# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scope-wide drain receipts, separate from publication and KV inheritance."""

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from itertools import chain
from threading import Lock

from .execution_authority import Revision
from .execution_version import ExecutionVersion

__all__ = [
    "FenceRequest",
    "FenceReceipt",
    "RevisionFence",
    "RequestDrainTracker",
    "RequestResourceClaim",
    "ScopeResourceClaim",
]


@dataclass(frozen=True)
class FenceRequest:
    """Identify a revision's drain within one worker lifetime."""

    session_id: str
    revision: Revision

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a nonempty string")
        if not isinstance(self.revision, Revision):
            raise TypeError("revision must be a Revision")


@dataclass(frozen=True)
class FenceReceipt:
    """All admitted work through the replaced epoch has finished resource access.

    This is historical evidence, not permission to resume or a KV reservation.
    """

    request: FenceRequest


@dataclass(frozen=True)
class RevisionFence:
    """Wait for drain without treating a timeout as completion or cancellation."""

    request: FenceRequest
    _future: Future[FenceReceipt] = field(repr=False, compare=False)

    def wait(self, timeout: float | None = 5.0) -> FenceReceipt:
        """Return the receipt or raise; waiting never grants successor authority."""
        return self._future.result(timeout=timeout)


@dataclass(frozen=True, eq=False)
class RequestResourceClaim:
    """One worker-local accessor whose completion is separate from its request.

    Only the exact object returned by its tracker can release the claim. Request
    IDs and owner names may be reused after completion; they are not release
    capabilities. A claim neither grants authority nor reserves native memory.
    """

    request_id: int
    version: ExecutionVersion
    owner: str


@dataclass(frozen=True, eq=False)
class ScopeResourceClaim:
    """A non-generating accessor, such as an ALIGN copy, that a later FENCE must drain.

    The authority owner must authorize acquisition. This tracks physical lifetime;
    it neither enqueues a request nor authorizes execution or KV reuse.
    """

    version: ExecutionVersion
    owner: str


class RequestDrainTracker:
    """Track engine admissions and resource accessors through their cleanup.

    Enqueue and registration share a lock so fast completion cannot overtake
    registration. The owner must close admission before asking whether a scope
    has drained. Resource-only release for recomputation is not completion.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: dict[int, ExecutionVersion] = {}
        self._resources: dict[RequestResourceClaim | ScopeResourceClaim, ExecutionVersion] = {}
        self._broadcasted: set[int] = set()
        self._error: Exception | None = None

    def enqueue(self, version: ExecutionVersion, submit: Callable[[], int]) -> int:
        """Register an enqueue atomically with terminal completion."""
        with self._lock:
            self._require_healthy()
            try:
                request_id = submit()
                if request_id in self._requests:
                    raise RuntimeError("request ID reused before terminal cleanup")
                self._requests[request_id] = version
                return request_id
            except Exception as error:
                # A failing enqueue may already have made work visible. Without
                # an ID there is no safe way to certify its completion.
                self._error = error
                raise

    def add_child(self, parent_id: int, child_id: int, version: ExecutionVersion) -> None:
        """Expand a still-pending parent before any of its children can execute."""
        with self._lock:
            self._require_healthy()
            if self._requests.get(parent_id) != version or child_id in self._requests:
                self._error = RuntimeError("child execution has no unique pending parent")
                raise self._error
            self._requests[child_id] = version

    def observe_broadcast(
        self, requests: tuple[tuple[int, ExecutionVersion], ...], *, source_rank: bool
    ) -> None:
        """Record an admission envelope received from the sole submitting rank.

        The source must still own each original enqueue. Peers register the same
        identities, even if their versions have since been revoked. This tracks
        already-admitted work; it does not grant new execution authority.

        The caller must join an all-rank acknowledgement before exposing the
        envelope to activation or cleanup. Otherwise the source could complete
        before a late peer records its ownership, yielding a false global drain.
        """
        with self._lock:
            self._require_healthy()
            try:
                if type(source_rank) is not bool or not isinstance(requests, tuple):
                    raise TypeError("invalid broadcast admission envelope")
                identities = set()
                for request_id, version in requests:
                    if type(request_id) is not int or not isinstance(version, ExecutionVersion):
                        raise TypeError("invalid broadcast execution identity")
                    if request_id in identities or request_id in self._broadcasted:
                        raise RuntimeError("request was broadcast more than once")
                    identities.add(request_id)
                    if source_rank:
                        if self._requests.get(request_id) != version:
                            raise RuntimeError("broadcast has no matching source admission")
                    elif request_id in self._requests:
                        raise RuntimeError("broadcast collides with a pending execution")
                self._requests.update(requests)
                self._broadcasted.update(identities)
            except Exception as error:
                self._error = error
                raise

    def fail(self, error: Exception) -> None:
        """Preserve uncertainty after failed distributed admission or execution."""
        with self._lock:
            if self._error is None:
                self._error = error

    def acquire_resource(
        self,
        request_id: int,
        version: ExecutionVersion,
        owner: str,
        *,
        parent: RequestResourceClaim | None = None,
    ) -> RequestResourceClaim:
        """Register an accessor before it can touch resources or outlive cleanup.

        The request must still be admitted, or an existing claim must explicitly
        cover the same request/version while a dependent accessor is registered.
        Taking a child claim and releasing its parent are serialized, so no
        apparent empty interval can let FENCE pass between the two owners.
        """
        if type(request_id) is not int or not isinstance(version, ExecutionVersion):
            raise TypeError("resource claim requires a request ID and execution version")
        if not isinstance(owner, str) or not owner:
            raise ValueError("resource owner must be a nonempty string")
        with self._lock:
            self._require_healthy()
            if parent is None:
                valid = self._requests.get(request_id) == version
            else:
                valid = (
                    isinstance(parent, RequestResourceClaim)
                    and self._resources.get(parent) == version
                    and parent.request_id == request_id
                    and parent.version == version
                )
            if not valid:
                self._error = RuntimeError("resource accessor has no matching live owner")
                raise self._error
            claim = RequestResourceClaim(request_id, version, owner)
            self._resources[claim] = version
            return claim

    def acquire_scope_resource(self, version: ExecutionVersion, owner: str) -> ScopeResourceClaim:
        """Track an independently authorized accessor without admitting generation."""
        if not isinstance(version, ExecutionVersion):
            raise TypeError("scope resource requires an execution version")
        if not isinstance(owner, str) or not owner:
            raise ValueError("resource owner must be a nonempty string")
        with self._lock:
            self._require_healthy()
            claim = ScopeResourceClaim(version, owner)
            self._resources[claim] = version
            return claim

    def has_resource(self, claim: RequestResourceClaim | ScopeResourceClaim) -> bool:
        """Observe a claim's lifetime; an unhealthy tracker cannot prove completion."""
        with self._lock:
            self._require_healthy()
            return claim in self._resources

    def release_resource(self, claim: RequestResourceClaim | ScopeResourceClaim) -> None:
        """Record proven accessor completion after its native release succeeds."""
        with self._lock:
            if (
                not isinstance(claim, (RequestResourceClaim, ScopeResourceClaim))
                or claim not in self._resources
            ):
                error = RuntimeError("resource completion has no matching live claim")
                if self._error is None:
                    self._error = error
                raise error
            del self._resources[claim]

    def complete(self, request_id: int, version: ExecutionVersion | None) -> None:
        """Record terminal cleanup, or removal of a never-activated queue item."""
        with self._lock:
            recorded = self._requests.get(request_id)
            if recorded is not None and recorded != version:
                self._error = RuntimeError("request execution identity changed before cleanup")
                raise self._error
            self._requests.pop(request_id, None)
            self._broadcasted.discard(request_id)

    def drained(self, version: ExecutionVersion) -> bool:
        """Check all epochs through version; callers must already exclude new admission."""
        with self._lock:
            self._require_healthy()
            return not any(
                (v.tenant_id, v.scope_id) == (version.tenant_id, version.scope_id)
                and v.epoch <= version.epoch
                for v in chain(self._requests.values(), self._resources.values())
            )

    def _require_healthy(self) -> None:
        if self._error is not None:
            raise RuntimeError("request drain is unknown; restart required") from self._error
