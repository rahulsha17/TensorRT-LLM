# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish execution authority and acknowledge revision lifecycle transitions."""

from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock, RLock, Thread
from typing import Literal, Protocol
from uuid import uuid4

from .execution_alignment import (
    AlignmentInput,
    AlignmentReceipt,
    AlignmentRequest,
    ExecutionPlacement,
    RevisionAlignment,
    WorkerAlignmentState,
)
from .execution_authority import AdvanceReceipt, Revision, ScopeRegistry, ScopeSnapshot
from .execution_fence import (
    FenceReceipt,
    FenceRequest,
    RequestDrainTracker,
    RequestResourceClaim,
    RevisionFence,
    ScopeResourceClaim,
)
from .execution_gate import ExecutionAuthorityError, HostExecution, HostExecutionGate
from .execution_version import ExecutionVersion

__all__ = [
    "AlignmentTransferPermit",
    "DeviceVersionPublisher",
    "ExecutionController",
    "LocalPublicationTransport",
    "PublicationReceipt",
    "PublicationTransport",
    "RevisionPublication",
    "VersionPublication",
    "VersionUpdate",
    "WorkerVersionTable",
    "WorkerIdentity",
]


@dataclass(frozen=True)
class WorkerIdentity:
    """Identify one bound worker table, independently of its publication endpoint."""

    session_id: str
    worker_id: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value for value in (self.session_id, self.worker_id)
        ):
            raise ValueError("worker identity requires an authority session and worker ID")


@dataclass(frozen=True)
class VersionUpdate:
    """Identify a publication within one executor lifetime.

    Args:
        session_id: Unique identity of the authority owner for this worker.
        sequence: Positive publication number assigned by that owner.
        version: Full execution identity, including its uint64 epoch.
        resume_prefix_length: Common cache-lookup bound for collective RESUME.
            None keeps ordinary single-worker reuse. The local receipt must
            cover this length, and a retry cannot change it.
        admit_requests: Whether this worker may execute this version. Pending
            revisions and non-destination placements still observe authority and
            drain old work, without permission to admit or commit successor work.
        source_only: Retain the selected prefix at a source placement without
            granting successor execution authority.
        release_source: Global ALIGN completed; retire the source hold without
            granting RESUME. Physical cleanup still belongs to the engine.
    """

    session_id: str
    sequence: int
    version: ExecutionVersion
    alignment: AlignmentRequest | None = None
    resume: AlignmentReceipt | None = None
    resume_prefix_length: int | None = None
    admit_requests: bool = True
    source_only: bool = False
    release_source: bool = False

    def __post_init__(self) -> None:
        if type(self.admit_requests) is not bool:
            raise TypeError("admit_requests must be a bool")
        if type(self.source_only) is not bool or type(self.release_source) is not bool:
            raise TypeError("source permissions must be bools")
        if self.resume is not None and not self.admit_requests:
            raise ValueError("RESUME must enable execution at its destination")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a nonempty string")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("sequence must be a positive integer")
        if not isinstance(self.version, ExecutionVersion):
            raise TypeError("version must be an ExecutionVersion")
        if self.alignment is not None and (
            not isinstance(self.alignment, AlignmentRequest)
            or self.alignment.session_id != self.session_id
            or self.alignment.revision.successor_version != self.version
        ):
            raise ValueError("alignment must belong to this publication")
        if self.resume is not None and (
            not isinstance(self.resume, AlignmentReceipt) or self.resume.request != self.alignment
        ):
            raise ValueError("RESUME must match this publication's alignment")
        if self.resume_prefix_length is not None and (
            self.resume is None
            or type(self.resume_prefix_length) is not int
            or not self.resume.request.revised_input.required_prefix_length
            <= self.resume_prefix_length
            <= self.resume.reusable_prefix_length
        ):
            raise ValueError("RESUME prefix must be covered by the live reservation")
        if self.source_only and (
            self.alignment is None
            or self.alignment.revised_input.transfer is None
            or self.admit_requests
            or self.resume is not None
        ):
            raise ValueError("source publication requires a non-generating transfer plan")
        if self.release_source and not self.source_only:
            raise ValueError("only a prefix source can release its hold")


@dataclass(frozen=True)
class PublicationReceipt:
    """Acknowledge worker observation, without proving FENCE.

    Args:
        update: Publication being acknowledged.
        observed_version: Version recorded by the worker when acknowledging.
            A newer epoch can cover an older publication, without making the
            older version current again.
    """

    update: VersionUpdate
    observed_version: ExecutionVersion

    def __post_init__(self) -> None:
        if not isinstance(self.update, VersionUpdate):
            raise TypeError("update must be a VersionUpdate")
        if not isinstance(self.observed_version, ExecutionVersion):
            raise TypeError("observed_version must be an ExecutionVersion")
        requested = self.update.version
        observed = self.observed_version
        if (requested.tenant_id, requested.scope_id) != (observed.tenant_id, observed.scope_id):
            raise ValueError("publication receipt belongs to a different scope")
        if observed.epoch < requested.epoch:
            raise ValueError("worker has not observed the requested epoch")


@dataclass(frozen=True)
class VersionPublication:
    """Handle for an asynchronous publication; waiting does not cancel it.

    Args:
        update: Version and publication identity being sent.
        _future: Completion owned by the controller.
    """

    update: VersionUpdate
    _future: Future[PublicationReceipt] = field(repr=False, compare=False)

    def wait(self, timeout: float | None = 5.0) -> PublicationReceipt:
        """Wait up to timeout seconds for observation, or raise the publication error."""
        return self._future.result(timeout=timeout)


@dataclass(frozen=True)
class RevisionPublication:
    """Separate a revision's acceptance from publication of the current version.

    Args:
        advance_receipt: Acceptance receipt, possibly historical on an old retry.
        publication: Publication of the scope's current version at this call.
        fence: Drain through this revision's replaced epoch, possibly historical.
    """

    advance_receipt: AdvanceReceipt
    publication: VersionPublication
    fence: RevisionFence
    alignment: RevisionAlignment | None = None


class DeviceVersionPublisher(Protocol):
    """Queue a device update without waiting for model execution."""

    def publish(self, version: ExecutionVersion) -> None: ...


@dataclass(frozen=True, eq=False)
class AlignmentTransferPermit:
    """Worker-local permission for one copy; never permission to generate.

    Only the table that issued this exact object can admit its operations.
    Transport code must also bind the exchange to the revised prefix identity.
    """

    request: AlignmentRequest
    request_id: int
    transfer_id: int
    direction: Literal["send", "receive"]
    prefix_length: int
    claim: ScopeResourceClaim


@dataclass
class _AlignmentTransfer:
    tracker: RequestDrainTracker
    alignment_state: WorkerAlignmentState
    accepted: bool = False
    error: Exception | None = None


class WorkerVersionTable:
    """Store one owner's versions; readers cannot choose or advance authority.

    Args:
        session_id: Authority owner permitted to publish to this table.
    """

    def __init__(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        self._session_id = session_id
        self._identity = WorkerIdentity(session_id, uuid4().hex)
        self._lock = RLock()
        self._versions: dict[tuple[str, str], VersionUpdate] = {}
        # Forward submission holds the device lock. Its advisory polls must not
        # take _lock, which the publisher holds while acquiring the device lock.
        self._observation_lock = Lock()
        self._observed_versions: dict[tuple[str, str], ExecutionVersion | None] = {}
        self._observation_failed = False
        self._closed = False
        self._gate = HostExecutionGate(self._lock, self._current_version_locked)
        self._device_publisher: DeviceVersionPublisher | None = None
        self._publication_error: Exception | None = None
        self.drain_tracker: RequestDrainTracker | None = None
        self.alignment_state: WorkerAlignmentState | None = None
        self._alignment_transfers: dict[AlignmentTransferPermit, _AlignmentTransfer] = {}

    def poll_fence(self, request: FenceRequest) -> FenceReceipt | None:
        """Check a drain without blocking publication or waiting on the GPU."""
        with self._lock:
            self._require_open()
            if request.session_id != self.session_id:
                raise ValueError("fence belongs to a different authority session")
            old = request.revision.expected_version
            current = self._current_version_locked(old)
            if current is None or current.epoch < request.revision.successor_version.epoch:
                return None
            if self.drain_tracker is None:
                raise RuntimeError("worker does not provide execution drain tracking")
            if not self.drain_tracker.drained(old):
                return None
            return FenceReceipt(request)

    def poll_alignment(self, request: AlignmentRequest) -> AlignmentReceipt | None:
        """A prepared prefix is usable only after the revision's drain completes."""
        with self._lock:
            self._require_open()
            if request.session_id != self.session_id:
                raise ValueError("alignment belongs to a different authority session")
            target = request.revision.successor_version
            current = self._current_version_locked(target)
            if current is None or current.epoch < target.epoch:
                return None
            if current != target:
                raise RuntimeError("alignment was superseded")
            if self.alignment_state is None:
                raise RuntimeError("worker does not support ALIGN")
            update = self._versions[(target.tenant_id, target.scope_id)]
            if update.alignment != request:
                # A deferred ALIGN query may arrive before its publication.
                # It cannot acknowledge a plan this worker has not observed.
                return None
            receipt = self.alignment_state.poll(request)
            copies = 0
            for permit, transfer in self._alignment_transfers.items():
                if permit.request != request:
                    continue
                if transfer.error is not None:
                    raise RuntimeError(
                        "ALIGN copy failed; a new revision is required"
                    ) from transfer.error
                if not transfer.accepted or self._alignment_transfer_live(permit, transfer):
                    return None
                copies += 1
            if request.revised_input.transfer is not None and not copies:
                return None
            if (
                receipt is None
                or self.poll_fence(FenceRequest(self.session_id, request.revision)) is None
            ):
                return None
            return receipt

    def bind_device_publisher(self, publisher: DeviceVersionPublisher) -> None:
        """Bind the engine's device table before any authority is published."""
        with self._lock:
            self._require_open()
            if self._versions or self._device_publisher is not None:
                raise RuntimeError("device publication must be bound before registration")
            self._device_publisher = publisher

    @contextmanager
    def guard_versions(self, versions: list[ExecutionVersion | None]) -> Iterator[list[bool]]:
        """Order short host state commits with mailbox updates; never wait for CUDA here."""
        with self._lock:
            self._require_open()
            yield [
                v is None
                or (
                    self._current_version_locked(v) == v
                    and self._versions[(v.tenant_id, v.scope_id)].admit_requests
                )
                for v in versions
            ]

    def observe_allowed(self, versions: tuple[ExecutionVersion | None, ...]) -> tuple[bool, ...]:
        """Poll authority for early exit, never for admission or token/KV commit.

        A positive observation can immediately become stale. A revoked version
        cannot become current again. This lock is never held across a CUDA call
        or while acquiring the main authority lock.
        """
        with self._observation_lock:
            return tuple(
                version is None
                or (
                    not self._observation_failed
                    and self._observed_versions.get((version.tenant_id, version.scope_id))
                    == version
                )
                for version in versions
            )

    @property
    def session_id(self) -> str:
        """Identity of the authority owner bound for this table's lifetime."""
        return self._session_id

    @property
    def identity(self) -> WorkerIdentity:
        """Read this worker lifetime's identity without granting execution authority."""
        with self._lock:
            self._require_open()
            return self._identity

    def apply(self, update: VersionUpdate) -> PublicationReceipt:
        """Record an owner update atomically; duplicate or delayed messages never rewind it."""
        if not isinstance(update, VersionUpdate):
            raise TypeError("update must be a VersionUpdate")
        if update.session_id != self.session_id:
            raise ValueError("publication belongs to a different authority session")
        key = (update.version.tenant_id, update.version.scope_id)
        with self._lock:
            self._require_open()
            previous = self._versions.get(key)
            if previous is not None:
                if update.sequence == previous.sequence and update != previous:
                    raise ValueError("publication sequence conflicts with the recorded version")
                if update.sequence < previous.sequence:
                    if update.version.epoch > previous.version.epoch:
                        raise ValueError("older publication sequence carries a newer epoch")
                    return PublicationReceipt(update, previous.version)
                if update.version.epoch < previous.version.epoch:
                    raise ValueError("newer publication sequence would rewind authority")
            if update.resume is not None:
                if self.poll_alignment(update.resume.request) != update.resume:
                    raise RuntimeError("RESUME requires the live ALIGN receipt and completed FENCE")
                assert self.alignment_state is not None
                self.alignment_state.authorize_resume(update.resume, update.resume_prefix_length)
            if update.release_source:
                assert update.alignment is not None
                if self.poll_alignment(update.alignment) is None:
                    raise RuntimeError("source release requires completed ALIGN and FENCE")
                assert self.alignment_state is not None
                self.alignment_state.release_source(update.alignment)
            if self.alignment_state is not None:
                self.alignment_state.observe(
                    update.version, update.alignment, source_only=update.source_only
                )
            self._versions[key] = update
            with self._observation_lock:
                self._observed_versions[key] = update.version if update.admit_requests else None
            self._gate.revoke_locked(update.version, admit_current=update.admit_requests)
            if self._device_publisher is not None:
                try:
                    self._device_publisher.publish(update.version)
                except Exception as error:
                    self._publication_error = error
                    with self._observation_lock:
                        self._observation_failed = True
                    self._gate.close_locked()
                    raise
            return PublicationReceipt(update, update.version)

    def _current_version_locked(self, version: ExecutionVersion) -> ExecutionVersion | None:
        update = self._versions.get((version.tenant_id, version.scope_id))
        return None if update is None else update.version

    @contextmanager
    def admission(self, version: ExecutionVersion) -> Iterator[HostExecution]:
        """Order worker enqueue with authority and acknowledged RESUME."""
        with self._gate.admission(version) as execution:
            update = self._versions[(version.tenant_id, version.scope_id)]
            if update.alignment is not None and update.resume is None:
                raise ExecutionAuthorityError("successor is pending ALIGN and RESUME")
            if not update.admit_requests:
                raise ExecutionAuthorityError("execution is not admitted at this worker")
            yield execution

    def current_version(self, tenant_id: str, scope_id: str) -> ExecutionVersion | None:
        """Observe a worker's version at this instant; unknown scopes return None."""
        with self._lock:
            self._require_open()
            update = self._versions.get((tenant_id, scope_id))
            return None if update is None else update.version

    def admit_resource(
        self, request_id: int, version: ExecutionVersion, owner: str
    ) -> RequestResourceClaim:
        """Claim an existing request's resource before publication can revoke it.

        Only the short host registration holds the authority lock. The caller
        constructs or publishes the accessor afterward, retaining this claim
        until physical cleanup finishes even if authority has since advanced.
        Stale admission is rejected without creating or releasing a claim.
        """
        with self.admission(version) as execution:
            try:
                self._require_open()
                if self.drain_tracker is None:
                    raise RuntimeError("worker does not provide execution drain tracking")
                return self.drain_tracker.acquire_resource(request_id, version, owner)
            finally:
                execution.finish()

    def _require_alignment_pending(self, request: AlignmentRequest) -> VersionUpdate:
        self._require_open()
        if not isinstance(request, AlignmentRequest) or request.session_id != self.session_id:
            raise ValueError("ALIGN copy belongs to another authority session")
        version = request.revision.successor_version
        update = self._versions.get((version.tenant_id, version.scope_id))
        if update is None or update.version != version or update.alignment != request:
            raise ExecutionAuthorityError("ALIGN copy requires the current published input")
        if update.resume is not None or update.release_source:
            raise ExecutionAuthorityError("ALIGN copy admission ends at RESUME")
        if self.alignment_state is None or self.drain_tracker is None:
            raise RuntimeError("ALIGN copy requires alignment and physical drain tracking")
        return update

    def _alignment_transfer_live(
        self, permit: AlignmentTransferPermit, transfer: _AlignmentTransfer
    ) -> bool:
        if self.drain_tracker is not transfer.tracker:
            raise RuntimeError("ALIGN copy drain tracker changed")
        if self.alignment_state is not transfer.alignment_state:
            raise RuntimeError("ALIGN copy reservation state changed")
        return transfer.tracker.has_resource(permit.claim)

    def admit_alignment_transfer(
        self,
        request: AlignmentRequest,
        fence: FenceReceipt,
        *,
        request_id: int,
        transfer_id: int,
        direction: Literal["send", "receive"],
        prefix_length: int,
    ) -> AlignmentTransferPermit:
        """Register one non-generating copy after the runtime's global FENCE.

        The trusted runtime supplies its completed cohort receipt. The worker
        rechecks its own FENCE and pending input under the publication lock.
        Callers must hold the source and destination memory before using this
        permit and retain the claim through physical cleanup. No native work or
        network call runs under this lock.
        """
        if (
            type(request_id) is not int
            or type(transfer_id) is not int
            or not 0 <= transfer_id < (1 << 63)
        ):
            raise ValueError("ALIGN copy requires a request ID and nonnegative int64 transfer ID")
        if direction not in ("send", "receive"):
            raise ValueError("invalid ALIGN copy direction")
        with self._lock:
            update = self._require_alignment_pending(request)
            if type(prefix_length) is not int or not 0 < prefix_length < len(
                request.revised_input.input_token_ids
            ):
                raise ValueError("ALIGN copy must leave the final input token for computation")
            plan = request.revised_input.transfer
            if plan is not None:
                expected_direction = "send" if update.source_only else "receive"
                if direction != expected_direction or prefix_length != plan.num_tokens:
                    raise ValueError("ALIGN copy does not match this endpoint's transfer plan")
                if update.source_only:
                    assert self.alignment_state is not None
                    if self.alignment_state.poll(request) is None:
                        raise RuntimeError("ALIGN copy requires the prepared source hold")
            expected = FenceReceipt(FenceRequest(self.session_id, request.revision))
            if (
                not isinstance(fence, FenceReceipt)
                or fence != expected
                or self.poll_fence(expected.request) != expected
            ):
                raise ExecutionAuthorityError("ALIGN copy requires completed FENCE")
            if any(
                p.transfer_id == transfer_id and p.direction == direction
                for p in self._alignment_transfers
            ):
                raise ValueError("ALIGN copy exchange cannot be reused")
            tracker = self.drain_tracker
            assert tracker is not None
            claim = tracker.acquire_scope_resource(
                request.revision.successor_version, f"align-{direction}-session"
            )
            assert self.alignment_state is not None
            try:
                self.alignment_state.retain_transfer(
                    request, lambda: not tracker.has_resource(claim)
                )
            except Exception:
                tracker.release_resource(claim)
                raise
            permit = AlignmentTransferPermit(
                request, request_id, transfer_id, direction, prefix_length, claim
            )
            self._alignment_transfers[permit] = _AlignmentTransfer(tracker, self.alignment_state)
            return permit

    @contextmanager
    def alignment_transfer_admission(self, permit: AlignmentTransferPermit) -> Iterator[None]:
        """Reserve a copy operation while its exact permit and physical owner are live."""
        with self._lock:
            self._require_open()
            transfer = self._alignment_transfers.get(permit)
            if transfer is None:
                raise ExecutionAuthorityError("unknown ALIGN copy permit")
            self._require_alignment_pending(permit.request)
            if (
                transfer.accepted
                or transfer.error is not None
                or not self._alignment_transfer_live(permit, transfer)
            ):
                raise ExecutionAuthorityError("ALIGN copy admission is closed")
            yield

    def accept_alignment_transfer(self, permit: AlignmentTransferPermit) -> None:
        """Record engine-validated copy success after physical cleanup.

        Resource release alone never calls this method. The engine must validate
        the copied prefix and its placement before accepting it. This receipt
        does not change the cache reservation or grant execution authority.
        """
        with self._lock:
            self._require_alignment_pending(permit.request)
            transfer = self._alignment_transfers.get(permit)
            if transfer is None:
                raise ExecutionAuthorityError("unknown ALIGN copy permit")
            if transfer.error is not None:
                raise RuntimeError("failed ALIGN copy cannot be accepted") from transfer.error
            if self._alignment_transfer_live(permit, transfer):
                raise RuntimeError("ALIGN copy still owns physical resources")
            transfer.accepted = True

    def fail_alignment_transfer(self, permit: AlignmentTransferPermit, error: Exception) -> None:
        """Reject reuse without releasing memory or erasing future FENCE obligations."""
        if not isinstance(error, Exception):
            raise TypeError("ALIGN failure must be an exception")
        with self._lock:
            transfer = self._alignment_transfers.get(permit)
            if transfer is None:
                raise ExecutionAuthorityError("unknown ALIGN copy permit")
            if transfer.error is None:
                transfer.error = error
            if transfer.accepted:
                # Contradictory evidence after acceptance invalidates authority;
                # it cannot silently leave a resumed successor running.
                self.close()

    def close(self) -> None:
        """Invalidate observations when the worker stops."""
        with self._lock:
            self._closed = True
            with self._observation_lock:
                self._observation_failed = True
            self._gate.close_locked()
            if self.alignment_state is not None:
                self.alignment_state.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("worker version table is closed")
        if self._publication_error is not None:
            raise RuntimeError(
                "device authority publication failed; restart required"
            ) from self._publication_error


class PublicationTransport(Protocol):
    """Transport owned by the controller's publisher thread."""

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        """Publish with bounded waits, checking stop while waiting."""
        ...

    def poll_fence(self, request: FenceRequest, stop: Event) -> FenceReceipt | None: ...

    def poll_alignment(self, request: AlignmentRequest, stop: Event) -> AlignmentReceipt | None: ...

    def close(self) -> None:
        """Release transport resources on their owning thread."""
        ...


class LocalPublicationTransport:
    """Publish directly to the host table of an in-process worker."""

    def __init__(self, table: WorkerVersionTable) -> None:
        self._table = table

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        if stop.is_set():
            raise RuntimeError("execution controller is closed")
        return self._table.apply(update)

    def poll_fence(self, request: FenceRequest, stop: Event) -> FenceReceipt | None:
        if stop.is_set():
            raise RuntimeError("execution controller is closed")
        return self._table.poll_fence(request)

    def poll_alignment(self, request: AlignmentRequest, stop: Event) -> AlignmentReceipt | None:
        if stop.is_set():
            raise RuntimeError("execution controller is closed")
        return self._table.poll_alignment(request)

    def close(self) -> None:
        pass


class ExecutionController:
    """Own a scope registry and serialize its publications through one transport.

    Args:
        session_id: Unique authority identity shared with the worker table.
        transport: Dedicated transport used only by the publisher thread.
        max_pending: Maximum queued or in-flight publications. An accepted
            revision beyond this limit gets a failed publication handle and can
            be retried once space is available.
        placements: Fixed frontend lifetimes supplied by runtime bootstrap to a
            placement-aware transport. A destination must belong to this set.

    Acceptance completes before publication. Publication failure never rolls
    back the registry. Retrying register/advance republishes the latest version
    after a failed attempt; a waiter timeout alone does not cancel that attempt.
    """

    def __init__(
        self,
        session_id: str,
        transport: PublicationTransport,
        max_pending: int = 256,
        *,
        placements: frozenset[ExecutionPlacement] = frozenset(),
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        if not isinstance(placements, frozenset) or any(
            not isinstance(p, ExecutionPlacement) or p.session_id != session_id for p in placements
        ):
            raise ValueError("placements must identify frontend lifetimes in this session")
        self._placements = placements
        self._session_id = session_id
        self._transport = transport
        self._max_pending = max_pending
        self._registry = ScopeRegistry()
        self._lock = Lock()
        self._gate = HostExecutionGate(self._lock, self._current_version_locked)
        self._stop = Event()
        self._sequence = 0
        self._latest: dict[tuple[str, str], VersionPublication] = {}
        self._pending: set[VersionPublication] = set()
        self._fences: dict[FenceRequest, RevisionFence] = {}
        self._pending_fences: dict[FenceRequest, RevisionFence] = {}
        self._alignment_inputs: dict[Revision, AlignmentInput | None] = {}
        self._alignments: dict[AlignmentRequest, RevisionAlignment] = {}
        self._pending_alignments: dict[AlignmentRequest, RevisionAlignment] = {}
        self._alignment_current: dict[ExecutionVersion, AlignmentRequest] = {}
        self._resumes: dict[ExecutionVersion, AlignmentReceipt] = {}
        self._queue: Queue[VersionPublication | FenceRequest | AlignmentRequest | None] = Queue()
        self._thread = Thread(target=self._run, name="retire_publication", daemon=True)
        self._thread.start()

    @property
    def closed(self) -> bool:
        """Whether this authority owner has stopped admitting work."""
        return self._stop.is_set()

    def register(self, version: ExecutionVersion) -> VersionPublication:
        """Register an initial version and schedule its publication."""
        with self._lock:
            self._require_open()
            snapshot = self._registry.register(version)
            return self._publish_current(snapshot.current_version)

    def advance(
        self, revision: Revision, *, alignment: AlignmentInput | None = None
    ) -> RevisionPublication:
        """Accept the revision and publish current authority, including on historical retries.

        Supply alignment here to reserve the matching prefix before releasing
        the old request. Otherwise, align() can bind the revised input later.
        Neither form opens successor admission before acknowledged RESUME.
        """
        with self._lock:
            self._require_open()
            self._validate_alignment(alignment)
            if revision in self._alignment_inputs and self._alignment_inputs[revision] != alignment:
                raise ValueError("revision retry changed its revised input")
            receipt = self._registry.advance(revision)
            self._alignment_inputs[revision] = alignment
            if alignment is not None:
                self._alignment_current[revision.successor_version] = AlignmentRequest(
                    self._session_id, revision, alignment
                )
            expected = revision.expected_version
            current = self._registry.snapshot(expected.tenant_id, expected.scope_id).current_version
            self._gate.revoke_locked(current)
            publication = self._publish_current(current)
            fence = self._request_fence(receipt.revision)
            aligned = (
                self._request_alignment(self._alignment_current[revision.successor_version])
                if alignment is not None
                else None
            )
            return RevisionPublication(receipt, publication, fence, aligned)

    def align(self, revision: Revision, alignment: AlignmentInput) -> RevisionPublication:
        """Bind revised input to an already accepted, still-current revision.

        Until this call, ADVANCE without alignment does not reserve any prefix.
        ALIGN looks up whatever compatible KV remains when the engine processes
        it. It can therefore return fewer tokens, including zero, than an ALIGN
        supplied with ADVANCE. Its receipt still requires completed FENCE.

        Wait for publication and alignment, then resume the receipt before
        submitting the successor. Exact retries keep the same request and hold;
        changing input, salt, or reuse requirements needs a new revision.
        """
        if not isinstance(revision, Revision):
            raise TypeError("revision must be a Revision")
        if not isinstance(alignment, AlignmentInput):
            raise TypeError("alignment must be an AlignmentInput")
        with self._lock:
            self._require_open()
            self._validate_alignment(alignment)
            if revision not in self._alignment_inputs:
                raise ValueError("ALIGN requires an accepted revision")
            version = revision.successor_version
            if not self._registry.is_current(version):
                raise ValueError("cannot align a superseded revision")
            request = AlignmentRequest(self._session_id, revision, alignment)
            previous = self._alignment_current.get(version)
            if previous is not None and previous != request:
                raise ValueError("ALIGN retry changed its revised input")
            self._alignment_current[version] = request
            return RevisionPublication(
                AdvanceReceipt(revision),
                self._publish_current(version),
                self._request_fence(revision),
                self._request_alignment(request),
            )

    def _request_fence(self, revision: Revision) -> RevisionFence:
        # Callers hold the controller lock.
        request = FenceRequest(self._session_id, revision)
        fence = self._fences.get(request)
        if fence is None or (fence._future.done() and fence._future.exception() is not None):
            fence = RevisionFence(request, Future())
            self._fences[request] = fence
            if len(self._pending_fences) >= self._max_pending:
                fence._future.set_exception(RuntimeError("fence backlog is full; retry revision"))
            else:
                self._pending_fences[request] = fence
                self._queue.put(request)
        return fence

    def _request_alignment(self, request: AlignmentRequest) -> RevisionAlignment:
        # Callers hold the controller lock.
        aligned = self._alignments.get(request)
        if aligned is None or (aligned._future.done() and aligned._future.exception() is not None):
            aligned = RevisionAlignment(request, Future())
            self._alignments[request] = aligned
            if len(self._pending_alignments) >= self._max_pending:
                aligned._future.set_exception(RuntimeError("ALIGN backlog is full; retry revision"))
            else:
                self._pending_alignments[request] = aligned
                self._queue.put(request)
        return aligned

    def resume(self, aligned: AlignmentReceipt) -> VersionPublication:
        """Open successor admission after a live worker recheck of ALIGN and FENCE.

        Wait for the returned publication before submitting exactly the aligned
        input with its successor execution version. A timeout is retryable with
        the same receipt and does not undo the worker's acceptance.
        """
        if not isinstance(aligned, AlignmentReceipt):
            raise TypeError("aligned must be an AlignmentReceipt")
        with self._lock:
            self._require_open()
            request = aligned.request
            version = request.revision.successor_version
            if request.session_id != self._session_id or not self._registry.is_current(version):
                raise ValueError("RESUME belongs to a different session or obsolete version")
            handle = self._alignments.get(request)
            if handle is None or not handle._future.done() or handle._future.result() != aligned:
                raise ValueError("RESUME requires this controller's completed ALIGN receipt")
            previous = self._resumes.get(version)
            if previous is not None and previous != aligned:
                raise ValueError("RESUME changed the reservation")
            self._resumes[version] = aligned
            return self._publish_current(version)

    def _current_version_locked(self, version: ExecutionVersion) -> ExecutionVersion | None:
        try:
            return self._registry.snapshot(version.tenant_id, version.scope_id).current_version
        except KeyError:
            return None

    @contextmanager
    def admission(
        self, version: ExecutionVersion, *, placement: ExecutionPlacement | None = None
    ) -> Iterator[HostExecution]:
        """Admit published initial work or an acknowledged, resumed successor."""
        with self._gate.admission(version) as execution:
            snapshot = self._registry.snapshot(version.tenant_id, version.scope_id)
            if snapshot.pending_revision is not None:
                raise ExecutionAuthorityError("successor is pending ALIGN and RESUME")
            publication = self._latest[(version.tenant_id, version.scope_id)]
            if not publication._future.done() or publication._future.exception() is not None:
                raise ExecutionAuthorityError("execution authority has not been published")
            request = self._alignment_current.get(version)
            if (
                request is not None
                and request.revised_input.destination is not None
                and request.revised_input.destination != placement
            ):
                raise ExecutionAuthorityError("successor belongs to another execution placement")
            yield execution

    def _validate_alignment(self, alignment: AlignmentInput | None) -> None:
        if alignment is not None and not isinstance(alignment, AlignmentInput):
            raise TypeError("alignment must be an AlignmentInput or None")
        if (
            alignment is not None
            and alignment.destination is not None
            and alignment.destination not in self._placements
        ):
            raise ValueError("ALIGN destination is not a member of this authority")
        if (
            alignment is not None
            and alignment.transfer is not None
            and alignment.transfer.source not in self._placements
        ):
            raise ValueError("ALIGN source is not a member of this authority")

    def snapshot(self, tenant_id: str, scope_id: str) -> ScopeSnapshot:
        """Read accepted authority, including when publication has failed."""
        return self._registry.snapshot(tenant_id, scope_id)

    def fail(self, error: Exception) -> None:
        """Revoke authority and fail waits without joining the publisher thread.

        Transport and liveness callbacks may use this while publication is in
        flight. Executor shutdown still calls close() to join the publisher.
        """
        if not isinstance(error, Exception):
            raise TypeError("execution failure requires an exception")
        with self._lock:
            if not self._stop.is_set():
                self._stop.set()
                self._gate.close_locked()
                for publication in self._pending:
                    publication._future.set_exception(error)
                self._pending.clear()
                for fence in self._pending_fences.values():
                    fence._future.set_exception(error)
                self._pending_fences.clear()
                for aligned in self._pending_alignments.values():
                    aligned._future.set_exception(error)
                self._pending_alignments.clear()
                self._queue.put(None)

    def close(self) -> None:
        """Fail outstanding waits and stop the publisher before releasing the transport."""
        self.fail(RuntimeError("execution controller is closed"))
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("execution publisher did not stop")

    def _require_open(self) -> None:
        if self._stop.is_set():
            raise RuntimeError("execution controller is closed")

    def _publish_current(self, version: ExecutionVersion) -> VersionPublication:
        key = (version.tenant_id, version.scope_id)
        previous = self._latest.get(key)
        if (
            previous is not None
            and previous.update.version == version
            and previous.update.alignment == self._alignment_current.get(version)
            and previous.update.resume == self._resumes.get(version)
        ):
            if not previous._future.done() or previous._future.exception() is None:
                return previous
            update = previous.update
        else:
            self._sequence += 1
            update = VersionUpdate(
                self._session_id,
                self._sequence,
                version,
                self._alignment_current.get(version),
                self._resumes.get(version),
                admit_requests=(
                    version in self._resumes
                    or self._registry.snapshot(version.tenant_id, version.scope_id).pending_revision
                    is None
                ),
            )
        publication = VersionPublication(update, Future())
        self._latest[key] = publication
        if len(self._pending) >= self._max_pending:
            publication._future.set_exception(
                RuntimeError("publication backlog is full; retry current version")
            )
            return publication
        self._pending.add(publication)
        self._queue.put(publication)
        return publication

    def _poll_one_fence(self) -> bool:
        with self._lock:
            if not self._pending_fences:
                return False
            request = next(iter(self._pending_fences))
            fence = self._pending_fences.pop(request)
            self._pending_fences[request] = fence
        try:
            receipt = self._transport.poll_fence(request, self._stop)
            if receipt is not None and (
                not isinstance(receipt, FenceReceipt) or receipt.request != request
            ):
                raise ValueError("acknowledgement does not match the fence")
        except Exception as error:
            with self._lock:
                if not fence._future.done():
                    fence._future.set_exception(error)
                self._pending_fences.pop(request, None)
            return True
        else:
            if receipt is not None:
                with self._lock:
                    if not fence._future.done():
                        fence._future.set_result(receipt)
                    self._pending_fences.pop(request, None)
                return True
            return False

    def _poll_one_alignment(self) -> bool:
        with self._lock:
            if not self._pending_alignments:
                return False
            request = next(iter(self._pending_alignments))
            aligned = self._pending_alignments.pop(request)
            self._pending_alignments[request] = aligned
        try:
            receipt = self._transport.poll_alignment(request, self._stop)
            if receipt is not None and (
                not isinstance(receipt, AlignmentReceipt) or receipt.request != request
            ):
                raise ValueError("acknowledgement does not match ALIGN")
        except Exception as error:
            with self._lock:
                if not aligned._future.done():
                    aligned._future.set_exception(error)
                self._pending_alignments.pop(request, None)
            return True
        else:
            if receipt is not None:
                with self._lock:
                    if not aligned._future.done():
                        aligned._future.set_result(receipt)
                    self._pending_alignments.pop(request, None)
                return True
            return False

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                fence_done = self._poll_one_fence()
                alignment_done = self._poll_one_alignment()
                with self._lock:
                    pending = max(len(self._pending_fences), len(self._pending_alignments))
                    # Drain ready receipts without an idle delay, but check the
                    # publication queue after every pair of lifecycle polls.
                    # Pending work still sleeps; bounded polling keeps many
                    # scopes from multiplying a single scope's polling period.
                    timeout = (
                        (0.0 if fence_done or alignment_done else max(0.001, 0.01 / pending))
                        if pending
                        else None
                    )
                try:
                    publication = self._queue.get(timeout=timeout)
                except Empty:
                    continue
                if publication is None or self._stop.is_set():
                    break
                if isinstance(publication, (FenceRequest, AlignmentRequest)):
                    continue
                try:
                    receipt = self._transport.publish(publication.update, self._stop)
                    if (
                        not isinstance(receipt, PublicationReceipt)
                        or receipt.update != publication.update
                        or (
                            publication.update.resume is not None
                            and receipt.observed_version != publication.update.version
                        )
                    ):
                        raise ValueError("acknowledgement does not match the publication")
                except Exception as error:
                    # A transport failure completes the waiter; it must not kill
                    # the publisher silently or undo an accepted revision.
                    with self._lock:
                        if not publication._future.done():
                            publication._future.set_exception(error)
                        self._pending.discard(publication)
                else:
                    with self._lock:
                        if not publication._future.done():
                            resume = publication.update.resume
                            if resume is not None and not self._registry.is_current(
                                publication.update.version
                            ):
                                publication._future.set_exception(
                                    RuntimeError("RESUME was superseded before acknowledgement")
                                )
                            else:
                                if resume is not None:
                                    self._registry.resume(resume.request.revision)
                                publication._future.set_result(receipt)
                        self._pending.discard(publication)
        finally:
            self._transport.close()
