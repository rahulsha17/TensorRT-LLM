# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish accepted execution versions without granting execution or reuse rights."""

from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock, RLock, Thread
from typing import Protocol

from .execution_alignment import (
    AlignmentInput,
    AlignmentReceipt,
    AlignmentRequest,
    RevisionAlignment,
    WorkerAlignmentState,
)
from .execution_authority import AdvanceReceipt, Revision, ScopeRegistry, ScopeSnapshot
from .execution_fence import FenceReceipt, FenceRequest, RequestDrainTracker, RevisionFence
from .execution_gate import ExecutionAuthorityError, HostExecution, HostExecutionGate
from .execution_version import ExecutionVersion

__all__ = [
    "DeviceVersionPublisher",
    "ExecutionController",
    "LocalPublicationTransport",
    "PublicationReceipt",
    "PublicationTransport",
    "RevisionPublication",
    "VersionPublication",
    "VersionUpdate",
    "WorkerVersionTable",
]


@dataclass(frozen=True)
class VersionUpdate:
    """Identify a publication within one executor lifetime.

    Args:
        session_id: Unique identity of the authority owner for this worker.
        sequence: Positive publication number assigned by that owner.
        version: Full execution identity, including its uint64 epoch.
    """

    session_id: str
    sequence: int
    version: ExecutionVersion
    alignment: AlignmentRequest | None = None

    def __post_init__(self) -> None:
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


class WorkerVersionTable:
    """Store one owner's versions; readers cannot choose or advance authority.

    Args:
        session_id: Authority owner permitted to publish to this table.
    """

    def __init__(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        self._session_id = session_id
        self._lock = RLock()
        self._versions: dict[tuple[str, str], VersionUpdate] = {}
        self._closed = False
        self._gate = HostExecutionGate(self._lock, self._current_version_locked)
        self._device_publisher: DeviceVersionPublisher | None = None
        self._publication_error: Exception | None = None
        self.drain_tracker: RequestDrainTracker | None = None
        self.alignment_state: WorkerAlignmentState | None = None

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
            receipt = self.alignment_state.poll(request)
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
            yield [v is None or self._current_version_locked(v) == v for v in versions]

    @property
    def session_id(self) -> str:
        """Identity of the authority owner bound for this table's lifetime."""
        return self._session_id

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
            if self.alignment_state is not None:
                self.alignment_state.observe(update.version, update.alignment)
            self._versions[key] = update
            self._gate.revoke_locked(update.version)
            if self._device_publisher is not None:
                try:
                    self._device_publisher.publish(update.version)
                except Exception as error:
                    self._publication_error = error
                    self._gate.close_locked()
                    raise
            return PublicationReceipt(update, update.version)

    def _current_version_locked(self, version: ExecutionVersion) -> ExecutionVersion | None:
        update = self._versions.get((version.tenant_id, version.scope_id))
        return None if update is None else update.version

    def admission(self, version: ExecutionVersion) -> AbstractContextManager[HostExecution]:
        """Order worker enqueue with mailbox updates; this does not fence GPU work."""
        return self._gate.admission(version)

    def current_version(self, tenant_id: str, scope_id: str) -> ExecutionVersion | None:
        """Observe a worker's version at this instant; unknown scopes return None."""
        with self._lock:
            self._require_open()
            update = self._versions.get((tenant_id, scope_id))
            return None if update is None else update.version

    def close(self) -> None:
        """Invalidate observations when the worker stops."""
        with self._lock:
            self._closed = True
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
    """Own a scope registry and serialize its publications to one worker.

    Args:
        session_id: Unique authority identity shared with the worker table.
        transport: Dedicated transport used only by the publisher thread.
        max_pending: Maximum queued or in-flight publications. An accepted
            revision beyond this limit gets a failed publication handle and can
            be retried once space is available.

    Acceptance completes before publication. Publication failure never rolls
    back the registry. Retrying register/advance republishes the latest version
    after a failed attempt; a waiter timeout alone does not cancel that attempt.
    """

    def __init__(
        self, session_id: str, transport: PublicationTransport, max_pending: int = 256
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
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
        self._queue: Queue[VersionPublication | FenceRequest | AlignmentRequest | None] = Queue()
        self._thread = Thread(target=self._run, name="retire_publication", daemon=True)
        self._thread.start()

    def register(self, version: ExecutionVersion) -> VersionPublication:
        """Register an initial version and schedule its publication."""
        with self._lock:
            self._require_open()
            snapshot = self._registry.register(version)
            return self._publish_current(snapshot.current_version)

    def advance(
        self, revision: Revision, *, alignment: AlignmentInput | None = None
    ) -> RevisionPublication:
        """Accept the revision and publish current authority, including on historical retries."""
        with self._lock:
            self._require_open()
            if alignment is not None and not isinstance(alignment, AlignmentInput):
                raise TypeError("alignment must be an AlignmentInput or None")
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
            request = FenceRequest(self._session_id, receipt.revision)
            fence = self._fences.get(request)
            if fence is None or (fence._future.done() and fence._future.exception() is not None):
                fence = RevisionFence(request, Future())
                self._fences[request] = fence
                if len(self._pending_fences) >= self._max_pending:
                    fence._future.set_exception(
                        RuntimeError("fence backlog is full; retry revision")
                    )
                else:
                    self._pending_fences[request] = fence
                    self._queue.put(request)
            aligned = None
            if alignment is not None:
                align_request = self._alignment_current[revision.successor_version]
                aligned = self._alignments.get(align_request)
                if aligned is None or (
                    aligned._future.done() and aligned._future.exception() is not None
                ):
                    aligned = RevisionAlignment(align_request, Future())
                    self._alignments[align_request] = aligned
                    if len(self._pending_alignments) >= self._max_pending:
                        aligned._future.set_exception(
                            RuntimeError("ALIGN backlog is full; retry revision")
                        )
                    else:
                        self._pending_alignments[align_request] = aligned
                        self._queue.put(align_request)
            return RevisionPublication(receipt, publication, fence, aligned)

    def _current_version_locked(self, version: ExecutionVersion) -> ExecutionVersion | None:
        try:
            return self._registry.snapshot(version.tenant_id, version.scope_id).current_version
        except KeyError:
            return None

    @contextmanager
    def admission(self, version: ExecutionVersion) -> Iterator[HostExecution]:
        """Admit registered initial work after publication, never a pending successor."""
        with self._gate.admission(version) as execution:
            snapshot = self._registry.snapshot(version.tenant_id, version.scope_id)
            if snapshot.pending_revision is not None:
                raise ExecutionAuthorityError("successor is pending ALIGN and RESUME")
            publication = self._latest[(version.tenant_id, version.scope_id)]
            if not publication._future.done() or publication._future.exception() is not None:
                raise ExecutionAuthorityError("execution authority has not been published")
            yield execution

    def snapshot(self, tenant_id: str, scope_id: str) -> ScopeSnapshot:
        """Read accepted authority, including when publication has failed."""
        return self._registry.snapshot(tenant_id, scope_id)

    def close(self) -> None:
        """Fail outstanding waits and stop the publisher before releasing the transport."""
        with self._lock:
            if not self._stop.is_set():
                self._stop.set()
                self._gate.close_locked()
                for publication in self._pending:
                    publication._future.set_exception(
                        RuntimeError("execution controller is closed")
                    )
                self._pending.clear()
                for fence in self._pending_fences.values():
                    fence._future.set_exception(RuntimeError("execution controller is closed"))
                self._pending_fences.clear()
                for aligned in self._pending_alignments.values():
                    aligned._future.set_exception(RuntimeError("execution controller is closed"))
                self._pending_alignments.clear()
                self._queue.put(None)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("execution publisher did not stop")

    def _require_open(self) -> None:
        if self._stop.is_set():
            raise RuntimeError("execution controller is closed")

    def _publish_current(self, version: ExecutionVersion) -> VersionPublication:
        key = (version.tenant_id, version.scope_id)
        previous = self._latest.get(key)
        if previous is not None and previous.update.version == version:
            if not previous._future.done() or previous._future.exception() is None:
                return previous
            update = previous.update
        else:
            self._sequence += 1
            update = VersionUpdate(
                self._session_id, self._sequence, version, self._alignment_current.get(version)
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

    def _poll_one_fence(self) -> None:
        with self._lock:
            if not self._pending_fences:
                return
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
        else:
            if receipt is not None:
                with self._lock:
                    if not fence._future.done():
                        fence._future.set_result(receipt)
                    self._pending_fences.pop(request, None)

    def _poll_one_alignment(self) -> None:
        with self._lock:
            if not self._pending_alignments:
                return
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
        else:
            if receipt is not None:
                with self._lock:
                    if not aligned._future.done():
                        aligned._future.set_result(receipt)
                    self._pending_alignments.pop(request, None)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._poll_one_fence()
                self._poll_one_alignment()
                with self._lock:
                    timeout = 0.01 if self._pending_fences or self._pending_alignments else None
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
                            publication._future.set_result(receipt)
                        self._pending.discard(publication)
        finally:
            self._transport.close()
