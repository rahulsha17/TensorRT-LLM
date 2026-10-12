# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish accepted execution versions without granting execution or reuse rights."""

from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Queue
from threading import Event, Lock, Thread
from typing import Protocol

from .execution_authority import AdvanceReceipt, Revision, ScopeRegistry, ScopeSnapshot
from .execution_version import ExecutionVersion

__all__ = [
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

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a nonempty string")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("sequence must be a positive integer")
        if not isinstance(self.version, ExecutionVersion):
            raise TypeError("version must be an ExecutionVersion")


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
    """

    advance_receipt: AdvanceReceipt
    publication: VersionPublication


class WorkerVersionTable:
    """Store one owner's versions; readers cannot choose or advance authority.

    Args:
        session_id: Authority owner permitted to publish to this table.
    """

    def __init__(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        self._session_id = session_id
        self._lock = Lock()
        self._versions: dict[tuple[str, str], VersionUpdate] = {}
        self._closed = False

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
            self._versions[key] = update
            return PublicationReceipt(update, update.version)

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

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("worker version table is closed")


class PublicationTransport(Protocol):
    """Transport owned by the controller's publisher thread."""

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        """Publish with bounded waits, checking stop while waiting."""
        ...

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
        self._stop = Event()
        self._sequence = 0
        self._latest: dict[tuple[str, str], VersionPublication] = {}
        self._pending: set[VersionPublication] = set()
        self._queue: Queue[VersionPublication | None] = Queue()
        self._thread = Thread(target=self._run, name="retire_publication", daemon=True)
        self._thread.start()

    def register(self, version: ExecutionVersion) -> VersionPublication:
        """Register an initial version and schedule its publication."""
        with self._lock:
            self._require_open()
            snapshot = self._registry.register(version)
            return self._publish_current(snapshot.current_version)

    def advance(self, revision: Revision) -> RevisionPublication:
        """Accept the revision and publish current authority, including on historical retries."""
        with self._lock:
            self._require_open()
            receipt = self._registry.advance(revision)
            expected = revision.expected_version
            current = self._registry.snapshot(expected.tenant_id, expected.scope_id).current_version
            return RevisionPublication(receipt, self._publish_current(current))

    def snapshot(self, tenant_id: str, scope_id: str) -> ScopeSnapshot:
        """Read accepted authority, including when publication has failed."""
        return self._registry.snapshot(tenant_id, scope_id)

    def close(self) -> None:
        """Fail outstanding waits and stop the publisher before releasing the transport."""
        with self._lock:
            if not self._stop.is_set():
                self._stop.set()
                for publication in self._pending:
                    publication._future.set_exception(
                        RuntimeError("execution controller is closed")
                    )
                self._pending.clear()
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
            update = VersionUpdate(self._session_id, self._sequence, version)
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

    def _run(self) -> None:
        try:
            while (publication := self._queue.get()) is not None:
                if self._stop.is_set():
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
