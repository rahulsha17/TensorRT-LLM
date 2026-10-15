# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host admission and output revocation for a remotely owned frontend.

The owner seals every frontend before accepting ADVANCE. Worker publication and
physical FENCE remain separate. A frontend opens initial work or a resumed
successor only after explicit activation of its exact observed publication.
"""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import RLock
from time import monotonic
from uuid import uuid4

from ..execution_authority import Revision
from ..execution_gate import ExecutionAuthorityError, HostExecution, HostExecutionGate
from ..execution_mailbox import PublicationReceipt, VersionUpdate, WorkerIdentity
from ..execution_version import ExecutionVersion

__all__ = ["FrontendAuthorityGate"]


class FrontendAuthorityGate:
    """Keep a fixed owner's local effects ordered with remote revision acceptance.

    Args:
        session_id: Authority session shared with this owner's worker tables.
        clock: Local monotonic clock used only for the owner lease.

    Network handlers call these short host transitions. An endpoint must also
    poll ``expire`` to wake consumers after owner loss. Expiry is terminal:
    delayed heartbeats, publications and activations cannot reopen this gate.
    Neither sealing nor expiry proves physical resource retirement.
    """

    def __init__(self, session_id: str, *, clock: Callable[[], float] = monotonic) -> None:
        self._identity = WorkerIdentity(session_id, uuid4().hex)
        self._lock = RLock()
        self._clock = clock
        self._owner: str | None = None
        self._lease_seconds = 0.0
        self._deadline = 0.0
        self._heartbeat = 0
        self._closed = False
        self._observed: dict[tuple[str, str], VersionUpdate] = {}
        self._active: dict[tuple[str, str], VersionUpdate] = {}
        self._initial: dict[tuple[str, str], ExecutionVersion] = {}
        self._floors: dict[tuple[str, str], ExecutionVersion] = {}
        self._seals: dict[tuple[str, str, str], Revision] = {}
        self._gate = HostExecutionGate(self._lock, self._current_locked)

    @property
    def identity(self) -> WorkerIdentity:
        """Identity of this frontend lifetime, distinct from its GPU workers."""
        return self._identity

    @property
    def closed(self) -> bool:
        with self._lock:
            self._expire_locked()
            return self._closed

    def bind(self, owner: str, lease_seconds: float) -> None:
        """Bind one owner incarnation; an exact retry does not extend its lease."""
        if not isinstance(owner, str) or not owner:
            raise ValueError("frontend requires an owner incarnation")
        if (
            type(lease_seconds) not in (int, float)
            or not math.isfinite(lease_seconds)
            or lease_seconds <= 0
        ):
            raise ValueError("frontend lease must be positive and finite")
        with self._lock:
            self._expire_locked()
            if self._closed:
                raise ExecutionAuthorityError("frontend authority is closed")
            if self._owner is not None:
                if self._owner != owner or self._lease_seconds != lease_seconds:
                    raise ValueError("frontend already has a different owner binding")
                return
            self._owner = owner
            self._lease_seconds = lease_seconds
            self._deadline = self._clock() + lease_seconds

    def renew(self, owner: str, sequence: int) -> int:
        """Renew with a fresh heartbeat; replayed heartbeats cannot postpone expiry."""
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("heartbeat sequence must be a positive integer")
        with self._lock:
            self._require_live_locked(owner)
            if sequence > self._heartbeat:
                self._heartbeat = sequence
                self._deadline = self._clock() + self._lease_seconds
            return self._heartbeat

    def check_owner(self, owner: str | None = None) -> None:
        """Check the bound owner without holding host authority across worker I/O."""
        with self._lock:
            self._require_live_locked(owner)

    def declare(self, owner: str, version: ExecutionVersion) -> None:
        """Establish a scope before asynchronous worker publication; do not admit work."""
        if not isinstance(version, ExecutionVersion):
            raise TypeError("scope declaration requires an execution version")
        key = (version.tenant_id, version.scope_id)
        with self._lock:
            self._require_live_locked(owner)
            observed = self._observed.get(key)
            known = self._floors.get(key) or (
                observed.version if observed is not None else self._initial.get(key)
            )
            if known is not None and known != version:
                raise ValueError("scope declaration conflicts with known frontend authority")
            self._initial.setdefault(key, version)

    def seal(self, owner: str, revision: Revision) -> Revision:
        """Revoke old host effects before the owner commits this revision.

        Exact retries retain their original result without disturbing a later
        epoch. Pending publications may lag an already sealed predecessor.
        """
        if not isinstance(revision, Revision):
            raise TypeError("seal requires a revision")
        expected = revision.expected_version
        key = (expected.tenant_id, expected.scope_id)
        seal_key = (*key, revision.revision_id)
        with self._lock:
            self._require_live_locked(owner)
            previous = self._seals.get(seal_key)
            if previous is not None:
                if previous != revision:
                    raise ValueError("revision ID conflicts with an earlier frontend seal")
                return previous
            observed = self._observed.get(key)
            known = self._floors.get(key) or (
                self._initial.get(key) if observed is None else observed.version
            )
            if known != expected:
                raise ValueError("frontend seal does not follow the known scope epoch")
            self._seals[seal_key] = revision
            self._floors[key] = revision.successor_version
            self._active.pop(key, None)
            self._gate.revoke_locked(revision.successor_version)
            return revision

    def observe(self, owner: str, update: VersionUpdate) -> PublicationReceipt:
        """Record publication after workers acknowledge it, without opening admission."""
        if not isinstance(update, VersionUpdate):
            raise TypeError("frontend publication requires a version update")
        key = (update.version.tenant_id, update.version.scope_id)
        with self._lock:
            self._require_live_locked(owner)
            if update.session_id != self.identity.session_id:
                raise ValueError("publication belongs to another authority session")
            previous = self._observed.get(key)
            initial = self._initial.get(key)
            if previous is None and initial is not None and initial != update.version:
                floor = self._floors.get(key)
                if floor is None or not initial.epoch <= update.version.epoch <= floor.epoch:
                    self._protocol_failure_locked("publication lacks a declared or sealed epoch")
            if previous is not None:
                if update.sequence == previous.sequence:
                    if update != previous:
                        self._protocol_failure_locked("conflicting publication sequence")
                    return PublicationReceipt(update, previous.version)
                if update.sequence < previous.sequence:
                    if update.version.epoch > previous.version.epoch:
                        self._protocol_failure_locked("older publication advances the epoch")
                    return PublicationReceipt(update, previous.version)
                if update.version.epoch < previous.version.epoch:
                    self._protocol_failure_locked("publication would rewind the epoch")
                if update.version != previous.version:
                    floor = self._floors.get(key)
                    if floor is None or floor.epoch < update.version.epoch:
                        self._protocol_failure_locked("new epoch lacks a frontend seal")
            self._observed[key] = update
            self._active.pop(key, None)
            if not update.admit_requests:
                self._gate.revoke_locked(update.version, admit_current=False)
            return PublicationReceipt(update, update.version)

    def activate(self, owner: str, update: VersionUpdate) -> bool:
        """Open the exact publication after global publication/RESUME acknowledgement.

        A delayed activation cannot reopen an epoch sealed by a newer revision.
        A revised scope requires RESUME even when ADVANCE supplied no ALIGN input.
        """
        if not isinstance(update, VersionUpdate):
            raise TypeError("frontend activation requires a version update")
        key = (update.version.tenant_id, update.version.scope_id)
        with self._lock:
            self._require_live_locked(owner)
            if self._observed.get(key) != update:
                return False
            if not update.admit_requests:
                raise ExecutionAuthorityError("execution is not admitted at this placement")
            floor = self._floors.get(key)
            if floor is not None:
                if update.version != floor:
                    return False
                if update.resume is None:
                    raise ExecutionAuthorityError("successor is pending ALIGN and RESUME")
            self._active[key] = update
            return True

    @contextmanager
    def admission(self, version: ExecutionVersion) -> Iterator[HostExecution]:
        """Order local enqueue and output commits with remote sealing and expiry."""
        with self._gate.admission(version) as execution:
            yield execution

    def expire(self) -> bool:
        """Poll from the endpoint's independent host loop to notify pending consumers."""
        with self._lock:
            self._expire_locked()
            return self._closed

    def close(self) -> None:
        """Revoke host effects without acknowledging or releasing physical resources."""
        with self._lock:
            self._closed = True
            self._gate.close_locked()

    def release(self, owner: str) -> None:
        """Let only the bound owner close authority; exact retries are harmless."""
        with self._lock:
            if self._owner is None or self._owner != owner:
                raise ValueError("message belongs to another owner incarnation")
            self.close()

    def _require_live_locked(self, owner: str | None = None) -> None:
        self._expire_locked()
        if self._closed or self._owner is None:
            raise ExecutionAuthorityError("frontend authority is unavailable")
        if owner is not None and owner != self._owner:
            raise ValueError("message belongs to another owner incarnation")

    def _expire_locked(self) -> None:
        if self._owner is not None and not self._closed and self._clock() >= self._deadline:
            self.close()

    def _current_locked(self, version: ExecutionVersion) -> ExecutionVersion | None:
        self._require_live_locked()
        active = self._active.get((version.tenant_id, version.scope_id))
        return None if active is None else active.version

    def _protocol_failure_locked(self, message: str) -> None:
        self.close()
        raise ExecutionAuthorityError(message)
