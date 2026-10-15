# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One runtime authority across executor frontends in separate processes."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, RLock, Thread, current_thread
from typing import TypeVar
from uuid import uuid4

from ..execution_alignment import AlignmentInput, AlignmentReceipt, AlignmentRequest
from ..execution_authority import Revision
from ..execution_collective import ExecutionParticipant
from ..execution_fence import FenceReceipt, FenceRequest
from ..execution_gate import ExecutionAuthorityError
from ..execution_mailbox import (
    ExecutionController,
    PublicationReceipt,
    PublicationTransport,
    RevisionPublication,
    VersionPublication,
    VersionUpdate,
)
from ..execution_placement import PlacementPublicationTransport, placement_update
from ..execution_version import ExecutionVersion
from .execution_frontend_ipc import FrontendAuthorityClient, _timeout
from .execution_member import ExecutionMemberEndpoint, MemberPublicationClient

__all__ = ["create_remote_execution_controller"]
_T = TypeVar("_T")


def _all_members(operation: str, calls: tuple[Callable[[], _T], ...]) -> tuple[_T, ...]:
    """Attempt every fixed member concurrently, including after a peer fails."""
    failures = []
    results = []
    with ThreadPoolExecutor(len(calls), thread_name_prefix="retire_remote") as pool:
        pending = [pool.submit(call) for call in calls]
        for index, future in enumerate(pending):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append((index, error))
    if failures:
        raise RuntimeError(
            f"{operation} failed for execution members {tuple(index for index, _ in failures)}"
        ) from failures[0][1]
    return tuple(results)


class _FrontendCohort:
    def __init__(
        self, members: tuple[ExecutionMemberEndpoint, ...], owner: str, timeout: float, lease: float
    ) -> None:
        self.placements = tuple(member.placement for member in members)
        self.clients = tuple(
            FrontendAuthorityClient(m.host, owner, timeout=timeout) for m in members
        )
        self._release_clients = tuple(
            FrontendAuthorityClient(m.host, owner, timeout=min(timeout, 0.5)) for m in members
        )
        self._lease = lease
        self._lock = RLock()
        self._stop = Event()
        self._closed = False
        self._failure: Exception | None = None
        self._fail_owner: Callable[[Exception], None] | None = None
        self._initial: dict[tuple[str, str], ExecutionVersion] = {}
        self._thread: Thread | None = None

    def _require_open(self) -> None:
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("remote frontend authority is closed") from self._failure

    def _call(self, operation: str, calls: tuple[Callable[[], _T], ...]) -> tuple[_T, ...]:
        self._require_open()
        try:
            result = _all_members(operation, calls)
        except Exception as error:
            self.fail(error)
            raise
        self._require_open()
        return result

    def bind(self) -> None:
        self._call("binding", tuple(lambda c=c: c.bind(self._lease) for c in self.clients))

    def start(self, fail_owner: Callable[[Exception], None]) -> None:
        with self._lock:
            self._require_open()
            self._fail_owner = fail_owner
            self._thread = Thread(target=self._heartbeat, name="retire_owner_lease", daemon=True)
            self._thread.start()

    def declare(self, version: ExecutionVersion) -> None:
        self._call("scope declaration", tuple(lambda c=c: c.declare(version) for c in self.clients))
        with self._lock:
            self._initial[(version.tenant_id, version.scope_id)] = version

    def seal(self, revision: Revision) -> None:
        self._call("host seal", tuple(lambda c=c: c.seal(revision) for c in self.clients))

    def acknowledge(self, update: VersionUpdate) -> None:
        local = tuple(placement_update(update, p) for p in self.placements)
        self._call(
            "host observation",
            tuple(lambda c=c, u=u: c.observe(u) for c, u in zip(self.clients, local)),
        )
        with self._lock:
            initial = self._initial.get((update.version.tenant_id, update.version.scope_id))
        activations = tuple(
            lambda c=c, u=u: c.activate(u)
            for c, u in zip(self.clients, local)
            if u.admit_requests and (u.resume is not None or u.version == initial)
        )
        if activations:
            opened = self._call("host activation", activations)
            if not all(opened):
                # A concurrent newer seal may legitimately overtake publication.
                # No stale activation can reopen a sealed frontend.
                raise ExecutionAuthorityError(
                    "frontend publication was superseded before activation"
                )

    def fail(self, error: Exception) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error
            self._stop.set()
            fail_owner = self._fail_owner
        if fail_owner is not None:
            fail_owner(error)

    def _heartbeat(self) -> None:
        sequence = 0
        while not self._stop.wait(self._lease / 4):
            sequence += 1
            try:
                self._call(
                    "heartbeat",
                    tuple(lambda c=c: c.renew(sequence, stop=self._stop) for c in self.clients),
                )
            except Exception:
                # _call has failed the owner; its publisher closes the cohort.
                return

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            heartbeat = self._thread
        try:
            _all_members("host release", tuple(c.release for c in self._release_clients))
        except RuntimeError:
            # An unreachable frontend still expires its lease. No release ACK
            # is used as proof of physical reclamation or permission to restart.
            pass
        if heartbeat is not None and heartbeat is not current_thread():
            heartbeat.join(timeout=1)


class _RemotePublicationTransport:
    def __init__(self, workers: PublicationTransport, frontends: _FrontendCohort) -> None:
        self._workers = workers
        self._frontends = frontends

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        self._frontends._require_open()
        receipt = self._workers.publish(update, stop)
        if stop.is_set():
            raise RuntimeError("remote authority stopped before publication acknowledgement")
        # Worker publication must be globally complete before any frontend opens
        # the initial version or a resumed successor.
        self._frontends.acknowledge(update)
        return receipt

    def poll_fence(self, request: FenceRequest, stop: Event) -> FenceReceipt | None:
        return self._workers.poll_fence(request, stop)

    def poll_alignment(self, request: AlignmentRequest, stop: Event) -> AlignmentReceipt | None:
        return self._workers.poll_alignment(request, stop)

    def close(self) -> None:
        try:
            self._workers.close()
        finally:
            self._frontends.close()


class _RemoteExecutionController(ExecutionController):
    def __init__(
        self, session: str, transport: PublicationTransport, frontends: _FrontendCohort
    ) -> None:
        self._transitions = Lock()
        self._frontends = frontends
        super().__init__(session, transport, placements=frozenset(frontends.placements))

    def register(self, version: ExecutionVersion) -> VersionPublication:
        if not isinstance(version, ExecutionVersion):
            raise TypeError("version must be an ExecutionVersion")
        with self._transitions:
            with self._lock:
                self._require_open()
                try:
                    current = self._registry.snapshot(version.tenant_id, version.scope_id)
                except KeyError:
                    current = None
                if current is not None and current.current_version != version:
                    raise ValueError("scope is already registered at a different epoch")
            if current is None:
                self._frontends.declare(version)
            return super().register(version)

    def advance(
        self, revision: Revision, *, alignment: AlignmentInput | None = None
    ) -> RevisionPublication:
        with self._transitions:
            with self._lock:
                self._require_open()
                self._validate_alignment(alignment)
                previous = self._registry.check_advance(revision)
                if (
                    revision in self._alignment_inputs
                    and self._alignment_inputs[revision] != alignment
                ):
                    raise ValueError("revision retry changed its revised input")
            if previous is None:
                # No registry/host-output lock is held across these bounded RPCs.
                # The registry CAS is repeated only after all frontend seal ACKs.
                self._frontends.seal(revision)
            return super().advance(revision, alignment=alignment)

    def align(self, revision: Revision, alignment: AlignmentInput) -> RevisionPublication:
        with self._transitions:
            return super().align(revision, alignment)

    def resume(self, aligned: AlignmentReceipt) -> VersionPublication:
        with self._transitions:
            return super().resume(aligned)


def create_remote_execution_controller(
    members: tuple[ExecutionMemberEndpoint, ...],
    *,
    timeout: float = 5.0,
    lease_seconds: float = 30.0,
) -> ExecutionController:
    """Bind one runtime authority to frontends prepared in separate processes.

    Args:
        members: Complete fixed membership returned by prepare_execution_member.
            All must share one fresh session and have distinct worker tables.
        timeout: Maximum time for each authenticated control round trip.
        lease_seconds: Owner lease at each frontend, at least four round trips.
            Heartbeats are independent of publication and renew all members in parallel.

    The caller owns the returned controller and uses the ordinary register,
    advance, align and resume APIs. ADVANCE validates first, waits for every host
    seal, then accepts the new epoch. Worker publication and physical FENCE remain
    asynchronous. Publication failure cannot roll back the accepted epoch.

    Partial host/control failure closes this authority. Recovery requires a new
    runtime session and normal worker teardown; a host lease cannot prove GPU
    drain. Untargeted ALIGN requires local reservations on every worker. An
    AlignmentInput.destination selects one member.placement; only that frontend
    activates the successor, after every member acknowledges publication/FENCE.
    This does not enable disaggregated transfers or remove device/topology guards.
    """
    timeout, lease_seconds = _timeout(timeout), _timeout(lease_seconds)
    if lease_seconds < 4 * timeout:
        raise ValueError("frontend lease must cover at least four control round trips")
    if not isinstance(members, tuple) or not members:
        raise ValueError("remote authority requires a nonempty tuple of members")
    if any(not isinstance(member, ExecutionMemberEndpoint) for member in members):
        raise TypeError("remote authority requires prepared execution member endpoints")
    identities = tuple(member.host.identity for member in members)
    session = identities[0].session_id
    if len(set(identities)) != len(members) or any(
        identity.session_id != session for identity in identities
    ):
        raise ValueError("remote authority requires distinct frontends in one session")
    addresses = [
        endpoint.address[0] for member in members for endpoint in (member.host, member.publication)
    ]
    if len(set(addresses)) != len(addresses):
        raise ValueError("remote authority contains duplicate control endpoints")

    owner = uuid4().hex
    frontends = _FrontendCohort(members, owner, timeout, lease_seconds)
    clients = tuple(MemberPublicationClient(m.publication, owner, timeout=timeout) for m in members)
    controller = None
    try:
        frontends.bind()
        descriptions = _all_members(
            "worker membership", tuple(client.describe for client in clients)
        )
        workers = [worker for group in descriptions for worker in group]
        if len(set(workers)) != len(workers):
            raise ValueError("remote authority contains a repeated worker table")
        participants = tuple(ExecutionParticipant(0, 0, group) for group in range(len(clients)))
        publication = PlacementPublicationTransport(
            participants,
            dict(zip(participants, clients)),
            dict(zip(participants, frontends.placements)),
        )
        controller = _RemoteExecutionController(
            session, _RemotePublicationTransport(publication, frontends), frontends
        )
        frontends.start(controller.fail)
        return controller
    except Exception:
        if controller is not None:
            controller.close()
        else:
            frontends.close()
        raise
