# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare an executor frontend for a runtime owner in another process."""

from dataclasses import dataclass
from threading import Event

from ..execution_alignment import AlignmentReceipt, AlignmentRequest
from ..execution_fence import FenceReceipt, FenceRequest
from ..execution_mailbox import (
    PublicationReceipt,
    PublicationTransport,
    VersionUpdate,
    WorkerIdentity,
)
from .execution_control import ExecutionControlMixin
from .execution_frontend import FrontendAuthorityGate
from .execution_frontend_ipc import (
    FrontendAuthorityClient,
    FrontendAuthorityEndpoint,
    FrontendAuthorityServer,
    _Argument,
    _Command,
    _Operation,
    _Value,
)
from .execution_group import _worker_keys

__all__ = ["ExecutionMemberEndpoint", "prepare_execution_member"]


@dataclass(frozen=True)
class ExecutionMemberEndpoint:
    """Authenticated bootstrap result for one frontend and its fixed worker group.

    Pass this descriptor only through the runtime's authenticated bootstrap.
    Both endpoints include secrets, hidden from their representations.
    """

    host: FrontendAuthorityEndpoint
    publication: FrontendAuthorityEndpoint

    def __post_init__(self) -> None:
        if not isinstance(self.host, FrontendAuthorityEndpoint) or not isinstance(
            self.publication, FrontendAuthorityEndpoint
        ):
            raise TypeError("execution member requires host and publication endpoints")
        if self.host.identity != self.publication.identity:
            raise ValueError("execution member endpoints belong to different frontend lifetimes")
        if self.host.address[0] == self.publication.address[0]:
            raise ValueError("host authority must not share the worker publication listener")


class MemberPublicationClient(FrontendAuthorityClient):
    """Keep worker traffic separate from host sealing and heartbeat connections."""

    def __init__(
        self, endpoint: FrontendAuthorityEndpoint, owner: str, *, timeout: float = 5.0
    ) -> None:
        super().__init__(endpoint, owner, timeout=timeout)
        self._closed = False

    def _exchange(
        self, operation: _Operation, argument: _Argument = None, *, stop: Event | None = None
    ) -> _Value:
        if self._closed:
            raise RuntimeError("execution member publication channel is closed")
        result = super()._exchange(operation, argument, stop=stop)
        if self._closed:
            raise RuntimeError("execution member publication channel is closed")
        return result

    def describe(self) -> tuple[WorkerIdentity, ...]:
        workers = self._exchange("describe")
        if (
            not isinstance(workers, tuple)
            or not workers
            or any(
                not isinstance(worker, WorkerIdentity)
                or worker.session_id != self._endpoint.identity.session_id
                for worker in workers
            )
            or len(set(workers)) != len(workers)
        ):
            raise ValueError("invalid execution member worker identities")
        return workers

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        value = self._exchange("publish", update, stop=stop)
        if not isinstance(value, PublicationReceipt) or value.update != update:
            raise ValueError("member acknowledgement does not match publication")
        return value

    def poll_fence(self, request: FenceRequest, stop: Event) -> FenceReceipt | None:
        value = self._exchange("fence", request, stop=stop)
        if value is not None and (not isinstance(value, FenceReceipt) or value.request != request):
            raise ValueError("member acknowledgement does not match FENCE")
        return value

    def poll_alignment(self, request: AlignmentRequest, stop: Event) -> AlignmentReceipt | None:
        value = self._exchange("alignment", request, stop=stop)
        if value is not None and (
            not isinstance(value, AlignmentReceipt) or value.request != request
        ):
            raise ValueError("member acknowledgement does not match ALIGN")
        return value

    def close(self) -> None:
        # Each exchange owns its socket and closes it before returning. Host
        # authority is released separately by the runtime coordinator.
        self._closed = True


class MemberPublicationServer(FrontendAuthorityServer):
    """Forward to this frontend's workers on a dedicated, socket-owning thread."""

    def __init__(
        self,
        gate: FrontendAuthorityGate,
        transport: PublicationTransport,
        workers: tuple[WorkerIdentity, ...],
        *,
        bind_host: str | None = None,
    ) -> None:
        self._transport = transport
        self._workers = workers
        super().__init__(gate, bind_host=bind_host)

    def _dispatch(self, command: _Command) -> _Value:
        if command.identity != self._gate.identity:
            raise ValueError("publication belongs to a different frontend lifetime")
        if (
            not isinstance(command.owner, str)
            or not command.owner
            or not isinstance(command.request_id, str)
            or not command.request_id
        ):
            raise ValueError("publication requires an owner and request identity")
        self._gate.check_owner(command.owner)
        argument = command.argument
        if command.operation == "describe" and argument is None:
            return self._workers
        if not isinstance(argument, (VersionUpdate, FenceRequest, AlignmentRequest)):
            raise TypeError("invalid worker publication command")
        if argument.session_id != self._gate.identity.session_id:
            raise ValueError("worker command belongs to another authority session")
        if command.operation == "publish" and isinstance(argument, VersionUpdate):
            result = self._transport.publish(argument, self._stop)
        elif command.operation == "fence" and isinstance(argument, FenceRequest):
            result = self._transport.poll_fence(argument, self._stop)
        elif command.operation == "alignment" and isinstance(argument, AlignmentRequest):
            result = self._transport.poll_alignment(argument, self._stop)
        else:
            raise ValueError("invalid worker publication command")
        # The host listener can expire or seal authority during worker I/O.
        # Sealing does not invalidate completion evidence; owner loss does.
        self._gate.check_owner(command.owner)
        return result

    def _run(self) -> None:
        try:
            super()._run()
        finally:
            try:
                self._transport.close()
            except Exception as error:
                self._failure = error
                self._gate.close()


def prepare_execution_member(
    frontend: ExecutionControlMixin, session_id: str, *, bind_host: str | None = None
) -> ExecutionMemberEndpoint:
    """Prepare this process's frontend before admission and export its fixed membership.

    Args:
        frontend: A running executor frontend. Existing worker/topology support
            checks still apply; this does not enable KV transfers or connectors.
        session_id: One fresh runtime-selected session for all members.
        bind_host: Concrete IPv4 address reachable by the authority owner. Worker
            endpoints may remain local because this process forwards their traffic.

    Exact bootstrap retries return the original endpoints. Partial failures leave
    the frontend unavailable until restart. Prepared members cannot create local
    controllers. After preparing the complete set, the runtime creates one remote
    execution controller from these descriptors and keeps it alive until teardown.
    """
    if not isinstance(frontend, ExecutionControlMixin):
        raise TypeError("execution member requires an executor frontend")
    gate = FrontendAuthorityGate(session_id)
    with frontend._execution_control_lock:
        frontend._require_execution_control_open()
        existing = frontend._execution_member_endpoint
        if existing is not None:
            if (
                existing.host.identity.session_id != session_id
                or frontend._execution_member_bind_host != bind_host
            ):
                raise ValueError("execution member bootstrap retry changed its binding")
            frontend._execution_frontend_server.check_health()
            frontend._execution_member_server.check_health()
            return existing
        if (
            frontend._execution_submit_started
            or frontend._execution_controller is not None
            or frontend._execution_version_table is not None
            or frontend._execution_mailbox_server is not None
            or frontend._execution_frontend_gate is not None
        ):
            raise RuntimeError("remote authority must start before admission or worker binding")
        transport = None
        host_server = member_server = None
        try:
            transport = frontend._create_execution_transport(session_id)
            workers = tuple(
                WorkerIdentity(session_id, worker)
                for worker in sorted(_worker_keys(transport, session_id))
            )
            host_server = FrontendAuthorityServer(gate, bind_host=bind_host)
            member_server = MemberPublicationServer(gate, transport, workers, bind_host=bind_host)
            endpoint = ExecutionMemberEndpoint(host_server.endpoint, member_server.endpoint)
            frontend._execution_frontend_gate = gate
            frontend._execution_frontend_server = host_server
            frontend._execution_member_server = member_server
            frontend._execution_member_endpoint = endpoint
            frontend._execution_member_bind_host = bind_host
            return endpoint
        except Exception as error:
            frontend._execution_control_error = error
            gate.close()
            cleanup = []
            if host_server is not None:
                cleanup.append(host_server.close)
            if member_server is not None:
                cleanup.append(member_server.close)
            elif transport is not None:
                cleanup.append(transport.close)
            failed = False
            for close in cleanup:
                try:
                    close()
                except Exception:
                    failed = True
            if failed:
                raise RuntimeError(
                    "execution member bootstrap and cleanup failed; restart required"
                ) from error
            raise
