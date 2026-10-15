# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authenticated host-only control for frontends owned by another process.

The listener never waits for worker publication, FENCE, ALIGN, or GPU work. Its
lease loop can therefore close host admission while those operations are stuck.
The authority coordinator must seal every member before accepting ADVANCE, and
must activate only after the complete worker publication has been acknowledged.
This channel alone does not implement that coordinator or prove physical drain.
"""

import math
from concurrent.futures import Future
from dataclasses import dataclass, field
from ipaddress import ip_address
from queue import Empty
from threading import Event, Thread
from time import monotonic
from typing import Literal
from uuid import uuid4

import zmq

from ..execution_authority import Revision
from ..execution_mailbox import PublicationReceipt, VersionUpdate, WorkerIdentity
from .execution_frontend import FrontendAuthorityGate
from .ipc import IpcQueue

__all__ = ["FrontendAuthorityEndpoint", "FrontendAuthorityServer", "FrontendAuthorityClient"]

_Operation = Literal["bind", "renew", "seal", "observe", "activate", "release"]
_Argument = float | int | Revision | VersionUpdate | None
_Value = PublicationReceipt | Revision | bool | int | None


@dataclass(frozen=True)
class FrontendAuthorityEndpoint:
    """Pass through authenticated runtime bootstrap; the address contains a secret."""

    identity: WorkerIdentity
    address: tuple[str, bytes] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, WorkerIdentity):
            raise TypeError("frontend endpoint requires a lifetime identity")
        if (
            not isinstance(self.address, tuple)
            or len(self.address) != 2
            or not isinstance(self.address[0], str)
            or not self.address[0]
            or not isinstance(self.address[1], bytes)
            or not self.address[1]
        ):
            raise ValueError("frontend endpoint requires an authenticated address")


@dataclass(frozen=True)
class _Command:
    identity: WorkerIdentity
    owner: str
    request_id: str
    operation: _Operation
    argument: _Argument = None


@dataclass(frozen=True)
class _Reply:
    command: _Command
    value: _Value = None
    failure: str | None = None


def _timeout(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("frontend timeout must be positive and finite")
    return value


class FrontendAuthorityClient:
    """Issue bounded host transitions without sharing sockets across threads.

    Args:
        endpoint: Descriptor from the frontend's authenticated runtime bootstrap.
        owner: Unique incarnation of the one runtime authority coordinator.
        timeout: Maximum time for each round trip. Timeout is not an ACK.
    """

    def __init__(
        self, endpoint: FrontendAuthorityEndpoint, owner: str, *, timeout: float = 5.0
    ) -> None:
        if not isinstance(endpoint, FrontendAuthorityEndpoint):
            raise TypeError("frontend client requires an endpoint")
        if not isinstance(owner, str) or not owner:
            raise ValueError("frontend client requires an owner incarnation")
        self._endpoint = endpoint
        self._owner = owner
        self._timeout = _timeout(timeout)

    def bind(self, lease_seconds: float) -> None:
        """Bind this owner once; retrying cannot renew an old lease."""
        if self._exchange("bind", _timeout(lease_seconds)) is not None:
            raise ValueError("invalid frontend binding acknowledgement")

    def renew(self, sequence: int) -> int:
        """Send a monotonically increasing heartbeat; duplicate sequences do not renew."""
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("heartbeat sequence must be a positive integer")
        value = self._exchange("renew", sequence)
        if type(value) is not int or value < sequence:
            raise ValueError("invalid frontend heartbeat acknowledgement")
        return value

    def seal(self, revision: Revision) -> Revision:
        """Wait for old host effects to be revoked at this member only."""
        if not isinstance(revision, Revision):
            raise TypeError("seal requires a revision")
        value = self._exchange("seal", revision)
        if not isinstance(value, Revision) or value != revision:
            raise ValueError("invalid frontend seal acknowledgement")
        return value

    def observe(self, update: VersionUpdate) -> PublicationReceipt:
        """Record the exact worker publication without opening successor admission."""
        if not isinstance(update, VersionUpdate):
            raise TypeError("observation requires a version update")
        value = self._exchange("observe", update)
        if not isinstance(value, PublicationReceipt) or value.update != update:
            raise ValueError("invalid frontend observation acknowledgement")
        return value

    def activate(self, update: VersionUpdate) -> bool:
        """Open a globally acknowledged publication, unless already superseded."""
        if not isinstance(update, VersionUpdate):
            raise TypeError("activation requires a version update")
        value = self._exchange("activate", update)
        if type(value) is not bool:
            raise ValueError("invalid frontend activation acknowledgement")
        return value

    def release(self) -> None:
        """Close host authority. Neither acknowledgement nor timeout proves GPU drain."""
        if self._exchange("release") is not None:
            raise ValueError("invalid frontend release acknowledgement")

    def _exchange(self, operation: _Operation, argument: _Argument = None) -> _Value:
        command = _Command(self._endpoint.identity, self._owner, uuid4().hex, operation, argument)
        queue = IpcQueue(
            self._endpoint.address,
            is_server=False,
            socket_type=zmq.DEALER,
            name="retire_frontend_client",
        )
        queue.socket.setsockopt(zmq.LINGER, 0)
        queue.socket.setsockopt(zmq.SNDHWM, 256)
        queue.socket.setsockopt(zmq.RCVHWM, 256)
        deadline = monotonic() + self._timeout
        try:
            while monotonic() < deadline:
                try:
                    queue.put_nowait(command)
                    break
                except zmq.Again:
                    # Bound both connection setup and acknowledgement waits.
                    queue.socket.poll(timeout=1, flags=zmq.POLLOUT)
            while (remaining := deadline - monotonic()) > 0:
                try:
                    reply = queue.get(timeout=min(remaining, 0.05))
                except Empty:
                    continue
                if not isinstance(reply, _Reply) or reply.command != command:
                    raise ValueError("frontend reply does not match this command and lifetime")
                if reply.failure is not None:
                    raise RuntimeError(reply.failure)
                return reply.value
            raise TimeoutError("frontend authority acknowledgement timed out")
        finally:
            queue.close()


class FrontendAuthorityServer:
    """Serve short host transitions and expire authority independently of worker I/O.

    Args:
        gate: Local frontend admission/output gate, not its worker version table.
        bind_host: Concrete IPv4 address reachable by the runtime authority owner.
            Omit for loopback. Bootstrap must protect the returned authentication key.
    """

    def __init__(self, gate: FrontendAuthorityGate, *, bind_host: str | None = None) -> None:
        if not isinstance(gate, FrontendAuthorityGate):
            raise TypeError("frontend listener requires a host authority gate")
        self._bind_address = None
        if bind_host is not None:
            host = ip_address(bind_host)
            if host.version != 4 or host.is_unspecified or host.is_multicast:
                raise ValueError("frontend listener requires a concrete IPv4 address")
            self._bind_address = (f"tcp://{host}:*", None)
        self._gate = gate
        self._stop = Event()
        self._failure: Exception | None = None
        self._ready: Future[FrontendAuthorityEndpoint] = Future()
        self._thread = Thread(target=self._run, name="retire_frontend_authority", daemon=True)
        self._thread.start()
        try:
            self.endpoint = self._ready.result(timeout=5)
        except Exception:
            self.close()
            raise

    def check_health(self) -> None:
        """Reject stopped listeners and expired or released authority."""
        if self._stop.is_set() or not self._thread.is_alive() or self._gate.closed:
            raise RuntimeError("frontend authority listener is unavailable") from self._failure

    def close(self) -> None:
        """Close admission before stopping the host listener; retain all physical owners."""
        self._stop.set()
        self._gate.close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("frontend authority listener did not stop")

    def _dispatch(self, command: _Command) -> _Value:
        if command.identity != self._gate.identity:
            raise ValueError("command belongs to a different frontend lifetime")
        if not isinstance(command.owner, str) or not command.owner or not command.request_id:
            raise ValueError("command requires an owner incarnation and request identity")
        owner, argument = command.owner, command.argument
        if command.operation == "bind":
            if not isinstance(argument, (int, float)):
                raise TypeError("binding requires a lease duration")
            self._gate.bind(owner, argument)
        elif command.operation == "renew":
            if type(argument) is not int:
                raise TypeError("heartbeat requires an integer sequence")
            return self._gate.renew(owner, argument)
        elif command.operation == "seal":
            if not isinstance(argument, Revision):
                raise TypeError("seal requires a revision")
            return self._gate.seal(owner, argument)
        elif command.operation == "observe":
            if not isinstance(argument, VersionUpdate):
                raise TypeError("observation requires a version update")
            return self._gate.observe(owner, argument)
        elif command.operation == "activate":
            if not isinstance(argument, VersionUpdate):
                raise TypeError("activation requires a version update")
            return self._gate.activate(owner, argument)
        elif command.operation == "release" and argument is None:
            self._gate.release(owner)
        else:
            raise ValueError("invalid frontend command")
        return None

    def _run(self) -> None:
        queue = None
        try:
            queue = IpcQueue(
                self._bind_address,
                is_server=True,
                socket_type=zmq.ROUTER,
                name="retire_frontend_authority",
            )
            queue.socket.setsockopt(zmq.LINGER, 0)
            queue.socket.setsockopt(zmq.SNDHWM, 256)
            queue.socket.setsockopt(zmq.RCVHWM, 256)
            self._ready.set_result(FrontendAuthorityEndpoint(self._gate.identity, queue.address))
            while not self._stop.is_set():
                self._gate.expire()
                try:
                    command = queue.get(timeout=0.05)
                except Empty:
                    continue
                if not isinstance(command, _Command):
                    raise ValueError("invalid frontend authority command")
                try:
                    reply = _Reply(command, self._dispatch(command))
                except (TypeError, ValueError, RuntimeError) as error:
                    reply = _Reply(command, failure=str(error))
                try:
                    queue.put_nowait(reply)
                except zmq.Again:
                    # A lost ACK is never permission to accept a revision. The
                    # coordinator may retry exact metadata or close authority.
                    continue
        except Exception as error:
            self._failure = error
            if not self._ready.done():
                self._ready.set_exception(error)
        finally:
            self._gate.close()
            if queue is not None:
                queue.close()
