# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated, authenticated IPC for host-side execution-version publication."""

import math
from concurrent.futures import Future
from dataclasses import dataclass
from ipaddress import ip_address
from queue import Empty
from threading import Event, Thread
from time import monotonic

import zmq

from ..execution_alignment import AlignmentReceipt, AlignmentRequest
from ..execution_fence import FenceReceipt, FenceRequest
from ..execution_mailbox import (
    PublicationReceipt,
    VersionUpdate,
    WorkerIdentity,
    WorkerVersionTable,
)
from .ipc import IpcQueue

__all__ = ["ExecutionMailboxServer", "IpcPublicationTransport"]


@dataclass(frozen=True)
class _IdentityQuery:
    session_id: str


@dataclass(frozen=True)
class _IdentityStatus:
    update: _IdentityQuery
    identity: WorkerIdentity


@dataclass(frozen=True)
class _PublicationFailure:
    update: VersionUpdate | FenceRequest | AlignmentRequest | _IdentityQuery
    message: str


@dataclass(frozen=True)
class _FenceStatus:
    update: FenceRequest
    receipt: FenceReceipt | None


@dataclass(frozen=True)
class _AlignmentStatus:
    update: AlignmentRequest
    receipt: AlignmentReceipt | None


class IpcPublicationTransport:
    """Connect to a worker mailbox from the publisher's owning thread.

    Args:
        address: Existing IpcQueue endpoint and authentication key.
        timeout: Maximum seconds for each send/acknowledgement round trip.
    """

    def __init__(self, address: tuple[str, bytes], timeout: float = 5.0) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("publication timeout must be positive and finite")
        self._address = address
        self._timeout = timeout
        self._queue: IpcQueue | None = None

    def identify(self, session_id: str) -> WorkerIdentity:
        """Inspect a prepared worker using a separate, bounded bootstrap connection.

        The temporary socket stays on the bootstrap thread. This transport's
        publication socket is still created later by its owning publisher.
        """
        query = _IdentityQuery(session_id)
        probe = IpcPublicationTransport(self._address, timeout=self._timeout)
        try:
            response = probe._exchange(query, Event())
            if (
                not isinstance(response, _IdentityStatus)
                or response.update != query
                or not isinstance(response.identity, WorkerIdentity)
                or response.identity.session_id != session_id
            ):
                raise ValueError("worker identity does not match authority bootstrap")
            return response.identity
        finally:
            probe.close()

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        """Send and await a matching acknowledgement, ignoring replies to earlier attempts."""
        response = self._exchange(update, stop)
        if not isinstance(response, PublicationReceipt):
            raise ValueError("invalid publication response")
        return response

    def poll_fence(self, request: FenceRequest, stop: Event) -> FenceReceipt | None:
        response = self._exchange(request, stop)
        if not isinstance(response, _FenceStatus):
            raise ValueError("invalid fence response")
        return response.receipt

    def poll_alignment(self, request: AlignmentRequest, stop: Event) -> AlignmentReceipt | None:
        response = self._exchange(request, stop)
        if not isinstance(response, _AlignmentStatus):
            raise ValueError("invalid ALIGN response")
        return response.receipt

    def _exchange(
        self, update: VersionUpdate | FenceRequest | AlignmentRequest | _IdentityQuery, stop: Event
    ) -> PublicationReceipt | _FenceStatus | _AlignmentStatus | _IdentityStatus:
        """Exchange one bounded status query on the publisher's socket."""
        if self._queue is None:
            self._queue = IpcQueue(self._address, is_server=False, name="retire_publisher")
            self._queue.socket.setsockopt(zmq.LINGER, 0)
            self._queue.socket.setsockopt(zmq.SNDHWM, 256)
        deadline = monotonic() + self._timeout
        while not stop.is_set():
            if monotonic() >= deadline:
                raise TimeoutError("execution version publication timed out")
            try:
                self._queue.put_nowait(update)
                break
            except zmq.Again:
                stop.wait(0.01)
        while not stop.is_set():
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("execution version acknowledgement timed out")
            try:
                response = self._queue.get(timeout=min(remaining, 0.05))
            except Empty:
                continue
            if not isinstance(
                response,
                (
                    PublicationReceipt,
                    _PublicationFailure,
                    _FenceStatus,
                    _AlignmentStatus,
                    _IdentityStatus,
                ),
            ):
                raise ValueError("invalid execution publication acknowledgement")
            if response.update != update:
                continue
            if isinstance(response, _PublicationFailure):
                raise RuntimeError(response.message)
            return response
        raise RuntimeError("execution controller is closed")

    def close(self) -> None:
        """Close the socket on its publisher thread without waiting for a dead peer."""
        if self._queue is not None:
            self._queue.close()
            self._queue = None


class ExecutionMailboxServer:
    """Record publications on a dedicated worker listener thread.

    Args:
        table: Host version table owned by this worker lifetime.
        bind_host: Concrete IPv4 worker address for Ray peers on another node.
            Omit to use the existing loopback endpoint.
    """

    def __init__(self, table: WorkerVersionTable, *, bind_host: str | None = None) -> None:
        self._bind_address = None
        if bind_host is not None:
            host = ip_address(bind_host)
            if host.version != 4 or host.is_unspecified or host.is_multicast:
                raise ValueError("execution mailbox requires a concrete worker address")
            self._bind_address = (f"tcp://{host}:*", None)
        self._table = table
        self._stop = Event()
        self._ready: Future[tuple[str, bytes]] = Future()
        self._failure: Exception | None = None
        self._thread = Thread(target=self._run, name="retire_mailbox", daemon=True)
        self._thread.start()
        try:
            self.address = self._ready.result(timeout=5)
        except Exception:
            self.close()
            raise

    def check_health(self) -> None:
        """Reject attachment to a stopped listener."""
        if self._stop.is_set() or not self._thread.is_alive():
            raise RuntimeError("execution mailbox is closed") from self._failure

    def close(self) -> None:
        """Invalidate the table and stop the listener before worker teardown."""
        self._stop.set()
        self._table.close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("execution mailbox listener did not stop")

    def _run(self) -> None:
        queue = None
        try:
            queue = IpcQueue(self._bind_address, is_server=True, name="retire_mailbox")
            queue.socket.setsockopt(zmq.LINGER, 0)
            queue.socket.setsockopt(zmq.SNDHWM, 256)
            self._ready.set_result(queue.address)
            while not self._stop.is_set():
                try:
                    update = queue.get(timeout=0.05)
                except Empty:
                    continue
                if not isinstance(
                    update, (VersionUpdate, FenceRequest, AlignmentRequest, _IdentityQuery)
                ):
                    raise ValueError("invalid execution publication")
                response: (
                    PublicationReceipt
                    | _FenceStatus
                    | _AlignmentStatus
                    | _IdentityStatus
                    | _PublicationFailure
                )
                try:
                    if isinstance(update, _IdentityQuery):
                        identity = self._table.identity
                        if identity.session_id != update.session_id:
                            raise ValueError("worker belongs to a different authority session")
                        response = _IdentityStatus(update, identity)
                    elif isinstance(update, AlignmentRequest):
                        response = _AlignmentStatus(update, self._table.poll_alignment(update))
                    elif isinstance(update, FenceRequest):
                        response = _FenceStatus(update, self._table.poll_fence(update))
                    else:
                        response = self._table.apply(update)
                except (TypeError, ValueError, RuntimeError) as error:
                    response = _PublicationFailure(update, str(error))
                try:
                    queue.put_nowait(response)
                except zmq.Again:
                    # The owner can retry the same update if its ack is lost.
                    continue
        except Exception as error:
            # Surface listener failure at the lifecycle boundary. No worker
            # observation remains usable after this thread exits.
            self._failure = error
            if not self._ready.done():
                self._ready.set_exception(error)
        finally:
            self._table.close()
            if queue is not None:
                queue.close()
