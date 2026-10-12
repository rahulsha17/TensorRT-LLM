# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated, authenticated IPC for host-side execution-version publication."""

import math
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty
from threading import Event, Thread
from time import monotonic

import zmq

from ..execution_mailbox import PublicationReceipt, VersionUpdate, WorkerVersionTable
from .ipc import IpcQueue

__all__ = ["ExecutionMailboxServer", "IpcPublicationTransport"]


@dataclass(frozen=True)
class _PublicationFailure:
    update: VersionUpdate
    message: str


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

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        """Send and await a matching acknowledgement, ignoring replies to earlier attempts."""
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
            if not isinstance(response, (PublicationReceipt, _PublicationFailure)):
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
    """

    def __init__(self, table: WorkerVersionTable) -> None:
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
            queue = IpcQueue(is_server=True, name="retire_mailbox")
            queue.socket.setsockopt(zmq.LINGER, 0)
            queue.socket.setsockopt(zmq.SNDHWM, 256)
            self._ready.set_result(queue.address)
            while not self._stop.is_set():
                try:
                    update = queue.get(timeout=0.05)
                except Empty:
                    continue
                if not isinstance(update, VersionUpdate):
                    raise ValueError("invalid execution publication")
                try:
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
