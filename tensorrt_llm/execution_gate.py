# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host admission and response commits ordered with execution authority changes."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from weakref import WeakSet

from .execution_version import ExecutionVersion
from .logger import logger

__all__ = ["ExecutionAuthorityError", "HostExecution", "HostExecutionGate"]


class ExecutionAuthorityError(RuntimeError):
    """The execution has no current authority, or its authority is unavailable."""


class HostExecution:
    """An admitted request whose host effects remain conditional on authority.

    The executor binds wakeup and cancellation callbacks while holding admission.
    Callbacks only enqueue notifications; they must not wait for inference or
    call back into authority. Cancellation is a request, not proof of FENCE.
    """

    def __init__(self, gate: "HostExecutionGate", version: ExecutionVersion) -> None:
        self._gate = gate
        self.version = version
        self._error: ExecutionAuthorityError | None = None
        self._wake: Callable[[ExecutionAuthorityError], None] | None = None
        self._cancel: Callable[[], None] | None = None

    def check(self) -> None:
        """Check at a result-delivery boundary; earlier results cannot be retracted."""
        self.commit(lambda: None)

    def commit(self, action: Callable[[], None]) -> None:
        """Commit a short host effect under the same lock as authority changes.

        The action must not block on inference, invoke user code, or reenter the
        controller. Executor use is limited to inserting into unbounded queues.
        """
        with self._gate.lock:
            if self._error is not None:
                raise self._error
            self._gate._check_locked(self.version)
            action()

    def finish(self) -> None:
        """Detach notifications after a terminal response; buffered reads still check authority."""
        with self._gate.lock:
            self._wake = None
            self._cancel = None
            self._gate._executions.discard(self)

    def _revoke_locked(self, error: ExecutionAuthorityError) -> None:
        if self._error is not None:
            return
        self._error = error
        wake, cancel = self._wake, self._cancel
        self._wake = self._cancel = None
        # Wake the consumer even if native cancellation reports a failure.
        if wake is not None:
            try:
                wake(error)
            except Exception as wake_error:
                logger.error(f"Failed to notify revoked execution: {wake_error}")
        if cancel is not None:
            try:
                cancel()
            except Exception as cancel_error:
                logger.error(f"Failed to enqueue revoked execution cancellation: {cancel_error}")


class HostExecutionGate:
    """Share the owner's lock instead of maintaining a second authority registry.

    Args:
        lock: Lock used by the owning controller or worker table.
        current: Read current authority with that lock already held.
    """

    def __init__(
        self,
        lock: AbstractContextManager[object],
        current: Callable[[ExecutionVersion], ExecutionVersion | None],
    ) -> None:
        self.lock = lock
        self._current = current
        self._closed = False
        self._executions: WeakSet[HostExecution] = WeakSet()

    def _check_locked(self, version: ExecutionVersion) -> None:
        if self._closed:
            raise ExecutionAuthorityError("execution authority is closed")
        current = self._current(version)
        if current != version:
            raise ExecutionAuthorityError(
                f"execution {version!r} is not current; authority is {current!r}"
            )

    @contextmanager
    def admission(self, version: ExecutionVersion) -> Iterator[HostExecution]:
        """Hold authority through the actual enqueue and callback binding."""
        with self.lock:
            self._check_locked(version)
            execution = HostExecution(self, version)
            self._executions.add(execution)
            try:
                yield execution
            except BaseException:
                self._executions.discard(execution)
                execution._wake = execution._cancel = None
                raise

    def revoke_locked(self, current: ExecutionVersion) -> None:
        """Invalidate requests in the changed scope while the owner holds its lock."""
        for execution in list(self._executions):
            version = execution.version
            if (version.tenant_id, version.scope_id) == (
                current.tenant_id,
                current.scope_id,
            ) and version != current:
                execution._revoke_locked(
                    ExecutionAuthorityError(f"execution {version!r} superseded by {current!r}")
                )
                self._executions.discard(execution)

    def close_locked(self) -> None:
        """Reject future commits and unblock outstanding consumers."""
        self._closed = True
        for execution in list(self._executions):
            execution._revoke_locked(ExecutionAuthorityError("execution authority is closed"))
        self._executions.clear()
