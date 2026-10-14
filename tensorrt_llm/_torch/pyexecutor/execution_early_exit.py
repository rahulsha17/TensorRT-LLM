# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-thread early-exit points outside CUDA capture and compilation."""

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from ...logger import logger

__all__ = [
    "ExecutionEarlyExit",
    "early_exit_scope",
    "execution_early_exit_point",
    "get_layer_early_exit_point",
    "execution_trace_enabled",
    "trace_execution",
]


class ExecutionEarlyExit(Exception):
    """The entire physical batch lost authority; discard its unfinished forward."""

    def __init__(self, boundary: str = "prelaunch", submitted: int = 0, total: int = 0) -> None:
        super().__init__(boundary)
        self.boundary = boundary
        self.submitted = submitted
        self.total = total


_layer_checks: ContextVar[bool] = ContextVar("execution_layer_checks", default=False)

_poll: ContextVar[Callable[[], bool] | None] = ContextVar("execution_early_exit", default=None)
_observer: ContextVar[Callable[[str, int, int, bool], None] | None] = ContextVar(
    "execution_boundary_observer", default=None
)


def execution_trace_enabled() -> bool:
    return logger.is_debug_enabled()


def trace_execution(phase: str, **fields: object) -> None:
    """Emit worker-local host timing; this does not measure GPU completion."""
    if execution_trace_enabled():
        logger.debug(
            "RETIRE_TRACE "
            + json.dumps({"phase": phase, "monotonic_ns": time.monotonic_ns(), **fields})
        )


@contextmanager
def early_exit_scope(
    poll: Callable[[], bool] | None,
    *,
    layer_checks: bool = True,
    observer: Callable[[str, int, int, bool], None] | None = None,
) -> Iterator[None]:
    """Bind a batch-local poll, restoring it even if model execution raises."""
    token = _poll.set(poll)
    layer_token = _layer_checks.set(layer_checks)
    observer_token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(observer_token)
        _poll.reset(token)
        _layer_checks.reset(layer_token)


def execution_early_exit_point(
    boundary: str = "prelaunch", submitted: int = 0, total: int = 0
) -> None:
    """Stop submission only when every real row is irrevocably obsolete.

    Call only outside compilation and graph capture. Unwinding does not release
    KV or prove GPU completion; the executor still records and drains all uses.
    """
    poll = _poll.get()
    obsolete = poll is not None and poll()
    observer = _observer.get()
    if observer is not None:
        observer(boundary, submitted, total, obsolete)
    if obsolete:
        raise ExecutionEarlyExit(boundary, submitted, total)


def get_layer_early_exit_point() -> Callable[..., None] | None:
    """Return the active prefill hook, or None for unarmed/short decode batches."""
    return execution_early_exit_point if _layer_checks.get() and _poll.get() is not None else None
