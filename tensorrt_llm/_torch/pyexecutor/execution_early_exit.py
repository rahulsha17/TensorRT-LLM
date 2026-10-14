# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-thread early-exit points outside CUDA capture and compilation."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = [
    "ExecutionEarlyExit",
    "early_exit_scope",
    "execution_early_exit_point",
    "get_layer_early_exit_point",
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


@contextmanager
def early_exit_scope(
    poll: Callable[[], bool] | None, *, layer_checks: bool = True
) -> Iterator[None]:
    """Bind a batch-local poll, restoring it even if model execution raises."""
    token = _poll.set(poll)
    layer_token = _layer_checks.set(layer_checks)
    try:
        yield
    finally:
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
    if poll is not None and poll():
        raise ExecutionEarlyExit(boundary, submitted, total)


def get_layer_early_exit_point() -> Callable[..., None] | None:
    """Return the active prefill hook, or None for unarmed/short decode batches."""
    return execution_early_exit_point if _layer_checks.get() and _poll.get() is not None else None
