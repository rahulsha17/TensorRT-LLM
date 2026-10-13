# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable revised input and worker-owned KV reservations for RETIRE ALIGN."""

from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol
from uuid import uuid4

from .execution_authority import Revision
from .execution_version import ExecutionVersion

__all__ = [
    "AlignmentInput",
    "AlignmentRequest",
    "AlignmentReceipt",
    "RevisionAlignment",
    "AlignmentSafetyError",
    "PrefixReservation",
    "WorkerAlignmentState",
]


@dataclass(frozen=True)
class AlignmentInput:
    """The runtime's complete revised text input and cache namespace.

    Tokens must be immutable. The final input token is always recomputed to
    obtain logits. min_reuse_tokens can require a prefix instead of allowing
    recomputation on a cache miss. Salt and LoRA identity use ordinary TRT-LLM
    reuse semantics; neither is derived from the execution epoch.
    """

    input_token_ids: tuple[int, ...]
    cache_salt: str | None = None
    lora_task_id: int | None = None
    min_reuse_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.input_token_ids, tuple) or not self.input_token_ids:
            raise ValueError("input_token_ids must be a nonempty tuple")
        if any(type(t) is not int or not 0 <= t < 2**31 for t in self.input_token_ids):
            raise ValueError("input tokens must be nonnegative int32 integers")
        if self.cache_salt is not None and not isinstance(self.cache_salt, str):
            raise TypeError("cache_salt must be a string or None")
        if self.lora_task_id is not None and (
            type(self.lora_task_id) is not int or not 0 <= self.lora_task_id < 2**64
        ):
            raise ValueError("lora_task_id must be a uint64 integer or None")
        if type(self.min_reuse_tokens) is not int or not (
            0 <= self.min_reuse_tokens < len(self.input_token_ids)
        ):
            raise ValueError("min_reuse_tokens must leave at least one input token to compute")


@dataclass(frozen=True)
class AlignmentRequest:
    """Bind revised input to one accepted transition and executor lifetime."""

    session_id: str
    revision: Revision
    revised_input: AlignmentInput

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a nonempty string")
        if not isinstance(self.revision, Revision):
            raise TypeError("revision must be a Revision")
        if not isinstance(self.revised_input, AlignmentInput):
            raise TypeError("revised_input must be an AlignmentInput")


@dataclass(frozen=True)
class AlignmentReceipt:
    """A reserved prefix after FENCE; RESUME must revalidate the live reservation.

    reservation_id identifies a worker-owned hold, not a GPU address. A later
    revision or shutdown invalidates the hold even if this receipt was delivered.
    """

    request: AlignmentRequest
    reservation_id: str
    reusable_prefix_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.reservation_id, str) or not self.reservation_id:
            raise ValueError("reservation_id must be a nonempty string")
        count = self.reusable_prefix_length
        if type(count) is not int or not (
            self.request.revised_input.min_reuse_tokens
            <= count
            < len(self.request.revised_input.input_token_ids)
        ):
            raise ValueError("invalid reserved prefix length")


@dataclass(frozen=True)
class RevisionAlignment:
    """Handle whose timeout neither releases the reservation nor resumes work."""

    request: AlignmentRequest
    _future: Future[AlignmentReceipt] = field(repr=False, compare=False)

    def wait(self, timeout: float | None = 5.0) -> AlignmentReceipt:
        return self._future.result(timeout=timeout)


class AlignmentSafetyError(RuntimeError):
    """Native reservation cleanup has unknown completion; resource release must stop."""


class PrefixReservation(Protocol):
    """Native cache ownership manipulated exclusively by the engine thread."""

    @property
    def num_tokens(self) -> int: ...

    def close(self) -> None: ...


@dataclass
class _AlignmentRecord:
    request: AlignmentRequest
    reservation: PrefixReservation | None = None
    receipt: AlignmentReceipt | None = None
    error: Exception | None = None


class WorkerAlignmentState:
    """Arm on the mailbox thread; acquire and release KV only on the engine thread.

    The release boundary shares this lock with publication so a newly observed
    plan cannot fall between reservation processing and old-owner release.
    Native cache operations may wait for copies, but never reenter authority.
    """

    def __init__(
        self,
        reserve: Callable[[AlignmentInput], PrefixReservation],
        wake: Callable[[], None],
        max_reservations: int = 256,
    ) -> None:
        self._lock = RLock()
        self._reserve = reserve
        self._wake = wake
        self._max_reservations = max_reservations
        self._current: dict[tuple[str, str], ExecutionVersion] = {}
        self._records: dict[AlignmentRequest, _AlignmentRecord] = {}
        self._closed = False
        self._wake_pending = False
        self._unsafe_error: AlignmentSafetyError | None = None

    def observe(self, version: ExecutionVersion, request: AlignmentRequest | None) -> None:
        """Arm before old cancellation; retries never duplicate a reservation."""
        with self._lock:
            if self._closed:
                return
            key = (version.tenant_id, version.scope_id)
            previous = self._current.get(key)
            if previous is not None and version.epoch < previous.epoch:
                return
            self._current[key] = version
            if request is not None and request not in self._records:
                record = _AlignmentRecord(request)
                occupied = {
                    (r.revision.successor_version.tenant_id, r.revision.successor_version.scope_id)
                    for r, entry in self._records.items()
                    if entry.error is None
                }
                if key not in occupied and len(occupied) >= self._max_reservations:
                    record.error = RuntimeError("ALIGN reservation capacity exhausted")
                self._records[request] = record
            if self._records and not self._wake_pending:
                self._wake_pending = True
                self._wake()

    def _is_current(self, request: AlignmentRequest) -> bool:
        version = request.revision.successor_version
        return self._current.get((version.tenant_id, version.scope_id)) == version

    def process(self) -> None:
        """Engine safe point: reserve successors before dropping superseded holds."""
        with self._lock:
            self._wake_pending = False
            if self._unsafe_error is not None:
                raise self._unsafe_error
            if not self._closed:
                for request, record in self._records.items():
                    if not self._is_current(request) or record.receipt is not None or record.error:
                        continue
                    try:
                        reservation = self._reserve(request.revised_input)
                        record.reservation = reservation
                        count = reservation.num_tokens
                        if (
                            not request.revised_input.min_reuse_tokens
                            <= count
                            < len(request.revised_input.input_token_ids)
                        ):
                            raise RuntimeError("reserved prefix does not satisfy revised input")
                        record.receipt = AlignmentReceipt(request, uuid4().hex, count)
                    except AlignmentSafetyError as error:
                        record.error = error
                        self._unsafe_error = error
                        raise
                    except Exception as error:
                        # Report a request-scoped acquisition error. Retain any
                        # partial ownership until the engine can close it below.
                        record.error = error
            for request, record in list(self._records.items()):
                if self._closed or not self._is_current(request) or record.error is not None:
                    if record.reservation is not None:
                        record.reservation.close()
                        record.reservation = None
                    if self._closed or not self._is_current(request):
                        del self._records[request]

    @contextmanager
    def releasing_owner(self) -> Iterator[None]:
        """Capture any armed prefix before ordinary cleanup drops the old KV hold."""
        with self._lock:
            self.process()
            yield

    def poll(self, request: AlignmentRequest) -> AlignmentReceipt | None:
        """Read status without touching native KV or CUDA on the mailbox thread."""
        with self._lock:
            if self._closed:
                raise RuntimeError("ALIGN is closed")
            if not self._is_current(request):
                raise RuntimeError("alignment was superseded")
            record = self._records.get(request)
            if record is None:
                return None
            if record.error is not None:
                raise RuntimeError(f"ALIGN failed: {record.error}") from record.error
            return record.receipt

    @contextmanager
    def resetting_cache(self) -> Iterator[None]:
        """Serialize a tree reset with arming, and reject loss of a promised prefix."""
        with self._lock:
            if any(
                (record.receipt is None and record.error is None)
                or (record.reservation is not None and record.reservation.num_tokens > 0)
                for record in self._records.values()
            ):
                raise RuntimeError("cannot reset prefix cache while ALIGN owns or prepares KV")
            yield

    def close(self) -> None:
        """Invalidate immediately; the engine thread performs native cleanup."""
        with self._lock:
            self._closed = True
            if self._records and not self._wake_pending:
                self._wake_pending = True
                self._wake()
