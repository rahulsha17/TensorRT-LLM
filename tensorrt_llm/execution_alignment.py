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
    "ExecutionPlacement",
    "PrefixTransferPlan",
    "AlignmentInput",
    "AlignmentRequest",
    "AlignmentReceipt",
    "RevisionAlignment",
    "AlignmentSafetyError",
    "PrefixReservation",
    "WorkerAlignmentState",
]


@dataclass(frozen=True)
class ExecutionPlacement:
    """One frontend lifetime within a runtime authority session, including its ranks."""

    session_id: str
    placement_id: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v for v in (self.session_id, self.placement_id)):
            raise ValueError("execution placement requires a session and frontend lifetime")


@dataclass(frozen=True)
class PrefixTransferPlan:
    """Runtime-selected source and complete prefix length for one destination.

    This identifies required work, not memory addresses or completion. Every
    source and destination rank must retain its own physical copy obligations.
    """

    source: ExecutionPlacement
    num_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, ExecutionPlacement):
            raise TypeError("prefix source must be an ExecutionPlacement")
        if type(self.num_tokens) is not int or not 0 < self.num_tokens < 2**31:
            raise ValueError("transfer prefix length must be a positive int32 integer")


@dataclass(frozen=True)
class AlignmentInput:
    """The runtime's complete revised text input and cache namespace.

    Tokens must be immutable. The final input token is always recomputed to
    obtain logits. min_reuse_tokens can require a prefix instead of allowing
    recomputation on a cache miss. Salt and LoRA identity use ordinary TRT-LLM
    reuse semantics; neither is derived from the execution epoch.

    destination selects one prepared frontend lifetime, including all its ranks.
    None retains common-prefix alignment across the complete cohort. Selection
    does not move KV. transfer additionally retains a source hold and requires
    both endpoints' copy acceptance. The engine must still allocate, copy and
    attach the actual destination prefix before ALIGN can complete.

    max_reuse_tokens bounds both reservation and successor cache lookup. A
    runtime enforcing a numerical contract supplies its validated boundary,
    or zero when it has no reuse evidence. None preserves ordinary cache reuse;
    token and namespace matches alone do not certify cold-output equivalence.
    """

    input_token_ids: tuple[int, ...]
    cache_salt: str | None = None
    lora_task_id: int | None = None
    min_reuse_tokens: int = 0
    destination: ExecutionPlacement | None = None
    transfer: PrefixTransferPlan | None = None
    max_reuse_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.destination is not None and not isinstance(self.destination, ExecutionPlacement):
            raise TypeError("destination must be an ExecutionPlacement or None")
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
        if self.max_reuse_tokens is not None and (
            type(self.max_reuse_tokens) is not int
            or not self.min_reuse_tokens <= self.max_reuse_tokens < len(self.input_token_ids)
        ):
            raise ValueError("max_reuse_tokens must cover required reuse and leave the final token")
        if self.transfer is not None:
            if not isinstance(self.transfer, PrefixTransferPlan):
                raise TypeError("transfer must be a PrefixTransferPlan or None")
            if self.destination is None or self.transfer.source == self.destination:
                raise ValueError("prefix transfer requires distinct source and destination")
            if self.transfer.source.session_id != self.destination.session_id:
                raise ValueError("prefix endpoints must share one authority session")
            if not self.min_reuse_tokens <= self.transfer.num_tokens < len(self.input_token_ids):
                raise ValueError("transfer must cover required reuse and leave the final token")
            if self.transfer.num_tokens > self.reuse_limit:
                raise ValueError("transfer exceeds max_reuse_tokens")

    @property
    def reuse_limit(self) -> int:
        """Maximum permitted cache lookup depth, excluding the final input token."""
        return (
            len(self.input_token_ids) - 1
            if self.max_reuse_tokens is None
            else self.max_reuse_tokens
        )

    @property
    def required_prefix_length(self) -> int:
        """The promised prefix includes any runtime-selected transfer."""
        return self.min_reuse_tokens if self.transfer is None else self.transfer.num_tokens


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
        destination = self.revised_input.destination
        if destination is not None and destination.session_id != self.session_id:
            raise ValueError("ALIGN destination belongs to another authority session")


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
            self.request.revised_input.required_prefix_length
            <= count
            <= self.request.revised_input.reuse_limit
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

    def settle(self) -> None: ...

    def close(self) -> None: ...


@dataclass
class _AlignmentRecord:
    request: AlignmentRequest
    reservation: PrefixReservation | None = None
    receipt: AlignmentReceipt | None = None
    error: Exception | None = None
    resumed: bool = False
    resume_prefix_length: int | None = None
    request_id: int | None = None
    attached: bool = False
    transfer_drains: list[Callable[[], bool]] = field(default_factory=list)
    source_only: bool = False
    source_released: bool = False


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
        *,
        prepare: Callable[[AlignmentRequest], None] | None = None,
    ) -> None:
        self._lock = RLock()
        self._reserve = reserve
        self._prepare = prepare
        self._wake = wake
        self._max_reservations = max_reservations
        self._current: dict[tuple[str, str], ExecutionVersion] = {}
        self._records: dict[AlignmentRequest, _AlignmentRecord] = {}
        self._closed = False
        self._wake_pending = False
        self._unsafe_error: AlignmentSafetyError | None = None

    def observe(
        self,
        version: ExecutionVersion,
        request: AlignmentRequest | None,
        *,
        source_only: bool = False,
    ) -> None:
        """Arm before old cancellation; retries never duplicate a reservation."""
        with self._lock:
            if self._closed:
                return
            key = (version.tenant_id, version.scope_id)
            previous = self._current.get(key)
            if previous is not None and version.epoch < previous.epoch:
                return
            self._current[key] = version
            if request in self._records and self._records[request].source_only != source_only:
                raise RuntimeError("ALIGN cannot change its source role")
            if request is not None and request not in self._records:
                record = _AlignmentRecord(request, source_only=source_only)
                occupied = {
                    (r.revision.successor_version.tenant_id, r.revision.successor_version.scope_id)
                    for r, entry in self._records.items()
                    if entry.error is None
                    and (entry.receipt is None or entry.reservation is not None)
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

    def retain_transfer(self, request: AlignmentRequest, drained: Callable[[], bool]) -> None:
        """Keep this reservation until a copy's physical owner has finished.

        The callback must only inspect host lifetime state and never acquire
        execution authority. Native cache cleanup remains on the engine thread.
        """
        with self._lock:
            record = self._records.get(request)
            if self._closed or not self._is_current(request) or record is None:
                raise RuntimeError("ALIGN copy has no current reservation plan")
            if record.error is not None or record.resumed or record.source_released:
                raise RuntimeError("ALIGN copy cannot extend a failed or resumed plan")
            record.transfer_drains.append(drained)

    def wake_reclaim(self) -> None:
        """Notify the engine after transfer cleanup, including during shutdown."""
        with self._lock:
            if self._records and not self._wake_pending:
                self._wake_pending = True
                self._wake()

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
                        if self._prepare is not None:
                            self._prepare(request)
                        reservation = self._reserve(request.revised_input)
                        record.reservation = reservation
                        count = reservation.num_tokens
                        if (
                            not request.revised_input.required_prefix_length
                            <= count
                            <= request.revised_input.reuse_limit
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
                if (
                    self._closed
                    or not self._is_current(request)
                    or record.error is not None
                    or record.source_released
                ):
                    reclaimed = self._reclaim(record)
                    if reclaimed and (self._closed or not self._is_current(request)):
                        del self._records[request]
            if self._closed and self._records:
                raise RuntimeError("ALIGN transfers still retain KV; teardown must wait")

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

    def snapshot(self) -> dict:
        """Copy logical reservation ownership without querying native KV or CUDA.

        Promised tokens count per reservation, not distinct physical blocks or
        HBM residency. A retained reservation with no receipt has unknown size.
        Only successful native cleanup clears its ownership flag.
        """
        with self._lock:
            records = []
            for request, record in self._records.items():
                version = request.revision.successor_version
                records.append(
                    {
                        "tenant_id": version.tenant_id,
                        "scope_id": version.scope_id,
                        "epoch": version.epoch,
                        "current": self._is_current(request),
                        "has_reservation": record.reservation is not None,
                        "promised_tokens": (
                            record.receipt.reusable_prefix_length if record.receipt else None
                        ),
                        "resumed": record.resumed,
                        "resume_prefix_length": record.resume_prefix_length,
                        "attached": record.attached,
                        "request_id": record.request_id,
                        "failed": record.error is not None,
                        "source_only": record.source_only,
                        "source_released": record.source_released,
                    }
                )
            held = [record for record in records if record["has_reservation"]]
            return {
                "closed": self._closed,
                "unsafe": self._unsafe_error is not None,
                "held_count": len(held),
                "promised_tokens_held": sum(record["promised_tokens"] or 0 for record in held),
                "unknown_prefix_count": sum(record["promised_tokens"] is None for record in held),
                "records": records,
            }

    def authorize_resume(self, receipt: AlignmentReceipt, prefix_length: int | None = None) -> None:
        """Validate a live reservation before opening worker admission; no CUDA calls."""
        with self._lock:
            if self.poll(receipt.request) != receipt:
                raise RuntimeError("RESUME does not match the live ALIGN reservation")
            if prefix_length is not None and (
                type(prefix_length) is not int
                or not receipt.request.revised_input.required_prefix_length
                <= prefix_length
                <= receipt.reusable_prefix_length
            ):
                raise ValueError("RESUME prefix is not covered by this reservation")
            record = self._records[receipt.request]
            if record.source_only:
                raise RuntimeError("a prefix source cannot RESUME successor execution")
            if record.resumed and record.resume_prefix_length != prefix_length:
                raise RuntimeError("RESUME cannot change the agreed prefix")
            record.resume_prefix_length = prefix_length
            record.resumed = True

    def release_source(self, request: AlignmentRequest) -> None:
        """Retire a source hold after global copy acceptance; cleanup stays on the engine.

        Keep its receipt for idempotent publication retries. It describes an
        already accepted copy after release, never permission to start another.
        """
        with self._lock:
            receipt = self.poll(request)
            record = self._records.get(request)
            if receipt is None or record is None or not record.source_only:
                raise RuntimeError("source release requires its prepared ALIGN hold")
            record.source_released = True
            self.wake_reclaim()

    def _record_for_version(self, version: ExecutionVersion) -> _AlignmentRecord | None:
        # Callers hold the state lock. Initial versions have no alignment record.
        if self._closed:
            raise RuntimeError("ALIGN is closed")
        if self._current.get((version.tenant_id, version.scope_id)) != version:
            raise RuntimeError("successor authority is no longer current")
        return next(
            (
                record
                for request, record in self._records.items()
                if request.revision.successor_version == version
            ),
            None,
        )

    def claim(
        self,
        version: ExecutionVersion,
        request_id: int,
        input_token_ids: tuple[int, ...],
        cache_salt: str | None,
        lora_task_id: int | None,
    ) -> int | None:
        """Bind one physical request and return its permitted cache-lookup limit."""
        with self._lock:
            record = self._record_for_version(version)
            if record is None:
                return
            if not record.resumed or record.error is not None:
                raise RuntimeError("successor is pending RESUME")
            revised = record.request.revised_input
            if (input_token_ids, cache_salt, lora_task_id) != (
                revised.input_token_ids,
                revised.cache_salt,
                revised.lora_task_id,
            ):
                raise ValueError("successor input does not match ALIGN")
            if record.request_id is not None and record.request_id != request_id:
                raise RuntimeError("RESUME already belongs to another request")
            record.request_id = request_id
            # Collective RESUME fixes an exact common prefix. Otherwise the
            # reservation guarantees a floor, while the runtime's bound also
            # permits compatible cache entries acquired after ALIGN.
            return (
                record.resume_prefix_length
                if record.resume_prefix_length is not None
                else revised.max_reuse_tokens
            )

    def pending_prefix_length(self, version: ExecutionVersion, request_id: int) -> int:
        """Return this request's unconsumed prefix promise, or zero when absent."""
        return self.pending_recovery_prefix(version, request_id) or 0

    def pending_recovery_prefix(self, version: ExecutionVersion, request_id: int) -> int | None:
        """Return a claimed successor's prefix, including zero; None means no pending handoff."""
        with self._lock:
            if self._closed or self._current.get((version.tenant_id, version.scope_id)) != version:
                return None
            record = self._record_for_version(version)
            if (
                record is None
                or not record.resumed
                or record.error is not None
                or record.request_id != request_id
                or record.attached
                or record.reservation is None
            ):
                return None
            assert record.receipt is not None
            return (
                record.resume_prefix_length
                if record.resume_prefix_length is not None
                else record.receipt.reusable_prefix_length
            )

    def has_resident_recovery(self, version: ExecutionVersion, request_id: int) -> bool:
        """Return an engine-thread scheduling hint, never new reuse authority.

        Only a current, claimed, unattached reservation can supply the hint.
        Backends without optional residency information keep ordinary ordering.
        """
        with self._lock:
            if self._closed or self._current.get((version.tenant_id, version.scope_id)) != version:
                return False
            record = self._record_for_version(version)
            if (
                record is None
                or not record.resumed
                or record.error is not None
                or record.request_id != request_id
                or record.attached
                or record.source_only
                or record.source_released
                or record.reservation is None
            ):
                return False
            return getattr(record.reservation, "has_resident_prefix", False) is True

    def confirm_attachment(
        self, version: ExecutionVersion, request_id: int, reused_tokens: int
    ) -> bool:
        """Validate ownership before forward; report the first successful attachment."""
        with self._lock:
            # A revision can race the scheduler. Stale candidates still pass
            # through the existing device/host fences; do not fail other rows.
            if self._current.get((version.tenant_id, version.scope_id)) != version:
                return False
            record = self._record_for_version(version)
            if record is None:
                return False
            if not record.resumed or record.request_id != request_id or record.error is not None:
                raise RuntimeError("request does not own this RESUME")
            newly_attached = not record.attached
            if newly_attached:
                promised = record.resume_prefix_length
                if promised is not None:
                    if reused_tokens != promised:
                        raise RuntimeError("successor did not acquire the agreed KV prefix")
                elif (
                    record.receipt is None or reused_tokens < record.receipt.reusable_prefix_length
                ):
                    raise RuntimeError("successor did not acquire the aligned KV prefix")
                elif reused_tokens > record.request.revised_input.reuse_limit:
                    raise RuntimeError("successor cache reuse exceeds the runtime bound")
                record.attached = True
            self._reclaim(record, settle=True)
            return newly_attached

    def _reclaim(self, record: _AlignmentRecord, *, settle: bool = True) -> bool:
        # Native calls belong to the engine thread. Preserve ownership and stop
        # further cleanup when a copy wait or release has unknown completion.
        if self._unsafe_error is not None:
            raise self._unsafe_error
        try:
            record.transfer_drains = [
                drained for drained in record.transfer_drains if not drained()
            ]
            if record.transfer_drains:
                return False
            if record.reservation is None:
                return True
            if settle:
                record.reservation.settle()
            record.reservation.close()
        except Exception as error:
            self._unsafe_error = AlignmentSafetyError(
                "RECLAIM completion is unknown; restart required"
            )
            record.error = self._unsafe_error
            raise self._unsafe_error from error
        record.reservation = None
        return True

    def terminal(self, version: ExecutionVersion, request_id: int) -> None:
        """Mark an abandoned claim; defer native release to an engine safe point."""
        with self._lock:
            for request, record in self._records.items():
                if request.revision.successor_version == version and (
                    record.request_id == request_id
                    or (record.resumed and record.request_id is None)
                ):
                    # Cancellation may arrive before activation binds the input.
                    record.request_id = request_id
                    record.error = RuntimeError("RESUME request is terminal")
                    if record.reservation is not None and not self._wake_pending:
                        self._wake_pending = True
                        self._wake()
                    break

    @contextmanager
    def resetting_cache(self) -> Iterator[None]:
        """Serialize a tree reset with arming, and reject loss of a promised prefix."""
        with self._lock:
            if any(
                (record.receipt is None and record.error is None)
                or (record.reservation is not None and record.reservation.num_tokens > 0)
                or bool(record.transfer_drains)
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
