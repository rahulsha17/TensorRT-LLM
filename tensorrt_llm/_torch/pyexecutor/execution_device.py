# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device authority and asynchronous token-commit guards for RETIRE execution."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

import torch
import triton
import triton.language as tl

from ..._utils import prefer_pinned
from ...execution_version import ExecutionVersion

__all__ = ["DeviceExecutionAuthority", "DeviceTokenCommit", "DeviceTokenCommitTicket"]


@triton.jit
def _publish_epoch(table, epoch, slot):
    # Publication and readers use GPU-scope atomics on the same 64-bit word.
    tl.atomic_xchg(table + slot, tl.load(epoch), sem="release", scope="gpu")


@triton.jit
def _check_versions(table, slots, expected, allowed, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row, row < COUNT, other=-1)
    epoch = tl.load(expected + row, row < COUNT, other=0)
    current = tl.atomic_add(
        table + tl.maximum(slot, 0), 0, mask=(row < COUNT) & (slot >= 0), sem="acquire", scope="gpu"
    )
    tl.store(allowed + row, (slot == -1) | (current == epoch), row < COUNT)


@triton.jit
def _commit_tokens(table, slots, expected, sequence_slots, candidates, committed, allowed):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    epoch = tl.load(expected + row)
    authorized = slot == -1
    if slot >= 0:
        current = tl.atomic_add(table + slot, 0, sem="acquire", scope="gpu")
        authorized = current == epoch
    tl.store(allowed + row, authorized)
    if authorized:
        # Single-step, single-beam tensors are contiguous and indexed by slot.
        sequence_slot = tl.load(sequence_slots + row)
        token = tl.load(candidates + sequence_slot)
        tl.store(committed + sequence_slot, token)


def _epoch_word(epoch: int) -> int:
    """Represent all uint64 epoch bits in torch.int64 without truncation."""
    return epoch if epoch < (1 << 63) else epoch - (1 << 64)


@dataclass
class _PendingUse:
    tensors: list[torch.Tensor]
    event: torch.cuda.Event | None = None


@dataclass(frozen=True)
class DeviceTokenCommit:
    tokens_device: torch.Tensor
    tokens_host: torch.Tensor
    allowed: list[bool]


@dataclass(frozen=True)
class DeviceTokenCommitTicket:
    """Own one batch's immutable metadata and asynchronous token-copy results.

    The host tensors may only be read after wait(). Device tokens are ready in
    the submitting stream; a consumer on another stream must wait for event.
    """

    authority: "DeviceExecutionAuthority"
    versions: tuple[ExecutionVersion | None, ...]
    sequence_slots: tuple[int, ...]
    tokens_device: torch.Tensor
    tokens_host: torch.Tensor
    allowed_host: torch.Tensor
    _use: _PendingUse

    @property
    def event(self) -> torch.cuda.Event:
        assert self._use.event is not None
        return self._use.event

    def wait(self) -> DeviceTokenCommit:
        self.authority._wait(self._use)
        return DeviceTokenCommit(self.tokens_device, self.tokens_host, self.allowed_host.tolist())


class DeviceExecutionAuthority:
    """Own stable scope slots and publish on a dedicated high-priority stream.

    Slots belong to (tenant, scope), never to reusable request/sequence slots.
    They are not recycled during this executor lifetime. Publication completion
    proves a table write only; it is not FENCE for kernels or KV transfers.

    Args:
        device: CUDA device index of the engine.
        capacity: Maximum number of distinct scopes in this executor lifetime.
    """

    _MAX_PENDING_USES = 256

    def __init__(self, device: int, capacity: int = 8192) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("device authority capacity must be positive")
        self.device = torch.device("cuda", device)
        self.capacity = capacity
        self._lock = Lock()
        self._slots: dict[tuple[str, str], int] = {}
        self._published: dict[tuple[str, str], tuple[int, torch.cuda.Event]] = {}
        self._pending: list[_PendingUse] = []
        self._failure: BaseException | None = None
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream(device=self.device, priority=-1)
            with torch.cuda.stream(self._stream):
                self._table = torch.zeros(capacity, dtype=torch.int64, device=self.device)
                self._ready = torch.cuda.Event()
                self._ready.record(self._stream)

    def _healthy(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "device authority is unavailable; restart required"
            ) from self._failure

    @contextmanager
    def submission(self) -> Iterator[None]:
        """Exclude publisher CUDA allocations during model capture/compilation.

        This protects CPU submission only, never GPU completion. Once the
        forward is enqueued, the publisher can run while its kernels execute.
        No worker-table lock may be acquired from inside this context.
        """
        with self._lock:
            self._healthy()
            yield

    def _collect_completed(self) -> None:
        self._pending[:] = [
            use for use in self._pending if use.event is None or not use.event.query()
        ]

    def _start_use(self, tensors: list[torch.Tensor]) -> _PendingUse:
        try:
            self._collect_completed()
        except BaseException as error:
            self._failure = error
            raise
        if len(self._pending) >= self._MAX_PENDING_USES:
            error = RuntimeError(
                "too many outstanding device authority operations; restart required"
            )
            self._failure = error
            raise error
        use = _PendingUse(tensors)
        self._pending.append(use)
        return use

    def _upload(self, values: list[int], dtype: torch.dtype, use: _PendingUse) -> torch.Tensor:
        host = torch.tensor(values, dtype=dtype, pin_memory=prefer_pinned())
        use.tensors.append(host)
        device = host.to(self.device, non_blocking=True)
        use.tensors.append(device)
        return device

    def _record(self, use: _PendingUse, stream: torch.cuda.Stream) -> None:
        event = torch.cuda.Event()
        event.record(stream)
        use.event = event
        self._table.record_stream(stream)
        self._collect_completed()

    def _wait(self, use: _PendingUse, timeout: float = 5.0) -> None:
        # No authority lock is held while waiting: the publisher must be able
        # to revoke a version even while its model stream is occupied.
        deadline = time.monotonic() + timeout
        try:
            assert use.event is not None
            while not use.event.query():
                self._healthy()
                if time.monotonic() >= deadline:
                    raise TimeoutError("device authority guard did not complete")
                time.sleep(0.001)
            self._healthy()
        except BaseException as error:
            with self._lock:
                self._failure = error
            raise

    def wait_event(self, event: torch.cuda.Event) -> None:
        """Wait for a request's submitted readers without holding authority locks."""
        self._wait(_PendingUse([], event))

    def publish(self, version: ExecutionVersion) -> None:
        """Queue one atomic epoch write without joining the execution stream."""
        key = (version.tenant_id, version.scope_id)
        with self._lock, torch.cuda.device(self.device):
            self._healthy()
            previous = self._published.get(key)
            if previous is not None:
                if version.epoch < previous[0]:
                    raise ValueError("device authority cannot move backwards")
                if version.epoch == previous[0]:
                    return
            if key not in self._slots:
                if len(self._slots) == self.capacity:
                    raise RuntimeError("device authority scope capacity exhausted")
                self._slots[key] = len(self._slots)
            use = self._start_use([])
            try:
                with torch.cuda.stream(self._stream):
                    epoch = self._upload([_epoch_word(version.epoch)], torch.int64, use)
                    _publish_epoch[(1,)](self._table, epoch, self._slots[key], num_warps=1)
                    self._record(use, self._stream)
                assert use.event is not None
                self._published[key] = (version.epoch, use.event)
            except BaseException as error:
                self._failure = error
                raise

    def _inputs(
        self, versions: list[ExecutionVersion | None], use: _PendingUse
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Called with _lock held, on the caller's CUDA stream.
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self._ready)
        slots, epochs = [], []
        for version in versions:
            if version is None:
                slots.append(-1)
                epochs.append(0)
                continue
            key = (version.tenant_id, version.scope_id)
            if key not in self._published:
                raise RuntimeError(f"scope {key!r} has no device publication")
            # Initial publication is visible before a request can pass a guard.
            # Later updates can still arrive concurrently and are read atomically.
            stream.wait_event(self._published[key][1])
            slots.append(self._slots[key])
            epochs.append(_epoch_word(version.epoch))
        return self._upload(slots, torch.int32, use), self._upload(epochs, torch.int64, use)

    def check(self, versions: list[ExecutionVersion | None]) -> list[bool]:
        """Observe device authority at a batch-entry boundary."""
        if not versions or all(version is None for version in versions):
            return [True] * len(versions)
        with self._lock, torch.cuda.device(self.device):
            self._healthy()
            use = self._start_use([])
            try:
                slots, expected = self._inputs(versions, use)
                allowed = torch.empty(len(versions), dtype=torch.bool, device=self.device)
                use.tensors.append(allowed)
                _check_versions[(triton.cdiv(len(versions), 128),)](
                    self._table, slots, expected, allowed, len(versions), 128
                )
                host = torch.empty(len(versions), dtype=torch.bool, pin_memory=prefer_pinned())
                use.tensors.append(host)
                host.copy_(allowed, non_blocking=True)
                self._record(use, torch.cuda.current_stream(self.device))
            except BaseException as error:
                self._failure = error
                raise
        self._wait(use)
        return host.tolist()

    def commit_tokens(
        self,
        versions: list[ExecutionVersion | None],
        sequence_slots: list[int],
        candidates: torch.Tensor,
        compact: bool,
    ) -> DeviceTokenCommit:
        """Synchronous convenience API; the executor uses enqueue_token_commit."""
        return self.enqueue_token_commit(versions, sequence_slots, candidates, compact).wait()

    def enqueue_token_commit(
        self,
        versions: list[ExecutionVersion | None],
        sequence_slots: list[int],
        candidates: torch.Tensor,
        compact: bool,
    ) -> DeviceTokenCommitTicket:
        """Copy only authorized candidate tokens to a distinct committed buffer.

        The atomic authority read is the device commit admission point. A commit
        admitted before publication may finish afterward; FENCE will account for
        in-flight effects. Candidate sampler buffers are private scratch, not KV
        or successor state, and may contain values for revoked requests.

        ``candidates`` must be a contiguous integer CUDA tensor of shape
        ``[1, num_sequence_slots, 1]``. The returned host tokens have shape
        ``[num_requests]`` for compact sampling or the candidate shape otherwise.
        """
        if len(versions) != len(sequence_slots) or not versions:
            raise ValueError("token commit requires one version and sequence slot per request")
        if (
            candidates.ndim != 3
            or candidates.shape[0] != 1
            or candidates.shape[2] != 1
            or not candidates.is_contiguous()
        ):
            raise ValueError(
                "device token guard requires contiguous single-step, single-beam tokens"
            )
        if candidates.device != self.device:
            raise ValueError("candidate tokens are on a different device")
        if candidates.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate tokens must be int32 or int64")
        if any(slot < 0 or slot >= candidates.shape[1] for slot in sequence_slots):
            raise ValueError("candidate sequence slot is out of bounds")
        if len(set(sequence_slots)) != len(sequence_slots):
            raise ValueError("candidate sequence slots must be unique")
        with self._lock, torch.cuda.device(self.device):
            self._healthy()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "RETIRE token commit must run after graph replay, outside capture"
                )
            use = self._start_use([candidates])
            try:
                slots, expected = self._inputs(versions, use)
                sequence_slots_device = self._upload(sequence_slots, torch.int32, use)
                committed = torch.zeros_like(candidates)
                allowed = torch.empty(len(versions), dtype=torch.bool, device=self.device)
                use.tensors.extend([committed, allowed])
                _commit_tokens[(len(versions),)](
                    self._table,
                    slots,
                    expected,
                    sequence_slots_device,
                    candidates,
                    committed,
                    allowed,
                    num_warps=1,
                )
                tokens = committed[0, sequence_slots_device, 0] if compact else committed
                host = torch.empty(tokens.shape, dtype=tokens.dtype, pin_memory=prefer_pinned())
                accepted = torch.empty(len(versions), dtype=torch.bool, pin_memory=prefer_pinned())
                use.tensors.extend([tokens, host, accepted])
                host.copy_(tokens, non_blocking=True)
                accepted.copy_(allowed, non_blocking=True)
                self._record(use, torch.cuda.current_stream(self.device))
            except BaseException as error:
                self._failure = error
                raise
        return DeviceTokenCommitTicket(
            self, tuple(versions), tuple(sequence_slots), committed, host, accepted, use
        )
