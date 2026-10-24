# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention row upload reuse and fail-closed submission protocol tests."""

from contextlib import nullcontext
from dataclasses import replace
from threading import RLock
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import execution_device as device
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only


class Tensor:
    def __init__(self, values: list[int], dtype: str, inference: bool = False) -> None:
        self.values = list(values)
        self.dtype, self.device = dtype, "cuda:0"
        self.shape = (len(values),)
        self.inference = inference
        self.version = 0
        self.pointer = id(self)
        self.offset = 0
        self.step = 1
        self.copies = 0
        self.failure: BaseException | None = None

    @property
    def _version(self) -> int:
        if self.inference:
            raise RuntimeError("Inference tensors do not track version counter.")
        return self.version

    def numel(self) -> int:
        return len(self.values)

    def data_ptr(self) -> int:
        return self.pointer

    def storage_offset(self) -> int:
        return self.offset

    def stride(self) -> tuple[int, ...]:
        return (self.step,)

    def copy_(self, other: "Tensor", non_blocking: bool) -> "Tensor":
        assert non_blocking
        self.copies += 1
        if self.failure is not None:
            raise self.failure
        self.values[:] = other.values
        if not self.inference:
            self.version += 1
        return self

    def record_stream(self, stream: "Stream") -> None:
        assert stream.cuda_stream == 7


class Stream:
    def __init__(self) -> None:
        self.cuda_stream = 7
        self.waited: list[Event] = []

    def wait_event(self, event: "Event") -> None:
        self.waited.append(event)
        if event.wait_failure is not None:
            raise event.wait_failure


class Event:
    def __init__(self) -> None:
        self.complete = False
        self.queries = 0
        self.recorded = False
        self.failure: BaseException | None = None
        self.wait_failure: BaseException | None = None

    def query(self) -> bool:
        self.queries += 1
        if self.failure is not None:
            raise self.failure
        return self.complete

    def record(self, stream: Stream) -> None:
        assert stream.cuda_stream == 7
        self.recorded = True


class Protocol:
    def __init__(self) -> None:
        self.stream = Stream()
        self.capture = False
        self.inference = False
        self.host: list[Tensor] = []
        self.events: list[Event] = []
        self.record_failure: BaseException | None = None

    def tensor(self, values: list[int], dtype: str, pin_memory: bool) -> Tensor:
        result = Tensor(values, dtype, self.inference)
        self.host.append(result)
        return result

    def event(self) -> Event:
        result = Event()
        if self.record_failure is not None:
            failure = self.record_failure

            def fail_record(stream: Stream) -> None:
                raise failure

            result.record = fail_record
        self.events.append(result)
        return result

    def full(self, shape: tuple[int, ...], value: int, dtype: str, device: str) -> Tensor:
        return Tensor([value] * shape[0], dtype, self.inference)

    def zeros(self, size: int, dtype: str, device: str) -> Tensor:
        return Tensor([0] * size, dtype, self.inference)


@pytest.fixture
def protocol(monkeypatch: pytest.MonkeyPatch) -> Protocol:
    value = Protocol()
    cuda = SimpleNamespace(
        device=lambda _: nullcontext(),
        current_stream=lambda _: value.stream,
        is_current_stream_capturing=lambda: value.capture,
        Event=value.event,
    )
    monkeypatch.setattr(
        device,
        "torch",
        SimpleNamespace(
            cuda=cuda,
            tensor=value.tensor,
            full=value.full,
            zeros=value.zeros,
            int32="int32",
            int64="int64",
            is_inference=lambda tensor: tensor.inference,
        ),
    )
    monkeypatch.setattr(device, "prefer_pinned", lambda: False)
    return value


@pytest.fixture
def authority(protocol: Protocol) -> device.DeviceExecutionAuthority:
    result = object.__new__(device.DeviceExecutionAuthority)
    result.device = "cuda:0"
    result._lock = RLock()
    result._pending = []
    result._failure = None
    result._ready = Event()
    result._table = Tensor([0] * 8, "int64")
    result._slots = {("tenant", "one"): 0, ("tenant", "two"): 1}
    result._published = {key: (3, Event()) for key in result._slots}
    return result


@pytest.mark.parametrize("inference", [False, True])
def test_attention_rows_two_uploads_then_zero_with_fresh_use(
    protocol: Protocol, authority: device.DeviceExecutionAuthority, inference: bool
) -> None:
    protocol.inference = inference
    rows = authority.create_attention_rows(3)
    version = ExecutionVersion("tenant", "one", 3)
    authority.prepare_attention_rows(rows, [version])
    assert len(protocol.host) == 2
    assert rows.slots.values == [0, -1, -1] and rows.expected.values == [3, 0, 0]
    first = authority._pending[-1]
    assert len(first.tensors) == 5
    rows.counters.values[:] = [2, 3, 4]
    waits = len(protocol.stream.waited)
    authority.prepare_attention_rows(rows, [version])
    assert len(protocol.host) == 2, "identical layout uploaded again"
    assert rows.slots.copies == rows.expected.copies == 1
    assert len(protocol.stream.waited) == waits + 2
    assert first.event.queries > 0
    assert authority._pending[-1] is not first
    assert authority._pending[-1].event.recorded
    assert authority._pending[-1].tensors == [rows.slots, rows.expected, rows.counters]
    assert rows.counters.values == [2, 3, 4]


def test_attention_rows_same_expected_waits_new_publication(
    protocol: Protocol, authority: device.DeviceExecutionAuthority
) -> None:
    rows = authority.create_attention_rows(2)
    version = ExecutionVersion("tenant", "one", (1 << 63) - 1)
    authority.prepare_attention_rows(rows, [version])
    newest = Event()
    authority._published[("tenant", "one")] = (1 << 63, newest)
    authority.prepare_attention_rows(rows, [version])
    assert len(protocol.host) == 2 and protocol.stream.waited[-1] is newest
    authority.prepare_attention_rows(rows, [replace(version, epoch=1 << 63)])
    assert len(protocol.host) == 4 and rows.expected.values == [-(1 << 63), 0]


def test_attention_rows_padding_order_scope_and_independent_storage(
    protocol: Protocol, authority: device.DeviceExecutionAuthority
) -> None:
    rows, other = authority.create_attention_rows(3), authority.create_attention_rows(3)
    first, second = ExecutionVersion("tenant", "one", 3), ExecutionVersion("tenant", "two", 3)
    for versions in ([first, second], [second, first], [first], [second], []):
        before = len(protocol.host)
        authority.prepare_attention_rows(rows, versions)
        assert len(protocol.host) == before + 2
    authority.prepare_attention_rows(rows, [None])
    assert len(protocol.host) == 10 and rows.slots.values == [-1] * 3
    authority.prepare_attention_rows(other, [None])
    assert len(protocol.host) == 12 and other._layout is not rows._layout


@pytest.mark.parametrize("change", ["replacement", "pointer", "offset", "stride", "version"])
def test_attention_rows_storage_changes_force_upload(
    protocol: Protocol, authority: device.DeviceExecutionAuthority, change: str
) -> None:
    rows = authority.create_attention_rows(2)
    authority.prepare_attention_rows(rows, [None])
    if change == "replacement":
        rows.expected = Tensor([91, 91], "int64")
    else:
        field = {"pointer": "pointer", "offset": "offset", "stride": "step", "version": "version"}[
            change
        ]
        setattr(rows.expected, field, getattr(rows.expected, field) + 1)
        rows.expected.values[:] = [91, 91]
    authority.prepare_attention_rows(rows, [None])
    assert len(protocol.host) == 4 and rows.expected.values == [0, 0]


@pytest.mark.parametrize("failure", ["copy", "record", "query", "publication", "pending"])
def test_attention_rows_failure_invalidates_and_poisons(
    protocol: Protocol, authority: device.DeviceExecutionAuthority, failure: str
) -> None:
    rows = authority.create_attention_rows(2)
    first = ExecutionVersion("tenant", "one", 3)
    authority.prepare_attention_rows(rows, [first])
    error = RuntimeError("injected failure")
    versions = [first]
    if failure == "copy":
        rows.expected.failure = error
        versions = []
    elif failure == "record":
        protocol.record_failure = error
    elif failure == "query":
        authority._pending[0].event.failure = error
    elif failure == "publication":
        authority._published[("tenant", "one")][1].wait_failure = error
    else:
        authority._MAX_PENDING_USES = len(authority._pending)
    before = len(protocol.host)
    with pytest.raises(RuntimeError) as raised:
        authority.prepare_attention_rows(rows, versions)
    assert rows._layout is None and authority._failure is raised.value
    assert len(protocol.host) == before + (2 if failure == "copy" else 0)
    with pytest.raises(RuntimeError, match="restart required"):
        authority.prepare_attention_rows(rows, [])
    assert authority._pending


def test_attention_rows_unpublished_scope_poison_even_after_hit(
    protocol: Protocol, authority: device.DeviceExecutionAuthority
) -> None:
    rows = authority.create_attention_rows(2)
    authority.prepare_attention_rows(rows, [])
    authority.prepare_attention_rows(rows, [None])
    with pytest.raises(RuntimeError, match="no device publication"):
        authority.prepare_attention_rows(rows, [ExecutionVersion("tenant", "missing", 3)])
    assert rows._layout is None and len(protocol.host) == 2


@pytest.mark.parametrize("invalid", ["owner", "capacity", "stream", "capture", "health"])
def test_attention_rows_hits_keep_validation(
    protocol: Protocol, authority: device.DeviceExecutionAuthority, invalid: str
) -> None:
    rows = authority.create_attention_rows(2)
    authority.prepare_attention_rows(rows, [])
    versions = []
    if invalid == "owner":
        rows.authority = object()
    elif invalid == "capacity":
        versions = [None] * 3
    elif invalid == "stream":
        protocol.stream.cuda_stream += 1
    elif invalid == "capture":
        protocol.capture = True
    else:
        authority._failure = RuntimeError("earlier failure")
    events = len(protocol.events)
    with pytest.raises((ValueError, RuntimeError)):
        authority.prepare_attention_rows(rows, versions)
    assert len(protocol.host) == 2 and len(protocol.events) == events
