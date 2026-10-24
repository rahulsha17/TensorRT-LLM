# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA attention row reuse, fresh authority and storage ownership regressions."""

import copy
import gc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor import execution_device as device
from tensorrt_llm._torch.pyexecutor.execution_attention import ExecutionAttentionBinding
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@contextmanager
def uploads(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[int, ...]]]:
    """Count the host row tensors only while preparing attention inputs."""
    calls: list[tuple[int, ...]] = []
    original = torch.tensor

    def tensor(values: list[int], *, dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
        calls.append(tuple(values))
        return original(values, dtype=dtype, pin_memory=pin_memory)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", tensor)
        yield calls


def allowed(
    authority: device.DeviceExecutionAuthority, rows: device.DeviceAttentionRows
) -> list[bool]:
    output = torch.empty(rows.slots.numel(), dtype=torch.bool, device=authority.device)
    device._check_versions[(1,)](
        authority._table, rows.slots, rows.expected, output, output.numel(), 128
    )
    return output.cpu().tolist()


@pytest.mark.parametrize("inference", [False, True])
def test_attention_row_reuse_observes_live_publication(
    monkeypatch: pytest.MonkeyPatch, inference: bool
) -> None:
    with torch.inference_mode(inference):
        authority = device.DeviceExecutionAuthority(torch.cuda.current_device())
        old = ExecutionVersion("tenant", "one", (1 << 63) - 1)
        neighbor = replace(old, scope_id="two", epoch=4)
        authority.publish(old)
        authority.publish(neighbor)
        rows = authority.create_attention_rows(4)
        assert torch.is_inference(rows.slots) is inference
        with uploads(monkeypatch) as calls:
            authority.prepare_attention_rows(rows, [old, neighbor])
            assert len(calls) == 2
            authority.prepare_attention_rows(rows, [old, neighbor])
            assert len(calls) == 2
        assert allowed(authority, rows) == [True] * 4
        current = replace(old, epoch=1 << 63)
        authority.publish(current)
        with uploads(monkeypatch) as calls:
            authority.prepare_attention_rows(rows, [old, neighbor])
            assert calls == []
        assert allowed(authority, rows) == [False, True, True, True]
        with uploads(monkeypatch) as calls:
            authority.prepare_attention_rows(rows, [current, neighbor])
            assert len(calls) == 2
        assert allowed(authority, rows) == [True] * 4
        assert rows.expected.cpu().tolist() == [-(1 << 63), 4, 0, 0]
        torch.cuda.synchronize()


def test_attention_row_reuse_padding_order_scope_and_independent_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = device.DeviceExecutionAuthority(torch.cuda.current_device())
    first = ExecutionVersion("tenant", "one", 3)
    second = replace(first, scope_id="two")
    authority.publish(first)
    authority.publish(second)
    rows, other = authority.create_attention_rows(3), authority.create_attention_rows(3)
    rows.counters.fill_(7)
    pointers = (rows.slots.data_ptr(), rows.expected.data_ptr())
    for versions in ([first, second], [second, first], [first], [second], []):
        with uploads(monkeypatch) as calls:
            authority.prepare_attention_rows(rows, versions)
            assert len(calls) == 2
            authority.prepare_attention_rows(rows, versions)
            assert len(calls) == 2
        expected = [authority._slots[(v.tenant_id, v.scope_id)] for v in versions]
        assert rows.slots.cpu().tolist() == expected + [-1] * (3 - len(versions))
        assert allowed(authority, rows) == [True] * 3
    with uploads(monkeypatch) as calls:
        authority.prepare_attention_rows(rows, [None])
        assert calls == []
        authority.prepare_attention_rows(other, [None])
        assert len(calls) == 2
    assert rows._layout is not other._layout
    assert (rows.slots.data_ptr(), rows.expected.data_ptr()) == pointers
    assert rows.counters.cpu().tolist() == [7, 7, 7]


def test_attention_row_reuse_metadata_copies_keep_separate_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = device.DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "one", 3)
    authority.publish(version)
    binding = ExecutionAttentionBinding(authority)
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata._seq_lens = metadata._seq_lens_kv = None
    metadata.max_num_requests = 2
    metadata.kv_cache_manager = object()
    metadata.kv_lens_cuda_runtime = torch.zeros(1, dtype=torch.int, device=authority.device)
    metadata.request_ids = [17]
    request = SimpleNamespace(
        py_request_id=17, py_execution_version=version, is_cuda_graph_dummy=False
    )
    batch = SimpleNamespace(all_requests=lambda: iter([request]))
    with uploads(monkeypatch) as calls:
        binding.prepare(metadata, batch)
        binding.prepare(metadata, batch)
        assert len(calls) == 2
        other = copy.copy(metadata)
        binding.prepare(other, batch)
        binding.prepare(other, batch)
        assert len(calls) == 4
    assert other.retire_sequence_slots is not metadata.retire_sequence_slots
    assert other.retire_expected_versions.cpu().tolist() == [3, 0]
    assert metadata.retire_expected_versions.cpu().tolist() == [3, 0]
    assert binding._rows[id(other)][1]._layout is not binding._rows[id(metadata)][1]._layout
    del other
    gc.collect()
    assert len(binding._rows) == 1


@pytest.mark.parametrize("change", ["replacement", "set_storage", "inplace"])
def test_attention_row_reuse_invalidates_changed_storage(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    # Exercise tracked storage mutations under an inference-mode caller.
    with torch.inference_mode(False):
        authority = device.DeviceExecutionAuthority(torch.cuda.current_device())
        rows = authority.create_attention_rows(3)
        assert not torch.is_inference(rows.expected)
    with torch.inference_mode():
        authority.prepare_attention_rows(rows, [])
        before_version = rows.expected._version
        if change == "replacement":
            rows.expected = torch.full_like(rows.expected, 91)
        elif change == "set_storage":
            rows.expected.set_(torch.full_like(rows.expected, 91))
        else:
            rows.expected.fill_(91)
        if change != "replacement":
            assert rows.expected._version > before_version
        with uploads(monkeypatch) as calls:
            authority.prepare_attention_rows(rows, [])
            assert len(calls) == 2
            authority.prepare_attention_rows(rows, [None])
            assert len(calls) == 2
        assert rows.expected.cpu().tolist() == [0, 0, 0]


@pytest.mark.parametrize("failure", ["copy", "record"])
def test_attention_row_reuse_copy_or_record_failure_poisons(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    authority = device.DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "one", 3)
    authority.publish(version)
    rows = authority.create_attention_rows(2)
    authority.prepare_attention_rows(rows, [version])
    error = RuntimeError("injected row submission failure")
    try:
        with monkeypatch.context() as patch:
            if failure == "copy":
                original = torch.Tensor.copy_

                def copy(
                    target: torch.Tensor, source: torch.Tensor, non_blocking: bool = False
                ) -> torch.Tensor:
                    if target is rows.expected:
                        raise error
                    return original(target, source, non_blocking=non_blocking)

                patch.setattr(torch.Tensor, "copy_", copy)
            else:
                original_record = authority._record

                def record(use: device._PendingUse, stream: torch.cuda.Stream) -> None:
                    original_record(use, stream)
                    raise error

                patch.setattr(authority, "_record", record)
            with pytest.raises(RuntimeError) as raised:
                authority.prepare_attention_rows(rows, [] if failure == "copy" else [version])
            assert raised.value is authority._failure is error and rows._layout is None
        with pytest.raises(RuntimeError, match="restart required"):
            authority.prepare_attention_rows(rows, [])
    finally:
        torch.cuda.synchronize()


def test_attention_row_reuse_rejects_capture_and_stream_changes() -> None:
    authority = device.DeviceExecutionAuthority(torch.cuda.current_device())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rows = authority.create_attention_rows(2)
        authority.prepare_attention_rows(rows, [])
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        # A nondefault submitting stream permits capture before the guard rejects it.
        with torch.cuda.graph(graph, stream=stream):
            with pytest.raises(RuntimeError, match="outside capture"):
                authority.prepare_attention_rows(rows, [])
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError, match="cannot move"):
            authority.prepare_attention_rows(rows, [])
    with torch.cuda.stream(stream):
        authority.prepare_attention_rows(rows, [])
        assert allowed(authority, rows) == [True, True]
