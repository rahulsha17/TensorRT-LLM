# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authority row ordering, storage lifetime and graph refresh regression tests."""

import copy
import gc
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.execution_attention import (
    ExecutionAttentionBinding,
    attention_versions,
)
from tensorrt_llm._torch.pyexecutor.execution_device import (
    DeviceExecutionAuthority,
    _check_versions,
)
from tensorrt_llm.execution_version import ExecutionVersion

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def request(request_id, version=None, *, dummy=False):
    return SimpleNamespace(
        py_request_id=request_id, py_execution_version=version, is_cuda_graph_dummy=dummy
    )


def test_attention_order_tracks_packer_and_repeated_padding():
    version = ExecutionVersion("tenant", "scope", 0)
    live = request(17, version)
    neighbor = request(21)
    padding = request(-3, dummy=True)
    assert attention_versions([21, -3, 17, -3], [live, neighbor, padding, padding]) == [
        None,
        None,
        version,
        None,
    ]


@pytest.mark.parametrize(
    "kind", ["unknown", "duplicate_real", "packed_duplicate", "alias", "tagged_padding"]
)
def test_attention_order_rejects_ambiguous_identity(kind):
    version = ExecutionVersion("tenant", "scope", 0)
    first = request(17, version)
    if kind == "unknown":
        ids, requests = [99], [first]
    elif kind == "duplicate_real":
        ids, requests = [17, 17], [first, first]
    elif kind == "packed_duplicate":
        ids, requests = [17, 17], [first]
    elif kind == "alias":
        ids, requests = [17], [first, request(17)]
    else:
        first.is_cuda_graph_dummy = True
        ids, requests = [17, 17], [first, first]
    with pytest.raises(ValueError):
        attention_versions(ids, requests)


def test_startup_opt_in_does_not_touch_cuda_when_disabled(monkeypatch):
    monkeypatch.setenv("TRTLLM_RETIRE_NATIVE_GUARDS", "0")
    assert ExecutionAttentionBinding.from_runner(None) is None
    monkeypatch.setenv("TRTLLM_RETIRE_NATIVE_GUARDS", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        ExecutionAttentionBinding.from_runner(None)


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("fault", [None, "pipeline", "attention_dp"])
def test_startup_binding_accepts_only_lockstep_tp(monkeypatch, tp, fault):
    from tensorrt_llm._torch.pyexecutor import execution_attention as binding_module

    monkeypatch.setenv("TRTLLM_RETIRE_NATIVE_GUARDS", "1")
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    authority = Mock()
    constructor = Mock(return_value=authority)
    monkeypatch.setattr(binding_module, "DeviceExecutionAuthority", constructor)
    runner = SimpleNamespace(
        _config=SimpleNamespace(
            dtype=torch.bfloat16,
            attention_backend=SimpleNamespace(Metadata=TrtllmAttentionMetadata),
            max_beam_width=1,
            is_spec_decode=False,
            is_encode_only=False,
            is_multimodal=False,
            enable_attention_dp=fault == "attention_dp",
        ),
        mapping=SimpleNamespace(world_size=tp * (2 if fault == "pipeline" else 1), tp_size=tp),
        sparse_attention_config=None,
    )
    if fault:
        with pytest.raises(NotImplementedError, match="TP only"):
            ExecutionAttentionBinding.from_runner(runner)
        constructor.assert_not_called()
    else:
        binding = ExecutionAttentionBinding.from_runner(runner)
        assert binding.authority is authority
        constructor.assert_called_once_with(0)


@CUDA
def test_graph_replay_reads_new_row_identities_and_current_epoch():
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "scope", (1 << 63) - 1)
    neighbor = replace(old, scope_id="neighbor", epoch=4)
    authority.publish(old)
    authority.publish(neighbor)
    rows = authority.create_attention_rows(4)
    addresses = (rows.slots.data_ptr(), rows.expected.data_ptr())
    allowed = torch.empty(4, dtype=torch.bool, device="cuda")

    def check():
        _check_versions[(1,)](authority._table, rows.slots, rows.expected, allowed, 4, 128)

    authority.prepare_attention_rows(rows, [None] * 4)
    check()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        check()
    authority.publish(replace(old, epoch=1 << 63))
    snapshots = []
    for index in range(12):
        versions = [old, neighbor, None] if index % 2 == 0 else [neighbor, None, old]
        # Preparation is called from inside the executor submission lock.
        with authority.submission():
            authority.prepare_attention_rows(rows, versions)
            graph.replay()
            snapshots.append(allowed.clone())
    torch.cuda.synchronize()
    assert addresses == (rows.slots.data_ptr(), rows.expected.data_ptr())
    for index, snapshot in enumerate(snapshots):
        assert snapshot.tolist() == (
            [False, True, True, True] if index % 2 == 0 else [True, True, False, True]
        )
    # Shrinking the batch clears identities in the unused rows too.
    authority.prepare_attention_rows(rows, [neighbor])
    assert rows.slots.tolist()[1:] == [-1, -1, -1]
    assert rows.expected.tolist()[1:] == [0, 0, 0]


@CUDA
def test_attention_rows_reject_wrong_owner_overflow_and_stream():
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    rows = authority.create_attention_rows(2)
    with pytest.raises(ValueError, match="capacity"):
        authority.prepare_attention_rows(rows, [None] * 3)
    other = DeviceExecutionAuthority(torch.cuda.current_device())
    with pytest.raises(ValueError, match="another authority"):
        other.prepare_attention_rows(rows, [])
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError, match="submission streams"):
            authority.prepare_attention_rows(rows, [])
    authority.prepare_attention_rows(rows, [None])
    assert rows.slots.tolist() == [-1, -1]


@CUDA
def test_unpublished_scope_cannot_become_untagged_attention():
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    rows = authority.create_attention_rows(2)
    with pytest.raises(RuntimeError, match="no device publication"):
        authority.prepare_attention_rows(rows, [ExecutionVersion("tenant", "missing", 0)])
    with pytest.raises(RuntimeError, match="restart required"):
        authority.prepare_attention_rows(rows, [None])


@CUDA
def test_metadata_copies_have_independent_buffers_and_release_them():
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "scope", 9)
    authority.publish(version)
    binding = ExecutionAttentionBinding(authority)
    # Only the fields consumed by the binder; native-op tests cover the full
    # metadata construction and V1/V2 paged attention invocation separately.
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata._seq_lens = metadata._seq_lens_kv = None
    metadata.max_num_requests = 4
    metadata.kv_cache_manager = object()
    metadata.kv_lens_cuda_runtime = torch.zeros(2, dtype=torch.int, device="cuda")
    metadata.request_ids = [17, 21]
    batch = SimpleNamespace(all_requests=lambda: iter([request(21), request(17, version)]))
    binding.prepare(metadata, batch)
    original_slots = metadata.retire_sequence_slots.clone()
    graph_metadata = copy.copy(metadata)
    graph_metadata.request_ids = [21, 17]
    binding.prepare(graph_metadata, batch)
    assert metadata.retire_current_versions is graph_metadata.retire_current_versions
    assert (
        metadata.retire_sequence_slots.data_ptr() != graph_metadata.retire_sequence_slots.data_ptr()
    )
    assert metadata.retire_expected_versions.tolist() == [9, 0, 0, 0]
    assert graph_metadata.retire_expected_versions.tolist() == [0, 9, 0, 0]
    assert torch.equal(metadata.retire_sequence_slots, original_slots)
    assert len(binding._rows) == 2
    del graph_metadata
    gc.collect()
    assert len(binding._rows) == 1


@CUDA
def test_missing_attention_request_is_rejected_before_staging():
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    binding = ExecutionAttentionBinding(authority)
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata._seq_lens = metadata._seq_lens_kv = None
    metadata.max_num_requests = 2
    metadata.kv_cache_manager = object()
    metadata.kv_lens_cuda_runtime = torch.zeros(1, dtype=torch.int, device="cuda")
    metadata.request_ids = [17]
    with pytest.raises(ValueError, match="no execution request"):
        binding.prepare(metadata, SimpleNamespace(all_requests=lambda: iter([])))
    assert not binding._rows
