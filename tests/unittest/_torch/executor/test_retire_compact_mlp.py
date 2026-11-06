# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-selected compact GEMMs preserve live row order across graph replays."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.retire_graph import (
    ConditionalCapture,
    compact_mlp_enabled,
    conditional_warmup,
    run_compact_mlp,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_compact_mlp_requires_conditional_graphs(monkeypatch):
    monkeypatch.setenv("TRTLLM_RETIRE_CONDITIONAL_GRAPHS", "0")
    monkeypatch.setenv("TRTLLM_RETIRE_COMPACT_MLP", "1")
    with pytest.raises(ValueError, match="requires conditional"):
        compact_mlp_enabled()


@CUDA
def test_live_rows_choose_one_bucket_and_return_to_original_positions(monkeypatch):
    monkeypatch.setenv("TRTLLM_RETIRE_CONDITIONAL_GRAPHS", "1")
    monkeypatch.setenv("TRTLLM_RETIRE_COMPACT_MLP", "1")
    rows, width = 8, 16
    metadata = SimpleNamespace(
        num_contexts=0,
        request_ids=list(range(rows)),
        retire_current_versions=torch.ones(rows, dtype=torch.int64, device="cuda"),
        retire_sequence_slots=torch.arange(rows, dtype=torch.int32, device="cuda"),
        retire_expected_versions=torch.ones(rows, dtype=torch.int64, device="cuda"),
        retire_real_rows=torch.ones(rows, dtype=torch.bool, device="cuda"),
    )
    hidden = torch.arange(rows * width, dtype=torch.float32, device="cuda").reshape(rows, width)
    weight = torch.eye(width, dtype=torch.float32, device="cuda") * 2

    def function(value):
        return value @ weight

    with conditional_warmup(metadata):
        run_compact_mlp(function, hidden, metadata)
    with ConditionalCapture(metadata) as graph:
        result = run_compact_mlp(function, hidden, metadata)
        # The last unconditional completion marker remains outside all bodies.
        published = result.clone()
        graph.bind_logits(published)
    try:
        for survivors, bucket in [
            ([0, 1, 2, 3, 4, 5, 6, 7], 8),
            ([6, 1], 2),
            ([7], 1),
            ([0, 3, 5], 4),
            ([0, 1, 2, 4, 6], 8),
        ]:
            metadata.retire_current_versions.fill_(2)
            metadata.retire_current_versions[survivors] = 1
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(published[survivors], hidden[survivors] * 2, rtol=0, atol=0)
            status = graph.native.status().tolist()
            selected = [
                upper
                for (_, lower, upper), flag in zip(graph.predicates, status)
                if lower >= 0 and flag
            ]
            assert selected == [bucket]
            assert status[-1] == 1
        metadata.retire_current_versions.fill_(2)
        graph.replay()
        torch.cuda.synchronize()
        # A stale snapshot from the previous replay cannot keep branches alive.
        assert graph.native.status().tolist() == [0] * len(graph.segments) + [1]
        assert published.eq(0).all()
    finally:
        graph.reset()


@CUDA
def test_untagged_live_row_and_padding_have_different_compaction_semantics(monkeypatch):
    monkeypatch.setenv("TRTLLM_RETIRE_CONDITIONAL_GRAPHS", "1")
    monkeypatch.setenv("TRTLLM_RETIRE_COMPACT_MLP", "1")
    metadata = SimpleNamespace(
        num_contexts=0,
        request_ids=list(range(4)),
        retire_current_versions=torch.tensor([2], dtype=torch.int64, device="cuda"),
        retire_sequence_slots=torch.tensor([0, -1, -1, -1], dtype=torch.int32, device="cuda"),
        retire_expected_versions=torch.ones(4, dtype=torch.int64, device="cuda"),
        retire_real_rows=torch.tensor([True, True, False, False], device="cuda"),
    )
    hidden = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)

    def function(value):
        return value + 3

    with conditional_warmup(metadata):
        run_compact_mlp(function, hidden, metadata)
    with ConditionalCapture(metadata) as graph:
        result = run_compact_mlp(function, hidden, metadata)
        output = result.clone()
        graph.bind_logits(output)
    try:
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output[1], hidden[1] + 3, rtol=0, atol=0)
        torch.testing.assert_close(output[[0, 2, 3]], torch.zeros_like(output[[0, 2, 3]]))
        selected = [
            upper
            for (_, lower, upper), flag in zip(graph.predicates, graph.native.status().tolist())
            if lower >= 0 and flag
        ]
        assert selected == [1]
    finally:
        graph.reset()
