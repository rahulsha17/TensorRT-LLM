# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CUDA graph branch selection, live neighbors and between-group revocation."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.retire_graph import (
    ConditionalCapture,
    conditional_graph_enabled,
    layer_group_size,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _metadata(slots=(0, -1), real=(True, False)):
    return SimpleNamespace(
        retire_current_versions=torch.tensor([1, 1], dtype=torch.int64, device="cuda"),
        retire_sequence_slots=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        retire_expected_versions=torch.ones(len(slots), dtype=torch.int64, device="cuda"),
        retire_real_rows=torch.tensor(real, dtype=torch.bool, device="cuda"),
    )


def test_conditional_graph_controls_are_explicit(monkeypatch):
    monkeypatch.delenv("TRTLLM_RETIRE_CONDITIONAL_GRAPHS", raising=False)
    assert not conditional_graph_enabled()
    monkeypatch.setenv("TRTLLM_RETIRE_CONDITIONAL_GRAPHS", "sometimes")
    with pytest.raises(ValueError, match="0 or 1"):
        conditional_graph_enabled()
    monkeypatch.setenv("TRTLLM_RETIRE_GRAPH_LAYER_GROUP", "0")
    with pytest.raises(ValueError, match="between 1 and 128"):
        layer_group_size()


@CUDA
@pytest.mark.parametrize("neighbor", ["padding", "untagged", "other_scope"])
def test_conditional_graph_keeps_live_neighbors_and_ignores_padding(neighbor):
    metadata = _metadata(
        slots=(0, 1 if neighbor == "other_scope" else -1),
        real=(True, neighbor != "padding"),
    )
    value = torch.zeros(1, device="cuda")
    with ConditionalCapture(metadata) as graph:
        value.add_(1)
        graph.boundary()
        value.add_(10)
    try:
        graph.replay()
        torch.cuda.synchronize()
        assert value.item() == 11
        assert graph.native.status().tolist() == [1, 1, 1]
        metadata.retire_current_versions[0] = 2
        value.zero_()
        graph.replay()
        torch.cuda.synchronize()
        live = neighbor != "padding"
        assert value.item() == (11 if live else 0)
        assert graph.native.status().tolist() == [int(live), int(live), 1]
        metadata.retire_expected_versions[0] = 2
        value.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert value.item() == 11
    finally:
        graph.reset()


@CUDA
def test_revocation_between_groups_skips_already_submitted_later_work():
    metadata = _metadata()
    value = torch.zeros(1, device="cuda")
    with ConditionalCapture(metadata) as graph:
        value.add_(1)
        # An ordered device publication is deterministic and proves the later
        # graph group observes authority at execution, not at host submission.
        metadata.retire_current_versions.fill_(2)
        graph.boundary()
        value.add_(100)
    try:
        graph.replay()
        torch.cuda.synchronize()
        assert value.item() == 1
        assert graph.native.status().tolist() == [1, 0, 1]
    finally:
        graph.reset()


@CUDA
def test_conditional_graph_rejects_submission_stream_changes():
    metadata = _metadata()
    value = torch.zeros(1, device="cuda")
    with ConditionalCapture(metadata) as graph:
        value.add_(1)
        graph.boundary()
        value.add_(2)
    try:
        graph.replay()
        torch.cuda.synchronize()
        with torch.cuda.stream(torch.cuda.Stream()):
            with pytest.raises(RuntimeError, match="submission streams"):
                graph.replay()
    finally:
        graph.reset()
