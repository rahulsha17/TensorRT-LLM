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


@CUDA
@pytest.mark.parametrize("cancellation", ["before", "between_groups", "mixed", "mixed_tagged"])
def test_unconditional_logits_sanitizer_protects_sampler_and_preserves_live_bits(cancellation):
    mixed = cancellation.startswith("mixed")
    metadata = _metadata(
        slots=(0, 1 if cancellation == "mixed_tagged" else -1, -1) if mixed else (0, -1),
        real=(True, True, False) if mixed else (True, False),
    )
    rows = metadata.retire_sequence_slots.numel()
    metadata.num_contexts = 0
    metadata.request_ids = list(range(rows))
    logits = torch.full((rows, 4), float("nan"), device="cuda")
    payload = logits.clone()
    if mixed:
        payload[1] = torch.tensor([-0.0, 1.0, -2.0, 3.0], device="cuda")
    scratch = torch.zeros(1, device="cuda")
    if cancellation == "before":
        metadata.retire_current_versions[0] = 2
    with ConditionalCapture(metadata) as graph:
        scratch.add_(1)
        if cancellation != "before":
            metadata.retire_current_versions[:1].fill_(2)
        graph.boundary()
        logits.copy_(payload)
        graph.bind_logits(logits)
    try:
        graph.replay()
        # Random sampling must be safe even though the executor rejects the
        # obsolete tokens later at its ordinary asynchronous commit boundary.
        sampled = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        torch.cuda.synchronize()
        assert torch.isfinite(logits).all()
        assert sampled.shape == (rows, 1)
        assert logits[0].eq(0).all() and logits[-1].eq(0).all()
        if mixed:
            assert torch.equal(logits[1].view(torch.int32), payload[1].view(torch.int32))
        assert graph.native.status()[-1].item() == 1
    finally:
        graph.reset()


@CUDA
def test_one_layer_group_can_skip_the_whole_forward(monkeypatch):
    monkeypatch.setenv("TRTLLM_RETIRE_GRAPH_LAYER_GROUP", "128")
    metadata = _metadata()
    value = torch.zeros(1, device="cuda")
    with ConditionalCapture(metadata) as graph:
        value.add_(7)
    try:
        graph.replay()
        torch.cuda.synchronize()
        assert value.item() == 7
        metadata.retire_current_versions[0] = 2
        value.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert value.item() == 0
        assert graph.native.status().tolist() == [0, 1]
    finally:
        graph.reset()


@CUDA
@pytest.mark.parametrize("width", [3, 16])
def test_conditional_copy_lowering_preserves_bytes_and_dependent_nodes(width):
    metadata = _metadata()
    # Offset one forces the byte path; an aligned multiple of eight uses words.
    offset = int(width == 3)
    source_storage = torch.arange(width + offset, dtype=torch.uint8, device="cuda")
    target_storage = torch.zeros(width + offset, dtype=torch.uint8, device="cuda")
    source = source_storage[offset:]
    target = target_storage[offset:]
    output = torch.zeros_like(target)
    with ConditionalCapture(metadata) as graph:
        target.copy_(source)
        target.add_(7)
        output.copy_(target)
    try:
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, source + 7)
        metadata.retire_current_versions[0] = 2
        output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert output.eq(0).all()
        assert graph.native.status().tolist() == [0, 1]
    finally:
        graph.reset()
