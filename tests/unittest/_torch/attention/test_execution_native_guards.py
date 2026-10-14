# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native operator qualification; executor/graph metadata binding is separate."""

import backend_case
import pytest
import torch
from backend_case import BackendCase, generate_inputs, run_backend

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.execution_device import _publish_epoch
from tensorrt_llm.bindings.internal import thop

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="native RETIRE attention qualification currently targets SM90",
)

PHASES = {
    "context": ([512, 512], [0, 0], 2),
    "cached_context": ([512, 512], [128, 128], 2),
    "decode_short": ([1, 1], [127, 127], 0),
    "decode_long": ([1, 1], [2047, 2047], 0),
    "mixed_short": ([512, 1], [0, 127], 1),
    "mixed_long": ([512, 1], [0, 2047], 1),
}


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("target_row", [0, 1])
@pytest.mark.parametrize("untagged_neighbor", [False, True])
@pytest.mark.parametrize("revoked", [False, True])
@pytest.mark.parametrize(
    "phase,capture",
    [(phase, False) for phase in PHASES]
    + [(phase, True) for phase in ("decode_short", "decode_long")],
)
def test_native_attention_preserves_live_rows(
    monkeypatch, use_v2, head_dim, target_row, untagged_neighbor, revoked, phase, capture
):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    original_post_init = TrtllmAttentionMetadata.__post_init__

    def paged_post_init(self):
        original_post_init(self)
        self.use_paged_context_fmha = True

    monkeypatch.setattr(TrtllmAttentionMetadata, "__post_init__", paged_post_init)
    lengths, cached, contexts = PHASES[phase]
    case = BackendCase(
        num_heads=32,
        num_kv_heads=8,
        head_dim=head_dim,
        seq_lens=lengths,
        num_cached_tokens=cached,
        num_contexts=contexts,
        dtype="bfloat16",
        page_size=16,
        use_kv_cache_manager_v2=use_v2,
        rope={"dim": head_dim, "theta": 1_000_000.0, "max_positions": 4096},
        fused_rope=True,
    )
    inputs = generate_inputs(case, seed=31)
    options = dict(kv_dtype=torch.bfloat16, kv_layout="HND", fuse_rope=True)
    baseline = run_backend(case, "TRTLLM", inputs, **options).clone()
    torch.cuda.synchronize()

    # Cross the signed boundary without losing any execution-version bits.
    old_epoch = (1 << 63) - 1
    new_epoch_word = -(1 << 63)
    table = torch.tensor([old_epoch, 9], device="cuda", dtype=torch.int64)
    slots = torch.full((2,), -1, device="cuda", dtype=torch.int32)
    expected = torch.zeros(2, device="cuda", dtype=torch.int64)
    counters = torch.zeros(3, device="cuda", dtype=torch.int64)
    row_slots = [0, -1 if untagged_neighbor else 1]
    row_epochs = [old_epoch, 0 if untagged_neighbor else 9]
    if target_row == 1:
        row_slots.reverse()
        row_epochs.reverse()
    row_slots = torch.tensor(row_slots, device="cuda", dtype=torch.int32)
    row_epochs = torch.tensor(row_epochs, device="cuda", dtype=torch.int64)
    epoch_word = torch.tensor([new_epoch_word], device="cuda", dtype=torch.int64)
    # Compile the atomic publisher before any graph capture.
    old_word = torch.tensor([old_epoch], device="cuda", dtype=torch.int64)
    _publish_epoch[(1,)](table, old_word, 0, num_warps=1)
    publisher = torch.cuda.Stream(priority=-1)
    native_attention = thop.attention
    original_forward = FallbackFmha.forward
    original_check = backend_case._assert_cache_contains_new_tokens
    observed = {}

    def instrument_forward(self, q, k, v, metadata, forward_args):
        buffer = metadata.kv_cache_manager.get_buffers(0, kv_layout="HND")
        blocks = [
            int(block)
            for block in metadata.kv_cache_manager.get_batch_cache_indices([target_row], 0)[0]
            if int(block) >= 0
        ]
        observed.update(buffer=buffer, blocks=blocks)
        original_forward(self, q, k, v, metadata, forward_args)
        torch.cuda.synchronize()
        observed["target_cache"] = buffer[blocks].clone()

    def guarded_attention(**kwargs):
        kwargs.update(
            retire_current_versions=table,
            retire_sequence_slots=slots,
            retire_expected_versions=expected,
            retire_counters=counters,
        )
        graph = None
        if capture:
            native_attention(**kwargs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                native_attention(**kwargs)
        # Capture initially saw untagged rows. Replay must consume new row
        # identities and a newly published epoch from the same stable addresses.
        slots.copy_(row_slots)
        expected.copy_(row_epochs)
        if revoked:
            observed["buffer"][observed["blocks"]] = -13
            publisher.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(publisher):
                _publish_epoch[(1,)](table, epoch_word, 0, num_warps=1)
            torch.cuda.current_stream().wait_stream(publisher)
        kwargs["output"].fill_(-21)
        counters.zero_()
        if graph is not None:
            graph.replay()
        else:
            native_attention(**kwargs)
        torch.cuda.synchronize()
        observed["output"] = kwargs["output"].clone()

    def check_live_cache(manager, layer, ids, lengths, cached, expected, **kwargs):
        if revoked:
            live = 1 - target_row
            original_check(
                manager,
                layer,
                [ids[live]],
                [lengths[live]],
                [cached[live]],
                [expected[live]],
                **kwargs,
            )
        else:
            original_check(manager, layer, ids, lengths, cached, expected, **kwargs)

    monkeypatch.setattr(FallbackFmha, "forward", instrument_forward)
    monkeypatch.setattr(thop, "attention", guarded_attention)
    monkeypatch.setattr(backend_case, "_assert_cache_contains_new_tokens", check_live_cache)
    actual = run_backend(case, "TRTLLM", inputs, **options)
    live_rows = slice(lengths[0], None) if target_row == 0 else slice(0, lengths[0])
    stale_rows = slice(0, lengths[0]) if target_row == 0 else slice(lengths[0], None)
    assert torch.equal(actual[live_rows], baseline[live_rows]), "live neighbor changed"
    if revoked:
        assert torch.all(observed["target_cache"] == -13), "stale KV store escaped"
        assert torch.all(observed["output"][stale_rows] == -21), "stale output store escaped"
        assert counters[:2].sum().item() > 0
        assert counters[2].item() > 0
    else:
        assert torch.equal(actual, baseline)
        assert counters.tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    "invalid",
    ["partial", "host_table", "slot_dtype", "expected_rows", "counter_size", "empty_table"],
)
def test_native_attention_rejects_invalid_authority_metadata(monkeypatch, invalid):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    case = BackendCase(
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        seq_lens=[16, 16],
        num_cached_tokens=[0, 0],
        num_contexts=2,
        dtype="bfloat16",
        page_size=16,
    )
    native_attention = thop.attention

    def malformed_attention(**kwargs):
        authority = dict(
            retire_current_versions=torch.zeros(2, device="cuda", dtype=torch.int64),
            retire_sequence_slots=torch.zeros(2, device="cuda", dtype=torch.int32),
            retire_expected_versions=torch.zeros(2, device="cuda", dtype=torch.int64),
            retire_counters=torch.zeros(3, device="cuda", dtype=torch.int64),
        )
        if invalid == "partial":
            authority["retire_counters"] = None
        elif invalid == "host_table":
            authority["retire_current_versions"] = authority["retire_current_versions"].cpu()
        elif invalid == "slot_dtype":
            authority["retire_sequence_slots"] = authority["retire_sequence_slots"].float()
        elif invalid == "expected_rows":
            authority["retire_expected_versions"] = authority["retire_expected_versions"][:1]
        elif invalid == "counter_size":
            authority["retire_counters"] = authority["retire_counters"][:1]
        elif invalid == "empty_table":
            authority["retire_current_versions"] = authority["retire_current_versions"][:0]
        kwargs.update(authority)
        native_attention(**kwargs)

    monkeypatch.setattr(thop, "attention", malformed_attention)
    with pytest.raises(RuntimeError, match="RETIRE authority|RETIRE attention"):
        run_backend(
            case, "TRTLLM", generate_inputs(case, seed=31), kv_dtype=torch.bfloat16, kv_layout="HND"
        )
