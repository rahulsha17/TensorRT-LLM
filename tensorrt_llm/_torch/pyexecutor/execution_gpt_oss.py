# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private, opt-in GPT-OSS V2 candidate; CUDA lifecycle qualification is separate."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..models.modeling_gpt_oss import GptOssForCausalLM
    from .engine.runners.decoder.runner import DecoderRunner
    from .py_executor import PyExecutor


@dataclass(frozen=True)
class _GptOssV2Layout:
    model: "GptOssForCausalLM"
    windows: tuple[int | None, ...]
    max_seq_len: int
    num_kv_heads: int
    head_dim: int


def _startup_gpt_oss_v2(runner: "DecoderRunner") -> _GptOssV2Layout | None:
    """Latch the narrow startup layout; an unset flag preserves legacy behavior."""
    setting = os.environ.get("TRTLLM_RETIRE_GPT_OSS_V2", "0")
    if setting == "0":
        return None
    if setting != "1":
        raise ValueError("TRTLLM_RETIRE_GPT_OSS_V2 must be 0 or 1")
    if os.environ.get("TRTLLM_RETIRE_NATIVE_GUARDS", "0") != "1":
        raise ValueError("GPT-OSS V2 RETIRE requires native attention guards at startup")
    config = runner._config
    model = runner.model
    if (
        type(model).__module__ != "tensorrt_llm._torch.models.modeling_gpt_oss"
        or type(model).__name__ != "GptOssForCausalLM"
        or runner.mapping.world_size != 1
        or not config.disable_overlap_scheduler
        or config.cuda_graph_config is not None
        or config.torch_compile_enabled
        or runner._torch_compile_backend is not None
        or config.enable_in_graph_sampling
        or config.is_spec_decode
        or config.spec_config is not None
        or config.is_multimodal
        or config.is_encode_only
        or config.enable_attention_dp
        or config.max_beam_width != 1
        or config.dtype != torch.bfloat16
        or runner.sparse_attention_config is not None
        or runner.moe_load_balancer is not None
    ):
        raise NotImplementedError(
            "GPT-OSS V2 RETIRE requires BF16 TP1 eager non-overlap without speculation or EPLB"
        )
    model_config = model.model_config
    pretrained = model_config.pretrained_config
    window = pretrained.sliding_window
    blocks = model.model.block
    if (
        model_config.moe_backend != "TRITON"
        or type(window) is not int
        or not 0 < window < config.max_seq_len
        or len(blocks) != pretrained.num_hidden_layers
        or len(blocks) < 2
        or type(pretrained.num_key_value_heads) is not int
        or pretrained.num_key_value_heads <= 0
        or type(pretrained.head_dim) is not int
        or pretrained.head_dim <= 0
    ):
        raise NotImplementedError("GPT-OSS V2 RETIRE requires Triton MoE and alternating windows")
    windows = tuple(window if i % 2 == 0 else None for i in range(len(blocks)))
    # These are the actual model attention windows, not inferred cache retention.
    # Learned attention_sinks affect softmax; they do not retain KV sink tokens.
    for i, (block, expected) in enumerate(zip(blocks, windows)):
        if (
            block.layer_idx != i
            or block.attn.sliding_window != expected
            or block.attn.attn_backend != "TRTLLM"
            or tuple(block.attn.sinks.shape) != (pretrained.num_attention_heads,)
            or block.attn.sinks.dtype != torch.float32
        ):
            raise NotImplementedError("GPT-OSS V2 RETIRE model attention layout changed")
    return _GptOssV2Layout(
        model, windows, config.max_seq_len, pretrained.num_key_value_heads, pretrained.head_dim
    )


def _validate_gpt_oss_v2_executor(executor: "PyExecutor") -> bool:
    """Allow only the startup-bound model and exact synchronous V2 cache layout.

    This does not alter V2 page holders, event waits, reuse matching or admission.
    Missing window pages must still shorten or reject ALIGN through native V2.
    """
    engine = executor.model_engine
    binding = getattr(engine, "execution_attention", None)
    layout = getattr(binding, "_gpt_oss_v2_layout", None)
    if layout is None:
        return False
    from ...runtime.kv_cache_manager_v2 import AttentionLayerConfig
    from .kv_cache.kv_cache_manager_v2 import BlockReusePolicy, KVCacheManagerV2, Role
    from .resource_manager import DataType

    manager = executor.kv_cache_manager
    if (
        type(layout) is not _GptOssV2Layout
        or engine.model is not layout.model
        or layout.model.model._retire_gpt_oss_v2 is not True
        or not executor._is_kv_manager_v2
        or type(manager) is not KVCacheManagerV2
        or executor.dist.world_size != 1
        or not executor.disable_overlap_scheduler
        or executor.drafter is not None
        or engine.is_spec_decode
        or engine.cuda_graph_config is not None
        or engine._torch_compile_enabled
        or engine._torch_compile_backend is not None
        or engine.enable_in_graph_sampling
        or engine.llm_args.lora_config is not None
    ):
        raise NotImplementedError(
            "GPT-OSS RETIRE candidate requires its startup model and V2 TP1 eager"
        )
    if (
        manager.dtype != DataType.BF16
        or not manager.is_vswa
        or manager.max_seq_len != layout.max_seq_len
        or tuple(manager.pp_layers) != tuple(range(len(layout.windows)))
        or tuple(manager.max_attention_window_vec) != layout.windows
        or tuple(manager.num_kv_heads_per_layer) != (layout.num_kv_heads,) * len(layout.windows)
        or tuple(manager.head_dim_per_layer) != (layout.head_dim,) * len(layout.windows)
        or manager.tokens_per_block != manager._ledger_tokens_per_block
        or not manager.enable_block_reuse
        or manager.block_reuse_policy != BlockReusePolicy.ALL_REUSABLE
        or manager.conversation_manager is not None
        or manager.is_draft
        or manager.is_disagg
        or manager._has_cp_helix
        or manager.max_draft_len
        or manager.num_extra_kv_tokens
        or manager.kv_compression_manages_history
        or manager._supports_dense_resume_capacity()
        or manager.supports_execution_context_overlap()
    ):
        raise NotImplementedError("GPT-OSS RETIRE requires unchanged window-aware synchronous V2")
    layers = manager.kv_cache_manager_py_config.layers
    if len(layers) != len(layout.windows):
        raise NotImplementedError("GPT-OSS RETIRE V2 layer membership changed")
    for i, (layer, window) in enumerate(zip(layers, layout.windows)):
        if (
            type(layer) is not AttentionLayerConfig
            or layer.layer_id != i
            or layer.sliding_window_size != window
            or layer.num_sink_tokens is not None
            or len(layer.buffers) != 2
            or {buffer.role for buffer in layer.buffers} != {Role.KEY, Role.VALUE}
            or any(
                buffer.is_sparse
                or buffer.tokens_per_block_override is not None
                or buffer.size
                != layout.num_kv_heads * layout.head_dim * 2 * manager.tokens_per_block
                for buffer in layer.buffers
            )
        ):
            raise NotImplementedError("GPT-OSS RETIRE V2 attention buffers or retention changed")
    return True
