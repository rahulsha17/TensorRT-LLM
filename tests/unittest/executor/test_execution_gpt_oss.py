# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local CPU/QA candidate contracts; device doubles do not qualify CUDA behavior."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_execution_device_guard import _executor
from test_execution_rate_policy import finalized, policy, request

from tensorrt_llm._torch.models.modeling_gpt_oss import Transformer
from tensorrt_llm._torch.pyexecutor.execution_alignment import KVAlignmentAdapter
from tensorrt_llm._torch.pyexecutor.execution_attention import ExecutionAttentionBinding
from tensorrt_llm._torch.pyexecutor.execution_early_exit import ExecutionEarlyExit, early_exit_scope
from tensorrt_llm._torch.pyexecutor.execution_gpt_oss import (
    _startup_gpt_oss_v2,
    _validate_gpt_oss_v2_executor,
)
from tensorrt_llm._torch.pyexecutor.execution_guard import ExecutionDeviceGuard
from tensorrt_llm._torch.pyexecutor.execution_rate_policy import _startup_policy, policy_step_scope
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.execution_alignment import AlignmentInput
from tensorrt_llm.execution_mailbox import WorkerVersionTable
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, BufferConfig

pytestmark = pytest.mark.cpu_only


def runner() -> SimpleNamespace:
    model_type = type(
        "GptOssForCausalLM", (), {"__module__": "tensorrt_llm._torch.models.modeling_gpt_oss"}
    )
    model = model_type()
    model.model_config = SimpleNamespace(
        moe_backend="TRITON",
        pretrained_config=SimpleNamespace(
            sliding_window=128,
            num_hidden_layers=24,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=64,
        ),
    )
    model.model = SimpleNamespace(
        _retire_gpt_oss_v2=False,
        block=[
            SimpleNamespace(
                layer_idx=i,
                attn=SimpleNamespace(
                    sliding_window=128 if i % 2 == 0 else None,
                    attn_backend="TRTLLM",
                    sinks=SimpleNamespace(shape=(64,), dtype=torch.float32),
                ),
            )
            for i in range(24)
        ],
    )
    return SimpleNamespace(
        model=model,
        mapping=SimpleNamespace(world_size=1),
        _torch_compile_backend=None,
        sparse_attention_config=None,
        moe_load_balancer=None,
        _config=SimpleNamespace(
            max_seq_len=16384,
            disable_overlap_scheduler=True,
            cuda_graph_config=None,
            torch_compile_enabled=False,
            enable_in_graph_sampling=False,
            is_spec_decode=False,
            spec_config=None,
            is_multimodal=False,
            is_encode_only=False,
            enable_attention_dp=False,
            max_beam_width=1,
            dtype=torch.bfloat16,
        ),
    )


@pytest.fixture
def candidate(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setenv("TRTLLM_RETIRE_GPT_OSS_V2", "1")
    monkeypatch.setenv("TRTLLM_RETIRE_NATIVE_GUARDS", "1")
    monkeypatch.setenv("TRTLLM_RETIRE_EARLY_EXIT", "1")
    value = runner()
    layout = _startup_gpt_oss_v2(value)
    executor = _executor()
    executor._is_kv_manager_v2 = True
    executor.executor_request_queue = Mock()
    executor.model_engine.model = value.model
    executor.model_engine._torch_compile_enabled = False
    executor.model_engine._torch_compile_backend = None
    executor.model_engine.llm_args = SimpleNamespace(lora_config=None)
    executor.model_engine.execution_attention = ExecutionAttentionBinding(
        Mock(), gpt_oss_v2_layout=layout
    )
    value.model.model._retire_gpt_oss_v2 = True
    manager = object.__new__(KVCacheManagerV2)
    manager.dtype = DataType.BF16
    manager.is_vswa = True
    manager.max_seq_len = layout.max_seq_len
    manager.pp_layers = list(range(24))
    manager.max_attention_window_vec = list(layout.windows)
    manager.num_kv_heads_per_layer = [8] * 24
    manager.head_dim_per_layer = [64] * 24
    manager.tokens_per_block = manager._ledger_tokens_per_block = 16
    manager.enable_block_reuse = True
    manager.impl = SimpleNamespace(cache_tier_list=["gpu", "host"])
    manager.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
    manager.conversation_manager = manager.kv_connector_manager = None
    manager.is_draft = manager.is_disagg = manager._has_cp_helix = False
    manager.kv_compression_manages_history = False
    manager.max_draft_len = manager.num_extra_kv_tokens = 0
    manager.max_beam_width = 1
    manager._can_publish_block_reuse = True
    manager.kv_cache_manager_py_config = SimpleNamespace(
        layers=[
            AttentionLayerConfig(
                layer_id=i,
                sliding_window_size=window,
                buffers=[BufferConfig(role=role, size=16384) for role in (Role.KEY, Role.VALUE)],
            )
            for i, window in enumerate(layout.windows)
        ]
    )
    executor.kv_cache_manager = manager
    executor.max_seq_len = layout.max_seq_len
    return SimpleNamespace(runner=value, executor=executor, manager=manager)


def test_unset_capability_reads_neither_model_nor_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_RETIRE_GPT_OSS_V2", raising=False)
    assert _startup_gpt_oss_v2(None) is None


@pytest.mark.parametrize("setting", ["", "true", "2"])
def test_bad_opt_in_fails_before_model_access(monkeypatch, setting) -> None:
    monkeypatch.setenv("TRTLLM_RETIRE_GPT_OSS_V2", setting)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _startup_gpt_oss_v2(None)


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("", "model", object()),
        ("mapping", "world_size", 2),
        ("", "_torch_compile_backend", object()),
        ("", "sparse_attention_config", object()),
        ("", "moe_load_balancer", object()),
        ("_config", "disable_overlap_scheduler", False),
        ("_config", "cuda_graph_config", {}),
        ("_config", "torch_compile_enabled", True),
        ("_config", "enable_in_graph_sampling", True),
        ("_config", "is_spec_decode", True),
        ("_config", "spec_config", object()),
        ("_config", "is_multimodal", True),
        ("_config", "is_encode_only", True),
        ("_config", "enable_attention_dp", True),
        ("_config", "max_beam_width", 2),
        ("_config", "dtype", torch.float32),
    ],
)
def test_startup_keeps_unsupported_modes_closed(candidate, target, field, value) -> None:
    obj = getattr(candidate.runner, target) if target else candidate.runner
    setattr(obj, field, value)
    with pytest.raises(NotImplementedError):
        _startup_gpt_oss_v2(candidate.runner)


@pytest.mark.parametrize("fault", ["native_off", "backend", "full", "window", "sink", "layers"])
def test_startup_validates_actual_model_layout(candidate, monkeypatch, fault) -> None:
    model = candidate.runner.model
    if fault == "native_off":
        monkeypatch.setenv("TRTLLM_RETIRE_NATIVE_GUARDS", "0")
    elif fault == "backend":
        model.model_config.moe_backend = "CUTLASS"
    elif fault == "full":
        model.model.block[0].attn.sliding_window = None
    elif fault == "window":
        model.model.block[1].attn.sliding_window = 128
    elif fault == "sink":
        model.model.block[0].attn.sinks.shape = (8,)
    else:
        model.model.block.pop()
    with pytest.raises((ValueError, NotImplementedError)):
        _startup_gpt_oss_v2(candidate.runner)


def test_capability_is_startup_bound_and_policy_is_separate(candidate, monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_RETIRE_BATCH_POLICY", raising=False)
    assert _startup_policy(candidate.runner) is None
    monkeypatch.setenv(
        "TRTLLM_RETIRE_BATCH_POLICY", '{"layer_horizon":4,"min_scheduled_tokens":32}'
    )
    assert _startup_policy(candidate.runner).config.layer_horizon == 4
    monkeypatch.delenv("TRTLLM_RETIRE_GPT_OSS_V2")
    with pytest.raises(NotImplementedError):
        _startup_policy(candidate.runner)
    # Once constructed the binding is fixed; changing the environment does not revoke it.
    assert _validate_gpt_oss_v2_executor(candidate.executor)
    candidate.executor.model_engine.execution_attention = ExecutionAttentionBinding(Mock())
    monkeypatch.setenv("TRTLLM_RETIRE_GPT_OSS_V2", "1")
    with pytest.raises(NotImplementedError, match="variable sliding windows"):
        ExecutionDeviceGuard.validate_executor(candidate.executor)


def test_guard_keeps_real_dense_optimizations_disabled(candidate) -> None:
    assert not candidate.manager._supports_dense_resume_capacity()
    assert not candidate.manager.supports_execution_context_overlap()
    guard = ExecutionDeviceGuard(candidate.executor, WorkerVersionTable("session"))
    assert guard._native_kv_guards
    assert not guard._async_context
    assert guard._kv_manager is candidate.manager


def test_real_startup_binding_latches_hook_only_after_guard_checks(candidate, monkeypatch) -> None:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    value = candidate.runner
    value.model.model._retire_gpt_oss_v2 = False
    value.mapping.tp_size = 1
    value._config.attention_backend = SimpleNamespace(Metadata=TrtllmAttentionMetadata)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    authority = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.execution_attention.DeviceExecutionAuthority",
        lambda device: authority,
    )
    monkeypatch.setenv("TLLM_FMHA_LIBS", "other")
    with pytest.raises(ValueError, match="fallback"):
        ExecutionAttentionBinding.from_runner(value)
    assert not value.model.model._retire_gpt_oss_v2
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    binding = ExecutionAttentionBinding.from_runner(value)
    assert binding.authority is authority
    assert binding._gpt_oss_v2_layout.model is value.model
    assert value.model.model._retire_gpt_oss_v2


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("", "_is_kv_manager_v2", False),
        ("", "disable_overlap_scheduler", False),
        ("", "drafter", object()),
        ("dist", "world_size", 2),
        ("model_engine", "model", object()),
        ("model_engine", "cuda_graph_config", {}),
        ("model_engine", "_torch_compile_enabled", True),
        ("model_engine", "_torch_compile_backend", object()),
        ("model_engine", "enable_in_graph_sampling", True),
        ("kv_cache_manager", "dtype", DataType.HALF),
        ("kv_cache_manager", "is_vswa", False),
        ("kv_cache_manager", "max_seq_len", 8192),
        ("kv_cache_manager", "max_attention_window_vec", [None] * 24),
        ("kv_cache_manager", "num_kv_heads_per_layer", [4] * 24),
        ("kv_cache_manager", "head_dim_per_layer", [128] * 24),
        ("kv_cache_manager", "pp_layers", list(reversed(range(24)))),
        ("kv_cache_manager", "_ledger_tokens_per_block", 32),
        ("kv_cache_manager", "block_reuse_policy", BlockReusePolicy.PER_REQUEST),
        ("kv_cache_manager", "conversation_manager", object()),
        ("kv_cache_manager", "kv_compression_manages_history", True),
    ],
)
def test_actual_executor_cannot_widen_startup_contract(candidate, target, field, value) -> None:
    obj = getattr(candidate.executor, target) if target else candidate.executor
    setattr(obj, field, value)
    with pytest.raises(NotImplementedError):
        ExecutionDeviceGuard.validate_executor(candidate.executor)


@pytest.mark.parametrize("fault", ["sink", "window", "sparse", "size", "role", "override"])
def test_native_layer_buffers_are_checked_independently(candidate, fault) -> None:
    layer = candidate.manager.kv_cache_manager_py_config.layers[0]
    buffers = list(layer.buffers)
    if fault == "sink":
        layer.num_sink_tokens = 128
    elif fault == "window":
        layer.sliding_window_size = 64
    elif fault == "sparse":
        buffers[0].is_sparse = True
    elif fault == "size":
        buffers[0].size = 0
    elif fault == "role":
        buffers[0].role = Role.VALUE
    else:
        buffers[0].tokens_per_block_override = 8
    layer.buffers = buffers
    with pytest.raises(NotImplementedError, match="buffers or retention"):
        ExecutionDeviceGuard.validate_executor(candidate.executor)


@pytest.mark.parametrize(
    "field", ["guided_decoder", "kv_cache_transceiver", "kv_connector_manager"]
)
def test_other_native_restrictions_are_not_bypassed(candidate, field) -> None:
    setattr(candidate.executor, field, object())
    with pytest.raises(NotImplementedError, match="RETIRE device guards"):
        ExecutionDeviceGuard.validate_executor(candidate.executor)


def test_windowed_alignment_keeps_native_suspended_hold_without_dense_lock(candidate) -> None:
    cache = SimpleNamespace(num_committed_tokens=112, close=Mock(), resume=Mock())
    candidate.manager.impl = SimpleNamespace(
        tokens_per_block=16,
        cache_tier_list=["gpu", "host"],
        get_life_cycle_pool_group_indices=lambda level: [0, 1],
        get_storage_statistics=lambda level: [SimpleNamespace(total=16), SimpleNamespace(total=8)],
        create_kv_cache=Mock(return_value=cache),
    )
    candidate.manager._derive_reuse_salt = Mock(return_value=42)
    adapter = KVAlignmentAdapter(candidate.executor, keep_resident=True)
    assert adapter._v2_reservation_block_limit() == 8
    revised = AlignmentInput(tuple(range(256)), cache_salt="test", max_reuse_tokens=128)
    reservation = adapter.reserve(revised)
    # The native matcher reports 112, not the requested 128; never round it up.
    assert reservation.num_tokens == 112
    assert adapter._reserved_v2_blocks == 7
    assert not reservation.has_resident_prefix
    cache.resume.assert_not_called()
    candidate.manager._derive_reuse_salt.assert_called_once_with("test")
    call = candidate.manager.impl.create_kv_cache.call_args
    assert call.args[0].salt == 42
    assert call.args[1] == list(range(128))
    assert call.kwargs["expected_prompt_length"] == 256
    reservation.close()
    reservation.close()
    cache.close.assert_called_once()
    assert adapter._reserved_v2_blocks == 0


@pytest.mark.parametrize("horizon,minimum,chunk", [(4, 1, 32), (0, 1, 32), (4, 1024, 32)])
@pytest.mark.parametrize("execution", ["eager", "capture", "compile"])
def test_real_model_loop_obeys_h4_policy_and_capture_guard(
    monkeypatch, horizon, minimum, chunk, execution
) -> None:
    calls = []
    probes = []
    fake_model = SimpleNamespace(
        _retire_gpt_oss_v2=True,
        embedding=lambda ids: "hidden",
        block=[
            Mock(side_effect=lambda **kw: (calls.append(len(calls)) or "hidden", None))
            for _ in range(24)
        ],
    )
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request(context_tokens=chunk)]
    step = finalized(policy(horizon, minimum), batch)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: execution == "compile")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: execution == "capture")
    with (
        policy_step_scope(step),
        early_exit_scope(lambda: False, observer=lambda *args: probes.append(args[1])),
    ):
        result = Transformer.forward(fake_model, None, input_ids=[1])
    assert result == "hidden" and len(calls) == 24
    assert probes == (
        [4, 8, 12, 16, 20] if horizon and chunk >= minimum and execution == "eager" else []
    )


def test_real_model_early_exit_stops_after_four_blocks_and_scope_restores() -> None:
    calls = []
    model = SimpleNamespace(
        _retire_gpt_oss_v2=True,
        embedding=lambda ids: "hidden",
        block=[
            Mock(side_effect=lambda **kw: (calls.append(len(calls)) or "hidden", None))
            for _ in range(24)
        ],
    )
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request(context_tokens=32)]
    with pytest.raises(ExecutionEarlyExit) as raised:
        with policy_step_scope(finalized(policy(4, 1), batch)), early_exit_scope(lambda: True):
            Transformer.forward(model, None, input_ids=[1])
    assert (raised.value.boundary, raised.value.submitted, raised.value.total) == (
        "eager_layer",
        4,
        24,
    )
    assert len(calls) == 4
    Transformer.forward(model, None, input_ids=[1])
    assert len(calls) == 28


def test_unopted_model_keeps_legacy_forward_even_with_ambient_poll() -> None:
    blocks = [Mock(return_value=("hidden", None)) for _ in range(24)]
    model = SimpleNamespace(_retire_gpt_oss_v2=False, embedding=lambda ids: "hidden", block=blocks)
    poll = Mock(side_effect=AssertionError("legacy GPT-OSS must not acquire layer polls"))
    with early_exit_scope(poll):
        assert Transformer.forward(model, None, input_ids=[1]) == "hidden"
    assert all(block.call_count == 1 for block in blocks)
    poll.assert_not_called()
