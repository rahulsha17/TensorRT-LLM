# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private, opt-in dense eager batch policy and its bounded observations."""

import json
import os
from collections import Counter, deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from ...execution_version import ExecutionVersion
    from .engine.runners.decoder.runner import DecoderRunner
    from .execution_device import DeviceTokenCommitTicket
    from .llm_request import LlmRequest
    from .sampler import SampleStateTorch
    from .sampler.sampler_features import SamplerEvent
    from .scheduler import ScheduledRequests


@dataclass(frozen=True)
class _RatePolicyConfig:
    layer_horizon: int
    min_scheduled_tokens: int


def _startup_config() -> _RatePolicyConfig | None:
    raw = os.environ.get("TRTLLM_RETIRE_BATCH_POLICY")
    if raw is None:
        return None
    fields = json.loads(raw)
    if not isinstance(fields, dict) or set(fields) != {"layer_horizon", "min_scheduled_tokens"}:
        raise ValueError("RETIRE batch policy requires layer_horizon and min_scheduled_tokens")
    if (
        type(fields["layer_horizon"]) is not int
        or fields["layer_horizon"] not in (0, 4)
        or type(fields["min_scheduled_tokens"]) is not int
        or fields["min_scheduled_tokens"] not in (1, 32, 1024)
    ):
        raise ValueError("RETIRE batch policy supports horizon 0/4 and minimum 1/32/1024")
    return _RatePolicyConfig(**fields)


@dataclass(frozen=True)
class _PolicyRow:
    request: "LlmRequest"
    request_id: int
    version: "ExecutionVersion | None"
    sequence_slot: int

    @classmethod
    def capture(cls, request: "LlmRequest") -> "_PolicyRow":
        if request.is_cuda_graph_dummy or request.is_dummy_request:
            raise NotImplementedError("RETIRE batch policy does not support dummy rows")
        if type(request.py_seq_slot) is not int or request.py_seq_slot < 0:
            raise ValueError("RETIRE batch policy requires an assigned sequence slot")
        return cls(
            request, request.py_request_id, request.py_execution_version, request.py_seq_slot
        )

    def matches(self, request: "LlmRequest") -> bool:
        return (
            self.request is request
            and self.request_id == request.py_request_id
            and self.version == request.py_execution_version
            and self.sequence_slot == request.py_seq_slot
        )


@dataclass(frozen=True)
class _BatchPolicyPlan:
    config: _RatePolicyConfig
    rows: tuple[_PolicyRow, ...]
    sampling_rows: tuple[_PolicyRow, ...]
    context_tokens: int
    decode_tokens: int
    predicates: bool
    layer_probes: bool

    def validate_rows(self, requests: list["LlmRequest"], *, sampling: bool = False) -> None:
        expected = self.sampling_rows if sampling else self.rows
        if len(requests) != len(expected) or any(
            not row.matches(request) for row, request in zip(expected, requests)
        ):
            raise RuntimeError("RETIRE policy batch identity changed")


class _RatePolicyState:
    """A startup policy; counters and the last 128 decisions contain no token data."""

    def __init__(self, config: _RatePolicyConfig) -> None:
        self.config = config
        self._lock = Lock()
        self._counts: Counter[str] = Counter()
        self._decisions: deque[dict[str, int | bool]] = deque(maxlen=128)

    def count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[name] += amount

    def decided(self, plan: _BatchPolicyPlan) -> None:
        with self._lock:
            self._counts["forward_plans"] += 1
            self._counts["predicate_plans" if plan.predicates else "boundary_plans"] += 1
            self._counts["context_tokens"] += plan.context_tokens
            self._counts["decode_tokens"] += plan.decode_tokens
            self._decisions.append(
                dict(
                    step=self._counts["forward_plans"],
                    context_tokens=plan.context_tokens,
                    decode_tokens=plan.decode_tokens,
                    real_rows=len(plan.rows),
                    tagged_rows=sum(row.version is not None for row in plan.rows),
                    predicates=plan.predicates,
                    layer_probes=plan.layer_probes,
                )
            )

    def snapshot(self) -> dict[str, object]:
        """Read host observations, not a GPU completion or kernel receipt."""
        with self._lock:
            return dict(
                layer_horizon=self.config.layer_horizon,
                min_scheduled_tokens=self.config.min_scheduled_tokens,
                counters=dict(self._counts),
                recent_decisions=[dict(row) for row in self._decisions],
                decision_retention_limit=128,
                observation="host dispatch decisions; no kernel completion claim",
            )


def _startup_policy(runner: "DecoderRunner") -> _RatePolicyState | None:
    config = _startup_config()
    if config is None:
        return None
    resolved = runner._config
    from .execution_gpt_oss import _startup_gpt_oss_v2

    supported_model = (
        type(runner.model).__module__ == "tensorrt_llm._torch.models.modeling_qwen3"
        and type(runner.model).__name__ == "Qwen3ForCausalLM"
    ) or _startup_gpt_oss_v2(runner) is not None
    if os.environ.get("TRTLLM_RETIRE_NATIVE_GUARDS", "0") != "1":
        raise ValueError("RETIRE batch policy requires native attention guards at startup")
    if os.environ.get("TRTLLM_RETIRE_EARLY_EXIT", "1") != "1":
        raise ValueError("RETIRE batch policy owns early-exit decisions; leave EARLY_EXIT enabled")
    if (
        runner.mapping.world_size != 1
        or not resolved.disable_overlap_scheduler
        or resolved.cuda_graph_config is not None
        or resolved.torch_compile_enabled
        or runner._torch_compile_backend is not None
        or resolved.is_spec_decode
        or resolved.spec_config is not None
        or resolved.is_multimodal
        or resolved.is_encode_only
        or resolved.enable_attention_dp
        or resolved.max_beam_width != 1
        or runner.sparse_attention_config is not None
        or not supported_model
    ):
        raise NotImplementedError(
            "RETIRE batch policy requires Qwen3 or opt-in GPT-OSS V2, TP1 eager non-overlap "
            "without speculation"
        )
    return _RatePolicyState(config)


class _PolicyStep:
    def __init__(self, owner: _RatePolicyState, batch: "ScheduledRequests") -> None:
        self.owner = owner
        self.batch = batch
        self.plan: _BatchPolicyPlan | None = None
        self.completed = False
        self.attention_prepared = False

    def finalize(self, batch: "ScheduledRequests", input_tokens: int, graph: bool) -> None:
        if self.plan is not None or batch is not self.batch or graph:
            raise RuntimeError("RETIRE policy requires one finalized eager physical batch")
        requests = list(batch.all_requests())
        rows = tuple(_PolicyRow.capture(request) for request in requests)
        if len({row.request_id for row in rows}) != len(rows) or len(
            {row.sequence_slot for row in rows}
        ) != len(rows):
            raise ValueError("RETIRE policy requires unique physical rows")
        context = sum(request.context_chunk_size for request in batch.context_requests)
        decode = len(batch.generation_requests)
        if (
            type(input_tokens) is not int
            or input_tokens != context + decode
            or any(request.context_chunk_size <= 0 for request in batch.context_requests)
            or any(request.py_draft_tokens for request in requests)
        ):
            raise RuntimeError("RETIRE policy token count differs from finalized unpadded input")
        sample_rows = tuple(
            _PolicyRow.capture(request)
            for request in (
                list(batch.context_requests_last_chunk) + list(batch.generation_requests)
            )
        )
        config = self.owner.config
        predicates = bool(
            config.layer_horizon
            and context + decode >= config.min_scheduled_tokens
            and any(row.version is not None for row in rows)
        )
        self.plan = _BatchPolicyPlan(
            config,
            rows,
            sample_rows,
            context,
            decode,
            predicates,
            predicates and all(row.version is not None for row in rows),
        )
        self.owner.decided(self.plan)

    def require_plan(self) -> _BatchPolicyPlan:
        if self.plan is None:
            raise RuntimeError("RETIRE policy was not finalized before execution")
        return self.plan


_step: ContextVar[_PolicyStep | None] = ContextVar("execution_rate_policy_step", default=None)


def current_policy_step() -> _PolicyStep | None:
    return _step.get()


@contextmanager
def policy_step_scope(step: _PolicyStep | None) -> Iterator[None]:
    token = _step.set(step)
    try:
        yield
    finally:
        _step.reset(token)


@dataclass(frozen=True)
class _PolicySampleReceipt:
    plan: _BatchPolicyPlan
    state: "SampleStateTorch"
    tokens_host: "torch.Tensor"
    tokens_device: "torch.Tensor"
    sampler_event: "SamplerEvent"
    ticket: "DeviceTokenCommitTicket | None"
    completion_event: "torch.cuda.Event | None"

    def validate(self, state: "SampleStateTorch") -> None:
        self.plan.validate_rows(state.requests, sampling=True)
        if (
            state is not self.state
            or state.host is None
            or state.device is None
            or state.host.new_tokens is not self.tokens_host
            or state.device.new_tokens is not self.tokens_device
            or state.sampler_event is not self.sampler_event
            or state.execution_ticket is not self.ticket
            or self.plan.predicates != (self.ticket is not None)
            or (self.completion_event is None) != self.plan.predicates
        ):
            raise RuntimeError("RETIRE boundary sample identity changed")
