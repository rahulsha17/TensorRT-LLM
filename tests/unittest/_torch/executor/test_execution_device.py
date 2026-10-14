# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA authority kernels. These tests do not need model weights."""

from dataclasses import replace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.execution_device import DeviceExecutionAuthority
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_device_epoch_preserves_all_uint64_bits_and_scope_isolation() -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    previous = None
    for epoch in [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]:
        version = ExecutionVersion("tenant", "scope", epoch)
        authority.publish(version)
        assert authority.check([version, None]) == [True, True]
        if previous is not None:
            assert authority.check([previous]) == [False]
        previous = version
    other_scope = replace(previous, scope_id="other", epoch=0)
    other_tenant = replace(other_scope, tenant_id="other")
    authority.publish(other_scope)
    authority.publish(other_tenant)
    authority.publish(replace(other_scope, epoch=1))
    assert authority.check([previous, other_scope, other_tenant]) == [True, False, True]


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_token_commit_rejects_stale_rows_without_changing_candidate_scratch(compact, dtype) -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "scope", 7)
    current = replace(old, epoch=8)
    other = replace(old, scope_id="other")
    authority.publish(old)
    authority.publish(other)
    authority.publish(current)
    candidates = torch.tensor([[[11], [22], [33], [44]]], dtype=dtype, device=authority.device)
    commit = authority.commit_tokens([old, other, None], [2, 0, 3], candidates, compact)
    assert commit.allowed == [False, True, True]
    assert commit.tokens_device.data_ptr() != candidates.data_ptr()
    assert candidates.flatten().tolist() == [11, 22, 33, 44]
    assert commit.tokens_device.flatten().tolist() == [11, 0, 0, 44]
    assert commit.tokens_host.flatten().tolist() == ([0, 11, 44] if compact else [11, 0, 0, 44])
    # A recycled sequence slot does not recycle its old scope's authority.
    successor = authority.commit_tokens([current], [2], candidates, compact)
    assert successor.allowed == [True]
    assert successor.tokens_device[0, 2, 0].item() == 33


def test_unknown_scopes_and_capacity_fail_instead_of_aliasing_epoch_zero() -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device(), capacity=1)
    version = ExecutionVersion("tenant", "scope", 0)
    authority.publish(version)
    authority.publish(version)
    with pytest.raises(RuntimeError, match="capacity"):
        authority.publish(replace(version, scope_id="other"))
    assert authority.check([version]) == [True]
    with pytest.raises(RuntimeError, match="no device publication"):
        authority.check([replace(version, scope_id="unknown")])
    with pytest.raises(RuntimeError, match="restart required"):
        authority.check([version])


def test_publication_does_not_join_an_occupied_model_stream() -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "scope", 0)
    new = replace(old, epoch=1)
    authority.publish(old)
    assert authority.check([old, old]) == [True, True]  # Warm the same check shape.
    model_stream = torch.cuda.Stream()
    model_done = torch.cuda.Event()
    with torch.cuda.stream(model_stream):
        torch.cuda._sleep(1_000_000_000)
        model_done.record()
    try:
        authority.publish(new)
        # The check uses the caller stream; the publisher uses its own stream.
        assert authority.check([old, new]) == [False, True]
        assert not model_done.query(), "publication unexpectedly waited for model execution"
    finally:
        model_done.synchronize()


@pytest.mark.parametrize("revoked", [False, True])
def test_context_frontier_requires_gpu_completion_before_host_acceptance(revoked):
    from threading import Thread
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.execution_guard import ExecutionDeviceGuard
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable

    old = ExecutionVersion("tenant", "scope", 0)
    guard = object.__new__(ExecutionDeviceGuard)
    guard.device = DeviceExecutionAuthority(torch.cuda.current_device())
    guard.table = WorkerVersionTable("session")
    guard.table.bind_device_publisher(guard.device)
    guard.table.apply(VersionUpdate("session", 1, old))
    assert guard.device.check([old]) == [True]
    guard._native_kv_guards = True
    guard._early_exit_enabled = False
    guard._uses = {}
    guard._prefill_frontiers = {}
    request = SimpleNamespace(
        py_execution_version=old,
        py_request_id=42,
        py_client_id=142,
        prepopulated_prompt_len=0,
        context_current_position=0,
        context_chunk_size=64,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_decoding_iter=0,
        is_finished=False,
    )

    def finish(reason):
        request.is_finished = True
        request.state = LlmRequestState.GENERATION_COMPLETE

    request.finish_by_reason = finish
    batch = ScheduledRequests()
    batch.context_requests_chunking = [request]
    stream = torch.cuda.Stream()
    marker = torch.zeros(1, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), guard.forward(batch):
        torch.cuda._sleep(1_000_000_000)
        marker.fill_(1)
    completion = guard._uses[id(request)].event
    assert not completion.query()
    wait = guard.device.wait_event

    def wait_with_revision(event):
        publisher = Thread(
            target=lambda: guard.table.apply(VersionUpdate("session", 2, replace(old, epoch=1)))
        )
        publisher.start()
        publisher.join(timeout=2)
        assert not publisher.is_alive(), "GPU wait prevented publication"
        wait(event)

    if revoked:
        guard.device.wait_event = wait_with_revision
    try:
        with guard.context_commit(batch) as accepted:
            assert completion.query()
            assert marker.item() == 1
            assert accepted.context_requests == ([] if revoked else [request])
            if not revoked:
                request.context_current_position = 64
        assert guard._prefill_frontiers[id(request)] == (0 if revoked else 64)
    finally:
        stream.synchronize()


def test_unknown_completion_keeps_buffers_and_fails_future_guards() -> None:
    from types import SimpleNamespace

    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "scope", 0)
    authority.publish(version)
    assert authority.check([version]) == [True]
    buffer = torch.ones(1, device=authority.device)
    use = SimpleNamespace(tensors=[buffer], event=SimpleNamespace(query=lambda: False))
    authority._pending.append(use)
    with pytest.raises(TimeoutError, match="did not complete"):
        authority._wait(use, timeout=0)
    assert use in authority._pending
    assert use.tensors[0] is buffer
    with pytest.raises(RuntimeError, match="restart required"):
        authority.check([version])


@pytest.mark.parametrize("compact", [False, True])
def test_overlapping_tickets_own_tokens_and_metadata(compact) -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "scope", 0)
    authority.publish(version)
    candidates = torch.zeros((1, 4, 1), dtype=torch.int32, device=authority.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Warm this stream's allocator pool as well as the kernels before timing submission.
    with torch.cuda.stream(stream):
        warmed = [
            authority.enqueue_token_commit([version], [2], candidates, compact) for _ in range(2)
        ]
    for ticket in warmed:
        ticket.wait()
    versions, slots = [version], [2]
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
        candidates.fill_(11)
        first = authority.enqueue_token_commit(versions, slots, candidates, compact)
        candidates.fill_(22)
        second = authority.enqueue_token_commit(versions, slots, candidates, compact)
    try:
        assert not first.event.query(), "token submission unexpectedly waited for the GPU"
        versions[0], slots[0] = replace(version, epoch=1), 0
        assert first.versions == second.versions == (version,)
        assert first.sequence_slots == second.sequence_slots == (2,)
        assert first.tokens_device.data_ptr() != second.tokens_device.data_ptr()
        assert first.tokens_host.data_ptr() != second.tokens_host.data_ptr()
        for ticket, expected in [(first, 11), (second, 22)]:
            commit = ticket.wait()
            assert commit.allowed == [True]
            assert commit.tokens_host.flatten().tolist() == (
                [expected] if compact else [0, 0, expected, 0]
            )
    finally:
        stream.synchronize()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_live_authority_after_graph_replay_and_compiled_output(compiled, captured) -> None:
    """The same candidate graph is replayed across revisions without capturing an epoch."""
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "scope", (1 << 63) - 1)
    new = replace(old, epoch=1 << 63)
    other = replace(old, scope_id="other")
    authority.publish(old)
    authority.publish(other)
    logits = torch.eye(6, device=authority.device)[[1, 2, 3, 4]]
    candidates = torch.empty((1, 4, 1), device=authority.device, dtype=torch.int32)

    def sample(scores, destination):
        destination.copy_(scores.argmax(dim=-1).reshape(1, -1, 1))

    produce = torch.compile(sample, fullgraph=True) if compiled else sample
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with authority.submission(), torch.cuda.stream(warmup):
        for _ in range(3):
            produce(logits, candidates)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = None
    if captured:
        graph = torch.cuda.CUDAGraph()
        with authority.submission(), torch.cuda.graph(graph):
            produce(logits, candidates)

    def run(versions):
        with authority.submission():
            if graph is not None:
                graph.replay()
            else:
                produce(logits, candidates)
        return authority.enqueue_token_commit(versions, [2, 0, 3], candidates, compact=True).wait()

    before = run([old, other, None])
    assert before.allowed == [True, True, True]
    assert before.tokens_host.tolist() == [3, 1, 4]
    authority.publish(new)
    after = run([old, other, None])
    assert after.allowed == [False, True, True]
    assert after.tokens_device.flatten().tolist() == [1, 0, 0, 4]
    assert after.tokens_host.tolist() == [0, 1, 4]
    # Reusing the same sequence slot with the current version doesn't inherit rejection.
    assert run([new, other, None]).tokens_host.tolist() == [3, 1, 4]
    assert candidates.flatten().tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("logprobs", [False, True])
def test_native_sampler_consumes_overlap_batch_containing_canceled_request(logprobs) -> None:
    """Previous-result cancellation must not change current sampler row alignment."""
    from tensorrt_llm._torch.pyexecutor.execution_guard import ExecutionDeviceGuard
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
    from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.bindings import SamplingConfig
    from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable

    version = ExecutionVersion("tenant", "scope", 7)
    guard = object.__new__(ExecutionDeviceGuard)
    guard.device = DeviceExecutionAuthority(torch.cuda.current_device())
    guard.table = WorkerVersionTable("session")
    guard.table.bind_device_publisher(guard.device)
    guard.table.apply(VersionUpdate("session", 1, version))
    guard._uses = {}
    guard._kv_manager = None
    guard._v1_manager = None
    guard._prefill_frontiers = {}
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=32,
            max_draft_len=0,
            max_num_sequences=4,
            max_beam_width=1,
            max_total_draft_tokens=0,
            disable_overlap_scheduler=False,
        )
    )
    requests = [
        LlmRequest(
            request_id=slot,
            seq_slot=slot,
            input_tokens=[1, 2],
            max_new_tokens=8,
            end_id=7,
            sampling_config=SamplingConfig(),
            is_streaming=False,
            return_log_probs=logprobs,
            execution_version=tag,
        )
        for slot, tag in [(2, version), (0, None)]
    ]
    logits = torch.full((2, 8), -100.0, device="cuda")
    logits[0, 3] = logits[1, 5] = 100.0
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = requests
    with guard.sampling(batch):
        previous = sampler.sample_async(batch, {"logits": logits}, [0, 1, 2])
        guard.commit_sample(batch, previous)
    # The current forward's batch is fixed before the previous result is consumed.
    for request in requests:
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
    batch = ScheduledRequests()
    batch.generation_requests = requests
    guard.table.apply(VersionUpdate("session", 2, replace(version, epoch=8)))
    with guard.token_commit(previous):
        sampler.update_requests(previous)
    assert requests[0].state == LlmRequestState.GENERATION_COMPLETE
    with guard.sampling(batch):
        current = sampler.sample_async(batch, {"logits": logits}, [0])
        guard.commit_sample(batch, current)
    with guard.token_commit(current):
        sampler.update_requests(current)
    assert requests[0].get_tokens(0) == [1, 2]
    assert requests[1].get_tokens(0) == [1, 2, 5, 5]
    for request in requests:
        guard.before_release(request)


def test_native_in_graph_sampling_is_guarded_after_replay() -> None:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
    from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.bindings import SamplingConfig
    from tensorrt_llm.sampling_params import SamplingParams

    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "scope", 7)
    authority.publish(version)
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=32,
            max_draft_len=0,
            max_num_sequences=4,
            max_beam_width=1,
            max_total_draft_tokens=0,
            disable_overlap_scheduler=False,
        )
    )
    params = SamplingParams(temperature=0.8, top_k=2)
    batch = ScheduledRequests()
    batch.generation_requests = [
        LlmRequest(
            request_id=slot,
            seq_slot=slot,
            input_tokens=[1, 2],
            max_new_tokens=8,
            sampling_config=SamplingConfig(params._get_sampling_config()),
            is_streaming=False,
            execution_version=tag,
        )
        for slot, tag in [(2, version), (0, None)]
    ]
    sample_type = sampler.get_sample_type(batch.all_requests())
    assert sample_type is SampleType.FAST
    sampler.stage_in_graph_sampling(batch, sample_type)
    logits = torch.full((2, 8), -1000.0, device="cuda")
    logits[0, 3] = logits[1, 5] = 1000.0
    outputs = {"logits": logits}
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with authority.submission(), torch.cuda.stream(warmup):
        for _ in range(3):
            sampler.sample_in_graph(outputs, sample_type)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with authority.submission(), torch.cuda.graph(graph):
        sampler.sample_in_graph(outputs, sample_type)
    graph.replay()
    accepted = authority.commit_tokens([version, None], [2, 0], sampler.store.new_tokens, True)
    assert accepted.tokens_host.tolist() == [3, 5]
    authority.publish(replace(version, epoch=8))
    graph.replay()
    rejected = authority.commit_tokens([version, None], [2, 0], sampler.store.new_tokens, True)
    assert rejected.allowed == [False, True]
    assert rejected.tokens_host.tolist() == [0, 5]
    assert sampler.store.new_tokens[0, 2, 0].item() == 3


@pytest.mark.parametrize("cached", [False, True])
def test_native_v1_alignment_release_never_publishes_unexecuted_blocks(cached, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tensorrt_llm._torch.pyexecutor.execution_alignment import KVAlignmentAdapter
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm.bindings import DataType, LayerType
    from tensorrt_llm.bindings import ModelConfig as NativeModelConfig
    from tensorrt_llm.bindings.executor import SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType, LlmRequest
    from tensorrt_llm.execution_alignment import AlignmentInput
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    config = NativeModelConfig(
        vocab_size=64,
        num_layers=1,
        num_attention_layers=1,
        num_rnn_layers=0,
        num_heads=2,
        hidden_size=64,
        data_type=DataType.HALF,
    )
    config.layer_types = [LayerType.ATTENTION]
    config.set_num_kv_heads(2)
    manager = KVCacheManager(
        kv_cache_config=KvCacheConfig(max_tokens=256, enable_block_reuse=True),
        kv_cache_type=CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=32,
        tokens_per_block=16,
        max_seq_len=64,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        model_config=config,
        max_beam_width=1,
    )
    tokens = [1] * 33
    if cached:
        seed = LlmRequest(
            request_id=10,
            max_new_tokens=1,
            input_tokens=tokens,
            sampling_config=SamplingConfig(),
            is_streaming=False,
            cache_salt="shared",
        )
        manager.impl.add_sequence_batch([(10, len(tokens), 1)], [seed])
        manager.impl.refresh_blocks()
        manager._stream.synchronize()
        # Seed native ownership metadata; no model reads these synthetic values.
        manager.impl.store_blocks_for_reuse(10, seed, False)
    executor = SimpleNamespace(
        kv_cache_manager=manager,
        max_seq_len=64,
        _is_kv_manager_v2=False,
        executor_request_queue=Mock(),
    )
    executor.executor_request_queue.reserve_request_id.return_value = 11
    probe = manager.probe_prefix_match_length
    # Exercise an advisory hit that has disappeared before native allocation.
    monkeypatch.setattr(manager, "probe_prefix_match_length", lambda *args: 32)
    hold = KVAlignmentAdapter(executor).reserve(AlignmentInput(tuple(tokens), cache_salt="shared"))
    assert hold.num_tokens == (32 if cached else 0)
    if cached:
        seed.is_dummy_request = True
        manager.impl.remove_sequence(10, seed, False)
    hold.settle()
    hold.close()
    hold.close()
    assert probe(tokens, None, "shared") == (32 if cached else 0)
    assert probe(tokens, None, "other") == 0


def test_breakable_graph_revocation_preserves_unsubmitted_segments() -> None:
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import BreakableCUDAGraph
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        early_exit_scope,
    )

    values = torch.zeros(3, device="cuda")
    views = list(values.unbind())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = BreakableCUDAGraph()
    revoked = False

    def revoke():
        nonlocal revoked
        revoked = True

    with torch.cuda.stream(stream):
        for value in views:
            value.add_(1)
        stream.synchronize()
        for value in views:
            segment = torch.cuda.CUDAGraph()
            with torch.cuda.graph(segment, stream=stream):
                value.add_(1)
            graph._segments.append(segment)
        graph._break_functions = [revoke, lambda: None]
        values.zero_()
        with early_exit_scope(lambda: revoked):
            with pytest.raises(ExecutionEarlyExit):
                graph.replay()
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    assert values.tolist() == [1, 0, 0]
    with torch.cuda.stream(stream):
        graph.replay()
        done.record()
    done.synchronize()
    assert values.tolist() == [2, 1, 1]
