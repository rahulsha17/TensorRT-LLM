# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
from tensorrt_llm._torch.pyexecutor.request_utils import (
    RequestBroadcaster,
    attach_py_objects_to_requests,
)
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.llmapi.llm import BaseLLM, PreprocessedInputs, RequestOutput
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

VERSION = ExecutionVersion("tenant", "scope", 7)


def _request(
    version: ExecutionVersion | None = VERSION, cache_salt: str | None = "shared-prefix"
) -> GenerationRequest:
    request = GenerationRequest(
        [1, 2, 3],
        SamplingParams(max_tokens=5, seed=1, n=2, top_k=2),
        cache_salt=cache_salt,
        execution_version=version,
    )
    request.set_id(42)
    return request


@pytest.mark.parametrize("version", [VERSION, None])
@pytest.mark.parametrize("cache_salt", ["shared-prefix", "", None])
def test_request_ipc_and_result_keep_version(
    version: ExecutionVersion | None, cache_salt: str | None
) -> None:
    request = pickle.loads(pickle.dumps(_request(version, cache_salt)))
    assert request.execution_version == version
    assert request.id == 42
    assert request.cache_salt == cache_salt
    result = GenerationResult(request)
    output = RequestOutput._from_generation_result(result)
    assert result.execution_version == version
    assert output.execution_version == version
    assert result.cache_salt == output.cache_salt == cache_salt
    with pytest.raises(AttributeError):
        output.cache_salt = "another-prefix"
    with pytest.raises(AttributeError):
        request.execution_version = ExecutionVersion("tenant", "scope", 8)


def test_revision_can_keep_the_predecessors_cache_namespace() -> None:
    revised_version = ExecutionVersion(VERSION.tenant_id, VERSION.scope_id, VERSION.epoch + 1)
    original = GenerationResult(pickle.loads(pickle.dumps(_request(VERSION))))
    revision = GenerationResult(pickle.loads(pickle.dumps(_request(revised_version))))
    assert original.execution_version != revision.execution_version
    assert original.cache_salt == revision.cache_salt == "shared-prefix"


def test_cache_namespace_is_independent_of_execution_identity() -> None:
    first = GenerationResult(_request(VERSION, "cache-a"))
    second = GenerationResult(_request(VERSION, "cache-b"))
    assert first.execution_version == second.execution_version == VERSION
    assert first.cache_salt == "cache-a"
    assert second.cache_salt == "cache-b"


def test_legacy_request_state_defaults_to_untagged() -> None:
    state = _request(None).__getstate__()
    state.pop("_execution_version")
    restored = GenerationRequest.__new__(GenerationRequest)
    restored.__setstate__(state)
    assert restored.execution_version is None
    assert restored.prompt_token_ids == [1, 2, 3]


def test_invalid_request_version_is_rejected() -> None:
    with pytest.raises(TypeError, match="execution_version"):
        _request(("tenant", "scope", 7))


class _Executor(GenerationExecutor):
    def __init__(self) -> None:
        self.submitted = []

    def _maybe_initialize_iteration_results(self) -> None:
        pass

    def submit(self, request: GenerationRequest) -> GenerationResult:
        request.set_id(len(self.submitted) + 1)
        self.submitted.append(request)
        return MagicMock(spec=GenerationResult)

    def abort_request(self, request_id: int) -> None:
        pass

    def shutdown(self) -> None:
        pass


@pytest.mark.parametrize("versions", [VERSION, [VERSION, None]])
def test_executor_submission_carries_versions(versions) -> None:
    executor = _Executor()
    executor.generate([[1], [2]], SamplingParams(max_tokens=1), execution_version=versions)
    expected = versions if isinstance(versions, list) else [VERSION, VERSION]
    assert [r.execution_version for r in executor.submitted] == expected
    assert executor.submitted[0].id != executor.submitted[1].id


def test_executor_rejects_batch_length_mismatch_before_submission() -> None:
    executor = _Executor()
    with pytest.raises(ValueError, match="length"):
        executor.generate([[1], [2]], SamplingParams(max_tokens=1), execution_version=[VERSION])
    assert executor.submitted == []


@pytest.mark.parametrize("versions", [VERSION, [VERSION, None], (VERSION, None)])
def test_llm_batch_forwards_versions(versions) -> None:
    llm = MagicMock()
    llm._is_unbatched_optional_inputs.return_value = False
    llm._item_at = BaseLLM._item_at
    BaseLLM.generate(llm, ["one", "two"], execution_version=versions, use_tqdm=False)
    expected = [VERSION, VERSION] if isinstance(versions, ExecutionVersion) else list(versions)
    assert [c.kwargs["execution_version"] for c in llm.generate_async.call_args_list] == expected


def test_llm_rejects_batch_length_mismatch_before_submission() -> None:
    llm = MagicMock()
    llm._is_unbatched_optional_inputs.return_value = False
    with pytest.raises(ValueError, match="length"):
        BaseLLM.generate(llm, ["one", "two"], execution_version=[VERSION], use_tqdm=False)
    llm.generate_async.assert_not_called()


@pytest.mark.parametrize("version", [VERSION, None])
def test_llm_async_submission_and_output_keep_version(version: ExecutionVersion | None) -> None:
    request = _request(version)
    llm = MagicMock()
    llm._encode_only = False
    llm._executor.is_shutdown.return_value = False
    llm._prepare_sampling_params.return_value = request.sampling_params
    llm.args.return_perf_metrics = False
    llm._executor.generate_async.return_value = GenerationResult(request)
    output = BaseLLM.generate_async(
        llm,
        PreprocessedInputs([1, 2, 3]),
        execution_version=version,
    )
    assert llm._executor.generate_async.call_args.kwargs["execution_version"] == version
    assert output.execution_version == version


@pytest.mark.parametrize("version", [VERSION, None])
def test_worker_and_native_broadcast_preserve_version(version: ExecutionVersion | None) -> None:
    worker = MagicMock()
    worker.llm_args.max_beam_width = 1
    worker.llm_args.return_perf_metrics = False
    worker._is_pytorch_backend = True
    worker.max_seq_len = 128
    worker.mapping.cp_size = 1
    worker.engine.enqueue_request.return_value = 99
    request = pickle.loads(pickle.dumps(_request(version)))
    assert BaseWorker._enqueue_request(worker, request) == 99
    native_request = worker.engine.enqueue_request.call_args.args[0]
    assert native_request.py_execution_version == version
    assert native_request.cache_salt == "shared-prefix"

    items = [RequestQueueItem(99, native_request)]
    py_objects = RequestBroadcaster._collect_py_objects(None, items)
    restored_items, restored_objects = pickle.loads(pickle.dumps((items, py_objects)))
    attach_py_objects_to_requests(restored_items, restored_objects)
    restored_request = restored_items[0].request
    llm_request = executor_request_to_llm_request(
        99,
        restored_request,
        child_req_ids=[100],
        exclude_last_generation_logits=False,
    )
    assert llm_request.py_execution_version == version
    assert llm_request.py_request_id == 99
    assert llm_request.child_requests[0].py_execution_version == version


def test_broadcast_keeps_request_versions_separate() -> None:
    other = ExecutionVersion("other-tenant", "scope", 7)
    source = [
        RequestQueueItem(
            1, SimpleNamespace(py_execution_version=VERSION, py_conversation_params=None)
        ),
        RequestQueueItem(
            2, SimpleNamespace(py_execution_version=other, py_conversation_params=None)
        ),
    ]
    payload = pickle.loads(pickle.dumps(RequestBroadcaster._collect_py_objects(None, source)))
    target = [RequestQueueItem(2, SimpleNamespace()), RequestQueueItem(1, SimpleNamespace())]
    attach_py_objects_to_requests(target, payload)
    assert target[0].request.py_execution_version == other
    assert target[1].request.py_execution_version == VERSION


def test_direct_native_request_remains_untagged() -> None:
    native_request = tllm.Request(input_token_ids=[1, 2, 3], max_tokens=5)
    llm_request = executor_request_to_llm_request(
        99,
        native_request,
        child_req_ids=[],
        exclude_last_generation_logits=False,
    )
    assert llm_request.py_execution_version is None
