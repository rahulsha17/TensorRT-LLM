# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use real result and executor methods, with inference replaced at the engine boundary."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from queue import Queue
from threading import Event
from unittest.mock import MagicMock

import pytest
import zmq

from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import (
    ExecutionController,
    LocalPublicationTransport,
    WorkerVersionTable,
)
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.execution_control import ExecutionControlMixin
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.utils import AsyncQueue
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 7)
REVISION = Revision("edit", VERSION, replace(VERSION, epoch=8))


@pytest.fixture
def controller():
    table = WorkerVersionTable("session")
    control = ExecutionController("session", LocalPublicationTransport(table))
    control.register(VERSION).wait()
    try:
        yield control
    finally:
        control.close()
        table.close()


def result_shell(controller, queue=None):
    result = object.__new__(GenerationResult)
    result._done = False
    result._terminal_error = None
    result._execution = None
    result.queue = Queue() if queue is None else queue
    result.aqueue = None
    result._handle_response = MagicMock()
    with controller.admission(VERSION) as execution:
        ExecutionControlMixin._bind_execution_result(result, execution)
    return result


@pytest.mark.parametrize("consume", ["_result_step", "result", "__next__"])
def test_buffered_output_is_rejected_by_sync_result_apis(controller, consume) -> None:
    result = result_shell(controller)
    assert result._publish_response(lambda: result.queue.put("old token"))
    controller.advance(REVISION)
    with pytest.raises(ExecutionAuthorityError):
        getattr(result, consume)()
    result._handle_response.assert_not_called()
    assert result.finished
    assert isinstance(result._terminal_error, ExecutionAuthorityError)
    with pytest.raises(ExecutionAuthorityError):
        result.result()


@pytest.mark.parametrize("consume", ["_aresult_step", "aresult", "__anext__"])
def test_buffered_output_is_rejected_by_async_result_apis(controller, consume) -> None:
    async def run():
        queue = AsyncQueue()
        result = result_shell(controller, queue.sync_q)
        result.aqueue = queue
        result._publish_response(lambda: queue.sync_q.put("old token"))
        controller.advance(REVISION)
        with pytest.raises(ExecutionAuthorityError):
            await getattr(result, consume)()
        result._handle_response.assert_not_called()

    asyncio.run(run())


def test_advance_wakes_a_blocked_sync_consumer(controller) -> None:
    waiting = Event()

    class ObservedQueue(Queue):
        def get(self, *args, **kwargs):
            waiting.set()
            return super().get(*args, **kwargs)

    result = result_shell(controller, ObservedQueue())
    with ThreadPoolExecutor(1) as pool:
        waiting_result = pool.submit(result._result_step, 5)
        assert waiting.wait(5)
        controller.advance(REVISION)
        with pytest.raises(ExecutionAuthorityError):
            waiting_result.result(5)
    result._handle_response.assert_not_called()


def test_advance_from_another_thread_wakes_async_consumer(controller) -> None:
    async def run():
        waiting = Event()

        class ObservedQueue(AsyncQueue):
            async def get(self, *args, **kwargs):
                waiting.set()
                return await super().get(*args, **kwargs)

        queue = ObservedQueue()
        result = result_shell(controller, queue.sync_q)
        result.aqueue = queue

        def advance():
            assert waiting.wait(5)
            controller.advance(REVISION)

        with ThreadPoolExecutor(1) as pool:
            advanced = pool.submit(advance)
            with pytest.raises(ExecutionAuthorityError):
                await asyncio.wait_for(result._aresult_step(), 5)
            advanced.result(5)
        result._handle_response.assert_not_called()

    asyncio.run(run())


def test_revision_during_postprocessing_rejects_delivery_without_deadlock(controller) -> None:
    result = result_shell(controller)
    result.queue.put("token")
    result._handle_response.side_effect = lambda _: controller.advance(REVISION)
    with pytest.raises(ExecutionAuthorityError):
        result._result_step()
    result._handle_response.assert_called_once_with("token")


def test_revoked_output_never_enters_consumer_queue(controller) -> None:
    result = result_shell(controller)
    controller.advance(REVISION)
    published = []
    assert not result._publish_response(lambda: published.append("stale"))
    assert published == []
    assert isinstance(result.queue.get(timeout=1), ExecutionAuthorityError)


def test_untagged_result_keeps_existing_delivery() -> None:
    result = object.__new__(GenerationResult)
    result._execution = None
    result._done = False
    result._terminal_error = None
    result.queue = Queue()
    result._handle_response = MagicMock()
    assert result._publish_response(lambda: result.queue.put("token"))
    result._result_step()
    result._handle_response.assert_called_once_with("token")


def test_request_output_copy_retains_gate(controller) -> None:
    request = GenerationRequest([1, 2], SamplingParams(max_tokens=1), execution_version=VERSION)
    request.set_id(42)
    result = GenerationResult(request)
    with controller.admission(VERSION) as execution:
        ExecutionControlMixin._bind_execution_result(result, execution)
    output = RequestOutput._from_generation_result(result)
    controller.advance(REVISION)
    with pytest.raises(ExecutionAuthorityError):
        output.result()


def worker_shell(controller):
    worker = MagicMock()
    worker.llm_args.max_beam_width = 1
    worker.llm_args.return_perf_metrics = False
    worker._is_pytorch_backend = True
    worker.max_seq_len = 128
    worker.mapping.cp_size = 1
    worker.engine.enqueue_request.return_value = 99
    worker._execution_admission = controller.admission
    worker._bind_execution_result = ExecutionControlMixin._bind_execution_result
    request = GenerationRequest([1, 2], SamplingParams(max_tokens=5), execution_version=VERSION)
    request.set_id(42)
    result = GenerationResult(request)
    worker._results = {42: result}
    return worker, request, result


def test_native_enqueue_is_gated_and_cancel_uses_native_id(controller) -> None:
    worker, request, result = worker_shell(controller)
    assert BaseWorker._enqueue_request(worker, request) == 99
    controller.advance(REVISION)
    worker.engine.cancel_request.assert_called_once_with(99)
    with pytest.raises(ExecutionAuthorityError):
        result.result()


def test_stale_native_submission_never_reaches_engine(controller) -> None:
    worker, request, _ = worker_shell(controller)
    controller.advance(REVISION)
    with pytest.raises(RequestError, match="not current"):
        BaseWorker._enqueue_request(worker, request)
    worker.engine.enqueue_request.assert_not_called()


def proxy_shell(controller):
    proxy = MagicMock()
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy._get_next_client_id.return_value = 42
    proxy._get_logprob_params.return_value = None
    proxy._results = {}
    proxy._execution_admission = controller.admission
    proxy._bind_execution_result = ExecutionControlMixin._bind_execution_result

    # GenerationResult stores a WeakMethod, so use an actual bound method.
    class ErrorHandler:
        def check(self):
            pass

    handler = ErrorHandler()
    proxy._handle_background_error = handler.check
    request = GenerationRequest([1, 2], SamplingParams(max_tokens=1), execution_version=VERSION)
    return proxy, request


def test_proxy_submit_checks_authority_at_request_queue(controller) -> None:
    proxy, request = proxy_shell(controller)
    result = GenerationExecutorProxy.submit(proxy, request)
    assert result._execution is not None
    proxy.request_queue.put_nowait.assert_called_once_with(request)
    controller.advance(REVISION)
    stale_request = GenerationRequest(
        [1, 2], SamplingParams(max_tokens=1), execution_version=VERSION
    )
    proxy._get_next_client_id.return_value = 43
    with pytest.raises(ExecutionAuthorityError):
        GenerationExecutorProxy.submit(proxy, stale_request)
    assert proxy.request_queue.put_nowait.call_count == 1


def test_full_ingress_fails_without_leaking_request_or_blocking_advance(controller) -> None:
    proxy, request = proxy_shell(controller)
    proxy.request_queue.put_nowait.side_effect = zmq.Again()
    with pytest.raises(RequestError, match="ingress is busy"):
        GenerationExecutorProxy.submit(proxy, request)
    assert proxy._results == {}
    controller.advance(REVISION).publication.wait()


def test_worker_submission_failure_removes_result_state(controller) -> None:
    worker, request, _ = worker_shell(controller)
    worker.start = lambda: None
    worker.rank = 0
    worker._get_logprob_params.return_value = None

    class ErrorHandler:
        def check(self):
            pass

    handler = ErrorHandler()
    worker._handle_background_error = handler.check
    worker._client_id_to_request_id = {}
    worker._pop_result = lambda client_id: BaseWorker._pop_result(worker, client_id)
    worker._enqueue_request = lambda req: BaseWorker._enqueue_request(worker, req)
    controller.advance(REVISION).publication.wait()
    with pytest.raises(RequestError, match="not current"):
        BaseWorker.submit(worker, request)
    assert worker._results == {}
    assert worker._client_id_to_request_id == {}
    worker.engine.enqueue_request.assert_not_called()
