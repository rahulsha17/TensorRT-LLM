# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import os
import threading
from concurrent.futures import Future
from typing import Callable, List, Optional

from .._utils import nvtx_range_debug
from ..execution_collective import CollectivePublicationTransport, ExecutionParticipant
from ..execution_mailbox import ExecutionController
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import _SyncQueue
from ..logger import logger
from .execution_mailbox import IpcPublicationTransport
from .postproc_worker import PostprocWorker
from .request import GenerationRequest
from .result import GenerationResult
from .rpc import RPCClient
from .rpc.rpc_common import get_unique_ipc_addr
from .utils import ErrorResponse, is_llm_response


class RpcExecutorMixin:
    """Mixin for executors that use RPC client for hot path communication.

    Provides:
    - RPC client initialization
    - Response handling loop
    - Main loop thread management
    - Shutdown logic for RPC components

    The inheriting class should call init_rpc_executor() to set up RPC client.
    """

    def init_rpc_executor(self):
        self.rpc_addr = get_unique_ipc_addr()
        self.hmac_key = os.urandom(32)
        self.rpc_client = RPCClient(self.rpc_addr, hmac_key=self.hmac_key)

        self._results = {}
        self._shutdown_event = threading.Event()
        self.main_loop_task_obj = None
        self.main_loop = None
        self.main_loop_thread = None

    def _create_execution_controller(self) -> ExecutionController:
        args = self.worker_kwargs["llm_args"]
        if (
            args.backend != "pytorch"
            or args.parallel_config.world_size != 1
            or getattr(self, "model_world_size", 1) != 1
            or getattr(self, "world_size", 1) != 1
            or args.num_serve_frontends != 1
        ):
            raise NotImplementedError(
                "execution publication requires one frontend and one PyTorch rank"
            )
        if self._shutdown_event.is_set() or self.rpc_client is None:
            raise RuntimeError("execution publication requires a running worker")
        address = self.rpc_client.start_execution_mailbox(self._execution_session_id).remote(
            timeout=5
        )
        participant = ExecutionParticipant(0, 0)
        transport = CollectivePublicationTransport(
            (participant,), {participant: IpcPublicationTransport(address)}
        )
        return ExecutionController(self._execution_session_id, transport)

    def setup_mainloop(
        self, tasks: Optional[List[Callable]] = None, thread_name: str = "rpc_proxy_main_loop"
    ):
        """Setup main loop thread with custom async tasks.

        Args:
            tasks: List of async coroutine functions to run.
            thread_name: Name for the main loop thread

        Note: Stats and kv_events are now fetched on-demand via direct RPC calls
        (get_stats, aget_stats, get_kv_events, aget_kv_events), so the default
        tasks only include the responses loop. Callers can still provide custom
        tasks including stats/kv_events loops if needed for streaming use cases.
        """
        if tasks is None:
            tasks = [
                self._fetch_responses_loop_async,
            ]

        async def main_loop_task():
            await asyncio.gather(*[task() for task in tasks])

        def _run_main_loop_task():
            """Local method to run the main loop task."""
            self.main_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.main_loop)

            self.main_loop_task_obj = self.main_loop.create_task(main_loop_task())
            try:
                self.main_loop.run_until_complete(self.main_loop_task_obj)
            except asyncio.CancelledError:
                pass  # Task cancellation is expected during shutdown
            finally:
                self.main_loop.close()

        self.main_loop_thread = threading.Thread(
            target=_run_main_loop_task, daemon=True, name=thread_name
        )
        self.main_loop_thread.start()
        atexit.register(self.shutdown)

    def submit(self, request: GenerationRequest) -> GenerationResult:
        request.set_id(self._get_next_client_id())
        return self._submit_rpc_generation(request)

    def _submit_rpc_generation(self, request: GenerationRequest) -> GenerationResult:
        """Bind authority and result routing before asynchronous ingress can reply."""
        logprob_params = self._get_logprob_params(request)
        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params,
        )
        sent = None
        with self._execution_admission(request.execution_version) as execution:
            self._bind_execution_result(result, execution)
            self._results[request.id] = result
            try:
                with nvtx_range_debug("RPCExecutor.submit", color="green", category="Proxy"):
                    if execution is None:
                        self.rpc_client.submit(request).remote(need_response=False)
                    else:
                        # Queue transport work without waiting under authority.
                        # A delayed send is rechecked by worker admission.
                        sent = self.rpc_client.submit(request).remote_future(need_response=False)
            except Exception:
                self._results.pop(request.id, None)
                raise
        if sent is not None:

            def ingress_done(future: Future) -> None:
                error = "RPC submission was cancelled" if future.cancelled() else future.exception()
                if error is not None and self._results.get(request.id) is result:
                    self.handle_responses([ErrorResponse(request.id, str(error), -1)])

            # A completed future may invoke its callback immediately. Register
            # outside admission because terminal handling also takes authority.
            sent.add_done_callback(ingress_done)
        return result

    def handle_responses(self, responses: list[GenerationResult]) -> bool:
        async_queues = []
        event_loop = None

        def process_res(res: list):
            for r in res:
                client_id = r.client_id
                nonlocal event_loop
                nonlocal async_queues

                result = self._results.get(client_id)
                if result is None:
                    logger.warning(f"Received response for unknown client_id: {client_id}")
                    continue

                queue = result.queue
                if isinstance(queue, _SyncQueue):
                    if result._publish_response(lambda: queue.put_nowait(r)):
                        async_queues.append(queue)
                        # all the loops are identical
                        event_loop = event_loop or queue.loop
                else:
                    result._publish_response(lambda: queue.put(r))

                # PostprocWorker.Output records arrive on the same stream when
                # postprocess parallelism is enabled (worker-side pool); they
                # carry their own is_final, mirroring the classic proxy's
                # dispatch_result_task handling.
                if (
                    (is_llm_response(r) and r.result.is_final)
                    or isinstance(r, ErrorResponse)
                    or (isinstance(r, PostprocWorker.Output) and r.is_final)
                ):
                    self._results.pop(client_id, None)
                    if result._execution is not None:
                        result._execution.finish()

        # Handle the case where responses might not be a list of lists
        if responses and not isinstance(responses[0], list):
            # If responses is a flat list, wrap it
            responses = [responses]

        for res in responses:
            global_tracer().log_instant("RPC.get")
            process_res(res)

        if async_queues:
            _SyncQueue.notify_many(event_loop, async_queues)

    async def _generic_fetch_loop_async(
        self, fetch_method_name: str, handler_method: Callable, method_name: str
    ):
        """Generic method for fetching data in a loop from RPC worker.

        Args:
            fetch_method_name: Name of the RPC client method to call
            handler_method: The handler method to call with the fetched data
            method_name: Name of the method for logging
        """
        try:
            fetch_method = getattr(self.rpc_client, fetch_method_name)
            async for data in fetch_method().remote_streaming():
                if self._shutdown_event.is_set():
                    return
                handler_method(data)
        except asyncio.CancelledError:
            logger.debug(f"{method_name} task cancelled")
        except Exception as e:
            logger.error(f"Error in {method_name}: {e}")
            raise

    async def _fetch_responses_loop_async(self):
        await self._generic_fetch_loop_async(
            fetch_method_name="fetch_responses_loop_async",
            handler_method=self.handle_responses,
            method_name="_fetch_responses_loop_async",
        )

    # NOTE: _fetch_stats_loop_async and _fetch_kv_cache_events_loop_async have been removed.
    # Stats and kv_events are now fetched on-demand via direct RPC calls
    # (get_stats, aget_stats, get_kv_events, aget_kv_events) instead of streaming loops.
