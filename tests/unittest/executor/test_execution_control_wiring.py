# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise control wiring on real executor classes without loading a model."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.proxy import GenerationExecutorFrontendProxy, GenerationExecutorProxy
from tensorrt_llm.executor.worker import GenerationExecutorWorker

pytestmark = pytest.mark.cpu_only


def test_executor_constructor_initializes_lazy_control() -> None:
    class Executor(GenerationExecutor):
        def submit(self, request):
            raise NotImplementedError

        def abort_request(self, request_id: int) -> None:
            raise NotImplementedError

        def shutdown(self) -> None:
            self._shutdown_execution_control()

    executor = Executor()
    try:
        assert executor._execution_controller is None
        with pytest.raises(NotImplementedError):
            executor.get_execution_controller()
    finally:
        executor.shutdown()


def test_direct_worker_shutdown_closes_publication_before_engine(monkeypatch) -> None:
    import torch

    worker = object.__new__(GenerationExecutorWorker)
    worker._init_execution_control()
    worker.doing_shutdown = False
    worker._is_pytorch_backend = True
    worker.llm_args = SimpleNamespace(
        backend="pytorch", num_serve_frontends=1, parallel_config=SimpleNamespace(world_size=1)
    )

    def engine_shutdown() -> None:
        with pytest.raises(RuntimeError, match="closed"):
            worker.get_execution_controller()

    configure = MagicMock()
    worker.engine = SimpleNamespace(
        can_enqueue_requests=lambda: False,
        shutdown=engine_shutdown,
        configure_execution_authority=configure,
    )
    worker._handle_background_error = MagicMock()
    monkeypatch.setattr("tensorrt_llm.executor.worker.mpi_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    controller = worker.get_execution_controller()
    configure.assert_called_once_with(worker.get_execution_version_table())
    try:
        version = ExecutionVersion("tenant", "scope", 0)
        assert controller.register(version).wait().observed_version == version
        assert worker.get_execution_version_table().current_version("tenant", "scope") == version
        worker.shutdown()
        assert not controller._thread.is_alive()
        assert worker.engine is None
    finally:
        worker._shutdown_execution_control()


def test_proxy_pre_shutdown_closes_publication_before_worker_shutdown_message() -> None:
    proxy = object.__new__(GenerationExecutorProxy)
    proxy._init_execution_control()
    proxy.workers_started = True
    proxy.doing_shutdown = False
    proxy._worker_process_monitor = MagicMock()
    proxy._abort_all_requests = MagicMock()
    proxy.mpi_futures = []

    def stop_worker(*args, **kwargs) -> None:
        assert proxy._execution_control_closed

    proxy.request_queue = SimpleNamespace(put_noblock=stop_worker)
    proxy.pre_shutdown()
    assert proxy._execution_control_closed
    with pytest.raises(RuntimeError, match="closed"):
        proxy.get_execution_controller()


def test_attached_frontend_rejects_publication_without_worker_attributes() -> None:
    proxy = object.__new__(GenerationExecutorFrontendProxy)
    proxy._init_execution_control()
    proxy.doing_shutdown = False
    proxy._num_frontends = 2
    with pytest.raises(NotImplementedError, match="one frontend"):
        proxy.get_execution_controller()
    assert proxy._execution_controller is None
    proxy._shutdown_execution_control()
