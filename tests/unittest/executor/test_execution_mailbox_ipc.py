# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
from concurrent.futures import Future
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_mailbox import ExecutionController, PublicationReceipt, VersionUpdate
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.execution_control import (
    ExecutionControlMixin,
    ProxyExecutionControlMixin,
    WorkerExecutionControlMixin,
)
from tensorrt_llm.executor.execution_mailbox import IpcPublicationTransport
from tensorrt_llm.executor.ipc import IpcQueue
from tensorrt_llm.executor.rpc.rpc_common import RPCRequest, RPCResponse
from tensorrt_llm.executor.rpc.rpc_server import RPCServer

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 0)


class _Worker(WorkerExecutionControlMixin, ExecutionControlMixin):
    def __init__(self, world_size: int = 1, frontends: int = 1, pytorch: bool = True) -> None:
        self._init_execution_control()
        self.doing_shutdown = False
        self._is_pytorch_backend = pytorch
        self.llm_args = SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=world_size), num_serve_frontends=frontends
        )
        self.engine = object()


@pytest.mark.parametrize("world_size", [1, 2])
def test_rpc_method_discovery_does_not_initialize_execution_control(world_size: int) -> None:
    worker = _Worker(world_size=world_size)
    server = RPCServer(worker)
    try:
        assert "start_execution_mailbox" in server._functions
        assert worker._execution_controller is None
        assert worker._execution_version_table is None
        assert worker._execution_mailbox_server is None
    finally:
        server.shutdown()
        worker._shutdown_execution_control()


def test_real_rpc_bootstrap_opens_separate_publication_channel() -> None:
    import zmq

    worker = _Worker()
    server = RPCServer(worker)
    server.bind("tcp://127.0.0.1:*")
    server.start()
    bootstrap = IpcQueue((server.address, server.hmac_key), is_server=False, socket_type=zmq.DEALER)
    bootstrap.socket.setsockopt(zmq.LINGER, 0)
    controller = None
    try:
        bootstrap.put(
            RPCRequest(
                request_id="bootstrap",
                method_name="start_execution_mailbox",
                args=("session",),
                kwargs={},
                timeout=5,
            )
        )
        response = bootstrap.get(timeout=5)
        assert isinstance(response, RPCResponse)
        assert response.error is None
        assert response.request_id == "bootstrap"
        assert response.result[0] != server.address
        controller = ExecutionController("session", IpcPublicationTransport(response.result))
        assert controller.register(VERSION).wait().observed_version == VERSION
        assert worker._execution_controller is None
    finally:
        if controller is not None:
            controller.close()
        bootstrap.close()
        server.shutdown()
        worker._shutdown_execution_control()


def _worker_process(connection, release) -> None:
    worker = _Worker()
    try:
        address = worker.start_execution_mailbox("session")
        connection.send(address)
        # This thread models an occupied execution loop. Only the dedicated
        # mailbox listener can process publications until it is released.
        if not release.wait(timeout=20):
            raise TimeoutError("test did not release worker")
        connection.send(worker.get_execution_version_table().current_version("tenant", "scope"))
    finally:
        worker._shutdown_execution_control()
        connection.close()


def test_publication_reaches_busy_spawned_worker_and_preserves_uint64_epoch(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_LLMAPI_ZMQ_DEBUG", "1")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    release = context.Event()
    process = context.Process(target=_worker_process, args=(child, release))
    process.start()
    child.close()
    controller = None
    try:
        # Native package imports in a spawned interpreter may take tens of seconds.
        assert parent.poll(60), "worker did not start"
        address = parent.recv()
        controller = ExecutionController("session", IpcPublicationTransport(address, timeout=2))
        original = replace(VERSION, epoch=(1 << 64) - 2)
        assert controller.register(original).wait().observed_version == original
        successor = replace(original, epoch=(1 << 64) - 1)
        update = controller.advance(Revision("edit", original, successor))
        assert update.publication.wait().observed_version == successor
        assert not release.is_set()
        release.set()
        assert parent.poll(5)
        assert parent.recv() == successor
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        release.set()
        if controller is not None:
            controller.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        parent.close()


def test_ipc_duplicates_and_reordered_updates_do_not_rewind_worker(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_LLMAPI_ZMQ_DEBUG", "1")
    worker = _Worker()
    transport = IpcPublicationTransport(worker.start_execution_mailbox("session"), timeout=1)
    try:
        newer = VersionUpdate("session", 2, replace(VERSION, epoch=1))
        older = VersionUpdate("session", 1, VERSION)
        assert transport.publish(newer, Event()).observed_version == newer.version
        assert transport.publish(older, Event()).observed_version == newer.version
        assert transport.publish(newer, Event()).observed_version == newer.version
        with pytest.raises(RuntimeError, match="authority session"):
            transport.publish(replace(newer, session_id="different"), Event())
        assert (
            worker.get_execution_version_table().current_version("tenant", "scope") == newer.version
        )
    finally:
        transport.close()
        worker._shutdown_execution_control()


def test_worker_shutdown_leaves_accepted_revision_visible_but_publication_fails() -> None:
    worker = _Worker()
    address = worker.start_execution_mailbox("session")
    controller = ExecutionController("session", IpcPublicationTransport(address, timeout=0.1))
    try:
        controller.register(VERSION).wait()
        worker._shutdown_execution_control()
        revised = controller.advance(Revision("edit", VERSION, replace(VERSION, epoch=1)))
        with pytest.raises(TimeoutError):
            revised.publication.wait()
        assert controller.snapshot("tenant", "scope").current_version.epoch == 1
    finally:
        controller.close()
        worker._shutdown_execution_control()


def test_abrupt_worker_death_fails_publication_without_resetting_authority() -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    release = context.Event()
    process = context.Process(target=_worker_process, args=(child, release))
    process.start()
    child.close()
    controller = None
    try:
        assert parent.poll(60), "worker did not start"
        controller = ExecutionController(
            "session", IpcPublicationTransport(parent.recv(), timeout=0.5)
        )
        controller.register(VERSION).wait()
        process.terminate()
        process.join(timeout=5)
        assert not process.is_alive()
        accepted = controller.advance(Revision("edit", VERSION, replace(VERSION, epoch=1)))
        with pytest.raises(TimeoutError):
            accepted.publication.wait()
        assert controller.snapshot("tenant", "scope").current_version.epoch == 1
    finally:
        # A killed waiter can leave multiprocessing.Event's semaphore unusable.
        # Teardown this child directly instead of signalling that event again.
        if controller is not None:
            controller.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent.close()


def test_unrelated_acknowledgements_are_ignored_before_matching_reply() -> None:
    import zmq

    address = Future()
    finished = Future()

    def reply() -> None:
        queue = None
        try:
            queue = IpcQueue(is_server=True)
            queue.socket.setsockopt(zmq.LINGER, 0)
            address.set_result(queue.address)
            update = queue.get(timeout=5)
            queue.put(PublicationReceipt(replace(update, session_id="old-worker"), update.version))
            queue.put(PublicationReceipt(replace(update, sequence=1), update.version))
            queue.put(PublicationReceipt(update, update.version))
            # Keep the socket alive until the consumer has received its reply.
            assert queue.get(timeout=5) == "received"
            finished.set_result(None)
        except Exception as error:
            if not address.done():
                address.set_exception(error)
            finished.set_exception(error)
        finally:
            if queue is not None:
                queue.close()

    thread = Thread(target=reply, daemon=True)
    thread.start()
    transport = IpcPublicationTransport(address.result(timeout=5), timeout=2)
    try:
        update = VersionUpdate("session", 2, VERSION)
        assert transport.publish(update, Event()) == PublicationReceipt(update, VERSION)
        transport._queue.put("received")
        finished.result(timeout=5)
    finally:
        transport.close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_controller_shutdown_interrupts_ipc_wait_without_waiting_for_transport_timeout() -> None:
    worker = _Worker()
    address = worker.start_execution_mailbox("session")
    worker._shutdown_execution_control()
    controller = ExecutionController("session", IpcPublicationTransport(address, timeout=60))
    publication = controller.register(VERSION)
    controller.close()
    with pytest.raises(RuntimeError, match="closed"):
        publication.wait()
    assert not controller._thread.is_alive()


def test_direct_executor_is_lazy_and_owns_exactly_one_controller() -> None:
    worker = _Worker()
    try:
        assert worker._execution_controller is None
        assert worker._execution_mailbox_server is None
        controller = worker.get_execution_controller()
        assert worker.get_execution_controller() is controller
        assert controller.register(VERSION).wait().observed_version == VERSION
        assert worker.get_execution_version_table().current_version("tenant", "scope") == VERSION
        assert worker._execution_mailbox_server is None
    finally:
        worker._shutdown_execution_control()
    with pytest.raises(RuntimeError, match="closed"):
        worker.get_execution_controller()
    assert not controller._thread.is_alive()


def test_one_worker_cannot_have_two_authority_owners() -> None:
    worker = _Worker()
    try:
        address = worker.start_execution_mailbox("session")
        assert worker.start_execution_mailbox("session") == address
        with pytest.raises(RuntimeError, match="different authority owner"):
            worker.start_execution_mailbox("another-session")
        with pytest.raises(RuntimeError, match="different authority owner"):
            worker.get_execution_controller()
    finally:
        worker._shutdown_execution_control()
    with pytest.raises(RuntimeError, match="closed"):
        worker.start_execution_mailbox("session")


@pytest.mark.parametrize("kwargs", [{"world_size": 2}, {"frontends": 2}, {"pytorch": False}])
def test_unsupported_workers_reject_control_before_creating_any_authority(kwargs) -> None:
    worker = _Worker(**kwargs)
    try:
        with pytest.raises(NotImplementedError):
            worker.get_execution_controller()
        with pytest.raises(NotImplementedError):
            worker.start_execution_mailbox("session")
        assert worker._execution_controller is None
        assert worker._execution_version_table is None
        assert worker._execution_mailbox_server is None
    finally:
        worker._shutdown_execution_control()


class _BootstrapCall:
    def __init__(self, worker: _Worker, session_id: str) -> None:
        self.worker = worker
        self.session_id = session_id

    def remote(self, timeout: float):
        assert timeout == 5
        return self.worker.start_execution_mailbox(self.session_id)


class _BootstrapClient:
    def __init__(self, worker: _Worker) -> None:
        self.worker = worker
        self.sessions = []

    def start_execution_mailbox(self, session_id: str) -> _BootstrapCall:
        self.sessions.append(session_id)
        return _BootstrapCall(self.worker, session_id)


class _Proxy(ProxyExecutionControlMixin, ExecutionControlMixin):
    def __init__(self, worker: _Worker) -> None:
        self._init_execution_control()
        self.doing_shutdown = False
        self._is_pytorch_backend = True
        self.model_world_size = 1
        self._num_frontends = 1
        self.workers_started = True
        self._engine_dead = False
        self.rpc_client = _BootstrapClient(worker)


def test_proxy_bootstraps_once_then_publications_use_dedicated_ipc(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_LLMAPI_ZMQ_DEBUG", "1")
    worker = _Worker()
    proxy = _Proxy(worker)
    try:
        controller = proxy.get_execution_controller()
        assert proxy.get_execution_controller() is controller
        assert controller.register(VERSION).wait().observed_version == VERSION
        result = controller.advance(Revision("edit", VERSION, replace(VERSION, epoch=1)))
        assert result.publication.wait().observed_version.epoch == 1
        assert proxy.rpc_client.sessions == [proxy._execution_session_id]
        assert worker._execution_controller is None
    finally:
        proxy._shutdown_execution_control()
        worker._shutdown_execution_control()


@pytest.mark.parametrize(
    "attribute, value",
    [("_num_frontends", 2), ("model_world_size", 2), ("_is_pytorch_backend", False)],
)
def test_proxy_rejects_unsupported_topology_without_bootstrap(attribute: str, value) -> None:
    worker = _Worker()
    proxy = _Proxy(worker)
    try:
        setattr(proxy, attribute, value)
        with pytest.raises(NotImplementedError):
            proxy.get_execution_controller()
        assert proxy.rpc_client.sessions == []
    finally:
        proxy._shutdown_execution_control()
        worker._shutdown_execution_control()


def test_unsupported_executor_has_no_silent_local_authority() -> None:
    class Unsupported(ExecutionControlMixin):
        doing_shutdown = False

    executor = Unsupported()
    executor._init_execution_control()
    try:
        with pytest.raises(NotImplementedError):
            executor.get_execution_controller()
        assert executor._execution_controller is None
    finally:
        executor._shutdown_execution_control()
