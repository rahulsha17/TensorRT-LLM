# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RPC frontend authority and result routing without inference or a Ray cluster."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from queue import Queue
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import WorkerVersionTable
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.execution_control import ExecutionControlMixin
from tensorrt_llm.executor.execution_mailbox import ExecutionMailboxServer
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.executor.rpc import RPCClient, RPCServer
from tensorrt_llm.executor.rpc.rpc_common import get_unique_ipc_addr
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin
from tensorrt_llm.executor.utils import EngineDeadError, ErrorResponse

pytestmark = pytest.mark.cpu_only
OLD = ExecutionVersion("tenant", "scope", 7)
NEW = replace(OLD, epoch=8)
REVISION = Revision("edit", OLD, NEW)


class Frontend(RpcExecutorMixin, ExecutionControlMixin):
    def __init__(self):
        self._init_execution_control()
        self.doing_shutdown = False
        self._shutdown_event = Event()
        self._results = {}
        self.rpc_client = Mock()
        self.worker_kwargs = {
            "llm_args": SimpleNamespace(
                backend="pytorch",
                parallel_config=SimpleNamespace(world_size=1),
                num_serve_frontends=1,
            )
        }
        self._get_logprob_params = Mock(return_value=None)
        self._handle_background_error = Mock()
        self._set_fatal_error = Mock()
        self._get_next_client_id = Mock(return_value=41)

    def close(self):
        self._shutdown_execution_control()


@dataclass
class _Request:
    execution_version: ExecutionVersion | None
    id: int | None = None
    disaggregated_params: None = None

    def set_id(self, identity):
        self.id = identity


def request(version=OLD):
    return _Request(version)


@pytest.fixture
def frontend(monkeypatch):
    def result_factory(req, **kwargs):
        result = object.__new__(GenerationResult)
        result._generation_request = req
        result._execution = None
        result._done = False
        result._terminal_error = None
        result.queue = Queue()
        result.aqueue = None
        result._handle_response = Mock()
        return result

    monkeypatch.setattr("tensorrt_llm.executor.rpc_proxy_mixin.GenerationResult", result_factory)
    owner = Frontend()
    table = WorkerVersionTable(owner._execution_session_id)
    server = ExecutionMailboxServer(table)
    owner.rpc_client.start_execution_mailbox.return_value.remote.return_value = server.address
    owner.rpc_client.submit_for_rpc.return_value.remote_future.return_value = Future()
    yield owner, table
    owner._shutdown_execution_control()
    server.close()


@contextmanager
def rpc_worker(submit):
    class Worker(RpcWorkerMixin):
        def submit(self, req):
            return submit(req)

    server = RPCServer(Worker())
    server.bind(get_unique_ipc_addr())
    server.start()
    try:
        with RPCClient(server.address, hmac_key=server.hmac_key) as client:
            acknowledgements = []
            original = client._call_future

            def capture_ack(*args, **kwargs):
                assert kwargs["__rpc_params"].need_response
                future = original(*args, **kwargs)
                acknowledgements.append(future)
                return future

            client._call_future = capture_ack
            yield client, acknowledgements
    finally:
        server.shutdown()


@pytest.mark.parametrize("version", [None, OLD])
def test_worker_admission_error_reaches_frontend_over_real_rpc(frontend, version):
    with closing(frontend[0]):
        owner, _ = frontend
        owner.get_execution_controller().register(OLD).wait()

        def reject(req):
            assert req.id == 41 and req.execution_version == version
            raise ValueError("worker admission rejected")

        with rpc_worker(reject) as (client, acknowledgements):
            owner.rpc_client = client
            result = owner.submit(request(version))
            error = result.queue.get(timeout=5)
            assert isinstance(error, ErrorResponse)
            assert "worker admission rejected" in error.error_msg
            assert "Worker traceback:" in error.error_msg
            assert acknowledgements[0].done()
            # Queue delivery precedes terminal cleanup under this lock.
            with owner._execution_control_lock:
                assert owner._results == {}
                if result._execution is not None:
                    assert result._execution._wake is None


@pytest.mark.parametrize("version", [None, OLD])
def test_admission_ack_is_serializable_and_does_not_complete_request(frontend, version):
    with closing(frontend[0]):
        owner, _ = frontend
        owner.get_execution_controller().register(OLD).wait()
        worker_handle = Lock()  # A worker-local result must never be sent over RPC.
        with rpc_worker(lambda req: worker_handle) as (client, acknowledgements):
            owner.rpc_client = client
            result = owner.submit(request(version))
            assert acknowledgements[0].result(5) is None
            assert result.queue.empty()
            assert owner._results[41] is result
            if result._execution is not None:
                assert result._execution._wake is not None
            owner.handle_responses([ErrorResponse(41, "ordinary terminal response", 9)])
            assert result.queue.get(timeout=5).error_msg == "ordinary terminal response"


def test_delayed_real_rpc_admission_does_not_block_revision(frontend):
    with closing(frontend[0]):
        owner, table = frontend
        control = owner.get_execution_controller()
        control.register(OLD).wait()
        entered, release = Event(), Event()

        def admit(req):
            entered.set()
            assert release.wait(5)
            with table.admission(req.execution_version):
                raise AssertionError("stale request must not be admitted")

        with rpc_worker(admit) as (client, acknowledgements):
            owner.rpc_client = client
            try:
                result = owner.submit(request())
                assert entered.wait(5)
                control.advance(REVISION).publication.wait()
                assert not acknowledgements[0].done()
                assert isinstance(result.queue.get(timeout=5), ExecutionAuthorityError)
            finally:
                release.set()
            finished = Event()
            acknowledgements[0].add_done_callback(lambda _: finished.set())
            assert finished.wait(5)
            assert acknowledgements[0].exception() is not None
            assert owner._results == {}
            assert result.queue.empty()


@pytest.mark.parametrize("version", [None, OLD])
def test_late_admission_error_does_not_reach_replacement_result(frontend, version):
    with closing(frontend[0]):
        owner, _ = frontend
        owner.get_execution_controller().register(OLD).wait()
        old_result = owner.submit(request(version))
        old_ack = owner.rpc_client.submit_for_rpc.return_value.remote_future.return_value
        owner.handle_responses([ErrorResponse(41, "request ended", 9)])
        assert old_result.queue.get_nowait().error_msg == "request ended"

        owner.rpc_client.submit_for_rpc.return_value.remote_future.return_value = Future()
        replacement = owner.submit(request(version))
        old_ack.set_exception(ValueError("late admission error"))
        assert old_result.queue.empty()
        assert replacement.queue.empty()
        assert owner._results[41] is replacement
        if replacement._execution is not None:
            assert replacement._execution._wake is not None


def test_rpc_bootstrap_is_lazy_and_uses_dedicated_authenticated_mailbox(frontend):
    with closing(frontend[0]):
        owner, table = frontend
        owner.rpc_client.start_execution_mailbox.assert_not_called()
        control = owner.get_execution_controller()
        assert control.register(OLD).wait().observed_version == OLD
        assert table.current_version("tenant", "scope") == OLD
        assert len(control._transport.participants) == 1
        assert owner.get_execution_controller() is control
        owner.rpc_client.start_execution_mailbox.assert_called_once_with(
            owner._execution_session_id
        )
        owner.rpc_client.start_execution_mailbox.return_value.remote.assert_called_once_with(
            timeout=5
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "other"),
        ("ranks", 2),
        ("frontends", 2),
        ("model_world_size", 2),
        ("world_size", 2),
    ],
)
def test_unsupported_topology_fails_before_bootstrap(frontend, field, value):
    owner, _ = frontend
    args = owner.worker_kwargs["llm_args"]
    if field == "ranks":
        args.parallel_config.world_size = value
    elif field == "frontends":
        args.num_serve_frontends = value
    elif field in ("model_world_size", "world_size"):
        setattr(owner, field, value)
    else:
        args.backend = value
    with pytest.raises(NotImplementedError, match="one frontend and one PyTorch rank"):
        owner.get_execution_controller()
    owner.rpc_client.start_execution_mailbox.assert_not_called()


def test_missing_authority_rejects_tagged_rpc_before_sending(frontend):
    owner, _ = frontend
    with pytest.raises(ExecutionAuthorityError, match="register"):
        owner.submit(request())
    owner.rpc_client.submit_for_rpc.assert_not_called()
    assert owner._results == {}


def test_pending_rpc_send_does_not_hold_authority_or_prevent_revoke(frontend):
    with closing(frontend[0]):
        owner, _ = frontend
        control = owner.get_execution_controller()
        control.register(OLD).wait()
        result = owner.submit(request())
        sent = owner.rpc_client.submit_for_rpc.return_value.remote_future.return_value
        assert not sent.done()
        owner.rpc_client.submit_for_rpc.return_value.remote.assert_not_called()
        control.advance(REVISION).publication.wait()
        with pytest.raises(ExecutionAuthorityError):
            result._result_step(0.1)
        assert not sent.done()
        sent.set_result(None)


def test_immediate_async_send_failure_runs_callback_outside_authority(frontend):
    with closing(frontend[0]):
        owner, _ = frontend
        control = owner.get_execution_controller()
        control.register(OLD).wait()
        failed = Future()
        failed.set_exception(OSError("send failed"))
        owner.rpc_client.submit_for_rpc.return_value.remote_future.return_value = failed
        handle = owner.handle_responses

        def check_unlocked(responses):
            assert not control._lock.locked()
            handle(responses)

        owner.handle_responses = check_unlocked
        result = owner.submit(request())
        error = result.queue.get_nowait()
        assert isinstance(error, ErrorResponse) and "send failed" in error.error_msg
        assert owner._results == {}
        assert result._execution._wake is None


def test_cancelled_transport_future_reports_terminal_failure(frontend):
    with closing(frontend[0]):
        owner, _ = frontend
        owner.get_execution_controller().register(OLD).wait()
        result = owner.submit(request())
        owner.rpc_client.submit_for_rpc.return_value.remote_future.return_value.cancel()
        error = result.queue.get_nowait()
        assert isinstance(error, ErrorResponse) and "cancelled" in error.error_msg
        assert owner._results == {}


def test_early_worker_reply_finds_result_registered_before_send(frontend):
    owner, _ = frontend

    def immediate_reply(**kwargs):
        assert 41 in owner._results
        owner.handle_responses([ErrorResponse(41, "worker reply", 9)])

    owner.rpc_client.submit_for_rpc.return_value.remote_future.side_effect = immediate_reply
    result = owner.submit(request(None))
    assert result.queue.get_nowait().error_msg == "worker reply"
    assert result._execution is None and owner._results == {}


def test_synchronous_send_rejection_removes_result_and_host_hold(frontend):
    with closing(frontend[0]):
        owner, _ = frontend
        control = owner.get_execution_controller()
        control.register(OLD).wait()
        owner.rpc_client.submit_for_rpc.return_value.remote_future.side_effect = RuntimeError(
            "send rejected"
        )
        with pytest.raises(RuntimeError, match="send rejected"):
            owner.submit(request())
        assert owner._results == {}
        assert not list(control._gate._executions)


def test_stale_terminal_response_is_dropped_and_detaches_notifications(frontend):
    with closing(frontend[0]):
        owner, _ = frontend
        control = owner.get_execution_controller()
        control.register(OLD).wait()
        result = owner.submit(request())
        control.advance(REVISION).publication.wait()
        assert isinstance(result.queue.get_nowait(), ExecutionAuthorityError)
        owner.handle_responses([ErrorResponse(41, "late worker reply", 9)])
        assert result.queue.empty() and owner._results == {}
        assert result._execution._wake is None
        with pytest.raises(ExecutionAuthorityError):
            result._check_execution_authority()


def test_unrelated_revision_preserves_result_delivery(frontend):
    with closing(frontend[0]):
        owner, _ = frontend
        control = owner.get_execution_controller()
        control.register(OLD).wait()
        result = owner.submit(request())
        other = replace(OLD, scope_id="other")
        control.register(other).wait()
        control.advance(Revision("other-edit", other, replace(other, epoch=8))).publication.wait()
        owner.handle_responses([ErrorResponse(41, "current reply", 9)])
        assert result.queue.get_nowait().error_msg == "current reply"


def test_shutdown_revokes_pending_rpc_result(frontend):
    owner, _ = frontend
    owner.get_execution_controller().register(OLD).wait()
    result = owner.submit(request())
    owner._shutdown_execution_control()
    assert isinstance(result.queue.get_nowait(), ExecutionAuthorityError)
    with pytest.raises(RuntimeError, match="closed"):
        owner.get_execution_controller()


@pytest.mark.parametrize("version", [None, OLD])
@pytest.mark.parametrize("failure", ["exception", "eof"])
def test_response_stream_loss_wakes_consumer_and_closes_admission(frontend, version, failure):
    owner, _ = frontend
    owner.get_execution_controller().register(OLD).wait()
    result = owner.submit(request(version))
    entered = Event()

    def consume():
        entered.set()
        with pytest.raises((EngineDeadError, ExecutionAuthorityError)):
            result._result_step(2)

    async def stream():
        if failure == "exception":
            raise OSError("producer gone")
        if False:
            yield None

    owner.rpc_client.fetch_responses_loop_async.return_value.remote_streaming.side_effect = stream
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(consume)
        assert entered.wait(1)
        asyncio.run(owner._fetch_responses_loop_async())
        waiting.result(3)
    assert owner._execution_control_closed
    owner._set_fatal_error.assert_called_once()
    assert owner._results == {}
    for next_version in (None, OLD):
        with pytest.raises(RuntimeError, match="restart required"):
            owner.submit(request(next_version))
    with pytest.raises(RuntimeError, match="closed"):
        owner._execution_controller.register(replace(OLD, scope_id="late"))


def test_rpc_failure_continues_after_controller_close_raises(frontend, monkeypatch):
    owner, _ = frontend
    result = owner.submit(request(None))
    original = OSError("worker died")
    with monkeypatch.context() as patch:
        patch.setattr(
            owner, "_shutdown_execution_control", Mock(side_effect=RuntimeError("close stuck"))
        )
        assert owner._fail_rpc_executor(original)
        assert not owner._fail_rpc_executor(RuntimeError("second error"))
    owner._set_fatal_error.assert_called_once_with(original)
    with pytest.raises(EngineDeadError):
        result._result_step(0.1)
    with pytest.raises(RuntimeError, match="restart required"):
        owner.submit(request(None))


def test_failure_serializes_with_submission_and_drops_late_response(frontend):
    owner, _ = frontend
    sending = Event()
    release = Event()

    def send(**kwargs):
        sending.set()
        assert release.wait(2)

    owner.rpc_client.submit_for_rpc.return_value.remote_future.side_effect = send
    with ThreadPoolExecutor(2) as pool:
        submitting = pool.submit(owner.submit, request(None))
        assert sending.wait(1)
        failing = pool.submit(owner._fail_rpc_executor, OSError("worker died"))
        release.set()
        result = submitting.result(2)
        assert failing.result(2)
    owner.handle_responses([ErrorResponse(41, "late reply", 9)])
    with pytest.raises(EngineDeadError):
        result._result_step(0.1)
    assert result.queue.empty()


def test_normal_stream_shutdown_does_not_mark_executor_failed(frontend):
    owner, _ = frontend
    owner._shutdown_event.set()

    async def stream():
        if False:
            yield None

    owner.rpc_client.fetch_responses_loop_async.return_value.remote_streaming.side_effect = stream
    asyncio.run(owner._fetch_responses_loop_async())
    owner._set_fatal_error.assert_not_called()


@pytest.mark.parametrize("version", [None, OLD])
def test_pending_transport_send_cannot_block_frontend_failure(frontend, version):
    owner, _ = frontend
    owner.get_execution_controller().register(OLD).wait()
    result = owner.submit(request(version))
    sent = owner.rpc_client.submit_for_rpc.return_value.remote_future.return_value
    assert not sent.done()
    owner.rpc_client.submit_for_rpc.return_value.remote.assert_not_called()
    assert owner._fail_rpc_executor(OSError("worker died during send"))
    with pytest.raises((EngineDeadError, ExecutionAuthorityError)):
        result._result_step(0.1)
    assert not sent.done()
    sent.set_exception(OSError("late send error"))
    assert not owner._results


@pytest.mark.parametrize("revoked", [False, True])
def test_async_consumer_is_not_notified_for_dropped_rpc_response(frontend, monkeypatch, revoked):
    with closing(frontend[0]):
        owner, _ = frontend
        control = owner.get_execution_controller()
        control.register(OLD).wait()
        result = owner.submit(request())

        class AsyncQueueBridge:
            notify_many = Mock()

            def __init__(self):
                self.loop = object()
                self.put_nowait = Mock()

        monkeypatch.setattr("tensorrt_llm.executor.rpc_proxy_mixin._SyncQueue", AsyncQueueBridge)
        bridge = AsyncQueueBridge()
        result.queue = bridge
        if revoked:
            control.advance(REVISION).publication.wait()
        response = ErrorResponse(41, "worker reply", 9)
        owner.handle_responses([response])
        if revoked:
            bridge.put_nowait.assert_not_called()
            bridge.notify_many.assert_not_called()
        else:
            bridge.put_nowait.assert_called_once_with(response)
            bridge.notify_many.assert_called_once_with(bridge.loop, [bridge])
        assert owner._results == {}
        assert result._execution._wake is None
