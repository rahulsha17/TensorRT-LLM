# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RPC frontend authority and result routing without inference or a Ray cluster."""

from concurrent.futures import Future
from dataclasses import replace
from queue import Queue
from threading import Event
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
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.executor.utils import ErrorResponse

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
        self._get_next_client_id = Mock(return_value=41)


def request(version=OLD):
    value = SimpleNamespace(id=None, execution_version=version, disaggregated_params=None)
    value.set_id = lambda identity: setattr(value, "id", identity)
    return value


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
    owner.rpc_client.submit.return_value.remote_future.return_value = Future()
    yield owner, table
    owner._shutdown_execution_control()
    server.close()


def test_rpc_bootstrap_is_lazy_and_uses_dedicated_authenticated_mailbox(frontend):
    owner, table = frontend
    owner.rpc_client.start_execution_mailbox.assert_not_called()
    control = owner.get_execution_controller()
    assert control.register(OLD).wait().observed_version == OLD
    assert table.current_version("tenant", "scope") == OLD
    assert len(control._transport.participants) == 1
    assert owner.get_execution_controller() is control
    owner.rpc_client.start_execution_mailbox.assert_called_once_with(owner._execution_session_id)
    owner.rpc_client.start_execution_mailbox.return_value.remote.assert_called_once_with(timeout=5)


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
    owner.rpc_client.submit.assert_not_called()
    assert owner._results == {}


def test_pending_rpc_send_does_not_hold_authority_or_prevent_revoke(frontend):
    owner, _ = frontend
    control = owner.get_execution_controller()
    control.register(OLD).wait()
    result = owner.submit(request())
    sent = owner.rpc_client.submit.return_value.remote_future.return_value
    assert not sent.done()
    owner.rpc_client.submit.return_value.remote.assert_not_called()
    control.advance(REVISION).publication.wait()
    with pytest.raises(ExecutionAuthorityError):
        result._result_step(0.1)
    assert not sent.done()
    sent.set_result(None)


def test_immediate_async_send_failure_runs_callback_outside_authority(frontend):
    owner, _ = frontend
    control = owner.get_execution_controller()
    control.register(OLD).wait()
    failed = Future()
    failed.set_exception(OSError("send failed"))
    owner.rpc_client.submit.return_value.remote_future.return_value = failed
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
    owner, _ = frontend
    owner.get_execution_controller().register(OLD).wait()
    result = owner.submit(request())
    owner.rpc_client.submit.return_value.remote_future.return_value.cancel()
    error = result.queue.get_nowait()
    assert isinstance(error, ErrorResponse) and "cancelled" in error.error_msg
    assert owner._results == {}


def test_early_worker_reply_finds_result_registered_before_send(frontend):
    owner, _ = frontend

    def immediate_reply(**kwargs):
        assert 41 in owner._results
        owner.handle_responses([ErrorResponse(41, "worker reply", 9)])

    owner.rpc_client.submit.return_value.remote.side_effect = immediate_reply
    result = owner.submit(request(None))
    assert result.queue.get_nowait().error_msg == "worker reply"
    assert result._execution is None and owner._results == {}


def test_synchronous_send_rejection_removes_result_and_host_hold(frontend):
    owner, _ = frontend
    control = owner.get_execution_controller()
    control.register(OLD).wait()
    owner.rpc_client.submit.return_value.remote_future.side_effect = RuntimeError("send rejected")
    with pytest.raises(RuntimeError, match="send rejected"):
        owner.submit(request())
    assert owner._results == {}
    assert not list(control._gate._executions)


def test_stale_terminal_response_is_dropped_and_detaches_notifications(frontend):
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


@pytest.mark.parametrize("revoked", [False, True])
def test_async_consumer_is_not_notified_for_dropped_rpc_response(frontend, monkeypatch, revoked):
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
