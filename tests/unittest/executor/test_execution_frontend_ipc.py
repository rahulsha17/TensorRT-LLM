# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real authenticated host control, including separately spawned frontends/owners."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_alignment import AlignmentInput, AlignmentReceipt, AlignmentRequest
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerIdentity
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.execution_frontend import FrontendAuthorityGate
from tensorrt_llm.executor.execution_frontend_ipc import (
    FrontendAuthorityClient,
    FrontendAuthorityEndpoint,
    FrontendAuthorityServer,
    _Reply,
)

pytestmark = pytest.mark.cpu_only
OLD = ExecutionVersion("tenant", "scope", 7)
NEW = replace(OLD, epoch=8)
REVISION = Revision("edit", OLD, NEW)
INITIAL = VersionUpdate("session", 1, OLD)


def _resume():
    request = AlignmentRequest("session", REVISION, AlignmentInput((1, 2, 3)))
    return VersionUpdate(
        "session", 3, NEW, alignment=request, resume=AlignmentReceipt(request, "hold", 2)
    )


@pytest.fixture
def endpoint():
    now = [100.0]
    gate = FrontendAuthorityGate("session", clock=lambda: now[0])
    server = FrontendAuthorityServer(gate)
    client = FrontendAuthorityClient(server.endpoint, "owner", timeout=1)
    client.bind(10)
    client.observe(INITIAL)
    client.activate(INITIAL)
    yield SimpleNamespace(now=now, gate=gate, server=server, client=client)
    server.close()


def test_ipc_seal_observe_resume_and_release(endpoint):
    client, gate = endpoint.client, endpoint.gate
    with gate.admission(OLD) as execution:
        wake = execution._wake = Mock()
    assert client.seal(REVISION) == REVISION
    wake.assert_called_once()
    assert not client.activate(INITIAL)
    advance = VersionUpdate("session", 2, NEW)
    assert client.observe(advance).observed_version == NEW
    with pytest.raises(RuntimeError, match="RESUME"):
        client.activate(advance)
    resume = _resume()
    assert client.observe(resume).update == resume
    assert client.activate(resume)
    with gate.admission(NEW) as execution:
        execution.check()
    client.release()
    client.release()
    with pytest.raises(ExecutionAuthorityError):
        execution.check()
    with pytest.raises(RuntimeError, match="unavailable"):
        endpoint.server.check_health()


def test_lease_listener_wakes_idle_consumer_without_another_owner_message(endpoint):
    with endpoint.gate.admission(OLD) as execution:
        wake = Event()
        execution._wake = lambda _: wake.set()
    endpoint.now[0] = 110
    assert wake.wait(2)
    with pytest.raises(RuntimeError, match="unavailable"):
        endpoint.client.renew(1)
    with pytest.raises(ExecutionAuthorityError):
        execution.check()


def test_concurrent_heartbeat_and_seal_use_separate_routed_connections(endpoint):
    with ThreadPoolExecutor(8) as pool:
        replies = [pool.submit(endpoint.client.renew, index) for index in range(1, 8)]
        seal = pool.submit(endpoint.client.seal, REVISION)
        assert seal.result(3) == REVISION
        for index, reply in enumerate(replies, 1):
            assert reply.result(3) >= index
    with pytest.raises(ExecutionAuthorityError):
        with endpoint.gate.admission(OLD):
            pytest.fail("seal did not revoke work")


@pytest.mark.parametrize("fault", ["session", "frontend", "owner"])
def test_wrong_lifetime_or_owner_cannot_revoke_the_live_frontend(endpoint, fault):
    descriptor = endpoint.server.endpoint
    owner = "owner"
    if fault == "owner":
        owner = "different"
    else:
        identity = descriptor.identity
        identity = replace(
            identity, **({"session_id": "other"} if fault == "session" else {"worker_id": "other"})
        )
        descriptor = replace(descriptor, identity=identity)
    client = FrontendAuthorityClient(descriptor, owner, timeout=1)
    with pytest.raises(RuntimeError):
        client.release()
    endpoint.server.check_health()
    with endpoint.gate.admission(OLD) as execution:
        execution.check()


def test_unreachable_frontend_times_out_without_an_ack():
    descriptor = FrontendAuthorityEndpoint(
        WorkerIdentity("session", "worker"), ("tcp://127.0.0.1:1", b"test-key")
    )
    with pytest.raises(TimeoutError):
        FrontendAuthorityClient(descriptor, "owner", timeout=0.05).bind(10)


@pytest.mark.parametrize("operation", ["bind", "renew", "seal", "observe", "activate", "release"])
def test_invalid_ack_value_is_never_success(endpoint, monkeypatch, operation):
    monkeypatch.setattr(endpoint.server, "_dispatch", lambda _: "invalid")
    argument = {
        "bind": 10,
        "renew": 1,
        "seal": REVISION,
        "observe": INITIAL,
        "activate": INITIAL,
    }
    with pytest.raises(ValueError, match="acknowledgement"):
        if operation == "release":
            endpoint.client.release()
        else:
            getattr(endpoint.client, operation)(argument[operation])


@pytest.mark.parametrize("fault", ["identity", "request", "operation", "type"])
def test_reply_must_match_this_exact_command(endpoint, monkeypatch, fault):
    from tensorrt_llm.executor.ipc import IpcQueue

    original = IpcQueue.put_nowait

    def corrupt(queue, value):
        if isinstance(value, _Reply):
            if fault == "type":
                value = "invalid"
            else:
                command = value.command
                change = {
                    "identity": {"identity": WorkerIdentity("session", "replacement")},
                    "request": {"request_id": "previous-request"},
                    "operation": {"operation": "release"},
                }[fault]
                value = replace(value, command=replace(command, **change))
        return original(queue, value)

    monkeypatch.setattr(IpcQueue, "put_nowait", corrupt)
    with pytest.raises(ValueError, match="reply"):
        endpoint.client.renew(1)


def test_invalid_authentication_fails_closed(endpoint):
    import zmq

    from tensorrt_llm.executor.ipc import IpcQueue

    wrong = replace(
        endpoint.server.endpoint,
        address=(endpoint.server.endpoint.address[0], b"wrong-authentication-key"),
    )
    queue = IpcQueue(wrong.address, is_server=False, socket_type=zmq.DEALER)
    queue.socket.setsockopt(zmq.LINGER, 0)
    with endpoint.gate.admission(OLD) as execution:
        wake = Event()
        execution._wake = lambda _: wake.set()
    try:
        queue.put_nowait("unauthenticated")
        assert wake.wait(2)
    finally:
        queue.close()
    with pytest.raises(RuntimeError):
        endpoint.server.check_health()
    with pytest.raises(ExecutionAuthorityError):
        execution.check()


def _remote_frontend(pipe):
    gate = FrontendAuthorityGate("session")
    server = FrontendAuthorityServer(gate)
    executions = {}
    wakes = []
    try:
        pipe.send(server.endpoint)
        while (operation := pipe.recv()) != "stop":
            try:
                if operation == "admit-old":
                    version = OLD
                elif operation == "admit-new":
                    version = NEW
                else:
                    version = None
                if version is not None:
                    with gate.admission(version) as execution:
                        execution._wake = lambda _: wakes.append("revoked")
                        executions[version] = execution
                    pipe.send("admitted")
                elif operation == "output-old":
                    executions[OLD].commit(lambda: None)
                    pipe.send("published")
                elif operation == "wakes":
                    pipe.send(len(wakes))
                else:
                    raise ValueError("unknown probe")
            except ExecutionAuthorityError:
                pipe.send("rejected")
    finally:
        server.close()
        pipe.close()


def _owner_then_exit(descriptor):
    client = FrontendAuthorityClient(descriptor, "owner")
    client.bind(1)
    client.observe(INITIAL)
    client.activate(INITIAL)


def test_spawned_frontends_reject_old_effects_before_accepting_successor():
    context = multiprocessing.get_context("spawn")
    processes, pipes, clients = [], [], []
    try:
        for _ in range(2):
            parent, child = context.Pipe()
            process = context.Process(target=_remote_frontend, args=(child,))
            process.start()
            child.close()
            processes.append(process)
            pipes.append(parent)
            assert parent.poll(15)
            client = FrontendAuthorityClient(parent.recv(), "owner")
            client.bind(60)
            client.observe(INITIAL)
            client.activate(INITIAL)
            clients.append(client)
            parent.send("admit-old")
            assert parent.poll(5) and parent.recv() == "admitted"
        for client in clients:
            client.seal(REVISION)
        for parent in pipes:
            for operation in ("admit-old", "admit-new", "output-old"):
                parent.send(operation)
                assert parent.poll(5) and parent.recv() == "rejected"
            parent.send("wakes")
            assert parent.poll(5) and parent.recv() == 1
        for client in clients:
            client.observe(_resume())
        for parent in pipes:
            parent.send("admit-new")
            assert parent.poll(5) and parent.recv() == "rejected"
        for client in clients:
            assert client.activate(_resume())
        for parent in pipes:
            parent.send("admit-new")
            assert parent.poll(5) and parent.recv() == "admitted"
    finally:
        for parent in pipes:
            parent.send("stop")
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(5)
            assert process.exitcode == 0
        for parent in pipes:
            parent.close()


def test_owner_process_exit_eventually_revokes_idle_frontend():
    now = [100.0]
    gate = FrontendAuthorityGate("session", clock=lambda: now[0])
    server = FrontendAuthorityServer(gate)
    context = multiprocessing.get_context("spawn")
    owner = context.Process(target=_owner_then_exit, args=(server.endpoint,))
    try:
        owner.start()
        owner.join(15)
        assert owner.exitcode == 0
        with gate.admission(OLD) as execution:
            wake = Event()
            execution._wake = lambda _: wake.set()
        now[0] = 101
        assert wake.wait(3)
        with pytest.raises(ExecutionAuthorityError):
            execution.commit(lambda: pytest.fail("published after owner loss"))
    finally:
        if owner.is_alive():
            owner.terminate()
            owner.join(5)
        server.close()


def test_endpoint_representation_does_not_expose_authentication_key(endpoint):
    assert "address" not in repr(endpoint.server.endpoint)
    assert repr(endpoint.server.endpoint.address[1]) not in repr(endpoint.server.endpoint)


def test_lost_seal_ack_requires_retry_and_cannot_reopen_old_work(endpoint, monkeypatch):
    from tensorrt_llm.executor.ipc import IpcQueue

    original = IpcQueue.put_nowait
    dropped = []

    def drop_ack(queue, value):
        if isinstance(value, _Reply) and value.command.operation == "seal" and not dropped:
            dropped.append(value)
            return None
        return original(queue, value)

    monkeypatch.setattr(IpcQueue, "put_nowait", drop_ack)
    client = FrontendAuthorityClient(endpoint.server.endpoint, "owner", timeout=0.1)
    with pytest.raises(TimeoutError):
        client.seal(REVISION)
    assert len(dropped) == 1
    with pytest.raises(ExecutionAuthorityError):
        with endpoint.gate.admission(OLD):
            pytest.fail("timed-out seal restored old authority")
    assert client.seal(REVISION) == REVISION
    assert not client.activate(INITIAL)
