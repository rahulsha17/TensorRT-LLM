# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote owner ordering over real IPC, worker tables, drain and ALIGN state."""

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_alignment import AlignmentInput, WorkerAlignmentState
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_collective import CollectivePublicationTransport, ExecutionParticipant
from tensorrt_llm.execution_fence import RequestDrainTracker
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import WorkerVersionTable
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.execution_control import (
    ExecutionControlMixin,
    WorkerExecutionControlMixin,
)
from tensorrt_llm.executor.execution_group import create_shared_execution_controller
from tensorrt_llm.executor.execution_mailbox import IpcPublicationTransport
from tensorrt_llm.executor.execution_member import prepare_execution_member
from tensorrt_llm.executor.execution_remote import create_remote_execution_controller

pytestmark = pytest.mark.cpu_only
OLD = ExecutionVersion("tenant", "scope", 7)
NEW = replace(OLD, epoch=8)
REVISION = Revision("edit", OLD, NEW)
INPUT = AlignmentInput((1, 2, 3, 4, 5), cache_salt="shared")


class Worker(WorkerExecutionControlMixin, ExecutionControlMixin):
    def __init__(self, tokens=3):
        self._init_execution_control()
        self.doing_shutdown = False
        self._is_pytorch_backend = True
        self.engine = object()
        self.llm_args = SimpleNamespace(
            num_serve_frontends=1, parallel_config=SimpleNamespace(world_size=1)
        )
        self.tokens = tokens
        self.holds = []

    def _reserve(self, _):
        hold = Mock(num_tokens=self.tokens)
        self.holds.append(hold)
        return hold

    def _create_execution_version_table(self, session):
        table = WorkerVersionTable(session)
        table.drain_tracker = RequestDrainTracker()
        table.alignment_state = WorkerAlignmentState(self._reserve, lambda: None)
        return table

    def close(self):
        self._shutdown_execution_control()
        if self._execution_version_table is not None:
            self._execution_version_table.alignment_state.process()


class Proxy(ExecutionControlMixin):
    def __init__(self, workers):
        self._init_execution_control()
        self.doing_shutdown = False
        self.workers = workers

    def _create_execution_transport(self, session):
        transports = tuple(
            IpcPublicationTransport(w.start_execution_mailbox(session)) for w in self.workers
        )
        if len(transports) == 1:
            return transports[0]
        participants = tuple(ExecutionParticipant(rank, 0) for rank in range(len(transports)))
        return CollectivePublicationTransport(
            participants, dict(zip(participants, transports)), publication_source=participants[0]
        )

    def close(self):
        self._shutdown_execution_control()


@pytest.fixture(params=["direct", "ipc", "nested", "mixed"])
def cluster(request):
    with ExitStack() as cleanup:
        workers = [Worker(2), Worker(3)]
        if request.param == "nested":
            workers.extend([Worker(3), Worker(3)])
        for worker in workers:
            cleanup.callback(worker.close)
        if request.param == "direct":
            frontends = tuple(workers)
        elif request.param == "nested":
            frontends = (Proxy(workers[:2]), Proxy(workers[2:]))
        elif request.param == "mixed":
            frontends = (workers[0], Proxy(workers[1:]))
        else:
            frontends = tuple(Proxy([worker]) for worker in workers)
        for frontend in frontends:
            cleanup.callback(frontend.close)
        endpoints = tuple(prepare_execution_member(frontend, "session") for frontend in frontends)
        controller = create_remote_execution_controller(endpoints)
        cleanup.callback(controller.close)
        yield SimpleNamespace(
            workers=workers,
            frontends=frontends,
            endpoints=endpoints,
            controller=controller,
            tables=[worker._execution_version_table for worker in workers],
        )


def reject(frontend, version):
    with pytest.raises(RuntimeError):
        with frontend._execution_admission(version):
            pytest.fail("unauthorized request was admitted")


def test_remote_scope_lifecycle_preserves_topology_and_common_prefix(cluster):
    controller = cluster.controller
    for frontend in cluster.frontends:
        with pytest.raises(RuntimeError, match="external runtime"):
            frontend.get_execution_controller()
        reject(frontend, OLD)
    controller.register(OLD).wait()
    active = []
    for frontend in cluster.frontends:
        with frontend._execution_admission(OLD) as execution:
            execution._wake, execution._cancel = Mock(), Mock()
            active.append((execution, execution._wake, execution._cancel))
    for table in cluster.tables:
        table.drain_tracker.enqueue(OLD, lambda: 42)
    revised = controller.advance(REVISION, alignment=INPUT)
    for execution, wake, cancel in active:
        with pytest.raises(ExecutionAuthorityError):
            execution.commit(lambda: pytest.fail("published after accepted ADVANCE"))
        wake.assert_called_once()
        cancel.assert_called_once()
    revised.publication.wait()
    for frontend in cluster.frontends:
        reject(frontend, OLD)
        reject(frontend, NEW)
    for table in cluster.tables:
        table.alignment_state.process()
    with pytest.raises(TimeoutError):
        revised.fence.wait(0.05)
    for table in cluster.tables[:-1]:
        table.drain_tracker.complete(42, OLD)
    with pytest.raises(TimeoutError):
        revised.alignment.wait(0.05)
    cluster.tables[-1].drain_tracker.complete(42, OLD)
    revised.fence.wait()
    receipt = revised.alignment.wait()
    assert receipt.reusable_prefix_length == 2
    controller.resume(receipt).wait()
    for frontend in cluster.frontends:
        with frontend._execution_admission(NEW) as execution:
            execution.check()
    for table, worker in zip(cluster.tables, cluster.workers):
        assert (
            table.alignment_state.claim(NEW, 43, INPUT.input_token_ids, INPUT.cache_salt, None) == 2
        )
        table.alignment_state.confirm_attachment(NEW, 43, 2)
        worker.holds[-1].close.assert_called_once()


def test_owner_does_not_accept_until_every_frontend_has_sealed(cluster, monkeypatch):
    controller = cluster.controller
    controller.register(OLD).wait()
    entered, release = Event(), Event()
    slow = cluster.frontends[-1]._execution_frontend_gate
    original = slow.seal

    def seal(owner, revision):
        entered.set()
        assert release.wait(3)
        return original(owner, revision)

    monkeypatch.setattr(slow, "seal", seal)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(controller.advance, REVISION)
        try:
            assert entered.wait(2)
            assert controller.snapshot("tenant", "scope").current_version == OLD
            assert not pending.done()
            with cluster.frontends[-1]._execution_admission(OLD) as execution:
                execution.check()
        finally:
            release.set()
        pending.result(3).publication.wait()
    assert controller.snapshot("tenant", "scope").current_version == NEW
    for frontend in cluster.frontends:
        reject(frontend, OLD)


def test_worker_publication_cannot_delay_host_seal_or_acceptance(cluster, monkeypatch):
    controller = cluster.controller
    entered, release = Event(), Event()
    server = cluster.frontends[0]._execution_member_server
    original = server._transport.publish

    def publish(update, stop):
        if update.version == OLD:
            entered.set()
            assert release.wait(3)
        return original(update, stop)

    monkeypatch.setattr(server._transport, "publish", publish)
    initial = controller.register(OLD)
    try:
        assert entered.wait(2)
        # Declaration allows immediate revision even before initial publication.
        revised = controller.advance(REVISION)
        assert controller.snapshot("tenant", "scope").current_version == NEW
        for frontend in cluster.frontends:
            reject(frontend, OLD)
            reject(frontend, NEW)
    finally:
        release.set()
    with pytest.raises(ExecutionAuthorityError, match="superseded"):
        initial.wait()
    revised.publication.wait()


def test_partial_host_seal_failure_never_accepts_or_reopens_old_epoch(cluster, monkeypatch):
    controller = cluster.controller
    controller.register(OLD).wait()
    monkeypatch.setattr(
        cluster.frontends[-1]._execution_frontend_gate,
        "seal",
        Mock(side_effect=RuntimeError("lost seal")),
    )
    with pytest.raises(RuntimeError, match="host seal"):
        controller.advance(REVISION)
    assert controller.closed
    assert controller.snapshot("tenant", "scope").current_version == OLD
    controller.close()
    for frontend in cluster.frontends:
        reject(frontend, OLD)
        reject(frontend, NEW)


def test_resume_cannot_open_any_frontend_before_all_worker_acks(cluster, monkeypatch):
    controller = cluster.controller
    controller.register(OLD).wait()
    revised = controller.advance(REVISION, alignment=INPUT)
    revised.publication.wait()
    for table in cluster.tables:
        table.alignment_state.process()
    aligned = revised.alignment.wait()
    entered, release = Event(), Event()
    server = cluster.frontends[-1]._execution_member_server
    original = server._transport.publish

    def publish(update, stop):
        if update.resume is not None:
            entered.set()
            assert release.wait(3)
        return original(update, stop)

    monkeypatch.setattr(server._transport, "publish", publish)
    resumed = controller.resume(aligned)
    try:
        assert entered.wait(2)
        for frontend in cluster.frontends:
            reject(frontend, NEW)
        assert controller.snapshot("tenant", "scope").pending_revision == REVISION
    finally:
        release.set()
    resumed.wait()
    for frontend in cluster.frontends:
        with frontend._execution_admission(NEW) as execution:
            execution.check()


def test_historical_retry_and_invalid_revisions_do_not_seal_again(cluster, monkeypatch):
    controller = cluster.controller
    controller.register(OLD).wait()
    first = controller.advance(REVISION)
    first.publication.wait()
    next_revision = Revision("edit-again", NEW, replace(NEW, epoch=9))
    controller.advance(next_revision).publication.wait()
    sealing = Mock(wraps=controller._frontends.seal)
    monkeypatch.setattr(controller._frontends, "seal", sealing)
    assert controller.advance(REVISION).advance_receipt == first.advance_receipt
    with pytest.raises(ValueError):
        controller.advance(replace(REVISION, revision_id="stale"))
    with pytest.raises(ValueError):
        controller.advance(Revision("edit", NEW, replace(NEW, epoch=9)))
    with pytest.raises(ValueError):
        controller.advance(REVISION, alignment=INPUT)
    sealing.assert_not_called()
    assert not controller.closed
    assert controller.snapshot("tenant", "scope").current_version.epoch == 9


def test_racing_revisions_have_one_cas_winner(cluster):
    controller = cluster.controller
    controller.register(OLD).wait()
    with ThreadPoolExecutor(2) as pool:
        pending = [
            pool.submit(controller.advance, replace(REVISION, revision_id=f"edit-{index}"))
            for index in range(2)
        ]
        results = []
        for future in pending:
            try:
                results.append(future.result())
            except ValueError:
                results.append(None)
    assert sum(result is not None for result in results) == 1
    assert not controller.closed
    assert controller.snapshot("tenant", "scope").current_version == NEW


def test_preparation_is_idempotent_and_cannot_mix_owner_modes(cluster):
    for frontend, endpoint in zip(cluster.frontends, cluster.endpoints):
        assert prepare_execution_member(frontend, "session") == endpoint
        with pytest.raises(ValueError):
            prepare_execution_member(frontend, "foreign")
    with pytest.raises(RuntimeError):
        create_shared_execution_controller(cluster.frontends)
    with pytest.raises(RuntimeError):
        create_remote_execution_controller(cluster.endpoints)
    cluster.controller.register(OLD).wait()
    for frontend in cluster.frontends:
        with frontend._execution_admission(OLD) as execution:
            execution.check()


def test_deferred_align_keeps_every_frontend_pending_until_resume(cluster):
    controller = cluster.controller
    controller.register(OLD).wait()
    revision = controller.advance(REVISION)
    revision.publication.wait()
    revision.fence.wait()
    for frontend in cluster.frontends:
        reject(frontend, NEW)
    deferred = controller.align(REVISION, INPUT)
    deferred.publication.wait()
    for table in cluster.tables:
        table.alignment_state.process()
    receipt = deferred.alignment.wait()
    for frontend in cluster.frontends:
        reject(frontend, NEW)
    controller.resume(receipt).wait()
    for frontend in cluster.frontends:
        with frontend._execution_admission(NEW) as execution:
            execution.check()


def test_partial_activation_failure_closes_authority_but_retains_worker_holds(cluster, monkeypatch):
    controller = cluster.controller
    controller.register(OLD).wait()
    revision = controller.advance(REVISION, alignment=INPUT)
    revision.publication.wait()
    for table in cluster.tables:
        table.alignment_state.process()
    aligned = revision.alignment.wait()
    monkeypatch.setattr(
        cluster.frontends[-1]._execution_frontend_gate,
        "activate",
        Mock(side_effect=RuntimeError("lost activation")),
    )
    with pytest.raises(RuntimeError):
        controller.resume(aligned).wait()
    assert controller.closed
    controller.close()
    for frontend in cluster.frontends:
        reject(frontend, NEW)
    for worker in cluster.workers:
        worker.holds[-1].close.assert_not_called()


def test_newer_revision_can_overtake_resume_without_reopening_old_frontends(cluster, monkeypatch):
    controller = cluster.controller
    controller.register(OLD).wait()
    first = controller.advance(REVISION, alignment=INPUT)
    first.publication.wait()
    for table in cluster.tables:
        table.alignment_state.process()
    aligned = first.alignment.wait()
    entered, release = Event(), Event()
    transport = cluster.frontends[-1]._execution_member_server._transport
    original = transport.publish

    def publish(update, stop):
        if update.resume is not None:
            entered.set()
            assert release.wait(3)
        return original(update, stop)

    monkeypatch.setattr(transport, "publish", publish)
    resumed = controller.resume(aligned)
    try:
        assert entered.wait(2)
        second = controller.advance(Revision("edit-again", NEW, replace(NEW, epoch=9)))
        assert controller.snapshot("tenant", "scope").current_version.epoch == 9
        for frontend in cluster.frontends:
            reject(frontend, NEW)
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="superseded"):
        resumed.wait()
    second.publication.wait()
    assert not controller.closed
    for frontend in cluster.frontends:
        reject(frontend, OLD)
        reject(frontend, NEW)


def test_partial_initial_declaration_never_registers_or_admits_scope(cluster, monkeypatch):
    monkeypatch.setattr(
        cluster.frontends[-1]._execution_frontend_gate,
        "declare",
        Mock(side_effect=RuntimeError("lost declaration")),
    )
    with pytest.raises(RuntimeError, match="scope declaration"):
        cluster.controller.register(OLD)
    with pytest.raises(KeyError):
        cluster.controller.snapshot("tenant", "scope")
    cluster.controller.close()
    for frontend in cluster.frontends:
        reject(frontend, OLD)


def test_prepared_frontend_cannot_admit_until_bound_or_create_a_local_owner():
    worker = Worker()
    try:
        prepare_execution_member(worker, "session")
        reject(worker, None)
        reject(worker, OLD)
        with pytest.raises(RuntimeError, match="external runtime"):
            worker.get_execution_controller()
    finally:
        worker.close()


def test_failed_publication_endpoint_bootstrap_closes_bound_host():
    worker = Worker()
    try:
        endpoint = prepare_execution_member(worker, "session")
        unreachable = replace(endpoint.publication, address=("tcp://127.0.0.1:1", b"test-key"))
        with pytest.raises(RuntimeError, match="worker membership"):
            create_remote_execution_controller(
                (replace(endpoint, publication=unreachable),), timeout=0.1, lease_seconds=1
            )
        assert worker._execution_frontend_gate.closed
        reject(worker, OLD)
    finally:
        worker.close()


def test_frontend_teardown_attempts_every_channel_after_an_error(monkeypatch):
    worker = Worker()
    prepare_execution_member(worker, "session")
    host = worker._execution_frontend_server
    original = host.close

    def fail_after_close():
        original()
        raise RuntimeError("host close failure")

    try:
        with monkeypatch.context() as patch:
            member_close = Mock(wraps=worker._execution_member_server.close)
            table_close = Mock(wraps=worker._execution_version_table.close)
            patch.setattr(host, "close", fail_after_close)
            patch.setattr(worker._execution_member_server, "close", member_close)
            patch.setattr(worker._execution_version_table, "close", table_close)
            with pytest.raises(RuntimeError, match="teardown"):
                worker.close()
            member_close.assert_called_once()
            table_close.assert_called_once()
            reject(worker, OLD)
    finally:
        worker.close()


def test_duplicate_workers_are_rejected_across_distinct_frontends():
    worker = Worker()
    fronts = (Proxy([worker]), Proxy([worker]))
    try:
        endpoints = tuple(prepare_execution_member(front, "session") for front in fronts)
        with pytest.raises(ValueError, match="repeated worker"):
            create_remote_execution_controller(endpoints)
        for front in fronts:
            reject(front, OLD)
    finally:
        for front in fronts:
            front.close()
        worker.close()


def test_lost_member_fails_owner_and_wakes_healthy_frontend():
    workers = (Worker(), Worker())
    controller = None
    try:
        endpoints = tuple(prepare_execution_member(worker, "session") for worker in workers)
        controller = create_remote_execution_controller(endpoints, timeout=0.25, lease_seconds=2)
        controller.register(OLD).wait()
        with workers[1]._execution_admission(OLD) as execution:
            wake = Event()
            execution._wake = lambda _: wake.set()
        workers[0].close()
        assert controller._stop.wait(3)
        assert wake.wait(3)
        reject(workers[1], OLD)
    finally:
        if controller is not None:
            controller.close()
        for worker in workers:
            worker.close()


def _frontend_process(pipe, tokens):
    worker = Worker(tokens)
    executions = {}
    try:
        pipe.send(prepare_execution_member(worker, "session"))
        while True:
            worker._execution_version_table.alignment_state.process()
            if not pipe.poll(0.01):
                continue
            operation = pipe.recv()
            if operation == "stop":
                return
            try:
                if operation == "old":
                    with worker._execution_admission(OLD) as execution:
                        worker._execution_version_table.drain_tracker.enqueue(OLD, lambda: 42)
                        executions[OLD] = execution
                    value = "admitted"
                elif operation == "old-output":
                    executions[OLD].check()
                    value = "published"
                elif operation == "drain":
                    worker._execution_version_table.drain_tracker.complete(42, OLD)
                    value = "drained"
                elif operation == "new":
                    with worker._execution_admission(NEW) as execution:
                        execution.check()
                        state = worker._execution_version_table.alignment_state
                        value = state.claim(NEW, 43, INPUT.input_token_ids, INPUT.cache_salt, None)
                        state.confirm_attachment(NEW, 43, value)
                else:
                    raise ValueError("unknown test operation")
                pipe.send(value)
            except RuntimeError as error:
                pipe.send(("rejected", str(error)))
    finally:
        worker.close()
        pipe.close()


def _probe(pipe, operation):
    pipe.send(operation)
    assert pipe.poll(10), operation
    return pipe.recv()


def test_full_lifecycle_with_two_frontends_in_separate_processes():
    context = multiprocessing.get_context("spawn")
    processes, pipes, endpoints = [], [], []
    controller = None
    try:
        for tokens in (2, 3):
            parent, child = context.Pipe()
            process = context.Process(target=_frontend_process, args=(child, tokens))
            process.start()
            child.close()
            processes.append(process)
            pipes.append(parent)
            assert parent.poll(15)
            endpoints.append(parent.recv())
        controller = create_remote_execution_controller(tuple(endpoints))
        controller.register(OLD).wait()
        for pipe in pipes:
            assert _probe(pipe, "old") == "admitted"
        revision = controller.advance(REVISION, alignment=INPUT)
        revision.publication.wait()
        for pipe in pipes:
            assert _probe(pipe, "old-output")[0] == "rejected"
            assert _probe(pipe, "new")[0] == "rejected"
        with pytest.raises(TimeoutError):
            revision.fence.wait(0.05)
        assert _probe(pipes[0], "drain") == "drained"
        with pytest.raises(TimeoutError):
            revision.fence.wait(0.05)
        assert _probe(pipes[1], "drain") == "drained"
        revision.fence.wait()
        aligned = revision.alignment.wait()
        assert aligned.reusable_prefix_length == 2
        controller.resume(aligned).wait()
        for pipe in pipes:
            assert _probe(pipe, "new") == 2
        controller.close()
        for pipe in pipes:
            assert _probe(pipe, "new")[0] == "rejected"
    finally:
        if controller is not None:
            controller.close()
        for pipe in pipes:
            try:
                pipe.send("stop")
            except (BrokenPipeError, EOFError):
                pass
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(5)
            assert process.exitcode == 0
        for pipe in pipes:
            pipe.close()


def _owner_process(pipe, endpoints):
    controller = create_remote_execution_controller(endpoints, timeout=0.5, lease_seconds=3)
    controller.register(OLD).wait()
    pipe.send("ready")
    assert pipe.recv() == "exit"
    os._exit(0)


def test_abrupt_owner_process_death_closes_bound_executor_admission():
    workers = (Worker(), Worker())
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    owner = None
    try:
        endpoints = tuple(prepare_execution_member(worker, "session") for worker in workers)
        owner = context.Process(target=_owner_process, args=(child, endpoints))
        owner.start()
        child.close()
        assert parent.poll(15) and parent.recv() == "ready"
        wakes = []
        for worker in workers:
            with worker._execution_admission(OLD) as execution:
                wake = Event()
                execution._wake = lambda _, wake=wake: wake.set()
                wakes.append((execution, wake))
        parent.send("exit")
        owner.join(5)
        assert owner.exitcode == 0
        for execution, wake in wakes:
            assert wake.wait(6)
            with pytest.raises(ExecutionAuthorityError):
                execution.check()
        for worker in workers:
            reject(worker, OLD)
            reject(worker, None)
    finally:
        if owner is not None and owner.is_alive():
            owner.terminate()
            owner.join(5)
        parent.close()
        for worker in workers:
            worker.close()
