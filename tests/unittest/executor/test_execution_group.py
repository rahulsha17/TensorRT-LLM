# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One revision owner across frontends, complete bootstrap and shutdown fencing."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from threading import Barrier, Event
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
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin

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
        self.hold = Mock(num_tokens=tokens)

    def _create_execution_version_table(self, session_id):
        table = WorkerVersionTable(session_id)
        table.drain_tracker = RequestDrainTracker()
        table.alignment_state = WorkerAlignmentState(lambda _: self.hold, lambda: None)
        return table

    def close(self):
        self._shutdown_execution_control()
        if self._execution_version_table is not None:
            self._execution_version_table.alignment_state.process()


class RemoteFrontend(RpcExecutorMixin, ExecutionControlMixin):
    def __init__(self, worker):
        self._init_execution_control()
        self.doing_shutdown = False
        self._shutdown_event = Event()
        self.worker_kwargs = {
            "llm_args": SimpleNamespace(
                backend="pytorch",
                num_serve_frontends=1,
                parallel_config=SimpleNamespace(world_size=1),
            )
        }
        # Bootstrap RPC is replaced; all subsequent mailbox traffic uses real
        # authenticated IPC and the production publication thread.
        self.rpc_client = SimpleNamespace(
            start_execution_mailbox=lambda session: SimpleNamespace(
                remote=lambda timeout: worker.start_execution_mailbox(session)
            )
        )

    def close(self):
        self._shutdown_execution_control()


class CohortFrontend(ExecutionControlMixin):
    def __init__(self, workers):
        self._init_execution_control()
        self.doing_shutdown = False
        self.workers = workers

    def _create_execution_transport(self, session_id):
        participants = tuple(ExecutionParticipant(rank, 0) for rank in range(len(self.workers)))
        transports = {
            participant: IpcPublicationTransport(worker.start_execution_mailbox(session_id))
            for participant, worker in zip(participants, self.workers)
        }
        return CollectivePublicationTransport(
            participants, transports, publication_source=participants[0]
        )

    def close(self):
        self._shutdown_execution_control()


@pytest.fixture(params=["local", "ipc", "nested", "mixed"])
def cluster(request):
    with ExitStack() as stack:
        workers = [
            Worker(tokens=2 if rank == 0 else 3)
            for rank in range(6 if request.param == "nested" else 2)
        ]
        for worker in workers:
            stack.callback(worker.close)
        if request.param == "local":
            frontends = tuple(workers)
        elif request.param == "ipc":
            frontends = tuple(RemoteFrontend(worker) for worker in workers)
        elif request.param == "mixed":
            frontends = (workers[0], RemoteFrontend(workers[1]))
        else:
            frontends = (CohortFrontend(workers[:2]), CohortFrontend(workers[2:]))
        for frontend in frontends:
            stack.callback(frontend.close)
        identities = [worker._execution_session_id for worker in workers]
        controller = create_shared_execution_controller(frontends)
        yield SimpleNamespace(
            frontends=frontends,
            workers=workers,
            identities=identities,
            controller=controller,
            tables=[worker.get_execution_version_table() for worker in workers],
        )


def test_shared_owner_preserves_worker_identity_and_revokes_every_frontend(cluster):
    controller = cluster.controller
    assert all(frontend.get_execution_controller() is controller for frontend in cluster.frontends)
    assert [worker._execution_session_id for worker in cluster.workers] == cluster.identities
    assert len({table.session_id for table in cluster.tables}) == 1
    assert cluster.tables[0].session_id not in cluster.identities
    controller.register(OLD).wait()
    executions = []
    for frontend in cluster.frontends:
        with frontend._execution_admission(OLD) as execution:
            execution._cancel = Mock()
            executions.append((execution, execution._cancel))
    revision = controller.advance(REVISION)
    for execution, cancel in executions:
        with pytest.raises(ExecutionAuthorityError):
            execution.check()
        cancel.assert_called_once()
    revision.publication.wait()
    for frontend in cluster.frontends:
        for version in (OLD, NEW):
            with pytest.raises(ExecutionAuthorityError):
                with frontend._execution_admission(version):
                    pass
    assert all(table.current_version("tenant", "scope") == NEW for table in cluster.tables)


def test_all_worker_fence_precedes_common_prefix_resume(cluster):
    controller = cluster.controller
    controller.register(OLD).wait()
    for table in cluster.tables:
        table.drain_tracker.enqueue(OLD, lambda: 42)
    revision = controller.advance(REVISION, alignment=INPUT)
    revision.publication.wait()
    for table in cluster.tables:
        table.alignment_state.process()
    for table in cluster.tables[:-1]:
        table.drain_tracker.complete(42, OLD)
    with pytest.raises(TimeoutError):
        revision.fence.wait(0.05)
    with pytest.raises(TimeoutError):
        revision.alignment.wait(0.05)
    cluster.tables[-1].drain_tracker.complete(42, OLD)
    revision.fence.wait()
    receipt = revision.alignment.wait()
    assert receipt.reusable_prefix_length == 2
    controller.resume(receipt).wait()
    for frontend in cluster.frontends:
        with frontend._execution_admission(NEW) as execution:
            assert execution.version == NEW
    for table, worker in zip(cluster.tables, cluster.workers):
        assert (
            table.alignment_state.claim(NEW, 43, INPUT.input_token_ids, INPUT.cache_salt, None) == 2
        )
        table.alignment_state.confirm_attachment(NEW, 43, 2)
        worker.hold.close.assert_called_once()


def test_lost_member_does_not_hide_revocation_from_healthy_groups(cluster):
    cluster.controller.register(OLD).wait()
    cluster.tables[0].close()
    revision = cluster.controller.advance(REVISION, alignment=INPUT)
    with pytest.raises(RuntimeError):
        revision.publication.wait()
    assert all(table.current_version("tenant", "scope") == NEW for table in cluster.tables[1:])
    with pytest.raises(RuntimeError):
        revision.fence.wait()
    for frontend in cluster.frontends:
        for version in (OLD, NEW):
            with pytest.raises(ExecutionAuthorityError):
                with frontend._execution_admission(version):
                    pass


@pytest.mark.parametrize("close_owner", [False, True])
def test_closing_owner_or_any_member_invalidates_all_frontend_admission(cluster, close_owner):
    cluster.controller.register(OLD).wait()
    if close_owner:
        cluster.controller.close()
    else:
        cluster.frontends[0].close()
    assert cluster.controller.closed
    for frontend in cluster.frontends:
        with pytest.raises(RuntimeError, match="closed"):
            frontend.get_execution_controller()
        for version in (None, OLD):
            with pytest.raises(RuntimeError, match="closed"):
                with frontend._execution_admission(version):
                    pass


@pytest.mark.parametrize("started", ["untagged", "controller", "mailbox", "closed"])
def test_existing_admission_or_binding_rejects_group_without_mutating_other_member(started):
    first, second = Worker(), Worker()
    try:
        if started == "untagged":
            with second._execution_admission(None):
                pass
        elif started == "controller":
            second.get_execution_controller()
        elif started == "mailbox":
            second.start_execution_mailbox("external")
        else:
            second.close()
        with pytest.raises(RuntimeError):
            create_shared_execution_controller((first, second))
        assert first._execution_version_table is None
        assert first._execution_controller is None
        with first._execution_admission(None):
            pass
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_partial_bootstrap_fails_every_member_even_when_cleanup_raises(cleanup_failure):
    first, second, third = Worker(), Worker(), Worker()
    second._create_execution_transport = Mock(side_effect=TimeoutError("worker unavailable"))
    if cleanup_failure:
        create = first._create_execution_transport

        def broken_cleanup(session):
            transport = create(session)
            transport.close = Mock(side_effect=RuntimeError("cleanup failed"))
            return transport

        first._create_execution_transport = broken_cleanup
    try:
        with pytest.raises((TimeoutError, RuntimeError)):
            create_shared_execution_controller((first, second, third))
        for frontend in (first, second, third):
            for version in (None, OLD):
                with pytest.raises(RuntimeError, match="restart required"):
                    with frontend._execution_admission(version):
                        pass
            with pytest.raises(RuntimeError, match="restart required"):
                frontend.get_execution_controller()
        assert third._execution_version_table is None
    finally:
        for frontend in (first, second, third):
            frontend.close()


@pytest.mark.parametrize("kind", ["ipc", "nested", "mixed"])
def test_aliases_of_the_same_worker_cannot_count_as_distinct_members(kind):
    worker = Worker()
    if kind == "nested":
        frontends = (CohortFrontend([worker]), CohortFrontend([worker]))
    elif kind == "mixed":
        frontends = (worker, RemoteFrontend(worker))
    else:
        frontends = (RemoteFrontend(worker), RemoteFrontend(worker))
    try:
        with pytest.raises(ValueError, match="repeated worker endpoint"):
            create_shared_execution_controller(frontends)
        assert all(frontend._execution_controller is None for frontend in frontends)
    finally:
        for frontend in frontends:
            frontend.close()
        worker.close()


@pytest.mark.parametrize("failure", [False, True])
def test_admission_waits_for_group_bootstrap_success_or_failure(failure):
    first, second = Worker(), Worker()
    entered, release, attempting, admitted = Event(), Event(), Event(), Event()
    create = second._create_execution_transport

    def bootstrap(session):
        entered.set()
        assert release.wait(3)
        if failure:
            raise TimeoutError("bootstrap unavailable")
        return create(session)

    def submit():
        attempting.set()
        with first._execution_admission(None):
            admitted.set()

    second._create_execution_transport = bootstrap
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            setup = pool.submit(create_shared_execution_controller, (first, second))
            assert entered.wait(3)
            admission = pool.submit(submit)
            assert attempting.wait(3)
            assert not admitted.wait(0.05)
            release.set()
            if failure:
                with pytest.raises(TimeoutError):
                    setup.result(3)
                with pytest.raises(RuntimeError, match="restart required"):
                    admission.result(3)
                assert not admitted.is_set()
            else:
                setup.result(3)
                admission.result(3)
                assert admitted.is_set()
    finally:
        release.set()
        first.close()
        second.close()


def test_competing_revisions_through_different_frontends_have_one_winner(cluster):
    cluster.controller.register(OLD).wait()
    barrier = Barrier(2, timeout=3)

    def revise(index):
        controller = cluster.frontends[index].get_execution_controller()
        barrier.wait()
        return controller.advance(replace(REVISION, revision_id=f"edit-{index}"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(revise, index) for index in range(2)]
    assert sum(future.exception() is None for future in futures) == 1
    winner = next(future.result() for future in futures if future.exception() is None)
    winner.publication.wait()
    assert cluster.controller.snapshot("tenant", "scope").current_version == NEW


def test_reversed_overlapping_group_bootstraps_do_not_deadlock():
    workers = (Worker(), Worker())
    barrier = Barrier(2, timeout=3)

    def bootstrap(frontends):
        barrier.wait()
        return create_shared_execution_controller(frontends)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(bootstrap, frontends) for frontends in (workers, workers[::-1])]
            for future in futures:
                try:
                    future.result(3)
                except RuntimeError as error:
                    assert "before admission or worker binding" in str(error)
        assert sum(future.exception() is None for future in futures) == 1
        assert workers[0].get_execution_controller() is workers[1].get_execution_controller()
    finally:
        for worker in workers:
            worker.close()


@pytest.mark.parametrize("kind", ["list", "empty", "single", "duplicate", "invalid"])
def test_invalid_membership_is_rejected_without_starting_workers(kind):
    worker = Worker()
    cases = {
        "list": [worker, worker],
        "empty": (),
        "single": (worker,),
        "duplicate": (worker, worker),
        "invalid": (worker, object()),
    }
    try:
        with pytest.raises((TypeError, ValueError)):
            create_shared_execution_controller(cases[kind])
        assert worker._execution_version_table is None
    finally:
        worker.close()
