# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap membership, admission ordering and all-worker failure behavior."""

import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Barrier, Condition, Event, RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_consensus import ExecutionStepConsensus
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import ExecutionController, WorkerVersionTable
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor import execution_bootstrap as bootstrap
from tensorrt_llm.executor.execution_bootstrap import (
    ExecutionBootstrapInfo,
    ExecutionBootstrapReady,
)
from tensorrt_llm.executor.execution_control import ExecutionControlMixin
from tensorrt_llm.executor.execution_mailbox import ExecutionMailboxServer, IpcPublicationTransport

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 1)


@pytest.mark.parametrize("host", ["0.0.0.0", "224.0.0.1", "::1", "host.invalid", "127.0.0.1:1"])
def test_mailbox_rejects_non_worker_bind_addresses(host):
    with pytest.raises(ValueError):
        ExecutionMailboxServer(WorkerVersionTable("session"), bind_host=host)


def test_explicit_worker_address_keeps_authenticated_publication():
    table = WorkerVersionTable("session")
    server = ExecutionMailboxServer(table, bind_host="127.0.0.1")
    controller = ExecutionController("session", IpcPublicationTransport(server.address))
    try:
        assert server.address[0].startswith("tcp://127.0.0.1:")
        assert isinstance(server.address[1], bytes) and server.address[1]
        assert controller.register(VERSION).wait().observed_version == VERSION
    finally:
        controller.close()
        server.close()


class Frontend(ExecutionControlMixin):
    def __init__(self, size=2):
        self._init_execution_control()
        self.doing_shutdown = False
        self._shutdown_event = Event()
        self.world_size = size
        self.workers = [object() for _ in range(size)]
        self.worker_kwargs = {
            "llm_args": SimpleNamespace(
                backend="pytorch",
                num_serve_frontends=1,
                parallel_config=SimpleNamespace(
                    world_size=size, tp_size=size, pp_size=1, cp_size=1, enable_attention_dp=False
                ),
            )
        }
        self.infos = [
            ExecutionBootstrapInfo(
                self._execution_session_id, f"worker-{rank}", rank, size, "127.0.0.1"
            )
            for rank in range(size)
        ]
        self.ready = [
            ExecutionBootstrapReady(info, (f"tcp://127.0.0.1:{20000 + rank}", b"key"))
            for rank, info in enumerate(self.infos)
        ]
        self.call_all = Mock(
            side_effect=lambda name, args, timeout: self.infos
            if name == "inspect_execution_bootstrap"
            else self.ready
        )
        self.abort = Mock()

    def _create_execution_controller(self):
        return bootstrap.create_execution_cohort(self, self.call_all, self.abort)


@pytest.fixture
def frontend():
    owner = Frontend()
    yield owner
    owner._shutdown_execution_control()


def test_bootstrap_attaches_all_distinct_workers_once(frontend):
    controller = frontend.get_execution_controller()
    assert frontend.get_execution_controller() is controller
    assert len(controller._transport.participants) == 2
    assert frontend.call_all.call_args_list[0].args == (
        "inspect_execution_bootstrap",
        (frontend._execution_session_id, 2),
        30.0,
    )
    assert frontend.call_all.call_args_list[1].args == (
        "start_execution_bootstrap",
        (frontend._execution_session_id, 2),
        90.0,
    )
    frontend.abort.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "other"),
        ("num_serve_frontends", 2),
        ("pp_size", 2),
        ("cp_size", 2),
        ("tp_size", 1),
        ("world_size", 4),
        ("enable_attention_dp", True),
    ],
)
def test_unsupported_bootstrap_has_no_worker_side_effects(frontend, field, value):
    args = frontend.worker_kwargs["llm_args"]
    target = args if field in ("backend", "num_serve_frontends") else args.parallel_config
    setattr(target, field, value)
    with pytest.raises(NotImplementedError):
        frontend.get_execution_controller()
    frontend.call_all.assert_not_called()
    frontend.abort.assert_not_called()
    with frontend._execution_admission(None):
        pass


def test_late_bootstrap_is_rejected_even_after_untagged_admission(frontend):
    with frontend._execution_admission(None):
        pass
    with pytest.raises(RuntimeError, match="before request admission"):
        frontend.get_execution_controller()
    frontend.call_all.assert_not_called()


def test_unregistered_tag_does_not_prevent_initial_bootstrap(frontend):
    with pytest.raises(ExecutionAuthorityError):
        frontend._execution_admission(VERSION)
    assert frontend.get_execution_controller() is not None


@pytest.mark.parametrize("fault", ["missing", "rank", "worker", "type"])
def test_preflight_membership_must_be_complete(frontend, fault):
    if fault == "missing":
        frontend.infos.pop()
    elif fault == "rank":
        frontend.infos[1] = replace(frontend.infos[1], rank=0)
    elif fault == "worker":
        frontend.infos[1] = replace(frontend.infos[1], worker_id=frontend.infos[0].worker_id)
    else:
        frontend.infos[1] = object()
    with pytest.raises(RuntimeError, match="preflight"):
        frontend.get_execution_controller()
    assert frontend.call_all.call_count == 1
    frontend.abort.assert_not_called()


@pytest.mark.parametrize(
    "fault", ["timeout", "missing", "identity", "duplicate_endpoint", "key", "host"]
)
def test_partial_bootstrap_permanently_closes_all_frontend_admission(frontend, fault):
    if fault == "timeout":
        frontend.call_all.side_effect = [frontend.infos, TimeoutError("worker did not park")]
    elif fault == "missing":
        frontend.ready.pop()
    elif fault == "identity":
        frontend.ready[1] = replace(
            frontend.ready[1], info=replace(frontend.infos[1], worker_id="replacement")
        )
    elif fault == "duplicate_endpoint":
        frontend.ready[1] = replace(frontend.ready[1], address=frontend.ready[0].address)
    elif fault == "key":
        frontend.ready[1] = replace(frontend.ready[1], address=("tcp://127.0.0.1:20001", b""))
    else:
        frontend.ready[1] = replace(frontend.ready[1], address=("tcp://127.0.0.2:20001", b"key"))
    with pytest.raises((RuntimeError, TimeoutError)):
        frontend.get_execution_controller()
    frontend.abort.assert_called_once()
    for version in (None, VERSION):
        with pytest.raises(RuntimeError, match="restart required"):
            frontend._execution_admission(version)
    with pytest.raises(RuntimeError, match="restart required"):
        frontend.get_execution_controller()
    assert frontend.call_all.call_count == 2


@pytest.mark.parametrize("failure", [False, True])
def test_untagged_admission_waits_for_complete_bootstrap(frontend, failure):
    entered, release, attempting, admitted = Event(), Event(), Event(), Event()

    def rpc(method, args, timeout):
        if method == "inspect_execution_bootstrap":
            return frontend.infos
        entered.set()
        assert release.wait(3)
        if failure:
            raise TimeoutError("bootstrap timed out")
        return frontend.ready

    def submit():
        attempting.set()
        with frontend._execution_admission(None):
            admitted.set()

    frontend.call_all.side_effect = rpc
    with ThreadPoolExecutor(max_workers=2) as pool:
        setup = pool.submit(frontend.get_execution_controller)
        assert entered.wait(3)
        ingress = pool.submit(submit)
        assert attempting.wait(3)
        assert not admitted.wait(0.05)
        release.set()
        if failure:
            with pytest.raises(TimeoutError):
                setup.result(timeout=3)
            with pytest.raises(RuntimeError, match="restart required"):
                ingress.result(timeout=3)
            assert not admitted.is_set()
        else:
            setup.result(timeout=3)
            ingress.result(timeout=3)
            assert admitted.is_set()


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("fault", [None, "install", "mailbox", "boundary"])
def test_worker_bootstrap_parks_all_ranks_before_installing(monkeypatch, size, fault):
    class TorchDist(SimpleNamespace):
        pass

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.distributed.communicator",
        SimpleNamespace(TorchDist=TorchDist),
    )
    from tensorrt_llm._torch.pyexecutor.execution_guard import ExecutionDeviceGuard

    monkeypatch.setattr(ExecutionDeviceGuard, "validate_executor", lambda *a, **kw: None)
    parked = Barrier(size)
    voting = Condition()
    votes = {}
    installed = set()
    resumed = []
    workers = []

    class Cohort:
        def __init__(self, session, rank, world):
            assert world == size

            def gather(vote):
                with voting:
                    votes[rank] = vote
                    voting.notify_all()
                    assert voting.wait_for(lambda: len(votes) == size, timeout=3)
                    return list(votes.values())

            self.consensus = ExecutionStepConsensus(session, rank, tuple(range(size)), gather)

    def server(table, bind_host):
        assert bind_host == "127.0.0.1"
        if fault == "mailbox" and table is workers[-1]._test_table:
            raise RuntimeError("mailbox failed")
        return SimpleNamespace(address=(f"tcp://127.0.0.1:{20000 + id(table) % 1000}", b"key"))

    monkeypatch.setattr(bootstrap, "GlooExecutionCohort", Cohort)
    monkeypatch.setattr(bootstrap, "ExecutionMailboxServer", server)

    for rank in range(size):
        worker = ExecutionControlMixin()
        worker._init_execution_control()
        worker.doing_shutdown = False
        worker._is_pytorch_backend = True
        worker.llm_args = SimpleNamespace(num_serve_frontends=1)
        worker._test_table = None
        engine = SimpleNamespace(
            dist=TorchDist(
                rank=rank, world_size=size, cluster_info=[("127.0.0.1", r) for r in range(size)]
            ),
            _execution_guard=None,
            is_shutdown=False,
            _sleep_wakeup_lock=RLock(),
        )

        @contextmanager
        def pause(*, drain, control_id, engine=engine, rank=rank):
            assert drain
            engine._active_control_id = control_id
            parked.wait(timeout=3)
            if fault == "boundary":
                engine._active_control_id = "other-action"
            try:
                yield
            finally:
                if fault is None:
                    assert len(installed) == size and len(votes) == size
                resumed.append(rank)

        def install(table, consensus, rank=rank, worker=worker, engine=engine):
            worker._test_table = table
            if fault == "install" and rank == size - 1:
                raise RuntimeError("guard setup failed")
            consensus.validate_membership("session", rank, tuple(range(size)))
            installed.add(rank)
            engine._execution_guard = SimpleNamespace(table=table)

        engine.control_action = pause
        engine.configure_execution_authority = install
        worker.engine = engine
        workers.append(worker)

    def start(rank):
        worker = workers[rank]
        if fault is not None:
            with pytest.raises(RuntimeError):
                bootstrap.start_execution_worker(worker, "session", size)
            with pytest.raises(RuntimeError, match="restart required"):
                worker._execution_admission(None)
            if worker._test_table is not None:
                with pytest.raises(RuntimeError, match="closed"):
                    worker._test_table.current_version("tenant", "scope")
        else:
            ready = bootstrap.start_execution_worker(worker, "session", size)
            assert ready.info.rank == rank
            assert worker._execution_version_table is worker._test_table

    with ThreadPoolExecutor(max_workers=size) as pool:
        list(pool.map(start, range(size)))
    assert set(resumed) == set(range(size))


@pytest.mark.parametrize("failure", [False, True])
def test_ray_shutdown_keeps_authority_and_collective_until_engine_stops(failure):
    from tensorrt_llm.executor.ray.gpu_worker import RayGPUWorker

    events = []

    def stop_engine():
        assert worker._execution_control_closed
        assert events == []
        events.append("engine stopped")
        if failure:
            raise RuntimeError("engine is still using the host group")

    worker = SimpleNamespace(
        _execution_host_collective=SimpleNamespace(
            close=lambda: events.append("collective closed")
        ),
        _execution_control_lock=RLock(),
        _execution_control_closed=False,
        _shutdown_execution_control=lambda: events.append("authority closed"),
        doing_shutdown=False,
        rank=0,
        engine=SimpleNamespace(shutdown=stop_engine),
        llm_args=SimpleNamespace(backend="pytorch"),
        _handle_background_error=Mock(),
    )
    if failure:
        with pytest.raises(RuntimeError, match="still using"):
            RayGPUWorker.shutdown(worker)
        assert events == ["engine stopped"]
        assert worker._execution_host_collective is not None
    else:
        RayGPUWorker.shutdown(worker)
        assert events == ["engine stopped", "authority closed", "collective closed"]
        assert worker._execution_host_collective is None and worker.engine is None
