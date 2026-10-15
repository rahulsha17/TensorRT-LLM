# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker loss must close authority without treating health as GPU drain evidence."""

from threading import Event, RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_mailbox import WorkerVersionTable
from tensorrt_llm.executor.execution_watchdog import ExecutionCohortWatchdog, check_execution_worker

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize(
    "fault", ["actor-dead", "timeout", "missing", "wrong-session", "wrong-rank"]
)
def test_cohort_failure_stops_monitor_and_reports_original_error(size, fault):
    original = TimeoutError("health deadline") if fault == "timeout" else OSError("actor died")
    replies = [("session", rank) for rank in range(size)]
    if fault == "missing":
        replies.pop()
    elif fault == "wrong-session":
        replies[-1] = ("previous-session", size - 1)
    elif fault == "wrong-rank":
        replies[-1] = ("session", 0)
    check = (
        Mock(side_effect=original)
        if fault in ("actor-dead", "timeout")
        else Mock(return_value=replies)
    )
    failed = Event()
    errors = []

    def fail(error):
        errors.append(error)
        failed.set()

    monitor = ExecutionCohortWatchdog("session", size, check, fail, interval=0.001)
    monitor.start()
    try:
        assert failed.wait(2)
        monitor._thread.join(1)
        assert not monitor._thread.is_alive()
        assert len(errors) == 1
        if fault in ("actor-dead", "timeout"):
            assert errors[0] is original
        else:
            assert "cohort" in str(errors[0])
        check.assert_called_once()
    finally:
        monitor.stop()


@pytest.mark.parametrize("error", [False, True])
def test_normal_stop_ignores_late_health_reply(error):
    checking, release = Event(), Event()
    fail = Mock()

    def check():
        checking.set()
        assert release.wait(2)
        if error:
            raise OSError("shutdown disconnected actor")
        return []

    monitor = ExecutionCohortWatchdog("session", 2, check, fail, interval=0.001)
    monitor.start()
    try:
        assert checking.wait(1)
        monitor.stop()
    finally:
        release.set()
        monitor._thread.join(2)
    fail.assert_not_called()


@pytest.mark.parametrize(
    "fault", [None, "session", "engine-error", "stopped", "dead-thread", "closed-table"]
)
def test_worker_health_uses_host_state_only(fault):
    table = WorkerVersionTable("session")
    engine = SimpleNamespace(
        _event_loop_error=RuntimeError("CUDA failure") if fault == "engine-error" else None,
        is_shutdown=fault == "stopped",
        worker_thread=SimpleNamespace(is_alive=lambda: fault != "dead-thread"),
        dist=SimpleNamespace(rank=1),
    )
    worker = SimpleNamespace(
        _execution_control_lock=RLock(),
        _require_execution_control_open=Mock(),
        _execution_version_table=table,
        engine=engine,
    )
    if fault == "closed-table":
        table.close()
    try:
        if fault is None:
            assert check_execution_worker(worker, "session") == ("session", 1)
        else:
            with pytest.raises(RuntimeError):
                check_execution_worker(worker, "other" if fault == "session" else "session")
    finally:
        table.close()


@pytest.mark.parametrize("size", [2, 4])
def test_ray_failure_kills_every_actor_even_if_one_kill_fails(monkeypatch, size):
    from tensorrt_llm.executor.execution_control import ExecutionControlMixin
    from tensorrt_llm.executor.ray import executor as ray_module

    owner = object.__new__(ray_module.RayExecutor)
    ExecutionControlMixin._init_execution_control(owner)
    owner.doing_shutdown = False
    owner._results = {}
    owner._set_fatal_error = Mock()
    owner.workers = [object() for _ in range(size)]
    owner._execution_watchdog = Mock()
    calls = []

    def kill(worker, *, no_restart):
        assert no_restart
        calls.append(worker)
        if worker is owner.workers[0]:
            raise RuntimeError("actor already gone")

    monkeypatch.setattr(ray_module.ray, "kill", kill)
    assert owner._fail_rpc_executor(OSError("rank lost"))
    assert calls == owner.workers
    owner._execution_watchdog.stop.assert_called_once()
    assert not owner._fail_rpc_executor(OSError("late stream failure"))
    assert calls == owner.workers
    with pytest.raises(RuntimeError, match="restart required"):
        owner._execution_admission(None)
