# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only liveness checks for an established Ray execution cohort."""

from threading import Event, Thread
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .ray.gpu_worker import RayGPUWorker


def check_execution_worker(worker: "RayGPUWorker", session_id: str) -> tuple[str, int]:
    """Check the local engine without waiting for a CUDA stream or a peer rank."""
    with worker._execution_control_lock:
        worker._require_execution_control_open()
        table = worker._execution_version_table
        if table is None or table.session_id != session_id:
            raise RuntimeError("execution health check does not match worker authority")
        engine = worker.engine
        if engine is None:
            raise RuntimeError("execution worker has no engine")
        if engine._event_loop_error is not None:
            raise RuntimeError("execution worker event loop failed") from engine._event_loop_error
        if engine.is_shutdown or not engine.worker_thread.is_alive():
            raise RuntimeError("execution worker event loop stopped")
        with table._lock:
            table._require_open()
        return session_id, engine.dist.rank


class ExecutionCohortWatchdog:
    """Fail the frontend if any actor dies or its bounded health RPC fails.

    The caller must bound ``check_workers``. Health proves only host liveness;
    it is never FENCE evidence or permission to release KV holds.
    """

    def __init__(
        self,
        session_id: str,
        world_size: int,
        check_workers: Callable[[], list[tuple[str, int]]],
        fail: Callable[[Exception], object],
        *,
        interval: float = 1.0,
    ) -> None:
        self._expected = [(session_id, rank) for rank in range(world_size)]
        self._check_workers = check_workers
        self._fail = fail
        self._interval = interval
        self._stop = Event()
        self._thread = Thread(target=self._run, name="retire_ray_health", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        # A bounded Ray RPC may still be returning. Its result is ignored after
        # this signal, so normal shutdown cannot become a cohort failure.
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                if self._check_workers() != self._expected:
                    raise RuntimeError("execution health reply does not match the worker cohort")
            except Exception as error:
                if not self._stop.is_set():
                    self._fail(error)
                return
