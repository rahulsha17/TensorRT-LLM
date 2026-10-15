# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CPU processes and Gloo; model/GPU coordination is tested separately."""

import multiprocessing
import time
from datetime import timedelta

import pytest
import torch.distributed as dist

from tensorrt_llm.execution_consensus import ExecutionRow
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.execution_bootstrap import GlooExecutionCohort

pytestmark = pytest.mark.cpu_only


def _worker(rank, size, store, fault, results):
    cohort = None
    try:
        dist.init_process_group(
            "gloo", init_method=store, rank=rank, world_size=size, timeout=timedelta(seconds=30)
        )
        cohort = GlooExecutionCohort("session", rank, size, timeout=2)
        assert dist.get_backend(cohort._group) == "gloo"
        assert cohort.consensus.agree("bootstrap_ready", (), ()) == ()
        row = ExecutionRow(7, ExecutionVersion("tenant", "scope", 1), 64, 0, 64, "context", 9)
        if fault == "missing" and rank == 1:
            time.sleep(4)
        elif fault:
            with pytest.raises(RuntimeError):
                cohort.consensus.agree(
                    "other" if fault == "phase" and rank else "step", (row,), (True,)
                )
            with pytest.raises(RuntimeError, match="restart required"):
                cohort.consensus.agree("retry", (), ())
        else:
            assert cohort.consensus.agree("step", (row,), (rank == 0,)) == (False,)
            assert cohort.consensus.agree("empty", (), ()) == ()
        cohort.close()
        cohort.close()
        with pytest.raises(RuntimeError, match="restart required"):
            cohort.consensus.agree("after_close", (), ())
        results.put((rank, True, None))
    except BaseException as error:
        results.put((rank, False, repr(error)))
        raise
    finally:
        if cohort is not None:
            cohort.close()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("size,fault", [(2, None), (4, None), (2, "phase"), (2, "missing")])
def test_real_host_collective_membership_and_failure(tmp_path, size, fault):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(
            target=_worker, args=(rank, size, (tmp_path / "store").as_uri(), fault, results)
        )
        for rank in range(size)
    ]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 90
        reports = [results.get(timeout=max(0.01, deadline - time.monotonic())) for _ in workers]
        for worker in workers:
            worker.join(timeout=max(0.01, deadline - time.monotonic()))
        assert {rank for rank, _, _ in reports} == set(range(size))
        assert all(passed for _, passed, _ in reports), reports
        assert all(worker.exitcode == 0 for worker in workers)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            if worker.pid is not None:
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=5)
        results.close()
