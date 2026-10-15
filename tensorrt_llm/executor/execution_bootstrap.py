# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install RETIRE on a paused, fixed set of Ray tensor-parallel workers."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from ipaddress import ip_address
from typing import TYPE_CHECKING

from ..execution_collective import CollectivePublicationTransport, ExecutionParticipant
from ..execution_consensus import ExecutionStepConsensus
from ..execution_mailbox import ExecutionController, WorkerVersionTable
from .execution_mailbox import ExecutionMailboxServer, IpcPublicationTransport

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from ..execution_consensus import _StepVote
    from .ray.executor import RayExecutor
    from .ray.gpu_worker import RayGPUWorker

__all__ = [
    "ExecutionBootstrapInfo",
    "ExecutionBootstrapReady",
    "GlooExecutionCohort",
    "inspect_execution_worker",
    "start_execution_worker",
    "create_execution_cohort",
]


@dataclass(frozen=True)
class ExecutionBootstrapInfo:
    session_id: str
    worker_id: str
    rank: int
    world_size: int
    host: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.session_id, str)
            or not self.session_id
            or not isinstance(self.worker_id, str)
            or not self.worker_id
            or type(self.rank) is not int
            or type(self.world_size) is not int
            or not 0 <= self.rank < self.world_size
            or self.world_size < 2
        ):
            raise ValueError("invalid execution bootstrap identity")
        host = ip_address(self.host)
        if host.version != 4 or host.is_unspecified or host.is_multicast:
            raise ValueError("execution bootstrap requires a concrete IPv4 worker address")


@dataclass(frozen=True)
class ExecutionBootstrapReady:
    info: ExecutionBootstrapInfo
    address: tuple[str, bytes]


class GlooExecutionCohort:
    """A separate CPU-only process group, created while model threads are paused.

    The RPC thread uses consensus for bootstrap readiness, then hands it to the
    engine thread. Group creation and collectives have bounded timeouts. The
    caller must kill the full worker set after a failed or timed-out bootstrap.
    Close only after the engine thread has stopped using the group.
    """

    def __init__(
        self, session_id: str, rank: int, world_size: int, *, timeout: float = 300.0
    ) -> None:
        import torch.distributed as dist

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("execution collective timeout must be positive and finite")
        self._dist = dist
        self._world_size = world_size
        self.consensus = ExecutionStepConsensus(
            session_id, rank, tuple(range(world_size)), self._gather
        )
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise ValueError("execution group does not match the default worker membership")
        self._group: ProcessGroup | None = dist.new_group(
            ranks=list(range(world_size)), backend="gloo", timeout=timedelta(seconds=timeout)
        )

    def _gather(self, vote: "_StepVote") -> "Sequence[_StepVote]":
        if self._group is None:
            raise RuntimeError("execution host collective is closed")
        values = [None] * self._world_size
        self._dist.all_gather_object(values, vote, group=self._group)
        return values

    def close(self) -> None:
        if self._group is not None:
            group, self._group = self._group, None
            self.consensus.fail(RuntimeError("execution host collective is closed"))
            self._dist.destroy_process_group(group)


def inspect_execution_worker(
    worker: "RayGPUWorker", session_id: str, world_size: int
) -> ExecutionBootstrapInfo:
    """Read-only preflight; unsupported workers must fail before any engine parks."""
    from .._torch.distributed.communicator import TorchDist
    from .._torch.pyexecutor.execution_guard import ExecutionDeviceGuard

    with worker._execution_control_lock:
        worker._require_execution_control_open()
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("execution bootstrap requires a session")
        if (
            not worker._is_pytorch_backend
            or worker.llm_args.num_serve_frontends != 1
            or worker.engine is None
            or not isinstance(worker.engine.dist, TorchDist)
        ):
            raise NotImplementedError("distributed RETIRE requires Ray PyTorch workers")
        engine = worker.engine
        if type(world_size) is not int or world_size < 2 or world_size != engine.dist.world_size:
            raise ValueError("execution bootstrap does not match model membership")
        if worker._execution_submit_started or worker._execution_version_table is not None:
            raise RuntimeError("distributed execution control must start before request admission")
        if engine._execution_guard is not None or engine.is_shutdown:
            raise RuntimeError("execution worker is not available for bootstrap")
        ExecutionDeviceGuard.validate_executor(engine, tensor_parallel=True)
        rank = engine.dist.rank
        host = str(ip_address(engine.dist.cluster_info[rank][0]))
        if ip_address(host).version != 4:
            raise NotImplementedError("execution mailbox requires an IPv4 worker address")
        return ExecutionBootstrapInfo(
            session_id, worker._execution_session_id, rank, world_size, host
        )


def start_execution_worker(
    worker: "RayGPUWorker", session_id: str, world_size: int
) -> ExecutionBootstrapReady:
    """Park the engine, install local authority, and require every peer's readiness."""
    with worker._execution_control_lock:
        info = inspect_execution_worker(worker, session_id, world_size)
        engine = worker.engine
        table = WorkerVersionTable(session_id)
        control_id = f"retire:{session_id}"
        try:
            with (
                engine._sleep_wakeup_lock,
                engine.control_action(drain=True, control_id=control_id),
            ):
                if engine._active_control_id != control_id:
                    raise RuntimeError("execution bootstrap reached a different control boundary")
                cohort = GlooExecutionCohort(session_id, info.rank, world_size)
                worker._execution_host_collective = cohort
                local_error = None
                try:
                    engine.configure_execution_authority(table, cohort.consensus)
                    worker._execution_mailbox_server = ExecutionMailboxServer(
                        table, bind_host=info.host
                    )
                    worker._execution_version_table = table
                except Exception as error:
                    local_error = error
                # No engine resumes until every peer has installed its guards.
                cohort.consensus.agree("bootstrap_ready", (), (), local_error=local_error)
                return ExecutionBootstrapReady(info, worker._execution_mailbox_server.address)
        except Exception as error:
            worker._execution_control_error = error
            table.close()
            # The frontend's bounded RPC wait kills every actor on failure.
            raise


def create_execution_cohort(
    frontend: "RayExecutor",
    call_all: Callable[[str, tuple, float], list],
    abort_workers: Callable[[], None],
) -> ExecutionController:
    """Bootstrap before admission under the frontend's execution-control lock.

    ``call_all`` must start every RPC before waiting and enforce the supplied
    deadline. ``abort_workers`` must attempt to kill every actor, even if some
    are already dead. A partial setup permanently closes frontend admission.
    """
    args = frontend.worker_kwargs["llm_args"]
    parallel = args.parallel_config
    size = frontend.world_size
    if (
        args.backend != "pytorch"
        or args.num_serve_frontends != 1
        or size < 2
        or parallel.world_size != size
        or parallel.tp_size != size
        or parallel.pp_size != 1
        or parallel.cp_size != 1
        or parallel.enable_attention_dp
    ):
        raise NotImplementedError("distributed RETIRE requires one frontend and tensor parallelism")
    if frontend._execution_submit_started:
        raise RuntimeError("distributed execution control must start before request admission")
    if frontend._shutdown_event.is_set() or len(frontend.workers) != size:
        raise RuntimeError("execution bootstrap requires the full running worker set")
    session = frontend._execution_session_id
    infos = call_all("inspect_execution_bootstrap", (session, size), 30.0)
    if (
        len(infos) != size
        or any(not isinstance(info, ExecutionBootstrapInfo) for info in infos)
        or any(
            (info.session_id, info.rank, info.world_size) != (session, rank, size)
            for rank, info in enumerate(infos)
        )
        or any(not info.worker_id for info in infos)
        or len({info.worker_id for info in infos}) != size
    ):
        raise RuntimeError("execution bootstrap preflight has invalid worker membership")
    try:
        ready = call_all("start_execution_bootstrap", (session, size), 90.0)
        if len(ready) != size or any(
            not isinstance(reply, ExecutionBootstrapReady) or reply.info != info
            for reply, info in zip(ready, infos)
        ):
            raise RuntimeError("execution bootstrap did not acknowledge every prepared worker")
        addresses = [reply.address for reply in ready]
        if any(
            not isinstance(address, tuple)
            or len(address) != 2
            or not isinstance(address[0], str)
            or not isinstance(address[1], bytes)
            or not address[1]
            for address in addresses
        ):
            raise RuntimeError("execution bootstrap returned an unauthenticated endpoint")
        if len({address[0] for address in addresses}) != size:
            raise RuntimeError("execution bootstrap returned duplicate worker endpoints")
        for info, address in zip(infos, addresses):
            prefix = f"tcp://{info.host}:"
            port = address[0].removeprefix(prefix)
            if (
                not address[0].startswith(prefix)
                or not port.isdecimal()
                or not 0 < int(port) < 65536
            ):
                raise RuntimeError("execution endpoint does not match its prepared worker")
        participants = tuple(ExecutionParticipant(rank, 0) for rank in range(size))
        transport = CollectivePublicationTransport(
            participants,
            {p: IpcPublicationTransport(address) for p, address in zip(participants, addresses)},
            publication_source=participants[0],
        )
        return ExecutionController(session, transport)
    except Exception as error:
        frontend._execution_control_error = error
        abort_workers()
        raise
