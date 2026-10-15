# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy execution-authority wiring for the direct and single-worker IPC executors."""

from contextlib import AbstractContextManager, nullcontext
from threading import RLock
from time import monotonic_ns
from typing import TYPE_CHECKING
from uuid import uuid4

from ..execution_gate import ExecutionAuthorityError, HostExecution
from ..execution_mailbox import (
    ExecutionController,
    LocalPublicationTransport,
    PublicationTransport,
    WorkerVersionTable,
)
from ..execution_version import ExecutionVersion
from .execution_mailbox import ExecutionMailboxServer, IpcPublicationTransport

__all__ = ["ExecutionControlMixin", "ProxyExecutionControlMixin", "WorkerExecutionControlMixin"]

if TYPE_CHECKING:
    from .result import GenerationResult


class ExecutionControlMixin:
    """Own one lazily created controller per frontend executor."""

    def _init_execution_control(self) -> None:
        self._execution_control_lock = RLock()
        self._execution_control_closed = False
        self._execution_control_error: Exception | None = None
        self._execution_submit_started = False
        self._execution_session_id = uuid4().hex
        self._execution_controller: ExecutionController | None = None
        self._execution_shared = False
        self._execution_version_table: WorkerVersionTable | None = None
        self._execution_mailbox_server: ExecutionMailboxServer | None = None

    def get_execution_controller(self) -> ExecutionController:
        """Get host authority for an explicitly supported executor topology.

        Register and revise scopes through this controller to publish accepted
        versions automatically. Access starts the control path on first use.
        Tagged submissions require registered authority. Host enqueue and result
        delivery are gated; GPU and transfer quiescence are separate work.
        """
        with self._execution_control_lock:
            self._require_execution_control_open()
            if self._execution_controller is None:
                self._execution_controller = self._create_execution_controller()
            return self._execution_controller

    def _require_execution_control_open(self) -> None:
        if self._execution_control_error is not None:
            raise RuntimeError(
                "executor execution control failed; restart required"
            ) from self._execution_control_error
        if self._execution_control_closed or self.doing_shutdown:
            raise RuntimeError("executor execution control is closed")
        if (
            self._execution_shared
            and self._execution_controller is not None
            and self._execution_controller.closed
        ):
            raise RuntimeError("shared execution authority is closed; restart required")

    def _create_execution_controller(self) -> ExecutionController:
        transport = self._create_execution_transport(self._execution_session_id)
        return ExecutionController(self._execution_session_id, transport)

    def _create_execution_transport(self, session_id: str) -> PublicationTransport:
        raise NotImplementedError("execution publication is not supported by this executor")

    def _execution_admission(
        self, version: ExecutionVersion | None
    ) -> AbstractContextManager[HostExecution | None]:
        with self._execution_control_lock:
            self._require_execution_control_open()
            if version is None:
                self._execution_submit_started = True
                return nullcontext(None)
            owner = self._execution_controller or self._execution_version_table
            if owner is None:
                raise ExecutionAuthorityError(
                    "register execution authority before submitting tagged work"
                )
            self._execution_submit_started = True
        return owner.admission(version)

    @staticmethod
    def _bind_execution_result(result: "GenerationResult", execution: HostExecution | None) -> None:
        # Called under admission, before any request can be revoked.
        result._execution = execution
        if execution is not None:
            execution._wake = result.queue.put

    def _shutdown_execution_control(self) -> None:
        # Shutdown can run after only part of executor initialization completed.
        if not hasattr(self, "_execution_control_lock"):
            return
        with self._execution_control_lock:
            self._execution_control_closed = True
            controller = self._execution_controller
            server = self._execution_mailbox_server
            table = self._execution_version_table
        if controller is not None:
            controller.close()
        if server is not None:
            server.close()
        elif table is not None:
            table.close()


class WorkerExecutionControlMixin:
    """Host the reader in a single PyTorch worker, with one authority owner."""

    def _check_execution_publication_support(self) -> None:
        self._require_execution_control_open()
        if (
            not self._is_pytorch_backend
            or self.llm_args.parallel_config.world_size != 1
            or self.llm_args.num_serve_frontends != 1
        ):
            raise NotImplementedError(
                "execution publication requires one frontend and one PyTorch rank"
            )
        if self.engine is None:
            raise RuntimeError("execution publication requires a running worker")

    def _bind_execution_version_table(self, session_id: str) -> WorkerVersionTable:
        self._check_execution_publication_support()
        if self._execution_version_table is None:
            self._execution_version_table = self._create_execution_version_table(session_id)
        elif self._execution_version_table.session_id != session_id:
            raise RuntimeError("worker already has a different authority owner")
        return self._execution_version_table

    def _create_execution_version_table(self, session_id: str) -> WorkerVersionTable:
        """Create host storage; concrete GPU workers also bind their device publisher."""
        return WorkerVersionTable(session_id)

    def get_execution_version_table(self) -> WorkerVersionTable:
        """Read the host table populated by this worker's authority owner."""
        with self._execution_control_lock:
            self._require_execution_control_open()
            if self._execution_version_table is None:
                raise RuntimeError("execution publication has not been initialized")
            return self._execution_version_table

    def _create_execution_transport(self, session_id: str) -> PublicationTransport:
        table = self._bind_execution_version_table(session_id)
        return LocalPublicationTransport(table)

    def start_execution_mailbox(self, session_id: str) -> tuple[str, bytes]:
        """Bootstrap the dedicated publication channel over the existing authenticated RPC."""
        with self._execution_control_lock:
            table = self._bind_execution_version_table(session_id)
            if self._execution_mailbox_server is None:
                self._execution_mailbox_server = ExecutionMailboxServer(table)
            self._execution_mailbox_server.check_health()
            return self._execution_mailbox_server.address

    def fetch_execution_reservations(self) -> dict:
        """Observe RETIRE holds through worker RPC without creating a controller.

        This reports logical ownership, not cache residency or a reusable
        receipt. Sampling does not release holds or drive engine cleanup.
        """
        start = monotonic_ns()
        with self._execution_control_lock:
            self._require_execution_control_open()
            table = self._execution_version_table
            state = table.alignment_state if table is not None else None
        snapshot = state.snapshot() if state is not None else {}
        return {
            "configured": table is not None,
            "alignment_configured": state is not None,
            "sample_start_ns": start,
            "sample_end_ns": monotonic_ns(),
            **snapshot,
        }


class ProxyExecutionControlMixin:
    """Keep authority at the sole frontend and publish to its IPC worker."""

    def _create_execution_transport(self, session_id: str) -> PublicationTransport:
        if self._num_frontends != 1 or not self._is_pytorch_backend or self.model_world_size != 1:
            raise NotImplementedError(
                "execution publication requires one frontend and one PyTorch rank"
            )
        if not self.workers_started or self._engine_dead or self.rpc_client is None:
            raise RuntimeError("execution publication requires a running worker")
        address = self.rpc_client.start_execution_mailbox(session_id).remote(timeout=5)
        return IpcPublicationTransport(address)
