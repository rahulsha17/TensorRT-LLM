# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-thread adapters that own real KV while a RETIRE successor is pending."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...execution_alignment import AlignmentInput, AlignmentSafetyError, PrefixReservation

__all__ = ["KVAlignmentAdapter"]

if TYPE_CHECKING:
    from .py_executor import PyExecutor


@dataclass
class _Reservation:
    num_tokens: int
    _release: Callable[[], None]
    _settle: Callable[[], None] = lambda: None
    _closed: bool = False

    def settle(self) -> None:
        self._settle()

    def close(self) -> None:
        if not self._closed:
            self._release()
            self._closed = True


class KVAlignmentAdapter:
    """Reserve compatible text prefixes using the executor's existing cache manager."""

    def __init__(self, executor: "PyExecutor") -> None:
        self._executor = executor
        self._manager = executor.kv_cache_manager

    def reserve(self, revised: AlignmentInput) -> PrefixReservation:
        """Return actual ownership, never merely a prefix probe or raw block IDs."""
        sequence_limit = self._executor.max_seq_len or self._manager.max_seq_len
        if len(revised.input_token_ids) > sequence_limit:
            raise ValueError("revised input exceeds max_seq_len")
        if not self._manager.enable_block_reuse or revised.reuse_limit == 0:
            return _Reservation(0, lambda: None)
        if self._executor._is_kv_manager_v2:
            return self._reserve_v2(revised)
        return self._reserve_v1(revised)

    def _reserve_v2(self, revised: AlignmentInput) -> PrefixReservation:
        from ...runtime.kv_cache_manager_v2 import ReuseScope

        manager = self._manager
        cache = manager.impl.create_kv_cache(
            ReuseScope(
                lora_id=revised.lora_task_id, salt=manager._derive_reuse_salt(revised.cache_salt)
            ),
            list(revised.input_token_ids[: revised.reuse_limit]),
            expected_prompt_length=len(revised.input_token_ids),
        )
        # Suspended caches retain native PageHolders without occupying an
        # executor slot. Pages may migrate between tiers but cannot be dropped.
        try:
            return _Reservation(
                int(cache.num_committed_tokens), cache.close, manager._stream.synchronize
            )
        except Exception:
            cache.close()
            raise

    def _reserve_v1(self, revised: AlignmentInput) -> PrefixReservation:
        from ...bindings.executor import SamplingConfig
        from ...bindings.internal.batch_manager import LlmRequest as CppLlmRequest

        manager = self._manager
        block_size = manager.tokens_per_block
        matched = manager.probe_prefix_match_length(
            list(revised.input_token_ids), revised.lora_task_id, revised.cache_salt
        )
        matched = min(matched, revised.reuse_limit // block_size * block_size)
        if matched == 0:
            return _Reservation(0, lambda: None)
        # A cache-only sequence acquires the same native references as an
        # ordinary request. One private tail block keeps the prefix immutable;
        # no model or sampler ever executes this sequence.
        tokens = list(revised.input_token_ids[: matched + 1])
        request_id = self._executor.executor_request_queue.reserve_request_id()
        request = CppLlmRequest(
            request_id=request_id,
            max_new_tokens=0,
            input_tokens=tokens,
            sampling_config=SamplingConfig(),
            is_streaming=False,
            lora_task_id=revised.lora_task_id,
            cache_salt=revised.cache_salt,
        )

        def release() -> None:
            # The binding requires a request. Mark it dummy only at release:
            # allocation must match cached blocks, but cleanup must never
            # publish this unexecuted sequence's private blocks for reuse.
            request.is_dummy_request = True
            manager.impl.remove_sequence(request_id, request, False)

        reservation = _Reservation(0, release, manager._stream.synchronize)
        try:
            manager.impl.sync_transfer_manager_with_buffer_manager()
            manager.impl.add_sequence_batch([(request_id, len(tokens), 1)], [request])
            manager.impl.refresh_blocks()
            manager._stream.synchronize()
            # Recheck the acquired result: pressure during allocation may have
            # changed what the advisory probe found.
            reservation.num_tokens = int(request.prepopulated_prompt_len)
            return reservation
        except Exception:
            try:
                manager.impl.refresh_blocks()
                manager._stream.synchronize()
                reservation.close()
            except Exception as cleanup_error:
                raise AlignmentSafetyError("V1 reservation cleanup is unknown") from cleanup_error
            raise
