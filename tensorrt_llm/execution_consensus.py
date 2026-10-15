# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agree on RETIRE execution decisions before changing a shared model schedule."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .execution_version import ExecutionVersion

__all__ = ["ExecutionRow", "ExecutionStepConsensus"]


@dataclass(frozen=True)
class ExecutionRow:
    """Logical row identity and progress; physical KV and sequence slots stay local."""

    request_id: int
    version: ExecutionVersion | None
    token_count: int
    context_position: int
    context_chunk_size: int
    state: str
    last_token_id: int | None = None


@dataclass(frozen=True)
class _StepVote:
    session_id: str
    rank: int
    sequence: int
    phase: str
    rows: tuple[ExecutionRow, ...]
    allowed: tuple[bool, ...]
    error: str | None = None


class ExecutionStepConsensus:
    """Intersect authority decisions over a fixed set of lockstep model ranks.

    The engine thread owns this object. ``gather`` must be a host-only collective
    over exactly ``ranks`` with bounded failure handling. Every rank must call at
    the same execution boundary, including empty and untagged batches. This is
    not a rendezvous for independently progressing pipeline stages.

    Token/KV commits call while their local host-authority lock is held, after
    device completion. Early-exit checks use observations without that lock.
    No CUDA work may be submitted or waited for by the collective itself.

    A missing member, changed row/order/progress, mismatched boundary, or failed
    collective poisons the coordinator. Later calls cannot manufacture a drain
    proof or resume execution. Bootstrap and distributed failure recovery remain
    the executor's responsibility; consensus is not FENCE or a KV reservation.
    """

    def __init__(
        self,
        session_id: str,
        rank: int,
        ranks: tuple[int, ...],
        gather: Callable[[_StepVote], Sequence[_StepVote]],
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("consensus requires an authority session")
        if (
            not isinstance(ranks, tuple)
            or not ranks
            or any(type(r) is not int or r < 0 for r in ranks)
            or len(set(ranks)) != len(ranks)
            or type(rank) is not int
            or rank not in ranks
        ):
            raise ValueError("consensus requires fixed unique rank membership")
        if not callable(gather):
            raise TypeError("consensus requires a host collective")
        self._session_id = session_id
        self._rank = rank
        self._ranks = frozenset(ranks)
        self._gather = gather
        self._sequence = 0
        self._failure: Exception | None = None

    def fail(self, error: Exception) -> None:
        """Invalidate this endpoint after an executor error outside a vote."""
        if self._failure is None:
            self._failure = error

    def validate_membership(self, session_id: str, rank: int, ranks: tuple[int, ...]) -> None:
        """Bind a coordinator only to the worker and session it was created for."""
        if self._failure is not None:
            raise RuntimeError("execution consensus failed; restart required") from self._failure
        if (session_id, rank, frozenset(ranks)) != (self._session_id, self._rank, self._ranks):
            raise ValueError("execution coordinator does not match worker membership")

    @staticmethod
    def _validate_rows(rows: tuple[ExecutionRow, ...], allowed: tuple[bool, ...]) -> None:
        if (
            not isinstance(rows, tuple)
            or not isinstance(allowed, tuple)
            or len(rows) != len(allowed)
        ):
            raise ValueError("execution rows and authority verdicts must have equal lengths")
        identities = set()
        for row, accepted in zip(rows, allowed):
            if (
                not isinstance(row, ExecutionRow)
                or type(row.request_id) is not int
                or type(accepted) is not bool
                or row.version is not None
                and not isinstance(row.version, ExecutionVersion)
                or any(
                    type(n) is not int or n < 0
                    for n in (row.token_count, row.context_position, row.context_chunk_size)
                )
                or not isinstance(row.state, str)
                or not row.state
                or (row.token_count == 0 and row.last_token_id is not None)
                or (
                    row.token_count > 0
                    and (type(row.last_token_id) is not int or row.last_token_id < 0)
                )
            ):
                raise ValueError("invalid execution row or authority verdict")
            if row.request_id in identities:
                raise ValueError("duplicate logical request in execution step")
            identities.add(row.request_id)
            if row.version is None and not accepted:
                raise ValueError("authority cannot reject an untagged row")

    def agree(
        self,
        phase: str,
        rows: tuple[ExecutionRow, ...],
        allowed: tuple[bool, ...],
        *,
        local_error: Exception | None = None,
    ) -> tuple[bool, ...]:
        """Require the same step on every rank; one rejection excludes the row everywhere."""
        if self._failure is not None:
            raise RuntimeError("execution consensus failed; restart required") from self._failure
        error = None
        try:
            if local_error is not None:
                if not isinstance(local_error, Exception):
                    raise TypeError("local execution failure must be an exception")
                error = str(local_error)
            if not isinstance(phase, str) or not phase:
                raise ValueError("execution consensus requires a named boundary")
            self._validate_rows(rows, allowed)
        except (TypeError, ValueError) as invalid:
            # Enter the collective even for malformed local input so peers see
            # the failure instead of waiting for a rank that returned early.
            error = str(invalid)
            phase, rows, allowed = "invalid", (), ()
        vote = _StepVote(self._session_id, self._rank, self._sequence, phase, rows, allowed, error)
        self._sequence += 1
        try:
            votes = tuple(self._gather(vote))
            if len(votes) != len(self._ranks) or any(not isinstance(v, _StepVote) for v in votes):
                raise RuntimeError("execution consensus is missing required ranks")
            if {v.rank for v in votes} != self._ranks:
                raise RuntimeError("execution consensus has invalid rank membership")
            if any(v.error is not None for v in votes):
                raise RuntimeError("execution consensus received invalid local input")
            if any(
                (v.session_id, v.sequence, v.phase, v.rows)
                != (vote.session_id, vote.sequence, vote.phase, vote.rows)
                for v in votes
            ):
                raise RuntimeError("execution ranks disagree on session, step, rows or progress")
            if next(v for v in votes if v.rank == self._rank) != vote:
                raise RuntimeError("execution consensus replaced the local vote")
            for received in votes:
                self._validate_rows(received.rows, received.allowed)
            return tuple(all(v.allowed[i] for v in votes) for i in range(len(rows)))
        except Exception as failure:
            self._failure = failure
            raise
