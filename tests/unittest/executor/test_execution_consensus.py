# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise fixed-membership execution votes, including asymmetric publications."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Condition
from time import monotonic
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_consensus import ExecutionRow, ExecutionStepConsensus
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 7)
ROW = ExecutionRow(42, VERSION, 64, 0, 64, "CONTEXT_INIT", 7)


def test_external_failure_is_permanent_and_preserves_first_cause():
    gather = Mock()
    consensus = ExecutionStepConsensus("session", 0, (0, 1), gather)
    original = RuntimeError("executor failed outside a vote")
    consensus.fail(original)
    consensus.fail(ValueError("later cleanup failure"))
    with pytest.raises(RuntimeError, match="restart required") as caught:
        consensus.agree("next", (), ())
    assert caught.value.__cause__ is original
    gather.assert_not_called()


class GatherGroup:
    """A bounded host collective with independent per-endpoint invocation order."""

    def __init__(self, size):
        self.size = size
        self.condition = Condition()
        self.rounds = {}

    def endpoint(self, rank):
        invocation = 0

        def gather(vote):
            nonlocal invocation
            with self.condition:
                round_id = invocation
                invocation += 1
                values = self.rounds.setdefault(round_id, {})
                assert rank not in values
                values[rank] = vote
                self.condition.notify_all()
                deadline = monotonic() + 3
                while len(values) != self.size:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError("test rank did not reach collective")
                    self.condition.wait(remaining)
                return [values[r] for r in range(self.size)]

        return gather


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize(
    "phase", ["schedule", "context_commit", "token_commit", "early_exit:layer:3:64"]
)
def test_one_rank_revocation_rejects_same_row_everywhere(size, phase):
    group = GatherGroup(size)
    rows = (ROW, replace(ROW, request_id=43, version=None))
    ranks = tuple(range(size))
    owners = [
        ExecutionStepConsensus("session", rank, ranks, group.endpoint(rank)) for rank in ranks
    ]

    def run(rank):
        current = owners[rank].agree(phase, rows, (True, True))
        revoked = owners[rank].agree(phase, rows, (rank != size - 1, True))
        empty = owners[rank].agree("schedule", (), ())
        return current, revoked, empty

    with ThreadPoolExecutor(max_workers=size) as pool:
        results = list(pool.map(run, ranks))
    assert results == [((True, True), (False, True), ())] * size


@pytest.mark.parametrize(
    "mismatch", ["session", "phase", "request", "version", "position", "order", "empty", "token"]
)
def test_asymmetric_steps_fail_on_every_rank(mismatch):
    group = GatherGroup(2)
    owners = [
        ExecutionStepConsensus(
            "other" if mismatch == "session" and rank else "session",
            rank,
            (0, 1),
            group.endpoint(rank),
        )
        for rank in (0, 1)
    ]
    rows = (ROW, replace(ROW, request_id=43))
    other = rows
    if mismatch == "request":
        other = (replace(ROW, request_id=99), rows[1])
    elif mismatch == "version":
        other = (replace(ROW, version=replace(VERSION, epoch=8)), rows[1])
    elif mismatch == "position":
        other = (replace(ROW, context_position=64), rows[1])
    elif mismatch == "token":
        other = (replace(ROW, last_token_id=8), rows[1])
    elif mismatch == "order":
        other = tuple(reversed(rows))
    elif mismatch == "empty":
        other = ()

    def run(rank):
        local = other if rank else rows
        phase = "token_commit" if mismatch == "phase" and rank else "context_commit"
        with pytest.raises(RuntimeError, match="disagree"):
            owners[rank].agree(phase, local, (True,) * len(local))
        with pytest.raises(RuntimeError, match="restart required"):
            owners[rank].agree("schedule", rows, (True, True))

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, (0, 1)))


@pytest.mark.parametrize(
    "bad_rows,bad_allowed",
    [
        ((ROW,), ()),
        ((ROW,), (1,)),
        ((ROW, ROW), (True, True)),
        ((replace(ROW, version=None),), (False,)),
        ((replace(ROW, token_count=-1),), (True,)),
        ((replace(ROW, last_token_id=None),), (True,)),
        ((replace(ROW, last_token_id=-1),), (True,)),
        ((replace(ROW, token_count=0),), (True,)),
    ],
)
def test_invalid_local_input_is_reported_to_peers(bad_rows, bad_allowed):
    group = GatherGroup(2)
    owners = [
        ExecutionStepConsensus("session", rank, (0, 1), group.endpoint(rank)) for rank in (0, 1)
    ]

    def run(rank):
        with pytest.raises(RuntimeError, match="invalid local input"):
            owners[rank].agree(
                "schedule", bad_rows if rank else (ROW,), bad_allowed if rank else (True,)
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, (0, 1)))


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "foreign", "sequence", "local", "type", "timeout"]
)
def test_transport_or_membership_failure_is_permanent(fault):
    def gather(vote):
        peer = replace(vote, rank=1)
        if fault == "missing":
            return [vote]
        if fault == "duplicate":
            return [vote, vote]
        if fault == "foreign":
            return [vote, replace(peer, rank=2)]
        if fault == "sequence":
            return [vote, replace(peer, sequence=vote.sequence + 1)]
        if fault == "local":
            return [replace(vote, allowed=(False,)), peer]
        if fault == "type":
            return [vote, None]
        raise TimeoutError("transport did not complete")

    collective = Mock(side_effect=gather)
    owner = ExecutionStepConsensus("session", 0, (0, 1), collective)
    with pytest.raises((RuntimeError, TimeoutError)):
        owner.agree("schedule", (ROW,), (True,))
    with pytest.raises(RuntimeError, match="restart required"):
        owner.agree("schedule", (ROW,), (True,))
    collective.assert_called_once()


@pytest.mark.parametrize(
    "rank,ranks", [(0, ()), (0, (0, 0)), (2, (0, 1)), (True, (0, 1)), (0, (0, -1)), (0, [0, 1])]
)
def test_invalid_membership_rejected_before_transport(rank, ranks):
    gather = Mock()
    with pytest.raises(ValueError, match="membership"):
        ExecutionStepConsensus("session", rank, ranks, gather)
    gather.assert_not_called()
