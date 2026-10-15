# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned exchanges cannot substitute input or model/cache semantics."""

from dataclasses import FrozenInstanceError, replace

import msgpack
import pytest

from tensorrt_llm._torch.disaggregation.native.execution_content import TransferPrefixIdentity
from tensorrt_llm._torch.disaggregation.native.execution_wire import (
    ExecutionTransferWire,
    MessageType,
    TransferExecutionIdentity,
    TransferIdentityError,
)
from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 2)
PREFIX = TransferPrefixIdentity.from_tokens("weights-and-format-v1", (1, 2, 3))


def _wire(prefix=PREFIX):
    table = WorkerVersionTable("session")
    table.apply(VersionUpdate("session", 1, VERSION))
    wire = ExecutionTransferWire()
    wire.bind(table)
    wire.register_local(91, "receive", VERSION, "salt", prefix)
    return wire, table


def test_default_positions_are_canonical_and_identity_is_immutable():
    assert PREFIX == TransferPrefixIdentity.from_tokens(
        "weights-and-format-v1", (1, 2, 3), position_ids=(0, 1, 2)
    )
    assert TransferPrefixIdentity.from_values(PREFIX.to_values()) == PREFIX
    with pytest.raises(FrozenInstanceError):
        PREFIX.num_tokens = 4


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cache_context_id="another-weights-revision"),
        dict(tokens=(1, 2, 4)),
        dict(tokens=(1, 2, 3, 4)),
        dict(position_ids=(1, 2, 3)),
        dict(lora_task_id=0),
        dict(lora_task_id=(1 << 64) - 1),
    ],
)
def test_input_semantics_are_distinct(kwargs):
    values = dict(cache_context_id="weights-and-format-v1", tokens=(1, 2, 3))
    values.update(kwargs)
    assert TransferPrefixIdentity.from_tokens(**values) != PREFIX


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cache_context_id=None),
        dict(cache_context_id=""),
        dict(cache_context_id=17),
        dict(tokens=[]),
        dict(tokens=()),
        dict(tokens=(True,)),
        dict(tokens=(-1,)),
        dict(tokens=(1 << 31,)),
        dict(tokens=("1",)),
        dict(position_ids=[0, 1, 2]),
        dict(position_ids=(0, 1)),
        dict(position_ids=(0, 1, True)),
        dict(position_ids=(0, 1, -1)),
        dict(lora_task_id=True),
        dict(lora_task_id=-1),
        dict(lora_task_id=1 << 64),
    ],
)
def test_invalid_prefix_input_is_rejected(kwargs):
    values = dict(cache_context_id="weights-and-format-v1", tokens=(1, 2, 3))
    values.update(kwargs)
    with pytest.raises(ValueError):
        TransferPrefixIdentity.from_tokens(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_tokens", 0),
        ("num_tokens", True),
        ("num_tokens", 1 << 31),
        ("token_digest", b""),
        ("token_digest", "a" * 32),
        ("position_digest", bytearray(32)),
        ("position_digest", b"a" * 33),
    ],
)
def test_wire_values_cannot_bypass_validation(field, value):
    with pytest.raises(ValueError):
        replace(PREFIX, **{field: value})


def test_large_prefix_uses_bounded_wire_metadata():
    content = TransferPrefixIdentity.from_tokens("weights", tuple(range(32768)))
    identity = TransferExecutionIdentity("session", VERSION, 91, None, content)
    assert len(identity.pack()) < 256
    assert TransferExecutionIdentity.unpack(identity.pack()) == identity


def test_tagged_identity_without_prefix_cannot_downgrade():
    wire, _ = _wire()
    with pytest.raises(TransferIdentityError, match="prefix identity"):
        wire.register_local(92, "send", VERSION, None)
    with pytest.raises(TransferIdentityError, match="discard"):
        wire.register_local(92, "send", None, None, PREFIX)
    old = [1, "session", "tenant", "scope", 2, 91, "salt"]
    with pytest.raises(TransferIdentityError, match="invalid transfer"):
        TransferExecutionIdentity.unpack(msgpack.packb(old, use_bin_type=True))


@pytest.mark.parametrize(
    "changed",
    [
        replace(PREFIX, cache_context_id="other"),
        replace(PREFIX, token_digest=b"x" * 32),
        replace(PREFIX, position_digest=b"x" * 32),
        replace(PREFIX, num_tokens=2),
        replace(PREFIX, lora_task_id=7),
    ],
)
def test_peer_prefix_constrains_local_session_before_admission(changed):
    sender, _ = _wire()
    receiver = ExecutionTransferWire()
    receiver.bind(WorkerVersionTable("session"))
    message = [MessageType.REQUEST_DATA, msgpack.packb({"unique_rid": 91})]
    assert receiver.decode(sender.encode(message)) == message
    with pytest.raises(TransferIdentityError, match="bound to another execution"):
        receiver.register_local(91, "send", VERSION, "salt", changed)
    receiver.register_local(91, "send", VERSION, "salt", PREFIX)


def test_matching_prefix_evidence_can_drain_after_revocation():
    wire, table = _wire()
    message = [MessageType.SESSION_QUIESCED, b"0", b"91"]
    encoded = wire.encode(message)
    table.apply(VersionUpdate("session", 2, replace(VERSION, epoch=3)))
    table.close()
    assert wire.decode(encoded) == message
    assert wire.encode(message) == encoded
