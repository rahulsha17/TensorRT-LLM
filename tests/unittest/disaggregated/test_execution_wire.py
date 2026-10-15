# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer identity survives revocation, delayed completion and local cleanup."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import pytest

from tensorrt_llm._torch.disaggregation.native import transfer
from tensorrt_llm._torch.disaggregation.native.execution_content import TransferPrefixIdentity
from tensorrt_llm._torch.disaggregation.native.execution_wire import (
    KV_RESULT_PREFIX,
    ExecutionTransferWire,
    MessageType,
    TransferExecutionIdentity,
    TransferIdentityError,
)
from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 7)
CONTENT = TransferPrefixIdentity.from_tokens("model-cache-v1", (1, 2, 3))
IDENTITY = TransferExecutionIdentity("authority", VERSION, 99, "cache", CONTENT)


def _body(kind: bytes, rid: int = 99) -> list[bytes]:
    if kind == MessageType.REQUEST_DATA:
        return [kind, msgpack.packb({"unique_rid": rid, "slice_id": 0}, use_bin_type=True)]
    if kind == MessageType.KV_AGENT_RESULT:
        return [kind, KV_RESULT_PREFIX.pack(0, rid, 1, True, 0, 4096), b"bounce", b"tail"]
    if kind == MessageType.AUX_AGENT_RESULT:
        return [kind, b"0", str(rid).encode(), b"SUCCESS"]
    if kind == MessageType.CANCEL_SESSION:
        return [kind, str(rid).encode(), b"0", b"endpoint"]
    return [MessageType.SESSION_QUIESCED, b"0", str(rid).encode()]


KINDS = [
    MessageType.REQUEST_DATA,
    MessageType.KV_AGENT_RESULT,
    MessageType.AUX_AGENT_RESULT,
    MessageType.CANCEL_SESSION,
    MessageType.SESSION_QUIESCED,
]


def _wire(
    *, register: bool = True, version: ExecutionVersion | None = VERSION, salt: str | None = "cache"
) -> tuple[ExecutionTransferWire, WorkerVersionTable]:
    table = WorkerVersionTable("authority")
    table.apply(VersionUpdate("authority", 1, VERSION))
    wire = ExecutionTransferWire()
    wire.bind(table)
    if register:
        wire.register_local(99, "receive", version, salt, CONTENT if version is not None else None)
    return wire, table


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("salt", [None, "", "cache", "租户"])
def test_round_trip_preserves_body_and_identity(kind, salt):
    wire, _ = _wire(salt=salt)
    message = _body(kind)
    encoded = wire.encode(message)
    assert encoded[2:] == message
    assert TransferExecutionIdentity.unpack(encoded[1]) == replace(IDENTITY, cache_salt=salt)
    assert wire.decode(encoded) == message
    assert message == _body(kind)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("state", ["revoked", "closed"])
def test_matching_old_evidence_remains_attributable_after_authority_ends(kind, state):
    wire, table = _wire()
    encoded = wire.encode(_body(kind))
    successor = replace(VERSION, epoch=8)
    table.apply(VersionUpdate("authority", 2, successor))
    if state == "closed":
        table.close()
    assert wire.decode(encoded) == _body(kind)
    assert wire.encode(_body(kind)) == encoded
    if state == "revoked":
        assert table.current_version("tenant", "scope") == successor
    # Matching identity is not permission to reuse the exchange for a successor.
    with pytest.raises(TransferIdentityError):
        wire.register_local(99, "receive", successor, "cache", CONTENT)


@pytest.mark.parametrize(
    "other",
    [
        replace(IDENTITY, authority_session="foreign"),
        replace(IDENTITY, version=replace(VERSION, tenant_id="foreign")),
        replace(IDENTITY, version=replace(VERSION, scope_id="foreign")),
        replace(IDENTITY, version=replace(VERSION, epoch=8)),
        replace(IDENTITY, transfer_id=100),
        replace(IDENTITY, cache_salt=None),
        replace(IDENTITY, prefix=replace(CONTENT, cache_context_id="another-model")),
        replace(IDENTITY, prefix=replace(CONTENT, num_tokens=2)),
        replace(IDENTITY, prefix=replace(CONTENT, token_digest=b"a" * 32)),
        replace(IDENTITY, prefix=replace(CONTENT, position_digest=b"a" * 32)),
        replace(IDENTITY, prefix=replace(CONTENT, lora_task_id=7)),
    ],
)
@pytest.mark.parametrize("kind", KINDS)
def test_mismatch_rejected_without_changing_original_exchange(other, kind):
    wire, _ = _wire()
    encoded = wire.encode(_body(kind))
    bad = [encoded[0], other.pack(), *encoded[2:]]
    with pytest.raises(TransferIdentityError):
        wire.decode(bad)
    assert wire.decode(encoded) == _body(kind)


@pytest.mark.parametrize("kind", KINDS)
def test_tagged_exchange_rejects_stripped_envelope(kind):
    wire, _ = _wire()
    with pytest.raises(TransferIdentityError):
        wire.decode(_body(kind))


@pytest.mark.parametrize("first_tagged", [False, True])
def test_tagged_and_untagged_sessions_cannot_alias(first_tagged):
    wire, _ = _wire(version=VERSION if first_tagged else None)
    with pytest.raises(TransferIdentityError):
        wire.register_local(
            99,
            "send",
            None if first_tagged else VERSION,
            "cache",
            None if first_tagged else CONTENT,
        )


@pytest.mark.parametrize("kind", [MessageType.REQUEST_DATA, MessageType.CANCEL_SESSION])
def test_gen_first_peer_binding_precedes_and_constrains_local_session(kind):
    peer, _ = _wire()
    local, _ = _wire(register=False)
    incoming = peer.encode(_body(kind))
    assert local.decode(incoming) == _body(kind)
    assert local.decode(incoming) == _body(kind)
    with pytest.raises(TransferIdentityError):
        local.register_local(99, "send", replace(VERSION, epoch=8), "cache", CONTENT)
    local.register_local(99, "send", VERSION, "cache", CONTENT)
    assert peer.decode(local.encode(_body(MessageType.SESSION_QUIESCED))) == _body(
        MessageType.SESSION_QUIESCED
    )


@pytest.mark.parametrize("kind", KINDS[1:3] + [MessageType.SESSION_QUIESCED])
@pytest.mark.parametrize("tagged", [False, True])
def test_unsolicited_completion_cannot_create_exchange(kind, tagged):
    peer, _ = _wire(version=VERSION if tagged else None)
    local, _ = _wire(register=False)
    with pytest.raises(TransferIdentityError):
        local.decode(peer.encode(_body(kind)))
    with pytest.raises(TransferIdentityError):
        local.encode(_body(kind))


def test_local_admission_cannot_reuse_id_even_with_same_identity():
    wire, table = _wire()
    wire.register_local(99, "send", VERSION, "cache", CONTENT)
    table.close()
    for direction in ("send", "receive"):
        with pytest.raises(TransferIdentityError, match="reused"):
            wire.register_local(99, direction, VERSION, "cache", CONTENT)


@pytest.mark.parametrize("action", ["send", "receive", "local"])
def test_authority_binding_cannot_follow_legacy_transfer_traffic(action):
    wire = ExecutionTransferWire()
    if action == "send":
        assert wire.encode(_body(MessageType.REQUEST_DATA)) == _body(MessageType.REQUEST_DATA)
    elif action == "receive":
        assert wire.decode(_body(MessageType.REQUEST_DATA)) == _body(MessageType.REQUEST_DATA)
    else:
        wire.register_local(99, "send", None, None)
    with pytest.raises(RuntimeError, match="before traffic"):
        wire.bind(WorkerVersionTable("authority"))


def test_peer_discovery_does_not_prevent_authority_binding():
    wire = ExecutionTransferWire()
    body = [MessageType.REGISTER_RANK_INFO, b"metadata"]
    assert wire.decode(wire.encode(body)) == body
    table = WorkerVersionTable("authority")
    wire.bind(table)
    wire.bind(table)
    with pytest.raises(RuntimeError):
        wire.bind(WorkerVersionTable("authority"))


def test_unbound_endpoint_rejects_versioned_peer():
    peer, _ = _wire()
    wire = ExecutionTransferWire()
    with pytest.raises(TransferIdentityError):
        wire.decode(peer.encode(_body(MessageType.REQUEST_DATA)))
    with pytest.raises(TransferIdentityError):
        wire.register_local(99, "send", VERSION, "cache", CONTENT)


@pytest.mark.parametrize("epoch", [0, (1 << 64) - 1])
def test_uint64_epoch_round_trip(epoch):
    identity = replace(IDENTITY, version=replace(VERSION, epoch=epoch))
    assert TransferExecutionIdentity.unpack(identity.pack()) == identity


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1],
        [True, "authority", "tenant", "scope", 7, 99, None, CONTENT.to_values()],
        [3, "authority", "tenant", "scope", 7, 99, None, CONTENT.to_values()],
        [2, "", "tenant", "scope", 7, 99, None, CONTENT.to_values()],
        [2, "authority", "", "scope", 7, 99, None, CONTENT.to_values()],
        [2, "authority", "tenant", "scope", True, 99, None, CONTENT.to_values()],
        [2, "authority", "tenant", "scope", -1, 99, None, CONTENT.to_values()],
        [2, "authority", "tenant", "scope", 7, True, None, CONTENT.to_values()],
        [2, "authority", "tenant", "scope", 7, -1, None, CONTENT.to_values()],
        [2, "authority", "tenant", "scope", 7, 1 << 63, None, CONTENT.to_values()],
        [2, "authority", "tenant", "scope", 7, 99, 4, CONTENT.to_values()],
    ],
)
def test_identity_schema_is_strict(values):
    with pytest.raises(TransferIdentityError):
        TransferExecutionIdentity.unpack(msgpack.packb(values, use_bin_type=True))


@pytest.mark.parametrize("data", [b"", b"\xc1", b"\x91", b"x" * 65537, IDENTITY.pack() + b"\x00"])
def test_malformed_identity_rejected(data):
    with pytest.raises(TransferIdentityError):
        TransferExecutionIdentity.unpack(data)


def test_oversized_identity_rejected_before_local_binding():
    wire, _ = _wire(register=False)
    with pytest.raises(TransferIdentityError, match="wire limit"):
        wire.register_local(99, "send", VERSION, "x" * 65536, CONTENT)
    wire.register_local(99, "send", VERSION, "cache", CONTENT)


@pytest.mark.parametrize(
    "body",
    [
        [],
        [MessageType.KV_AGENT_RESULT],
        [MessageType.KV_AGENT_RESULT, b"short"],
        [MessageType.REQUEST_DATA, b"\xc1"],
        [MessageType.REQUEST_DATA, msgpack.packb({"unique_rid": True})],
        [MessageType.REQUEST_DATA, msgpack.packb({})],
        [MessageType.REQUEST_DATA, msgpack.packb([])],
        [MessageType.AUX_AGENT_RESULT, b"0", b"99"],
        [MessageType.CANCEL_SESSION, b"99", b"0"],
        [MessageType.CANCEL_SESSION, b"-1"],
        [MessageType.SESSION_QUIESCED, b"0", b"invalid"],
        [MessageType.REQUEST_DATA, None],
    ],
)
def test_malformed_body_rejected(body):
    wire, _ = _wire()
    with pytest.raises(TransferIdentityError):
        wire.decode(body)


def test_envelope_cannot_wrap_nontransfer_or_another_envelope():
    wire, _ = _wire()
    encoded = wire.encode(_body(MessageType.REQUEST_DATA))
    with pytest.raises(TransferIdentityError):
        wire.encode(encoded)
    with pytest.raises(TransferIdentityError):
        wire.decode(encoded[:2] + [MessageType.TERMINATION, b"extra"])
    with pytest.raises(TransferIdentityError):
        wire.decode(encoded[:3])


def test_racing_peer_and_local_registration_choose_one_identity():
    wire, _ = _wire(register=False)
    peer, _ = _wire(salt="peer")
    incoming = peer.encode(_body(MessageType.REQUEST_DATA))
    barrier = Barrier(2)

    def attempt(peer_first):
        barrier.wait(timeout=5)
        try:
            if peer_first:
                wire.decode(incoming)
            else:
                wire.register_local(99, "send", VERSION, "local", CONTENT)
            return True
        except TransferIdentityError:
            return False

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(attempt, (True, False))) == [False, True]


@pytest.mark.parametrize(
    "cls,kind,handler",
    [
        (transfer.Sender, MessageType.REQUEST_DATA, "_respond_with_kv"),
        (transfer.Sender, MessageType.CANCEL_SESSION, "_handle_cancel_session"),
        (transfer.Receiver, MessageType.KV_AGENT_RESULT, "_process_kv_agent_result"),
        (transfer.Receiver, MessageType.AUX_AGENT_RESULT, "_process_aux_agent_result"),
        (transfer.Receiver, MessageType.SESSION_QUIESCED, "_process_session_quiesced"),
        (transfer.Receiver, MessageType.CANCEL_SESSION, "_handle_cancel_session"),
    ],
)
def test_native_listener_validates_before_dispatch_and_keeps_old_evidence(cls, kind, handler):
    owner = object.__new__(cls)
    owner._execution_wire, table = _wire()
    owner._messenger = Mock()
    callback = Mock()
    setattr(owner, handler, callback)
    cls._start_listener(owner)
    receive = owner._messenger.start_listener.call_args.args[0]
    body = _body(kind)
    encoded = owner._execution_wire.encode(body)
    wrong = [encoded[0], replace(IDENTITY, cache_salt="other").pack(), *body]
    for rejected in (wrong, body, encoded[:3]):
        assert receive([b"peer", *rejected]) is True
        callback.assert_not_called()
    table.close()
    receive([b"peer", *encoded])
    callback.assert_called_once()
    assert callback.call_args.args[-1] == body


def test_native_send_wraps_late_settlement_after_table_close():
    wire, table = _wire()
    table.close()
    messenger = Mock()
    owner = SimpleNamespace(_execution_wire=wire)
    body = _body(MessageType.SESSION_QUIESCED)
    transfer._send_control_message(owner, messenger, body)
    assert wire.decode(messenger.send.call_args.args[0]) == body
    assert messenger.send.call_args.args[0] != body
