# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import (
    AsyncTransferManager,
    TransferOwner,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import NoFreeSlotsError, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.execution_fence import RequestDrainTracker
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only


def create_mock_request(request_id: int):
    """Create a mock LlmRequest with the given request ID."""
    request = MagicMock()
    request.py_request_id = request_id
    request.py_execution_version = None
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    return request


def create_mock_resource_manager(
    kv_cache_manager=None,
    seq_slot_manager=None,
    spec_resource_manager=None,
):
    """Create a mock ResourceManager with the specified resource managers."""
    resource_manager = MagicMock()
    resource_manager.resource_managers = {}

    if kv_cache_manager is not None:
        resource_manager.resource_managers[ResourceManagerType.KV_CACHE_MANAGER] = kv_cache_manager

    if seq_slot_manager is not None:
        resource_manager.resource_managers[ResourceManagerType.SEQ_SLOT_MANAGER] = seq_slot_manager

    if spec_resource_manager is not None:
        resource_manager.resource_managers[ResourceManagerType.SPEC_RESOURCE_MANAGER] = (
            spec_resource_manager
        )

    return resource_manager


def test_start_transfer_single_request():
    """Test starting a single transfer."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    seq_slot_manager = MagicMock()
    resource_manager = create_mock_resource_manager(
        kv_cache_manager=kv_cache_manager, seq_slot_manager=seq_slot_manager
    )
    manager = AsyncTransferManager(resource_manager)

    request = create_mock_request(42)
    manager.start_transfer(request, owner=TransferOwner.TRANSCEIVER)

    # Check request is tracked
    assert 42 in manager._requests_in_transfer

    transfer_metadata = manager._request_transfer_metadata[42]

    assert transfer_metadata.block_id == 100
    assert transfer_metadata.counter == 1

    # Check state was updated
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    # Check KV cache manager was called
    kv_cache_manager.store_blocks_for_reuse.assert_called_once_with(request, True)

    # Check seq slot manager was called to free resources
    seq_slot_manager.free_resources.assert_called_once_with(request)

    manager.end_transfer(request, owner=TransferOwner.TRANSCEIVER)
    kv_cache_manager.unpin_blocks_by_id.assert_called_once()


@pytest.mark.parametrize("first_owner", list(TransferOwner))
def test_two_transfer_owners_release_independently(first_owner: TransferOwner) -> None:
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    manager = AsyncTransferManager(create_mock_resource_manager(kv_cache_manager=kv_cache_manager))
    request = create_mock_request(42)
    second_owner = next(owner for owner in TransferOwner if owner is not first_owner)
    for owner in (first_owner, second_owner):
        manager.start_transfer(request, owner=owner)
    assert manager._request_transfer_metadata[42].counter == 2
    kv_cache_manager.store_blocks_for_reuse.assert_called_once()

    assert not manager.end_transfer(request, owner=first_owner)
    assert not manager.end_transfer(request, owner=first_owner)  # duplicate completion
    assert manager.has_transfer(request, owner=second_owner)
    kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    assert manager.end_transfer(request, owner=second_owner)
    kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(100)
    assert not manager.end_transfer(request, owner=second_owner)
    assert not manager.has_any_inflight_requests()


@pytest.mark.parametrize("owner", list(TransferOwner))
def test_duplicate_start_preserves_the_original_hold(owner: TransferOwner) -> None:
    cache, slots = MagicMock(), MagicMock()
    manager = AsyncTransferManager(create_mock_resource_manager(cache, slots))
    request = create_mock_request(42)
    manager.start_transfer(request, owner=owner)

    with pytest.raises(RuntimeError, match="already holds"):
        manager.start_transfer(request, owner=owner)

    assert manager._request_transfer_metadata[42].counter == 1
    cache.store_blocks_for_reuse.assert_called_once()
    slots.free_resources.assert_called_once()
    cache.unpin_blocks_by_id.assert_not_called()


@pytest.mark.parametrize("operation", ["start_transfer", "end_transfer", "has_transfer"])
@pytest.mark.parametrize("mutation", ["request", "version", "request_id"])
def test_mismatched_identity_cannot_touch_live_claims(operation: str, mutation: str) -> None:
    cache = MagicMock()
    manager = AsyncTransferManager(create_mock_resource_manager(cache))
    request = create_mock_request(42)
    request.py_execution_version = ExecutionVersion("tenant", "scope", 1)
    tracker = RequestDrainTracker()
    tracker.enqueue(request.py_execution_version, lambda: 42)
    manager.bind_execution_drain(tracker)
    manager.start_transfer(request, owner=TransferOwner.TRANSCEIVER)

    if mutation == "request":
        other = create_mock_request(42)
        other.py_execution_version = request.py_execution_version
    elif mutation == "version":
        other = request
        other.py_execution_version = ExecutionVersion("tenant", "scope", 2)
    else:
        other = create_mock_request(43)
        manager.start_transfer(other, owner=TransferOwner.CONNECTOR)
        other.py_request_id = 42

    with pytest.raises(RuntimeError, match="identity changed"):
        getattr(manager, operation)(other, owner=TransferOwner.CONNECTOR)

    assert manager._requests_in_transfer[42] is request
    assert manager._request_transfer_metadata[42].version.epoch == 1
    assert manager._request_transfer_metadata[42].owners == {TransferOwner.TRANSCEIVER}
    cache.unpin_blocks_by_id.assert_not_called()


def test_late_completion_cannot_release_a_new_request_with_the_same_id() -> None:
    cache = MagicMock()
    manager = AsyncTransferManager(create_mock_resource_manager(cache))
    old, new = create_mock_request(42), create_mock_request(42)
    manager.start_transfer(old, owner=TransferOwner.TRANSCEIVER)
    assert manager.end_transfer(old, owner=TransferOwner.TRANSCEIVER)
    manager.start_transfer(new, owner=TransferOwner.TRANSCEIVER)
    cache.unpin_blocks_by_id.reset_mock()

    with pytest.raises(RuntimeError, match="identity changed"):
        manager.end_transfer(old, owner=TransferOwner.TRANSCEIVER)

    assert manager.has_transfer(new, owner=TransferOwner.TRANSCEIVER)
    cache.unpin_blocks_by_id.assert_not_called()


@pytest.mark.parametrize("owner", list(TransferOwner))
def test_failed_unpin_keeps_ownership_and_request_state(owner: TransferOwner) -> None:
    cache = MagicMock()
    manager = AsyncTransferManager(create_mock_resource_manager(cache))
    request = create_mock_request(42)
    manager.start_transfer(request, owner=owner)
    cache.unpin_blocks_by_id.side_effect = RuntimeError("unpin failed")

    with pytest.raises(RuntimeError, match="unpin failed"):
        manager.end_transfer(request, owner=owner)

    assert manager.has_transfer(request, owner=owner)
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    assert manager.has_any_inflight_requests()
    cache.unpin_blocks_by_id.side_effect = None
    with pytest.raises(RuntimeError, match="release is unknown"):
        manager.end_transfer(request, owner=owner)
    with pytest.raises(RuntimeError, match="release is unknown"):
        manager.start_transfer(request, owner=owner)
    cache.unpin_blocks_by_id.assert_called_once()


@pytest.mark.parametrize("operation", ["start_transfer", "end_transfer", "has_transfer"])
def test_unknown_owner_does_not_change_resources(operation: str) -> None:
    cache, slots = MagicMock(), MagicMock()
    manager = AsyncTransferManager(create_mock_resource_manager(cache, slots))
    request = create_mock_request(42)
    with pytest.raises(TypeError, match="TransferOwner"):
        getattr(manager, operation)(request, owner="transceiver")
    assert not manager.has_any_inflight_requests()
    cache.store_blocks_for_reuse.assert_not_called()
    slots.free_resources.assert_not_called()


def test_transfer_without_storing_blocks():
    """Test starting a transfer with should_store_blocks=False."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 0
    spec_resource_manager = MagicMock()
    resource_manager = create_mock_resource_manager(
        kv_cache_manager=kv_cache_manager, spec_resource_manager=spec_resource_manager
    )
    manager = AsyncTransferManager(resource_manager, should_store_blocks=False)

    request = create_mock_request(42)
    manager.start_transfer(request, owner=TransferOwner.TRANSCEIVER)

    # Check request is tracked
    assert 42 in manager._requests_in_transfer
    transfer_metadata = manager._request_transfer_metadata[42]
    assert transfer_metadata.block_id is None  # No block stored
    assert transfer_metadata.counter == 1

    # Check KV cache manager was NOT called
    kv_cache_manager.store_blocks_for_reuse.assert_not_called()
    spec_resource_manager.free_resources.assert_called_once_with(request)

    assert manager.end_transfer(request, owner=TransferOwner.TRANSCEIVER)

    kv_cache_manager.unpin_blocks_by_id.assert_not_called()


@pytest.mark.parametrize("first_owner", list(TransferOwner))
@pytest.mark.parametrize("store_blocks", [False, True])
def test_transfer_drain_survives_terminal_request(first_owner, store_blocks):
    cache = MagicMock()
    manager = AsyncTransferManager(create_mock_resource_manager(cache), store_blocks)
    tracker = RequestDrainTracker()
    manager.bind_execution_drain(tracker)
    request = create_mock_request(42)
    version = request.py_execution_version = ExecutionVersion("tenant", "scope", 1)
    tracker.enqueue(version, lambda: 42)
    manager.start_transfer(request, owner=first_owner)
    tracker.complete(42, version)
    second_owner = next(owner for owner in TransferOwner if owner is not first_owner)
    manager.start_transfer(request, owner=second_owner)
    assert not tracker.drained(version)
    assert not manager.end_transfer(request, owner=first_owner)
    assert not manager.end_transfer(request, owner=first_owner)
    assert not tracker.drained(version)
    assert manager.end_transfer(request, owner=second_owner)
    assert tracker.drained(version)
    assert cache.unpin_blocks_by_id.call_count == int(store_blocks)


def test_tagged_transfer_requires_tracking_before_native_mutation():
    cache, slots = MagicMock(), MagicMock()
    manager = AsyncTransferManager(create_mock_resource_manager(cache, slots))
    request = create_mock_request(42)
    request.py_execution_version = ExecutionVersion("tenant", "scope", 1)
    with pytest.raises(RuntimeError, match="requires execution drain"):
        manager.start_transfer(request, owner=TransferOwner.TRANSCEIVER)
    cache.store_blocks_for_reuse.assert_not_called()
    slots.free_resources.assert_not_called()
    assert not manager.has_any_inflight_requests()


@pytest.mark.parametrize("existing", ["tracker", "transfer"])
def test_transfer_tracker_binding_cannot_replace_existing_ownership(existing):
    manager = AsyncTransferManager(create_mock_resource_manager(MagicMock()))
    tracker = RequestDrainTracker()
    if existing == "tracker":
        manager.bind_execution_drain(tracker)
        manager.bind_execution_drain(tracker)
    else:
        manager.start_transfer(create_mock_request(42), owner=TransferOwner.TRANSCEIVER)
    with pytest.raises(RuntimeError, match="bound once"):
        manager.bind_execution_drain(RequestDrainTracker())


@pytest.mark.parametrize("failure", ["slot", "pin", "unpin"])
@pytest.mark.parametrize("tagged", [False, True])
def test_uncertain_native_transfer_ownership_poison_fence(failure, tagged):
    cache, slots = MagicMock(), MagicMock()
    manager = AsyncTransferManager(create_mock_resource_manager(cache, slots))
    tracker = RequestDrainTracker()
    manager.bind_execution_drain(tracker)
    request = create_mock_request(42)
    version = ExecutionVersion("tenant", "scope", 1)
    if tagged:
        request.py_execution_version = version
        tracker.enqueue(version, lambda: 42)
    method = {
        "slot": slots.free_resources,
        "pin": cache.store_blocks_for_reuse,
        "unpin": cache.unpin_blocks_by_id,
    }[failure]
    method.side_effect = RuntimeError("native state unknown")
    if failure == "unpin":
        manager.start_transfer(request, owner=TransferOwner.TRANSCEIVER)
        operation = manager.end_transfer
    else:
        operation = manager.start_transfer
    with pytest.raises(RuntimeError, match="native state unknown"):
        operation(request, owner=TransferOwner.TRANSCEIVER)
    if tagged:
        tracker.complete(42, version)
        with pytest.raises(RuntimeError, match="drain is unknown"):
            tracker.drained(version)
    assert manager.has_transfer(request, owner=TransferOwner.TRANSCEIVER)
    method.side_effect = None
    with pytest.raises(RuntimeError, match="release is unknown"):
        manager.end_transfer(request, owner=TransferOwner.TRANSCEIVER)
    method.assert_called_once()


@pytest.mark.parametrize("released", [False, True])
def test_failed_drain_completion_cannot_repeat_native_unpin(monkeypatch, released):
    cache = MagicMock()
    manager = AsyncTransferManager(create_mock_resource_manager(cache))
    tracker = RequestDrainTracker()
    manager.bind_execution_drain(tracker)
    request = create_mock_request(42)
    version = request.py_execution_version = ExecutionVersion("tenant", "scope", 1)
    tracker.enqueue(version, lambda: 42)
    manager.start_transfer(request, owner=TransferOwner.TRANSCEIVER)
    tracker.complete(42, version)
    original_release = tracker.release_resource

    def fail_completion(claim):
        if released:
            original_release(claim)
        raise RuntimeError("completion failed")

    monkeypatch.setattr(tracker, "release_resource", fail_completion)
    with pytest.raises(RuntimeError, match="completion failed"):
        manager.end_transfer(request, owner=TransferOwner.TRANSCEIVER)
    with pytest.raises(RuntimeError, match="drain is unknown"):
        tracker.drained(version)
    assert manager.has_transfer(request, owner=TransferOwner.TRANSCEIVER)
    with pytest.raises(RuntimeError, match="release is unknown"):
        manager.end_transfer(request, owner=TransferOwner.TRANSCEIVER)
    cache.unpin_blocks_by_id.assert_called_once()


def test_resource_claim_precedes_slot_and_pin_side_effects():
    entered, proceed = Event(), Event()
    cache, slots = MagicMock(), MagicMock()
    manager = AsyncTransferManager(create_mock_resource_manager(cache, slots))
    tracker = RequestDrainTracker()
    manager.bind_execution_drain(tracker)
    request = create_mock_request(42)
    version = request.py_execution_version = ExecutionVersion("tenant", "scope", 1)
    tracker.enqueue(version, lambda: 42)

    def release_slots(req):
        entered.set()
        assert proceed.wait(5)

    slots.free_resources.side_effect = release_slots
    with ThreadPoolExecutor(1) as pool:
        started = pool.submit(manager.start_transfer, request, owner=TransferOwner.TRANSCEIVER)
        try:
            assert entered.wait(5)
            tracker.complete(42, version)
            assert not tracker.drained(version)
        finally:
            proceed.set()
        started.result(5)
    assert manager.end_transfer(request, owner=TransferOwner.TRANSCEIVER)
    assert tracker.drained(version)


def test_end_transfer_preserves_error_state():
    """Test that end_transfer does not overwrite error state."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    request = create_mock_request(42)
    manager.start_transfer(request, owner=TransferOwner.TRANSCEIVER)

    # Set error state before end_transfer
    request.state = LlmRequestState.DISAGG_TRANS_ERROR

    manager.end_transfer(request, owner=TransferOwner.TRANSCEIVER)

    # Error state should be preserved
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR


def test_requests_in_transfer():
    """Test that requests_in_transfer returns correct mapping."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    request1 = create_mock_request(1)
    request2 = create_mock_request(2)
    request3 = create_mock_request(3)

    manager.start_transfer(request1, owner=TransferOwner.TRANSCEIVER)
    manager.start_transfer(request2, owner=TransferOwner.TRANSCEIVER)
    manager.start_transfer(request3, owner=TransferOwner.TRANSCEIVER)

    in_transfer = manager.requests_in_transfer()

    assert len(in_transfer) == 3
    assert in_transfer[1] is request1
    assert in_transfer[2] is request2
    assert in_transfer[3] is request3


def _slot_request(request_id: int) -> SimpleNamespace:
    """Request stub with the attributes SeqSlotManager and the transfer manager read."""
    return SimpleNamespace(
        request_id=request_id,
        py_request_id=request_id,
        py_execution_version=None,
        state=LlmRequestState.CONTEXT_INIT,
        seq_slot=None,
        py_seq_slot=None,
        is_disagg_generation_init_state=False,
        is_disagg_generation_transmission_complete=False,
        return_perf_metrics=False,
    )


def test_start_transfer_frees_the_sequence_slot_so_the_next_request_fits():
    """A context-only request entering transfer keeps only its KV blocks: its
    sequence slot and speculative-decoding resources are released at once, so
    with a single slot the next request can be prepared while the first one is
    still transferring (the slot leak fixed in #6032)."""
    seq_slots = SeqSlotManager(max_num_sequences=1)
    spec_resource_manager = MagicMock()
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    manager = AsyncTransferManager(
        create_mock_resource_manager(
            kv_cache_manager=kv_cache_manager,
            seq_slot_manager=seq_slots,
            spec_resource_manager=spec_resource_manager,
        )
    )

    def prepare(request):
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [request]
        seq_slots.prepare_resources(batch)

    first, second = _slot_request(1), _slot_request(2)
    prepare(first)
    assert first.py_seq_slot == 0
    with pytest.raises(NoFreeSlotsError):
        prepare(second)

    manager.start_transfer(first, owner=TransferOwner.TRANSCEIVER)

    prepare(second)
    assert second.py_seq_slot == 0
    spec_resource_manager.free_resources.assert_called_once_with(first)
    assert list(manager.requests_in_transfer()) == [1]
