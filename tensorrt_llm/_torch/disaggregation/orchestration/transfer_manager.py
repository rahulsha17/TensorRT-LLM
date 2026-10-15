# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Dict, Optional

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.logger import logger


class TransferOwner(Enum):
    """Independent accessors that must each release their request's KV hold."""

    TRANSCEIVER = "transceiver"
    CONNECTOR = "connector"


class AsyncTransferManager:
    """
    Handle asynchronous transfer of KV cache after a request has completed.
    When running with both the KV cache transceiver and the KV cache connector, we must ensure
    that BOTH transfers (if any) are completed before we can release the KV cache blocks.
    The AsyncTransferManager has a few key responsibilities:
    1. Track requests in transfer.
    2. Pin blocks for reuse while blocks are in transfer.
    3. Unpin blocks after all transfers are complete.

    Only the executor thread mutates these holds. A completion identifies one
    accessor of the original request; it does not grant execution authority or
    establish that a received prefix is reusable.

    TODO(jthomson04): This only handles async send/saving, not loading. Loading kv cache is
    handled through a separate codepath. Eventually, we'll want to merge these two paths.
    """

    class RequestTransferMetadata:
        def __init__(self, block_id: Optional[int], version: ExecutionVersion | None) -> None:
            self.block_id = block_id
            self.version = version
            self.owners: set[TransferOwner] = set()
            self.release_error: Exception | None = None

        @property
        def counter(self) -> int:
            return len(self.owners)

    def __init__(self, resource_manager: "ResourceManager", should_store_blocks: bool = True):
        self.resource_manager = resource_manager
        self.kv_cache_manager = resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER
        )

        self.should_store_blocks = should_store_blocks

        # Mapping of request id to the LlmRequest
        self._requests_in_transfer: Dict[int, LlmRequest] = dict()

        # Mapping of request id to the request metadata
        self._request_transfer_metadata: Dict[int, self.RequestTransferMetadata] = dict()

    def requests_in_transfer(self) -> Dict[int, LlmRequest]:
        return self._requests_in_transfer

    def _metadata(self, request: LlmRequest) -> RequestTransferMetadata | None:
        """Match the original allocation owner before changing any hold."""
        req_id = request.py_request_id
        metadata = self._request_transfer_metadata.get(req_id)
        if metadata is not None and (
            self._requests_in_transfer[req_id] is not request
            or metadata.version != request.py_execution_version
        ):
            raise RuntimeError("KV transfer request identity changed while resources are owned")
        return metadata

    def has_transfer(self, request: LlmRequest, *, owner: TransferOwner) -> bool:
        """Whether this accessor still holds the original request's resources."""
        if not isinstance(owner, TransferOwner):
            raise TypeError("transfer owner must be a TransferOwner")
        metadata = self._metadata(request)
        return metadata is not None and owner in metadata.owners

    def start_transfer(self, request: LlmRequest, *, owner: TransferOwner) -> None:
        """
        Called when a Cache transceiver or connector transfer is started.
        1. Register the accessor's unique claim for this request.
        2. Releases all resources except for the KV cache, if not already released.
        3. Store KV cache blocks for reuse.
        """

        req_id = request.py_request_id
        if not isinstance(owner, TransferOwner):
            raise TypeError("transfer owner must be a TransferOwner")
        metadata = self._metadata(request)
        if metadata is not None and metadata.release_error is not None:
            raise RuntimeError(
                "KV transfer release is unknown; restart required"
            ) from metadata.release_error
        if metadata is not None and owner in metadata.owners:
            raise RuntimeError("KV transfer owner already holds this request")

        if metadata is None:
            for resource_mgr_type in (
                ResourceManagerType.SEQ_SLOT_MANAGER,
                ResourceManagerType.SPEC_RESOURCE_MANAGER,
            ):
                if (
                    resource_mgr_type in self.resource_manager.resource_managers
                    and self.resource_manager.resource_managers[resource_mgr_type] is not None
                ):
                    self.resource_manager.resource_managers[resource_mgr_type].free_resources(
                        request
                    )

            request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

            if self.should_store_blocks:
                block_id = self.kv_cache_manager.store_blocks_for_reuse(request, True)
            else:
                block_id = None

            self._requests_in_transfer[req_id] = request
            metadata = self.RequestTransferMetadata(block_id, request.py_execution_version)
            self._request_transfer_metadata[req_id] = metadata

        metadata.owners.add(owner)

    def end_transfer(self, request: LlmRequest, *, owner: TransferOwner) -> bool:
        """
        Called after a send of KV cache is complete.
        1. Release only the named accessor's claim, at most once.
        2. If there are no more inflight transfers for this request, unpin the blocks and mark the request as complete.

        Returns:
            bool: True if the request should be terminated after call to end_transfer
        """
        if not self.has_transfer(request, owner=owner):
            logger.warning(
                f"Request {request.py_request_id} has no {owner.value} claim in transfer manager"
            )
            return False

        transfer_metadata = self._request_transfer_metadata[request.py_request_id]
        if transfer_metadata.release_error is not None:
            raise RuntimeError(
                "KV transfer release is unknown; restart required"
            ) from transfer_metadata.release_error
        if len(transfer_metadata.owners) == 1:
            # Keep the owner discoverable if unpinning fails. No terminal
            # response or FENCE may treat that failure as a successful release.
            if self.should_store_blocks:
                try:
                    self.kv_cache_manager.unpin_blocks_by_id(transfer_metadata.block_id)
                except Exception as error:
                    # An exception may follow a partial native release. A retry
                    # could decrement a different owner's pin a second time.
                    transfer_metadata.release_error = error
                    raise

            self._requests_in_transfer.pop(request.py_request_id)
            self._request_transfer_metadata.pop(request.py_request_id)
            transfer_metadata.owners.remove(owner)
            # We don't want to overwrite any error state.
            if request.state != LlmRequestState.DISAGG_TRANS_ERROR:
                request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE

            return True

        transfer_metadata.owners.remove(owner)
        return False

    def has_any_inflight_requests(self) -> bool:
        return len(self._requests_in_transfer) > 0
