# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N-gram proposal state follows request lifetime, including cancellation."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.ngram import NGramPoolManager

pytestmark = pytest.mark.cpu_only


def manager(public=False):
    return NGramPoolManager(
        SimpleNamespace(
            tokens_per_gen_step=5,
            max_matching_ngram_size=3,
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=public,
        ),
        max_num_requests=8,
    )


@pytest.mark.parametrize("completed_batch", [False, True])
def test_private_pool_release_preserves_neighbor_and_reinitializes_reused_id(completed_batch):
    pool = manager()
    tokens = [11, 12, 13, 14] * 4
    assert pool.get_draft_tokens(tokens, 41, 128)
    assert pool.get_draft_tokens(tokens, 42, 128)
    neighbor = deepcopy(pool.pool[42])
    cursor = pool.start_index[42]
    request = SimpleNamespace(request_id=41, state=LlmRequestState.GENERATION_COMPLETE)
    if completed_batch:
        pool.update_resources(SimpleNamespace(all_requests=lambda: iter([request])))
    else:
        pool.free_resources(request)
    pool.free_resources(request)
    assert 41 not in pool.pool and 41 not in pool.start_index
    assert pool.pool[42] == neighbor and pool.start_index[42] == cursor
    # Request IDs can be reused. Old proposal state must not supply new drafts.
    replacement = [91, 92, 93, 94] * 3
    actual = pool.get_draft_tokens(replacement, 41, 128)
    assert actual == manager().get_draft_tokens(replacement, 41, 128)
    assert all(11 not in pattern for pattern in pool.pool[41])


def test_public_pool_release_drops_only_request_cursor():
    pool = manager(public=True)
    pool.get_draft_tokens([11, 12, 13, 14] * 4, 41, 128)
    contents = deepcopy(pool.pool)
    pool.free_resources(SimpleNamespace(request_id=41))
    assert pool.pool == contents
    assert 41 not in pool.start_index


def test_completed_private_cursor_is_released_even_without_a_pool_entry():
    pool = manager()
    pool.start_index[41] = 12
    request = SimpleNamespace(request_id=41, state=LlmRequestState.GENERATION_COMPLETE)
    pool.update_resources(SimpleNamespace(all_requests=lambda: iter([request])))
    assert pool.start_index == {}
