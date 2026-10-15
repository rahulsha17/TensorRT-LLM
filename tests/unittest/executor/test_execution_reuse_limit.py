# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V2 lookup evidence must honor the same bound as collective RESUME."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("backoff", [0, 2])
@pytest.mark.parametrize(
    "execution_limit,other_limit,expected",
    [
        (None, None, 16),
        (0, None, 0),
        (4, None, 4),
        (7, None, 7),
        (8, 4, 4),
        (4, 8, 4),
        (0, 8, 0),
        (8, 0, 0),
    ],
)
def test_context_evidence_intersects_limits_without_losing_lookahead(
    backoff, execution_limit, other_limit, expected
):
    manager = object.__new__(KVCacheManagerV2)
    tokens = tuple(range(17))
    manager.reuse_match_backoff = backoff
    manager._reuse_token_source = lambda request: tokens
    manager._augment_tokens_for_block_reuse = lambda source, request, end: source[:end]
    request = SimpleNamespace(execution_reuse_limit=execution_limit)
    evidence = manager._context_reuse_tokens(request, other_limit)
    end = 0 if expected == 0 else min(expected + backoff, len(tokens))
    assert evidence == tokens[:end]
