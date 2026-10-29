<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental RETIRE GPT-OSS V2 candidate

This is a separate, unqualified extension of the native RETIRE implementation
and finalized-batch policy. It adds a narrowly validated variable-window path
and a GPT-OSS model layer hook. It has not been built or run on a GPU. Earlier
Qwen results and wheels do not qualify it.

## Private startup contract

Set `TRTLLM_RETIRE_GPT_OSS_V2=1`, `TRTLLM_RETIRE_NATIVE_GUARDS=1`, and
`TLLM_FMHA_LIBS=fallback` before constructing the LLM. An unset or `0` capability
keeps the previous guard restrictions and GPT-OSS model loop behavior. Other
values fail startup. The capability is latched in the attention binding and
model; changing the environment after construction cannot enable it.

The candidate requires the exact native `GptOssForCausalLM`, Triton MoE,
SM90 BF16 activations, TRTLLM attention, TP1, eager execution, disabled overlap,
and a V2 BF16 KV manager with ordinary key/value buffers. It rejects compilation,
graphs and in-graph sampling, speculation, beams, attention DP, sparse or
recurrent caches, EPLB, LoRA, guided decoding, and KV connectors/transfers.
It requires ordinary `ALL_REUSABLE` block reuse without conversation or
compressed-history management. V1 variable-window ALIGN remains unsupported.

`TRTLLM_RETIRE_BATCH_POLICY` remains optional and separate. Its original JSON
contract selects horizon `0` or `4` and minimum total scheduled tokens `1`, `32`,
or `1024`. With horizon `4`, the new model hook checks positive multiples of four
layers, excluding layer zero, when every actual row is tagged and the finalized
batch meets the threshold. Horizon `0` and below-threshold batches retain the
existing event-checked host token acceptance path. Mixed untagged batches cannot
use a whole-batch early exit. Compilation and capture also suppress the model
hook defensively, although the candidate rejects those configurations.

## Cache layout and ownership

Startup records the actual model object and alternating sliding/full attention
windows. Controller bootstrap validates the actual V2 layer membership, window
vector, layer buffer roles, BF16 cache, sequence bound and scheduling mode against
that record. Full-retention overrides do not qualify. GPT-OSS's learned
`attention_sinks` are softmax terms; they are not KV sink tokens, and this path
does not add KV sink retention.

The extension does not change allocation, native page holders, events, prefix
matching or cleanup. `ExecutionDeviceGuard.context_commit` still waits for
submitted native work before advancing accepted context. `before_release` still
settles uses and accepted prefix publication before cache release. V2's
`preserve_execution_prefix` commits only drained, accepted complete blocks.
`KVAlignmentAdapter._reserve_v2` obtains a native suspended cache and retains its
page holders. The hold may migrate; it is not a promise of HBM residency.

V2's radix matching checks required pages separately for every lifecycle at the
final reuse endpoint: full-attention history plus the sliding window's required
tail. A missing page can shorten or reject a match. The adapter reports the
native match, never the requested reuse limit. An earlier accepted prefix is not
proof its window pages remain reusable after subsequent allocation. Required
reuse must still fail closed if unavailable. This candidate neither forces full
retention nor manufactures a reuse receipt.

Dense-only resume capacity locks and asynchronous context publication remain
disabled for this layout. Existing synchronous FENCE semantics remain unchanged.
The backing-tier ALIGN budget conservatively charges every lifecycle and does
not assert where the pages currently reside.

## Required GPU qualification

Before claiming support or launching paper reproduction:

1. Verify the exact checkpoint and its native load route. The paper's
   `openai/gpt-oss-20b` revision `6cee5e81ee83917806bbde320786a8fb61efebee`
   has MXFP4 expert weights, 24 alternating sliding-128/full layers, 64 query
   heads, eight KV heads and head dimension 64. The SM90 source route supports
   `W4A16_MXFP4` with Triton MoE and BF16 activations; this is not a BF16-weight
   checkpoint. Record the actual loaded implementation, quantization, routing
   and finalize-fusion settings. Source compatibility is not a GPU result.
2. Read back the actual V2 windows, per-layer and per-pool membership, page sizes,
   GPU/host quotas and capacity. A dense Qwen token-capacity figure cannot be
   carried over as equivalent capacity for heterogeneous pools. Preserve the
   natural sliding window. Record hardware differences and admission/pauses
   separately from requested concurrency.
3. Exercise prefill and decode revisions around both block and window boundaries,
   including incomplete forwards, partial tails and endpoints that move backward.
   Require genuine active work, publication/FENCE order, no obsolete publication,
   native successor attachment, and zero retained ownership after termination.
   Check both positive ALIGN and missing-window-page rejection against the native
   pool/lifecycle observations, under low pressure and host-backed churn.
4. Demonstrate actual h4 probes/exits and h0/below-threshold host acceptance using
   the finalized batch counters and controlled interruption. Confirm full native
   stream/event drain after a model-layer exit; a host hook does not interrupt a
   running kernel or prove saved GPU time.
5. Repeat cold and mode-matched numerical controls using the same sampler and
   batch shape. Classify numerical differences separately from lifecycle failure.
   Cover stale admission, failed ALIGN/RESUME, cancellation and latest-epoch
   terminal cleanup. Keep V1, TP/PP, overlap, graphs and speculative negative
   controls rejected.

The new tests are local developer/QA tests, not additions to a CI test list.
Host tests execute actual Python guard, policy, model-loop and adapter methods
with explicit native/CUDA doubles. They cover opt-in/default behavior, mode and
layout rejection, h4 dispatch and restoration, and conservative suspended-hold
accounting. They cannot validate native page lifetime, CUDA event ordering,
fallback attention with sinks/windows, Triton MoE execution or model accuracy.

## Source boundaries

- `execution_gpt_oss.py`: startup capability and actual executor/cache validation.
- `execution_attention.py`: immutable startup binding and model-hook latch.
- `execution_guard.py`: narrow exception to the variable-window guard.
- `execution_rate_policy.py`: the existing policy may select this validated model.
- `modeling_gpt_oss.py`: opt-in checks between blocks; no attention or MoE kernel changes.
- `test_execution_gpt_oss.py`: candidate host contracts and actual method tests.

The candidate starts from native commit
`7534f3e5d78cb738828d10423a7a42b67ef7d4aa` and includes the test-only metadata
fixture correction from `853815dd35cb0c02bc0753bdb7f503f658d4494f`.
