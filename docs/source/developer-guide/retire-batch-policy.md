<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental RETIRE batch policy

This private startup policy selects native device predicates and layer probes
from each finalized physical batch. It is an opt-in Python engine extension.
It does not change the behavior of a process that leaves
`TRTLLM_RETIRE_BATCH_POLICY` unset. It requires a separately identified source
build; results from an earlier wheel do not qualify this extension.

## Startup contract

Set these variables before creating the LLM:

```sh
export TRTLLM_RETIRE_NATIVE_GUARDS=1
export TRTLLM_RETIRE_EARLY_EXIT=1
export TLLM_FMHA_LIBS=fallback
export TRTLLM_RETIRE_BATCH_POLICY='{"layer_horizon":4,"min_scheduled_tokens":32}'
```

The JSON must contain exactly those two integer fields. `layer_horizon` accepts
`0` or `4`; `min_scheduled_tokens` accepts `1`, `32`, or `1024`. Invalid values
fail startup. This is an internal experiment contract, not a public LLM argument.

The supported configuration is dense Qwen3, TP1, BF16 on SM90, TRTLLM attention,
eager execution with the overlap scheduler disabled. CUDA graphs, compilation,
speculation, beam search, multimodal/encoder-only models, attention DP, and
sparse attention are rejected. Existing native authority validation continues
to reject recurrent or variable-sliding-window KV, guided decoding, and KV
connectors or transfers. The extension does not broaden native engine support.

A separate [GPT-OSS V2 candidate](retire-gpt-oss-v2.md) adds an explicit private
opt-in for one validated variable-window layout. It preserves the defaults above
and remains unqualified on GPUs; the Qwen qualification evidence does not apply
to it.

## Batch and acceptance semantics

The runner freezes the actual ordered request objects, runtime IDs, execution
versions, and sequence slots after preparing the eager batch. It checks actual
input length against the sum of context chunk tokens and decode tokens. The
threshold therefore uses total scheduled tokens, including decode rows.

With horizon `4`, a batch at or above the threshold and containing tagged work
uses native attention and sampled-token device predicates. Layer probes run
only when every real row is tagged, at positive multiples of four; layer zero
is excluded. A mixed batch cannot use a whole-batch layer exit. Horizon `0`, or
a batch below the threshold, clears the four attention predicate references
and retains their owned buffers for subsequent use. Ordinary prelaunch
authority checks remain enabled in both modes.

A batch without device predicates still waits for native sampling completion,
checks asynchronous copy failures, and waits for its recorded sampling-stream
completion event before the host authority gate can accept a token. Its receipt
binds the exact ordered sampling rows and token buffers and is consumed once.
Armed batches retain the existing native device ticket as well as the identity
receipt. Intermediate context chunks with no sampled rows issue no token
receipt. Native context completion waits, use records, cancellation, and
terminal FENCE ownership rules remain in force in both modes.

## Bounded observations

In a local TP1 executor, after authority bootstrap, read:

```python
snapshot = executor.engine._execution_guard._rate_policy.snapshot()
```

The same state is available before bootstrap at
`executor.engine.model_engine.execution_attention.rate_policy`. These are private
implementation paths. The snapshot contains the resolved policy, cumulative
counters, and the last 128 finalized decisions. Each decision includes actual
context/decode token counts, real/tagged row counts, and predicate/layer choices.
Absent counter keys mean zero. Copying a snapshot takes a host lock and adds
measurement overhead.

Counters distinguish finalized `forward_plans`, `predicate_plans` and
`boundary_plans`; context/decode token totals; attention preparations;
prelaunch/layer probes and exits; device/boundary sample dispatches and host
commit checks; and `host_rejected_rows`. A finalized plan can subsequently exit
or fail. These are host dispatch and acceptance observations, not GPU kernel
completion, saved FLOPs, physical KV ownership, or device-predicate hit receipts.
The bounded decision history does not provide a complete per-batch trace for a
long workload.

## Qualification status

The targeted host suite has 534 passing cases using the actual Python method
bodies with explicit native/CUDA doubles. Four pre-existing parameterized
attachment cases that depend on native bindings are excluded from that host
run. Repository lint and formatting checks pass. These checks cover threshold
selection, metadata clearing, exact sample identity, one-use receipts, event
ordering, failed asynchronous copies, failed forward cleanup, and legacy guard
behavior. They do not validate CUDA kernels, real stream/event behavior, or
native attachment bindings. GPU qualification and performance results for this
separate extension are pending.
