<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# RETIRE optimization ideas

This backlog separates existing mechanisms from proposed experiments. None of
the proposed changes has a performance claim. The reference implementation is
the V46b merge, `89714ede9c`; lifecycle behavior and ownership requirements are
described in [RETIRE](retire.md).

This branch implements a first local V2 handoff before old cleanup on top of that
reference. Its ownership and fallback contract is documented in
[Local handoff before old cleanup](retire.md#local-handoff-before-old-cleanup).
Host tests do not establish CUDA correctness or a serving speedup.

## Existing mechanisms and benchmark context

The reference implementation already includes capacity-aware V2 resume, bounded
admission retries, incremental cold-page promotion, short protection of resident
ALIGN prefixes, cached immutable authority metadata, packed token/verdict copies,
and asynchronous context receipts for eligible single-rank V2 overlap execution.
Scheduling progress alone never establishes reusable KV: a completed authority
receipt, accepted prefix, exact allocation ownership, and the latest-reader
completion checks still govern publication.

Early exit stops wholly obsolete batches before further eager layer groups or
breakable graph segments. Native attention predicates can skip eligible future
attention work and suppress stores. They do not terminate arbitrary running
kernels, and a live row prevents a whole-batch exit. Multi-block attention must
preserve required scratch writes and semaphore arrivals.

The surveyed `2026-10-27-recovery-fairness-r6` replay harness is a separate
experiment artifact. It uses Qwen3-32B, TP1 BF16, V2, overlap, chunked prefill with
a 4,096-token budget, and decode graph batch sizes `[1, 2, 4, 8, 16]`. Its prefill
uses eager execution: it supplies neither `prefill_cuda_graph_backend` nor
`torch_compile_config`. RETIRE arms enable native attention guards and host early
exit. This setup does not measure conditional CUDA graphs or establish the
benefit of breakable prefill graphs. The later prefill fair-share policy is also
separate from the V46b reference; it is not an existing feature of this base.

## Measuring handoff before cleanup

The archived Yanxin `concurrent_replay.py` and the October 27 matrix reproduction
wait for publication, then full FENCE, then ALIGN and RESUME inside their
`advance()` helper. That sequence delays successor submission until old cleanup
even when the runtime can provide a safe ALIGN earlier. It cannot measure the
latency benefit of handoff before cleanup. Preserve those historical inputs;
use a separately versioned harness for a new experiment.

The new harness should wait for publication, ALIGN, and acknowledged RESUME,
then submit the exact aligned successor. Retain the independent FENCE future
and record its physical-cleanup completion timestamp separately, without putting
that wait back on the successor submission path. Keep all times relative to the
same revision event; `fence_s` must continue to mean full drain, not ALIGN
readiness. Observe eventual FENCE completion and failures before declaring the
run complete. Measure revision-to-first-token and useful throughput as well as
receipt timing, while retaining the existing workload and baseline controls.

## Proposed: GPU-conditional layer groups

A conditional CUDA graph could check live authority on the GPU immediately
before each group of decoder layers. If every real row is obsolete, its
conditional body would skip that group. Later groups would also remain skipped,
while an unconditional completion path records that the forward was incomplete.
This could avoid queued future layer computation even after the CPU has submitted
the graph, without a host round trip for every group. CUDA provides conditional
graph bodies controlled by device code; integration must be checked against the
actual CUDA, driver, PyTorch capture, and allocator versions. See NVIDIA's
[CUDA Graphs documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

The first prototype should target TP1 and a wholly obsolete batch. Keep the
authority table and graph row identities at stable addresses, refresh identities
before replay, and read the current epoch at execution time. Do not capture a
Python authority decision as a permanent graph constant. A positive check admits
the next group; a revision immediately afterward can still leave that group
running. This is cooperative skipping of future work, not kernel preemption.

A batch containing a live tagged row or an untagged neighbor must execute the
group. Skipping individual stale rows would require separate row compaction or
compatible operator support. Native per-row guards and token acceptance gates
remain necessary. A skipped group cannot leave stale activation buffers to be
sampled or let the current unfinished chunk extend the accepted KV frontier.
Completion accounting must run even when every body is skipped, and ownership
must remain held until all submitted uses finish. TP support would additionally
require consistent decisions before rank collectives; independent rank-local
conditions are insufficient.

The expected benefit is less obsolete attention and MLP work after revocation.
The costs are condition kernels, graph/control overhead, extra capture variants,
and potentially smaller efficient execution units. Compare eager prefill,
existing breakable graphs, and conditional groups at several group sizes. Measure
both interruption savings and no-revision overhead. More frequent checks are
useful only when they avoid enough work to pay their cost.

## Other experiments

| Candidate | Existing foundation | Proposed change and tradeoff |
| --- | --- | --- |
| Safe handoff before reclamation | This branch adds local V2 sharing of committed full blocks while isolated old owners drain; FENCE still waits for terminal cleanup. | Qualify the first path, measure retained-memory costs, and investigate broader layouts or prefix coverage only with equivalent lifetime proofs. Retained old allocations may worsen admission under pressure. |
| Completion wakeups | Mailbox publication wakes its queue; FENCE/ALIGN completion is polled. A lone pending lifecycle operation uses a 10-ms polling interval. | Notify on completion rather than waiting for the next poll. Retain bounded fallback polling and shutdown/error behavior. Attribute delay before assuming this dominates. |
| Ticket buffer reuse | Token-commit metadata and attention row layouts are cached; compact tokens and verdicts share a transfer. | Pool temporary device and pinned-host buffers after ticket completion. Avoid allocation/zeroing/copy work without allowing a later step to overwrite an outstanding result. |
| Adaptive prefill chunks | Chunking already limits work per scheduling pass. | Use smaller chunks when interruption responsiveness matters and larger chunks for stable work. Compare saved obsolete work and neighbor latency against reduced GEMM efficiency and extra launch overhead. |
| Prefix locality | ALIGN can briefly protect already resident prefixes and yield optional GPU locks under pressure. | Favor a successor location or short residency window that avoids measured transfers. Do not eagerly restore every held prefix or weaken guaranteed prefix ownership. |
| Completed-prefix publication | Eligible async receipts decouple scheduling from accepted KV progress; publication waits for the latest reader. | Profile completed KV that remains unpublished, then reduce avoidable publication delay. Never infer completeness from allocated capacity or publish through an outstanding reader. |
| Revision bursts | Exact retries reuse their protocol records, and superseded work loses authority. | Avoid restoring or executing intermediate successors already superseded. Revoke promptly; preserve accepted revision records, explicit prefix promises, and ownership cleanup. Coalescing work is not silently deleting protocol obligations. |
| Recovery fairness | Recovery admission is bounded and existing work can drain. A later experimental policy shares recovery compute with waiting ordinary prefill. | Evaluate that policy independently from handoff changes, using ordinary and recovery latency distributions. A compute share cannot solve absent memory capacity and must not release promised reservations. |
| Mixed-batch stale work | Live rows prevent whole-batch exit; native predicates cover selected attention work/stores. | Compact stale rows at safe chunk boundaries or investigate revision-aware batching. Include packing, graph-shape, utilization, and live-neighbor costs. |

Implementation entry points are `execution_device.py` and `execution_guard.py`
under `tensorrt_llm/_torch/pyexecutor/`, the V2 scheduler and KV manager in that
tree, and `tensorrt_llm/execution_mailbox.py`. Existing breakable graph replay is
in `tensorrt_llm/_torch/pyexecutor/breakable_cuda_graph/`.

## Validation and decision order

First record publication, last submitted work, device completion, safe prefix
acquisition, FENCE/ALIGN availability, successor admission, first token, and final
reclamation. Distinguish cumulative timestamps from phase durations. Debug traces
perturb timing; use them for attribution separately from uninstrumented runs.

Start with the smallest representative cells from the existing Yanxin-style
replay suite, preserving its trace, seeds, model, native fixes, cache capacity,
provider, and declared numerical policy. Then replay the full comparison matrix.
Retain failed and stalled cells; report incomplete coverage rather than averaging
only successful cells. Synthetic tokens and injected revisions do not by
themselves establish application quality or a production-workload result.

For each candidate, compare native cancellation with ordinary prefix reuse,
tagged execution without revisions, and RETIRE with revisions. Change one
mechanism at a time. Exercise interruption before launch, within prefill, during
decode, and during cleanup, with both wholly obsolete and mixed live batches.
Include hot and cold prefixes, cache pressure, rapid supersession, graph replay
with changed identities, and cleanup failures. Verify output behavior against
both the matched baseline and the declared cold-reference contract, alongside
accepted frontiers, retained promises, and eventual reclamation.

Report revision-to-first-token, useful throughput, no-revision overhead,
ordinary-request and recovery p95/p99 latency, transferred KV bytes, and retained
capacity over time. A faster receipt alone is not a faster successor. Prioritize
completion wakeups and buffer reuse when their costs are visible; use the phase
breakdown to decide whether conditional graphs, chunking, locality, or handoff
restructuring addresses the dominant delay.
