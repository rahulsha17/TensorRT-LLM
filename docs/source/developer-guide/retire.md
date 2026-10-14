<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# RETIRE: execution versions and authority

RETIRE lets a runtime replace work when its context changes. The runtime decides
what to replace and supplies the revised input. TRT-LLM will use that decision to
stop obsolete work and safely hand useful KV to the replacement.

The Python LLM and executor APIs carry version metadata. A separate host-side
`ScopeRegistry` records which version is current and accepts revisions from the
runtime. An executor controller publishes accepted versions to a worker's host
table and GPU table. Tagged requests require this authority: admission, batch
entry, token commit, and result delivery reject obsolete versions. Revocation
requests ordinary cancellation; it does not prove that all GPU work stopped or
that KV is safe to inherit.

## Identity

`ExecutionVersion(tenant_id, scope_id, epoch)` names one version of a scope:

- **Tenant** is the namespace in which the runtime assigns scope IDs.
- **Scope** is the work the runtime replaces together. It can contain several requests.
- **Epoch** is the version number within that scope. The runtime must advance it
  for a revision and must not reuse a retired `(tenant_id, scope_id, epoch)`.

Request IDs still identify individual submissions. Two requests may have the
same execution version. The tag does not authenticate a tenant or establish that
a version is current.

`StateTag(version, generation)` additionally identifies an allocation generation.
Generation belongs to the resource owner and is independent of epoch. The type
is defined, but it is not attached to KV allocations yet. A state
tag is neither a memory hold nor proof of safe reuse.

Both types are immutable and support Python pickle serialization. Tenant and
scope IDs must be nonempty strings. Epoch and generation must be integers in
`[0, 2**64)`; Boolean values are rejected.

## Passing a version

```python
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.sampling_params import SamplingParams

version = ExecutionVersion(tenant_id="app", scope_id="task-42", epoch=0)
executor.get_execution_controller().register(version).wait(timeout=5)
output = executor.generate_async(
    [1, 2, 3], SamplingParams(max_tokens=16),
    execution_version=version, cache_salt="app-cache"
)
assert output.execution_version == version
assert output.cache_salt == "app-cache"
```

The Python LLM API accepts the same argument, using the controller of its backing
executor. Batched `generate` accepts either one
version for the whole batch or a sequence with one version per prompt. An entry
may be `None` for an untagged request. Omitting the argument preserves existing
behavior.

The request and its result associate the version with `cache_salt`. Both result
properties are read-only views of the submitted request. A revision can keep the
same salt to allow matching in the same cache namespace. Salt stays separate
from execution identity: requests in one scope version can use different cache
namespaces. Matching salt alone does not prove that KV is available or safe to
inherit; those checks belong to ALIGN.

The Python executor carries the tag along this path:

```text
LLM.generate / generate_async
  -> GenerationRequest (including proxy IPC serialization)
  -> bindings.executor.Request.py_execution_version
  -> request broadcast's Python metadata payload
  -> LlmRequest.py_execution_version (also copied to child requests)

GenerationRequest -> GenerationResult / RequestOutput.execution_version
```

The result handle keeps its identity for streaming, completion, and errors. The
tag is separate from `cache_salt`, native request IDs, and transfer attempt IDs.
Native request serialization does not include dynamic Python attributes, so
worker broadcast explicitly carries the tag in its existing Python metadata
payload. This is not a new C++ wire field or an HTTP/gRPC protocol field.

## Accepting a revision

The runtime decides when context has changed and which work to replace. RETIRE
checks that the runtime is replacing the current version. For example, if epoch
7 is current, a revision expecting 7 can advance it to 8. If another revision
already advanced it to 8, a competing revision still expecting 7 is rejected.
The runtime then decides whether its change still applies to the newer context;
it should not simply relabel an old revision with a newer expected epoch.

`ScopeRegistry` provides this check and update under one lock. It keeps separate
records for each `(tenant_id, scope_id)`. Cache salt and allocation generation
are not part of this lookup.

```python
from tensorrt_llm.execution_authority import Revision, ScopeRegistry
from tensorrt_llm.execution_version import ExecutionVersion

authority = ScopeRegistry()
old = ExecutionVersion("app", "task-42", 7)
new = ExecutionVersion("app", "task-42", 8)
authority.register(old)

# The runtime keeps this ID when retrying the same revision.
revision = Revision("edit-123", expected_version=old, successor_version=new)
receipt = authority.advance(revision)

assert authority.is_current(new)
assert not authority.is_current(old)
assert authority.snapshot("app", "task-42").pending_revision == revision
assert authority.advance(revision) == receipt  # Retry: no second advance.
```

Registration creates a scope at the runtime's chosen initial epoch. Repeating
registration at the current version is harmless; registering a different
version for an existing scope fails. Revisions must advance exactly one epoch
within the same tenant and scope. Unknown scopes fail instead of being created
by a revision.

`revision_id` identifies one runtime revision within a tenant and scope. It is
separate from request IDs and execution versions. Retrying the same descriptor
returns its original receipt, even if the scope has since advanced again. That
historical receipt does not make its version current again. Reusing the ID for
a different transition fails. A different ID claiming the same old version is
a competing revision, not a retry.

`Revision` identifies the transition. For a KV handoff, pass an immutable
`AlignmentInput` alongside it to `control.advance`, as shown in step 7 below.
The controller binds those exact tokens and cache namespace to the revision ID.
Retries must supply the same input, including whether ALIGN was requested.
An accepted successor remains pending until RESUME is acknowledged.

The registry is an in-memory object in one process. The runtime must route all
revisions for a scope through the same instance. Creating separate registries
on each worker would not provide one authority. Records and retry receipts are
retained for the registry's lifetime, so metadata grows with scopes and accepted
revisions. Scope deletion, retry-history cleanup, and recovery after restart are
not implemented yet; do not reset a registry while its versions are in use.

An accepted revision changes logical authority only. Its receipt is not proof
that workers saw the update, old work stopped, or KV is safe. Likewise,
`is_current` and `snapshot` are observations that can become stale immediately;
a check followed by a later launch or output is not an atomic authority gate.

## Publishing to a worker

Runtime integrations that own a `GenerationExecutor` should use its controller
to register and revise scopes. The controller owns one registry and schedules
publication as part of each call. Using a standalone `ScopeRegistry` still only
changes that local registry.

```python
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_version import ExecutionVersion

control = executor.get_execution_controller()
old = ExecutionVersion("app", "task-42", 7)
new = ExecutionVersion("app", "task-42", 8)

initial_publication = control.register(old)
initial_publication.wait(timeout=5)

result = control.advance(Revision("edit-123", old, new))
accepted = result.advance_receipt
observed = result.publication.wait(timeout=5)
assert accepted.revision.successor_version == new
assert observed.observed_version == new
```

There are two distinct results:

- **Acceptance:** the authority selected the new version. This happens before
  `advance` returns.
- **Observation:** the worker recorded that version, or a newer version of the
  same scope. This is what the publication handle waits for. It is not FENCE,
  GPU visibility, or permission to launch a successor.

For direct execution, a publisher thread updates the worker's host table. For
IPC execution, the existing authenticated RPC connection opens a dedicated
mailbox on first use. Subsequent updates use that mailbox's own listener,
independently of the normal request queue and model execution loop. Each socket
is created, used, and closed on its owning thread.

```text
Runtime -> executor controller -> ScopeRegistry accepts epoch 8
                       |
                       +-> publisher -> worker mailbox -> host table records 8
                                                |
                       publication handle <-----+ acknowledgement
```

The first supported setup is one frontend and one PyTorch rank, using the direct
`GenerationExecutorWorker` or classic IPC `GenerationExecutorProxy`. RPC/Ray
executors, attached frontends, and multi-rank configurations reject controller
creation. Publication starts only when the runtime explicitly requests the
controller. Untagged generation calls keep their existing behavior. Tagged work
requires explicit registration and a successful initial publication; missing
authority fails instead of silently running without checks.

Updates carry the full execution version, a publication sequence, and an
executor-session identity. Epochs retain their full 64-bit range. Worker tables
are isolated by tenant and scope; an unknown scope is not treated as epoch zero.
Duplicate updates are harmless, and delayed updates cannot move the table
backward. A worker can have only one authority owner for its lifetime.

An old revision retry returns its original acceptance receipt, but its
publication handle refers to the scope's latest accepted version. On a failed
publication, retrying registration or revision resends that latest version.
Acceptance is never rolled back. A timeout from `wait` alone does not cancel an
in-flight publication; its acknowledgement may still arrive.

The controller limits queued and in-flight publications to 256. When full, it
keeps the authority decision and returns a publication handle that raises an
error. The runtime can retry after capacity becomes available. Registry history
still has the lifetime described above. Shutdown fails outstanding waits and
closes the control channel before worker teardown. Restart recovery is not
implemented; a new controller cannot attach to an already owned worker table.

## Rejecting obsolete work and output

For a tagged request, the executor holds authority through the actual request
enqueue. Preparation such as constructing a native request happens beforehand.
The proxy checks against its controller; the IPC worker checks again against
its mailbox table before native enqueue. Unknown, older, and future versions
cannot enter through these gates.
Tagged proxy submission uses a nonblocking send. If the ingress queue is busy,
submission raises `RequestError` and can be retried; it cannot stall ADVANCE while
waiting for the worker to drain that queue.

ADVANCE immediately revokes old requests at the frontend. Result queue insertion
and ADVANCE take the same lock: the insertion either commits before the revision
or is rejected. This still works if publication to the worker is delayed or
fails. All requests in the revised scope are affected; other scopes and tenants
keep running.

```text
Runtime accepts epoch 8
  -> frontend closes epoch 7's output gate and wakes waiting consumers
  -> mailbox delivers epoch 8 to worker
       -> worker rejects new epoch 7 enqueues
       -> worker enqueues cancellation for admitted epoch 7 requests

Epoch 8 remains pending until RESUME is acknowledged.
```

Waiting result consumers receive `ExecutionAuthorityError`. Sync, async, and
streaming APIs also check buffered responses before handling them and again
before delivery. A revision during postprocessing prevents delivery of that
result. Postprocessing runs outside the authority lock so application hooks can
submit a revision without deadlocking.

The guarantee ends at the executor's queue and result-delivery boundaries.
Responses committed before ADVANCE cannot be recalled. Already returned Python
objects and effects performed by application hooks are not undone. Internal IPC
and postprocessing queues can still carry old responses; the frontend rejects
them before consumer publication. Applications must use the result APIs rather
than read their internal queues directly.

Cancellation uses the native request ID and the existing engine cancellation
queue. A direct worker requests cancellation at ADVANCE; an IPC worker does so
when its mailbox observes the update. Cancellation failure is logged without
restoring output authority. Scheduling iterations, kernels, and transfers already
in flight may continue until the engine processes cancellation. No new KV holds,
early resource releases or FENCE guarantees are introduced by these host gates.

ADVANCE publication alone does not allow successor submission. The successor
remains pending until ALIGN and RESUME complete. Registering the same version or retrying an
old revision does not bypass this restriction.

## GPU authority and token commit (step 5)

Creating the controller installs device guards on the worker. The implemented
scope is one GPU with single-beam `TorchSampler`:

| Part | Execution paths covered |
| --- | --- |
| 5A | Eager execution with KV manager V1 |
| 5B | KV manager V2, including its prefix reuse tree |
| 5C | Overlap scheduling with either KV manager |
| 5D | CUDA graph replay, `torch.compile`, and sampling inside the model graph |

These options can be combined where TRT-LLM itself supports the combination.
Controller setup still rejects multiple ranks, speculative decoding,
transfers/connectors, encoder-decoder models, guided decoding, recurrent KV,
and variable sliding windows. Request validation rejects multimodal input,
disaggregated requests, draft tokens, and logits callbacks. Those modes need
separate integration work; they are not silently run without guards.

The device table assigns a stable slot to each `(tenant, scope)` and stores all
64 epoch bits. It allows 8,192 distinct scopes per executor lifetime. Slots are
never recycled, and request/sequence IDs do not select authority slots. Cache
salt stays separate. Exhausting capacity fails publication; restarting the
worker is required before using a fresh table.

```text
Runtime ADVANCE
   |-> frontend revokes old output immediately
   `-> worker mailbox updates host authority
          `-> dedicated CUDA stream queues an atomic epoch update

Host authority check -> schedule -> model forward / graph replay
                                           |
                                           v
                                  sample candidate tokens
                                           |
                                           v
                       GPU guard copies authorized tokens only
                                           |
                                           v
                        host authority check -> commit tokens
                                           |
                                           v
                                 existing output gates
```

Publication uses a dedicated high-priority CUDA stream. Epoch writes and reads
use GPU-scope release/acquire atomics. A mailbox receipt means the worker
recorded the version and queued its device update. It does **not** wait for GPU
visibility or prove that old work finished.

Before scheduling, the executor checks the worker's host table. This does not
wait for GPU work. After sampling, a device kernel copies authorized rows into
a separate token buffer. Each sampling step owns its buffers and an immutable
snapshot of versions and sequence slots. A later batch cannot overwrite that
step's result by reusing the sampler's scratch buffer.

The executor waits for the result at its normal token-consumption point. It then
rechecks host authority while committing tokens. A revision after the GPU check
still blocks that host commit. A changed execution tag or sequence slot fails
the commit. Untagged requests and other scopes keep their tokens in a mixed
batch, with the sampler's original row order intact. Batches without tags skip
the extra token kernel and readback.

### V2 reuse (5B)

Context progress and V2 prefix publication each check host authority. Once the
worker observes a revision, the obsolete request cannot extend its published
prefix through these paths. A canceled prefill cannot return to generation.

Before ordinary cleanup, the executor waits for that request's recorded GPU
uses. V2 then stops committing that request's cache. This preserves the tokens
already committed, including a partial block when V2 supports it. It does not
append the uncommitted tail or clear the reuse tree. Other requests sharing the
prefix keep their own holds.

Without ALIGN, a prefix remains subject to ordinary cache eviction. Step 7
adds a native hold for the successor when revised input accompanies ADVANCE.

### Overlap (5C)

With overlap, the next forward can already be reading a request's KV when the
previous token result is rejected. The executor records completion after each
forward and sampling submission, including partial failures. Its existing stream
ordering makes the latest event cover earlier uses of the same request.

Cleanup waits for that latest event before releasing KV or request slots. A
failed or missing completion event prevents release. Ownership records use the
request object's identity, so recycled native IDs and sequence slots do not
alias an old request's record. GPU submission stays asynchronous; waits happen
at result consumption and resource release.

### Graphs and compilation (5D)

The model and any sampling inside its graph produce candidate tokens. The
authority kernel runs **after** the forward, outside capture and compilation.
Every replay reads the live authority table; it never captures an epoch as a
constant. This also covers compiled forwards and batches padded for graphs.

A lock excludes publisher CUDA allocations while the CPU submits, captures, or
compiles a forward. It is released when submission returns, without joining the
GPU. The publisher can then run while the model executes. Cold capture or
compilation can delay worker publication; the frontend output gate is already
revoked by ADVANCE.

### What step 5 does not prove

These checks are admission points, not kernel preemption. An already submitted
forward can still write KV, and sampling can write candidate scratch. Step 5
prevents rejected tokens from becoming committed request state; it does not
guard every attention/KV store. The local cleanup wait supplies the GPU completion proof for step 6 below.
Successors remain pending until RESUME is acknowledged.

Commit buffers are allocated per step. No performance gain is claimed yet.
Device failures never roll authority back. Publication failures close host gates;
device guard failures remain sticky and reject later guarded work. Staging
buffers are retained until their CUDA events complete. Unknown completion keeps
the affected buffers and ownership records alive. At most 256 unfinished device
operations may retain staging buffers; reaching that limit fails authority
instead of growing the queue indefinitely.

## FENCE (step 6)

A revision now has a separate fence handle:

```python
result = control.advance(Revision("edit-123", old, new))
result.publication.wait(timeout=5)  # Worker observed the authority update.
receipt = result.fence.wait(timeout=5)  # Old execution has drained.
assert receipt.request.revision.expected_version == old
```

FENCE covers every tagged request in the scope through the replaced epoch,
including earlier epochs still draining. It counts work from engine enqueue,
through waiting and active execution, until terminal cleanup. Additional return
sequences are counted separately before activation, so a parent finishing cannot
hide a running child. Other scopes, tenants, and untagged requests do not delay
this fence.

```text
ADVANCE closes old admission and output
  -> worker observes the new epoch
  -> queued requests are removed or proceed to ordinary cancellation
  -> active requests finish their recorded GPU uses
  -> terminal cleanup completes
  -> FENCE receipt is returned

Successor stays pending -> ALIGN -> RESUME
```

The receipt requires both closed worker admission and an empty drain record for
the affected epochs. Admission and polling share the worker authority lock;
enqueue registration also orders with terminal completion. The engine's existing
GPU waits happen outside those locks. Freeing resources for recomputation does
not complete a request. A canceled queue item counts as drained only after it is
actually removed from the waiting queue, before activation.

The same API works for direct and single-worker IPC execution. The publisher
queries fence status without blocking the worker mailbox on GPU completion.
Pending queries rotate so one busy scope does not prevent other fences from
completing. At most 256 fences are pending by default; overflow returns an error
handle and can be retried. Completed receipts are retained for the controller's
lifetime, like revision retry history.

A wait timeout leaves the fence pending. Retrying the same revision reuses its
successful or pending fence; a failed status exchange can be retried. A later
revision does not change the meaning of an older receipt. Shutdown fails pending
waits. Unknown enqueue completion or inconsistent request identity fails closed;
unknown GPU completion or failed cleanup never removes the pending request.

FENCE uses ordinary terminal cleanup. It proves that the old execution has
finished accessing its resources, but by itself does not reserve KV. When the
revision includes ALIGN input, the worker acquires a separate native hold before
releasing the old owner's resources, after the GPU completion wait. A fence receipt is neither an allocation-generation token nor
permission to run the successor. It also does not wait for obsolete IPC responses
to disappear; existing output gates prevent those responses from being delivered.

The supported configurations remain those of step 5. Multi-rank execution and
KV transfers/connectors need their own completion evidence before they can
participate in this protocol.

## ALIGN (step 7)

The runtime supplies the complete revised token input with ADVANCE. This lets
the worker reserve useful KV before cancellation frees the old owner's cache.

```python
from tensorrt_llm.execution_alignment import AlignmentInput

revised_input = AlignmentInput(
    input_token_ids=(1, 2, 3, 4, 9, 10),
    cache_salt="app-cache",
    lora_task_id=None,
)
result = control.advance(Revision("edit-123", old, new), alignment=revised_input)
result.fence.wait(timeout=5)
aligned = result.alignment.wait(timeout=5)
print(aligned.reusable_prefix_length)
```

The worker matches the revised tokens in its existing cache tree, using the
supplied salt and LoRA identity. Its manager fixes the model and KV layout.
Neither epoch nor scope ID changes the cache namespace. A token mismatch ends
the usable prefix; a changed salt or LoRA identity searches a different namespace.
The match can use compatible KV from other requests too. V2 uses prefixes published by its ordinary commit protocol. V1 also publishes the
old request's completed, block-aligned prefill prefix after draining its GPU uses,
before ALIGN probes the tree. A native frozen frontier bounds every subsequent
V1 store and release, including a zero-token frontier. An interrupted chunk that
did not finish all layers is never inherited.

A probe alone does not count as ALIGN. The worker acquires native ownership:

- **V2:** a suspended native cache holds the matched pages without taking an
  executor sequence slot. Held pages can migrate between cache tiers but cannot
  be dropped. Partial prefixes follow V2's existing reuse rules.
- **V1:** a cache-only native sequence holds the acquired prefix. It also needs
  one private tail block per applicable pool. It never runs a model or publishes
  new KV. V1 reports the acquired prefix, which can be shorter than its initial
  probe, and waits for any native onboarding copies before returning it.

The final revised input token is left for recomputation so the successor can
obtain logits. A cache miss is a valid zero-length alignment: the successor will
need to prefill the input. Set `min_reuse_tokens` to require a minimum prefix;
insufficient reuse then fails ALIGN instead. Invalid tokens and requests longer
than the engine's sequence limit fail validation.

```text
ADVANCE(new version, revised input)
  -> mailbox arms the reservation plan and revokes old work
  -> engine matches revised tokens and acquires native KV holds
  -> old request drains and releases its own resources
  -> FENCE completes
  -> ALIGN returns a reservation ID and the held prefix length

Successor remains pending until RESUME.
```

Reservation work runs on the engine thread, outside model capture and compilation.
A queue marker wakes an idle engine. Old-owner release and plan arming share a
lock, so an observed plan cannot be skipped between the reservation check and
resource release. Requests that finished before the worker observed the plan
can only contribute KV still present in the ordinary cache. ALIGN does not
promise to recover already evicted data.

A later revision invalidates the older reservation. The engine acquires a new
reservation before closing an older one, allowing repeated revisions to retain
their common prefix. Other scopes keep their own reservations. Waiting with a
timeout does not release a hold. Native acquisition failure is reported to the
runtime and never resumes the successor; unknown copy completion also blocks
old resource release. A failed acquisition remains failed for that revision.
Use a new revision for different input or a different reuse requirement.

Reservations are bounded to 256 scopes per worker. A replacement in the same
scope can temporarily hold both versions while acquiring the new prefix.
Revision inputs and receipts are retained with controller retry history for its
lifetime; this metadata is not yet compacted. Shutdown invalidates reservations
immediately, then closes native ownership on the engine thread before manager
teardown. Cache reset is rejected while ALIGN owns or prepares a prefix. Engine
control actions that may change weights or cache storage invalidate ALIGN; a
fresh executor is required for further alignment after those actions.

The receipt is a record of preparation, not a permanent right to use memory.
RESUME must check that its version and reservation ID are still live, verify the
submitted input, and acquire the successor's own KV before releasing this hold.
No GPU pointers are exposed. Successor submission is still rejected at this
step, and no performance gain is claimed before native validation.

## RESUME (step 8)

RESUME lets the runtime submit the revised request after ALIGN is complete:

```python
from tensorrt_llm.sampling_params import SamplingParams

aligned = result.alignment.wait()
control.resume(aligned).wait()
successor = executor.generate_async(
    prompt_token_ids=list(revised_input.input_token_ids),
    sampling_params=SamplingParams(max_tokens=64),
    cache_salt=revised_input.cache_salt,
    execution_version=aligned.request.revision.successor_version,
)
```

The worker rechecks the session, current version, reservation ID, and completed
FENCE. Only after its acknowledgement does the frontend clear the pending
revision and allow submission. A timeout does not roll back worker acceptance;
retry `resume` with the same receipt. A newer revision always wins over a late
RESUME acknowledgement.

The successor must use exactly the tokens, cache salt, and LoRA identity supplied
to ALIGN. The worker checks them before scheduling and binds the reservation to
one physical request. A changed prompt, another request ID, or an invalidated
reservation is rejected. This step supports one returned sequence per revision;
additional physical requests need their own revision or scope. Once claimed,
a failed or canceled successor requires a new revision to try a different request.

V1 and V2 use their normal request allocation paths to acquire their own cache
references while the ALIGN hold is still present. Before the first model
forward, RETIRE verifies that the acquired prefix is at least the length ALIGN
promised. If it is shorter, the request fails before model execution. An aligned
cache miss (zero tokens) allows ordinary recomputation. Later scheduler-driven
recomputation of the same request uses the normal cache path.

```text
ALIGN hold remains alive
  -> RESUME acknowledgement opens admission
  -> validate revised input and bind request ID
  -> normal V1/V2 allocation acquires successor KV
  -> verify prefix before first forward
  -> execute with all authority checks still active
```

This check runs at the common forward boundary, outside model capture. Eager,
compiled, CUDA graph, and overlap execution use the same handoff logic; RESUME
does not remove the host or device fences, including for in-graph sampling.
Native GPU runs are still required to validate these combinations end to end.
The existing one-rank, text-only, single-beam restrictions still apply; this
does not add disaggregated transfer, speculative decoding, or multimodal support.

RECLAIM releases the extra ALIGN hold after successful attachment, as described
below. Cache resets remain blocked while a pending hold protects a prefix.

## RECLAIM (step 9)

RECLAIM runs automatically on the engine thread. Once RESUME verifies that the
successor acquired the promised prefix, it waits for the cache allocation copies
to finish, then releases only the separate ALIGN hold. The successor keeps its
own references. V1 removes the cache-only sequence without publishing its unused
tail; V2 closes the suspended cache without closing the successor cache.

A claimed successor that terminates before attachment also releases its hold at
the next engine safe point. Terminal notification itself never calls CUDA.
A later revision acquires its new hold before releasing an older unused hold.
An unsubmitted revision stays reserved until the runtime supersedes it or closes
the executor; a timeout alone does not abandon it.

Copy-wait or native-release failure stops further cleanup and requires worker
restart. The ownership record remains present rather than assuming release was
safe. Successful release returns reservation capacity. A small retry record
remains, so replaying RESUME cannot acquire a second hold or bind another request.
This is KV reclamation, not deletion of scope or epoch history.

These operations stay outside graph capture and use the same boundary with
V1/V2, eager, overlap, graph, and compiled execution. The copy wait occurs once
per handoff, not on every decoding step.

## Measuring prefix residence

For V2, the worker RPC `fetch_kv_cache_residency(input_tokens, cache_salt,
lora_task_id)` reports the pages that match those tokens in the ordinary reuse
namespace. It returns the matched token count, sampling timestamps, cache-tier
names, and each block/lifecycle's tier, slot and readiness. The native method
`KVCacheManager.probe_reuse_residency` takes the manager's shared lock and returns
copied values. The probe does not reserve, restore, or change eviction priority.
It requires a matching native build.

Use `cache_tiers[page.cache_level] == "GPU_MEM"` together with `page.ready`
to identify pages ready in HBM at the time of the sample. A tree hit can include
host or disk pages. Missing pages have no tier or slot; windowed caches may
legitimately omit old pages. Report each lifecycle separately rather than adding
their token counts together. `matched_token_count` clips each page to the query's
matched prefix; `recorded_token_count` describes the underlying page.

Sample before and after cache pressure, and before RESUME, to show migration
without triggering restoration. These observations can become stale immediately
after the query. A slot number is not an allocation identity or permission to
read memory; ALIGN's live reservation still establishes successor ownership.
The probe observes the supplied tokens, including the final token if present;
RESUME may recompute that token for logits. V1 does not support this RPC.

## Early exit during prefill

For an interruption test, resolve the frontend request ID through the worker's
`get_runtime_request_id` RPC before matching iteration `requestStats` IDs.
The mapping exists only while the backend request is live. It does not prove
that a reported prefill position is still the active position when a revision
arrives.

Set `TLLM_LOG_LEVEL=info` before launching the worker to retain early-exit events;
the default error level filters them out. Missing events at that level do not
show whether an exit happened.

An actual early exit emits `RETIRE_EARLY_EXIT` with frontend and runtime request
IDs, the prefill position, and the execution boundary. For `eager_layer`,
`submitted` and `total` count decoder layers; for `graph_segment`, they count
replay segments. A `prelaunch` event does not prove a mid-forward stop. These
counts describe work that was submitted, not GPU completion or kernel
preemption. FENCE still establishes completion separately.

For a separate timing diagnostic, enable `TLLM_LOG_LEVEL_BY_MODULE=debug:_torch`.
`RETIRE_TRACE` records worker-local monotonic timestamps for prefill submission,
each supported boundary and its authority verdict, and the device publisher's
entry after the host authority changed. This distinguishes a revision that
arrives while CPU submission is still running from one that arrives after the
forward has already been submitted. A forward-end record does not mean the GPU
has completed. Debug logging perturbs timing; do not pool these diagnostic runs
with uninstrumented performance measurements.

REVISION can now save unfinished computation as well as retain completed KV.
The runtime still calls ADVANCE with the revised input; no new runtime API is
needed. Chunked prefill must be enabled and the prompt must span multiple chunks
if the revision is to inherit an earlier completed chunk.

```text
Chunk 1 completes all layers -> accepted prefix frontier
Chunk 2 starts -> runtime sends ADVANCE
  -> next supported execution boundary observes revocation
  -> stop submitting the rest of the wholly revoked batch
  -> record completion of work already submitted
  -> drain old GPU uses
  -> publish only the accepted, completed prefix and retain its pages
  -> ALIGN acquires ownership -> FENCE/ALIGN receipts -> RESUME
  -> successor prefills the remaining input
```

| Execution path | Early-exit boundary |
| --- | --- |
| Eager Qwen3 prefill, at least 32 scheduled context tokens | Before each group of four decoder layers |
| Breakable prefill CUDA graph | Before each replay segment |
| Monolithic CUDA graph or compiled model | Before submitting the model/graph |
| Other eager models | Before submitting the model |

Both KV managers and eager/overlap schedulers use the same batch guard and
completion tracking. These boundaries are implemented, not yet performance
qualified. Already-launched kernels are not preempted. Compilation and capture
never record a Python authority decision into the graph. A monolithic graph has
no internal host interruption point; adding one requires graph partitioning.

V2 context progress and ordinary tree publication occur at separate executor
steps. If a revision arrives between them, cleanup first waits for outstanding
GPU work, then publishes complete blocks up to the previously accepted context
frontier. It never derives that frontier from allocated capacity or the current
unfinished chunk. A suspended cache keeps its already committed prefix without
being restored just for this step. Other owners keep their existing holds.

Only a batch whose every real row has lost authority may exit. A live request
from another scope or an untagged neighbor makes the complete batch continue.
Output and context-commit gates still reject the obsolete rows. This preserves
live neighbors but does not eliminate the stale row's dense computation in a
mixed batch. Per-row compaction is separate work.

Layer checks use a separate read-only authority observation lock. They never
acquire the worker's main authority lock while holding the device-submission
lock. A positive observation is advisory; admission and commit still use their
existing gates. Short decode steps do not run the per-layer polls.

An early-exit signal skips sampling and context advancement for that batch. It
is not a model failure and does not cancel other active requests. The guard's
completion events remain live until normal terminal cleanup. Source preparation
waits only for obsolete executions in the revised scope, not unrelated scopes.
FENCE still requires cleanup; this change does not move unsafe release past RESUME.

**Build requirement:** the frozen V1 frontier adds a native request field and
binding. Rebuild the C++ library and Python bindings from this branch together.
The earlier performance campaign's wheel plus a Python-only overlay is not a
matching build for this change.

For validation, run the existing execution mailbox/guard/alignment suites, the
native `LlmRequestTest.retiredKvFrontierCannotGrow` and
`KVCacheManagerTest.RetiredPrefillPublishesOnlyFrozenCompletePrefix` tests, and
`tests/unittest/_torch/executor/test_execution_device.py` on CUDA. The host tests
cover early-exit unwinding, mixed/live neighbors, blocked publication, zero and
partial frontiers, graph replay cleanup, and failed completion evidence. The
CUDA test checks that skipped graph segments do not write and that the graph
can subsequently replay normally.

The next performance campaign must interrupt during active prefill, verify the
observed completed frontier and skipped work, and compare against native prefix
reuse as well as cold restart. Submit the revision's successor as soon as its
handoff permits, without waiting for unrelated cache churn. Report
revision-to-first-token, no-revision overhead, and live-neighbor latency. The
older post-prefill campaign does not measure these new hooks, and its speedup
numbers must not be presented as validation of this change.

## Testing

### Native attention predicates

`thop.attention` accepts an optional set of four tensors: the current epoch
table, per-row scope slots, expected epochs, and rejection counters. All four
must be present together. `TrtllmAttentionMetadata` carries these inputs to the
fallback FMHA provider; other providers reject them. Null inputs retain the
ordinary attention path.

The native implementation covers source-built SM90 BF16 attention with head
size 64 or 128, paged KV, and a single beam. It checks authority before eligible
attention tiles and before attention-output and KV stores. Epoch reads use
GPU-scope acquire atomics, matching the device publisher. Untagged live rows
use the exact sentinel `(slot=-1, expected_epoch=0)`. Multi-block reductions
still complete their scratch writes and semaphore arrivals; they cannot take
independent early exits without stranding other blocks.

These predicates do not establish FENCE or permit resource release. Callers
must retain the authority tensors and KV until all submitted GPU uses finish.
Captured graphs require stable addresses, with row identities refreshed before
each replay. Existing cubins without the authority ABI cannot serve a guarded
call. Rebuild native libraries and bindings together.

**Integration status:** the executor does not yet bind its version table to
these attention metadata fields. Automatic binding, overlap lifetime handling,
and graph initialization remain required before real-model runs exercise the
native predicates. This is an operator implementation and qualification step,
not a claim of end-to-end operator suppression.

`tests/unittest/_torch/attention/test_execution_native_guards.py` checks V1/V2,
fresh/cached context, short/long generation, mixed batches, live tagged and
untagged neighbors, epoch wrap across the signed boundary, malformed metadata,
and captured decode replay after identity changes. It compares live outputs
exactly with the unguarded operator and verifies that revoked rows leave output
and KV sentinels untouched. The existing H100 attention test-list entry includes
this file. Native build and GPU results are required to qualify the port.

### Executor lifecycle

The CPU CI list includes authority, mailbox, IPC, host gates, and executor wiring.
Guard tests cover revisions during result waits, mixed scopes, context-only
chunks, V2 commit races, slot changes, recycled request IDs, overlapping readers,
partial submission failures, and refusal to free without completion. The mode
matrix checks configuration acceptance; it does not establish GPU correctness.
They also check that an old result cannot alter a reused slot's pending-token
counter, and that publication cannot issue CUDA calls during capture submission.

`tests/unittest/_torch/executor/test_execution_device.py` runs without model
weights on CUDA. It covers uint64 epochs, scope isolation, compact/full token
layouts, asynchronous tickets, scratch-buffer reuse, live authority after graph
replay and compilation, and the native sampler consuming an overlap batch that
still contains a canceled request. A captured native sampling test checks that
in-graph candidates still pass through live authority after replay.
V2's native test suite checks that stopping
one owner preserves a shared partial prefix without publishing the tail.
Both GPU test directories are included in the H100 CI list.

Local isolated host checks can run without torch or the native bindings, using
test doubles for those dependencies. They are not a substitute for the native
CPU suite, CUDA tests, or real inference. GPU validation must include V1/V2,
overlap on/off, eager/compiled/graph execution, and in-graph sampling, with
revisions during prefill, generation, and cleanup.

## Next steps

Validate the complete lifecycle on native GPU runs across V1/V2 and execution
modes. Distributed authority, disaggregated transfers, scope-history compaction,
and recovery after restart remain separate work. Authority checks stay active
after RESUME so another revision can supersede the new work too.
