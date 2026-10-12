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

The descriptor currently contains version metadata only. Its pending successor
is a record of the latest accepted revision, not a runnable request. Revised
input and the KV inheritance contract will be added with ALIGN. The runtime
must keep a revision ID bound to the same revision; this metadata layer cannot
compare input that has not been supplied to it.

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

Epoch 8 remains pending until ALIGN and RESUME are implemented.
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

An accepted successor cannot be submitted yet, even after its publication is
acknowledged. It remains pending. Registering the same version or retrying an
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

A prefix can remain available through ordinary cache reuse, but it is still
subject to eviction. Step 5 does not reserve it for a successor. ALIGN will
provide that handoff and validate which prefix the successor may inherit.

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
guard every attention/KV store. The local cleanup wait is not the successor FENCE
protocol. Successors remain pending until FENCE, ALIGN, and RESUME are implemented.

Commit buffers are allocated per step. No performance gain is claimed yet.
Device failures never roll authority back. Publication failures close host gates;
device guard failures remain sticky and reject later guarded work. Staging
buffers are retained until their CUDA events complete. Unknown completion keeps
the affected buffers and ownership records alive. At most 256 unfinished device
operations may retain staging buffers; reaching that limit fails authority
instead of growing the queue indefinitely.

## Testing

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

The next step is FENCE: prove that old work can no longer access the memory we
intend to preserve or release. Then ALIGN establishes the successor's KV,
RESUME allows it to run, and RECLAIM releases obsolete holds. RESUME will keep authority
checks active so another revision can supersede the new work too.
