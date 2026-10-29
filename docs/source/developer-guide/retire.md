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
If ADVANCE omitted the input, `control.align(revision, revised_input)` can bind
it later while the revision is still current. This is a separate operation;
changing the arguments of an ADVANCE retry remains an error.
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

The execution guard requires one frontend and the PyTorch backend. The direct
`GenerationExecutorWorker` and classic IPC `GenerationExecutorProxy` have model
coverage. The RPC and Ray paths now use the same worker mailbox and frontend
authority gates; their model qualification is still pending. Ray additionally
connects lockstep tensor-parallel workers through the bootstrap below. Attached
frontends, PP, CP, attention DP and multi-rank MPI/RPC remain unsupported. Publication starts
only when the runtime explicitly requests the
controller. Untagged generation calls keep their existing behavior. Tagged work
requires explicit registration and a successful initial publication; missing
authority fails instead of silently running without checks.

RPC and Ray register result routing before sending a request. Submission runs
asynchronously so a congested ingress cannot hold the authority lock and delay
ADVANCE. The worker rechecks authority when the request arrives, then replies
with an admission acknowledgement. Admission errors, including the worker
traceback, and transport failures reach the result. The RPC client registers the
reply future before sending, so a fast acknowledgement cannot be dropped while
the send is still pending. A successful acknowledgement
does not finish the request or establish FENCE; tokens use the response stream
and pass the frontend authority gate before entering the consumer queue. A late
admission error cannot affect a replacement result. Shutdown closes authority
before tearing down workers.

`CollectivePublicationTransport` provides the control-plane aggregation needed
for multi-rank integration. The executor supplies the complete expected set of
`ExecutionParticipant(tp_rank, pp_rank, kv_group)` coordinates and one authenticated
endpoint for each worker. Missing, duplicate, or extra participants are rejected;
the set cannot shrink when a worker fails.

```text
                         +-> TP/PP worker 0 -> local observation/drain/hold
controller -> transport -+-> TP/PP worker 1 -> local observation/drain/hold
                         +-> remaining required workers
             |
             +-> acknowledge only after every required worker answers
```

Publication and FENCE require every participant. ALIGN chooses the shortest
reserved prefix and RESUME binds that common lookup limit on every worker.
The transport retains each worker's reservation ID and translates the aggregate
receipt back to those exact local receipts on RESUME. The frontend stays closed
if only some workers acknowledge. A retry rechecks every local reservation;
an acknowledgement timeout does not release memory. A later revision or worker
teardown reclaims abandoned holds through the existing worker lifecycle.

This transport is tested with local tables and authenticated IPC mailboxes. It
is connected to RPC/Ray bootstrap, and does not itself coordinate model
collectives or per-layer exit. Ray TP attaches the engine coordination described
below; other distributed topologies remain unsupported. Closing
the transport only closes channels; it does not certify drain or native cleanup.

`ExecutionStepConsensus` supplies the next part of that integration: an engine
boundary vote over a fixed set of ranks. Votes carry the authority session,
step number, boundary, ordered request versions, and logical progress. A row
can proceed only if every rank accepts it. Missing ranks, different rows or
progress, and failed collectives invalidate the coordinator until restart.
Invalid local input still enters the collective so peers can observe failure.

### One authority across executor frontends

A runtime that owns several executor frontends in one process can bind them to
one controller before any frontend admits requests or creates its own authority:

```python
from tensorrt_llm.executor.execution_group import create_shared_execution_controller

controller = create_shared_execution_controller((first_executor, second_executor))
controller.register(version).wait()
assert first_executor.get_execution_controller() is controller
assert second_executor.get_execution_controller() is controller
```

The frontends share one registry and one admission/result gate. A revision
accepted through either frontend immediately revokes old host results on both.
Each worker group keeps its existing TP coordination. Publication, FENCE and
RESUME require every group, including every rank inside each group.

Bootstrap locks all frontend admission paths until every member is attached.
Workers report an immutable identity for their bound table over authenticated
IPC. Two handles for the same worker cannot count as two participants, even
when one handle is local and the other uses IPC. The worker identity query uses
a temporary connection; the publisher creates its own socket on its own thread.
A partial bootstrap failure disables all member frontends until restart.

The group is fixed for that authority's lifetime. Closing any member's control
path closes the shared controller and blocks new submissions on all members.
Shut down each executor normally to stop its workers and finish cleanup;
closing authority alone does not prove that GPU or transfer access has ended.

This supplies a common owner for separate placements, but does not yet enable
disaggregated RETIRE. Existing transfer and connector guards remain in place.
Untargeted ALIGN reserves on every member and uses the shortest common prefix.
A destination can instead select one prepared placement, as described below.
Transferred-content validation and physical transfer qualification are still
required before prefill/decode transfer can be enabled.
For frontends in separate runtime processes, use the external owner described
below. Independent local controllers with matching epochs are separate authorities.

### Frontend control across processes

The runtime prepares each frontend, collects the descriptors through its
authenticated bootstrap, and creates one owner:

```python
# Run in each frontend process before any request admission. Use the same fresh
# session_id, chosen by the runtime. For another host, supply its concrete IPv4
# address as bind_host so the owner can reach the two control listeners.
from tensorrt_llm.executor.execution_member import prepare_execution_member

member = prepare_execution_member(executor, session_id)

# Run in the runtime process after collecting every member descriptor.
from tensorrt_llm.executor.execution_remote import create_remote_execution_controller

controller = create_remote_execution_controller(tuple(member_descriptors))
controller.register(initial_version).wait()
```

The returned controller has the same `advance`, `align`, and `resume` methods as
a local owner. Each prepared executor uses its remote frontend gate for admission
and result delivery. Calling `get_execution_controller()` on that executor raises
an error; it cannot silently create another owner. Exact preparation retries
return the existing descriptor. Membership is fixed, and bootstrap rejects two
frontends that report the same worker table, including nested worker groups.

Each frontend has two authenticated listeners: one for host authority and one
that forwards publication, FENCE and ALIGN to its existing workers. Worker
connections stay on their own listener thread. A blocked worker call therefore
cannot prevent host sealing or lease expiry. Endpoint identity includes the
authority session and a unique frontend lifetime. A replacement process cannot
accept messages for the previous lifetime. Keep the descriptor's authentication
keys inside the trusted runtime control plane.

The order is: seal every frontend, accept ADVANCE at the owner, publish
to workers, complete FENCE and ALIGN, then activate the acknowledged RESUME.
Sealing immediately rejects old admission and output and wakes old consumers.
It does not wait for a worker and does not certify that GPU or transfer access
has stopped. A canceled request can still have resource claims keeping FENCE
pending. The owner validates the expected epoch and revision ID before sealing,
then repeats the registry CAS after every seal ACK. Invalid or historical
revisions do not seal the current epoch again. A scope is declared before its
initial worker publication, so it can be revised while that publication is still
in flight.

Observation alone cannot admit a successor. Activation must name the exact
observed publication, and a revised scope requires RESUME. Duplicate seals and
delayed activations cannot restore an older epoch. Each endpoint binds one owner
incarnation; fresh heartbeat sequences renew its lease. Replayed heartbeats do
not extend it. The default lease is 30 seconds with a five-second control timeout.
Heartbeats run independently of publication and renew all frontends in parallel.
Expiry closes host authority permanently and wakes consumers; it never releases
KV memory. Losing a member fails the owner and closes the reachable host gates.
An unreachable frontend relies on its lease; owner loss is not instantaneous
across a network partition.

On partial sealing or host activation failure, the owner closes authority and
fails outstanding waits. It never rolls back to the old epoch. A newer revision
may overtake an in-flight RESUME; the delayed activation cannot reopen that
successor. Normal shutdown closes the controller and each executor. Recovery
requires a fresh runtime session and worker teardown: closing host authority or
waiting for its lease is not proof of safe physical reclamation.

The control path is tested over real IPC and separately spawned processes with
worker version tables, drain tracking and ALIGN reservations. It preserves the
existing supported executor topologies. Existing transfer and connector guards
remain enabled. Host tests do not qualify GPU execution or disaggregated transfer.

### Choosing where the successor runs

The runtime can choose one frontend as the successor's destination. A placement
identifies that frontend lifetime and all of its workers; it is separate from a
scope, execution epoch, request ID and cache salt. Use the identity returned by
bootstrap, rather than constructing one from a network address or GPU number:

```python
# The remote runtime already collected the complete set of member descriptors.
revised = AlignmentInput(
    tuple(revised_tokens),
    cache_salt="conversation-123",
    destination=members[1].placement,
)
advanced = controller.advance(revision, alignment=revised)
advanced.publication.wait()
advanced.fence.wait()
aligned = advanced.alignment.wait()
controller.resume(aligned).wait()
# Submit the successor through members[1]'s frontend process.
```

For frontends sharing one runtime process, use `frontend.get_execution_placement()`
after `create_shared_execution_controller(frontends)`. The destination can also
be supplied to a deferred `controller.align(revision, revised)` call.

Every member still receives ADVANCE, revokes old work, and contributes to FENCE.
An outstanding transfer at the old placement therefore keeps FENCE and ALIGN
pending even if the destination already has its prefix. Only the destination
reserves KV, unless the runtime also names a transfer source as described below,
and admits the successor after all publication acknowledgements.
Other workers record the current epoch with execution disabled. Their request
admission, host commits and early-exit observations remain closed for that epoch;
matching old completion evidence can still retire resources. ADVANCE without
ALIGN also leaves worker admission disabled until RESUME.

Within the destination, its TP ranks must agree on a common prefix as before.
Other frontends' cache misses no longer shorten that prefix. Omitting the
destination retains the all-placement common-prefix behavior. An unknown or
foreign-session destination is rejected before ADVANCE seals any frontend or
changes the registry. Retries must keep the same destination; moving a successor
requires a new epoch. Old ALIGN receipts cannot authorize the new placement.

This selects locally owned KV; it does not copy KV from the old placement.
Transferred bytes still need content/layout validation and physical retirement
proof before disaggregated RETIRE can be enabled. Host/IPC tests cover destination
selection, rank agreement, global FENCE, disabled non-destinations, partial failure
and retargeting. Real-model and transfer qualification remain required.

### Retaining a prefix at a different source

The runtime can add a `PrefixTransferPlan` to the revised input:

```python
from tensorrt_llm.execution_alignment import PrefixTransferPlan

revised = AlignmentInput(
    tuple(revised_tokens),
    cache_salt="conversation-123",
    destination=members[1].placement,
    transfer=PrefixTransferPlan(source=members[0].placement, num_tokens=1024),
)
```

Both placements must belong to the prepared authority cohort. The prefix must
cover `min_reuse_tokens` and leave the last input token to recompute. Source,
destination, prefix length and revised input are fixed for this revision; a
change requires a new epoch.

The source receives ALIGN with execution disabled. Its engine retains a real
cache reservation before dropping the old request's hold. After global FENCE,
that source can obtain send permits for the planned prefix; the destination can
obtain receive permits for the same prefix. Neither permit allows generation.
An ordinary cache hit cannot satisfy the transfer plan: every source and
destination rank must complete and explicitly accept its copy before global
ALIGN returns. A source acquisition failure is reported even when no copy has
started or another endpoint is still pending.

RESUME opens execution only at the destination and retires the source hold at an
engine safe point. The source retains its acceptance receipt for lost-ACK
retries, but cannot start another copy or execute the successor. A revision
during transfer closes admission while keeping the hold until physical cleanup.
The destination keeps its hold until the successor acquires the promised prefix.

This implements control routing and source retention, not destination import.
The current cache adapters reserve existing prefixes; a cold destination still
fails acquisition. Runtime staging allocation, page selection, transport setup
and validated cache attachment remain required before enabling this path in
disaggregated serving. Local, IPC and nested collective tests use reservation
fixtures and are not GPU transfer qualification.

### Coordinating model execution

The device guard coordinates scheduling, token/context commits, and early exit.
Empty sampler results and untagged batches still enter their expected votes.
Device waits finish before acquiring host authority. A failed wait, missing
ticket or closed table is reported in the vote peers are expecting. Commit
completion checks include the resulting token count and last token, so equal
lengths cannot hide different accepted samples. V2 prefix preservation has a
completion vote after its device waits, outside the host-authority lock.

Forward entry checks row order and progress even with early exit disabled.
Forward completion distinguishes a normal return from an early exit and includes
the exact exit boundary. GPU-use records remain live until normal cleanup.
Physical allocation identifiers remain local; different sequence slots are
valid. The native context-chunk accessor is read only during context, since
decode and terminal requests no longer permit it. Single-rank execution does
not enter a collective.

Mirrored admissions use another agreement point, after restoring request
versions and before adding requests to the waiting queue. The submitting rank
must still hold the original enqueue; every peer records the same request in
its drain tracker. All ranks acknowledge the envelope before any can activate
or finish it. This prevents a late peer from receiving a request after the
source has already reported drain. Revoked versions remain tracked until
cleanup; recording the envelope does not restore their authority. A failed
acknowledgement invalidates drain evidence even if local cleanup later finishes.

Loop checks cover the stop decision, whether a new batch can run, and whether
an overlap sampler batch remains to be consumed. They compare both batches'
rows before any rank takes the corresponding branch. An unexpected error
outside a guarded step cannot use request-local recovery when rank coordination
is attached. The existing executor crash watchdog is armed before invalidating
authority and drain evidence, then normal cleanup runs. Cleanup does not turn
a failed distributed execution into a successful FENCE.

Ray TP bootstrap now attaches these hooks before the first request, including
untagged requests. It first checks every worker without changing engine state.
Then rank 0 broadcasts a control request. All engine threads park at that same
scheduler boundary, with device work drained. The worker RPC threads create a
separate CPU Gloo group and install local guards and authenticated mailboxes.
Every worker must acknowledge readiness before any engine resumes. Frontend
admission stays blocked throughout setup. A partial setup or timeout invalidates
the frontend and kills the entire actor set; ordinary inference cannot continue
on a partly configured group.

```text
Frontend: block admission -> check every worker -> start all bootstrap RPCs
Workers:  park engines -> create Gloo group -> install guards -> agree ready
Frontend: verify identities/endpoints -> create controller -> allow admission
Runtime:  publish to peers -> publish to rank 0 -> broadcast scheduler wake
```

Preflight RPCs have a 30-second deadline; bootstrap has a 90-second deadline.
Engine votes use a separate, bounded 300-second host collective, allowing for
uneven compilation time. Worker mailboxes bind concrete IPv4 addresses. Rank 0
receives publications last, so peers arm ALIGN or reclamation before its wake
reaches them. Only rank 0 enqueues these wakes. The host group closes after the
engine shuts down.

After bootstrap, the frontend checks every Ray worker once per second with a
30-second RPC deadline. A dead actor, failed engine thread, wrong session or
missing reply closes frontend authority and asks Ray to kill every actor without
restarting it. Pending requests receive terminal errors, and late responses and
new submissions are rejected. RPC response-stream errors and unexpected stream
termination also fail pending requests. Normal shutdown stops health polling
before stopping the engines. These checks use host state only: a healthy reply
is not a FENCE acknowledgement or evidence that GPU access has ended. Failure
does not authorize KV reuse from the failed cohort; recovery requires a fresh
executor. Actual GPU fault qualification remains required.

Two/four-process CPU Gloo tests exercise agreement, mismatched phases and a
missing peer. Simulated-worker tests cover partial installation and admission
races. Real Ray TP model qualification is still required. ALIGN chooses the
shortest prefix covered by every worker's reservation. RESUME sends that length
alongside each worker's original receipt and opens frontend admission only after
all workers acknowledge it. Pipeline stages require separate ordering and remain
rejected.

For example, if two workers hold 128 and 256 compatible tokens, both successors
start at token 128. The native V1 match and capacity estimates and the V2 cache
lookup honor the same request-bound limit. This limits allocation itself; it
does not rewind a cursor into shared blocks. The first forward requires exactly
the agreed count before releasing either original hold. A zero common prefix
means both workers recompute. A runtime minimum above the available prefix fails
ALIGN. Lost acknowledgements can be retried, but cannot change the chosen length.

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
scope uses single-beam `TorchSampler`, on one GPU or Ray TP with coordinated
bootstrap. Ray TP model qualification is pending:

| Part | Execution paths covered |
| --- | --- |
| 5A | Eager execution with KV manager V1 |
| 5B | KV manager V2, including its prefix reuse tree |
| 5C | Overlap scheduling with either KV manager |
| 5D | CUDA graph replay, `torch.compile`, and sampling inside the model graph |

These options can be combined where TRT-LLM itself supports the combination.
Controller setup still rejects unsupported distributed topologies, speculative decoding,
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

Successor stays pending until ALIGN -> RESUME; eligible local isolation can
complete those steps while the drain above is still pending.
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
releasing the old owner's resources. The local isolated path can acquire that
hold before old GPU completion. A fence receipt is neither an allocation-generation token nor
permission to run the successor. It also does not wait for obsolete IPC responses
to disappear; existing output gates prevent those responses from being delivered.

Worker-local resource claims keep FENCE pending when a transfer outlives the
request's terminal cleanup. The transfer manager registers a separate claim for
the transceiver and connector before releasing the sequence slot or pinning KV.
Each accessor releases its claim only after completion; the last claim follows
successful native unpinning. A dependent accessor can take a claim from a live
parent even after the request leaves the executor. The transition is atomic with
FENCE polling, and a reused request ID cannot release an earlier allocation's
claim. Uncertain setup or release makes drain unknown and requires a restart.

Python transfer sessions serialize the entire close operation, including
auxiliary-slot release, bounce cleanup and unregistering the session. A duplicate
close cannot report success while those operations are running. If cleanup
raises after a possible side effect, later close attempts keep failing and do
not repeat that cleanup. A refusal because DMA or retirement is still pending
can be retried once it drains. This makes successful close a usable lifetime
boundary for resource claims.

The Python transfer worker takes a separate claim before constructing each tagged
send or receive session. This includes early pipelined sends and receives that
never enter the completed-context transfer manager. The claim uses the local
request ID and its execution version, not the peer's transfer ID. It survives
request cleanup and cancellation, then releases after session cleanup succeeds.
Construction or cleanup uncertainty poisons drain; request-ID reuse cannot release
a successor's claim. Binding requires physical ownership and a retirement
watchdog and must happen before any session is admitted. Other transceivers
reject this binding. These internal hooks do not enable disaggregated RETIRE.

Session admission also requires the executor's bound worker authority table.
The epoch check and resource claim share the publication lock. If publication
wins, a stale or unknown epoch is rejected before allocation or session setup;
ordinary rejection does not poison drain. If the claim wins, publication can
continue while session construction runs. That claim keeps FENCE pending until
physical cleanup, even if the request is canceled during construction. A pending
ALIGN/RESUME cannot admit an ordinary request session. The explicit ALIGN copy
path below has separate permission. Binding only a drain tracker is insufficient
for tagged sessions.

The native Python transfer worker also binds each exchange to an immutable wire
identity: authority session, tenant, scope, epoch, transfer ID, cache salt and
expected prefix identity.
Requests, KV and auxiliary results, cancellation and quiescence messages carry
that identity. A listener rejects a mismatched or stripped identity before
dispatch. Peer requests may arrive before the local session; later local admission
must match the peer's binding. Both endpoints must support the envelope.

The prefix identity records the prompt length, token and position digests, LoRA
task ID and an immutable model/cache context ID. It excludes newly sampled tokens
beyond `prompt_len`. Tagged sessions currently require single-beam text input.
The worker snapshots this identity before session construction and checks it
again before returning the session. A mismatch with the peer prevents admission;
a change during construction keeps resource ownership until physical cleanup.

The runtime must provide `TransferWorkerConfig.execution_cache_context_id`, shared
only by workers with compatible model weights, positional semantics, adapter
identities and KV numerical format. A mutable model name or worker address is not
sufficient. This is an internal configuration contract; runtime wiring is still
required. Tagged sessions without it fail before acquiring a session claim.
Existing peer checks still validate physical layouts. Untagged transfers keep
their existing behavior. Identity schema 2 carries the prefix inside the existing
envelope framing; old tagged schema 1 messages are rejected without downgrade.

These digests describe expected inputs. They do not checksum GPU memory, establish
HBM residence, prove that a copy finished, or authorize RESUME.

Matching old completion and cancellation messages remain usable after revocation
or authority shutdown. They identify the old transfer and can help it drain;
they cannot restore authority. The worker retains bindings after session cleanup
and forbids reusing a local transfer ID in the same direction. This prevents a
delayed message from reaching a different allocation. These bindings currently
last for the worker's lifetime and are not compacted. They identify exchanges
within the trusted control plane; they do not authenticate a network peer.

Ordinary transfers check current authority and completed RESUME before sender
task admission, destination publication reservation and backend submission. The
check requires a local session in the matching direction. A sender rechecks after
queued preparation, so an earlier task admission cannot authorize a stale DMA
submission. Rejected receive admission fails the unpublished task and session;
it does not leave the receiver waiting for a result that cannot arrive.

These are short host reservations under the publication lock. Session and
retirement locks are acquired first; network sends, gather work, backend calls
and completion waits run outside authority. If a reservation wins before ADVANCE,
its physical owners remain held while that operation finishes or fails. Rejection
of a later slice cannot release previously published KV or auxiliary state.
Matching completion evidence can still retire those owners after revocation.
FENCE waits for physical drain and successful session cleanup, including when
rejection happened before any destination was published.

An ALIGN copy can run before RESUME through an `AlignmentTransferPermit`. This
worker-local object names one revision, local staging request, exchange,
direction and prefix length. It grants no generation or output authority.
The runtime supplies its completed global FENCE receipt; the worker rechecks its
own FENCE and the exact currently published revised input. The copied prefix must
match those tokens, salt and LoRA ID, use the agreed positions, and leave the
final input token for computation. A caller cannot reuse a retired exchange ID.

The native transfer worker accepts `alignment` and `fence` when creating this
session. Its resource claim does not enqueue a generation request. Sender task
admission, receive publication and backend submission check the copy permit.
A later revision closes those checks immediately. The claim remains live until
physical cleanup, blocks a later FENCE, and keeps the associated ALIGN reservation
from being reclaimed. Shutdown refuses cache teardown while a copy retains it.

Copy cleanup and prefix acceptance are separate. Even after the session closes,
ALIGN and RESUME remain blocked until the engine validates the result. The engine
calls `finish_alignment_transfer(session, validate_prefix)` after successful
transfer completion. The callback must validate and attach the retained prefix
at the intended placement; it runs outside authority locks. A racing revision
rejects acceptance while still allowing safe physical cleanup. Failed validation
requires a new revision. Multiple copies must all finish and be accepted.
Contradictory evidence after acceptance closes the worker's authority.

These claims account for resource lifetime; they do not publish authority or
prove the contents of transferred memory. End-to-end disaggregated RETIRE still
needs runtime wiring of shared authority and cache context across prefill and
decode, destination staging ownership, physical page selection and the engine
callback that validates and attaches the received prefix. `PrefixTransferPlan`
routes source retention and acceptance; it does not supply those engine pieces
or turn a cache lookup count into HBM evidence. Physical retirement must
be qualified for each supported layout and transport. Device guards continue to
reject transfer and connector configurations until those paths are integrated
and qualified.

## ALIGN (step 7)

Supplying the complete revised token input with ADVANCE lets the worker reserve
useful KV before cancellation frees the old owner's cache.

```python
from tensorrt_llm.execution_alignment import AlignmentInput

revised_input = AlignmentInput(
    input_token_ids=(1, 2, 3, 4, 9, 10),
    cache_salt="app-cache",
    lora_task_id=None,
)
result = control.advance(Revision("edit-123", old, new), alignment=revised_input)
result.publication.wait(timeout=5)
aligned = result.alignment.wait(timeout=5)
print(aligned.reusable_prefix_length)
```

The runtime can also revoke first and supply the revised input later:

```python
revision = Revision("edit-123", old, new)
advance = control.advance(revision)
advance.publication.wait(timeout=5)

# The runtime may wait for a tool or obtain the revised input here.
# No successor reservation protects the cache during this gap.
result = control.align(revision, revised_input)
result.publication.wait(timeout=5)
aligned = result.alignment.wait(timeout=5)
control.resume(aligned).wait(timeout=5)
```

Separate ALIGN searches what remains in the cache at that time. Evicted KV is
recomputed unless `min_reuse_tokens` requires a hit. It uses the same V1/V2
reservation, multi-rank prefix agreement and RESUME checks as input supplied
with ADVANCE. It cannot align an unknown or superseded revision, change input
on a retry, or admit work before successor-safe ALIGN and acknowledged RESUME. Once supplied,
the revised input is immutable regardless of which call supplied it.

### Local handoff before old cleanup

Eligible single-rank V2 overlap execution can return ALIGN and acknowledge RESUME
while obsolete GPU work and terminal cleanup remain pending. It shares only
already committed, immutable full blocks, obtained through a separate suspended
native cache. It does not publish the old request's unfinished context or activate
the reservation. Normal native readiness dependencies still order successor reads.
The successor receives its own execution slot and writable suffix, with reuse
fixed to the acquired prefix length even if the cache tree subsequently grows.

The engine stops scheduling each obsolete source and retains its exact allocation,
slot, completion events and context receipts. Every admitted old request through
the replaced epoch must be isolated or finished; unknown queue entries and
independent resource accessors prevent early readiness. New accessors cannot be
added to a certified old scope. FENCE keeps its full-drain meaning and completes
only after resource cleanup succeeds.

After normal result handling retires an old request, the engine polls its latest
GPU event and context receipts at scheduling safe points. It also retains the
request while a previous overlap batch still refers to its sampler rows. Ready
owners are released independently of successor attachment. Pending owners count
against admission capacity, prevent idle blocking and shutdown, and defer cache
rebalancing and control actions. Cleanup failure preserves ownership and fails
readiness and drain checks.

This path uses the existing native-guarded asynchronous-context eligibility:
dense reusable V2, single rank, overlap enabled, no speculative decoding or joint
reuse, and no conversation state. Device-guard restrictions also exclude KV
transfers/connectors, multimodal requests and recurrent state. Unsupported modes,
LoRA sources, partial-block matches or insufficient retained prefixes use the existing drained
preparation path. The early path may reuse less KV because it cannot publish a
just-completed source frontier by waiting. `min_reuse_tokens` can require that
frontier and therefore select the drained path.

Wait for publication, ALIGN and acknowledged RESUME before submitting the
successor. Observe `result.fence.wait()` later when full old-resource cleanup is
needed; waiting for it before RESUME intentionally serializes the handoff. Retired
suffixes still occupy memory and slots, so early readiness does not guarantee
immediate scheduling or a measured speedup. CUDA and serving qualification are
required for this implementation.

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

### Bound reuse for a numerical contract

Matching tokens and intact KV bytes establish structural validity. They do not
guarantee the same output as a cold prefill: decode and prefill attention can
round differently, and later layers can amplify that difference. This matters
when a revision includes tokens generated by its predecessor.

The runtime can pass `max_reuse_tokens` in `AlignmentInput` to limit reuse to a
boundary validated for its model, backend, execution mode and output contract.
V1 rounds down to complete blocks; V2 applies its normal prefix rules within
the bound. On a single worker, the acquired reservation is the guaranteed
minimum; the runtime's maximum remains the successor's lookup limit. For
example, a 2K-token reservation with a 12K-token maximum permits a later 6K
compatible cache hit, but never reuse beyond 12K. Multi-worker RESUME still
requires the exact agreed common prefix. The isolated local handoff also fixes
the exact acquired prefix, rather than allowing a larger later cache hit.
The suffix is recomputed. Other
requests keep their own ordinary cache access.

V2 also enforces the bound when committing computed KV. Committing can adopt
an existing tree entry, so limiting the initial lookup alone is insufficient.
If a matching block is beyond the bound, V2 keeps the request's recomputed
pages and stops further publication for that request. New blocks can still be
published until such a collision. Other owners and tree entries are unchanged.
This can reduce reuse available to a later revision, but preserves the current
revision's chosen recomputation boundary.

Use `max_reuse_tokens=0` when the runtime has no acceptable reuse evidence.
This forces cold successor prefill even if the same namespace contains a full
match. `None` retains ordinary TRT-LLM reuse semantics; it is not a numerical
certificate. No fixed token count is valid for every model or workload, and
the engine does not generate a numerical attestation. A requested minimum or
transfer length above the maximum is rejected before publication. Retries
cannot change the bound.

```text
ADVANCE(new version, revised input)
  -> mailbox arms the reservation plan and revokes old work
  -> engine matches revised tokens and acquires native KV holds
  -> engine proves source isolation OR waits for full old-resource drain
  -> ALIGN returns a reservation ID and the held prefix length
  -> acknowledged RESUME opens successor admission

Old cleanup independently completes FENCE. Unsupported isolation paths wait
for FENCE before returning ALIGN.

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

When ALIGN specifies `max_reuse_tokens`, attachment must stay within that
runtime bound and cover the reserved prefix. An explicit common-prefix
agreement instead requires exactly that length on every worker. The native
request retains its applicable limit during rescheduling and V2 commit-time
rebasing, so later cache access cannot bypass it.

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
The text-only, single-beam and supported-topology restrictions still apply; this
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
| Eager Qwen3, Qwen2 or dense Llama/Mistral prefill, at least 32 scheduled context tokens | Before each group of four decoder layers |
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

Accepted decode steps extend the same frontier beyond the input prompt. After
the device ticket completes, host authority must still accept the sample before
its input KV becomes inheritable. The newly sampled token is excluded: its KV
is written by the next forward. A rejected or failed sample cannot advance the
frontier. Terminal cleanup bounds V1 publication by this accepted frontier,
including ordinary completion and cancellation without a successor.

Tagged V2 requests keep committing after prefill. When accepted decode progress
completes another block, the guard drains the latest GPU reader before publishing
it; overlap may already have submitted the next forward. The drain does not hold
host authority. V2 then uses its normal commit/rebase machinery and stops
committing during terminal cleanup. A stopped or suspended cache is never
reopened by this path. Untagged requests retain their existing prefill-end
transition. The extra block-boundary waits can reduce decode overlap and belong
in the no-revision overhead measurements.

Generated-prefix inheritance requires rebuilt V1 bindings and the V2
`is_commit_allowed` property from this branch. A Python overlay on the earlier
wheel is insufficient. Run the generated-frontier host tests, native
`LlmRequestTest.retiredKvFrontierIncludesWrittenGeneratedInputs`, native
`KVCacheManagerTest.RetiredDecodePublishesOnlyAcceptedGeneratedPrefix`, and
`test_retire_native_decode_commit_survives_owner_close` on CUDA before model
qualification. The implementation is not yet qualified by GPU/model results.

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

## Explicit Torch BLAS preference

Cached and cold computation can use different matrix shapes even with identical
prefix tokens. A model whose output contract requires exact cold equivalence
must validate those shapes on its actual software and hardware configuration.
Changing the BLAS preference is one explicit configuration choice; it does not
certify numerical equivalence or replace the reuse ceiling.

The experimental PyTorch option is:

```yaml
torch_blas_preference: cublaslt
```

The default is `None`, which leaves Torch's existing preference unchanged. An
explicit preference is assigned before model construction and checked after
configuration resolution, at warmup and on the execution thread. It preserves
the BF16 reduction and split-K settings, which must both be enabled. Unsupported
Torch readbacks or a subsequent change to protected settings fail initialization
or execution rather than silently using a different policy.

The option currently requires a fresh local TP1 decoder worker without Torch
compilation, speculative decoding, encoder-only execution or a KV connector.
It cannot attach to an existing worker. Since the preference is process-wide,
explicit and unset policies cannot share a worker process; a failed explicit
initialization requires a fresh process. A preference does not guarantee the
library or kernel selected for every operation. Keep the same declared policy
in native and RETIRE comparisons, and separately validate the complete model,
revision and output contract before interpreting performance measurements.

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
GPU-scope acquire loads, matching the device publisher. Untagged live rows
use the exact sentinel `(slot=-1, expected_epoch=0)`. Multi-block reductions
still complete their scratch writes and semaphore arrivals; they cannot take
independent early exits without stranding other blocks.

These predicates do not establish FENCE or permit resource release. Callers
must retain the authority tensors and KV until all submitted GPU uses finish.
Captured graphs require stable addresses, with row identities refreshed before
each replay. Existing cubins without the authority ABI cannot serve a guarded
call. Rebuild native libraries and bindings together.

When reusing a build directory, pass `--generate_fmha` to
`scripts/build_wheel.py`. The FMHA generation stamp records completion, not
the generator's source revision. Without regeneration, a new wheel can retain
an older dispatch table that selects unguarded head-size-128 cubins. Keep the
operator gate enabled: a wheel's source commit alone does not verify its
generated kernels.

Enable executor binding before constructing the LLM:

```bash
export TRTLLM_RETIRE_NATIVE_GUARDS=1
export TLLM_FMHA_LIBS=fallback
```

The decoder creates its authority table before warmup. When the runtime later
registers a scope, the controller adopts that same table. The runner stages
identities in the attention packer's order after input preparation, before
eager/compiled execution or graph replay. Each metadata object owns separate,
stable buffers with capacity for all sequence rows; unused entries are reset
to the untagged sentinel. This also covers prefill graph buckets whose sequence
count changes. Immutable host uploads stay alive until their copies complete,
and the buffers can only be updated on their original model submission stream.

With native predicates enabled, tagged prefill chunks wait for their recorded
GPU completion before host progress or a reusable prefix is accepted. Publication
remains unlocked during that wait, and authority is checked again afterward.
This prevents a revision from suppressing queued writes to an already accepted
prefix. It can reduce CPU/GPU overlap during tagged prefill; include that cost in
performance comparisons. Untagged batches, decode-only batches and the path
without native predicates keep their existing scheduling behavior.

This opt-in requires the matching native build. It currently targets the
SM90 BF16 decoder path above, with V1 or V2 KV and single-rank or coordinated TP
execution. TP model qualification remains pending. Enabling it does
not remove the executor's unsupported-feature checks. Native operator tests,
binding tests and real-model eager/overlap/compiled/graph qualification must
pass before claiming end-to-end operator suppression or performance gains.

`tests/unittest/_torch/attention/test_execution_native_guards.py` checks V1/V2,
fresh/cached context, short/long generation, mixed batches, live tagged and
untagged neighbors, epoch wrap across the signed boundary, malformed metadata,
and captured decode replay after identity changes. It compares live outputs
exactly with the unguarded operator and verifies that revoked rows leave output
and KV sentinels untouched. The existing H100 attention test-list entry includes
this file. Native build and GPU results are required to qualify the port.
`test_execution_attention_binding.py` additionally checks packed row order,
repeated graph-padding requests, buffer ownership, graph replay after epoch and
identity changes, batch shrinkage, stream constraints, and metadata lifetimes.

### Executor lifecycle

The Qwen3, Qwen2 and dense Llama decoder loops check authority every four
eager prefill layers. Mistral uses that same Llama loop. An exit stops only
unsubmitted layers; the executor still drains recorded device uses before
reclaiming the request. Compilation and graph capture omit these Python layer
checks. Graph replay uses the separate prelaunch and breakable-segment hooks.
These hooks do not enable unsupported model features or parallelism. Model
fixtures and production checkpoints need their own GPU qualification.

For mechanism ablations, set `TRTLLM_RETIRE_EARLY_EXIT=0` before starting the
worker to disable host prelaunch, layer and graph-segment exits. The default is
`1`. This leaves authority publication, stale-output rejection, FENCE and KV
ownership checks active. Native attention predicates have their own startup
switch, `TRTLLM_RETIRE_NATIVE_GUARDS`; disable both to measure an arm without
either form of early stopping. Keep the same attention provider in every arm.

Cross early stopping off/on with explicit inheritance off/on. With inheritance
off, call `advance(revision)` without an alignment input, wait for publication
and FENCE, then submit the successor under the new version. With inheritance
on, supply the revised input and wait for ALIGN and RESUME as usual. Keep
ordinary prefix reuse and cache salts equivalent in all four arms. This tests
the benefit of an explicit handoff, not the benefit of disabling native reuse.
Run native cancellation as a separate baseline. Preserve correctness failures
and report both total revision latency and recovery after any imposed wait.

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

## Observing reservation ownership

`executor.collective_rpc("fetch_execution_reservations")` samples worker-owned
ALIGN holds without starting RETIRE, querying CUDA, or releasing anything.
The report distinguishes a promised prefix, a still-held reservation, and a
reservation whose native cleanup has completed. Superseding an epoch can make
a reservation obsolete before the engine has released it; a publication or
FENCE receipt alone does not establish that this extra hold is gone.

`held_count` counts live reservation objects. `promised_tokens_held` sums their
known promises; overlapping reservations can refer to the same physical pages.
`unknown_prefix_count` includes retained partial reservations without a completed
receipt. An unsafe cleanup remains visible with its ownership flag set.
These are logical ownership counters, not HBM bytes or physical occupancy.
Use the separate V2 page-residency query for page locations. Both reports are
observations and cannot authorize reuse.

## Next steps

Validate the complete lifecycle on native GPU runs across V1/V2 and execution
modes. Distributed authority, disaggregated transfers, scope-history compaction,
and recovery after restart remain separate work. Authority checks stay active
after RESUME so another revision can supersede the new work too.
