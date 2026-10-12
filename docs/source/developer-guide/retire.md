<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# RETIRE: execution versions and authority

RETIRE lets a runtime replace work when its context changes. The runtime decides
what to replace and supplies the revised input. TRT-LLM will use that decision to
stop obsolete work and safely hand useful KV to the replacement.

The Python LLM and executor APIs carry version metadata. A separate host-side
`ScopeRegistry` records which version is current and accepts revisions from the
runtime. An executor controller now publishes accepted versions to a worker's
host table. Supplying a version or publishing an update does not yet stop work,
block output, or protect KV.

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

version = ExecutionVersion(tenant_id="app", scope_id="task-42", epoch=0)
output = llm.generate(
    "Explain this function", execution_version=version, cache_salt="app-cache"
)
assert output.execution_version == version
assert output.cache_salt == "app-cache"
```

`generate_async` accepts the same argument. Batched `generate` accepts either one
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
controller. Existing generation calls continue to carry metadata only.

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

## Testing

The CPU CI list includes authority, mailbox, IPC, and executor-wiring tests.
They cover concurrent revisions, lost acknowledgements, delayed updates,
publication to a spawned process while its main thread is occupied, real RPC
bootstrap, topology guards, bounded pending work, and shutdown. These tests do
not run model inference or establish GPU execution safety.

## Next steps

The next step will check authority where work and output are admitted. Later
steps establish GPU publication and FENCE, align the
successor's KV, resume it, and reclaim old resources. RESUME will keep authority
checks active so another revision can supersede the new work too.
