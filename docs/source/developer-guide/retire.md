<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# RETIRE: request version identity

RETIRE lets a runtime replace work when its context changes. The runtime decides
what to replace and supplies the revised input. TRT-LLM will use that decision to
stop obsolete work and safely hand useful KV to the replacement.

The first step adds **metadata only** to the Python LLM and executor APIs.
Supplying a version does not cancel requests, select an active version, change
scheduling, protect KV, or affect cache matching.

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
Generation belongs to the resource owner and is independent of epoch. This step
defines the type; it does not attach state tags to KV allocations yet. A state
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

## Next step

Add a scope authority that accepts a revision only against the current version.
Mailbox publication, stale-work checks, fencing, KV alignment, resume, and
reclamation follow in separate review steps.
