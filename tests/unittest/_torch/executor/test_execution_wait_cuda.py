# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA completion waiting, concurrent publication and sticky failure regressions."""

import faulthandler
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import cuda.bindings.runtime as cudart
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.execution_device import DeviceExecutionAuthority, _PendingUse
from tensorrt_llm.bindings.internal import runtime as runtime_bindings
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _HostStreamGate:
    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()
        self.timed_out = Event()

    def __call__(self) -> None:
        # CUDA calls are forbidden inside a CUDA host function.
        self.entered.set()
        if not self.release.wait(30):
            self.timed_out.set()


@pytest.mark.parametrize("revoke_owner", [False, True])
def test_pending_cuda_wait_allows_publication_and_fresh_token_verdict(revoke_owner: bool) -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "scope", (1 << 63) - 1)
    neighbor = replace(old, scope_id="neighbor")
    new = replace(old if revoke_owner else neighbor, epoch=1 << 63)
    authority.publish(old)
    authority.publish(neighbor)
    candidates = torch.tensor([[[41], [29]]], device=authority.device, dtype=torch.int32)
    assert authority.enqueue_token_commit([old, None], [0, 1], candidates, True).wait().allowed == [
        True,
        True,
    ]
    assert authority.check([old, neighbor]) == [True, True]
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    completion = torch.cuda.Event()
    gate = _HostStreamGate()
    first_miss, finished = Event(), Event()
    failures: list[BaseException] = []

    class ObservedEvent:
        def query(self) -> bool:
            ready = completion.query()
            if not ready:
                first_miss.set()
            return ready

    def wait() -> None:
        try:
            authority.wait_event(ObservedEvent())
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()

    waiter = Thread(target=wait, daemon=True)
    faulthandler.dump_traceback_later(120, exit=True)
    try:
        runtime_bindings.launch_hostfunc(stream.cuda_stream, True, False, gate)
        completion.record(stream)
        assert gate.entered.wait(30), "CUDA host gate did not start"
        assert not gate.timed_out.is_set() and not completion.query()
        waiter.start()
        assert first_miss.wait(2), "wait did not observe the real pending event"
        authority.publish(new)
        assert not finished.is_set(), "publication did not run during the pending wait"
        assert not completion.query(), "publication unexpectedly joined the occupied stream"
        assert not gate.release.is_set() and not gate.timed_out.is_set()
        gate.release.set()
        assert finished.wait(5)
        waiter.join(1)
        assert not waiter.is_alive() and failures == [] and completion.query()
        assert not gate.timed_out.is_set()
        old_result = authority.enqueue_token_commit([old, None], [0, 1], candidates, True).wait()
        assert old_result.allowed == [not revoke_owner, True]
        assert old_result.tokens_host.tolist() == ([0, 29] if revoke_owner else [41, 29])
        current = new if revoke_owner else old
        live = authority.enqueue_token_commit([current, None], [0, 1], candidates, True).wait()
        assert live.allowed == [True, True] and live.tokens_host.tolist() == [41, 29]
        assert torch.equal(live.tokens_device, candidates)
        assert candidates.flatten().tolist() == [41, 29]
        authority._healthy()
    finally:
        gate.release.set()
        try:
            if waiter.ident is not None:
                waiter.join(6)
            assert not waiter.is_alive()
        finally:
            try:
                stream.synchronize()
            finally:
                faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("failure_kind", ["timeout", "query"])
def test_wait_failure_retains_cuda_buffers_and_poison(failure_kind: str) -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    buffer = torch.ones(16, device=authority.device, dtype=torch.uint8)
    stream = torch.cuda.Stream()
    completion = torch.cuda.Event()
    gate = _HostStreamGate()
    stream.wait_stream(torch.cuda.current_stream())
    failure = RuntimeError("controlled query failure")

    def fail_query() -> bool:
        raise failure

    event = completion if failure_kind == "timeout" else SimpleNamespace(query=fail_query)
    use = _PendingUse([buffer], event)
    authority._pending.append(use)
    faulthandler.dump_traceback_later(120, exit=True)
    try:
        runtime_bindings.launch_hostfunc(stream.cuda_stream, True, False, gate)
        (result,) = cudart.cudaMemsetAsync(
            buffer.data_ptr(), 37, buffer.numel(), stream.cuda_stream
        )
        assert int(result) == 0
        completion.record(stream)
        assert gate.entered.wait(30), "CUDA host gate did not start"
        assert not gate.timed_out.is_set() and not completion.query()
        expected = TimeoutError if failure_kind == "timeout" else RuntimeError
        with pytest.raises(expected) as error:
            authority._wait(use, timeout=0)
        if failure_kind == "query":
            assert error.value is failure
        assert authority._failure is error.value
        assert use in authority._pending and use.tensors[0] is buffer
        with pytest.raises(RuntimeError, match="restart required") as later:
            authority.check([None])
        assert later.value.__cause__ is error.value
        assert not gate.release.is_set() and not gate.timed_out.is_set()
        assert not completion.query(), "failed wait released its pending CUDA operation"
        assert use in authority._pending and use.tensors[0] is buffer
        gate.release.set()
        stream.synchronize()
        assert not gate.timed_out.is_set() and completion.query()
        assert torch.all(buffer == 37).item()
    finally:
        gate.release.set()
        try:
            stream.synchronize()
        finally:
            faulthandler.cancel_dump_traceback_later()
