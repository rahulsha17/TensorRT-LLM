/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>
#include <vector>

#if CUDART_VERSION < 12030
#error RETIRE conditional graphs require CUDA 12.3 or newer
#endif

namespace
{
void checkCuda(cudaError_t result)
{
    TORCH_CHECK(result == cudaSuccess, cudaGetErrorString(result));
}

__global__ void checkAuthority(cudaGraphConditionalHandle handle, int64_t* table, int const* slots,
    int64_t const* expected, bool const* realRows, int rows, int capacity, int* status, int index, int const* count,
    int lower, int upper)
{
    __shared__ int live;
    if (threadIdx.x == 0)
    {
        live = 0;
    }
    __syncthreads();
    for (int row = threadIdx.x; row < rows; row += blockDim.x)
    {
        if (!realRows[row])
        {
            continue;
        }
        int const slot = slots[row];
        // Unknown/invalid slots fail open for work, never for output authority.
        bool allowed = slot < 0 || slot >= capacity;
        if (!allowed)
        {
            cuda::atomic_ref<int64_t, cuda::thread_scope_device> current(table[slot]);
            allowed = current.load(cuda::memory_order_acquire) == expected[row];
        }
        if (allowed)
        {
            atomicExch(&live, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int const selected = live && (lower < 0 || (*count > lower && *count <= upper));
        status[index] = selected;
        cudaGraphSetConditional(handle, selected);
    }
}

void validateTensor(at::Tensor const& tensor, at::ScalarType dtype, c10::Device device)
{
    TORCH_CHECK(
        tensor.device() == device && tensor.scalar_type() == dtype && tensor.dim() == 1 && tensor.is_contiguous(),
        "RETIRE graph metadata requires contiguous one-dimensional CUDA tensors on one device");
}

class ConditionalGraph
{
public:
    ConditionalGraph(std::vector<int64_t> const& segments, at::Tensor table, at::Tensor slots, at::Tensor expected,
        at::Tensor realRows, std::vector<at::Tensor> counts, std::vector<int64_t> const& lowerBounds,
        std::vector<int64_t> const& upperBounds)
        : mTable(std::move(table))
        , mSlots(std::move(slots))
        , mExpected(std::move(expected))
        , mRealRows(std::move(realRows))
        , mCounts(std::move(counts))
    {
        TORCH_CHECK(mTable.is_cuda(), "RETIRE graph authority must reside on CUDA");
        c10::cuda::CUDAGuard guard(mTable.device());
        validateTensor(mTable, at::kLong, mTable.device());
        validateTensor(mSlots, at::kInt, mTable.device());
        validateTensor(mExpected, at::kLong, mTable.device());
        validateTensor(mRealRows, at::kBool, mTable.device());
        TORCH_CHECK(!segments.empty() && mSlots.numel() == mExpected.numel() && mSlots.numel() == mRealRows.numel()
                && mSlots.numel() <= INT_MAX && mTable.numel() > 0 && mTable.numel() <= INT_MAX,
            "Invalid RETIRE graph row layout");
        TORCH_CHECK(mCounts.size() == segments.size() && lowerBounds.size() == segments.size()
                && upperBounds.size() == segments.size(),
            "RETIRE graph predicates must describe every segment");
        for (size_t index = 0; index < mCounts.size(); ++index)
        {
            validateTensor(mCounts[index], at::kInt, mTable.device());
            TORCH_CHECK(mCounts[index].numel() >= 1 && lowerBounds[index] >= -1 && upperBounds[index] <= INT_MAX
                    && (lowerBounds[index] == -1 || upperBounds[index] > lowerBounds[index]),
                "Invalid RETIRE graph bucket bounds");
        }
        mStatus = at::zeros({static_cast<int64_t>(segments.size()) + 1}, mSlots.options());
        try
        {
            checkCuda(cudaGraphCreate(&mGraph, 0));
            cudaGraphNode_t previous = nullptr;
            for (size_t index = 0; index < segments.size(); ++index)
            {
                cudaGraphConditionalHandle handle;
                checkCuda(cudaGraphConditionalHandleCreate(&handle, mGraph, 0, 0));
                int64_t* tablePtr = mTable.data_ptr<int64_t>();
                int const* slotsPtr = mSlots.data_ptr<int>();
                int64_t const* expectedPtr = mExpected.data_ptr<int64_t>();
                bool const* realRowsPtr = mRealRows.data_ptr<bool>();
                int rows = static_cast<int>(mSlots.numel());
                int capacity = static_cast<int>(mTable.numel());
                int* statusPtr = mStatus.data_ptr<int>();
                int segmentIndex = static_cast<int>(index);
                int const* countPtr = mCounts[index].data_ptr<int>();
                int lower = static_cast<int>(lowerBounds[index]);
                int upper = static_cast<int>(upperBounds[index]);
                void* args[] = {&handle, &tablePtr, &slotsPtr, &expectedPtr, &realRowsPtr, &rows, &capacity, &statusPtr,
                    &segmentIndex, &countPtr, &lower, &upper};
                cudaKernelNodeParams kernel{};
                kernel.func = reinterpret_cast<void*>(checkAuthority);
                kernel.gridDim = dim3(1);
                kernel.blockDim = dim3(128);
                kernel.kernelParams = args;
                cudaGraphNode_t predicate;
                checkCuda(cudaGraphAddKernelNode(
                    &predicate, mGraph, previous ? &previous : nullptr, previous ? 1 : 0, &kernel));
                cudaGraphNodeParams condition{};
                condition.type = cudaGraphNodeTypeConditional;
                condition.conditional.handle = handle;
                condition.conditional.type = cudaGraphCondTypeIf;
                condition.conditional.size = 1;
#if CUDART_VERSION >= 13000
                checkCuda(cudaGraphAddNode(&previous, mGraph, &predicate, nullptr, 1, &condition));
#else
                checkCuda(cudaGraphAddNode(&previous, mGraph, &predicate, 1, &condition));
#endif
                cudaGraphNode_t child;
                checkCuda(cudaGraphAddChildGraphNode(&child, condition.conditional.phGraph_out[0], nullptr, 0,
                    reinterpret_cast<cudaGraph_t>(segments[index])));
            }
            // This node always runs, even when every model segment is skipped.
            cudaMemsetParams completed{};
            completed.dst = mStatus.data_ptr<int>() + segments.size();
            completed.value = 1;
            completed.elementSize = sizeof(int);
            completed.width = 1;
            completed.height = 1;
            cudaGraphNode_t terminal;
            checkCuda(cudaGraphAddMemsetNode(&terminal, mGraph, &previous, 1, &completed));
            checkCuda(cudaGraphInstantiate(&mExec, mGraph, 0));
            checkCuda(cudaEventCreateWithFlags(&mDone, cudaEventDisableTiming));
        }
        catch (...)
        {
            destroy();
            throw;
        }
    }

    ~ConditionalGraph()
    {
        c10::cuda::CUDAGuard guard(mTable.device());
        if (mLaunched)
        {
            cudaStreamSynchronize(mStream);
        }
        destroy();
    }

    void replay()
    {
        c10::cuda::CUDAGuard guard(mTable.device());
        TORCH_CHECK(mExec, "Cannot replay a closed RETIRE graph");
        auto stream = at::cuda::getCurrentCUDAStream(mTable.get_device()).stream();
        TORCH_CHECK(!mLaunched || stream == mStream, "RETIRE graphs cannot change submission streams");
        mStream = stream;
        checkCuda(cudaMemsetAsync(mStatus.data_ptr(), 0, mStatus.nbytes(), stream));
        checkCuda(cudaGraphLaunch(mExec, stream));
        mLaunched = true;
        checkCuda(cudaEventRecord(mDone, stream));
    }

    void close()
    {
        c10::cuda::CUDAGuard guard(mTable.device());
        if (mLaunched)
        {
            // Teardown also covers a launch whose subsequent event record failed.
            checkCuda(cudaStreamSynchronize(mStream));
            mLaunched = false;
        }
        destroy();
    }

    at::Tensor status() const
    {
        return mStatus;
    }

private:
    void destroy() noexcept
    {
        if (mExec)
        {
            cudaGraphExecDestroy(mExec);
            mExec = nullptr;
        }
        if (mGraph)
        {
            cudaGraphDestroy(mGraph);
            mGraph = nullptr;
        }
        if (mDone)
        {
            cudaEventDestroy(mDone);
            mDone = nullptr;
        }
    }

    at::Tensor mTable, mSlots, mExpected, mRealRows, mStatus;
    std::vector<at::Tensor> mCounts;
    cudaGraph_t mGraph{};
    cudaGraphExec_t mExec{};
    cudaEvent_t mDone{};
    cudaStream_t mStream{};
    bool mLaunched{false};
};
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<ConditionalGraph>(m, "ConditionalGraph")
        .def(pybind11::init<std::vector<int64_t> const&, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
            std::vector<at::Tensor>, std::vector<int64_t> const&, std::vector<int64_t> const&>())
        .def("replay", &ConditionalGraph::replay)
        .def("close", &ConditionalGraph::close)
        .def("status", &ConditionalGraph::status);
}
