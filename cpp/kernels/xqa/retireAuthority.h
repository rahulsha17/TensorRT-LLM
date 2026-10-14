/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifndef GENERATE_CUBIN
#include <cstdint>
#endif

// This POD is shared verbatim by the TRT launcher and generated FMHA kernels.
// Append it to parameter structs: never change the prebuilt-cubin ABI prefix.
namespace retire
{
#ifdef GENERATE_CUBIN
using uint64_t = unsigned long long;
using int32_t = int;
#endif
static_assert(sizeof(uint64_t) == 8 && sizeof(int32_t) == 4);

struct Authority
{
    uint64_t const* currentVersions = nullptr;
    int32_t const* sequenceSlots = nullptr;
    uint64_t const* expectedVersions = nullptr;
    unsigned long long* counters = nullptr;
    int32_t capacity = 0;
    int32_t rows = 0;
    int32_t rowOffset = 0;

#ifdef __CUDACC__
    // A predicate is not a drain certificate. Physical ownership remains held
    // until the executor records and observes completion of all model work.
    __device__ __forceinline__ bool allows(int32_t localRow, int32_t counter) const
    {
        if (currentVersions == nullptr)
        {
            return true;
        }
        int32_t const row = localRow + rowOffset;
        bool allowed = false;
        if (row >= 0 && row < rows)
        {
            int32_t const slot = sequenceSlots[row];
            uint64_t const expected = expectedVersions[row];
            if (slot == -1 && expected == 0)
            {
                allowed = true; // exact sentinel for an untagged live neighbor
            }
            else if (slot >= 0 && slot < capacity)
            {
                uint64_t current;
                // Pair with the GPU-scope release publication without making
                // every attention thread perform a read-modify-write.
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
                             : "=l"(current)
                             : "l"(currentVersions + slot)
                             : "memory");
                allowed = current == expected;
            }
        }
        if (!allowed && counters != nullptr)
        {
            atomicAdd(counters + counter, 1ULL);
        }
        return allowed;
    }
#endif
};
} // namespace retire
