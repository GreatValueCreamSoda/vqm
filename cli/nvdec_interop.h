// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#pragma once

#include <array>
#include <memory>

#include <cuda.h>

#include "vqm_vulkan.h"

namespace vqm::cli {

class CurrentCudaContext {
public:
    explicit CurrentCudaContext(CUcontext context);
    ~CurrentCudaContext();
    CurrentCudaContext(const CurrentCudaContext&) = delete;
    CurrentCudaContext& operator=(const CurrentCudaContext&) = delete;
};

// The pointers and byte strides describe a CUDA hardware frame. For
// semiplanar frames, the second pointer is the interleaved U/V plane.
struct NvdecSource {
    std::array<CUdeviceptr, 3> data{};
    std::array<unsigned, 3> stride{};
    CUstream stream = nullptr;
    bool semiplanar = false;
    unsigned shift = 0;
};

// A reusable frame buffer. The interop device must outlive the buffer.
class NvdecBuffer {
public:
    NvdecBuffer();
    ~NvdecBuffer();
    NvdecBuffer(const NvdecBuffer&) = delete;
    NvdecBuffer& operator=(const NvdecBuffer&) = delete;

private:
    friend class NvdecInterop;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NvdecInterop {
public:
    NvdecInterop();
    ~NvdecInterop();
    NvdecInterop(const NvdecInterop&) = delete;
    NvdecInterop& operator=(const NvdecInterop&) = delete;

    [[nodiscard]] const VulkanDevice& vulkan() const noexcept;
    [[nodiscard]] CUcontext context() const noexcept;
    [[nodiscard]] int cuda_ordinal() const noexcept;

    // Repack on the GPU and return a Vulkan-owned planar buffer. Keep the
    // buffer alive and finish scoring it before repacking or destroying it.
    // The decoded CUDA frame may be released as soon as Pack returns.
    [[nodiscard]] VulkanBufferView Pack(
        NvdecBuffer& dst, Size size, const ImageFormat& format,
        const NvdecSource& src
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vqm::cli
