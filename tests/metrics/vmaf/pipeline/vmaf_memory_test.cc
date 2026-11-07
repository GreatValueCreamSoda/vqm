// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF: Copyright (c) Netflix, Inc. BSD-2-Clause-Patent;
// see licenses/NETFLIX-VMAF-LICENSE.

#include <cstdint>
#include <limits>

#include "support.h"
#include "vmaf.h"
#include "metrics/vmaf/vmaf_memory.h"
#include "vmaf_v1.h"
#include "vulkan/vulkan_support.h"

namespace {
using namespace vqm;
using namespace vqm::internal;

constexpr Size kSize{512, 384};
constexpr std::uint64_t kPixels = kSize.width * kSize.height;

VmafV1Config V1Config() {
    VmafV1Config config;
    config.size = kSize;
    config.input_format.type = PixelType::Byte;
    config.input_format.family = ColorFamily::Yuv;
    config.input_format.matrix = MatrixCoefficients::Bt709;
    config.input_format.depth = 8;
    config.input_format.subsample_w = 1;
    config.input_format.subsample_h = 1;
    config.gpu_resident_images = 0;
    config.streaming_inputs = false;
    return config;
}

void SetLimits(VulkanContext& context) {
    context.limits.maxStorageBufferRange =
        std::numeric_limits<std::uint32_t>::max();
    context.limits.maxComputeSharedMemorySize = 49152;
    context.limits.maxComputeWorkGroupInvocations = 1024;
    context.limits.maxComputeWorkGroupSize[0] = 1024;
    context.limits.maxComputeWorkGroupSize[1] = 1024;
    context.limits.maxComputeWorkGroupSize[2] = 64;
    context.limits.maxComputeWorkGroupCount[0] = 65535;
    context.limits.maxComputeWorkGroupCount[1] = 65535;
    context.limits.maxComputeWorkGroupCount[2] = 65535;
    context.limits.maxPushConstantsSize = 128;
}
}  // namespace

VQM_TEST(TestVmafMemoryPlansAccountForSourcesAndStaging) {
    VulkanContext context;
    SetLimits(context);
    VmafConfig v0{
        .size = kSize, .gpu_resident_images = 0, .streaming_inputs = false};
    VmafV1Config v1 = V1Config();
    const auto v0_base = PlanVmafMemory(v0, context);
    const auto v1_base = PlanVmafV1Memory(v1, context);
    CHECK(v0_base.fallback_staging_bytes == (2 * 44 + 14) * 4);
    CHECK(v1_base.fallback_staging_bytes == (2 * 44 + 17) * 4);

    v0.gpu_resident_images = 1;
    v1.gpu_resident_images = 1;
    const auto v0_resident = PlanVmafMemory(v0, context);
    const auto v1_resident = PlanVmafV1Memory(v1, context);
    CHECK(v0_resident.arena_bytes == v0_base.arena_bytes + 16 * kPixels);
    CHECK(v1_resident.arena_bytes == v1_base.arena_bytes + 16 * kPixels);
    CHECK(
        v0_resident.fallback_staging_bytes ==
        v0_base.fallback_staging_bytes + 16 * kPixels
    );
    CHECK(
        v1_resident.fallback_staging_bytes ==
        v1_base.fallback_staging_bytes + 16 * kPixels
    );

    v0.streaming_inputs = true;
    v1.streaming_inputs = true;
    const auto v0_streaming = PlanVmafMemory(v0, context);
    const auto v1_streaming = PlanVmafV1Memory(v1, context);
    CHECK(v0_streaming.arena_bytes == v0_resident.arena_bytes + 32 * kPixels);
    CHECK(v1_streaming.arena_bytes == v1_resident.arena_bytes + 32 * kPixels);
    CHECK(
        v0_streaming.fallback_staging_bytes ==
        v0_base.fallback_staging_bytes + 32 * kPixels
    );
    CHECK(
        v1_streaming.fallback_staging_bytes ==
        v1_base.fallback_staging_bytes + 32 * kPixels
    );
}

VQM_TEST(TestVmafMemoryPlanUsesDeviceLimitsAndValidatesConfig) {
    VulkanContext context;
    SetLimits(context);
    VmafConfig v0{
        .size = kSize, .gpu_resident_images = 0, .streaming_inputs = false};
    auto v1 = V1Config();
    const auto fused = PlanVmafMemory(v0, context);
    context.limits.maxComputeSharedMemorySize = 16384;
    const auto fallback = PlanVmafMemory(v0, context);
    CHECK(fallback.arena_bytes == fused.arena_bytes + 12 * kPixels);

    context.limits.maxStorageBufferRange = 1;
    CHECK_THROWS_AS(PlanVmafMemory(v0, context), std::invalid_argument);
    CHECK_THROWS_AS(PlanVmafV1Memory(v1, context), std::invalid_argument);
    SetLimits(context);
    v0.size = {std::numeric_limits<std::size_t>::max(), 20};
    CHECK_THROWS_AS(PlanVmafMemory(v0, context), std::invalid_argument);
    v1.input_format.family = ColorFamily::Rgb;
    v1.input_format.matrix = MatrixCoefficients::Rgb;
    v1.input_format.subsample_w = 0;
    v1.input_format.subsample_h = 0;
    CHECK(PlanVmafV1Memory(v1, context).arena_bytes > 0);
    v1 = V1Config();
    v1.size.width = 511;
    CHECK_THROWS_AS(PlanVmafV1Memory(v1, context), std::invalid_argument);
}

VQM_TEST(TestVmafV1MemoryFitsWithoutUnusedDistortedDetailBands) {
    VulkanContext context;
    SetLimits(context);
    // The 512x384 native pipeline fits in 14 MB. Retaining the obsolete
    // distorted-detail plane adds 589,824 bytes and exceeds this buffer budget.
    context.limits.maxStorageBufferRange = 14'000'000;
    const auto plan = PlanVmafV1Memory(V1Config(), context);
    CHECK(plan.arena_bytes <= context.limits.maxStorageBufferRange);
}

VQM_TEST(TestVmafV0DispatchLimitBoundaries) {
    VulkanContext context;
    SetLimits(context);
    VmafConfig config{
        .size = kSize, .gpu_resident_images = 0, .streaming_inputs = false};
    auto& limits = context.limits;
    // The source/luma grid needs 32 groups in X; 32x8 motion needs 48 in Y.
    limits.maxComputeWorkGroupCount[0] = 32;
    limits.maxComputeWorkGroupCount[1] = 48;
    limits.maxComputeWorkGroupCount[2] = 1;
    limits.maxComputeWorkGroupSize[0] = 256;
    limits.maxComputeWorkGroupSize[1] = 16;
    limits.maxComputeWorkGroupSize[2] = 1;
    limits.maxComputeWorkGroupInvocations = 256;
    limits.maxComputeSharedMemorySize = 6 * 256 * sizeof(float);
    limits.maxPushConstantsSize = 14 * sizeof(std::uint32_t);
    CHECK(PlanVmafMemory(config, context).arena_bytes > 0);

    --limits.maxComputeWorkGroupCount[0];
    CHECK_THROWS_AS(PlanVmafMemory(config, context), std::invalid_argument);
    ++limits.maxComputeWorkGroupCount[0];
    --limits.maxComputeWorkGroupCount[1];
    CHECK_THROWS_AS(PlanVmafMemory(config, context), std::invalid_argument);
    ++limits.maxComputeWorkGroupCount[1];
    --limits.maxComputeWorkGroupCount[2];
    CHECK_THROWS_AS(PlanVmafMemory(config, context), std::invalid_argument);
    ++limits.maxComputeWorkGroupCount[2];
    --limits.maxComputeWorkGroupSize[0];
    CHECK_THROWS_AS(PlanVmafMemory(config, context), std::invalid_argument);
    ++limits.maxComputeWorkGroupSize[0];
    --limits.maxComputeWorkGroupSize[1];
    CHECK_THROWS_AS(PlanVmafMemory(config, context), std::invalid_argument);
    ++limits.maxComputeWorkGroupSize[1];
    --limits.maxComputeWorkGroupInvocations;
    CHECK_THROWS_AS(PlanVmafMemory(config, context), std::invalid_argument);
    ++limits.maxComputeWorkGroupInvocations;
    --limits.maxComputeSharedMemorySize;
    CHECK_THROWS_AS(PlanVmafMemory(config, context), std::invalid_argument);
    ++limits.maxComputeSharedMemorySize;
    --limits.maxPushConstantsSize;
    CHECK_THROWS_AS(PlanVmafMemory(config, context), std::invalid_argument);
}

VQM_TEST(TestVmafV1DispatchLimitBoundaries) {
    VulkanContext context;
    SetLimits(context);
    auto config = V1Config();
    auto& limits = context.limits;
    // The 512x384 case uses 2500 covariance groups in x, 48 CAMBI
    // contrast groups in y, and four batched SpEED filter planes in z.
    limits.maxComputeWorkGroupCount[0] = 2500;
    limits.maxComputeWorkGroupCount[1] = 48;
    limits.maxComputeWorkGroupCount[2] = 4;
    limits.maxComputeWorkGroupSize[0] = 256;
    limits.maxComputeWorkGroupSize[1] = 16;
    limits.maxComputeWorkGroupSize[2] = 1;
    limits.maxComputeWorkGroupInvocations = 256;
    limits.maxComputeSharedMemorySize = 16004;
    limits.maxPushConstantsSize = 19 * sizeof(std::uint32_t);
    CHECK(PlanVmafV1Memory(config, context).arena_bytes > 0);

    --limits.maxComputeWorkGroupCount[0];
    CHECK_THROWS_AS(PlanVmafV1Memory(config, context), std::invalid_argument);
    ++limits.maxComputeWorkGroupCount[0];
    --limits.maxComputeWorkGroupCount[1];
    CHECK_THROWS_AS(PlanVmafV1Memory(config, context), std::invalid_argument);
    ++limits.maxComputeWorkGroupCount[1];
    --limits.maxComputeWorkGroupCount[2];
    CHECK_THROWS_AS(PlanVmafV1Memory(config, context), std::invalid_argument);
    ++limits.maxComputeWorkGroupCount[2];
    --limits.maxComputeSharedMemorySize;
    CHECK_THROWS_AS(PlanVmafV1Memory(config, context), std::invalid_argument);
    ++limits.maxComputeSharedMemorySize;
    --limits.maxPushConstantsSize;
    CHECK_THROWS_AS(PlanVmafV1Memory(config, context), std::invalid_argument);
    ++limits.maxPushConstantsSize;

    // When packing all scales would exceed x, CAMBI uses its rectangular
    // fallback. Only the largest single scale's groups must fit.
    config.size = {1536, 1024};
    limits.maxComputeWorkGroupCount[0] = 6144;
    limits.maxComputeWorkGroupCount[1] = 128;
    CHECK(PlanVmafV1Memory(config, context).arena_bytes > 0);
    --limits.maxComputeWorkGroupCount[0];
    CHECK_THROWS_AS(PlanVmafV1Memory(config, context), std::invalid_argument);
}
