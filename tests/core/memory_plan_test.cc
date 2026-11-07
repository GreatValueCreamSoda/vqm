// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "support.h"
#include "metrics/ssimulacra2/layout.h"
#include "metrics/butteraugli/layout.h"

namespace {
void Configure(vqm::internal::VulkanContext& context) {
    context.limits.maxComputeSharedMemorySize = 32768;
    context.limits.maxComputeWorkGroupInvocations = 1024;
    context.limits.maxComputeWorkGroupSize[0] = 1024;
    context.limits.maxComputeWorkGroupSize[1] = 1024;
    context.limits.maxComputeWorkGroupCount[0] = 65535;
    context.limits.maxComputeWorkGroupCount[1] = 65535;
    context.limits.maxComputeWorkGroupCount[2] = 65535;
    context.limits.maxStorageBufferRange = UINT32_MAX;
    context.limits.minStorageBufferOffsetAlignment = 64;
    context.sixteen_bit_storage = true;
}

VQM_TEST(TestSsimulacra2DispatchLimits) {
    using namespace vqm;
    using namespace vqm::internal;
    VulkanContext context;
    Configure(context);
    Ssimulacra2Config config{.size = {1920, 1080}};
    config.gpu_resident_images = 0;
    config.streaming_inputs = false;
    auto& limits = context.limits.maxComputeWorkGroupCount;
    // 32x32 scale: 60 by 34 groups, with three channel groups in z.
    limits[0] = 60;
    limits[1] = 34;
    limits[2] = 3;
    CHECK(PlanSsimulacra2Memory(config, context).arena_bytes > 0);
    limits[0] = 59;
    CHECK_THROWS_AS(
        PlanSsimulacra2Memory(config, context), std::invalid_argument
    );
    limits[0] = 60;
    limits[1] = 33;
    CHECK_THROWS_AS(
        PlanSsimulacra2Memory(config, context), std::invalid_argument
    );
    limits[1] = 34;
    limits[2] = 2;
    CHECK_THROWS_AS(
        PlanSsimulacra2Memory(config, context), std::invalid_argument
    );

    // The 16x32 fallback has twice as many y groups as the normal tile.
    context.limits.maxComputeSharedMemorySize = 16384;
    limits[1] = 68;
    limits[2] = 3;
    // The reduction still needs 256 lanes in X with the smaller scale tile.
    context.limits.maxComputeWorkGroupSize[0] = 255;
    CHECK_THROWS_AS(
        PlanSsimulacra2Memory(config, context), std::invalid_argument
    );
    context.limits.maxComputeWorkGroupSize[0] = 256;
    CHECK(PlanSsimulacra2Memory(config, context).arena_bytes > 0);
    limits[1] = 67;
    CHECK_THROWS_AS(
        PlanSsimulacra2Memory(config, context), std::invalid_argument
    );

    Configure(context);
    config.size = {8, 8};
    // Even tiny images dispatch 18 norm groups in the reduction.
    limits[0] = 17;
    CHECK_THROWS_AS(
        PlanSsimulacra2Memory(config, context), std::invalid_argument
    );
    limits[0] = 18;
    CHECK(PlanSsimulacra2Memory(config, context).arena_bytes > 0);
    config.chroma_filter = static_cast<ChromaFilter>(255);
    CHECK_THROWS_AS(
        PlanSsimulacra2Memory(config, context), std::invalid_argument
    );
}

VQM_TEST(TestButteraugliDispatchLimits) {
    using namespace vqm;
    using namespace vqm::internal;
    VulkanContext context;
    Configure(context);
    ButteraugliConfig config{.size = {1920, 1080}};
    config.gpu_resident_images = 0;
    config.streaming_inputs = false;
    auto& limits = context.limits.maxComputeWorkGroupCount;
    // The linear diff and reduction passes need ceil(1920*1080/256)
    // groups; tiled passes need 34 y groups and three channels in z.
    limits[0] = 8100;
    limits[1] = 34;
    limits[2] = 3;
    // Check the fixed workgroup dimensions and largest shared-memory tile.
    context.limits.maxComputeWorkGroupSize[0] = 255;
    CHECK_THROWS_AS(
        PlanButteraugliMemory(config, context), std::invalid_argument
    );
    context.limits.maxComputeWorkGroupSize[0] = 256;
    context.limits.maxComputeSharedMemorySize = 29375;
    CHECK_THROWS_AS(
        PlanButteraugliMemory(config, context), std::invalid_argument
    );
    context.limits.maxComputeSharedMemorySize = 29376;
    CHECK(PlanButteraugliMemory(config, context).arena_bytes > 0);
    limits[0] = 8099;
    CHECK_THROWS_AS(
        PlanButteraugliMemory(config, context), std::invalid_argument
    );
    limits[0] = 8100;
    limits[1] = 33;
    CHECK_THROWS_AS(
        PlanButteraugliMemory(config, context), std::invalid_argument
    );
    limits[1] = 34;
    limits[2] = 2;
    CHECK_THROWS_AS(
        PlanButteraugliMemory(config, context), std::invalid_argument
    );

    Configure(context);
    config.size = {8000, 2200};
    // 17.6 megapixels is within a 4 GiB storage range but exceeds the
    // common 65,535-group dispatch limit on the flat kernels.
    CHECK_THROWS_AS(
        PlanButteraugliMemory(config, context), std::invalid_argument
    );
    limits[0] = 68750;
    CHECK(PlanButteraugliMemory(config, context).arena_bytes > 0);
    limits[0] = 68749;
    CHECK_THROWS_AS(
        PlanButteraugliMemory(config, context), std::invalid_argument
    );

    Configure(context);
    config.size = {33, 17};
    limits[0] = 3;
    limits[1] = 1;
    limits[2] = 3;
    CHECK(PlanButteraugliMemory(config, context).arena_bytes > 0);
    config.chroma_filter = static_cast<ChromaFilter>(255);
    CHECK_THROWS_AS(
        PlanButteraugliMemory(config, context), std::invalid_argument
    );
}
}  // namespace

VQM_TEST(TestSsimulacra2MemoryPlan) {
    using namespace vqm;
    using namespace vqm::internal;
    VulkanContext context;
    Configure(context);
    constexpr std::size_t pixels = 1920u * 1080u;
    Ssimulacra2Config config;
    config.size = {1920, 1080};
    config.streaming_inputs = false;
    config.gpu_resident_images = 0;
    const auto base = PlanSsimulacra2Memory(config, context);
    config.streaming_inputs = true;
    const auto stream = PlanSsimulacra2Memory(config, context);
    CHECK(stream.arena_bytes == base.arena_bytes + 32 * pixels);
    CHECK(
        stream.fallback_staging_bytes ==
        base.fallback_staging_bytes + 32 * pixels
    );
    config.streaming_inputs = false;
    config.gpu_resident_images = 2;
    const auto resident = PlanSsimulacra2Memory(config, context);
    CHECK(resident.arena_bytes == base.arena_bytes + 32 * pixels);
    CHECK(
        resident.fallback_staging_bytes ==
        base.fallback_staging_bytes + 16 * pixels
    );
    config.gpu_resident_images = 0;
    config.half_planes = true;
    CHECK(PlanSsimulacra2Memory(config, context).arena_bytes < base.arena_bytes);
    context.sixteen_bit_storage = false;
    CHECK_THROWS_AS(PlanSsimulacra2Memory(config, context), std::runtime_error);
    config.half_planes = false;
    config.input_bytes = 4096;
    CHECK(
        PlanSsimulacra2Memory(config, context).arena_bytes ==
        base.arena_bytes + 4096
    );
    config.input_bytes = std::numeric_limits<std::size_t>::max();
    CHECK_THROWS_AS(
        PlanSsimulacra2Memory(config, context), std::invalid_argument
    );
    config.input_bytes = 0;
    context.limits.maxStorageBufferRange =
        static_cast<std::uint32_t>(base.arena_bytes - 1);
    CHECK_THROWS_AS(
        PlanSsimulacra2Memory(config, context), std::invalid_argument
    );
}

VQM_TEST(TestButteraugliMemoryPlan) {
    using namespace vqm;
    using namespace vqm::internal;
    VulkanContext context;
    Configure(context);
    constexpr std::size_t pixels = 1920u * 1080u;
    ButteraugliConfig config;
    config.size = {1920, 1080};
    config.streaming_inputs = false;
    config.gpu_resident_images = 0;
    const auto base = PlanButteraugliMemory(config, context);
    // The two chains reuse temporary planes across the image and diff stages.
    // Keep a generous budget to catch an accidental return to separate scratch
    // allocations without pinning Vulkan allocation alignment or table sizes.
    CHECK(base.arena_bytes < 140 * pixels);
    config.streaming_inputs = true;
    const auto stream = PlanButteraugliMemory(config, context);
    CHECK(stream.arena_bytes == base.arena_bytes + 32 * pixels);
    CHECK(
        stream.fallback_staging_bytes ==
        base.fallback_staging_bytes + 32 * pixels
    );
    config.streaming_inputs = false;
    config.gpu_resident_images = 2;
    const auto resident = PlanButteraugliMemory(config, context);
    CHECK(resident.arena_bytes == base.arena_bytes + 32 * pixels);
    CHECK(
        resident.fallback_staging_bytes ==
        base.fallback_staging_bytes + 16 * pixels
    );
    config.gpu_resident_images = 0;
    config.input_bytes = 4096;
    CHECK(
        PlanButteraugliMemory(config, context).arena_bytes ==
        base.arena_bytes + 4096
    );
    config.input_bytes = std::numeric_limits<std::size_t>::max();
    CHECK_THROWS_AS(
        PlanButteraugliMemory(config, context), std::invalid_argument
    );
    config.input_bytes = 0;
    context.limits.maxStorageBufferRange =
        static_cast<std::uint32_t>(base.arena_bytes - 1);
    CHECK_THROWS_AS(
        PlanButteraugliMemory(config, context), std::invalid_argument
    );
}
