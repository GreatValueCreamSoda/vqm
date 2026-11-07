// VQM resource and budget contracts for ColorVideoVDP.
// BSD-3-Clause; see LICENSE.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "support.h"
#include "cvvdp.h"
#include "metrics/cvvdp/cvvdp_internal.h"
#include "vqm_memory.h"
#include "vqm_metrics.h"

namespace {
using namespace vqm;
using namespace vqm::internal;

void ConfigureLimits(VulkanContext& context) {
    auto& limits = context.limits;
    limits.maxComputeSharedMemorySize = 32768;
    limits.maxComputeWorkGroupInvocations = 256;
    limits.maxComputeWorkGroupSize[0] = 256;
    limits.maxComputeWorkGroupSize[1] = 8;
    limits.maxComputeWorkGroupSize[2] = 1;
    limits.maxComputeWorkGroupCount[0] = 65535;
    limits.maxComputeWorkGroupCount[1] = 65535;
    limits.maxComputeWorkGroupCount[2] = 1;
    limits.maxStorageBufferRange = UINT32_MAX;
    limits.maxPushConstantsSize = 128;
    limits.maxPerStageDescriptorStorageBuffers = 6;
    limits.maxDescriptorSetStorageBuffers = 6;
}

MemoryAllocation Allocation(std::uint64_t bytes, std::uint32_t heap) {
    return {bytes, bytes, 0, heap};
}

}  // namespace

VQM_TEST(TestCvvdpAuxiliaryBudgetAccounting) {
    MemoryScenario scenario;
    scenario.supported = true;
    scenario.arena = Allocation(10, 0);
    scenario.staging = Allocation(20, 1);
    scenario.auxiliary = {Allocation(30, 0), Allocation(40, 1)};
    // Each worker consumes 40 bytes on heap 0 and 60 on heap 1. Histories
    // remain separate allocations, even when some occupy the same heap.
    constexpr std::array<std::uint64_t, 2> budgets{400, 600};
    CHECK(MaxWorkersForBudget(scenario, budgets) == 10);
    constexpr std::array<std::uint64_t, 2> reduced{400, 599};
    CHECK(MaxWorkersForBudget(scenario, reduced) == 9);
    scenario.auxiliary.back().heap = 2;
    CHECK_THROWS_AS(
        MaxWorkersForBudget(scenario, budgets), std::invalid_argument
    );
    scenario.auxiliary.back().heap = 0;
    scenario.arena.allocation_bytes = std::numeric_limits<std::uint64_t>::max();
    CHECK_THROWS_AS(MaxWorkersForBudget(scenario, budgets), std::overflow_error);
}

VQM_TEST(TestCvvdpAuxiliaryWorstCaseBudget) {
    MetricMemoryRequirements requirements;
    requirements.direct.supported = true;
    requirements.direct.arena = Allocation(10, 0);
    requirements.direct.auxiliary = {Allocation(30, 1)};
    requirements.staged.supported = true;
    requirements.staged.arena = Allocation(20, 0);
    requirements.staged.staging = Allocation(5, 1);
    requirements.staged.auxiliary = {Allocation(30, 1)};
    constexpr std::array<std::uint64_t, 2> budgets{200, 350};
    CHECK(
        MaxWorkersForBudgetWorstCase(std::span(&requirements, 1), budgets) == 10
    );
    // Independent metrics add their per-heap worst cases, rather than taking
    // a single largest allocation or sharing one clip's history between workers.
    const std::array pair{requirements, requirements};
    CHECK(MaxWorkersForBudgetWorstCase(pair, budgets) == 5);
}

VQM_TEST(TestCvvdpMemoryPlannerGeometryAndRateMatrix) {
    VulkanContext context;
    ConfigureLimits(context);
    CvvdpConfig config;
    config.gpu_resident_images = 0;
    config.streaming_inputs = false;
    for (const auto size : {Size{1920, 1080}, Size{3840, 2160}}) {
        config.size = size;
        for (const auto fps : {24u, 30u, 60u, 120u}) {
            config.fps_numerator = fps;
            const auto plan = PlanCvvdpMemory(config, context);
            const auto taps = CvvdpTemporalTaps(config);
            const std::uint64_t expected =
                3 * size.width * size.height * taps * 4;
            CHECK(plan.auxiliary_bytes.size() == 2);
            CHECK(plan.auxiliary_bytes[0] == expected);
            CHECK(plan.auxiliary_bytes[1] == expected);
            CHECK(plan.arena_bytes > 0);
            CHECK(plan.fallback_staging_bytes > 0);
        }
    }
    config.temporal = false;
    CHECK(PlanCvvdpMemory(config, context).auxiliary_bytes.empty());
    config.temporal = true;
    config.fps_numerator = 240;
    CHECK_THROWS_AS(PlanCvvdpMemory(config, context), std::invalid_argument);
    config.size = {1, 32};
    CHECK_THROWS_AS(PlanCvvdpMemory(config, context), std::invalid_argument);
    config.size = {std::numeric_limits<std::size_t>::max(), 32};
    CHECK_THROWS_AS(PlanCvvdpMemory(config, context), std::invalid_argument);
}

VQM_TEST(TestCvvdpMemoryPlannerStorageAndDispatchLimits) {
    VulkanContext context;
    ConfigureLimits(context);
    CvvdpConfig config;
    config.size = {1920, 1080};
    config.gpu_resident_images = 0;
    config.streaming_inputs = false;
    config.fps_numerator = 60;
    const auto plan = PlanCvvdpMemory(config, context);
    context.limits.maxStorageBufferRange =
        static_cast<std::uint32_t>(plan.auxiliary_bytes[0] - 1);
    CHECK_THROWS_AS(PlanCvvdpMemory(config, context), std::invalid_argument);
    ConfigureLimits(context);
    context.limits.maxPerStageDescriptorStorageBuffers = 5;
    CHECK_THROWS_AS(PlanCvvdpMemory(config, context), std::invalid_argument);
    // Image-only pipelines retain the existing four-binding portability floor.
    config.temporal = false;
    context.limits.maxPerStageDescriptorStorageBuffers = 4;
    context.limits.maxDescriptorSetStorageBuffers = 4;
    CHECK(PlanCvvdpMemory(config, context).arena_bytes > 0);
    ConfigureLimits(context);
    context.limits.maxComputeWorkGroupCount[0] = 8099;
    CHECK_THROWS_AS(PlanCvvdpMemory(config, context), std::invalid_argument);
    ConfigureLimits(context);
    context.limits.maxComputeWorkGroupInvocations = 255;
    CHECK_THROWS_AS(PlanCvvdpMemory(config, context), std::invalid_argument);
}

VQM_TEST(TestCvvdpNonfiniteStatusAndHistoryRecovery) {
    if (!vqm::test::GpuAvailable()) return;
    const auto original = vqm::test::MakeGradient(0);
    const auto distorted = vqm::test::MakeGradient(1);
    auto invalid = original;
    invalid.format.type = PixelType::Float;
    invalid.data.resize(RequiredBytes(invalid.format, invalid.size));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(invalid.data.data(), &nan, sizeof(nan));
    CvvdpConfig config;
    config.size = original.size;
    const auto requirements = GetMemoryRequirements(config);
    for (const auto policy : {MemoryPolicy::Direct, MemoryPolicy::Staged}) {
        const auto& path = policy == MemoryPolicy::Direct ? requirements.direct
                                                          : requirements.staged;
        if (!path.supported) continue;
        config.memory_policy = policy;
        CvvdpPipeline pipeline(config);
        CHECK_THROWS_AS(
            pipeline.Push(invalid, distorted), std::invalid_argument
        );
        CHECK_THROWS_AS(pipeline.Finalize(), std::logic_error);
        pipeline.Reset();
        pipeline.Push(original, distorted);
        // Processing the deferred first frame precedes the next upload's
        // validation result. Reset must clear those accumulated features and
        // ignore every invalid history slot, including on the staged path.
        CHECK_THROWS_AS(
            pipeline.Push(invalid, distorted), std::invalid_argument
        );
        CHECK_THROWS_AS(pipeline.Push(original, distorted), std::logic_error);
        pipeline.Reset();
        CHECK(!pipeline.Finalize());
        pipeline.Reset();
        pipeline.Push(original, distorted);
        pipeline.Push(original, distorted);
        const auto actual = pipeline.Finalize();
        CvvdpPipeline fresh(config);
        fresh.Push(original, distorted);
        fresh.Push(original, distorted);
        const auto expected = fresh.Finalize();
        CHECK(actual && expected);
        if (actual && expected) CHECK_NEAR(actual->score, expected->score, 1e-7);
        CHECK(
            pipeline.GetMemoryUsage().uses_staging ==
            (policy == MemoryPolicy::Staged)
        );
    }
}

VQM_TEST(TestCvvdpUnifiedFormatValidationBeforeAllocation) {
    MetricsConfig config;
    config.size = {32, 32};
    config.cvvdp = CvvdpConfig{.size = config.size, .temporal = false};
    config.ssimulacra2 = Ssimulacra2Config{.size = config.size};
    // Log100 is accepted by the shared image contract but is outside CVVDP's
    // supported photometry. Reject the configuration before another metric
    // allocates or processes it; neither check needs a Vulkan device.
    config.original_format.transfer = TransferCharacteristics::Log100;
    CHECK_THROWS_AS(Metrics(config), std::invalid_argument);
    CHECK_THROWS_AS(GetMemoryRequirements(config), std::invalid_argument);
    config.original_format = {};
    config.distorted_format.transfer = TransferCharacteristics::Log100;
    CHECK_THROWS_AS(Metrics(config), std::invalid_argument);
    CHECK_THROWS_AS(GetMemoryRequirements(config), std::invalid_argument);
    config.distorted_format = {};
    config.original_format.alpha = AlphaType::Straight;
    CHECK_THROWS_AS(Metrics(config), std::invalid_argument);
    CHECK_THROWS_AS(GetMemoryRequirements(config), std::invalid_argument);
    config.original_format = {};
    config.original_format.transfer = TransferCharacteristics::AribB67;
    config.cvvdp->display = CvvdpDisplay::StandardHdr();
    config.cvvdp->display.ambient_illuminance = 0;
    CHECK_THROWS_AS(Metrics(config), std::invalid_argument);
    CHECK_THROWS_AS(GetMemoryRequirements(config), std::invalid_argument);
}
