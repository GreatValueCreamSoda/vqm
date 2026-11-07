// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "support.h"
#include "vqm_memory.h"

namespace {
void CheckUsage(
    const vqm::MetricMemoryRequirements& planned,
    const vqm::PipelineMemoryUsage& used
) {
    const auto& selected = used.uses_staging ? planned.staged : planned.direct;
    CHECK(selected.supported);
    CHECK(used.arena.buffer_bytes == selected.arena.buffer_bytes);
    CHECK(used.arena.allocation_bytes == selected.arena.allocation_bytes);
    CHECK(used.arena.memory_type == selected.arena.memory_type);
    CHECK(used.arena.heap == selected.arena.heap);
    if (used.uses_staging) {
        CHECK(used.staging.buffer_bytes == selected.staging.buffer_bytes);
        CHECK(
            used.staging.allocation_bytes == selected.staging.allocation_bytes
        );
        CHECK(used.staging.memory_type == selected.staging.memory_type);
        CHECK(used.staging.heap == selected.staging.heap);
    }
}

template <class Pipeline, class Config>
void CheckPolicy(Config config, vqm::MemoryPolicy policy) {
    config.memory_policy = policy;
    const auto planned = vqm::GetMemoryRequirements(config);
    const auto& selected =
        policy == vqm::MemoryPolicy::Direct ? planned.direct : planned.staged;
    const auto& excluded =
        policy == vqm::MemoryPolicy::Direct ? planned.staged : planned.direct;
    CHECK(!excluded.supported);
    if (!selected.supported) {
        CHECK_THROWS_AS(Pipeline(config), std::runtime_error);
        return;
    }
    Pipeline pipeline(config);
    const auto used = pipeline.GetMemoryUsage();
    CHECK(used.uses_staging == (policy == vqm::MemoryPolicy::Staged));
    CheckUsage(planned, used);
}
}  // namespace

VQM_TEST(TestInvalidMemoryPolicyRejectedBeforeDeviceSelection) {
    const auto invalid = static_cast<vqm::MemoryPolicy>(255);
    vqm::Ssimulacra2Config ss;
    ss.memory_policy = invalid;
    CHECK_THROWS_AS(vqm::GetMemoryRequirements(ss), std::invalid_argument);
    vqm::ButteraugliConfig butter;
    butter.memory_policy = invalid;
    CHECK_THROWS_AS(vqm::GetMemoryRequirements(butter), std::invalid_argument);
    vqm::VmafConfig v0;
    v0.memory_policy = invalid;
    CHECK_THROWS_AS(vqm::GetMemoryRequirements(v0), std::invalid_argument);
    vqm::VmafV1Config v1;
    v1.memory_policy = invalid;
    CHECK_THROWS_AS(vqm::GetMemoryRequirements(v1), std::invalid_argument);
    vqm::SharedMetricConfig shared;
    shared.memory_policy = invalid;
    CHECK_THROWS_AS(vqm::GetMemoryRequirements(shared), std::invalid_argument);
}

VQM_TEST(TestWorkerBudgetCountsAllocationsPerHeap) {
    vqm::MemoryScenario scenario;
    scenario.supported = true;
    scenario.arena = {.buffer_bytes = 90, .allocation_bytes = 100, .heap = 0};
    scenario.staging = {.buffer_bytes = 45, .allocation_bytes = 50, .heap = 1};
    const std::array<std::uint64_t, 2> separate{450, 130};
    CHECK(vqm::MaxWorkersForBudget(scenario, separate) == 2);
    scenario.staging.heap = 0;
    CHECK(vqm::MaxWorkersForBudget(scenario, separate) == 3);
    const std::array combined{scenario, scenario};
    CHECK(vqm::MaxWorkersForBudget(combined, separate) == 1);
    vqm::MetricMemoryRequirements alternatives;
    alternatives.direct = scenario;
    alternatives.direct.staging = {};
    alternatives.staged = scenario;
    alternatives.staged.staging.heap = 1;
    const std::array two{alternatives, alternatives};
    CHECK(vqm::MaxWorkersForBudgetWorstCase(two, separate) == 1);
    CHECK_THROWS_AS(
        vqm::MaxWorkersForBudget(scenario, std::span<const std::uint64_t>{}),
        std::invalid_argument
    );
    scenario.staging.allocation_bytes =
        std::numeric_limits<std::uint64_t>::max();
    CHECK_THROWS_AS(
        vqm::MaxWorkersForBudget(scenario, separate), std::overflow_error
    );
    scenario.supported = false;
    CHECK_THROWS_AS(
        vqm::MaxWorkersForBudget(scenario, separate), std::invalid_argument
    );
}

VQM_TEST(TestAvailableHeapBudgetsAreOptionalAndSaturate) {
    vqm::MetricMemoryRequirements requirements;
    requirements.heaps.resize(2);
    CHECK(!vqm::AvailableHeapBudgets(requirements).has_value());
    requirements.heaps[0].budget_available = true;
    requirements.heaps[0].budget_bytes = 1000;
    requirements.heaps[0].usage_bytes = 250;
    requirements.heaps[1].budget_available = true;
    requirements.heaps[1].budget_bytes = 100;
    requirements.heaps[1].usage_bytes = 125;
    const auto budgets = vqm::AvailableHeapBudgets(requirements);
    CHECK(budgets.has_value());
    if (budgets) {
        CHECK((*budgets)[0] == 750);
        CHECK((*budgets)[1] == 0);
    }
}

VQM_TEST(TestMemoryQueryReportsDeviceRequirements) {
    if (!vqm::test::GpuAvailable()) {
        vqm::test::MarkSkip("no usable Vulkan device", true);
        return;
    }
    constexpr vqm::Size kSize{320, 320};
    vqm::Ssimulacra2Config ssimu;
    ssimu.size = kSize;
    const auto ss = vqm::GetMemoryRequirements(ssimu);
    CHECK(!ss.heaps.empty());
    CHECK(ss.direct.supported || ss.staged.supported);
    const auto& actual = ss.direct.supported ? ss.direct : ss.staged;
    CHECK(actual.arena.buffer_bytes > 0);
    CHECK(actual.arena.allocation_bytes >= actual.arena.buffer_bytes);
    CHECK(actual.arena.heap < ss.heaps.size());
    if (ss.staged.supported) {
        CHECK(ss.staged.staging.buffer_bytes > 0);
        CHECK(
            ss.staged.staging.allocation_bytes >= ss.staged.staging.buffer_bytes
        );
        CHECK(ss.staged.staging.heap < ss.heaps.size());
    }
    {
        vqm::Ssimulacra2Pipeline pipeline(ssimu);
        CheckUsage(ss, pipeline.GetMemoryUsage());
    }
    vqm::ButteraugliConfig butter;
    butter.size = kSize;
    const auto b = vqm::GetMemoryRequirements(butter);
    CHECK(b.direct.supported || b.staged.supported);
    {
        vqm::ButteraugliPipeline pipeline(butter);
        CheckUsage(b, pipeline.GetMemoryUsage());
    }
    vqm::VmafConfig v0;
    v0.size = kSize;
    const auto a = vqm::GetMemoryRequirements(v0);
    CHECK(a.direct.supported || a.staged.supported);
    {
        vqm::VmafPipeline pipeline(v0);
        CheckUsage(a, pipeline.GetMemoryUsage());
    }
    vqm::VmafV1Config v1;
    v1.size = kSize;
    v1.input_format.type = vqm::PixelType::Byte;
    v1.input_format.family = vqm::ColorFamily::Yuv;
    v1.input_format.matrix = vqm::MatrixCoefficients::Bt709;
    v1.input_format.depth = 8;
    v1.input_format.subsample_w = 1;
    v1.input_format.subsample_h = 1;
    const auto b1 = vqm::GetMemoryRequirements(v1);
    CHECK(b1.direct.supported || b1.staged.supported);
    {
        vqm::VmafV1Pipeline pipeline(v1);
        CheckUsage(b1, pipeline.GetMemoryUsage());
    }

    vqm::SharedMetricConfig shared;
    shared.size = kSize;
    const auto planned = vqm::GetMemoryRequirements(shared);
    vqm::SharedMetricWorkspace workspace(shared);
    CheckUsage(planned, workspace.GetMemoryUsage());
}

VQM_TEST(TestStrictMemoryPoliciesMatchQueries) {
    if (!vqm::test::GpuAvailable()) {
        vqm::test::MarkSkip("no usable Vulkan device", true);
        return;
    }
    constexpr vqm::Size kSize{320, 320};
    vqm::Ssimulacra2Config ss;
    ss.size = kSize;
    vqm::ButteraugliConfig butter;
    butter.size = kSize;
    vqm::VmafConfig v0;
    v0.size = kSize;
    vqm::VmafV1Config v1;
    v1.size = kSize;
    v1.input_format.type = vqm::PixelType::Byte;
    v1.input_format.family = vqm::ColorFamily::Yuv;
    v1.input_format.matrix = vqm::MatrixCoefficients::Bt709;
    v1.input_format.depth = 8;
    v1.input_format.subsample_w = 1;
    v1.input_format.subsample_h = 1;
    vqm::SharedMetricConfig shared;
    shared.size = kSize;
    for (auto policy : {vqm::MemoryPolicy::Direct, vqm::MemoryPolicy::Staged}) {
        CheckPolicy<vqm::Ssimulacra2Pipeline>(ss, policy);
        CheckPolicy<vqm::ButteraugliPipeline>(butter, policy);
        CheckPolicy<vqm::VmafPipeline>(v0, policy);
        CheckPolicy<vqm::VmafV1Pipeline>(v1, policy);
        CheckPolicy<vqm::SharedMetricWorkspace>(shared, policy);
    }
    const auto invalid = static_cast<vqm::MemoryPolicy>(255);
    ss.memory_policy = invalid;
    butter.memory_policy = invalid;
    v0.memory_policy = invalid;
    v1.memory_policy = invalid;
    shared.memory_policy = invalid;
    CHECK_THROWS_AS(vqm::Ssimulacra2Pipeline(ss), std::invalid_argument);
    CHECK_THROWS_AS(vqm::ButteraugliPipeline(butter), std::invalid_argument);
    CHECK_THROWS_AS(vqm::VmafPipeline(v0), std::invalid_argument);
    CHECK_THROWS_AS(vqm::VmafV1Pipeline(v1), std::invalid_argument);
    CHECK_THROWS_AS(vqm::SharedMetricWorkspace(shared), std::invalid_argument);
}

VQM_TEST(TestStrictMemoryPoliciesPreserveImageScores) {
    if (!vqm::test::GpuAvailable()) {
        vqm::test::MarkSkip("no usable Vulkan device", true);
        return;
    }
    const auto reference = vqm::test::MakeGradient(0);
    const auto distorted = vqm::test::MakeGradient(1);
    vqm::Ssimulacra2Config ss;
    ss.size = reference.size;
    ss.input_bytes = vqm::RequiredBytes(reference.format, reference.size);
    vqm::ButteraugliConfig butter;
    butter.size = reference.size;
    double ss_reference = 0;
    double butter_reference = 0;
    {
        vqm::Ssimulacra2Pipeline baseline(ss);
        ss_reference = baseline.Compute(reference, distorted).Score();
    }
    {
        vqm::ButteraugliPipeline baseline(butter);
        butter_reference = baseline.Compute(reference, distorted).score;
    }
    for (auto policy : {vqm::MemoryPolicy::Direct, vqm::MemoryPolicy::Staged}) {
        ss.memory_policy = policy;
        butter.memory_policy = policy;
        const auto ss_plan = vqm::GetMemoryRequirements(ss);
        const auto butter_plan = vqm::GetMemoryRequirements(butter);
        const auto ss_supported = policy == vqm::MemoryPolicy::Direct
                                      ? ss_plan.direct.supported
                                      : ss_plan.staged.supported;
        const auto butter_supported = policy == vqm::MemoryPolicy::Direct
                                          ? butter_plan.direct.supported
                                          : butter_plan.staged.supported;
        if (ss_supported) {
            vqm::Ssimulacra2Pipeline pipeline(ss);
            CHECK_NEAR(
                pipeline.Compute(reference, distorted).Score(), ss_reference,
                1e-5
            );
            if (policy == vqm::MemoryPolicy::Staged)
                CHECK_THROWS_AS(
                    pipeline.AllocateImage(reference.format), std::runtime_error
                );
        }
        if (butter_supported) {
            vqm::ButteraugliPipeline pipeline(butter);
            CHECK_NEAR(
                pipeline.Compute(reference, distorted).score, butter_reference,
                1e-5
            );
        }
        vqm::SharedMetricConfig shared;
        shared.size = reference.size;
        shared.memory_policy = policy;
        const auto shared_plan = vqm::GetMemoryRequirements(shared);
        const auto shared_supported = policy == vqm::MemoryPolicy::Direct
                                          ? shared_plan.direct.supported
                                          : shared_plan.staged.supported;
        if (shared_supported) {
            vqm::SharedMetricWorkspace workspace(shared);
            CHECK_NEAR(
                workspace.ComputeSsimulacra2(reference, distorted).Score(),
                ss_reference, 1e-5
            );
            CHECK_NEAR(
                workspace.ComputeButteraugli(reference, distorted).score,
                butter_reference, 1e-5
            );
            CHECK_NEAR(
                workspace.ComputeSsimulacra2(reference, distorted).Score(),
                ss_reference, 1e-5
            );
        }
    }
}
