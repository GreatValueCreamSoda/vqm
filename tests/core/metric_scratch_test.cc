#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "metrics/metric_scratch.h"
#include "support.h"
#include "vqm_metrics.h"

namespace {
using namespace vqm;

Image YuvFrame(std::size_t seed) {
    Image image;
    image.size = {216, 80};
    image.format.family = ColorFamily::Yuv;
    image.format.matrix = MatrixCoefficients::Bt709;
    image.format.transfer = TransferCharacteristics::Bt709;
    image.format.type = PixelType::Byte;
    image.format.depth = 8;
    image.data.resize(RequiredBytes(image.format, image.size));
    for (std::size_t index = 0; index < image.data.size(); ++index)
        image.data[index] =
            static_cast<std::uint8_t>(32 + (index * 13 + seed * 29) % 192);
    return image;
}

std::uint64_t AllocationBytes(const PipelineMemoryUsage& usage) {
    auto bytes = usage.arena.allocation_bytes + usage.staging.allocation_bytes;
    for (const auto& allocation : usage.auxiliary)
        bytes += allocation.allocation_bytes;
    return bytes;
}

void CheckScenario(
    const PipelineMemoryUsage& usage, const MetricMemoryRequirements& memory
) {
    const auto& scenario = usage.uses_staging || !memory.direct.supported
                               ? memory.staged
                               : memory.direct;
    CHECK(scenario.supported);
    CHECK(usage.arena.buffer_bytes == scenario.arena.buffer_bytes);
    CHECK(usage.arena.allocation_bytes == scenario.arena.allocation_bytes);
    CHECK(usage.staging.allocation_bytes == scenario.staging.allocation_bytes);
    CHECK(usage.auxiliary.size() == scenario.auxiliary.size());
    for (std::size_t index = 0;
         index < std::min(usage.auxiliary.size(), scenario.auxiliary.size());
         ++index) {
        CHECK(
            usage.auxiliary[index].buffer_bytes ==
            scenario.auxiliary[index].buffer_bytes
        );
        CHECK(
            usage.auxiliary[index].allocation_bytes ==
            scenario.auxiliary[index].allocation_bytes
        );
        CHECK(usage.auxiliary[index].heap == scenario.auxiliary[index].heap);
    }
}

MetricsConfig Config(Size size, MemoryPolicy policy) {
    MetricsConfig config;
    config.size = size;
    config.memory_policy = policy;
    config.ssimulacra2.emplace();
    config.butteraugli.emplace();
    config.vmaf_v0.emplace();
    config.vmaf_v1.emplace();
    config.vmaf_v1->hfr = true;
    config.cvvdp.emplace();
    config.psnr_hvs.emplace();
    return config;
}

template <class Pipelines>
void CheckAllMemory(const Pipelines& metrics, const MetricsConfig& config) {
    const auto actual = metrics.GetMemoryUsage();
    const auto planned = GetMemoryRequirements(config);
    CHECK(actual.scratch.has_value());
    CHECK(planned.scratch.has_value());
    CheckScenario(*actual.scratch, *planned.scratch);
    std::uint64_t total = AllocationBytes(*actual.scratch);
    const auto check = [&](const auto& usage, const auto& memory) {
        CHECK(usage.has_value());
        CHECK(memory.has_value());
        CHECK(usage->arena.buffer_bytes == 0);
        CHECK(usage->staging.buffer_bytes == 0);
        CheckScenario(*usage, *memory);
        total += AllocationBytes(*usage);
    };
    check(actual.spatial, planned.spatial);
    check(actual.vmaf_v0, planned.vmaf_v0);
    check(actual.vmaf_v1, planned.vmaf_v1);
    check(actual.cvvdp, planned.cvvdp);
    check(actual.psnr_hvs, planned.psnr_hvs);
    std::uint64_t maximum = 0;
    std::uint64_t independent = 0;
    const auto add = [&](auto metric) {
        metric.size = config.size;
        metric.gpu_resident_images = 0;
        metric.memory_policy = config.memory_policy;
        if constexpr (requires { metric.input_format; })
            metric.input_format = config.original_format;
        const auto requirements = GetMemoryRequirements(metric);
        CHECK(requirements.scratch.reusable);
        maximum = std::max(maximum, requirements.scratch.arena_bytes);
        const auto& scenario = actual.scratch->uses_staging
                                   ? requirements.staged
                                   : requirements.direct;
        independent +=
            scenario.arena.allocation_bytes + scenario.staging.allocation_bytes;
        for (const auto& allocation : scenario.auxiliary)
            independent += allocation.allocation_bytes;
    };
    add(*config.ssimulacra2);
    add(*config.butteraugli);
    add(*config.vmaf_v0);
    add(*config.vmaf_v1);
    add(*config.cvvdp);
    add(*config.psnr_hvs);
    CHECK(actual.scratch->arena.buffer_bytes == maximum);
    CHECK(total < independent);
}

void CheckAllScores(MemoryPolicy policy) {
    const auto original = YuvFrame(0);
    const auto distorted = YuvFrame(99);
    auto config = Config(original.size, policy);
    config.original_format = original.format;
    config.distorted_format = distorted.format;
    Metrics shared(config);
    CheckAllMemory(shared, config);
    std::vector<MetricsConfig> configs;
    for (std::size_t index = 0; index < 6; ++index) {
        auto single = config;
        if (index != 0) single.ssimulacra2.reset();
        if (index != 1) single.butteraugli.reset();
        if (index != 2) single.vmaf_v0.reset();
        if (index != 3) single.vmaf_v1.reset();
        if (index != 4) single.cvvdp.reset();
        if (index != 5) single.psnr_hvs.reset();
        configs.push_back(single);
    }
    std::vector<Metrics> independent;
    for (const auto& single : configs) independent.emplace_back(single);
    const auto compare = [](const MetricsResult& actual,
                            const MetricsResult& expected) {
        CHECK(actual.frame_index == expected.frame_index);
        const auto check = [](const auto& value, const auto& reference) {
            if (reference) {
                CHECK(value.has_value());
                if (value) CHECK_NEAR(*value, *reference, 1e-9);
            }
        };
        check(actual.ssimulacra2, expected.ssimulacra2);
        check(actual.butteraugli, expected.butteraugli);
        check(actual.vmaf_v0, expected.vmaf_v0);
        check(actual.vmaf_v1, expected.vmaf_v1);
        check(actual.psnr_hvs, expected.psnr_hvs);
    };
    for (int sequence = 0; sequence < 2; ++sequence) {
        const std::size_t frames = sequence == 0 ? 5 : 1;
        std::vector<MetricsResult> emitted;
        std::array<std::vector<MetricsResult>, 6> expected;
        for (std::size_t frame = 0; frame < frames; ++frame) {
            const auto source = YuvFrame(frame % 3);
            if (const auto value = shared.Compute(source, distorted))
                emitted.push_back(*value);
            for (std::size_t index = 0; index < independent.size(); ++index) {
                if (const auto value =
                        independent[index].Compute(source, distorted))
                    expected[index].push_back(*value);
            }
        }
        while (const auto value = shared.Flush()) emitted.push_back(*value);
        CHECK(emitted.size() == frames);
        for (std::size_t index = 0; index < independent.size(); ++index) {
            while (const auto value = independent[index].Flush())
                expected[index].push_back(*value);
            CHECK(expected[index].size() == frames);
            for (const auto& value : expected[index]) {
                CHECK(value.frame_index < emitted.size());
                if (value.frame_index < emitted.size())
                    compare(emitted[value.frame_index], value);
            }
        }
        const auto final = shared.Finalize();
        const auto reference = independent[4].Finalize();
        CHECK(final.cvvdp.has_value());
        CHECK(reference.cvvdp.has_value());
        if (final.cvvdp && reference.cvvdp)
            CHECK_NEAR(final.cvvdp->score, reference.cvvdp->score, 1e-9);
        shared.Reset();
        for (auto& metric : independent) metric.Reset();
    }
}
}  // namespace

VQM_TEST(TestAllMetricScratchAndPrivateHistory) {
    if (!vqm::test::GpuAvailable()) return;
    CheckAllScores(vqm::MemoryPolicy::Automatic);
    CheckAllScores(vqm::MemoryPolicy::Staged);
}

VQM_TEST(TestMetricPipelinesAllMetricMemory) {
    if (!vqm::test::GpuAvailable()) return;
    const auto image = YuvFrame(0);
    for (const auto policy : {MemoryPolicy::Automatic, MemoryPolicy::Staged}) {
        auto config = Config(image.size, policy);
        config.original_format = image.format;
        config.distorted_format = image.format;
        MetricPipelines pipelines(config);
        CHECK(pipelines.Ssimulacra2() != nullptr);
        CHECK(pipelines.Butteraugli() != nullptr);
        CHECK(pipelines.VmafV0() != nullptr);
        CHECK(pipelines.VmafV1() != nullptr);
        CHECK(pipelines.Cvvdp() != nullptr);
        CHECK(pipelines.PsnrHvs() != nullptr);
        CheckAllMemory(pipelines, config);
    }
}

VQM_TEST(TestSharedScratchPrivateWorkerBudgets) {
    using namespace vqm;
    MetricsMemoryRequirements requirements;
    requirements.scratch.emplace();
    requirements.scratch->direct.supported = true;
    requirements.scratch->direct.arena = {.allocation_bytes = 100, .heap = 0};
    requirements.scratch->staged.supported = true;
    requirements.scratch->staged.arena = {.allocation_bytes = 80, .heap = 0};
    requirements.scratch->staged.staging = {.allocation_bytes = 20, .heap = 1};
    requirements.vmaf_v0.emplace();
    requirements.vmaf_v0->direct.supported = true;
    requirements.vmaf_v0->direct.auxiliary.push_back(
        {.allocation_bytes = 20, .heap = 0}
    );
    requirements.vmaf_v0->staged = requirements.vmaf_v0->direct;
    requirements.cvvdp.emplace();
    requirements.cvvdp->direct.supported = true;
    requirements.cvvdp->direct.auxiliary.push_back(
        {.allocation_bytes = 50, .heap = 1}
    );
    requirements.cvvdp->staged = requirements.cvvdp->direct;
    CHECK(
        MaxWorkersForBudgetWorstCase(
            requirements, std::array<std::uint64_t, 2>{240, 140}
        ) == 2
    );
    CHECK(
        MaxWorkersForBudgetWorstCase(
            requirements, std::array<std::uint64_t, 2>{239, 140}
        ) == 1
    );
    CHECK(
        MaxWorkersForBudgetWorstCase(
            requirements, std::array<std::uint64_t, 2>{240, 139}
        ) == 1
    );
}

VQM_TEST(TestSharedScratchSubmissionOwnership) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm;
    const auto original = vqm::test::MakeGradient(0);
    const auto distorted = vqm::test::MakeGradient(1);
    Ssimulacra2Config config{.size = original.size};
    config.gpu_resident_images = 0;
    const auto memory = GetMemoryRequirements(config);
    internal::MetricScratchBuilder planner;
    planner.Add(memory);
    const auto scratch = planner.Build(nullptr);
    auto first = internal::MetricScratchAccess::Create<Ssimulacra2Pipeline>(
        config, nullptr, scratch
    );
    auto second = internal::MetricScratchAccess::Create<Ssimulacra2Pipeline>(
        config, nullptr, scratch
    );
    const auto token = first->Submit(original, distorted);
    CHECK_THROWS_AS(second->Submit(original, distorted), std::logic_error);
    CHECK_THROWS_AS(
        internal::MetricScratchAccess::Create<Ssimulacra2Pipeline>(
            config, nullptr, scratch
        ),
        std::logic_error
    );
    const auto score = first->Finish(token).Score();
    CHECK_NEAR(second->Compute(original, distorted).Score(), score, 0.0);
    first.reset();
    CHECK_NEAR(second->Compute(original, distorted).Score(), score, 0.0);
    config.gpu_resident_images = 1;
    CHECK(!GetMemoryRequirements(config).scratch.reusable);
    CHECK_THROWS_AS(
        planner.Add(GetMemoryRequirements(config)), std::invalid_argument
    );
}
