// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>

#include "butteraugli.h"
#include "support.h"
#include "vmaf.h"
#include "vmaf_v1.h"
#include "vqm_metrics.h"
#include "vqm_shared.h"
#include "vqm_views.h"
#include "metrics/vmaf/pipeline/vmaf_sdr_fixtures.h"

namespace {
using namespace vqm;

struct Pair {
    Image original;
    Image distorted;
};

Pair MakePair(bool alternate) {
    using namespace vqm::test::sdr;
    return {MakeSource(
                PixelType::Float, ColorFamily::Rgb, AlphaType::None, alternate
            ),
            MakeSource(
                PixelType::Byte, ColorFamily::Rgb, AlphaType::None, !alternate
            )};
}

MetricsConfig AllConfig(const Pair& pair) {
    MetricsConfig config;
    config.size = pair.original.size;
    config.original_format = pair.original.format;
    config.distorted_format = pair.distorted.format;
    config.ssimulacra2 = Ssimulacra2Config{.size = config.size};
    config.butteraugli = ButteraugliConfig{.size = config.size};
    VmafConfig v0{.size = config.size};
    v0.chroma_sampling = VmafChromaSampling::Yuv420;
    config.vmaf_v0 = v0;
    VmafV1Config v1;
    v1.size = config.size;
    v1.chroma_sampling = VmafChromaSampling::Yuv420;
    config.vmaf_v1 = v1;
    return config;
}

ImageView View(const Image& image) {
    ImageView view;
    view.size = image.size;
    view.format = image.format;
    const auto planes = internal::PlaneLayout(image.format, image.size);
    for (std::size_t i = 0; i < planes.size(); ++i) {
        if (!planes[i].present()) continue;
        view.planes[i] = {reinterpret_cast<const std::byte*>(
                              image.data.data() + planes[i].offset
                          ),
                          planes[i].width * planes[i].sample_bytes};
    }
    return view;
}

void CheckMemoryRequirements(
    const std::optional<PipelineMemoryUsage>& actual,
    const std::optional<MetricMemoryRequirements>& planned
) {
    CHECK(actual.has_value() == planned.has_value());
    if (!actual || !planned) return;
    const auto& scenario = (actual->uses_staging || !planned->direct.supported)
                               ? planned->staged
                               : planned->direct;
    CHECK(scenario.supported);
    CHECK(actual->arena.buffer_bytes == scenario.arena.buffer_bytes);
    CHECK(actual->arena.allocation_bytes == scenario.arena.allocation_bytes);
    CHECK(actual->staging.buffer_bytes == scenario.staging.buffer_bytes);
    CHECK(actual->staging.allocation_bytes == scenario.staging.allocation_bytes);
    CHECK(actual->auxiliary.size() == scenario.auxiliary.size());
    if (actual->auxiliary.size() != scenario.auxiliary.size()) return;
    for (std::size_t index = 0; index < actual->auxiliary.size(); ++index) {
        const auto& allocation = actual->auxiliary[index];
        const auto& expected = scenario.auxiliary[index];
        CHECK(allocation.buffer_bytes == expected.buffer_bytes);
        CHECK(allocation.allocation_bytes == expected.allocation_bytes);
        CHECK(allocation.memory_type == expected.memory_type);
        CHECK(allocation.heap == expected.heap);
    }
}

}  // namespace

VQM_TEST(TestMetricsWorstCaseWorkerBudget) {
    MetricsMemoryRequirements requirements;
    requirements.spatial.emplace();
    requirements.spatial->direct.supported = true;
    requirements.spatial->direct.arena = {.allocation_bytes = 100, .heap = 0};
    requirements.spatial->staged.supported = true;
    requirements.spatial->staged.arena = {.allocation_bytes = 80, .heap = 0};
    requirements.spatial->staged.staging = {.allocation_bytes = 50, .heap = 1};

    requirements.vmaf_v0.emplace();
    requirements.vmaf_v0->direct.supported = true;
    requirements.vmaf_v0->direct.arena = {.allocation_bytes = 60, .heap = 0};
    requirements.vmaf_v0->staged.supported = true;
    requirements.vmaf_v0->staged.arena = {.allocation_bytes = 70, .heap = 0};
    requirements.vmaf_v0->staged.staging = {.allocation_bytes = 10, .heap = 1};

    requirements.vmaf_v1.emplace();
    requirements.vmaf_v1->direct.supported = true;
    requirements.vmaf_v1->direct.arena = {.allocation_bytes = 40, .heap = 1};
    requirements.vmaf_v1->staged.supported = true;
    requirements.vmaf_v1->staged.arena = {.allocation_bytes = 20, .heap = 0};
    requirements.vmaf_v1->staged.staging = {.allocation_bytes = 80, .heap = 1};

    // Per-worker worst-case totals: heap 0 = 100+70+20 = 190 bytes;
    // heap 1 = 50+10+80 = 140 bytes. Each component contributes its maximum
    // per heap, even if those maxima come from different allocation paths.
    constexpr std::array<std::uint64_t, 2> budgets{760, 420};
    CHECK(MaxWorkersForBudgetWorstCase(requirements, budgets) == 3);
    constexpr std::array<std::uint64_t, 2> roomier{760, 560};
    CHECK(MaxWorkersForBudgetWorstCase(requirements, roomier) == 4);
    CHECK_THROWS_AS(
        MaxWorkersForBudgetWorstCase(
            requirements, std::span<const std::uint64_t>(budgets).first(1)
        ),
        std::invalid_argument
    );

    requirements.vmaf_v0.reset();
    requirements.vmaf_v1.reset();
    CHECK(MaxWorkersForBudgetWorstCase(requirements, roomier) == 7);
    requirements.spatial->staged.supported = false;
    CHECK(MaxWorkersForBudgetWorstCase(requirements, roomier) == 7);
    requirements.spatial->direct.supported = false;
    CHECK_THROWS_AS(
        MaxWorkersForBudgetWorstCase(requirements, roomier),
        std::invalid_argument
    );
    requirements.spatial.reset();
    CHECK_THROWS_AS(
        MaxWorkersForBudgetWorstCase(requirements, roomier),
        std::invalid_argument
    );
}

VQM_TEST(TestMetricsAllFrameAlignedScores) {
    if (!vqm::test::GpuAvailable()) return;
    const std::array<Pair, 3> pairs{MakePair(false), MakePair(true),
                                    MakePair(false)};
    const auto config = AllConfig(pairs[0]);
    Metrics metrics(config);
    const auto memory = metrics.GetMemoryUsage();
    const auto requirements = GetMemoryRequirements(config);
    CheckMemoryRequirements(memory.scratch, requirements.scratch);
    CheckMemoryRequirements(memory.spatial, requirements.spatial);
    CheckMemoryRequirements(memory.vmaf_v0, requirements.vmaf_v0);
    CheckMemoryRequirements(memory.vmaf_v1, requirements.vmaf_v1);
    Ssimulacra2Pipeline ssimu(*config.ssimulacra2);
    ButteraugliPipeline butter(*config.butteraugli);
    VmafPipeline v0(*config.vmaf_v0);
    VmafV1Pipeline v1(*config.vmaf_v1);

    std::array<double, 3> expected_ssimu{};
    std::array<double, 3> expected_butter{};
    std::array<double, 3> expected_v0{};
    std::array<double, 3> expected_v1{};
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const auto& p = pairs[i];
        expected_ssimu[i] = ssimu.Compute(p.original, p.distorted).Score();
        expected_butter[i] = butter.Compute(p.original, p.distorted).score;
        if (const auto frame = v0.Push(p.original, p.distorted)) {
            CHECK(frame->index < pairs.size());
            if (frame->index < pairs.size())
                expected_v0[frame->index] = frame->score;
        }
        if (const auto frame = v1.Push(p.original, p.distorted)) {
            CHECK(frame->index < pairs.size());
            if (frame->index < pairs.size())
                expected_v1[frame->index] = frame->score;
        }
    }
    if (const auto frame = v0.Flush()) {
        CHECK(frame->index < pairs.size());
        if (frame->index < pairs.size())
            expected_v0[frame->index] = frame->score;
    }
    if (const auto frame = v1.Flush()) {
        CHECK(frame->index < pairs.size());
        if (frame->index < pairs.size())
            expected_v1[frame->index] = frame->score;
    }

    const auto check = [&](const MetricsResult& result, std::size_t index) {
        CHECK(result.frame_index == index);
        CHECK(result.ssimulacra2.has_value());
        CHECK(result.butteraugli.has_value());
        CHECK(result.vmaf_v0.has_value());
        CHECK(result.vmaf_v1.has_value());
        if (result.ssimulacra2)
            CHECK_NEAR(*result.ssimulacra2, expected_ssimu[index], 1e-9);
        if (result.butteraugli)
            CHECK_NEAR(*result.butteraugli, expected_butter[index], 1e-9);
        if (result.vmaf_v0)
            CHECK_NEAR(*result.vmaf_v0, expected_v0[index], 1e-9);
        if (result.vmaf_v1)
            CHECK_NEAR(*result.vmaf_v1, expected_v1[index], 1e-9);
    };

    CHECK(!metrics.Compute(pairs[0].original, pairs[0].distorted));
    // Mix owning and borrowed host input modes within the same sequence.
    const auto second =
        metrics.Compute(View(pairs[1].original), View(pairs[1].distorted));
    CHECK(second.has_value());
    if (second) check(*second, 0);
    const auto third = metrics.Compute(pairs[2].original, pairs[2].distorted);
    CHECK(third.has_value());
    if (third) check(*third, 1);
    const auto last = metrics.Flush();
    CHECK(last.has_value());
    if (last) check(*last, 2);
    CHECK(!metrics.Flush());
    CHECK_THROWS_AS(
        metrics.Compute(pairs[0].original, pairs[0].distorted), std::logic_error
    );

    metrics.Reset();
    CHECK(!metrics.Compute(pairs[0].original, pairs[0].distorted));
    const auto still = metrics.Flush();
    CHECK(still.has_value());
    if (still) {
        CHECK(still->frame_index == 0);
        CHECK(still->ssimulacra2.has_value());
        CHECK(still->butteraugli.has_value());
        CHECK(still->vmaf_v0.has_value());
        CHECK(still->vmaf_v1.has_value());
        if (still->ssimulacra2)
            CHECK_NEAR(*still->ssimulacra2, expected_ssimu[0], 1e-9);
        if (still->butteraugli)
            CHECK_NEAR(*still->butteraugli, expected_butter[0], 1e-9);
        if (still->vmaf_v0) CHECK_NEAR(*still->vmaf_v0, expected_v0[0], 1e-9);
    }
}

VQM_TEST(TestMetricsHfrTailStaysFrameAligned) {
    if (!vqm::test::GpuAvailable()) return;
    const auto pair = MakePair(false);
    auto config = AllConfig(pair);
    config.vmaf_v1->hfr = true;
    Metrics metrics(config);
    const auto submit = [&] {
        return metrics.Compute(pair.original, pair.distorted);
    };
    const auto check = [](const std::optional<MetricsResult>& frame,
                          std::uint64_t index) {
        CHECK(frame.has_value());
        if (!frame) return;
        CHECK(frame->frame_index == index);
        CHECK(frame->ssimulacra2.has_value());
        CHECK(frame->vmaf_v0.has_value());
        CHECK(frame->vmaf_v1.has_value());
    };
    CHECK(!submit());
    CHECK(!submit());
    check(submit(), 0);
    const auto finalized = metrics.Finalize();
    CHECK(finalized.frames == 3);
    check(finalized.penultimate_frame, 1);
    check(finalized.last_frame, 2);
    CHECK(!metrics.Finalize().last_frame);

    metrics.Reset();
    CHECK(!submit());
    CHECK(!submit());
    check(metrics.Flush(), 0);
    const auto remaining = metrics.Finalize();
    CHECK(!remaining.penultimate_frame);
    check(remaining.last_frame, 1);
    CHECK(!metrics.Flush());
}

VQM_TEST(TestMetricsSelectionAndMemoryUsage) {
    if (!vqm::test::GpuAvailable()) return;
    const auto pair = MakePair(false);
    MetricsConfig config;
    config.size = pair.original.size;
    config.original_format = pair.original.format;
    config.distorted_format = pair.distorted.format;
    CHECK_THROWS_AS(Metrics(config), std::invalid_argument);

    config.ssimulacra2 = Ssimulacra2Config{.size = config.size};
    Metrics one(config);
    Ssimulacra2Config single{.size = config.size};
    single.gpu_resident_images = 0;
    Ssimulacra2Pipeline standalone(single);
    const auto memory = one.GetMemoryUsage();
    const auto requirement = GetMemoryRequirements(config);
    CheckMemoryRequirements(memory.scratch, requirement.scratch);
    CheckMemoryRequirements(memory.spatial, requirement.spatial);
    CheckMemoryRequirements(memory.vmaf_v0, requirement.vmaf_v0);
    CheckMemoryRequirements(memory.vmaf_v1, requirement.vmaf_v1);
    CHECK(memory.spatial.has_value());
    CHECK(!memory.vmaf_v0);
    CHECK(!memory.vmaf_v1);
    if (memory.spatial) {
        CHECK(
            memory.scratch->arena.allocation_bytes ==
            standalone.GetMemoryUsage().arena.allocation_bytes
        );
    }
    const auto single_result = one.Compute(pair.original, pair.distorted);
    CHECK(single_result.has_value());
    if (single_result) {
        CHECK(single_result->frame_index == 0);
        CHECK(single_result->ssimulacra2.has_value());
        CHECK(!single_result->butteraugli);
        CHECK(!single_result->vmaf_v0);
        CHECK(!single_result->vmaf_v1);
        if (single_result->ssimulacra2)
            CHECK_NEAR(
                *single_result->ssimulacra2,
                standalone.Compute(pair.original, pair.distorted).Score(), 1e-9
            );
    }
    CHECK(!one.Flush());

    config.butteraugli = ButteraugliConfig{.size = config.size};
    Metrics shared(config);
    SharedMetricConfig shared_config{.size = config.size};
    SharedMetricWorkspace expected(shared_config);
    const auto shared_memory = shared.GetMemoryUsage();
    const auto shared_requirement = GetMemoryRequirements(config);
    CheckMemoryRequirements(shared_memory.scratch, shared_requirement.scratch);
    CheckMemoryRequirements(shared_memory.spatial, shared_requirement.spatial);
    CheckMemoryRequirements(shared_memory.vmaf_v0, shared_requirement.vmaf_v0);
    CheckMemoryRequirements(shared_memory.vmaf_v1, shared_requirement.vmaf_v1);
    CHECK(shared_memory.spatial.has_value());
    if (shared_memory.spatial)
        CHECK(
            shared_memory.scratch->arena.allocation_bytes ==
            expected.GetMemoryUsage().arena.allocation_bytes
        );
}

VQM_TEST(TestMetricsPackedInputsAndComputeImage) {
    if (!vqm::test::GpuAvailable()) return;
    const auto pair = MakePair(false);
    Metrics metrics(AllConfig(pair));
    const auto original = std::as_bytes(std::span(pair.original.data));
    const auto distorted = std::as_bytes(std::span(pair.distorted.data));
    CHECK_THROWS_AS(
        metrics.Compute(original.first(original.size() - 1), distorted),
        std::invalid_argument
    );
    CHECK_THROWS_AS(
        metrics.Compute(original, distorted.first(distorted.size() - 1)),
        std::invalid_argument
    );
    const auto first = metrics.ComputeImage(original, distorted);
    const auto second = metrics.ComputeImage(pair.original, pair.distorted);
    CHECK(first.frame_index == 0);
    CHECK(second.frame_index == 0);
    CHECK(first.ssimulacra2.has_value());
    CHECK(first.butteraugli.has_value());
    CHECK(first.vmaf_v0.has_value());
    CHECK(first.vmaf_v1.has_value());
    if (first.ssimulacra2 && second.ssimulacra2)
        CHECK_NEAR(*first.ssimulacra2, *second.ssimulacra2, 1e-9);
    if (first.butteraugli && second.butteraugli)
        CHECK_NEAR(*first.butteraugli, *second.butteraugli, 1e-9);
    if (first.vmaf_v0 && second.vmaf_v0)
        CHECK_NEAR(*first.vmaf_v0, *second.vmaf_v0, 1e-9);
    if (first.vmaf_v1 && second.vmaf_v1)
        CHECK_NEAR(*first.vmaf_v1, *second.vmaf_v1, 1e-9);

    CHECK(!metrics.Compute(pair.original, pair.distorted));
    CHECK_THROWS_AS(metrics.ComputeImage(original, distorted), std::logic_error);
    const auto result = metrics.Flush();
    CHECK(result.has_value());
    if (result) CHECK(result->frame_index == 0);
}

VQM_TEST(TestMetricsDistinctChromaFilters) {
    if (!vqm::test::GpuAvailable()) return;
    const auto rgb = MakePair(false);
    const auto original = vqm::test::sdr::ReferenceYuv(rgb.original, true);
    const auto distorted = vqm::test::sdr::ReferenceYuv(rgb.distorted, true);
    MetricsConfig config;
    config.size = original.size;
    config.original_format = original.format;
    config.distorted_format = distorted.format;
    config.ssimulacra2 = Ssimulacra2Config{.size = config.size};
    config.ssimulacra2->chroma_filter = ChromaFilter::Lanczos;
    config.butteraugli = ButteraugliConfig{.size = config.size};
    config.butteraugli->chroma_filter = ChromaFilter::Bicubic;

    Ssimulacra2Pipeline ssimu(*config.ssimulacra2);
    ButteraugliPipeline butter(*config.butteraugli);
    const auto expected_ssimu = ssimu.Compute(original, distorted).Score();
    const auto expected_butter = butter.Compute(original, distorted).score;
    Metrics metrics(config);
    const auto result = metrics.Compute(original, distorted);
    CHECK(result.has_value());
    if (result) {
        CHECK(result->ssimulacra2.has_value());
        CHECK(result->butteraugli.has_value());
        if (result->ssimulacra2)
            CHECK_NEAR(*result->ssimulacra2, expected_ssimu, 1e-9);
        if (result->butteraugli)
            CHECK_NEAR(*result->butteraugli, expected_butter, 1e-9);
    }
}

VQM_TEST(TestMetricsSharedAlphaBackground) {
    if (!vqm::test::GpuAvailable()) return;
    const Pair pair{
        vqm::test::sdr::MakeSource(
            PixelType::Byte, ColorFamily::Rgb, AlphaType::Straight, false
        ),
        vqm::test::sdr::MakeSource(
            PixelType::Byte, ColorFamily::Rgb, AlphaType::Straight, true
        )};
    auto config = AllConfig(pair);
    config.background = 0.2f;
    config.vmaf_v0->background = 0.8f;  // top-level background wins
    config.vmaf_v1->background = 0.8f;

    Ssimulacra2Pipeline ssimu(*config.ssimulacra2);
    ButteraugliPipeline butter(*config.butteraugli);
    VmafConfig v0_config = *config.vmaf_v0;
    VmafV1Config v1_config = *config.vmaf_v1;
    v0_config.background = config.background;
    v1_config.background = config.background;
    VmafPipeline v0(v0_config);
    VmafV1Pipeline v1(v1_config);
    const auto expected_ssimu =
        ssimu.Compute(pair.original, pair.distorted, config.background).Score();
    const auto expected_butter =
        butter.Compute(pair.original, pair.distorted, config.background).score;
    const auto expected_v0 = v0.Push(pair.original, pair.distorted);
    CHECK(!v1.Push(pair.original, pair.distorted));
    const auto expected_v1 = v1.Flush();
    CHECK(expected_v0.has_value());
    CHECK(expected_v1.has_value());
    Metrics metrics(config);
    const auto result = metrics.ComputeImage(pair.original, pair.distorted);
    CHECK(result.ssimulacra2.has_value());
    CHECK(result.butteraugli.has_value());
    CHECK(result.vmaf_v0.has_value());
    CHECK(result.vmaf_v1.has_value());
    if (result.ssimulacra2)
        CHECK_NEAR(*result.ssimulacra2, expected_ssimu, 1e-9);
    if (result.butteraugli)
        CHECK_NEAR(*result.butteraugli, expected_butter, 1e-9);
    if (result.vmaf_v0 && expected_v0)
        CHECK_NEAR(*result.vmaf_v0, expected_v0->score, 1e-9);
    if (result.vmaf_v1 && expected_v1)
        CHECK_NEAR(*result.vmaf_v1, expected_v1->score, 1e-9);
}

VQM_TEST(TestMetricsFormatValidationDoesNotConsumeFrame) {
    if (!vqm::test::GpuAvailable()) return;
    const auto pair = MakePair(false);
    auto config = AllConfig(pair);
    Metrics metrics(config);
    auto invalid = pair.distorted;
    invalid.format.transfer = TransferCharacteristics::Bt709;
    CHECK_THROWS_AS(
        metrics.Compute(pair.original, invalid), std::invalid_argument
    );
    invalid = pair.distorted;
    invalid.size.width -= 1;
    CHECK_THROWS_AS(
        metrics.Compute(pair.original, invalid), std::invalid_argument
    );
    CHECK(!metrics.Compute(pair.original, pair.distorted));
    const auto result = metrics.Flush();
    CHECK(result.has_value());
    if (result) CHECK(result->frame_index == 0);
}

VQM_TEST(TestMetricsTemporalSelections) {
    if (!vqm::test::GpuAvailable()) return;
    const auto pair = MakePair(false);
    auto config = AllConfig(pair);
    config.ssimulacra2.reset();
    config.butteraugli.reset();
    config.vmaf_v1.reset();
    Metrics only_v0(config);
    const auto v0_usage = only_v0.GetMemoryUsage();
    const auto v0_plan = GetMemoryRequirements(config);
    CHECK(!v0_usage.spatial);
    CHECK(!v0_usage.vmaf_v1);
    CheckMemoryRequirements(v0_usage.vmaf_v0, v0_plan.vmaf_v0);
    const auto first = only_v0.Compute(pair.original, pair.distorted);
    CHECK(first.has_value());
    if (first) {
        CHECK(first->frame_index == 0);
        CHECK(first->vmaf_v0.has_value());
        CHECK(!first->ssimulacra2);
        CHECK(!first->butteraugli);
        CHECK(!first->vmaf_v1);
    }
    CHECK(!only_v0.Flush());

    config = AllConfig(pair);
    config.ssimulacra2.reset();
    config.butteraugli.reset();
    config.vmaf_v0.reset();
    Metrics only_v1(config);
    const auto v1_usage = only_v1.GetMemoryUsage();
    const auto v1_plan = GetMemoryRequirements(config);
    CHECK(!v1_usage.spatial);
    CHECK(!v1_usage.vmaf_v0);
    CheckMemoryRequirements(v1_usage.vmaf_v1, v1_plan.vmaf_v1);
    CHECK(!only_v1.Compute(pair.original, pair.distorted));
    const auto final = only_v1.Flush();
    CHECK(final.has_value());
    if (final) {
        CHECK(final->frame_index == 0);
        CHECK(final->vmaf_v1.has_value());
        CHECK(!final->ssimulacra2);
        CHECK(!final->butteraugli);
        CHECK(!final->vmaf_v0);
    }
    CHECK(!only_v1.Flush());
    only_v1.Reset();
    CHECK(!only_v1.Flush());  // empty sequence
    CHECK_THROWS_AS(
        only_v1.Compute(pair.original, pair.distorted), std::logic_error
    );
    only_v1.Reset();
    const auto still = only_v1.ComputeImage(pair.original, pair.distorted);
    CHECK(still.frame_index == 0);
    CHECK(still.vmaf_v1.has_value());
}
