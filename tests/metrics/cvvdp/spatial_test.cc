// ColorVideoVDP: Copyright (c) 2024, Graphics and Displays group.
// MIT; see licenses/COLORVIDEOVDP-LICENSE.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "support.h"
#include "metrics/cvvdp/cvvdp_spatial.h"
#include "vulkan/vulkan_runtime.h"

namespace {
using namespace vqm;
using namespace vqm::internal;

constexpr float kPixelsPerDegree = 60.0f;

struct ProbeLayout {
    std::uint32_t reference = 0;
    std::uint32_t distorted;
    std::uint32_t scratch;
    std::uint32_t tables;
    std::uint32_t partials;
    std::uint32_t results;
    std::uint32_t words;

    explicit ProbeLayout(Size size) {
        const auto pixels = static_cast<std::uint32_t>(size.width * size.height);
        distorted = 4 * pixels;
        scratch = 8 * pixels;
        tables = scratch + static_cast<std::uint32_t>(
                               CvvdpSpatial::ScratchWords(size, kPixelsPerDegree)
                           );
        partials = tables + static_cast<std::uint32_t>(
                                CvvdpSpatial::TableWords(size, kPixelsPerDegree)
                            );
        results =
            partials +
            static_cast<std::uint32_t>(4 * CvvdpSpatial::PartialCount(size));
        words = results + static_cast<std::uint32_t>(
                              4 * CvvdpSpatial::Levels(size, kPixelsPerDegree)
                          );
    }
};

class SpatialProbe {
public:
    explicit SpatialProbe(Size size)
        : size_(size),
          layout_(size),
          runtime_(AcquireVulkanContext(), Config(size)) {
        runtime_.CreateArena(Bytes(layout_.words));
        if (!runtime_.arena_visible())
            runtime_.CreateStaging(Bytes(2 * layout_.words));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(layout_.words));
        spatial_.emplace(runtime_);
    }

    std::vector<float> Compute(std::uint32_t channels, bool identical = false) {
        std::vector<float> data(layout_.words);
        FillInputs(data, channels, identical);
        CvvdpSpatial::FillTables(
            size_, kPixelsPerDegree,
            std::span(data).subspan(
                layout_.tables, layout_.partials - layout_.tables
            )
        );
        float* upload = runtime_.arena_visible() ? runtime_.arena_map()
                                                 : runtime_.staging_map();
        std::copy(data.begin(), data.end(), upload);
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        runtime_.BindDescriptorSet();
        if (runtime_.arena_visible()) {
            runtime_.HostToComputeBarrier();
        } else {
            const std::array copies{VulkanRuntime::CopyRange{0, 0, data.size()}};
            runtime_.RecordStagedCopies(copies);
        }
        spatial_->Record(
            size_, kPixelsPerDegree, channels, layout_.reference,
            layout_.distorted, layout_.scratch, layout_.tables, layout_.partials,
            layout_.results
        );
        const std::size_t count =
            channels * CvvdpSpatial::Levels(size_, kPixelsPerDegree);
        if (runtime_.arena_visible()) {
            runtime_.ComputeToHostBarrier();
        } else {
            runtime_.RecordReadback(layout_.results, layout_.words, count);
        }
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        const float* values = runtime_.arena_visible()
                                  ? runtime_.arena_map() + layout_.results
                                  : runtime_.staging_map() + layout_.words;
        return {values, values + count};
    }

private:
    static VulkanRuntime::Config Config(Size size) {
        VulkanRuntime::Config config{};
        config.base_size = size;
        config.base_pixels = size.width * size.height;
        config.streaming = false;
        config.push_constant_size = CvvdpSpatial::PushBytes();
        return config;
    }

    // These inputs are independently evaluated by the pinned original Python
    // process_block_of_frames; the checked-in features below are its output.
    void FillInputs(
        std::span<float> data, std::uint32_t channels, bool identical
    ) const {
        const std::size_t area = size_.width * size_.height;
        for (std::size_t y = 0; y < size_.height; ++y) {
            for (std::size_t x = 0; x < size_.width; ++x) {
                const std::array<float, 4> reference{
                    10.0f + static_cast<float>(x % 7) +
                        2.0f * static_cast<float>(y % 5),
                    0.3f + static_cast<float>((x + 3 * y) % 11) / 50.0f,
                    0.2f + static_cast<float>((3 * x + y) % 13) / 60.0f,
                    static_cast<float>(
                        static_cast<int>(x % 3) - static_cast<int>(y % 3)
                    ) / 10.0f};
                const std::array<float, 4> distorted{
                    reference[0] + static_cast<float>(
                                       static_cast<int>((x + 2 * y) % 5) - 2
                                   ) / 4.0f,
                    reference[1] + ((x + y) % 2 != 0 ? 0.02f : -0.01f),
                    reference[2] * 0.93f,
                    reference[3] + 0.1f * static_cast<float>(x % 2)};
                for (std::size_t channel = 0; channel < channels; ++channel) {
                    const std::size_t index =
                        channel * area + y * size_.width + x;
                    data[layout_.reference + index] = reference[channel];
                    data[layout_.distorted + index] =
                        identical ? reference[channel] : distorted[channel];
                }
            }
        }
    }

    Size size_;
    ProbeLayout layout_;
    VulkanRuntime runtime_;
    std::optional<CvvdpSpatial> spatial_;
};

// Generated independently by tests/make_cvvdp_oracles.py, not this shader.
#include "cvvdp_spatial_reference.inc"

}  // namespace

VQM_TEST(TestCvvdpSpatialOriginalFeatures) {
    if (!vqm::test::GpuAvailable()) return;
    for (const auto& fixture : kCvvdpSpatialReferences) {
        SpatialProbe probe({fixture.width, fixture.height});
        const auto actual = probe.Compute(fixture.channels);
        CHECK(actual.size() == fixture.features.size());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            CHECK_NEAR(
                actual[index], fixture.features[index],
                std::max(5e-6, std::abs(fixture.features[index]) * 5e-5)
            );
        }
    }
}

VQM_TEST(TestCvvdpSpatialIdentityAndReuse) {
    if (!vqm::test::GpuAvailable()) return;
    SpatialProbe probe({33, 24});
    for (const auto channels : {3u, 4u, 3u}) {
        const auto actual = probe.Compute(channels, true);
        CHECK(std::all_of(actual.begin(), actual.end(), [](float value) {
            return value == 0.0f;
        }));
    }
}

VQM_TEST(TestCvvdpSpatialPlanning) {
    CHECK(CvvdpSpatial::Levels({2, 2}, 60.0f) == 1);
    CHECK(CvvdpSpatial::Levels({32, 24}, 60.0f) == 4);
    CHECK(CvvdpSpatial::Levels({1920, 1080}, 60.0f) == 9);
    CHECK_THROWS_AS(CvvdpSpatial::Levels({0, 32}, 60), std::invalid_argument);
    CHECK_THROWS_AS(CvvdpSpatial::Levels({32, 32}, 0), std::invalid_argument);
    CHECK_THROWS_AS(CvvdpSpatial::Levels({1, 32}, 60), std::invalid_argument);
}
