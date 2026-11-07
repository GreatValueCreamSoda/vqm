// VMAF ADM3: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>
#include "support.h"
#include "metrics/vmaf/features/vmaf_adm.h"
#include "vulkan/vulkan_runtime.h"

namespace {
using namespace vqm;
using namespace vqm::internal;
class Adm3Device {
public:
    explicit Adm3Device(Size size)
        : size_(size),
          pixels_(size.width * size.height),
          runtime_(vqm::internal::AcquireVulkanContext(), Config(size)) {
        std::size_t next = 0;
        auto take = [&](std::size_t count) {
            const auto result = static_cast<std::uint32_t>(next);
            next += count;
            return result;
        };
        auto level = size;
        for (std::size_t i = 0; i < 5; ++i) {
            const auto count = level.width * level.height;
            buffers_.reference[i] = take(count);
            buffers_.distorted[i] = take(count);
            if (i < 4) level = VmafAdmWavelet::OutputSize(level);
        }
        const auto bands = 3 * VmafAdm::BandPixels(size);
        buffers_.reference_details = take(bands);
        buffers_.restored = take(bands);
        buffers_.additive = take(bands);
        buffers_.partials = take(9 * VmafAdm::PartialCount(size));
        buffers_.results = take(13);
        runtime_.CreateArena(vqm::internal::Bytes(next));
        runtime_.CreateStaging(vqm::internal::Bytes(next));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, vqm::internal::Bytes(next));
        adm_.emplace(runtime_);
    }

    std::array<float, 13> Compute(
        const std::vector<float>& reference, const std::vector<float>& distorted,
        bool inspect_partials = false
    ) {
        std::copy(reference.begin(), reference.end(), runtime_.staging_map());
        std::copy(
            distorted.begin(), distorted.end(), runtime_.staging_map() + pixels_
        );
        const auto partial_words = 9 * VmafAdm::PartialCount(size_);
        if (inspect_partials)
            std::fill_n(
                runtime_.staging_map() + 2 * pixels_, partial_words,
                std::numeric_limits<float>::quiet_NaN()
            );
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{
            VulkanRuntime::CopyRange{0, buffers_.reference[0], pixels_},
            VulkanRuntime::CopyRange{pixels_, buffers_.distorted[0], pixels_},
            VulkanRuntime::CopyRange{2 * pixels_, buffers_.partials,
                                     partial_words},
        };
        runtime_.RecordStagedCopies(
            std::span(copies.data(), inspect_partials ? 3u : 2u)
        );
        runtime_.BindDescriptorSet();
        adm_->Record(size_, buffers_);
        runtime_.RecordReadback(buffers_.results, 0, 13);
        if (inspect_partials)
            runtime_.RecordReadback(buffers_.partials, 13, partial_words);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        std::array<float, 13> result{};
        std::copy_n(runtime_.staging_map(), result.size(), result.begin());
        if (inspect_partials)
            CheckPartialWrites(runtime_.staging_map() + result.size());
        return result;
    }

private:
    void CheckPartialWrites(const float* partials) const {
        const Size band = VmafAdmWavelet::OutputSize(size_);
        const Size next = VmafAdmWavelet::OutputSize(band);
        const auto columns = (band.width + 15u) / 16u;
        const auto rows = (band.height + 15u) / 16u;
        // The following scales overwrite only this prefix of the first
        // scale's partials. All other slots still expose its border writes.
        const auto reused =
            ((next.width + 15u) / 16u) * ((next.height + 15u) / 16u);
        for (std::size_t i = 0; i < 9 * columns * rows; ++i)
            CHECK(std::isfinite(partials[i]));
        const auto left = (band.width - 5u) / 10u;
        const auto top = (band.height - 5u) / 10u;
        const auto check_zero_group = [partials](std::size_t group) {
            for (std::size_t term = 0; term < 9; ++term) {
                const float value = partials[9 * group + term];
                CHECK(value == 0.0f);
                CHECK(!std::signbit(value));
            }
        };
        std::size_t excluded = 0;
        for (std::size_t group = reused; group < columns * rows; ++group) {
            bool scored_pixel = false;
            for (std::size_t y = (group / columns) * 16u;
                 y < std::min((group / columns + 1u) * 16u, band.height); ++y)
                for (std::size_t x = (group % columns) * 16u;
                     x < std::min((group % columns + 1u) * 16u, band.width); ++x)
                    scored_pixel |= x >= left && x < band.width - left &&
                                    y >= top && y < band.height - top;
            if (scored_pixel) continue;
            ++excluded;
            check_zero_group(group);
        }
        CHECK(excluded > 0);
        // Hand-selected left/right groups beyond the reused prefix must be
        // empty; the larger odd case also exposes a whole bottom-center tile.
        const std::array edge_groups{
            ((reused + columns - 1u) / columns) * columns, columns * rows - 1u};
        CHECK(left >= 16u);
        for (auto group : edge_groups) {
            CHECK(group >= reused);
            CHECK(group < columns * rows);
            check_zero_group(group);
        }
        if (top >= 16u) {
            const auto bottom_center = (rows - 1u) * columns + columns / 2u;
            CHECK(bottom_center >= reused);
            check_zero_group(bottom_center);
        }
    }

    static VulkanRuntime::Config Config(Size size) {
        VulkanRuntime::Config config;
        config.base_size = size;
        config.base_pixels = size.width * size.height;
        config.push_constant_size = VmafAdm::PushBytes();
        config.streaming = false;
        return config;
    }
    Size size_;
    std::size_t pixels_;
    VulkanRuntime runtime_;
    VmafAdmBuffers buffers_{};
    std::optional<VmafAdm3> adm_;
};
}  // namespace

VQM_TEST(TestVmafAdm3AgainstIntegerLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{64, 48};
    Adm3Device device(size);
    // Pinned scalar libvmaf integer adm extractor, v1.0.16 3H model options,
    // 10-bit, deterministic noisy video; CLI JSON rounded to six decimals.
    constexpr std::array<double, 4> expected{0.945501, 0.949053, 0.936053,
                                             0.957450};
    for (std::size_t frame = 0; frame < 4; ++frame) {
        std::vector<float> a(size.width * size.height), b(a.size());
        for (std::size_t y = 0; y < size.height; ++y)
            for (std::size_t x = 0; x < size.width; ++x) {
                const int base = static_cast<int>(
                    16 +
                    (7 * x + 13 * y + 17 * frame + 29 * ((x / 7 + frame) % 2)) %
                        220
                );
                const int noise =
                    static_cast<int>((41 * x + 37 * y + 17 * frame) % 81) - 40;
                a[y * size.width + x] = static_cast<float>(base - 128);
                b[y * size.width + x] =
                    static_cast<float>(std::clamp(base + noise, 0, 255) - 128);
            }
        const auto actual = device.Compute(a, b);
        CHECK_NEAR(actual[12], expected[frame], 0.0002);
    }
}

VQM_TEST(TestVmafAdm3ExcludedGroupsOverwritePartials) {
    if (!vqm::test::GpuAvailable()) return;
    for (Size size : {Size{513, 385}, Size{512, 256}}) {
        Adm3Device device(size);
        std::vector<float> reference(size.width * size.height),
            distorted(reference.size());
        for (std::size_t i = 0; i < reference.size(); ++i)
            reference[i] = static_cast<float>((i * 47u) % 251u) - 128.0f;
        for (bool identity : {true, false, true}) {
            for (std::size_t i = 0; i < distorted.size(); ++i)
                distorted[i] =
                    identity
                        ? reference[i]
                        : static_cast<float>((i * 31u + 19u) % 251u) - 128.0f;
            const auto result = device.Compute(reference, distorted, true);
            for (float value : result) CHECK(std::isfinite(value));
            if (identity) CHECK_NEAR(result[12], 1.0, 2e-6);
        }
    }
}

VQM_TEST(TestVmafAdm3HighBitDepthAndMoving) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{64, 48};
    Adm3Device device(size);
    // Pinned integer ADM3 with v1.0.16 3H model options and the deterministic
    // sequences generated by tests/make_vmaf_fixtures.py. The integer oracle
    // has depth-dependent quantisation, so these bound a difference between
    // implementations rather than equality of intermediate arithmetic.
    constexpr std::array<std::array<double, 4>, 3> expected{{
        {0.992767, 0.993196, 0.993986, 0.993892},
        {0.999859, 0.999864, 0.999871, 0.999886},
        {0.998539, 0.998626, 0.998816, 0.998970},
    }};
    for (std::size_t fixture = 0; fixture < expected.size(); ++fixture) {
        for (std::size_t frame = 0; frame < 4; ++frame) {
            std::vector<float> a(size.width * size.height), b(a.size());
            for (std::size_t y = 0; y < size.height; ++y)
                for (std::size_t x = 0; x < size.width; ++x) {
                    const int base = static_cast<int>(
                        16 + (7 * x + 13 * y + 17 * frame +
                              29 * ((x / 7 + frame) % 2)) %
                                 220
                    );
                    const auto i = y * size.width + x;
                    a[i] = static_cast<float>(base - 128);
                    if (fixture == 0) {
                        const int delta = static_cast<int>((x + y + frame) % 9);
                        b[i] = static_cast<float>(
                            std::min(235, base + delta) - 128
                        );
                    } else {
                        const int shift = fixture == 1 ? 2 : 4;
                        const int delta =
                            1 + static_cast<int>(
                                    (x + y + frame) % ((1u << shift) - 1)
                                );
                        b[i] = a[i] + static_cast<float>(delta) /
                                          static_cast<float>(1u << shift);
                    }
                }
            const auto actual = device.Compute(a, b);
            CHECK_NEAR(actual[12], expected[fixture][frame], 0.0002);
        }
    }
}
