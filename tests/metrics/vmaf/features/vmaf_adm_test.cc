// VMAF ADM: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "support.h"
#include "metrics/vmaf/features/vmaf_adm.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

namespace {

using vqm::Size;
using vqm::internal::VmafAdmWavelet;
using vqm::internal::VmafAdmWaveletBuffers;
using vqm::internal::VulkanRuntime;

// Independent direct 2-D convolution: the GPU uses two separable passes
// inside each invocation. Libvmaf's 4-tap Daubechies-2 coefficients and
// edge rule come from adm_tools.c:43-44,1017-1153 at the pinned revision.
std::array<std::vector<float>, 4> ReferenceWavelet(
    const std::vector<float>& source, Size size
) {
    constexpr std::array<float, 4> kLow{0.482962913144690f, 0.836516303737469f,
                                        0.224143868041857f, -0.129409522550921f};
    constexpr std::array<float, 4> kHigh{-0.129409522550921f,
                                         -0.224143868041857f, 0.836516303737469f,
                                         -0.482962913144690f};
    const auto reflect = [](int pos, int extent) {
        if (pos < 0) return -pos;
        if (pos >= extent) return 2 * extent - pos - 1;
        return pos;
    };
    const Size output = VmafAdmWavelet::OutputSize(size);
    std::array<std::vector<float>, 4> result;
    for (auto& plane : result) plane.resize(output.width * output.height);
    for (std::size_t y = 0; y < output.height; ++y) {
        for (std::size_t x = 0; x < output.width; ++x) {
            std::array<double, 4> sums{};
            for (int fy = 0; fy < 4; ++fy) {
                const auto sy = static_cast<std::size_t>(reflect(
                    2 * static_cast<int>(y) - 1 + fy,
                    static_cast<int>(size.height)
                ));
                for (int fx = 0; fx < 4; ++fx) {
                    const auto sx = static_cast<std::size_t>(reflect(
                        2 * static_cast<int>(x) - 1 + fx,
                        static_cast<int>(size.width)
                    ));
                    const double sample = source[sy * size.width + sx];
                    const auto ix = static_cast<std::size_t>(fx);
                    const auto iy = static_cast<std::size_t>(fy);
                    sums[0] += sample * kLow[iy] * kLow[ix];
                    sums[1] += sample * kHigh[iy] * kLow[ix];
                    sums[2] += sample * kLow[iy] * kHigh[ix];
                    sums[3] += sample * kHigh[iy] * kHigh[ix];
                }
            }
            const std::size_t index = y * output.width + x;
            for (std::size_t channel = 0; channel < result.size(); ++channel)
                result[channel][index] = static_cast<float>(sums[channel]);
        }
    }
    return result;
}

class WaveletDevice {
public:
    explicit WaveletDevice(Size size)
        : size_(size),
          pixels_(size.width * size.height),
          output_(VmafAdmWavelet::OutputSize(size)),
          bands_(output_.width * output_.height),
          runtime_(vqm::internal::AcquireVulkanContext(), Config(size)),
          buffers_{0, static_cast<std::uint32_t>(pixels_),
                   static_cast<std::uint32_t>(pixels_ + bands_),
                   static_cast<std::uint32_t>(pixels_ + 2 * bands_),
                   static_cast<std::uint32_t>(pixels_ + 3 * bands_)} {
        const std::size_t total = pixels_ + 4 * bands_;
        runtime_.CreateArena(vqm::internal::Bytes(total));
        runtime_.CreateStaging(vqm::internal::Bytes(total));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, vqm::internal::Bytes(total));
        wavelet_.emplace(runtime_);
    }

    std::array<std::vector<float>, 4> Transform(const std::vector<float>& luma) {
        if (luma.size() != pixels_) throw std::invalid_argument("test geometry");
        std::copy(luma.begin(), luma.end(), runtime_.staging_map());
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{VulkanRuntime::CopyRange{0, 0, pixels_}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        wavelet_->Record(size_, buffers_);
        runtime_.RecordReadback(buffers_.low, pixels_, 4 * bands_);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        std::array<std::vector<float>, 4> result;
        for (std::size_t c = 0; c < result.size(); ++c) {
            const float* begin = runtime_.staging_map() + pixels_ + c * bands_;
            result[c].assign(begin, begin + bands_);
        }
        return result;
    }

private:
    static VulkanRuntime::Config Config(Size size) {
        VulkanRuntime::Config config;
        config.base_size = size;
        config.base_pixels = size.width * size.height;
        config.push_constant_size = VmafAdmWavelet::PushBytes();
        config.streaming = false;
        return config;
    }

    Size size_;
    std::size_t pixels_;
    Size output_;
    std::size_t bands_;
    VulkanRuntime runtime_;
    VmafAdmWaveletBuffers buffers_;
    std::optional<VmafAdmWavelet> wavelet_;
};

}  // namespace

VQM_TEST(TestVmafAdmWaveletGeometry) {
    CHECK(VmafAdmWavelet::OutputSize({2, 2}) == (Size{1, 1}));
    CHECK(VmafAdmWavelet::OutputSize({17, 19}) == (Size{9, 10}));
    CHECK_THROWS_AS(VmafAdmWavelet::OutputSize({1, 8}), std::invalid_argument);
    CHECK_THROWS_AS(
        VmafAdmWavelet::OutputSize({std::numeric_limits<std::size_t>::max(), 3}),
        std::invalid_argument
    );
}

VQM_TEST(TestVmafAdmWaveletVulkan) {
    if (!vqm::test::GpuAvailable()) return;
    for (Size size : {Size{2, 2}, Size{17, 19}, Size{64, 48}, Size{33, 17}}) {
        std::vector<float> source(size.width * size.height);
        for (std::size_t y = 0; y < size.height; ++y) {
            for (std::size_t x = 0; x < size.width; ++x) {
                source[y * size.width + x] =
                    static_cast<float>((x * 13 + y * 7) % 93) - 50.0f;
            }
        }
        source[0] = 100.0f;
        WaveletDevice device(size);
        const auto expected = ReferenceWavelet(source, size);
        const auto actual = device.Transform(source);
        for (std::size_t c = 0; c < actual.size(); ++c) {
            for (std::size_t i = 0; i < actual[c].size(); ++i) {
                CHECK_NEAR(actual[c][i], expected[c][i], 1e-4);
            }
        }
        if (size == Size{17, 19}) {
            // Pinned libvmaf adm_dwt2_s output, including all four edges and
            // one interior coefficient, in low/H/V/D order.
            constexpr std::array<std::size_t, 5> indices{0, 8, 81, 89, 49};
            constexpr std::array<std::array<double, 4>, 5> oracle{{
                {17.6434231, -30.6871815, -32.8833351, 7.53607035},
                {-58.0621758, -2.5621798, 11.2583294, -0.000000298023224},
                {-29.2583294, 6.06217766, -4.7583313, 0.0},
                {0.00000011920929, 6.06217766, 11.2583303, 0.000000238418579},
                {55.0344429, 34.8750038, -11.6250019, -20.1350937},
            }};
            for (std::size_t n = 0; n < indices.size(); ++n) {
                for (std::size_t c = 0; c < actual.size(); ++c)
                    CHECK_NEAR(actual[c][indices[n]], oracle[n][c], 1e-4);
            }
        }
    }
}

namespace {
using vqm::internal::VmafAdm;
using vqm::internal::VmafAdmBuffers;

class AdmDevice {
public:
    explicit AdmDevice(Size size)
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
        buffers_.partials = take(6 * VmafAdm::PartialCount(size));
        buffers_.results = take(9);
        runtime_.CreateArena(vqm::internal::Bytes(next));
        runtime_.CreateStaging(vqm::internal::Bytes(next));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, vqm::internal::Bytes(next));
        adm_.emplace(runtime_);
    }

    std::array<float, 9> Compute(
        const std::vector<float>& reference, const std::vector<float>& distorted,
        bool inspect_partials = false
    ) {
        std::copy(reference.begin(), reference.end(), runtime_.staging_map());
        std::copy(
            distorted.begin(), distorted.end(), runtime_.staging_map() + pixels_
        );
        const auto partial_words = 6 * VmafAdm::PartialCount(size_);
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
        runtime_.RecordReadback(buffers_.results, 0, 9);
        if (inspect_partials)
            runtime_.RecordReadback(buffers_.partials, 9, partial_words);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        std::array<float, 9> result{};
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
        // Subsequent scales reuse a prefix of the first scale's partials.
        // Inspect zero slots only beyond that prefix; every slot must be
        // finite after replacing the uploaded NaN poison on every call.
        const auto reused =
            ((next.width + 15u) / 16u) * ((next.height + 15u) / 16u);
        for (std::size_t i = 0; i < 6 * columns * rows; ++i)
            CHECK(std::isfinite(partials[i]));
        const auto left = (band.width - 5u) / 10u;
        const auto top = (band.height - 5u) / 10u;
        const auto check_zero_group = [partials](std::size_t group) {
            for (std::size_t term = 0; term < 6; ++term) {
                const float value = partials[6 * group + term];
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
    std::optional<VmafAdm> adm_;
};
}  // namespace

VQM_TEST(TestVmafAdmGeometry) {
    CHECK_THROWS_AS(VmafAdm::BandPixels({16, 20}), std::invalid_argument);
    CHECK(VmafAdm::BandPixels({17, 19}) == 90);
    CHECK(VmafAdm::PartialCount({64, 48}) == 4);
}

VQM_TEST(TestVmafAdmOracleVulkan) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{64, 48};
    AdmDevice device(size);
    // Exact outputs from the pinned libvmaf adm-debug oracle. The first
    // four rows use 8-bit moving pairs; the next four retain low 10-bit data.
    constexpr std::array<std::array<double, 9>, 8> oracle{{
        {24.63272476, 24.80878258, 39.48160172, 39.90427017, 65.30679321,
         65.25995636, 75.11885071, 75.25345612, 0.9966549401},
        {24.51716042, 24.69342232, 40.55188751, 41.15293503, 76.67151642,
         76.93891144, 71.74388885, 71.74284363, 0.9951350935},
        {24.44251823, 24.59205627, 40.83214188, 41.29332352, 79.84722137,
         79.44378662, 69.28631592, 69.29192352, 0.9990080540},
        {24.20008087, 24.34774590, 39.78666687, 40.31094742, 62.81310272,
         63.31930161, 112.49855042, 112.48556519, 0.9951545289},
        {24.80954170, 24.80878258, 39.89665222, 39.90427017, 65.26040649,
         65.25995636, 75.25270081, 75.25345612, 0.9999650922},
        {24.69419098, 24.69342232, 41.14656448, 41.15293503, 76.93776703,
         76.93891144, 71.74408722, 71.74284363, 0.9999743497},
        {24.58981705, 24.59205627, 41.28977585, 41.29332352, 79.44464111,
         79.44378662, 69.29196930, 69.29192352, 0.9999772314},
        {24.34479523, 24.34774590, 40.30983734, 40.31094742, 63.31845093,
         63.31930161, 112.48558807, 112.48556519, 0.9999796704},
    }};
    for (std::size_t test = 0; test < oracle.size(); ++test) {
        const int frame = static_cast<int>(test % 4);
        const bool lowbits = test >= 4;
        std::vector<float> reference(size.width * size.height),
            distorted(reference.size());
        for (std::size_t y = 0; y < size.height; ++y)
            for (std::size_t x = 0; x < size.width; ++x) {
                const int xx = static_cast<int>(x), yy = static_cast<int>(y);
                const int base = 16 + (7 * xx + 13 * yy + 17 * frame +
                                       29 * ((xx / 7 + frame) % 2)) %
                                          220;
                const float delta =
                    lowbits
                        ? float(1 + (xx + yy + frame) % 3) / 4.0f
                        : float(
                              std::min(235, base + (xx + yy + frame) % 9) - base
                          );
                reference[y * size.width + x] = float(base - 128);
                distorted[y * size.width + x] = float(base - 128) + delta;
            }
        const auto actual = device.Compute(reference, distorted);
        for (std::size_t i = 0; i < 8; ++i)
            CHECK_NEAR(actual[i], oracle[test][i], 5e-4);
        CHECK_NEAR(actual[8], oracle[test][8], 2e-6);
    }
}

VQM_TEST(TestVmafAdmIdentityVulkan) {
    if (!vqm::test::GpuAvailable()) return;
    for (Size size : {Size{17, 19}, Size{64, 48}, Size{95, 73}}) {
        AdmDevice device(size);
        std::vector<float> image(size.width * size.height, -128.0f);
        for (int pattern = 0; pattern < 2; ++pattern) {
            if (pattern == 1) {
                for (std::size_t i = 0; i < image.size(); ++i)
                    image[i] = static_cast<float>((i * 47) % 256) - 128.0f;
            }
            const auto result = device.Compute(image, image);
            CHECK_NEAR(result[8], 1.0, 2e-6);
        }
    }
}

VQM_TEST(TestVmafAdmOddGeometryAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{33, 19};
    AdmDevice device(size);
    // Pinned libvmaf float ADM debug extractor on planar 8-bit YUV444.
    // The flat case also checks the norm's per-scale noise floor.
    constexpr std::array<std::array<double, 9>, 3> oracle{{
        {5.02074718, 5.02074718, 3.36105347, 3.36105347, 2.33042431, 2.33042431,
         1.71707129, 1.71707129, 1.0},
        {5.02074718, 5.02074862, 3.36105347, 3.36105680, 2.33042431, 2.33043003,
         1.71707129, 1.71708012, 0.9999984463},
        {7.77936459, 15.40177917, 5.75630379, 24.96350670, 9.17595291,
         52.63217926, 20.58029938, 93.75516510, 0.2318142487},
    }};
    for (std::size_t test = 0; test < oracle.size(); ++test) {
        std::vector<float> reference(size.width * size.height),
            distorted(reference.size());
        for (std::size_t y = 0; y < size.height; ++y) {
            for (std::size_t x = 0; x < size.width; ++x) {
                const auto i = y * size.width + x;
                if (test == 0)
                    reference[i] = distorted[i] = 0.0f;
                else if (test == 1) {
                    reference[i] = -128.0f;
                    distorted[i] = (x / 3 + y / 3) % 2 ? 127.0f : -128.0f;
                } else {
                    reference[i] =
                        static_cast<float>((7 * x + 13 * y + 16) % 256) - 128.0f;
                    distorted[i] =
                        static_cast<float>((11 * x + 5 * y + 31) % 256) - 128.0f;
                }
            }
        }
        const auto actual = device.Compute(reference, distorted);
        for (std::size_t i = 0; i < 8; ++i)
            CHECK_NEAR(actual[i], oracle[test][i], 5e-4);
        CHECK_NEAR(actual[8], oracle[test][8], 2e-6);
    }
}

VQM_TEST(TestVmafAdmExcludedGroupsOverwritePartials) {
    if (!vqm::test::GpuAvailable()) return;
    for (Size size : {Size{513, 385}, Size{512, 256}}) {
        AdmDevice device(size);
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
            if (identity) CHECK_NEAR(result[8], 1.0, 2e-6);
        }
    }
}
