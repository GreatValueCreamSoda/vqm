// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>
#include "support.h"
#include "metrics/vmaf/speed/vmaf_speed.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

namespace {
using namespace vqm::internal;
class SpeedDevice {
public:
    explicit SpeedDevice(bool allow_wide = true)
        : runtime_(AcquireVulkanContext(), Config()) {
        constexpr std::uint32_t p = 320 * 320, o = 20 * 20;
        std::uint32_t offset = 0;
        ref_ = offset;
        offset += p;
        dist_ = offset;
        offset += p;
        f_.scratch = offset;
        offset += static_cast<std::uint32_t>(
            VmafSpeedFilter::ScratchWords({320, 320})
        );
        f_.down = offset;
        offset += o;
        s_.reference = offset;
        offset += o;
        s_.distorted = offset;
        offset += o;
        s_.scratch = offset;
        offset += static_cast<std::uint32_t>(VmafSpeedStats::ScratchWords());
        s_.partials = offset;
        offset +=
            static_cast<std::uint32_t>(VmafSpeedStats::PartialCount({20, 20}));
        s_.result = offset;
        offset += 2;
        runtime_.CreateArena(Bytes(offset));
        runtime_.CreateStaging(Bytes(2 * p + 2));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(offset));
        filter_.emplace(runtime_);
        stats_.emplace(runtime_, allow_wide);
    }
    std::array<float, 2> Compute(
        std::uint32_t plane, std::uint32_t frame = 0, bool identity = false,
        bool ref_flat = false, bool dist_flat = false, bool low_bits = false
    ) {
        constexpr std::uint32_t p = 320 * 320;
        for (std::uint32_t y = 0; y < 320; y++)
            for (std::uint32_t x = 0; x < 320; x++) {
                const auto val =
                    16 + ((x * (7 + frame * 2) + y * 13 + (x / 7) * 23 +
                           (y / 11) * 31 + plane * 19 + frame * 33) %
                          220);
                const auto dis =
                    (val + ((x * 13 + y * (3 + frame) + plane * 17) %
                            (41 + frame * 7))) %
                    256;
                const float ref_fraction =
                    low_bits ? float((x + y + plane) % 4) * 0.25f : 0.0f;
                const float dist_fraction =
                    low_bits ? float((x * 3 + y + frame) % 4) * 0.25f : 0.0f;
                runtime_.staging_map()[y * 320 + x] =
                    ref_flat ? 0.0f : float(val) + ref_fraction - 128.0f;
                runtime_.staging_map()[p + y * 320 + x] =
                    dist_flat
                        ? 0.0f
                        : float(identity ? val : dis) + dist_fraction - 128.0f;
            }
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{VulkanRuntime::CopyRange{0, ref_, p},
                                VulkanRuntime::CopyRange{p, dist_, p}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        f_.input = ref_;
        f_.result = s_.reference;
        filter_->Record({320, 320}, f_);
        f_.input = dist_;
        f_.result = s_.distorted;
        filter_->Record({320, 320}, f_);
        stats_->Record({20, 20}, s_);
        runtime_.RecordReadback(s_.result, 2 * p, 2);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        return {runtime_.staging_map()[2 * p],
                runtime_.staging_map()[2 * p + 1]};
    }

private:
    static VulkanRuntime::Config Config() {
        VulkanRuntime::Config c;
        c.base_size = {320, 320};
        c.base_pixels = 320 * 320;
        c.push_constant_size = VmafSpeedFilter::PushBytes();
        c.streaming = false;
        return c;
    }
    VulkanRuntime runtime_;
    std::optional<VmafSpeedFilter> filter_;
    std::optional<VmafSpeedStats> stats_;
    VmafSpeedFilterBuffers f_{};
    VmafSpeedStatsBuffers s_{};
    std::uint32_t ref_ = 0, dist_ = 0;
};
}  // namespace

VQM_TEST(TestVmafSpeedStatsGeometry) {
    CHECK(VmafSpeedStats::PartialCount({20, 20}) == 1);
    CHECK_THROWS_AS(VmafSpeedStats::PartialCount({4, 5}), std::invalid_argument);
    CHECK_THROWS_AS(VmafSpeedStats::PartialCount({5, 4}), std::invalid_argument);
    CHECK_THROWS_AS(
        VmafSpeedStats::PartialCount(
            {std::numeric_limits<std::size_t>::max(), 5}
        ),
        std::invalid_argument
    );
}

VQM_TEST(TestVmafSpeedStatsBatchedChannels) {
    if (!vqm::test::GpuAvailable()) return;
    // Different planes in odd geometry exercise bank addressing and cropped
    // blocks independently of the filter. Compare batched work against two
    // separate statistics dispatch sequences, including singular V imputation.
    constexpr vqm::Size size{37, 29};
    constexpr std::uint32_t pixels = size.width * size.height;
    const auto groups =
        static_cast<std::uint32_t>(VmafSpeedStats::PartialCount(size));
    constexpr auto scratch =
        static_cast<std::uint32_t>(VmafSpeedStats::ScratchWords());
    const VmafSpeedStatsBuffers batch{0, 2 * pixels, 4 * pixels,
                                      4 * pixels + 2 * scratch,
                                      4 * pixels + 2 * scratch + 2 * groups};
    const auto separate = batch.result + 4;
    VulkanRuntime::Config config;
    config.base_size = size;
    config.base_pixels = pixels;
    config.push_constant_size = VmafSpeedStats::PushBytes();
    config.streaming = false;
    VulkanRuntime runtime(AcquireVulkanContext(), config);
    runtime.CreateArena(Bytes(separate + 4));
    runtime.CreateStaging(Bytes(4 * pixels + 8));
    runtime.CreateCommandObjects();
    runtime.CreateDescriptors(0, Bytes(separate + 4));
    VmafSpeedStats stats(runtime);
    for (bool singular_v : {false, true}) {
        for (std::uint32_t plane = 0; plane < 4; ++plane) {
            for (std::uint32_t i = 0; i < pixels; ++i) {
                auto hash = i * 0x9e3779b9u + plane * 0x85ebca6bu;
                hash ^= hash >> 16;
                hash *= 0x7feb352du;
                hash ^= hash >> 15;
                runtime.staging_map()[plane * pixels + i] =
                    singular_v && plane == 1
                        ? 0.0f
                        : static_cast<float>(hash % 1024) / 64.0f;
            }
        }
        runtime.ResetCommandPool();
        runtime.BeginCommandBuffer();
        const std::array copies{VulkanRuntime::CopyRange{0, 0, 4 * pixels}};
        runtime.RecordStagedCopies(copies);
        runtime.BindDescriptorSet();
        CHECK_THROWS_AS(stats.Record(size, batch, 0), std::invalid_argument);
        CHECK_THROWS_AS(stats.Record(size, batch, 3), std::invalid_argument);
        auto overflow = batch;
        overflow.reference = std::numeric_limits<std::uint32_t>::max() - pixels;
        CHECK_THROWS_AS(stats.Record(size, overflow, 2), std::invalid_argument);
        stats.Record(size, batch, 2);
        for (std::uint32_t channel = 0; channel < 2; ++channel) {
            stats.Record(
                size, {batch.reference + channel * pixels,
                       batch.distorted + channel * pixels,
                       batch.scratch + channel * scratch,
                       batch.partials + channel * groups, separate + 2 * channel}
            );
        }
        runtime.RecordReadback(batch.result, 4 * pixels, 8);
        runtime.EndCommandBuffer();
        runtime.SubmitAndWait();
        const auto* result = runtime.staging_map() + 4 * pixels;
        for (std::size_t i = 0; i < 4; ++i) CHECK(result[i] == result[i + 4]);
        CHECK(result[1] == 0.0f);
        CHECK(result[3] == (singular_v ? 1.0f : 0.0f));
    }
}
VQM_TEST(TestVmafSpeedStatsAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    // Pinned v1 feature options and full feature-extractor outputs, generated
    // independently with tests/metrics/vmaf/speed/vmaf_speed_stats_oracle.cc
    // (320x320 YUV444).
    SpeedDevice d;
    constexpr std::array<std::array<double, 2>, 4> expected{{
        {10.98688793182373, 12.925295829772949},
        {14.350398063659668, 13.212528228759766},
        {23.101078033447266, 25.090614318847656},
        {18.231754302978516, 16.174461364746094},
    }};
    constexpr std::array<std::array<double, 2>, 4> expected_10bit{{
        {10.986575126647949, 12.924886703491211},
        {14.350759506225586, 13.21332836151123},
        {23.101585388183594, 25.090232849121094},
        {18.231607437133789, 16.174606323242188},
    }};
    for (std::uint32_t frame = 0; frame < expected.size(); ++frame) {
        for (std::uint32_t channel = 1; channel <= 2; ++channel) {
            const auto result = d.Compute(channel, frame);
            CHECK_NEAR(result[0], expected[frame][channel - 1], 0.03);
            CHECK(result[1] == 0.0f);
            const auto high =
                d.Compute(channel, frame, false, false, false, true);
            CHECK_NEAR(high[0], expected_10bit[frame][channel - 1], 0.03);
            CHECK(high[1] == 0.0f);
        }
    }
    const auto identity = d.Compute(1, 0, true);
    CHECK_NEAR(identity[0], 0.0, 1e-4);
    CHECK(identity[1] == 0.0f);
    const auto flat = d.Compute(1, 0, true, true, true);
    CHECK_NEAR(flat[0], 0.0, 1e-4);
    CHECK(flat[1] == 1.0f);
    const auto flat_ref = d.Compute(1, 0, false, true, false);
    CHECK_NEAR(flat_ref[0], 0.0, 1e-4);
    CHECK(flat_ref[1] == 1.0f);
    const auto flat_dist = d.Compute(1, 0, false, false, true);
    CHECK_NEAR(flat_dist[0], 0.0, 1e-4);
    CHECK(flat_dist[1] == 1.0f);
}

namespace {
class EigenDevice {
public:
    struct Result {
        std::array<float, 625> factor;
        std::array<float, 25> values;
        float status;
    };
    explicit EigenDevice(bool allow_wide, bool force_legacy_qr = false)
        : runtime_(AcquireVulkanContext(), Config()) {
        runtime_.CreateArena(Bytes(kWords));
        runtime_.CreateStaging(Bytes(2 * kWords));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(kWords));
        const auto& limits = runtime_.context().limits;
        const bool wide = allow_wide &&
                          limits.maxComputeWorkGroupInvocations >= 512 &&
                          limits.maxComputeWorkGroupSize[0] >= 512;
        pipeline_ = runtime_.CreatePipeline(
            force_legacy_qr ? "vmaf_speed_stats_eigen_legacy"
            : wide          ? "vmaf_speed_stats_eigen_wide"
                            : "vmaf_speed_stats_eigen"
        );
    }
    ~EigenDevice() {
        vkDestroyPipeline(runtime_.context().device, pipeline_, nullptr);
    }
    Result Compute(const std::array<float, 625>& matrix) {
        std::fill_n(runtime_.staging_map(), kWords, 0.0f);
        for (std::uint32_t ch = 0; ch < 2; ++ch)
            std::copy(
                matrix.begin(), matrix.end(),
                runtime_.staging_map() + 50 + ch * 625
            );
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{VulkanRuntime::CopyRange{0, 0, kWords}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        const std::array<std::uint32_t, 9> pc{};
        runtime_.Dispatch(pipeline_, pc.data(), sizeof(pc), 2, 2, 1);
        runtime_.ComputeBarrier();
        runtime_.RecordReadback(0, kWords, kWords);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        Result result;
        const auto* out = runtime_.staging_map() + kWords;
        std::copy_n(out + 1300, result.factor.size(), result.factor.begin());
        std::copy_n(out + 2550, result.values.size(), result.values.begin());
        result.status = out[2600] == 0.0f ? out[2602] : out[2600];
        // Both independent channel workgroups see the same matrix.
        for (std::size_t i = 0; i < result.values.size(); ++i)
            CHECK(
                result.values[i] == out[2575 + i] ||
                (std::isnan(result.values[i]) && std::isnan(out[2575 + i]))
            );
        CHECK(result.status == (out[2601] == 0.0f ? out[2603] : out[2601]));
        return result;
    }

private:
    static constexpr std::uint32_t kWords = 2604;
    static VulkanRuntime::Config Config() {
        VulkanRuntime::Config c;
        c.base_size = {25, 25};
        c.base_pixels = 625;
        c.push_constant_size = VmafSpeedStats::PushBytes();
        c.streaming = false;
        return c;
    }
    VulkanRuntime runtime_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

bool IsCpuVulkanDevice() {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(
        AcquireVulkanContext()->physical_device, &properties
    );
    return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
}

// For A = Q diag(spectrum) Q' and its rounded input K, row-sum norms bound
// both input rounding and E = LL' - K. Weyl's bound gives lambda_min(K) >=
// lambda_min(A) - ||K-A||. If delta = ||E|| / lambda_min(K) < 1, the relative
// error of rhs' (LL')^-1 rhs is at most delta / (1-delta). This checks the
// variance against the measured factorization residual rather than assuming
// the same forward error for well- and ill-conditioned matrices.
double VariancePerturbationBound(
    const std::array<double, 625>& basis, const std::array<double, 25>& spectrum,
    const std::array<float, 625>& matrix, const std::array<float, 625>& factor
) {
    double factor_residual = 0.0, input_rounding = 0.0;
    for (std::size_t i = 0; i < 25; ++i) {
        double factor_row = 0.0, rounding_row = 0.0;
        for (std::size_t j = 0; j < 25; ++j) {
            double reconstructed = 0.0, unrounded = 0.0;
            for (std::size_t k = 0; k < 25; ++k) {
                reconstructed += double(factor[i * 25 + k]) * factor[j * 25 + k];
                unrounded += basis[i * 25 + k] * spectrum[k] * basis[j * 25 + k];
            }
            factor_row += std::fabs(reconstructed - matrix[i * 25 + j]);
            rounding_row += std::fabs(unrounded - matrix[i * 25 + j]);
        }
        factor_residual = std::max(factor_residual, factor_row);
        input_rounding = std::max(input_rounding, rounding_row);
    }
    const double minimum = *std::min_element(spectrum.begin(), spectrum.end());
    const double eigenvalue_lower_bound = minimum - input_rounding;
    CHECK(eigenvalue_lower_bound > 0.0);
    if (eigenvalue_lower_bound <= 0.0) return 0.0;
    const double delta = factor_residual / eigenvalue_lower_bound;
    CHECK(delta < 1.0);
    if (delta >= 1.0) return 0.0;
    return delta / (1.0 - delta);
}
}  // namespace

VQM_TEST(TestVmafSpeedCovarianceFactorization) {
    if (!vqm::test::GpuAvailable()) return;
    const bool cpu_device = IsCpuVulkanDevice();
    for (bool allow_wide : {false, true}) {
        EigenDevice device(allow_wide);
        // Construct dense covariance matrices with known spectra, including a
        // repeated spectrum, a small regular eigenvalue, and rank deficiency.
        // Check the eigenvalues independently, then reconstruct K = L L' and
        // verify its variance solve against a known solution on the rounded K.
        for (std::uint32_t variant = 0; variant < 4; ++variant) {
            std::array<double, 625> basis{};
            std::array<double, 25> spectrum{};
            const std::size_t first = variant == 3 ? 1 : 0;
            const double n = double(25 - first);
            for (std::size_t k = 0; k < 25; ++k) {
                spectrum[k] = variant == 0 ? 1.0 : 0.25 + double(k);
                if (k < first) continue;
                for (std::size_t i = first; i < 25; ++i)
                    basis[i * 25 + k] =
                        std::sqrt((k == first ? 1.0 : 2.0) / n) *
                        std::cos(
                            std::acos(-1.0) * (double(i - first) + 0.5) *
                            double(k - first) / n
                        );
            }
            if (variant == 2) spectrum[0] = 1e-4;
            if (variant == 3) spectrum[0] = 0.0;
            std::array<float, 625> matrix{};
            for (std::size_t i = 0; i < 25; ++i)
                for (std::size_t j = i; j < 25; ++j) {
                    double value = 0.0;
                    for (std::size_t k = 0; k < 25; ++k)
                        value +=
                            basis[i * 25 + k] * spectrum[k] * basis[j * 25 + k];
                    matrix[i * 25 + j] = matrix[j * 25 + i] =
                        static_cast<float>(value);
                }
            const auto result = device.Compute(matrix);
            CHECK(result.status == (variant == 3 ? 1.0f : 0.0f));
            auto sorted = result.values;
            std::sort(sorted.begin(), sorted.end());
            for (std::size_t k = 0; k < 25; ++k)
                CHECK_NEAR(sorted[k], spectrum[k], 4e-5);
            if (variant != 3) {
                for (std::size_t i = 0; i < 25; ++i)
                    for (std::size_t j = 0; j < 25; ++j) {
                        double reconstructed = 0.0;
                        for (std::size_t k = 0; k <= std::min(i, j); ++k)
                            reconstructed += double(result.factor[i * 25 + k]) *
                                             result.factor[j * 25 + k];
                        CHECK_NEAR(reconstructed, matrix[i * 25 + j], 1e-4);
                        if (j > i) CHECK(result.factor[i * 25 + j] == 0.0f);
                    }
                // Choose a solution aligned with the smallest known mode.
                // Forming RHS with the rounded matrix avoids attributing its
                // input quantization error to the GPU factorization.
                std::array<double, 25> exact{}, rhs{}, forward{};
                for (std::size_t i = 0; i < 25; ++i)
                    exact[i] = basis[i * 25] / spectrum[0];
                for (std::size_t i = 0; i < 25; ++i)
                    for (std::size_t j = 0; j < 25; ++j)
                        rhs[i] += double(matrix[i * 25 + j]) * exact[j];
                double expected_variance = 0.0, actual_variance = 0.0;
                for (std::size_t i = 0; i < 25; ++i) {
                    expected_variance += rhs[i] * exact[i];
                    double value = rhs[i];
                    for (std::size_t j = 0; j < i; ++j)
                        value -= double(result.factor[i * 25 + j]) * forward[j];
                    forward[i] = value / result.factor[i * 25 + i];
                    actual_variance += forward[i] * forward[i];
                }
                if (variant == 2) {
                    const double bound = VariancePerturbationBound(
                        basis, spectrum, matrix, result.factor
                    );
                    CHECK_NEAR(
                        actual_variance, expected_variance,
                        std::fabs(expected_variance) * bound
                    );
                }
                // Keep the established hardware-GPU pin. CPU Vulkan's
                // contraction differs: its ill-conditioned case is judged by
                // the residual bound above, with unchanged eigenvalue and
                // per-entry factor reconstruction checks on every device.
                if (!cpu_device || variant != 2) {
                    CHECK_NEAR(
                        actual_variance, expected_variance,
                        std::fabs(expected_variance) *
                            (variant == 2 ? 2e-3 : 1e-5)
                    );
                }
            }
        }
        std::array<float, 625> invalid{};
        invalid[0] = std::numeric_limits<float>::quiet_NaN();
        CHECK(device.Compute(invalid).status == 2.0f);
    }
}

VQM_TEST(TestVmafSpeedAdversarialSpectra) {
    if (!vqm::test::GpuAvailable()) return;
    for (bool allow_wide : {false, true}) {
        EigenDevice device(allow_wide);
        for (unsigned variant = 0; variant < 7; ++variant) {
            std::array<float, 625> matrix{};
            std::array<double, 25> expected{};
            for (unsigned i = 0; i < 25; ++i) {
                float value;
                if (variant <= 2)
                    value = static_cast<float>(std::pow(10.0, -6.0 + 0.5 * i));
                else if (variant == 3)
                    value = 100.0f + static_cast<float>(i % 5) * 1e-4f;
                else if (variant == 4)
                    value = static_cast<float>(1e-25 * (i + 1));
                else if (variant == 5)
                    value = i == 0 ? -2.0f : float(i + 1);
                else
                    value = i < 2 ? 4e-6f : float(i + 1);
                if (i == 0 && variant == 0) value = std::nextafter(1e-6f, 0.0f);
                if (i == 0 && variant == 2) value = std::nextafter(1e-6f, 1.0f);
                matrix[i * 25 + i] = value;
                expected[i] = value;
            }
            if (variant == 6) {
                matrix[1] = matrix[25] = 1e-6f;
                // The two eigenvalues of [a b; b a] are exactly a +/- b.
                expected[0] = double(matrix[0]) - matrix[1];
                expected[1] = double(matrix[0]) + matrix[1];
            }
            std::sort(expected.begin(), expected.end());
            const auto result = device.Compute(matrix);
            auto actual = result.values;
            std::sort(actual.begin(), actual.end());
            const float expected_status =
                variant == 0 || variant == 4 || variant == 5 ? 1.0f : 0.0f;
            CHECK(result.status == expected_status);
            for (unsigned i = 0; i < 25; ++i) {
                // Independent exact diagonal/2x2 spectra: bound relative
                // error even when the overall dynamic range spans 12 orders.
                const double tolerance =
                    std::max(1e-34, std::fabs(expected[i]) * 5e-6);
                CHECK_NEAR(actual[i], expected[i], tolerance);
            }
        }
        // Dense orthogonal transforms with known spectra exercise the
        // reduction as well as the tridiagonal solver. Input is rounded to
        // FP32 exactly once, just like the existing spectral oracle.
        for (unsigned variant = 0; variant < 3; ++variant) {
            std::array<double, 625> basis{};
            std::array<double, 25> expected{};
            for (unsigned k = 0; k < 25; ++k) {
                expected[k] = variant == 2 ? 1.0 + (k % 5) * 1e-4 : 0.25 + k;
                if (k == 0 && variant < 2)
                    expected[k] = variant == 0 ? 1e-4 : 1e-5;
                for (unsigned i = 0; i < 25; ++i)
                    basis[i * 25 + k] = std::sqrt((k == 0 ? 1.0 : 2.0) / 25.0) *
                                        std::cos(
                                            std::acos(-1.0) * (double(i) + 0.5) *
                                            double(k) / 25.0
                                        );
            }
            std::array<float, 625> matrix{};
            for (unsigned i = 0; i < 25; ++i)
                for (unsigned j = i; j < 25; ++j) {
                    double value = 0.0;
                    for (unsigned k = 0; k < 25; ++k)
                        value +=
                            basis[i * 25 + k] * expected[k] * basis[j * 25 + k];
                    matrix[i * 25 + j] = matrix[j * 25 + i] =
                        static_cast<float>(value);
                }
            std::sort(expected.begin(), expected.end());
            const auto result = device.Compute(matrix);
            auto actual = result.values;
            std::sort(actual.begin(), actual.end());
            CHECK(result.status == 0.0f);
            for (unsigned i = 0; i < 25; ++i)
                CHECK_NEAR(actual[i], expected[i], 4e-5);
        }
        // Finite inputs can overflow a Householder intermediate. This is an
        // explicit invalid matrix, not a falsely regular one.
        std::array<float, 625> overflow{};
        overflow[0] = overflow[26] = overflow[52] = 1e30f;
        overflow[2] = overflow[50] = 1e30f;
        CHECK(device.Compute(overflow).status == 2.0f);
    }
}

VQM_TEST(TestVmafSpeedStatsEigenFallback) {
    if (!vqm::test::GpuAvailable()) return;
    SpeedDevice wide;
    SpeedDevice portable(false);
    for (std::uint32_t plane = 1; plane <= 2; ++plane) {
        const auto expected = wide.Compute(plane, 1);
        const auto actual = portable.Compute(plane, 1);
        CHECK_NEAR(actual[0], expected[0], 1e-5);
        CHECK(actual[1] == expected[1]);
    }
    const auto flat = portable.Compute(1, 0, true, true, true);
    CHECK(flat[0] == 0.0f);
    CHECK(flat[1] == 1.0f);
}

VQM_TEST(TestVmafSpeedGuardPreservesLegacyClassification) {
    if (!vqm::test::GpuAvailable()) return;
    // These exact tridiagonal FP32 matrices exposed status changes when the
    // original QR solver was replaced by unguarded Sturm bisection. Status is
    // the legacy QR result, not a claim that it is the exact mathematical
    // classification near 1e-6. The guard should preserve that behavior.
    // Cases 502, 534 and 9541 come from the seeded 10,000-matrix stress run.
    // The last reports factorization failure on the measured NVIDIA device;
    // CPU Vulkan instead classifies its spectrum as singular. A forced-QR
    // shader checks the old behavior on the same device without assuming
    // identical floating-point contraction across Vulkan implementations.
    constexpr std::array<std::array<float, 49>, 3> inputs{{
        std::array<float, 49>{
            0x1.b93ab20000000p-1f,  0x1.937a3a0000000p-1f,
            0x1.9fadc00000000p+0f,  0x1.ecd3080000000p-1f,
            0x1.4b96ce0000000p+0f,  0x1.2c9d7e0000000p+0f,
            0x1.72a8640000000p-1f,  0x1.14340c0000000p+0f,
            0x1.c488b40000000p-1f,  0x1.b57d860000000p-1f,
            0x1.5092580000000p+0f,  0x1.46c3f80000000p-1f,
            0x1.adca6e0000000p-1f,  0x1.86642e0000000p-1f,
            0x1.c924680000000p-1f,  0x1.9e93c80000000p+0f,
            0x1.5dbbb40000000p-1f,  0x1.bbaafe0000000p-1f,
            0x1.2e18860000000p+0f,  0x1.4569420000000p+0f,
            0x1.4efe6e0000000p+0f,  0x1.d49e3e0000000p-1f,
            0x1.02cc220000000p+0f,  0x1.a25d0e0000000p-1f,
            0x1.88a2c40000000p+0f,  -0x1.9005c00000000p-2f,
            -0x1.c935400000000p-1f, -0x1.c0e05e0000000p-2f,
            0x1.268f280000000p-1f,  0x1.2f71580000000p-1f,
            -0x1.111fcc0000000p-2f, 0x1.20ff640000000p-1f,
            -0x1.9742380000000p-3f, 0x1.d3fb0a0000000p-2f,
            0x1.7693280000000p-1f,  -0x1.7f85180000000p-5f,
            0x1.e9ef180000000p-4f,  0x1.f6b4ce0000000p-4f,
            -0x1.2172a40000000p-6f, -0x1.b31d860000000p-1f,
            -0x1.70ac3e0000000p-4f, -0x1.6f84180000000p-4f,
            0x1.7533d60000000p-1f,  0x1.6a157a0000000p-1f,
            -0x1.4cc1a80000000p-1f, 0x1.894e120000000p-2f,
            0x1.50c0da0000000p-2f,  -0x1.e607240000000p-4f,
            0x1.988cd80000000p-1f,
        },
        std::array<float, 49>{
            0x1.ca38d40000000p-1f,  0x1.cfd0700000000p+1f,
            0x1.1d4fe40000000p-2f,  0x1.27a6020000000p-1f,
            0x1.f7cc1c0000000p+1f,  0x1.c87da40000000p-1f,
            0x1.8036000000000p-2f,  0x1.8029440000000p-2f,
            0x1.057bd00000000p+1f,  0x1.23bb560000000p+1f,
            0x1.a8910a0000000p-1f,  0x1.f777c60000000p-2f,
            0x1.9f97620000000p-1f,  0x1.ce78c60000000p-3f,
            0x1.6a1e480000000p+0f,  0x1.06d1c20000000p+1f,
            0x1.c489a00000000p-2f,  0x1.82185c0000000p-1f,
            0x1.9cff1e0000000p-1f,  0x1.d0c5a00000000p-3f,
            0x1.2723fc0000000p-2f,  0x1.0a9f7c0000000p-3f,
            0x1.20538e0000000p-1f,  0x1.27b3e40000000p-2f,
            0x1.c30d000000000p+0f,  -0x1.31550e0000000p-2f,
            -0x1.c925ba0000000p-3f, 0x1.d8e2ea0000000p-5f,
            -0x1.d7f0c60000000p-3f, -0x1.97a7960000000p-5f,
            -0x1.30d2f20000000p-4f, 0x1.a4e1620000000p-6f,
            0x1.4b4e7e0000000p-3f,  0x1.b2d1c20000000p-2f,
            -0x1.6ae8dc0000000p-1f, 0x1.17a73e0000000p-2f,
            0x1.8564460000000p-4f,  -0x1.16b4180000000p-2f,
            0x1.8d542a0000000p-2f,  0x1.67dcd80000000p-1f,
            -0x1.bb96700000000p-3f, 0x1.9ad3ea0000000p-2f,
            0x1.9ce7200000000p-3f,  -0x1.6187b00000000p-3f,
            -0x1.8fed780000000p-3f, -0x1.572a2a0000000p-6f,
            0x1.6d798e0000000p-3f,  -0x1.a7ed020000000p-9f,
            -0x1.b550380000000p-2f,
        },
        std::array<float, 49>{
            0x1.f0f01a0000000p+8f,  0x1.2075f80000000p+7f,
            0x1.c538560000000p+6f,  0x1.6b75e20000000p+2f,
            0x1.e4b3880000000p+2f,  0x1.ffab460000000p+0f,
            0x1.f4c1d60000000p+2f,  0x1.a802b60000000p+2f,
            0x1.f7a5380000000p+0f,  0x1.d910be0000000p-2f,
            0x1.b9f4680000000p+8f,  0x1.101d060000000p+4f,
            0x1.ae31ee0000000p+1f,  0x1.4a9e800000000p+9f,
            0x1.1d24a40000000p+0f,  0x1.3a6de80000000p+5f,
            0x1.5c00ba0000000p+4f,  0x1.8a68ec0000000p+9f,
            0x1.41bdee0000000p+3f,  0x1.6e3fae0000000p+9f,
            0x1.5abf040000000p+12f, 0x1.9434d00000000p+7f,
            0x1.1a81c20000000p+2f,  0x1.35e7180000000p+2f,
            0x1.12731e0000000p+6f,  0x1.17ccc60000000p+6f,
            0x1.362bc80000000p+4f,  -0x1.90d2f80000000p+4f,
            -0x1.70e9ec0000000p-6f, 0x1.e503040000000p-2f,
            0x1.30bc780000000p-1f,  0x1.f0d90c0000000p+0f,
            0x1.baba380000000p+1f,  0x1.8b31440000000p-5f,
            -0x1.6c994e0000000p+0f, -0x1.b20e660000000p+5f,
            0x1.4da91a0000000p+1f,  0x1.9ac7bc0000000p+0f,
            -0x1.5fceec0000000p+4f, 0x1.961f5c0000000p-1f,
            0x1.bd49580000000p+1f,  0x1.81f7a60000000p+3f,
            0x1.5fd7e60000000p+6f,  -0x1.d6d9b00000000p-1f,
            -0x1.2e3fde0000000p+5f, -0x1.ac6a300000000p+7f,
            0x1.d3c8bc0000000p+4f,  0x1.0d41c20000000p-3f,
            0x1.eb9e120000000p+0f,
        },
    }};
    constexpr std::array<float, 3> legacy_status{0.0f, 1.0f, 2.0f};
    EigenDevice legacy(false, true);
    const bool cpu_device = IsCpuVulkanDevice();
    for (bool allow_wide : {false, true}) {
        EigenDevice device(allow_wide);
        for (std::size_t n = 0; n < inputs.size(); ++n) {
            std::array<float, 625> matrix{};
            for (unsigned i = 0; i < 25; ++i) matrix[i * 25 + i] = inputs[n][i];
            for (unsigned i = 0; i < 24; ++i)
                matrix[i * 25 + i + 1] = matrix[(i + 1) * 25 + i] =
                    inputs[n][25 + i];
            const auto actual = device.Compute(matrix);
            CHECK(actual.status == legacy.Compute(matrix).status);
            if (!cpu_device) CHECK(actual.status == legacy_status[n]);
        }
        // A well-conditioned spectrum stays outside the fallback band and
        // exercises independent bisection. The eigenvalues are known exactly.
        std::array<float, 625> regular{};
        for (unsigned i = 0; i < 25; ++i) regular[i * 25 + i] = float(i + 1);
        const auto result = device.Compute(regular);
        CHECK(result.status == 0.0f);
        auto values = result.values;
        std::sort(values.begin(), values.end());
        for (unsigned i = 0; i < 25; ++i)
            CHECK_NEAR(values[i], float(i + 1), 4e-5);
    }
}
