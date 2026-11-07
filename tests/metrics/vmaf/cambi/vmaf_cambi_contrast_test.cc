// VMAF CAMBI: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <optional>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "support.h"
#include "metrics/vmaf/cambi/vmaf_cambi.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

namespace {
#include "vmaf_cambi_contrast_reference.inc"
#include "vmaf_cambi_contrast_reference33.inc"

// Reused by the independent direct-window oracle and storage-variant checks.
void FillHistogramEdgesFixture(
    vqm::Size size, unsigned test_case, float* staging
) {
    const std::size_t pixels = size.width * size.height;
    constexpr std::array<int, 12> levels{65,  66,  67,  68,  69,  100,
                                         101, 561, 562, 563, 564, 565};
    for (std::size_t y = 0; y < size.height; ++y) {
        for (std::size_t x = 0; x < size.width; ++x) {
            const auto at = y * size.width + x;
            staging[2 * pixels + at] = 123.0f;
            if (test_case == 4 || test_case == 5) {
                // A wholly masked map and an unmasked out-of-band map
                // must overwrite every sentinel, including partial tiles.
                staging[at] = test_case == 4 ? 100.0f : 900.0f;
                staging[pixels + at] = test_case == 4 ? 0.0f : 1.0f;
            } else if (test_case == 3) {
                // Dense windows exercise neighboring bins near 33²
                // counts, including simultaneous removals and additions
                // as their shared window moves across narrow stripes.
                staging[at] = x % 31 < 2 ? 100.0f : 99.0f;
                staging[pixels + at] = 1.0f;
            } else if (test_case >= 6) {
                // Every output row is active and adjacent rows slide
                // between varied odd/even bins and band boundaries.
                staging[at] =
                    static_cast<float>(67 + ((x * 19 + y * 37 + x / 5) % 497));
                staging[pixels + at] = 1.0f;
            } else {
                staging[at] =
                    static_cast<float>(levels[(x / 4 + y / 7) % levels.size()]);
                // Entire inactive output rows surround active rows whose
                // windows include masked samples. Straddle band boundaries
                // 67 and 563.
                staging[pixels + at] =
                    (y / 4) % 3 != 0 && x % 11 != 0 ? 1.0f : 0.0f;
            }
        }
    }
}

class ContrastDevice {
public:
    explicit ContrastDevice(
        vqm::Size size, bool allow_half = true, bool enabled_float16 = true,
        bool allow_subgroups = true
    )
        : size_(size),
          pixels_(size.width * size.height),
          runtime_(Context(enabled_float16), Config(size)) {
        buffers_ = {0, static_cast<std::uint32_t>(pixels_),
                    static_cast<std::uint32_t>(2 * pixels_),
                    static_cast<std::uint32_t>(3 * pixels_),
                    static_cast<std::uint32_t>(4 * pixels_)};
        runtime_.CreateArena(vqm::internal::Bytes(5 * pixels_));
        runtime_.CreateStaging(vqm::internal::Bytes(4 * pixels_));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, vqm::internal::Bytes(5 * pixels_));
        contrast_.emplace(runtime_, allow_half, allow_subgroups);
    }
    std::vector<float> Compute(unsigned test_case, unsigned window) {
        FillHistogramEdgesFixture(size_, test_case, runtime_.staging_map());
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{vqm::internal::VulkanRuntime::CopyRange{
                                    0, buffers_.input, 2 * pixels_},
                                vqm::internal::VulkanRuntime::CopyRange{
                                    2 * pixels_, buffers_.contrast, pixels_}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        contrast_->Record(size_, window, buffers_);
        runtime_.RecordReadback(buffers_.filtered, 2 * pixels_, 2 * pixels_);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        const auto* result = runtime_.staging_map() + 2 * pixels_;
        return {result, result + 2 * pixels_};
    }
    bool UsesHalfSource() const { return contrast_->UsesHalfSource(); }
    bool UsesSubgroupPipeline() const {
        return contrast_->UsesSubgroupPipeline();
    }

private:
    static std::shared_ptr<vqm::internal::VulkanContext> Context(bool enabled) {
        auto context = vqm::internal::AcquireVulkanContext();
        if (enabled) return context;
        // A borrowed device may support Float16 physically without promising
        // that the logical device enabled it. Keep its portable module.
        const vqm::VulkanDevice device{context->physical_device,
                                       context->device,
                                       context->queue,
                                       context->queue_family,
                                       context->sixteen_bit_storage,
                                       false};
        return vqm::internal::BorrowVulkanContext(device);
    }
    static vqm::internal::VulkanRuntime::Config Config(vqm::Size size) {
        vqm::internal::VulkanRuntime::Config config;
        config.base_size = size;
        config.base_pixels = size.width * size.height;
        config.push_constant_size =
            vqm::internal::VmafCambiContrast::PushBytes();
        config.streaming = false;
        return config;
    }
    vqm::Size size_;
    std::size_t pixels_;
    vqm::internal::VulkanRuntime runtime_;
    vqm::internal::VmafCambiContrastBuffers buffers_;
    std::optional<vqm::internal::VmafCambiContrast> contrast_;
};

}  // namespace

VQM_TEST(TestVmafCambiContrastGeometry) {
    using vqm::internal::VmafCambiContrast;
    CHECK(VmafCambiContrast::Pixels({960, 540}, 17) == 960 * 540);
    CHECK(VmafCambiContrast::Pixels({1920, 1080}, 33) == 1920 * 1080);
    CHECK_THROWS_AS(
        VmafCambiContrast::Pixels({31, 29}, 4), std::invalid_argument
    );
    CHECK_THROWS_AS(
        VmafCambiContrast::Pixels({31, 29}, 35), std::invalid_argument
    );
    CHECK_THROWS_AS(
        VmafCambiContrast::Pixels({1921, 1080}, 17), std::invalid_argument
    );
}

VQM_TEST(TestVmafCambiContrastAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::internal;
    constexpr vqm::Size size{31, 29};
    constexpr std::size_t pixels = size.width * size.height;
    constexpr std::array<std::uint32_t, 4> bases{76, 177, 304, 558};
    constexpr VmafCambiContrastBuffers buffers{0, pixels, 2 * pixels, 3 * pixels,
                                               4 * pixels};
    VulkanRuntime::Config config;
    config.base_size = size;
    config.base_pixels = pixels;
    config.push_constant_size = VmafCambiContrast::PushBytes();
    config.streaming = false;
    VulkanRuntime runtime(AcquireVulkanContext(), config);
    runtime.CreateArena(Bytes(5 * pixels));
    runtime.CreateStaging(Bytes(4 * pixels));
    runtime.CreateCommandObjects();
    runtime.CreateDescriptors(0, Bytes(5 * pixels));
    VmafCambiContrast contrast(runtime);
    for (std::size_t y = 0; y < size.height; ++y) {
        for (std::size_t x = 0; x < size.width; ++x) {
            auto index = y * size.width + x;
            runtime.staging_map()[index] =
                static_cast<float>(bases[(x / 8) % 4] + ((x / 2 + y / 3) % 5));
            runtime.staging_map()[pixels + index] =
                (x / 7 + y / 6) % 7 != 0 ? 1.0f : 0.0f;
        }
    }
    for (const unsigned window : {11u, 17u}) {
        runtime.ResetCommandPool();
        runtime.BeginCommandBuffer();
        const std::array copies{
            VulkanRuntime::CopyRange{0, buffers.input, 2 * pixels}};
        runtime.RecordStagedCopies(copies);
        runtime.BindDescriptorSet();
        contrast.Record(size, window, buffers);
        runtime.RecordReadback(buffers.filtered, 2 * pixels, pixels);
        runtime.RecordReadback(buffers.contrast, 3 * pixels, pixels);
        runtime.EndCommandBuffer();
        runtime.SubmitAndWait();
        std::uint64_t filtered_hash = 14695981039346656037ULL;
        for (std::size_t i = 0; i < pixels; ++i) {
            const auto filtered = static_cast<std::uint32_t>(
                runtime.staging_map()[2 * pixels + i]
            );
            filtered_hash = (filtered_hash ^ filtered) * 1099511628211ULL;
            const auto expected =
                window == 11 ? kContrastOracle11[i] : kContrastOracle17[i];
            CHECK_NEAR(runtime.staging_map()[3 * pixels + i], expected, 1e-4);
        }
        CHECK(filtered_hash == kFilteredHash);
    }
}

VQM_TEST(TestVmafCambiContrastLargestWindowOddEdges) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::internal;
    constexpr vqm::Size size{65, 49};
    constexpr std::size_t pixels = size.width * size.height;
    constexpr std::array<std::uint32_t, 4> bases{76, 177, 304, 558};
    constexpr VmafCambiContrastBuffers buffers{0, pixels, 2 * pixels, 3 * pixels,
                                               4 * pixels};
    VulkanRuntime::Config config;
    config.base_size = size;
    config.base_pixels = pixels;
    config.push_constant_size = VmafCambiContrast::PushBytes();
    config.streaming = false;
    VulkanRuntime runtime(AcquireVulkanContext(), config);
    runtime.CreateArena(Bytes(5 * pixels));
    runtime.CreateStaging(Bytes(3 * pixels));
    runtime.CreateCommandObjects();
    runtime.CreateDescriptors(0, Bytes(5 * pixels));
    VmafCambiContrast contrast(runtime);
    for (std::size_t y = 0; y < size.height; ++y) {
        for (std::size_t x = 0; x < size.width; ++x) {
            const auto index = y * size.width + x;
            runtime.staging_map()[index] =
                static_cast<float>(bases[(x / 8) % 4] + ((x / 2 + y / 3) % 5));
            runtime.staging_map()[pixels + index] =
                (x / 7 + y / 6) % 7 != 0 ? 1.0f : 0.0f;
        }
    }
    runtime.ResetCommandPool();
    runtime.BeginCommandBuffer();
    const std::array copies{
        VulkanRuntime::CopyRange{0, buffers.input, 2 * pixels}};
    runtime.RecordStagedCopies(copies);
    runtime.BindDescriptorSet();
    contrast.Record(size, 33, buffers);
    runtime.RecordReadback(buffers.contrast, 2 * pixels, pixels);
    runtime.EndCommandBuffer();
    runtime.SubmitAndWait();
    for (std::size_t i = 0; i < pixels; ++i)
        CHECK_NEAR(
            runtime.staging_map()[2 * pixels + i], kContrastOracle33[i], 1e-4
        );
}

VQM_TEST(TestVmafCambiContrastHistogramEdges) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::internal;
    constexpr vqm::Size size{65, 37};
    constexpr std::size_t pixels = size.width * size.height;
    constexpr VmafCambiContrastBuffers buffers{0, pixels, 2 * pixels, 3 * pixels,
                                               4 * pixels};
    VulkanRuntime::Config config;
    config.base_size = size;
    config.base_pixels = pixels;
    config.push_constant_size = VmafCambiContrast::PushBytes();
    config.streaming = false;
    VulkanRuntime runtime(AcquireVulkanContext(), config);
    runtime.CreateArena(Bytes(5 * pixels));
    runtime.CreateStaging(Bytes(4 * pixels));
    runtime.CreateCommandObjects();
    runtime.CreateDescriptors(0, Bytes(5 * pixels));
    VmafCambiContrast contrast(runtime);
    for (unsigned test_case = 0; test_case < 8; ++test_case) {
        const unsigned window = test_case == 0                       ? 1u
                                : (test_case == 1 || test_case == 6) ? 17u
                                                                     : 33u;
        FillHistogramEdgesFixture(size, test_case, runtime.staging_map());
        runtime.ResetCommandPool();
        runtime.BeginCommandBuffer();
        const std::array copies{
            VulkanRuntime::CopyRange{0, buffers.input, 2 * pixels},
            VulkanRuntime::CopyRange{2 * pixels, buffers.contrast, pixels}};
        runtime.RecordStagedCopies(copies);
        runtime.BindDescriptorSet();
        contrast.Record(size, window, buffers);
        runtime.RecordReadback(buffers.filtered, 2 * pixels, 2 * pixels);
        runtime.EndCommandBuffer();
        runtime.SubmitAndWait();
        const auto* filtered = runtime.staging_map() + 2 * pixels;
        const auto* map = runtime.staging_map() + 3 * pixels;
        const int radius = static_cast<int>(window / 2);
        for (int y = 0; y < static_cast<int>(size.height); ++y) {
            for (int x = 0; x < static_cast<int>(size.width); ++x) {
                const auto at = static_cast<std::size_t>(y) * size.width +
                                static_cast<std::size_t>(x);
                const int center = static_cast<int>(filtered[at]);
                float expected = 0.0f;
                if (runtime.staging_map()[pixels + at] != 0.0f && center >= 67 &&
                    center <= 563) {
                    // Independent direct-window oracle: no sliding state,
                    // atomics, tiles or packed histogram indices.
                    std::array<int, 9> bins{};
                    for (int yy = y - radius; yy <= y + radius; ++yy) {
                        if (yy < 0 || yy >= static_cast<int>(size.height))
                            continue;
                        for (int xx = x - radius; xx <= x + radius; ++xx) {
                            if (xx < 0 || xx >= static_cast<int>(size.width))
                                continue;
                            const auto sample =
                                static_cast<std::size_t>(yy) * size.width +
                                static_cast<std::size_t>(xx);
                            if (runtime.staging_map()[pixels + sample] == 0.0f)
                                continue;
                            const int value = static_cast<int>(filtered[sample]);
                            const int delta = value - center;
                            if (value >= 67 && value <= 563 && delta >= -4 &&
                                delta <= 4)
                                ++bins[static_cast<std::size_t>(delta + 4)];
                        }
                    }
                    constexpr std::array<int, 4> tvi{182, 309, 436, 563};
                    for (std::size_t d = 1; d <= 4; ++d) {
                        if (center + 4 > tvi[d - 1] ||
                            center + 4 + static_cast<int>(d) <= 78)
                            continue;
                        const int p0 = bins[4];
                        const int other = std::max(bins[4 - d], bins[4 + d]);
                        if (p0 + other != 0) {
                            const float value =
                                static_cast<float>(
                                    static_cast<int>(d) * p0 * other
                                ) *
                                (1.0f / static_cast<float>(p0 + other));
                            expected = std::max(expected, value);
                        }
                    }
                }
                CHECK_NEAR(map[at], expected, 1e-4);
            }
        }
    }
}

VQM_TEST(TestVmafCambiHalfSourceMatchesIntegerAndRequiresEnabledFeature) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr vqm::Size size{65, 37};
    ContrastDevice automatic(size), portable(size, false),
        borrowed_without_float16(size, true, false);
    const auto context = vqm::internal::AcquireVulkanContext();
    CHECK(automatic.UsesHalfSource() == context->shader_float16);
    CHECK(!portable.UsesHalfSource());
    CHECK(!borrowed_without_float16.UsesHalfSource());
    for (unsigned test_case = 0; test_case < 8; ++test_case) {
        const unsigned window = test_case == 0                       ? 1u
                                : (test_case == 1 || test_case == 6) ? 17u
                                                                     : 33u;
        const auto expected = portable.Compute(test_case, window);
        const auto actual = automatic.Compute(test_case, window);
        CHECK(actual.size() == expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
            CHECK(
                std::bit_cast<std::uint32_t>(actual[i]) ==
                std::bit_cast<std::uint32_t>(expected[i])
            );
        if (test_case == 3) {
            const auto fallback =
                borrowed_without_float16.Compute(test_case, window);
            CHECK(fallback.size() == expected.size());
            for (std::size_t i = 0; i < fallback.size(); ++i)
                CHECK(
                    std::bit_cast<std::uint32_t>(fallback[i]) ==
                    std::bit_cast<std::uint32_t>(expected[i])
                );
        }
    }
}

VQM_TEST(TestVmafCambiSubgroupEligibility) {
    using vqm::internal::VmafCambiContrast;
    VkPhysicalDeviceSubgroupProperties properties{};
    CHECK(!VmafCambiContrast::SupportsSubgroupRows(properties));
    properties.subgroupSize = 32;
    properties.supportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
    properties.supportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT;
    CHECK(VmafCambiContrast::SupportsSubgroupRows(properties));
    for (const std::uint32_t size : {1u, 16u, 64u, 128u}) {
        properties.subgroupSize = size;
        CHECK(!VmafCambiContrast::SupportsSubgroupRows(properties));
    }
    properties.subgroupSize = 32;
    properties.supportedStages = VK_SHADER_STAGE_VERTEX_BIT;
    CHECK(!VmafCambiContrast::SupportsSubgroupRows(properties));
    properties.supportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
    properties.supportedOperations = VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    CHECK(!VmafCambiContrast::SupportsSubgroupRows(properties));
    properties.supportedOperations |= VK_SUBGROUP_FEATURE_BASIC_BIT;
    CHECK(VmafCambiContrast::SupportsSubgroupRows(properties));
}

VQM_TEST(TestVmafCambiSubgroupAndPortableModulesMatchExactly) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr vqm::Size size{65, 37};
    ContrastDevice portable_uint(size, false, true, false);
    ContrastDevice portable_half(size, true, true, false);
    ContrastDevice subgroup_uint(size, false, true, true);
    ContrastDevice subgroup_half(size, true, true, true);
    ContrastDevice borrowed_uint(size, true, false, true);
    ContrastDevice borrowed_portable(size, true, false, false);
    const auto context = vqm::internal::AcquireVulkanContext();
    VkPhysicalDeviceSubgroupProperties properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 physical{};
    physical.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physical.pNext = &properties;
    vkGetPhysicalDeviceProperties2(context->physical_device, &physical);
    const bool eligible =
        vqm::internal::VmafCambiContrast::SupportsSubgroupRows(properties);
    CHECK(!portable_uint.UsesSubgroupPipeline());
    CHECK(!portable_half.UsesSubgroupPipeline());
    CHECK(subgroup_uint.UsesSubgroupPipeline() == eligible);
    CHECK(subgroup_half.UsesSubgroupPipeline() == eligible);
    CHECK(borrowed_uint.UsesSubgroupPipeline() == eligible);
    CHECK(!borrowed_portable.UsesSubgroupPipeline());
    CHECK(!portable_uint.UsesHalfSource());
    CHECK(portable_half.UsesHalfSource() == context->shader_float16);
    CHECK(subgroup_half.UsesHalfSource() == context->shader_float16);
    CHECK(!borrowed_uint.UsesHalfSource());
    CHECK(!borrowed_portable.UsesHalfSource());
    for (unsigned test_case = 0; test_case < 8; ++test_case) {
        const unsigned window = test_case == 0                       ? 1u
                                : (test_case == 1 || test_case == 6) ? 17u
                                                                     : 33u;
        const auto expected = portable_uint.Compute(test_case, window);
        for (auto* device : {&portable_half, &subgroup_uint, &subgroup_half,
                             &borrowed_uint, &borrowed_portable}) {
            const auto actual = device->Compute(test_case, window);
            CHECK(actual.size() == expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i)
                CHECK(
                    std::bit_cast<std::uint32_t>(actual[i]) ==
                    std::bit_cast<std::uint32_t>(expected[i])
                );
        }
    }
}
