// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF: Copyright (c) Netflix, Inc. BSD-2-Clause-Patent;
// see licenses/NETFLIX-VMAF-LICENSE.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "support.h"
#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "vulkan/vulkan_runtime.h"

namespace {

using vqm::Size;
using vqm::internal::Bytes;
using vqm::internal::FitVmafInto;
using vqm::internal::VmafFitGeometry;
using vqm::internal::VmafResizer;
using vqm::internal::VulkanRuntime;

float Sinc(float x) {
    return x == 0.0f ? 1.0f
                     : std::sin(3.14159265358979323846f * x) /
                           (3.14159265358979323846f * x);
}

float Kernel(float x, std::uint32_t filter) {
    x = std::abs(x);
    if (filter == 1) {
        if (x < 1.0f) return 1.0f + x * x * (-2.5f + x * 1.5f);
        if (x < 2.0f) return 2.0f + x * (-4.0f + x * (2.5f - 0.5f * x));
        return 0.0f;
    }
    return x < 3.0f ? Sinc(x) * Sinc(x / 3.0f) : 0.0f;
}

std::uint32_t ClampIndex(int index, std::uint32_t size) {
    return static_cast<std::uint32_t>(
        std::clamp(index, 0, static_cast<int>(size) - 1)
    );
}

float SampleAxis(
    const std::vector<float>& plane, std::uint32_t offset, std::uint32_t stride,
    std::uint32_t in_size, std::uint32_t out_size, std::uint32_t out_pos,
    std::uint32_t filter
) {
    if (filter == 0)
        return plane
            [offset + ClampIndex(static_cast<int>(out_pos), in_size) * stride];
    const float scale =
        static_cast<float>(in_size) / static_cast<float>(out_size);
    const float pos = (static_cast<float>(out_pos) + 0.5f) * scale - 0.5f;
    const float support_scale = std::max(1.0f, scale);
    const float radius = filter == 1 ? 2.0f : 3.0f;
    const float support = radius * support_scale;
    const int first = static_cast<int>(std::ceil(pos - support));
    const int last = static_cast<int>(std::floor(pos + support));
    float sum = 0.0f;
    float total = 0.0f;
    for (int i = first; i <= last; ++i) {
        const float weight =
            Kernel((static_cast<float>(i) - pos) / support_scale, filter);
        sum += weight * plane[offset + ClampIndex(i, in_size) * stride];
        total += weight;
    }
    if (std::abs(total) < 1e-20f)
        return plane
            [offset +
             ClampIndex(static_cast<int>(std::round(pos)), in_size) * stride];
    return sum / total;
}

std::vector<float> ResizeReference(
    const std::vector<float>& source, Size source_size, Size target_size,
    float pad
) {
    const VmafFitGeometry geometry = FitVmafInto(source_size, target_size);
    const bool horizontal = geometry.content.width != source_size.width;
    std::vector<float> intermediate;
    const std::vector<float>* vertical_source = &source;
    Size vertical_size = source_size;
    if (horizontal) {
        intermediate.resize(geometry.content.width * source_size.height);
        const std::uint32_t filter =
            geometry.content.width > source_size.width ? 1u : 2u;
        for (std::size_t y = 0; y < source_size.height; ++y) {
            for (std::size_t x = 0; x < geometry.content.width; ++x) {
                intermediate[y * geometry.content.width + x] = SampleAxis(
                    source, static_cast<std::uint32_t>(y * source_size.width), 1,
                    static_cast<std::uint32_t>(source_size.width),
                    static_cast<std::uint32_t>(geometry.content.width),
                    static_cast<std::uint32_t>(x), filter
                );
            }
        }
        vertical_source = &intermediate;
        vertical_size = {geometry.content.width, source_size.height};
    }
    std::vector<float> result(target_size.width * target_size.height, pad);
    const std::uint32_t filter =
        geometry.content.height == source_size.height
            ? 0u
            : (geometry.content.height > source_size.height ? 1u : 2u);
    for (std::size_t y = 0; y < geometry.content.height; ++y) {
        for (std::size_t x = 0; x < geometry.content.width; ++x) {
            const std::size_t target_index =
                (geometry.y + y) * target_size.width + geometry.x + x;
            result[target_index] = SampleAxis(
                *vertical_source, static_cast<std::uint32_t>(x),
                static_cast<std::uint32_t>(vertical_size.width),
                static_cast<std::uint32_t>(vertical_size.height),
                static_cast<std::uint32_t>(geometry.content.height),
                static_cast<std::uint32_t>(y), filter
            );
        }
    }
    return result;
}

class ResizeDevice {
public:
    ResizeDevice(Size source, Size target)
        : source_(source),
          target_(target),
          geometry_(FitVmafInto(source, target)),
          source_pixels_(source.width * source.height),
          target_pixels_(target.width * target.height),
          horizontal_(geometry_.content.width != source.width),
          scratch_pixels_(
              horizontal_ ? geometry_.content.width * source.height : 0
          ),
          scratch_(static_cast<std::uint32_t>(source_pixels_)),
          destination_(
              static_cast<std::uint32_t>(source_pixels_ + scratch_pixels_)
          ),
          runtime_(
              vqm::internal::AcquireVulkanContext(),
              Config(source, source_pixels_)
          ) {
        runtime_.CreateArena(Bytes(destination_ + target_pixels_));
        runtime_.CreateStaging(Bytes(source_pixels_ + target_pixels_));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(destination_ + target_pixels_));
        resizer_.emplace(runtime_);
    }

    std::vector<float> Compute(const std::vector<float>& source, float pad) {
        CHECK(source.size() == source_pixels_);
        std::copy(source.begin(), source.end(), runtime_.staging_map());
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{VulkanRuntime::CopyRange{0, 0, source_pixels_}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        const Size vertical_source{geometry_.content.width, source_.height};
        std::uint32_t input = 0;
        if (horizontal_) {
            resizer_->RecordHorizontal(0, scratch_, source_, vertical_source);
            runtime_.ComputeBarrier();
            input = scratch_;
        }
        resizer_->RecordVertical(
            input, destination_, vertical_source, geometry_, target_, pad
        );
        runtime_.RecordReadback(destination_, source_pixels_, target_pixels_);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        return std::vector<float>(
            runtime_.staging_map() + source_pixels_,
            runtime_.staging_map() + source_pixels_ + target_pixels_
        );
    }

private:
    static VulkanRuntime::Config Config(Size size, std::size_t pixels) {
        VulkanRuntime::Config config;
        config.base_size = size;
        config.base_pixels = pixels;
        config.push_constant_size = VmafResizer::PushBytes();
        config.streaming = false;
        return config;
    }

    Size source_;
    Size target_;
    VmafFitGeometry geometry_;
    std::size_t source_pixels_;
    std::size_t target_pixels_;
    bool horizontal_;
    std::size_t scratch_pixels_;
    std::uint32_t scratch_;
    std::uint32_t destination_;
    VulkanRuntime runtime_;
    std::optional<VmafResizer> resizer_;
};

}  // namespace

VQM_TEST(TestVmafFitResampler) {
    if (!vqm::test::GpuAvailable()) return;
    struct Case {
        Size source;
        Size target;
    };
    constexpr std::array cases{
        Case{{64, 48}, {128, 128}}, Case{{128, 96}, {64, 48}},
        Case{{32, 24}, {64, 24}}, Case{{24, 32}, {48, 32}}};
    for (const Case& test : cases) {
        std::vector<float> source(test.source.width * test.source.height);
        for (std::size_t y = 0; y < test.source.height; ++y) {
            for (std::size_t x = 0; x < test.source.width; ++x) {
                source[y * test.source.width + x] =
                    static_cast<float>((37 * x + 19 * y) % 251) - 125.0f;
            }
        }
        constexpr float kPad = -112.0f;
        ResizeDevice device(test.source, test.target);
        const auto actual = device.Compute(source, kPad);
        const auto expected =
            ResizeReference(source, test.source, test.target, kPad);
        CHECK(actual.size() == expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
            CHECK_NEAR(actual[i], expected[i], 1e-4);
    }
}
