// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF input scaling: Copyright (c) Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "support.h"
#include "vmaf.h"
#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "metrics/vmaf/features/vmaf_hdr.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"
#include "metrics/vmaf/pipeline/vmaf_sdr_fixtures.h"

namespace {

using namespace vqm;
using namespace vqm::internal;

constexpr Size kSize{33, 17};
constexpr std::size_t kPixels = kSize.width * kSize.height;
constexpr std::uint32_t kDesc0 = 8 * kPixels;
constexpr std::uint32_t kDesc1 = kDesc0 + kSourceDescWords;
constexpr std::uint32_t kOutput = kDesc1 + kSourceDescWords;
constexpr std::size_t kArenaFloats = kOutput + 2 * kPixels;
constexpr std::size_t kReadback = 8 * kPixels + 2 * kSourceDescWords;

ImageFormat Format(PixelType type, std::uint8_t depth, PixelRange range) {
    ImageFormat format;
    format.family = ColorFamily::Grey;
    format.matrix = MatrixCoefficients::Bt709;
    format.type = type;
    format.depth = depth;
    format.range = range;
    return format;
}

Image ImageFor(const ImageFormat& format, std::uint32_t seed) {
    Image image{kSize, format, {}};
    image.data.resize(RequiredBytes(format, kSize));
    const unsigned depth = VmafLumaDepth(format);
    const std::uint32_t mask = (1u << depth) - 1u;
    for (std::size_t i = 0; i < kPixels; ++i) {
        const auto sample = static_cast<std::uint16_t>(
            (static_cast<std::uint32_t>(i) * 97u + seed) & mask
        );
        if (format.type == PixelType::Byte) {
            image.data[i] = static_cast<std::uint8_t>(sample);
        } else {
            std::memcpy(image.data.data() + 2 * i, &sample, 2);
        }
    }
    return image;
}

float Expected(const Image& image, std::size_t i) {
    std::uint16_t sample = 0;
    if (image.format.type == PixelType::Byte) {
        sample = image.data[i];
    } else {
        std::memcpy(&sample, image.data.data() + 2 * i, 2);
    }
    const std::uint32_t depth = VmafLumaDepth(image.format);
    return static_cast<float>(sample) / static_cast<float>(1u << (depth - 8)) -
           128.0f;
}

class LumaHarness {
public:
    explicit LumaHarness(std::shared_ptr<VulkanContext> context)
        : runtime_(std::move(context), Config()) {
        runtime_.CreateArena(Bytes(kArenaFloats));
        if (!runtime_.arena_visible()) {
            runtime_.CreateStaging(Bytes(kReadback + 2 * kPixels));
        }
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(kArenaFloats));
        converter_ = std::make_unique<VmafLumaConverter>(runtime_);
    }

    std::vector<float> Run(const Image& original, const Image& distorted) {
        runtime_.UploadStreaming(original, distorted);
        return Convert({}, original.format, distorted.format);
    }

    std::vector<float> Run(
        const ImageView& original, const ImageView& distorted
    ) {
        runtime_.UploadStreaming(original, distorted);
        return Convert({}, original.format, distorted.format);
    }

    std::vector<float> Run(
        const VulkanBufferView& original, const VulkanBufferView& distorted
    ) {
        const auto sources = runtime_.ResolveExternalInputs(original, distorted);
        return Convert(sources, original.format, distorted.format);
    }

    std::vector<float> RunPlanes(
        const Image& original, const Image& distorted, std::uint32_t plane
    ) {
        runtime_.UploadStreaming(original, distorted);
        return Convert({}, original.format, distorted.format, plane);
    }

private:
    static VulkanRuntime::Config Config() {
        VulkanRuntime::Config config;
        config.stream_region = {0, 4 * kPixels};
        config.stream_desc = {kDesc0, kDesc1};
        config.staging_desc_offset = 8 * kPixels;
        config.staging_image_bytes = 16 * kPixels;
        config.base_pixels = kPixels;
        config.base_size = kSize;
        config.streaming = true;
        config.push_constant_size = VmafLumaConverter::PushBytes();
        return config;
    }

    std::vector<float> Convert(
        const std::array<VulkanRuntime::Source, 2>& sources,
        const ImageFormat& original, const ImageFormat& distorted,
        std::uint32_t plane = 0
    ) {
        runtime_.PrepareSourceBindings(sources);
        runtime_.BeginCommandBuffer();
        (void)runtime_.PublishSources(sources);
        converter_->RecordPlane(0, kDesc0, kOutput, kSize, original, plane);
        converter_->RecordPlane(
            1, kDesc1, kOutput + kPixels, kSize, distorted, plane
        );
        if (runtime_.arena_visible()) {
            runtime_.ComputeToHostBarrier();
        } else {
            runtime_.RecordReadback(kOutput, kReadback, 2 * kPixels);
        }
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        const float* values = runtime_.arena_visible()
                                  ? runtime_.arena_map() + kOutput
                                  : runtime_.staging_map() + kReadback;
        return {values, values + 2 * kPixels};
    }

    VulkanRuntime runtime_;
    std::unique_ptr<VmafLumaConverter> converter_;
};

class ExternalLumaBuffer {
public:
    ExternalLumaBuffer(
        const VulkanContext& context, const Image& image, VkDeviceSize prefix,
        std::size_t stride
    )
        : device_(context.device), prefix_(prefix), stride_(stride) {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = prefix + stride * image.size.height;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(vkCreateBuffer(device_, &info, nullptr, &buffer_), "test buffer");
        try {
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device_, buffer_, &requirements);
            std::uint32_t type = context.memory_properties.memoryTypeCount;
            for (std::uint32_t i = 0;
                 i < context.memory_properties.memoryTypeCount; ++i) {
                const auto flags =
                    context.memory_properties.memoryTypes[i].propertyFlags;
                const auto required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if ((requirements.memoryTypeBits & (1u << i)) &&
                    (flags & required) == required) {
                    type = i;
                    break;
                }
            }
            if (type == context.memory_properties.memoryTypeCount)
                throw std::runtime_error("test: no host-coherent buffer memory");
            VkMemoryAllocateInfo alloc{};
            alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc.allocationSize = requirements.size;
            alloc.memoryTypeIndex = type;
            Check(
                vkAllocateMemory(device_, &alloc, nullptr, &memory_),
                "test memory"
            );
            Check(vkBindBufferMemory(device_, buffer_, memory_, 0), "test bind");
            void* mapping = nullptr;
            Check(
                vkMapMemory(device_, memory_, 0, info.size, 0, &mapping),
                "test map"
            );
            const std::size_t row =
                image.size.width *
                (image.format.type == PixelType::Byte ? 1u : 2u);
            for (std::size_t y = 0; y < image.size.height; ++y) {
                std::memcpy(
                    static_cast<std::byte*>(mapping) + prefix + y * stride,
                    image.data.data() + y * row, row
                );
            }
            vkUnmapMemory(device_, memory_);
        } catch (...) {
            Destroy();
            throw;
        }
    }
    ~ExternalLumaBuffer() { Destroy(); }
    ExternalLumaBuffer(const ExternalLumaBuffer&) = delete;
    ExternalLumaBuffer& operator=(const ExternalLumaBuffer&) = delete;

    VulkanBufferView View(const Image& image) const {
        VulkanBufferView result{buffer_, prefix_, stride_ * image.size.height,
                                image.size, image.format};
        result.planes[0] = {0, stride_};
        return result;
    }

private:
    void Destroy() noexcept {
        if (buffer_ != VK_NULL_HANDLE)
            vkDestroyBuffer(device_, buffer_, nullptr);
        if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);
    }
    VkDevice device_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize prefix_;
    std::size_t stride_;
};

}  // namespace

VQM_TEST(TestVmafPipelineBorrowedBuffers) {
    if (!vqm::test::GpuAvailable()) return;
    const auto context = AcquireVulkanContext();
    const VulkanDevice device{context->physical_device, context->device,
                              context->queue, context->queue_family,
                              context->sixteen_bit_storage};
    const auto format = Format(PixelType::Word, 10, PixelRange::Limited);
    const Image a = ImageFor(format, 3), b = ImageFor(format, 19);
    ExternalLumaBuffer external_a(*context, a, 4, 2 * kSize.width + 8);
    ExternalLumaBuffer external_b(*context, b, 12, 2 * kSize.width + 12);
    VmafPipeline pipeline({.size = kSize}, device);
    const auto host = pipeline.Push(a, b);
    CHECK(host.has_value());
    (void)pipeline.Flush();
    pipeline.Reset();
    const auto gpu = pipeline.Push(external_a.View(a), external_b.View(b));
    CHECK(gpu.has_value());
    if (host && gpu) {
        CHECK_NEAR(gpu->score, host->score, 1e-6);
        CHECK_NEAR(gpu->adm2, host->adm2, 1e-7);
        for (std::size_t scale = 0; scale < 4; ++scale)
            CHECK_NEAR(gpu->vif[scale], host->vif[scale], 1e-7);
    }
    (void)pipeline.Flush();
    pipeline.Reset();
    const auto host_again = pipeline.Push(a, b);
    if (host && host_again) CHECK_NEAR(host_again->score, host->score, 1e-6);
}

VQM_TEST(TestVmafLumaFormatValidation) {
    const ImageFormat byte = Format(PixelType::Byte, 8, PixelRange::Limited);
    CHECK(VmafLumaDepth(byte) == 8);
    for (std::uint8_t depth : std::array<std::uint8_t, 3>{10, 12, 16}) {
        CHECK(
            VmafLumaDepth(Format(PixelType::Word, depth, PixelRange::Full)) ==
            depth
        );
    }
    auto invalid = byte;
    invalid.family = ColorFamily::Rgb;
    CHECK_THROWS_AS(VmafLumaDepth(invalid), std::invalid_argument);
    invalid = byte;
    invalid.alpha = AlphaType::Straight;
    CHECK_THROWS_AS(VmafLumaDepth(invalid), std::invalid_argument);
    invalid = byte;
    invalid.type = PixelType::Float;
    CHECK_THROWS_AS(VmafLumaDepth(invalid), std::invalid_argument);
    invalid = byte;
    invalid.depth = 7;
    CHECK_THROWS_AS(VmafLumaDepth(invalid), std::invalid_argument);
    invalid = Format(PixelType::Word, 9, PixelRange::Full);
    CHECK_THROWS_AS(VmafLumaDepth(invalid), std::invalid_argument);
}

VQM_TEST(TestVmafCodedChromaPlanes) {
    auto format = Format(PixelType::Word, 10, PixelRange::Limited);
    format.family = ColorFamily::Yuv;
    format.subsample_w = 1;
    format.subsample_h = 1;
    CHECK((VmafLumaConverter::PlaneSize(kSize, format, 0) == kSize));
    constexpr Size subsampled{kSize.width / 2, kSize.height / 2};
    CHECK((VmafLumaConverter::PlaneSize(kSize, format, 1) == subsampled));
    CHECK((VmafLumaConverter::PlaneSize(kSize, format, 2) == subsampled));
    CHECK_THROWS_AS(
        VmafLumaConverter::PlaneSize(kSize, format, 3), std::invalid_argument
    );
    auto grey = format;
    grey.family = ColorFamily::Grey;
    CHECK_THROWS_AS(
        VmafLumaConverter::PlaneSize(kSize, grey, 1), std::invalid_argument
    );
    if (!vqm::test::GpuAvailable()) return;

    // This harness deliberately keeps its odd geometry; use 4:4:4 here
    // because the shared input contract requires subsampled sizes to divide.
    format.subsample_w = 0;
    format.subsample_h = 0;
    constexpr Size chroma = kSize;
    Image a{kSize, format, {}}, b{kSize, format, {}};
    a.data.resize(RequiredBytes(format, kSize));
    b.data.resize(a.data.size());
    constexpr auto aligned = [](std::size_t n) {
        return (n + 3) & ~std::size_t{3};
    };
    constexpr std::size_t chroma_bytes = 2 * chroma.width * chroma.height;
    constexpr std::size_t u_offset = aligned(2 * kPixels);
    constexpr std::size_t v_offset = aligned(u_offset + chroma_bytes);
    for (std::size_t plane = 1; plane <= 2; ++plane) {
        const std::size_t offset = plane == 1 ? u_offset : v_offset;
        for (std::size_t i = 0; i < chroma.width * chroma.height; ++i) {
            const std::uint16_t av =
                static_cast<std::uint16_t>(400 + plane * 100 + i % 61);
            const std::uint16_t bv = static_cast<std::uint16_t>(av + 1 + plane);
            std::memcpy(a.data.data() + offset + 2 * i, &av, 2);
            std::memcpy(b.data.data() + offset + 2 * i, &bv, 2);
        }
    }
    LumaHarness harness(AcquireVulkanContext());
    for (std::uint32_t plane = 1; plane <= 2; ++plane) {
        const auto actual = harness.RunPlanes(a, b, plane);
        for (std::size_t i = 0; i < chroma.width * chroma.height; ++i) {
            const float av = static_cast<float>(400 + plane * 100 + i % 61);
            CHECK(actual[i] == av / 4.0f - 128.0f);
            CHECK(
                actual[kPixels + i] ==
                (av + 1.0f + static_cast<float>(plane)) / 4.0f - 128.0f
            );
        }
    }
}

VQM_TEST(TestVmafLumaCodedPrecisionAndStrides) {
    if (!vqm::test::GpuAvailable()) return;
    const auto context = AcquireVulkanContext();
    LumaHarness harness(context);
    for (const auto [type, depth] :
         {std::pair{PixelType::Byte, std::uint8_t{8}},
          std::pair{PixelType::Word, std::uint8_t{10}},
          std::pair{PixelType::Word, std::uint8_t{12}},
          std::pair{PixelType::Word, std::uint8_t{16}}}) {
        const auto a = ImageFor(Format(type, depth, PixelRange::Limited), 512);
        const auto b = ImageFor(Format(type, depth, PixelRange::Full), 513);
        const auto actual = harness.Run(a, b);
        for (std::size_t i = 0; i < kPixels; ++i) {
            CHECK(actual[i] == Expected(a, i));
            CHECK(actual[kPixels + i] == Expected(b, i));
        }
        const std::size_t row = kSize.width * (type == PixelType::Byte ? 1 : 2);
        const std::size_t stride = row + 8;
        std::vector<std::byte> a_padded(stride * kSize.height);
        std::vector<std::byte> b_padded(stride * kSize.height);
        for (std::size_t y = 0; y < kSize.height; ++y) {
            std::memcpy(
                a_padded.data() + y * stride, a.data.data() + y * row, row
            );
            std::memcpy(
                b_padded.data() + y * stride, b.data.data() + y * row, row
            );
        }
        ImageView av{a.size, a.format, {}};
        ImageView bv{b.size, b.format, {}};
        av.planes[0] = {a_padded.data(), stride};
        bv.planes[0] = {b_padded.data(), stride};
        const auto views = harness.Run(av, bv);
        CHECK(views == actual);
        // Each source has a distinct buffer, offset, and padded row stride.
        const std::size_t external_stride = (row + 7u) & ~std::size_t{3};
        const ExternalLumaBuffer source_a(*context, a, 4, external_stride);
        const ExternalLumaBuffer source_b(*context, b, 8, external_stride);
        const auto external = harness.Run(source_a.View(a), source_b.View(b));
        CHECK(external == actual);
        if (depth == 10) {
            CHECK(actual[0] == 0.0f);
            CHECK(actual[kPixels] == 0.25f);
        }
    }
}

namespace {
class SdrHarness {
public:
    explicit SdrHarness(
        std::shared_ptr<VulkanContext> context, Size size = kSize
    )
        : size_(size), runtime_(std::move(context), Config(size)) {
        runtime_.CreateArena(Bytes(kOutput + 3 * kPixels));
        if (!runtime_.arena_visible())
            runtime_.CreateStaging(Bytes(kReadback + 3 * kPixels));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(kOutput + 3 * kPixels));
        converter_ = std::make_unique<VmafSdrConverter>(runtime_);
    }

    std::vector<float> Run(const Image& image, bool subsampled) {
        const auto chroma_width =
            subsampled ? (size_.width + 1) / 2 : size_.width;
        const auto chroma_height =
            subsampled ? (size_.height + 1) / 2 : size_.height;
        const auto pixels =
            static_cast<std::uint32_t>(size_.width * size_.height);
        const auto chroma_area =
            static_cast<std::uint32_t>(chroma_width * chroma_height);
        const auto count = pixels + 2 * chroma_area;
        runtime_.UploadStreaming(image, image);
        runtime_.PrepareSourceBindings({});
        runtime_.BeginCommandBuffer();
        (void)runtime_.PublishSources({});
        converter_->Record(
            0, kDesc0, kOutput, kOutput + pixels, kOutput + pixels + chroma_area,
            size_, image.format, subsampled, true, 0.5f
        );
        if (runtime_.arena_visible())
            runtime_.ComputeToHostBarrier();
        else
            runtime_.RecordReadback(kOutput, kReadback, count);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        const float* values = runtime_.arena_visible()
                                  ? runtime_.arena_map() + kOutput
                                  : runtime_.staging_map() + kReadback;
        return {values, values + count};
    }

private:
    static VulkanRuntime::Config Config(Size size) {
        VulkanRuntime::Config config;
        config.stream_region = {0, 4 * kPixels};
        config.stream_desc = {kDesc0, kDesc1};
        config.staging_desc_offset = 8 * kPixels;
        config.staging_image_bytes = 16 * kPixels;
        config.base_pixels = kPixels;
        config.base_size = size;
        config.streaming = true;
        config.push_constant_size = VmafSdrConverter::PushBytes();
        return config;
    }
    Size size_;
    VulkanRuntime runtime_;
    std::unique_ptr<VmafSdrConverter> converter_;
};
}  // namespace

// Compare every converted sample to an independent scalar reference, including
// clamped odd-size edges of 4:2:0. The tolerance allows a few 16-bit codes for
// GPU FP32 transcendental and matrix rounding.
VQM_TEST(TestVmafSdrConvertedPlanes) {
    if (!vqm::test::GpuAvailable()) return;
    SdrHarness harness(AcquireVulkanContext());
    const auto original = vqm::test::sdr::MakeSource(
        PixelType::Float, ColorFamily::Rgb, AlphaType::Premultiplied, false,
        kSize
    );
    const auto expected_sample = [](double value) {
        return std::nearbyint(std::clamp(value * 256.0, 0.0, 65535.0)) / 256.0 -
               128.0;
    };
    for (bool subsampled : {false, true}) {
        const auto values = harness.Run(original, subsampled);
        const std::size_t cw = subsampled ? (kSize.width + 1) / 2 : kSize.width;
        const std::size_t ch =
            subsampled ? (kSize.height + 1) / 2 : kSize.height;
        const std::size_t ca = cw * ch;
        for (std::size_t y = 0; y < kSize.height; ++y)
            for (std::size_t x = 0; x < kSize.width; ++x) {
                const auto index = y * kSize.width + x;
                const auto expected =
                    vqm::test::sdr::ExpectedYuv(original, index, 0.5);
                CHECK_NEAR(values[index], expected_sample(expected[0]), 0.02);
                if (!subsampled) {
                    CHECK_NEAR(
                        values[kPixels + index], expected_sample(expected[1]),
                        0.02
                    );
                    CHECK_NEAR(
                        values[kPixels + ca + index],
                        expected_sample(expected[2]), 0.02
                    );
                }
            }
        if (subsampled) {
            for (std::size_t y = 0; y < ch; ++y)
                for (std::size_t x = 0; x < cw; ++x) {
                    double u = 0.0, v = 0.0;
                    for (std::size_t dy = 0; dy < 2; ++dy)
                        for (std::size_t dx = 0; dx < 2; ++dx) {
                            const auto px =
                                std::min(2 * x + dx, kSize.width - 1);
                            const auto py =
                                std::min(2 * y + dy, kSize.height - 1);
                            const auto expected = vqm::test::sdr::ExpectedYuv(
                                original, py * kSize.width + px, 0.5
                            );
                            u += expected[1];
                            v += expected[2];
                        }
                    const auto index = y * cw + x;
                    CHECK_NEAR(
                        values[kPixels + index], expected_sample(u / 4), 0.02
                    );
                    CHECK_NEAR(
                        values[kPixels + ca + index], expected_sample(v / 4),
                        0.02
                    );
                }
        }
    }
}

VQM_TEST(TestVmafSdrSubsampledYuvSourceEdges) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{32, 16};
    ImageFormat format;
    format.type = PixelType::Byte;
    format.family = ColorFamily::Yuv;
    format.matrix = MatrixCoefficients::Bt709;
    format.transfer = TransferCharacteristics::Bt709;
    format.range = PixelRange::Limited;
    format.subsample_w = 1;
    format.subsample_h = 1;
    Image image{size, format, {}};
    image.data.resize(RequiredBytes(format, size));
    const auto planes = PlaneLayout(format, size);
    for (std::size_t y = 0; y < size.height; ++y)
        for (std::size_t x = 0; x < size.width; ++x)
            image.data[planes[0].offset + y * size.width + x] =
                static_cast<std::uint8_t>(64 + (3 * x + 7 * y) % 128);
    // Non-neutral constant chroma makes every interpolation edge independently
    // predictable, including the final source row and column.
    for (unsigned plane = 1; plane < 3; ++plane)
        std::fill_n(
            image.data.begin() +
                static_cast<std::ptrdiff_t>(planes[plane].offset),
            planes[plane].width * planes[plane].height,
            plane == 1 ? std::uint8_t{112} : std::uint8_t{144}
        );
    SdrHarness harness(AcquireVulkanContext(), size);
    for (bool subsampled : {false, true}) {
        const auto values = harness.Run(image, subsampled);
        const auto pixels = size.width * size.height;
        const auto cw = subsampled ? size.width / 2 : size.width;
        const auto ch = subsampled ? size.height / 2 : size.height;
        const auto chroma_area = cw * ch;
        for (std::size_t i = 0; i < pixels; ++i)
            CHECK_NEAR(
                values[i],
                static_cast<double>(image.data[planes[0].offset + i]) - 128.0,
                0.02
            );
        for (std::size_t i = 0; i < chroma_area; ++i) {
            CHECK_NEAR(values[pixels + i], -16.0, 0.02);
            CHECK_NEAR(values[pixels + chroma_area + i], 16.0, 0.02);
        }
    }
}

VQM_TEST(TestVmafHdrQuantizesToLimited16BitCodes) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{32, 16};
    constexpr std::uint32_t luma = 32 * 16, chroma = luma / 4;
    constexpr std::array output{kOutput, kOutput + luma,
                                kOutput + luma + chroma};
    constexpr auto count = luma + 2 * chroma;
    constexpr auto scratch = kOutput + count;
    const ImageFormat format{
        .type = PixelType::Word,
        .family = ColorFamily::Yuv,
        .range = PixelRange::Full,
        .matrix = MatrixCoefficients::Bt2100ICtCp,
        .transfer = TransferCharacteristics::St2084,
        .primaries = ColorPrimaries::Bt2020,
        .subsample_w = 1,
        .subsample_h = 1,
        .depth = 10,
    };
    Image image{size, format, {}};
    constexpr std::array<std::uint16_t, 8> samples{0,   64,  129, 503,
                                                   512, 643, 921, 1023};
    image.data.resize(RequiredBytes(format, size));
    for (std::size_t i = 0; i < count; ++i) {
        const auto code = samples[i % samples.size()];
        std::memcpy(image.data.data() + 2 * i, &code, sizeof(code));
    }

    VulkanRuntime::Config config;
    config.stream_region = {0, 4 * kPixels};
    config.stream_desc = {kDesc0, kDesc1};
    config.staging_desc_offset = 8 * kPixels;
    config.staging_image_bytes = 16 * kPixels;
    config.base_pixels = luma;
    config.base_size = size;
    config.streaming = true;
    config.push_constant_size = VmafHdrConverter::PushBytes();
    VulkanRuntime runtime(AcquireVulkanContext(), config);
    const auto arena_bytes = Bytes(scratch + 2 * luma);
    runtime.CreateArena(arena_bytes);
    if (!runtime.arena_visible())
        runtime.CreateStaging(Bytes(kReadback + count));
    runtime.CreateCommandObjects();
    runtime.CreateDescriptors(0, arena_bytes);
    VmafHdrConverter converter(runtime);
    runtime.UploadStreaming(image, image);
    runtime.PrepareSourceBindings({});
    runtime.BeginCommandBuffer();
    (void)runtime.PublishSources({});
    converter.Record(
        0, kDesc0, format, size, size, {{}, scratch, scratch + luma}, output
    );
    if (runtime.arena_visible())
        runtime.ComputeToHostBarrier();
    else
        runtime.RecordReadback(kOutput, kReadback, count);
    runtime.EndCommandBuffer();
    runtime.SubmitAndWait();
    const float* values = runtime.arena_visible()
                              ? runtime.arena_map() + kOutput
                              : runtime.staging_map() + kReadback;
    for (std::size_t i = 0; i < count; ++i) {
        const double input = samples[i % samples.size()];
        const double code = i < luma
                                ? 4096.0 + input * 56064.0 / 1023.0
                                : 32768.0 + (input - 512.0) * 57344.0 / 1023.0;
        CHECK(values[i] == std::nearbyint(code) / 256.0 - 128.0);
    }
}
