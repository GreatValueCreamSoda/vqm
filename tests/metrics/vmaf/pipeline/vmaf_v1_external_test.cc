// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// VMAF: Copyright (c) Netflix, Inc. BSD-2-Clause-Patent;
// see licenses/NETFLIX-VMAF-LICENSE.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include "input.h"
#include "support.h"
#include "temporal_test_support.h"
#include "vmaf_v1.h"
#include "vmaf_sdr_fixtures.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"

namespace {
using namespace vqm;
using namespace vqm::internal;
constexpr Size kSize{224, 224};
Image Frame(bool distorted) {
    ImageFormat fmt;
    fmt.type = PixelType::Word;
    fmt.depth = 10;
    fmt.family = ColorFamily::Yuv;
    fmt.range = PixelRange::Limited;
    fmt.matrix = MatrixCoefficients::Bt709;
    fmt.subsample_w = 1;
    fmt.subsample_h = 1;
    Image image{kSize, fmt, {}};
    image.data.resize(RequiredBytes(fmt, kSize));
    const auto planes = PlaneLayout(fmt, kSize);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t y = 0; y < planes[c].height; ++y)
            for (std::size_t x = 0; x < planes[c].width; ++x) {
                int v = 64 + static_cast<int>((3 * x + 7 * y + 149 * c) % 850);
                if (distorted)
                    v = (v + static_cast<int>((5 * x + 11 * y + 13 * c) % 15)) %
                        1024;
                const auto sample = static_cast<std::uint16_t>(v);
                const auto pos =
                    planes[c].offset + 2 * (y * planes[c].width + x);
                std::memcpy(image.data.data() + pos, &sample, 2);
            }
    return image;
}
class Buffer {
public:
    Buffer(
        const VulkanContext& context, const Image& image, VkDeviceSize prefix,
        std::size_t padding
    )
        : device_(context.device),
          prefix_(prefix),
          size_(image.size),
          format_(image.format) {
        const auto planes = PlaneLayout(image.format, image.size);
        VkDeviceSize cursor = 0;
        for (std::size_t p = 0; p < 3; ++p) {
            const auto stride =
                planes[p].sample_bytes * planes[p].width + padding;
            offsets_[p] = cursor;
            strides_[p] = stride;
            cursor += stride * planes[p].height;
        }
        bytes_ = cursor;
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = prefix + bytes_;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(vkCreateBuffer(device_, &info, nullptr, &buffer_), "test buffer");
        try {
            VkMemoryRequirements req{};
            vkGetBufferMemoryRequirements(device_, buffer_, &req);
            std::uint32_t type = context.memory_properties.memoryTypeCount;
            constexpr auto wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            for (std::uint32_t i = 0;
                 i < context.memory_properties.memoryTypeCount; ++i) {
                if ((req.memoryTypeBits & (1u << i)) &&
                    (context.memory_properties.memoryTypes[i].propertyFlags &
                     wanted) == wanted) {
                    type = i;
                    break;
                }
            }
            if (type == context.memory_properties.memoryTypeCount)
                throw std::runtime_error("no coherent test memory");
            VkMemoryAllocateInfo alloc{};
            alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc.allocationSize = req.size;
            alloc.memoryTypeIndex = type;
            Check(
                vkAllocateMemory(device_, &alloc, nullptr, &memory_),
                "test allocation"
            );
            Check(vkBindBufferMemory(device_, buffer_, memory_, 0), "test bind");
            Write(image);
        } catch (...) {
            Destroy();
            throw;
        }
    }
    ~Buffer() { Destroy(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void Write(const Image& image) {
        if (image.size != size_ || !SameImageFormat(image.format, format_))
            throw std::invalid_argument("test buffer: incompatible image");
        void* mapping = nullptr;
        Check(
            vkMapMemory(device_, memory_, 0, prefix_ + bytes_, 0, &mapping),
            "test map"
        );
        const auto planes = PlaneLayout(image.format, image.size);
        for (std::size_t p = 0; p < 3; ++p)
            for (std::size_t y = 0; y < planes[p].height; ++y)
                std::memcpy(
                    static_cast<std::byte*>(mapping) + prefix_ + offsets_[p] +
                        strides_[p] * y,
                    image.data.data() + planes[p].offset +
                        planes[p].sample_bytes * planes[p].width * y,
                    planes[p].sample_bytes * planes[p].width
                );
        vkUnmapMemory(device_, memory_);
    }
    VulkanBufferView View() const {
        VulkanBufferView view{buffer_, prefix_, bytes_, size_, format_};
        for (std::size_t p = 0; p < 3; ++p)
            view.planes[p] = {offsets_[p], strides_[p]};
        return view;
    }

private:
    void Destroy() noexcept {
        if (buffer_ != VK_NULL_HANDLE)
            vkDestroyBuffer(device_, buffer_, nullptr);
        if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        memory_ = VK_NULL_HANDLE;
    }
    VkDevice device_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize prefix_, bytes_ = 0;
    Size size_;
    ImageFormat format_;
    std::array<VkDeviceSize, 3> offsets_{}, strides_{};
};
}  // namespace

VQM_TEST(TestVmafV1BorrowedPaddedBuffers) {
    if (!vqm::test::GpuAvailable()) return;
    const auto context = AcquireVulkanContext();
    const VulkanDevice device{context->physical_device, context->device,
                              context->queue, context->queue_family,
                              context->sixteen_bit_storage};
    const Image a = Frame(false), b = Frame(true);
    Buffer ba(*context, a, 4, 8), bb(*context, b, 12, 12);
    VmafV1Pipeline pipeline({.size = kSize, .input_format = a.format}, device);
    CHECK(!pipeline.Push(a, b));
    const auto host = pipeline.Flush();
    CHECK(host.has_value());
    pipeline.Reset();
    CHECK(!pipeline.Push(ba.View(), bb.View()));
    const auto external = pipeline.Flush();
    CHECK(external.has_value());
    if (host && external) {
        CHECK_NEAR(external->cambi, host->cambi, 1e-6);
        CHECK_NEAR(external->speed_chroma_uv, host->speed_chroma_uv, 1e-6);
        CHECK_NEAR(external->adm3, host->adm3, 1e-6);
        CHECK_NEAR(external->score, host->score, 1e-5);
    }
    pipeline.Reset();
    CHECK(!pipeline.Push(a, b));
    const auto again = pipeline.Flush();
    if (host && again) CHECK_NEAR(again->score, host->score, 1e-5);
}

VQM_TEST(TestVmafV1BorrowedSdrRgbBuffers) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::test::sdr;
    const auto context = AcquireVulkanContext();
    const VulkanDevice device{context->physical_device, context->device,
                              context->queue, context->queue_family,
                              context->sixteen_bit_storage};
    const auto original =
        MakeSource(PixelType::Float, ColorFamily::Rgb, AlphaType::None, false);
    const auto distorted =
        MakeSource(PixelType::Float, ColorFamily::Rgb, AlphaType::None, true);
    Buffer a(*context, original, 4, 8), b(*context, distorted, 12, 12);
    VmafV1Config config;
    config.size = original.size;
    config.input_format = original.format;
    config.chroma_sampling = VmafChromaSampling::Yuv444;
    VmafV1Pipeline pipeline(config, device);
    CHECK(!pipeline.Push(original, distorted));
    const auto host = pipeline.Flush();
    CHECK(host.has_value());
    pipeline.Reset();
    CHECK(!pipeline.Push(a.View(), b.View()));
    const auto external = pipeline.Flush();
    CHECK(external.has_value());
    if (host && external) {
        CHECK_NEAR(external->cambi, host->cambi, 1e-6);
        CHECK_NEAR(external->speed_chroma_uv, host->speed_chroma_uv, 1e-6);
        CHECK_NEAR(external->adm3, host->adm3, 1e-6);
        CHECK_NEAR(external->score, host->score, 1e-5);
    }
}

VQM_TEST(TestVmafV1BorrowedSubsampledSourceExactEnd) {
    if (!vqm::test::GpuAvailable()) return;
    const auto context = AcquireVulkanContext();
    const VulkanDevice device{context->physical_device, context->device,
                              context->queue, context->queue_family,
                              context->sixteen_bit_storage};
    const auto original = Frame(false);
    const auto distorted = Frame(true);
    // No row or end padding: the final V-plane row ends exactly at the
    // descriptor's range and at the VkBuffer boundary.
    Buffer a(*context, original, 4, 0), b(*context, distorted, 12, 0);
    VmafV1Config config;
    config.size = original.size;
    config.input_format = original.format;
    config.chroma_sampling = VmafChromaSampling::Yuv444;
    VmafV1Pipeline pipeline(config, device);
    CHECK(!pipeline.Push(original, distorted));
    const auto host = pipeline.Flush();
    CHECK(host.has_value());
    pipeline.Reset();
    CHECK(!pipeline.Push(a.View(), b.View()));
    const auto external = pipeline.Flush();
    CHECK(external.has_value());
    if (host && external) {
        CHECK_NEAR(external->cambi, host->cambi, 1e-6);
        CHECK_NEAR(external->speed_chroma_uv, host->speed_chroma_uv, 1e-6);
        CHECK_NEAR(external->adm3, host->adm3, 1e-6);
        CHECK_NEAR(external->score, host->score, 1e-5);
    }
}

VQM_TEST(TestVmafV1BorrowedBufferReusePreservesTemporalHistory) {
    if (!vqm::test::GpuAvailable()) return;
    const auto context = AcquireVulkanContext();
    const VulkanDevice device{context->physical_device, context->device,
                              context->queue, context->queue_family,
                              context->sixteen_bit_storage};
    const auto pairs = vqm::test::temporal::MakeClip(kSize, 6);
    VmafV1Config config;
    config.size = kSize;
    config.input_format = pairs[0].original.format;
    VmafV1Pipeline host(config, device), borrowed(config, device);
    const auto expected = vqm::test::temporal::RunClip<VmafV1Frame>(host, pairs);
    Buffer original(*context, pairs[0].original, 4, 8);
    Buffer distorted(*context, pairs[0].distorted, 12, 12);
    std::vector<VmafV1Frame> actual;
    for (const auto& pair : pairs) {
        original.Write(pair.original);
        distorted.Write(pair.distorted);
        if (const auto frame = borrowed.Push(original.View(), distorted.View()))
            actual.push_back(*frame);
        // Push promises private history: callers may overwrite both source
        // buffers immediately, including before the deferred tail is emitted.
        original.Write(pairs[0].distorted);
        distorted.Write(pairs[0].original);
    }
    if (const auto last = borrowed.Flush()) actual.push_back(*last);
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
        CHECK(actual[i].index == expected[i].index);
        CHECK_NEAR(actual[i].cambi, expected[i].cambi, 1e-6);
        CHECK_NEAR(actual[i].speed_chroma_uv, expected[i].speed_chroma_uv, 1e-6);
        CHECK_NEAR(actual[i].adm3, expected[i].adm3, 1e-6);
        CHECK_NEAR(actual[i].motion3, expected[i].motion3, 1e-6);
        CHECK_NEAR(actual[i].score, expected[i].score, 1e-5);
    }
    vqm::test::temporal::CheckPooled<VmafV1Frame>(borrowed, actual);
}
