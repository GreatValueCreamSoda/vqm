// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
// CVVDP oracle: Graphics and Displays Group, MIT; see licenses/.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "cvvdp.h"
#include "support.h"
#include "vmaf.h"
#include "vqm_memory.h"
#include "vqm_metrics.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"

namespace {
using namespace vqm;
using namespace vqm::internal;

constexpr Size kSize{32, 32};

VulkanDevice Borrow(const VulkanContext& context) {
    return {context.physical_device,
            context.device,
            context.queue,
            context.queue_family,
            context.sixteen_bit_storage,
            context.shader_float16};
}

Image SolidChannel(std::size_t channel) {
    std::array<std::vector<float>, 4> values;
    for (std::size_t plane = 0; plane < 3; ++plane)
        values[plane].assign(
            kSize.width * kSize.height, plane == channel ? 1.0f : 0.0f
        );
    return vqm::test::PackPlanes(ImageFormat{}, kSize, values);
}

class PaddedHost {
public:
    explicit PaddedHost(const Image& image) {
        view_.size = image.size;
        view_.format = image.format;
        const auto layout = PlaneLayout(image.format, image.size);
        for (std::size_t plane = 0; plane < layout.size(); ++plane) {
            if (!layout[plane].present()) continue;
            const auto& geometry = layout[plane];
            const std::size_t row_bytes = geometry.width * geometry.sample_bytes;
            const std::size_t stride = row_bytes + 12;
            pixels_[plane].assign(stride * geometry.height + 4, std::byte{0x7F});
            view_.planes[plane] = {pixels_[plane].data() + 4, stride};
            for (std::size_t row = 0; row < geometry.height; ++row)
                std::memcpy(
                    pixels_[plane].data() + 4 + stride * row,
                    image.data.data() + geometry.offset + row_bytes * row,
                    row_bytes
                );
        }
    }
    [[nodiscard]] const ImageView& View() const { return view_; }

private:
    ImageView view_;
    std::array<std::vector<std::byte>, 4> pixels_;
};

// Caller-owned padded storage; updates are permitted only after the synchronous
// metric input-consumption boundary. The caller flushes even coherent heaps so
// the same fixture also works when HOST_VISIBLE is noncoherent.
class CallerBuffer {
public:
    CallerBuffer(const VulkanContext& context, const Image& image)
        : context_(context) {
        view_.size = image.size;
        view_.format = image.format;
        view_.offset = 12;
        const auto layout = PlaneLayout(image.format, image.size);
        std::size_t cursor = 4;
        for (std::size_t plane = 0; plane < layout.size(); ++plane) {
            if (!layout[plane].present()) continue;
            const auto& geometry = layout[plane];
            const auto stride = geometry.width * geometry.sample_bytes + 12;
            view_.planes[plane] = {cursor, stride};
            cursor += stride * geometry.height + 4;
        }
        view_.bytes = cursor;
        try {
            Allocate(view_.offset + view_.bytes);
            Write(image);
        } catch (...) {
            Destroy();
            throw;
        }
    }
    ~CallerBuffer() { Destroy(); }
    CallerBuffer(const CallerBuffer&) = delete;
    CallerBuffer& operator=(const CallerBuffer&) = delete;

    [[nodiscard]] const VulkanBufferView& View() const { return view_; }
    void Write(const Image& image) {
        if (image.size != view_.size ||
            !SameImageFormat(image.format, view_.format))
            throw std::invalid_argument("test buffer format changed");
        void* mapped = nullptr;
        Check(
            vkMapMemory(context_.device, memory_, 0, VK_WHOLE_SIZE, 0, &mapped),
            "interop map"
        );
        const auto layout = PlaneLayout(image.format, image.size);
        for (std::size_t plane = 0; plane < layout.size(); ++plane) {
            if (!layout[plane].present()) continue;
            const auto& geometry = layout[plane];
            const std::size_t bytes = geometry.width * geometry.sample_bytes;
            for (std::size_t row = 0; row < geometry.height; ++row)
                std::memcpy(
                    static_cast<std::byte*>(mapped) + view_.offset +
                        view_.planes[plane].offset_bytes +
                        view_.planes[plane].stride_bytes * row,
                    image.data.data() + geometry.offset + bytes * row, bytes
                );
        }
        VkMappedMemoryRange flush{};
        flush.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        flush.memory = memory_;
        flush.size = VK_WHOLE_SIZE;
        const auto result =
            vkFlushMappedMemoryRanges(context_.device, 1, &flush);
        vkUnmapMemory(context_.device, memory_);
        Check(result, "interop flush");
    }

private:
    void Allocate(VkDeviceSize bytes) {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(
            vkCreateBuffer(context_.device, &info, nullptr, &view_.buffer),
            "interop buffer"
        );
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(
            context_.device, view_.buffer, &requirements
        );
        const auto& properties = context_.memory_properties;
        std::uint32_t type = properties.memoryTypeCount;
        for (std::uint32_t index = 0; index < properties.memoryTypeCount;
             ++index) {
            if ((requirements.memoryTypeBits & (1u << index)) &&
                (properties.memoryTypes[index].propertyFlags &
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                type = index;
                break;
            }
        }
        if (type == properties.memoryTypeCount)
            throw std::runtime_error("no host-visible test buffer memory");
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        Check(
            vkAllocateMemory(context_.device, &allocation, nullptr, &memory_),
            "interop allocation"
        );
        Check(
            vkBindBufferMemory(context_.device, view_.buffer, memory_, 0),
            "interop bind"
        );
    }
    void Destroy() noexcept {
        if (view_.buffer != VK_NULL_HANDLE)
            vkDestroyBuffer(context_.device, view_.buffer, nullptr);
        if (memory_ != VK_NULL_HANDLE)
            vkFreeMemory(context_.device, memory_, nullptr);
        view_.buffer = VK_NULL_HANDLE;
        memory_ = VK_NULL_HANDLE;
    }
    const VulkanContext& context_;
    VulkanBufferView view_;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
};

class Semaphores {
public:
    explicit Semaphores(VulkanContext& context) : context_(context) {}
    ~Semaphores() {
        for (auto semaphore : semaphores_)
            vkDestroySemaphore(context_.device, semaphore, nullptr);
    }
    Semaphores(const Semaphores&) = delete;
    Semaphores& operator=(const Semaphores&) = delete;
    VkSemaphore Make() {
        VkSemaphoreCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        Check(
            vkCreateSemaphore(context_.device, &info, nullptr, &semaphore),
            "interop semaphore"
        );
        semaphores_.push_back(semaphore);
        return semaphore;
    }
    void Signal(VkSemaphore semaphore) {
        VkSubmitInfo submission{};
        submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submission.signalSemaphoreCount = 1;
        submission.pSignalSemaphores = &semaphore;
        std::lock_guard lock(context_.queue_mutex);
        Check(
            vkQueueSubmit(context_.queue, 1, &submission, VK_NULL_HANDLE),
            "interop signal"
        );
    }
    void Consume(VkSemaphore semaphore) {
        const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo submission{};
        submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submission.waitSemaphoreCount = 1;
        submission.pWaitSemaphores = &semaphore;
        submission.pWaitDstStageMask = &stage;
        VkFenceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        Check(
            vkCreateFence(context_.device, &info, nullptr, &fence),
            "interop fence"
        );
        {
            std::lock_guard lock(context_.queue_mutex);
            Check(
                vkQueueSubmit(context_.queue, 1, &submission, fence),
                "interop wait"
            );
        }
        const VkResult waited =
            vkWaitForFences(context_.device, 1, &fence, VK_TRUE, 5000000000ull);
        if (waited != VK_SUCCESS) {
            std::fprintf(stderr, "CVVDP interop signal was not completed\n");
            std::_Exit(EXIT_FAILURE);
        }
        vkDestroyFence(context_.device, fence, nullptr);
    }

private:
    VulkanContext& context_;
    std::vector<VkSemaphore> semaphores_;
};

void CheckStillSources(MemoryPolicy policy, const VulkanContext& context) {
    const auto a = vqm::test::MakeGradient(0), b = vqm::test::MakeGradient(1);
    const PaddedHost host_a(a), host_b(b);
    CallerBuffer external_a(context, a), external_b(context, b);
    CvvdpConfig config;
    config.size = a.size;
    config.memory_policy = policy;
    config.temporal = false;
    CvvdpPipeline pipeline(config, Borrow(context));
    const auto first = pipeline.Upload(a), second = pipeline.Upload(b);
    const double expected = pipeline.ComputeImage(a, b).score;
    CHECK(expected < 10.0);
    CHECK_NEAR(
        pipeline.ComputeImage(host_a.View(), host_b.View()).score, expected, 1e-7
    );
    CHECK_NEAR(
        pipeline.ComputeImage(external_a.View(), external_b.View()).score,
        expected, 1e-7
    );
    CHECK_NEAR(pipeline.ComputeImage(first, second).score, expected, 1e-7);
    CHECK(
        pipeline.ComputeImage(external_b.View(), external_b.View()).score == 10.0
    );
    CHECK_NEAR(pipeline.ComputeImage(a, b).score, expected, 1e-7);
    CHECK_NEAR(
        pipeline.ComputeImage(external_a.View(), external_b.View()).score,
        expected, 1e-7
    );
    CHECK(
        pipeline.GetMemoryUsage().uses_staging ==
        (policy == MemoryPolicy::Staged)
    );
    pipeline.Release(first);
    pipeline.Release(second);
}

void CheckTemporalReuse(MemoryPolicy policy, VulkanContext& context) {
    const auto green = SolidChannel(1), red = SolidChannel(0),
               blue = SolidChannel(2);
    CallerBuffer original(context, green), distorted(context, red);
    CvvdpConfig config;
    config.size = kSize;
    config.memory_policy = policy;
    config.gpu_resident_images = 0;
    CvvdpPipeline pipeline(config, Borrow(context));
    Semaphores semaphores(context);
    const auto ready = semaphores.Make(), consumed = semaphores.Make();
    const std::array waits{VulkanSemaphoreWait{ready}};
    const std::array signals{consumed};
    semaphores.Signal(ready);
    pipeline.Push(original.View(), distorted.View(), {waits, signals});
    semaphores.Consume(consumed);
    // Overwrite immediately after Push 0. The deferred first pair must already
    // live in private storage, even though spatial filtering has not run yet.
    distorted.Write(blue);
    pipeline.Push(original.View(), distorted.View());
    const PaddedHost host_green(green), host_red(red), host_blue(blue);
    for (std::size_t frame = 2; frame < 30; ++frame)
        pipeline.Push(
            host_green.View(), frame % 2 ? host_blue.View() : host_red.View()
        );
    const auto result = pipeline.Finalize();
    CHECK(result.has_value());
    if (result) {
        CHECK(result->frames == 30);
        // Independent original CPU FP32 oracle: video_negative in
        // cvvdp_reference.inc, tests/make_cvvdp_oracles.py artifact 6.
        CHECK_NEAR(result->score, -4.7883110046386719, 1e-3);
    }
    pipeline.Reset();
    distorted.Write(red);
    pipeline.Push(original.View(), distorted.View());
    // Finalize of one deferred frame uses the true single-image algorithm.
    distorted.Write(blue);
    CHECK_NEAR(pipeline.Finalize()->score, 6.1910214424133301, 1e-3);
}

MetricsConfig UnifiedConfig(const Image& input, bool mixed) {
    MetricsConfig config;
    config.size = input.size;
    config.original_format = input.format;
    config.distorted_format = input.format;
    config.cvvdp = CvvdpConfig{.size = input.size};
    if (mixed) {
        config.ssimulacra2 = Ssimulacra2Config{.size = input.size};
        config.vmaf_v0 = VmafConfig{.size = input.size};
    }
    return config;
}

void CheckMetadataOnly(const MetricsResult& frame, std::uint64_t index) {
    CHECK(frame.frame_index == index);
    CHECK(!frame.cvvdp);
    CHECK(!frame.ssimulacra2);
    CHECK(!frame.butteraugli);
    CHECK(!frame.vmaf_v0);
    CHECK(!frame.vmaf_v1);
}

}  // namespace

VQM_TEST(TestCvvdpHostViewsAndBorrowedSourceSwitching) {
    if (!vqm::test::GpuAvailable()) return;
    const auto context = AcquireVulkanContext();
    CvvdpConfig config;
    config.size = kSize;
    config.temporal = false;
    const auto requirements = GetMemoryRequirements(config, Borrow(*context));
    if (requirements.direct.supported)
        CheckStillSources(MemoryPolicy::Direct, *context);
    if (requirements.staged.supported)
        CheckStillSources(MemoryPolicy::Staged, *context);
    CHECK(requirements.direct.supported || requirements.staged.supported);
}

VQM_TEST(TestCvvdpBorrowedTemporalHistoryAndSignals) {
    if (!vqm::test::GpuAvailable()) return;
    const auto context = AcquireVulkanContext();
    CvvdpConfig config;
    config.size = kSize;
    const auto requirements = GetMemoryRequirements(config, Borrow(*context));
    if (requirements.direct.supported)
        CheckTemporalReuse(MemoryPolicy::Direct, *context);
    if (requirements.staged.supported)
        CheckTemporalReuse(MemoryPolicy::Staged, *context);
    CHECK(requirements.direct.supported || requirements.staged.supported);
}

VQM_TEST(TestCvvdpUnifiedClipAndStillSemantics) {
    if (!vqm::test::GpuAvailable()) return;
    const auto green = SolidChannel(1), red = SolidChannel(0),
               blue = SolidChannel(2);
    Metrics metrics(UnifiedConfig(green, false));
    const auto empty = metrics.Finalize();
    CHECK(empty.frames == 0 && !empty.cvvdp && !empty.last_frame);
    metrics.Reset();
    for (std::uint64_t index = 0; index < 3; ++index) {
        const auto frame = metrics.Compute(green, index % 2 ? blue : red);
        CHECK(frame.has_value());
        if (frame) CheckMetadataOnly(*frame, index);
    }
    const auto final = metrics.Finalize();
    CHECK(final.frames == 3 && !final.last_frame && final.cvvdp);
    CHECK(final.cvvdp && final.cvvdp->frames == 3);
    const auto again = metrics.Finalize();
    CHECK(!again.last_frame);
    CHECK(
        again.cvvdp && final.cvvdp && again.cvvdp->score == final.cvvdp->score
    );
    CHECK(!metrics.Flush());
    CHECK_THROWS_AS(metrics.Compute(green, red), std::logic_error);
    metrics.Reset();
    const auto still = metrics.ComputeImage(green, red);
    CHECK(still.cvvdp.has_value());
    if (still.cvvdp) CHECK_NEAR(*still.cvvdp, 6.1910214424133301, 1e-3);
    CHECK(metrics.ComputeImage(green, green).cvvdp == 10.0);
    const auto reset_summary = metrics.Finalize();
    CHECK(reset_summary.frames == 0 && !reset_summary.cvvdp);
}

VQM_TEST(TestCvvdpUnifiedMixedMetricTail) {
    if (!vqm::test::GpuAvailable()) return;
    const auto a = vqm::test::MakeGradient(0), b = vqm::test::MakeGradient(1);
    auto config = UnifiedConfig(a, true);
    Metrics metrics(config);
    CvvdpPipeline reference(*config.cvvdp);
    VmafPipeline vmaf(*config.vmaf_v0);
    std::vector<double> expected_vmaf;
    std::vector<MetricsResult> emitted;
    for (std::uint64_t index = 0; index < 3; ++index) {
        reference.Push(a, b);
        if (const auto ready = vmaf.Push(a, b))
            expected_vmaf.push_back(ready->score);
        if (const auto ready = metrics.Compute(a, b)) emitted.push_back(*ready);
    }
    if (const auto tail = vmaf.Flush()) expected_vmaf.push_back(tail->score);
    const auto final = metrics.Finalize();
    CHECK(final.last_frame.has_value());
    if (final.last_frame) emitted.push_back(*final.last_frame);
    CHECK(emitted.size() == 3 && expected_vmaf.size() == 3);
    for (std::size_t index = 0; index < emitted.size(); ++index) {
        CHECK(emitted[index].frame_index == index);
        CHECK(!emitted[index].cvvdp);
        CHECK(emitted[index].ssimulacra2.has_value());
        CHECK(emitted[index].vmaf_v0.has_value());
        if (emitted[index].vmaf_v0 && index < expected_vmaf.size())
            CHECK_NEAR(*emitted[index].vmaf_v0, expected_vmaf[index], 1e-7);
    }
    CHECK(final.frames == 3 && final.cvvdp);
    if (final.cvvdp)
        CHECK_NEAR(final.cvvdp->score, reference.Finalize()->score, 1e-7);
    CHECK(!metrics.Finalize().last_frame);
    CHECK(!metrics.Flush());
    metrics.Reset();
    const auto still = metrics.ComputeImage(a, b);
    CHECK(
        still.frame_index == 0 && still.cvvdp && still.vmaf_v0 &&
        still.ssimulacra2
    );
    reference.Reset();
    if (still.cvvdp)
        CHECK_NEAR(*still.cvvdp, reference.ComputeImage(a, b).score, 1e-7);
}

VQM_TEST(TestCvvdpUnifiedBorrowedSemaphoreRouting) {
    if (!vqm::test::GpuAvailable()) return;
    const auto context = AcquireVulkanContext();
    const auto a = vqm::test::MakeGradient(0), b = vqm::test::MakeGradient(1);
    CallerBuffer original(*context, a), distorted(*context, b);
    auto config = UnifiedConfig(a, true);
    config.streaming_inputs = false;
    Metrics metrics(config, Borrow(*context));
    Semaphores semaphores(*context);
    const auto ready = semaphores.Make(), consumed = semaphores.Make();
    const std::array waits{VulkanSemaphoreWait{ready}};
    const std::array signals{consumed};
    semaphores.Signal(ready);
    const auto still = metrics.ComputeImage(
        original.View(), distorted.View(), {waits, signals}
    );
    semaphores.Consume(consumed);
    CHECK(still.cvvdp && still.vmaf_v0 && still.ssimulacra2);
    CvvdpPipeline standalone({.size = a.size, .temporal = false});
    if (still.cvvdp)
        CHECK_NEAR(*still.cvvdp, standalone.ComputeImage(a, b).score, 1e-7);
    semaphores.Signal(ready);
    const auto frame =
        metrics.Compute(original.View(), distorted.View(), {waits, signals});
    semaphores.Consume(consumed);
    CHECK(frame.has_value());
    if (frame) CHECK(!frame->cvvdp);
    const auto summary = metrics.Finalize();
    CHECK(summary.frames == 1 && summary.cvvdp && !summary.last_frame);
    if (still.cvvdp && summary.cvvdp)
        CHECK_NEAR(*still.cvvdp, summary.cvvdp->score, 1e-7);
}

VQM_TEST(TestCvvdpDisabledStreamingRejectsHostInputs) {
    if (!vqm::test::GpuAvailable()) return;
    const auto a = vqm::test::MakeGradient(0), b = vqm::test::MakeGradient(1);
    PaddedHost original(a), distorted(b);
    CvvdpConfig config{.size = a.size, .streaming_inputs = false};
    const auto requirements = GetMemoryRequirements(config);
    CvvdpPipeline reference({.size = a.size});
    const auto expected = reference.ComputeImage(a, b).score;
    for (const auto policy : {MemoryPolicy::Direct, MemoryPolicy::Staged}) {
        const auto& path = policy == MemoryPolicy::Direct ? requirements.direct
                                                          : requirements.staged;
        if (!path.supported) continue;
        config.memory_policy = policy;
        CvvdpPipeline pipeline(config);
        const auto resident_a = pipeline.Upload(a),
                   resident_b = pipeline.Upload(b);
        CHECK_THROWS_AS(pipeline.ComputeImage(a, b), std::invalid_argument);
        CHECK_THROWS_AS(
            pipeline.ComputeImage(original.View(), distorted.View()),
            std::invalid_argument
        );
        CHECK_THROWS_AS(pipeline.Push(a, b), std::invalid_argument);
        CHECK_THROWS_AS(
            pipeline.Push(original.View(), distorted.View()),
            std::invalid_argument
        );
        // Rejected host writes must neither corrupt immutable tables nor
        // advance the clip, and both memory policies retain the resident path.
        CHECK_NEAR(
            pipeline.ComputeImage(resident_a, resident_b).score, expected, 1e-7
        );
        pipeline.Push(resident_a, resident_b);
        CHECK(pipeline.Finalize()->frames == 1);
    }
}
