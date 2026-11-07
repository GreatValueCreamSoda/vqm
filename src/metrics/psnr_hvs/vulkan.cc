// Copyright 2001-2012 Xiph.Org and contributors.
// BSD-2-Clause; see licenses/XIPH-PSNR-HVS-LICENSE.
// The metric follows libvmaf's Xiph implementation at revision 4611e47,
// libvmaf/src/feature/third_party/xiph/psnr_hvs.c. GPU pass boundaries and
// parallel error reduction are VQM's; no coefficient images are retained.

#include "metrics/psnr_hvs/vulkan.h"

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "input.h"
#include "metrics/psnr_hvs/layout.h"
#include "metrics/psnr_hvs/tables.h"
#include "timing.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"

namespace vqm::internal {
namespace {

using Source = VulkanRuntime::Source;
using Push = psnr_hvs_layout::Push;

void ValidatePair(const ImageFormat& a, const ImageFormat& b, Size size) {
    ValidatePsnrHvsFormat(a, size);
    ValidatePsnrHvsFormat(b, size);
    if (!SameImageFormat(a, b))
        throw std::invalid_argument("psnr_hvs: input formats must match");
}

float Decibels(float normalized_error) {
    return -10.0f * std::log10(normalized_error);
}

}  // namespace

struct PsnrHvsGpuPipeline::Impl {
    PsnrHvsConfig config;
    std::shared_ptr<VulkanContext> context;
    psnr_hvs_layout::Layout layout;
    std::unique_ptr<VulkanRuntime> runtime;
    std::array<VkPipeline, 3> pipelines{};
    std::vector<ImageFormat> resident_formats;
    std::unordered_map<std::uint64_t, ImageFormat> input_formats;
    std::array<std::uint32_t, 3> block_counts{};
    std::uint32_t peak = 255;

    Impl(const PsnrHvsConfig& value, const VulkanDevice* device)
        : config(value),
          context(RequireVulkanContext(device)),
          layout(psnr_hvs_layout::BuildLayout(config, *context)),
          resident_formats(config.gpu_resident_images) {
        VulkanRuntime::Config runtime_config;
        runtime_config.stream_region = layout.stream_region;
        runtime_config.stream_desc = layout.stream_desc;
        runtime_config.resident_slots = layout.resident;
        runtime_config.input_pool = layout.input_pool;
        runtime_config.input_pool_size = layout.input_pool_size;
        runtime_config.staging_desc_offset = layout.staging_desc;
        runtime_config.staging_image_bytes = layout.staging.image_bytes;
        runtime_config.base_pixels = config.size.width * config.size.height;
        runtime_config.base_size = config.size;
        runtime_config.streaming = config.streaming_inputs;
        runtime_config.memory_policy = config.memory_policy;
        runtime_config.push_constant_size = sizeof(Push);
        runtime = std::make_unique<VulkanRuntime>(context, runtime_config);
        runtime->CreateArena(layout.bytes);
        if (!runtime->arena_visible())
            runtime->CreateStaging(Bytes(layout.staging_words));
        runtime->CreateCommandObjects();
        runtime->CreateDescriptors(0, layout.bytes);
        InitializeTables();
        pipelines = runtime->CreatePipelines(
            std::array{"psnr_hvs_prepare", "psnr_hvs_blocks", "psnr_hvs_reduce"}
        );
    }

    void InitializeTables() {
        static_assert(kPsnrHvsTables.size() == psnr_hvs_layout::kTableWords);
        constexpr std::size_t bytes = kPsnrHvsTables.size() * sizeof(float);
        if (runtime->arena_visible()) {
            // The first PublishSources supplies host-to-compute visibility;
            // these immutable metadata words never overlap input uploads.
            std::memcpy(
                runtime->arena_map() + layout.tables, kPsnrHvsTables.data(),
                bytes
            );
            return;
        }
        std::memcpy(
            runtime->staging_map() + layout.staging_tables,
            kPsnrHvsTables.data(), bytes
        );
        runtime->PublishStaged(
            layout.tables, layout.staging_tables, psnr_hvs_layout::kTableWords
        );
    }

    ~Impl() {
        if (runtime->pending()) {
            try {
                runtime->WaitFence();
            } catch (...) {
                // Device loss must not turn destruction into an exception.
            }
        }
        runtime->DestroyPipelines(pipelines);
    }

    void RequireStreaming() const {
        runtime->RequireIdle();
        if (!config.streaming_inputs)
            throw std::invalid_argument("psnr_hvs: streaming_inputs is false");
    }

    void RecordPrepare(
        const std::array<Source, 2>& sources, const ImageFormat& format
    ) {
        runtime->OpenStage(Stage::PsnrHvsPrepare);
        (void)runtime->PublishSources(sources);
        for (std::size_t image = 0; image < 2; ++image) {
            runtime->BindSource(image);
            for (std::uint32_t plane = 0; plane < 3; ++plane) {
                Push push;
                push.desc_base = layout.stream_desc[image];
                push.plane = plane;
                push.width = static_cast<std::uint32_t>(
                    config.size.width >> (plane == 0 ? 0 : format.subsample_w)
                );
                push.height = static_cast<std::uint32_t>(
                    config.size.height >> (plane == 0 ? 0 : format.subsample_h)
                );
                push.destination =
                    layout.prepared[image] + plane * layout.plane_stride;
                runtime->Dispatch(
                    pipelines[0], &push, sizeof(push), Groups(push.width, 16),
                    Groups(push.height, 8), 1
                );
            }
        }
        runtime->CloseStage();
        runtime->ComputeBarrier();
    }

    void RecordBlocks(const ImageFormat& format) {
        runtime->OpenStage(Stage::PsnrHvsBlocks);
        for (std::uint32_t plane = 0; plane < 3; ++plane) {
            Push push;
            push.plane = plane;
            push.tables = layout.tables;
            push.width = static_cast<std::uint32_t>(
                config.size.width >> (plane == 0 ? 0 : format.subsample_w)
            );
            push.height = static_cast<std::uint32_t>(
                config.size.height >> (plane == 0 ? 0 : format.subsample_h)
            );
            push.reference = layout.prepared[0] + plane * layout.plane_stride;
            push.distorted = layout.prepared[1] + plane * layout.plane_stride;
            push.destination = layout.partials[plane];
            push.blocks_x = (push.width - 8u) / 7u + 1u;
            const auto blocks_y = (push.height - 8u) / 7u + 1u;
            block_counts[plane] = push.blocks_x * blocks_y;
            runtime->Dispatch(
                pipelines[1], &push, sizeof(push), push.blocks_x, blocks_y, 1
            );
        }
        runtime->CloseStage();
        runtime->ComputeBarrier();
    }

    void RecordReduce() {
        runtime->OpenStage(Stage::PsnrHvsReduce);
        std::array<std::uint32_t, 3> groups{};
        bool second_pass = false;
        for (std::uint32_t plane = 0; plane < 3; ++plane) {
            groups[plane] = Groups(
                block_counts[plane], psnr_hvs_layout::kReduceItemsPerGroup
            );
            second_pass = second_pass || groups[plane] > 1;
            Push push;
            push.reference = layout.partials[plane];
            push.destination = groups[plane] > 1 ? layout.reduction[plane]
                                                 : layout.result + plane;
            push.blocks = block_counts[plane];
            runtime->Dispatch(
                pipelines[2], &push, sizeof(push), groups[plane], 1, 1
            );
        }
        if (second_pass) {
            runtime->ComputeBarrier();
            for (std::uint32_t plane = 0; plane < 3; ++plane) {
                if (groups[plane] <= 1) continue;
                Push push;
                push.reference = layout.reduction[plane];
                push.destination = layout.result + plane;
                push.blocks = groups[plane];
                runtime->Dispatch(pipelines[2], &push, sizeof(push), 1, 1, 1);
            }
        }
        runtime->CloseStage();
        if (runtime->arena_visible())
            runtime->ComputeToHostBarrier();
        else
            runtime->RecordReadback(layout.result, layout.staging_result, 3);
    }

    std::uint64_t SubmitSources(
        const std::array<Source, 2>& sources, const ImageFormat& a,
        const ImageFormat& b, const VulkanSubmitSync* sync = nullptr
    ) {
        runtime->RequireIdle();
        ValidatePair(a, b, config.size);
        runtime->PrepareSourceBindings(sources);
        peak = (1u << EffectiveDepth(a)) - 1u;
        const auto token = runtime->BeginSubmission();
        runtime->BeginCommandBuffer();
        runtime->BindDescriptorSet();
        RecordPrepare(sources, a);
        RecordBlocks(a);
        RecordReduce();
        runtime->EndCommandBuffer();
        if (sync != nullptr)
            runtime->QueueSubmit(*sync);
        else
            runtime->QueueSubmit();
        return token;
    }

    std::uint64_t SubmitExternal(
        const VulkanBufferView& original, const VulkanBufferView& distorted,
        const VulkanSubmitSync* sync
    ) {
        runtime->RequireIdle();
        if (context->owns_device)
            throw std::invalid_argument(
                "psnr_hvs: external buffers require a caller-owned device"
            );
        ValidatePair(original.format, distorted.format, config.size);
        if (sync != nullptr) VulkanRuntime::ValidateSubmitSync(*sync);
        return SubmitSources(
            runtime->ResolveExternalInputs(original, distorted), original.format,
            distorted.format, sync
        );
    }
};

PsnrHvsGpuPipeline::PsnrHvsGpuPipeline(
    const PsnrHvsConfig& config, const VulkanDevice* device
)
    : impl_(std::make_unique<Impl>(config, device)) {}

PsnrHvsGpuPipeline::~PsnrHvsGpuPipeline() = default;

void PsnrHvsGpuPipeline::RequireIdle() const { impl_->runtime->RequireIdle(); }

PipelineMemoryUsage PsnrHvsGpuPipeline::GetMemoryUsage() const {
    return impl_->runtime->GetMemoryUsage();
}

std::optional<std::uint32_t> PsnrHvsGpuPipeline::Upload(
    const Image& image, std::size_t capacity
) {
    RequireIdle();
    CheckImage(image, impl_->config.size, "psnr_hvs");
    ValidatePsnrHvsFormat(image.format, impl_->config.size);
    const auto slot = impl_->runtime->UploadResident(image, capacity);
    if (slot) impl_->resident_formats.at(*slot) = image.format;
    return slot;
}

void PsnrHvsGpuPipeline::Release(std::uint32_t slot) {
    impl_->runtime->ReleaseResident(slot);
}

ImageBuffer PsnrHvsGpuPipeline::AllocateImage(const ImageFormat& format) {
    RequireIdle();
    ValidatePsnrHvsFormat(format, impl_->config.size);
    const auto image = impl_->runtime->AllocateImage(format);
    try {
        impl_->input_formats.emplace(image.slot, format);
    } catch (...) {
        impl_->runtime->ReleaseInput(image);
        throw;
    }
    return image;
}

std::byte* PsnrHvsGpuPipeline::MapImage(ImageBuffer image) {
    return impl_->runtime->MapImage(image);
}

void PsnrHvsGpuPipeline::Release(ImageBuffer image) {
    impl_->runtime->ReleaseInput(image);
    impl_->input_formats.erase(image.slot);
}

std::uint64_t PsnrHvsGpuPipeline::Submit(
    const Image& original, const Image& distorted
) {
    impl_->RequireStreaming();
    CheckImage(original, impl_->config.size, "psnr_hvs original");
    CheckImage(distorted, impl_->config.size, "psnr_hvs distorted");
    ValidatePair(original.format, distorted.format, impl_->config.size);
    impl_->runtime->UploadStreaming(original, distorted);
    return impl_->SubmitSources({}, original.format, distorted.format);
}

std::uint64_t PsnrHvsGpuPipeline::Submit(
    const ImageView& original, const ImageView& distorted
) {
    impl_->RequireStreaming();
    ValidatePair(original.format, distorted.format, impl_->config.size);
    impl_->runtime->UploadStreaming(original, distorted);
    return impl_->SubmitSources({}, original.format, distorted.format);
}

std::uint64_t PsnrHvsGpuPipeline::Submit(
    std::uint32_t original, std::uint32_t distorted
) {
    RequireIdle();
    return impl_->SubmitSources(
        {Source{Source::Kind::Resident, original},
         Source{Source::Kind::Resident, distorted}},
        impl_->resident_formats.at(original),
        impl_->resident_formats.at(distorted)
    );
}

std::uint64_t PsnrHvsGpuPipeline::Submit(
    std::uint32_t original, const Image& distorted
) {
    impl_->RequireStreaming();
    CheckImage(distorted, impl_->config.size, "psnr_hvs distorted");
    const auto& format = impl_->resident_formats.at(original);
    ValidatePair(format, distorted.format, impl_->config.size);
    impl_->runtime->UploadStreamingDistorted(distorted);
    return impl_->SubmitSources(
        {Source{Source::Kind::Resident, original}, Source{}}, format,
        distorted.format
    );
}

std::uint64_t PsnrHvsGpuPipeline::Submit(
    const ImageBuffer& original, const ImageBuffer& distorted
) {
    const auto sources = impl_->runtime->ResolveInputs(original, distorted);
    return impl_->SubmitSources(
        sources, impl_->input_formats.at(original.slot),
        impl_->input_formats.at(distorted.slot)
    );
}

std::uint64_t PsnrHvsGpuPipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted
) {
    return impl_->SubmitExternal(original, distorted, nullptr);
}

std::uint64_t PsnrHvsGpuPipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    const VulkanSubmitSync& sync
) {
    return impl_->SubmitExternal(original, distorted, &sync);
}

PsnrHvsScore PsnrHvsGpuPipeline::Finish(std::uint64_t token) {
    impl_->runtime->FinishSubmission(token);
    VQM_TIMED(PsnrHvsScore);
    const float* sums =
        impl_->runtime->arena_visible()
            ? impl_->runtime->arena_map() + impl_->layout.result
            : impl_->runtime->staging_map() + impl_->layout.staging_result;
    PsnrHvsScore result;
    std::array<float, 3> errors{};
    const float peak = static_cast<float>(impl_->peak);
    for (std::size_t plane = 0; plane < 3; ++plane) {
        float error =
            sums[plane] / static_cast<float>(64u * impl_->block_counts[plane]);
        error /= peak * peak;
        errors[plane] = error;
        result.normalized_error[plane] = error;
    }
    result.y = Decibels(errors[0]);
    result.cb = Decibels(errors[1]);
    result.cr = Decibels(errors[2]);
    result.score = Decibels(0.8f * errors[0] + 0.1f * (errors[1] + errors[2]));
    return result;
}

}  // namespace vqm::internal
