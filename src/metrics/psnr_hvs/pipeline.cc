// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include "psnr_hvs.h"

#include <memory>

#include "handles.h"
#include "input.h"
#include "vulkan.h"

namespace vqm {

struct PsnrHvsPipeline::Impl {
    explicit Impl(
        const PsnrHvsConfig& config, const VulkanDevice* device,
        std::shared_ptr<internal::SharedMetricStorage> scratch = {}
    )
        : size(config.size),
          gpu(std::make_unique<internal::PsnrHvsGpuPipeline>(
              config, device, std::move(scratch)
          )) {
        resident.Resize(config.gpu_resident_images);
    }

    Size size;
    std::unique_ptr<internal::PsnrHvsGpuPipeline> gpu;
    internal::ResidentTable resident;
};

PsnrHvsPipeline::PsnrHvsPipeline(PsnrHvsConfig config)
    : impl_(std::make_unique<Impl>(config, nullptr)) {}

PsnrHvsPipeline::PsnrHvsPipeline(
    PsnrHvsConfig config, const VulkanDevice& device
)
    : impl_(std::make_unique<Impl>(config, &device)) {}

PsnrHvsPipeline::~PsnrHvsPipeline() = default;
PsnrHvsPipeline::PsnrHvsPipeline(
    PsnrHvsConfig config, const VulkanDevice* device,
    std::shared_ptr<internal::SharedMetricStorage> scratch
)
    : impl_(std::make_unique<Impl>(config, device, std::move(scratch))) {}
PsnrHvsPipeline::PsnrHvsPipeline(PsnrHvsPipeline&&) noexcept = default;
PsnrHvsPipeline& PsnrHvsPipeline::operator=(PsnrHvsPipeline&&) noexcept =
    default;

PsnrHvsScore PsnrHvsPipeline::Compute(
    const Image& original, const Image& distorted
) {
    return Finish(Submit(original, distorted));
}

PsnrHvsScore PsnrHvsPipeline::Compute(
    const ImageView& original, const ImageView& distorted
) {
    return Finish(Submit(original, distorted));
}

UploadedImage PsnrHvsPipeline::Upload(const Image& image) {
    impl_->gpu->RequireIdle();
    internal::CheckImage(image, impl_->size, "image");
    return impl_->resident.Upload(
        image.size, [this, &image](std::size_t capacity) {
            return impl_->gpu->Upload(image, capacity);
        }
    );
}

void PsnrHvsPipeline::Release(UploadedImage image) {
    impl_->gpu->RequireIdle();
    impl_->resident.Release(image, [this](std::uint32_t slot) {
        impl_->gpu->Release(slot);
    });
}

PsnrHvsScore PsnrHvsPipeline::Compute(
    const UploadedImage& original, const UploadedImage& distorted
) {
    return Finish(Submit(original, distorted));
}

PsnrHvsScore PsnrHvsPipeline::Compute(
    const UploadedImage& original, const Image& distorted
) {
    return Finish(Submit(original, distorted));
}

std::uint64_t PsnrHvsPipeline::Submit(
    const Image& original, const Image& distorted
) {
    return impl_->gpu->Submit(original, distorted);
}

std::uint64_t PsnrHvsPipeline::Submit(
    const ImageView& original, const ImageView& distorted
) {
    return impl_->gpu->Submit(original, distorted);
}

std::uint64_t PsnrHvsPipeline::Submit(
    const UploadedImage& original, const UploadedImage& distorted
) {
    impl_->gpu->RequireIdle();
    const auto& reference =
        impl_->resident.Require(original, impl_->size, "original");
    const auto& comparison =
        impl_->resident.Require(distorted, impl_->size, "distorted");
    return impl_->gpu->Submit(reference.device, comparison.device);
}

std::uint64_t PsnrHvsPipeline::Submit(
    const UploadedImage& original, const Image& distorted
) {
    impl_->gpu->RequireIdle();
    const auto& reference =
        impl_->resident.Require(original, impl_->size, "original");
    return impl_->gpu->Submit(reference.device, distorted);
}

ImageBuffer PsnrHvsPipeline::AllocateImage(const ImageFormat& format) {
    return impl_->gpu->AllocateImage(format);
}

std::byte* PsnrHvsPipeline::MapImage(ImageBuffer buffer) {
    return impl_->gpu->MapImage(buffer);
}

void PsnrHvsPipeline::Release(ImageBuffer buffer) {
    impl_->gpu->Release(buffer);
}

PsnrHvsScore PsnrHvsPipeline::Compute(
    const ImageBuffer& original, const ImageBuffer& distorted
) {
    return Finish(Submit(original, distorted));
}

std::uint64_t PsnrHvsPipeline::Submit(
    const ImageBuffer& original, const ImageBuffer& distorted
) {
    return impl_->gpu->Submit(original, distorted);
}

PsnrHvsScore PsnrHvsPipeline::Compute(
    const VulkanBufferView& original, const VulkanBufferView& distorted
) {
    return Finish(Submit(original, distorted));
}

std::uint64_t PsnrHvsPipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted
) {
    return impl_->gpu->Submit(original, distorted);
}

std::uint64_t PsnrHvsPipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    const VulkanSubmitSync& sync
) {
    return impl_->gpu->Submit(original, distorted, sync);
}

PsnrHvsScore PsnrHvsPipeline::Finish(std::uint64_t token) {
    return impl_->gpu->Finish(token);
}

PipelineMemoryUsage PsnrHvsPipeline::GetMemoryUsage() const {
    return impl_->gpu->GetMemoryUsage();
}

}  // namespace vqm
