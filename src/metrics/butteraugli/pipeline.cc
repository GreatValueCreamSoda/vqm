// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.

// The public Butteraugli facade: geometry setup, the resident pool and the
// handle contracts live here (mirroring src/metrics/ssimulacra2/pipeline.cc),
// the metric itself in src/metrics/butteraugli/tables.cc and
// src/metrics/butteraugli/vulkan.cc.

#include "butteraugli.h"
#include "vulkan.h"

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "metrics/butteraugli/butteraugli_internal.h"
#include "handles.h"
#include "input.h"

namespace vqm {

struct ButteraugliPipeline::Impl {
    explicit Impl(
        const ButteraugliConfig& config, const VulkanDevice* device = nullptr
    );

    UploadedImage Upload(const Image& image);
    void Release(UploadedImage image);
    std::uint64_t Submit(
        const UploadedImage& original, const UploadedImage& distorted,
        float background
    );
    std::uint64_t Submit(
        const UploadedImage& original, const Image& distorted, float background
    );
    [[nodiscard]] internal::ButteraugliSource ResidentSource(
        const UploadedImage& image, const char* name
    ) const;

    Size size;
    std::unique_ptr<internal::ButteraugliGpuPipeline> gpu;
    internal::ResidentTable resident;
};

namespace {

ButteraugliScore Convert(const internal::ButteraugliResult& result) {
    return ButteraugliScore{result.score, result.distance_p3};
}

std::uint64_t Generation(std::uint64_t slot) {
    return slot >> internal::kResidentIndexBits;
}

}  // namespace

ButteraugliPipeline::Impl::Impl(
    const ButteraugliConfig& config, const VulkanDevice* device
)
    : size(config.size) {
    internal::ValidateButteraugliConfig(config);
    resident.Resize(config.gpu_resident_images);
    const internal::ButteraugliGeometry geometry =
        internal::PlanButteraugliGeometry(config.size);
    internal::ButteraugliParams params;
    params.hf_asymmetry = config.hf_asymmetry;
    params.xmul = config.xmul;
    params.intensity_target = config.intensity_target;
    gpu = internal::CreateVulkanButteraugliPipeline(
        geometry, params, config.gpu_resident_images, config.chroma_filter,
        config.streaming_inputs, config.input_bytes, config.memory_policy, device
    );
    if (!gpu) throw std::runtime_error("vulkan: no usable compute device");
}

UploadedImage ButteraugliPipeline::Impl::Upload(const Image& image) {
    gpu->RequireIdle();
    internal::CheckImage(image, size, "image");
    return resident.Upload(image.size, [this, &image](std::size_t capacity) {
        return gpu->Upload(image, capacity);
    });
}

void ButteraugliPipeline::Impl::Release(UploadedImage image) {
    gpu->RequireIdle();
    resident.Release(image, [this](std::uint32_t slot) {
        gpu->ReleaseSlot(slot);
    });
}

internal::ButteraugliSource ButteraugliPipeline::Impl::ResidentSource(
    const UploadedImage& image, const char* name
) const {
    internal::ButteraugliSource source;
    source.kind = internal::ButteraugliSource::Kind::Resident;
    source.index = resident.Require(image, size, name).device;
    source.generation = Generation(image.slot);
    return source;
}

std::uint64_t ButteraugliPipeline::Impl::Submit(
    const UploadedImage& original, const UploadedImage& distorted,
    float background
) {
    gpu->RequireIdle();
    return gpu->Submit(
        ResidentSource(original, "original"),
        ResidentSource(distorted, "distorted"), background
    );
}

std::uint64_t ButteraugliPipeline::Impl::Submit(
    const UploadedImage& original, const Image& distorted, float background
) {
    gpu->RequireIdle();
    return gpu->Submit(
        ResidentSource(original, "original"), distorted, background
    );
}

ButteraugliPipeline::ButteraugliPipeline(ButteraugliConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

ButteraugliPipeline::ButteraugliPipeline(
    ButteraugliConfig config, const VulkanDevice& device
)
    : impl_(std::make_unique<Impl>(config, &device)) {}

ButteraugliPipeline::~ButteraugliPipeline() = default;
ButteraugliPipeline::ButteraugliPipeline(ButteraugliPipeline&&) noexcept =
    default;
ButteraugliPipeline& ButteraugliPipeline::operator=(
    ButteraugliPipeline&&
) noexcept = default;

PipelineMemoryUsage ButteraugliPipeline::GetMemoryUsage() const {
    return impl_->gpu->GetMemoryUsage();
}

ButteraugliScore ButteraugliPipeline::Compute(
    const Image& original, const Image& distorted, float background
) {
    return Finish(Submit(original, distorted, background));
}

ButteraugliScore ButteraugliPipeline::Compute(
    const ImageView& original, const ImageView& distorted, float background
) {
    return Finish(Submit(original, distorted, background));
}

UploadedImage ButteraugliPipeline::Upload(const Image& image) {
    return impl_->Upload(image);
}

void ButteraugliPipeline::Release(UploadedImage image) { impl_->Release(image); }

ButteraugliScore ButteraugliPipeline::Compute(
    const UploadedImage& original, const UploadedImage& distorted,
    float background
) {
    return Finish(Submit(original, distorted, background));
}

ButteraugliScore ButteraugliPipeline::Compute(
    const UploadedImage& original, const Image& distorted, float background
) {
    return Finish(Submit(original, distorted, background));
}

std::uint64_t ButteraugliPipeline::Submit(
    const Image& original, const Image& distorted, float background
) {
    return impl_->gpu->Submit(original, distorted, background);
}

std::uint64_t ButteraugliPipeline::Submit(
    const ImageView& original, const ImageView& distorted, float background
) {
    return impl_->gpu->Submit(original, distorted, background);
}

std::uint64_t ButteraugliPipeline::Submit(
    const UploadedImage& original, const UploadedImage& distorted,
    float background
) {
    return impl_->Submit(original, distorted, background);
}

std::uint64_t ButteraugliPipeline::Submit(
    const UploadedImage& original, const Image& distorted, float background
) {
    return impl_->Submit(original, distorted, background);
}

ButteraugliScore ButteraugliPipeline::Finish(std::uint64_t token) {
    return Convert(impl_->gpu->Finish(token));
}

ImageBuffer ButteraugliPipeline::AllocateImage(const ImageFormat& format) {
    return impl_->gpu->AllocateImage(format);
}

std::byte* ButteraugliPipeline::MapImage(ImageBuffer buffer) {
    return impl_->gpu->MapImage(buffer);
}

void ButteraugliPipeline::Release(ImageBuffer buffer) {
    impl_->gpu->Release(buffer);
}

ButteraugliScore ButteraugliPipeline::Compute(
    const ImageBuffer& original, const ImageBuffer& distorted, float background
) {
    return Finish(Submit(original, distorted, background));
}

std::uint64_t ButteraugliPipeline::Submit(
    const ImageBuffer& original, const ImageBuffer& distorted, float background
) {
    return impl_->gpu->Submit(original, distorted, background);
}

ButteraugliScore ButteraugliPipeline::Compute(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    float background
) {
    return Finish(Submit(original, distorted, background));
}

std::uint64_t ButteraugliPipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    float background
) {
    return impl_->gpu->Submit(original, distorted, background);
}

std::uint64_t ButteraugliPipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    const VulkanSubmitSync& sync, float background
) {
    return impl_->gpu->Submit(original, distorted, sync, background);
}

}  // namespace vqm
