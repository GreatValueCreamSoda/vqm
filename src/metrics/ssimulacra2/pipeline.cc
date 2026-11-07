// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "handles.h"
#include "input.h"
#include "metrics/ssimulacra2/ssimulacra2_internal.h"
#include "ssimulacra2.h"
#include "vulkan.h"

// The metric pipeline: decoded samples -> XYB -> 2x downsample -> Gaussian
// blur -> error maps -> norms, all of it on the Vulkan compute device. The
// weighted score mapping is Ssimulacra2Result::Score() in score.cc; input may
// be any declared transfer function and primaries, decoded on the GPU.

namespace vqm::internal {

Ssimulacra2ScalePlan PlanSsimulacra2Scales(Size size) {
    if (size.width < kSsimulacra2MinDimension ||
        size.height < kSsimulacra2MinDimension) {
        throw std::invalid_argument("minimum image size is 8x8 pixels");
    }
    // The previous level controls whether the next level is emitted, so the
    // last scale can itself be below 8 pixels.
    Ssimulacra2ScalePlan result;
    auto width = size.width;
    auto height = size.height;
    while (result.count < kSsimulacra2MaxScales &&
           width >= kSsimulacra2MinDimension &&
           height >= kSsimulacra2MinDimension) {
        if (result.count != 0) {
            width = width / 2 + width % 2;
            height = height / 2 + height % 2;
        }
        result.sizes[result.count++] = {width, height};
    }
    return result;
}

}  // namespace vqm::internal

namespace vqm {

struct Ssimulacra2Pipeline::Impl {
    explicit Impl(
        const Ssimulacra2Config& config, const VulkanDevice* device = nullptr
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

    Size size;
    std::unique_ptr<internal::Ssimulacra2GpuPipeline> gpu;

    // Resident image pool, one entry per Ssimulacra2Config::gpu_resident_images.
    internal::ResidentTable resident;
};

Ssimulacra2Pipeline::Impl::Impl(
    const Ssimulacra2Config& config, const VulkanDevice* device
)
    : size(config.size) {
    // Construct the kernel first to preserve initialization failure ordering.
    const auto kernel =
        internal::MakeGaussian(internal::kSsimulacra2GaussianSigma);
    const auto scales = internal::PlanSsimulacra2Scales(config.size);

    if (config.gpu_resident_images > internal::kResidentIndexMask) {
        throw std::invalid_argument(
            "gpu_resident_images is limited to " +
            std::to_string(internal::kResidentIndexMask) + " slots"
        );
    }
    resident.Resize(config.gpu_resident_images);
    gpu = internal::CreateVulkanSsimulacra2Pipeline(
        {scales.count, scales.sizes.data()}, kernel, config.gpu_resident_images,
        config.half_planes, config.chroma_filter, config.streaming_inputs,
        config.input_bytes, config.memory_policy, device
    );
    if (!gpu) throw std::runtime_error("vulkan: no usable compute device");
}

UploadedImage Ssimulacra2Pipeline::Impl::Upload(const Image& image) {
    gpu->RequireIdle();
    internal::CheckImage(image, size, "image");
    return resident.Upload(image.size, [this, &image](std::size_t capacity) {
        return gpu->Upload(image, capacity);
    });
}

void Ssimulacra2Pipeline::Impl::Release(UploadedImage image) {
    gpu->RequireIdle();
    resident.Release(image, [this](std::uint32_t slot) { gpu->Release(slot); });
}

std::uint64_t Ssimulacra2Pipeline::Impl::Submit(
    const UploadedImage& original, const UploadedImage& distorted,
    float background
) {
    gpu->RequireIdle();
    const internal::ResidentTable::Slot& reference_slot =
        resident.Require(original, size, "original");
    const internal::ResidentTable::Slot& distorted_slot =
        resident.Require(distorted, size, "distorted");
    return gpu->Submit(reference_slot.device, distorted_slot.device, background);
}

std::uint64_t Ssimulacra2Pipeline::Impl::Submit(
    const UploadedImage& original, const Image& distorted, float background
) {
    gpu->RequireIdle();
    const internal::ResidentTable::Slot& reference_slot =
        resident.Require(original, size, "original");
    return gpu->Submit(reference_slot.device, distorted, background);
}

Ssimulacra2Pipeline::Ssimulacra2Pipeline(Ssimulacra2Config config)
    : impl_(std::make_unique<Impl>(config)) {}
Ssimulacra2Pipeline::Ssimulacra2Pipeline(
    Ssimulacra2Config config, const VulkanDevice& device
)
    : impl_(std::make_unique<Impl>(config, &device)) {}

Ssimulacra2Pipeline::~Ssimulacra2Pipeline() = default;
Ssimulacra2Pipeline::Ssimulacra2Pipeline(Ssimulacra2Pipeline&&) noexcept =
    default;
Ssimulacra2Pipeline& Ssimulacra2Pipeline::operator=(
    Ssimulacra2Pipeline&&
) noexcept = default;

PipelineMemoryUsage Ssimulacra2Pipeline::GetMemoryUsage() const {
    return impl_->gpu->GetMemoryUsage();
}

Ssimulacra2Result Ssimulacra2Pipeline::Compute(
    const Image& original, const Image& distorted, float background
) {
    return Finish(Submit(original, distorted, background));
}

Ssimulacra2Result Ssimulacra2Pipeline::Compute(
    const ImageView& original, const ImageView& distorted, float background
) {
    return Finish(Submit(original, distorted, background));
}

UploadedImage Ssimulacra2Pipeline::Upload(const Image& image) {
    return impl_->Upload(image);
}

void Ssimulacra2Pipeline::Release(UploadedImage image) { impl_->Release(image); }

Ssimulacra2Result Ssimulacra2Pipeline::Compute(
    const UploadedImage& original, const UploadedImage& distorted,
    float background
) {
    return Finish(Submit(original, distorted, background));
}

Ssimulacra2Result Ssimulacra2Pipeline::Compute(
    const UploadedImage& original, const Image& distorted, float background
) {
    return Finish(Submit(original, distorted, background));
}

std::uint64_t Ssimulacra2Pipeline::Submit(
    const Image& original, const Image& distorted, float background
) {
    return impl_->gpu->Submit(original, distorted, background);
}

std::uint64_t Ssimulacra2Pipeline::Submit(
    const ImageView& original, const ImageView& distorted, float background
) {
    return impl_->gpu->Submit(original, distorted, background);
}

std::uint64_t Ssimulacra2Pipeline::Submit(
    const UploadedImage& original, const UploadedImage& distorted,
    float background
) {
    return impl_->Submit(original, distorted, background);
}

std::uint64_t Ssimulacra2Pipeline::Submit(
    const UploadedImage& original, const Image& distorted, float background
) {
    return impl_->Submit(original, distorted, background);
}

Ssimulacra2Result Ssimulacra2Pipeline::Finish(std::uint64_t token) {
    return impl_->gpu->Finish(token);
}

ImageBuffer Ssimulacra2Pipeline::AllocateImage(const ImageFormat& format) {
    return impl_->gpu->AllocateImage(format);
}

std::byte* Ssimulacra2Pipeline::MapImage(ImageBuffer buffer) {
    return impl_->gpu->MapImage(buffer);
}

void Ssimulacra2Pipeline::Release(ImageBuffer buffer) {
    impl_->gpu->Release(buffer);
}

Ssimulacra2Result Ssimulacra2Pipeline::Compute(
    const ImageBuffer& original, const ImageBuffer& distorted, float background
) {
    return Finish(Submit(original, distorted, background));
}

std::uint64_t Ssimulacra2Pipeline::Submit(
    const ImageBuffer& original, const ImageBuffer& distorted, float background
) {
    return impl_->gpu->Submit(original, distorted, background);
}

Ssimulacra2Result Ssimulacra2Pipeline::Compute(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    float background
) {
    return Finish(Submit(original, distorted, background));
}

std::uint64_t Ssimulacra2Pipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    float background
) {
    return impl_->gpu->Submit(original, distorted, background);
}

std::uint64_t Ssimulacra2Pipeline::Submit(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    const VulkanSubmitSync& sync, float background
) {
    return impl_->gpu->Submit(original, distorted, sync, background);
}

}  // namespace vqm

namespace vqm::internal {

GaussianKernel MakeGaussian(double sigma) {
    GaussianKernel kernel;
    kernel.radius =
        static_cast<int>(std::ceil(kSsimulacra2GaussianSupport * sigma));
    const std::size_t taps = 2 * static_cast<std::size_t>(kernel.radius) + 1;
    std::vector<double> weights(taps);
    double sum = 0.0;
    for (std::size_t i = 0; i < taps; ++i) {
        const double offset = static_cast<double>(i) - kernel.radius;
        weights[i] = std::exp(-0.5 * (offset / sigma) * (offset / sigma));
        sum += weights[i];
    }
    kernel.weights.resize(taps);
    for (std::size_t i = 0; i < taps; ++i) {
        kernel.weights[i] = static_cast<float>(weights[i] / sum);
    }
    return kernel;
}

}  // namespace vqm::internal
