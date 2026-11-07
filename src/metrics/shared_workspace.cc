// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "vqm_shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "metrics/butteraugli/butteraugli_internal.h"
#include "metrics/ssimulacra2/ssimulacra2_internal.h"
#include "metrics/butteraugli/vulkan.h"
#include "metrics/shared_workspace_internal.h"
#include "metrics/ssimulacra2/layout.h"
#include "metrics/butteraugli/layout.h"
#include "metrics/ssimulacra2/vulkan.h"
#include "vulkan/vulkan_shared_storage.h"
#include "vulkan/vulkan_support.h"

namespace vqm {
namespace {
using namespace internal;

struct SharedConfigs {
    Ssimulacra2Config ssimu;
    ButteraugliConfig butter;
};

SharedConfigs MakeSharedConfigs(const SharedMetricConfig& config) {
    SharedConfigs result;
    result.ssimu.size = config.size;
    result.ssimu.gpu_resident_images = 0;
    result.ssimu.streaming_inputs = config.streaming_inputs;
    result.ssimu.half_planes = config.half_planes;
    result.ssimu.chroma_filter = config.chroma_filter;
    result.ssimu.memory_policy = config.memory_policy;
    result.butter.size = config.size;
    result.butter.gpu_resident_images = 0;
    result.butter.streaming_inputs = config.streaming_inputs;
    result.butter.chroma_filter =
        config.butter_chroma_filter.value_or(config.chroma_filter);
    result.butter.hf_asymmetry = config.hf_asymmetry;
    result.butter.xmul = config.xmul;
    result.butter.intensity_target = config.intensity_target;
    result.butter.memory_policy = config.memory_policy;
    return result;
}
}  // namespace

namespace internal {
BufferMemoryPlan PlanSharedMetricMemory(
    const SharedMetricConfig& config, const VulkanContext& context
) {
    const auto configs = MakeSharedConfigs(config);
    const auto a = PlanSsimulacra2Memory(configs.ssimu, context);
    const auto b = PlanButteraugliMemory(configs.butter, context);
    return {std::max(a.arena_bytes, b.arena_bytes),
            std::max(a.fallback_staging_bytes, b.fallback_staging_bytes)};
}
}  // namespace internal

struct SharedMetricWorkspace::Impl {
    enum class ActiveMetric { Ssimulacra2, Butteraugli };

    explicit Impl(const SharedMetricConfig& config, const VulkanDevice* device) {
        if (config.size.width < kSsimulacra2MinDimension ||
            config.size.height < kSsimulacra2MinDimension)
            throw std::invalid_argument("minimum image size is 8x8 pixels");
        if (!std::isfinite(config.hf_asymmetry) || config.hf_asymmetry <= 0 ||
            !std::isfinite(config.intensity_target) ||
            config.intensity_target <= 0 || !std::isfinite(config.xmul) ||
            config.xmul < 0)
            throw std::invalid_argument("invalid Butteraugli configuration");
        context = RequireVulkanContext(device);

        const auto configs = MakeSharedConfigs(config);
        const auto& a = configs.ssimu;
        const auto& b = configs.butter;
        const auto plan = PlanSharedMetricMemory(config, *context);
        storage = std::make_shared<SharedMetricStorage>(
            context, plan.arena_bytes, plan.fallback_staging_bytes,
            config.memory_policy
        );

        const auto scales = PlanSsimulacra2Scales(config.size);
        ssimu = CreateSharedVulkanSsimulacra2Pipeline(
            {scales.count, scales.sizes.data()},
            MakeGaussian(kSsimulacra2GaussianSigma), a, context, storage
        );
        const ButteraugliParams params{b.hf_asymmetry, b.xmul,
                                       b.intensity_target};
        butter = CreateSharedVulkanButteraugliPipeline(
            PlanButteraugliGeometry(config.size), params, b, context, storage
        );
    }

    void Activate(ActiveMetric selected) {
        ssimu->RequireIdle();
        butter->RequireIdle();
        if (active == selected) return;
        // Both metrics publish overlapping tables. A failed activation must
        // leave neither metric cached as active, so a retry republishes them.
        active.reset();
        if (selected == ActiveMetric::Ssimulacra2)
            ssimu->ActivateShared();
        else
            butter->ActivateShared();
        active = selected;
    }
    template <class... Args>
    Ssimulacra2Result ComputeSsimu(Args&&... args) {
        Activate(ActiveMetric::Ssimulacra2);
        const auto token = ssimu->Submit(std::forward<Args>(args)...);
        return ssimu->Finish(token);
    }
    template <class... Args>
    ButteraugliScore ComputeButter(Args&&... args) {
        Activate(ActiveMetric::Butteraugli);
        const auto token = butter->Submit(std::forward<Args>(args)...);
        const auto result = butter->Finish(token);
        return {result.score, result.distance_p3};
    }

    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<SharedMetricStorage> storage;
    std::unique_ptr<Ssimulacra2GpuPipeline> ssimu;
    std::unique_ptr<ButteraugliGpuPipeline> butter;
    std::optional<ActiveMetric> active;
};

SharedMetricWorkspace::SharedMetricWorkspace(SharedMetricConfig config)
    : impl_(std::make_unique<Impl>(config, nullptr)) {}
SharedMetricWorkspace::SharedMetricWorkspace(
    SharedMetricConfig config, const VulkanDevice& device
)
    : impl_(std::make_unique<Impl>(config, &device)) {}
SharedMetricWorkspace::~SharedMetricWorkspace() = default;
SharedMetricWorkspace::SharedMetricWorkspace(SharedMetricWorkspace&&) noexcept =
    default;
SharedMetricWorkspace& SharedMetricWorkspace::operator=(
    SharedMetricWorkspace&&
) noexcept = default;

Ssimulacra2Result SharedMetricWorkspace::ComputeSsimulacra2(
    const Image& a, const Image& b, float background
) {
    return impl_->ComputeSsimu(a, b, background);
}
Ssimulacra2Result SharedMetricWorkspace::ComputeSsimulacra2(
    const ImageView& a, const ImageView& b, float background
) {
    return impl_->ComputeSsimu(a, b, background);
}
Ssimulacra2Result SharedMetricWorkspace::ComputeSsimulacra2(
    const VulkanBufferView& a, const VulkanBufferView& b, float background
) {
    return impl_->ComputeSsimu(a, b, background);
}
Ssimulacra2Result SharedMetricWorkspace::ComputeSsimulacra2(
    const VulkanBufferView& a, const VulkanBufferView& b,
    const VulkanSubmitSync& sync, float background
) {
    return impl_->ComputeSsimu(a, b, sync, background);
}
ButteraugliScore SharedMetricWorkspace::ComputeButteraugli(
    const Image& a, const Image& b, float background
) {
    return impl_->ComputeButter(a, b, background);
}
ButteraugliScore SharedMetricWorkspace::ComputeButteraugli(
    const ImageView& a, const ImageView& b, float background
) {
    return impl_->ComputeButter(a, b, background);
}
ButteraugliScore SharedMetricWorkspace::ComputeButteraugli(
    const VulkanBufferView& a, const VulkanBufferView& b, float background
) {
    return impl_->ComputeButter(a, b, background);
}
ButteraugliScore SharedMetricWorkspace::ComputeButteraugli(
    const VulkanBufferView& a, const VulkanBufferView& b,
    const VulkanSubmitSync& sync, float background
) {
    return impl_->ComputeButter(a, b, sync, background);
}
PipelineMemoryUsage SharedMetricWorkspace::GetMemoryUsage() const {
    return impl_->storage->backing().GetMemoryUsage();
}
}  // namespace vqm
