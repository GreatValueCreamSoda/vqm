#include "metric_pipelines_internal.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "image_view.h"
#include "input.h"
#include "metrics/cvvdp/cvvdp_internal.h"
#include "metrics/metric_scratch.h"
#include "metrics/psnr_hvs/layout.h"
#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "metrics/vmaf/features/vmaf_hdr.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"

namespace vqm {
namespace {
struct SelectedConfigs {
    std::optional<Ssimulacra2Config> ssimulacra2;
    std::optional<ButteraugliConfig> butteraugli;
    std::optional<VmafConfig> vmaf_v0;
    std::optional<VmafV1Config> vmaf_v1;
    std::optional<CvvdpConfig> cvvdp;
    std::optional<PsnrHvsConfig> psnr_hvs;
};

SelectedConfigs SelectConfigs(const MetricsConfig& config) {
    if (!config.ssimulacra2 && !config.butteraugli && !config.vmaf_v0 &&
        !config.vmaf_v1 && !config.cvvdp && !config.psnr_hvs)
        throw std::invalid_argument("metrics: select at least one metric");
    if (!std::isfinite(config.background) || config.background < 0.0f ||
        config.background > 1.0f)
        throw std::invalid_argument(
            "metrics: background must be between 0 and 1"
        );
    internal::ValidateImage(
        config.original_format, config.size, "original format"
    );
    internal::ValidateImage(
        config.distorted_format, config.size, "distorted format"
    );
    const bool hdr =
        config.vmaf_v1 && config.vmaf_v1->model == VmafV1Model::Hdr4K;
    if (hdr) {
        internal::ValidateVmafHdrFormat(config.original_format);
        internal::ValidateVmafHdrFormat(config.distorted_format);
    }
    if (config.vmaf_v0 || (config.vmaf_v1 && !hdr)) {
        internal::ValidateVmafSdrFormat(config.original_format);
        internal::ValidateVmafSdrFormat(config.distorted_format);
    }
    if (config.cvvdp) {
        internal::ValidateCvvdpInput(
            config.original_format, config.cvvdp->display
        );
        internal::ValidateCvvdpInput(
            config.distorted_format, config.cvvdp->display
        );
    }
    if (config.psnr_hvs) {
        internal::ValidatePsnrHvsFormat(config.original_format, config.size);
        internal::ValidatePsnrHvsFormat(config.distorted_format, config.size);
        if (!internal::SameImageFormat(
                config.original_format, config.distorted_format
            ))
            throw std::invalid_argument(
                "metrics: PSNR-HVS requires matching native input formats"
            );
    }

    SelectedConfigs result;
    result.ssimulacra2 = config.ssimulacra2;
    result.butteraugli = config.butteraugli;
    result.vmaf_v0 = config.vmaf_v0;
    result.vmaf_v1 = config.vmaf_v1;
    result.cvvdp = config.cvvdp;
    result.psnr_hvs = config.psnr_hvs;
    // Unified evaluation accepts streamed or borrowed inputs. Resident pools
    // belong to the individual metric APIs.
    const auto configure_inputs = [&](auto& metric) {
        metric.size = config.size;
        metric.streaming_inputs = config.streaming_inputs;
        metric.memory_policy = config.memory_policy;
        metric.gpu_resident_images = 0;
    };
    if (result.ssimulacra2) {
        auto& metric = *result.ssimulacra2;
        configure_inputs(metric);
        metric.input_bytes = 0;
    }
    if (result.butteraugli) {
        auto& metric = *result.butteraugli;
        configure_inputs(metric);
        metric.input_bytes = 0;
    }
    if (result.vmaf_v0) {
        auto& metric = *result.vmaf_v0;
        configure_inputs(metric);
        metric.background = config.background;
    }
    if (result.vmaf_v1) {
        auto& metric = *result.vmaf_v1;
        configure_inputs(metric);
        metric.input_format = config.original_format;
        metric.background = config.background;
        if (!hdr && metric.chroma_sampling == VmafChromaSampling::Native &&
            !internal::SameImageFormat(
                config.original_format, config.distorted_format
            ))
            throw std::invalid_argument(
                "metrics: VMAF v1 Native requires matching input formats"
            );
    }
    if (result.cvvdp) configure_inputs(*result.cvvdp);
    if (result.psnr_hvs) {
        configure_inputs(*result.psnr_hvs);
        result.psnr_hvs->input_bytes = 0;
    }
    return result;
}

template <class Config>
MetricMemoryRequirements Requirements(
    const Config& config, const VulkanDevice* device
) {
    return device ? GetMemoryRequirements(config, *device)
                  : GetMemoryRequirements(config);
}

struct PlannedMetrics {
    MetricsMemoryRequirements requirements;
    internal::MetricScratchBuilder scratch;
};

PlannedMetrics PlanMemory(
    const SelectedConfigs& selected, MemoryPolicy policy,
    const VulkanDevice* device
) {
    PlannedMetrics result;
    internal::BufferMemoryPlan spatial;
    const auto add_spatial = [&](const auto& config) {
        if (!config) return;
        const auto memory = Requirements(*config, device);
        result.scratch.Add(memory);
        spatial.auxiliary_bytes.insert(
            spatial.auxiliary_bytes.end(),
            memory.scratch.persistent_bytes.begin(),
            memory.scratch.persistent_bytes.end()
        );
    };
    add_spatial(selected.ssimulacra2);
    add_spatial(selected.butteraugli);
    if (selected.ssimulacra2 || selected.butteraugli)
        result.requirements.spatial =
            internal::QueryMemoryPlan(spatial, policy, device);
    const auto add = [&](const auto& config, auto& destination) {
        if (!config) return;
        const auto memory = Requirements(*config, device);
        result.scratch.Add(memory);
        destination =
            internal::PrivateMemoryRequirements(memory, policy, device);
    };
    add(selected.vmaf_v0, result.requirements.vmaf_v0);
    add(selected.vmaf_v1, result.requirements.vmaf_v1);
    add(selected.cvvdp, result.requirements.cvvdp);
    add(selected.psnr_hvs, result.requirements.psnr_hvs);
    result.requirements.scratch =
        internal::QueryMemoryPlan(result.scratch.MemoryPlan(), policy, device);
    return result;
}

template <class Pipeline, class Config>
std::unique_ptr<Pipeline> MakePipeline(
    const Config& config, const VulkanDevice* device,
    const std::shared_ptr<internal::SharedMetricStorage>& scratch
) {
    return internal::MetricScratchAccess::Create<Pipeline>(
        config, device, scratch
    );
}

}  // namespace

MetricsMemoryRequirements GetMemoryRequirements(const MetricsConfig& config) {
    return PlanMemory(SelectConfigs(config), config.memory_policy, nullptr)
        .requirements;
}
MetricsMemoryRequirements GetMemoryRequirements(
    const MetricsConfig& config, const VulkanDevice& device
) {
    return PlanMemory(SelectConfigs(config), config.memory_policy, &device)
        .requirements;
}
std::size_t MaxWorkersForBudgetWorstCase(
    const MetricsMemoryRequirements& requirements,
    std::span<const std::uint64_t> heap_budgets
) {
    std::vector<MetricMemoryRequirements> components;
    components.reserve(6);
    if (requirements.scratch) components.push_back(*requirements.scratch);
    if (requirements.spatial) components.push_back(*requirements.spatial);
    if (requirements.vmaf_v0) components.push_back(*requirements.vmaf_v0);
    if (requirements.vmaf_v1) components.push_back(*requirements.vmaf_v1);
    if (requirements.cvvdp) components.push_back(*requirements.cvvdp);
    if (requirements.psnr_hvs) components.push_back(*requirements.psnr_hvs);
    return MaxWorkersForBudgetWorstCase(
        std::span<const MetricMemoryRequirements>(components), heap_budgets
    );
}

MetricPipelines::Impl::Impl(
    MetricsConfig pipeline_config, const VulkanDevice* device, bool share_single
)
    : config(std::move(pipeline_config)) {
    const auto selected = SelectConfigs(config);
    const auto count = static_cast<unsigned>(!!selected.ssimulacra2) +
                       static_cast<unsigned>(!!selected.butteraugli) +
                       static_cast<unsigned>(!!selected.vmaf_v0) +
                       static_cast<unsigned>(!!selected.vmaf_v1) +
                       static_cast<unsigned>(!!selected.cvvdp) +
                       static_cast<unsigned>(!!selected.psnr_hvs);
    if (share_single || count > 1) {
        const auto memory = PlanMemory(selected, config.memory_policy, device);
        scratch = memory.scratch.Build(device, config.memory_policy);
    }
    if (selected.ssimulacra2)
        ssimu = MakePipeline<Ssimulacra2Pipeline>(
            *selected.ssimulacra2, device, scratch
        );
    if (selected.butteraugli)
        butter = MakePipeline<ButteraugliPipeline>(
            *selected.butteraugli, device, scratch
        );
    if (selected.vmaf_v0)
        v0 = MakePipeline<VmafPipeline>(*selected.vmaf_v0, device, scratch);
    if (selected.vmaf_v1)
        v1 = MakePipeline<VmafV1Pipeline>(*selected.vmaf_v1, device, scratch);
    if (selected.cvvdp)
        cvvdp = MakePipeline<CvvdpPipeline>(*selected.cvvdp, device, scratch);
    if (selected.psnr_hvs)
        psnr_hvs =
            MakePipeline<PsnrHvsPipeline>(*selected.psnr_hvs, device, scratch);
}

MetricsMemoryUsage MetricPipelines::Impl::GetMemoryUsage() const {
    MetricsMemoryUsage result;
    if (scratch) result.scratch = scratch->backing().GetMemoryUsage();
    if (ssimu) result.spatial = ssimu->GetMemoryUsage();
    if (butter) {
        if (result.spatial)
            internal::AppendPrivateMemory(
                *result.spatial, butter->GetMemoryUsage()
            );
        else
            result.spatial = butter->GetMemoryUsage();
    }
    if (v0) result.vmaf_v0 = v0->GetMemoryUsage();
    if (v1) result.vmaf_v1 = v1->GetMemoryUsage();
    if (cvvdp) result.cvvdp = cvvdp->GetMemoryUsage();
    if (psnr_hvs) result.psnr_hvs = psnr_hvs->GetMemoryUsage();
    return result;
}

MetricPipelines::MetricPipelines(MetricsConfig config)
    : impl_(std::make_unique<Impl>(std::move(config), nullptr, false)) {}
MetricPipelines::MetricPipelines(
    MetricsConfig config, const VulkanDevice& device
)
    : impl_(std::make_unique<Impl>(std::move(config), &device, false)) {}
MetricPipelines::~MetricPipelines() = default;
MetricPipelines::MetricPipelines(MetricPipelines&&) noexcept = default;
MetricPipelines& MetricPipelines::operator=(MetricPipelines&&) noexcept =
    default;
Ssimulacra2Pipeline* MetricPipelines::Ssimulacra2() noexcept {
    return impl_->ssimu.get();
}
ButteraugliPipeline* MetricPipelines::Butteraugli() noexcept {
    return impl_->butter.get();
}
VmafPipeline* MetricPipelines::VmafV0() noexcept { return impl_->v0.get(); }
VmafV1Pipeline* MetricPipelines::VmafV1() noexcept { return impl_->v1.get(); }
CvvdpPipeline* MetricPipelines::Cvvdp() noexcept { return impl_->cvvdp.get(); }
PsnrHvsPipeline* MetricPipelines::PsnrHvs() noexcept {
    return impl_->psnr_hvs.get();
}
MetricsMemoryUsage MetricPipelines::GetMemoryUsage() const {
    return impl_->GetMemoryUsage();
}
}  // namespace vqm
