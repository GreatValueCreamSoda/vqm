#pragma once

#include "vqm_metrics.h"

namespace vqm {

struct MetricPipelines::Impl {
    Impl(MetricsConfig config, const VulkanDevice* device, bool share_single);
    [[nodiscard]] MetricsMemoryUsage GetMemoryUsage() const;

    MetricsConfig config;
    std::shared_ptr<internal::SharedMetricStorage> scratch;
    std::unique_ptr<Ssimulacra2Pipeline> ssimu;
    std::unique_ptr<ButteraugliPipeline> butter;
    std::unique_ptr<VmafPipeline> v0;
    std::unique_ptr<VmafV1Pipeline> v1;
    std::unique_ptr<CvvdpPipeline> cvvdp;
    std::unique_ptr<PsnrHvsPipeline> psnr_hvs;
};

}  // namespace vqm
