#include <array>
#include <type_traits>
#include <stdexcept>

#include "butteraugli.h"
#include "cvvdp.h"
#include "vmaf.h"
#include "vmaf_v1.h"
#include "vqm.h"
#include "vqm_alpha.h"
#include "vqm_memory.h"
#include "vqm_metrics.h"
#include "vqm_shared.h"
#include "vqm_pool.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"

#ifdef VQM_PROBE_VULKAN_SDK
#include "vqm_sdk_probe.h"
#endif

static_assert(
    std::is_constructible_v<vqm::Ssimulacra2Pipeline, vqm::Ssimulacra2Config>
);
static_assert(
    std::is_constructible_v<vqm::ButteraugliPipeline, vqm::ButteraugliConfig>
);
static_assert(std::is_constructible_v<vqm::VmafPipeline, vqm::VmafConfig>);
static_assert(std::is_constructible_v<vqm::VmafV1Pipeline, vqm::VmafV1Config>);
static_assert(std::is_constructible_v<vqm::Metrics, vqm::MetricsConfig>);
static_assert(std::is_constructible_v<vqm::CvvdpPipeline, vqm::CvvdpConfig>);
static_assert(
    std::is_constructible_v<vqm::SharedMetricWorkspace, vqm::SharedMetricConfig>
);

int main() {
    try {
        vqm::VmafV1Pipeline invalid({.size = {1, 1}});
        return 1;
    } catch (const std::invalid_argument&) {
        // Exercises installed V1 symbols without requiring a GPU.
    }
    const auto display = vqm::CvvdpDisplay::StandardFhd();
    if (display.pixels_per_degree < 37.0) return 1;
    vqm::ImageFormat format;
    format.type = vqm::PixelType::Byte;
    vqm::MemoryScenario scenario;
    scenario.supported = true;
    scenario.arena.buffer_bytes = 100;
    scenario.arena.allocation_bytes = 100;
    const std::array<std::uint64_t, 1> budgets{250};
    return vqm::RequiredBytes(format, {8, 8}) == 192 &&
                   vqm::MaxWorkersForBudget(scenario, budgets) == 2
               ? 0
               : 1;
}
