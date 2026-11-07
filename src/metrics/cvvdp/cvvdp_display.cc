// Original photometry/matrices: Graphics and Displays Group (2024), MIT.
// See pycvvdp/display_model.py and color_spaces.json at the pinned revision.
#include "cvvdp_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "cvvdp_model_data.inc"
#include "input.h"

namespace vqm {
CvvdpDisplay CvvdpDisplay::Standard4K() { return {}; }
CvvdpDisplay CvvdpDisplay::StandardFhd() {
    CvvdpDisplay out;
    out.pixels_per_degree = internal::cvvdp_data::standard_fhd_ppd;
    return out;
}
CvvdpDisplay CvvdpDisplay::StandardHdr() {
    CvvdpDisplay out;
    out.peak_luminance = 1500.0;
    out.contrast = 1000000.0;
    out.ambient_illuminance = 10.0;
    return out;
}
}  // namespace vqm

namespace vqm::internal {
namespace {
std::uint32_t TransferKind(TransferCharacteristics transfer) {
    switch (transfer) {
        case TransferCharacteristics::Srgb:
        case TransferCharacteristics::Bt709:
        case TransferCharacteristics::Bt601:
        case TransferCharacteristics::Bt2020_10:
        case TransferCharacteristics::Bt2020_12:
            return 0;
        case TransferCharacteristics::Linear:
            return 1;
        case TransferCharacteristics::St2084:
            return 2;
        case TransferCharacteristics::AribB67:
            return 3;
        default:
            throw std::invalid_argument("cvvdp: unsupported transfer function");
    }
}

const std::array<float, 9>& DklMatrix(ColorPrimaries primaries) {
    switch (primaries) {
        case ColorPrimaries::Bt709:
            return cvvdp_data::rgb709_to_dkl;
        case ColorPrimaries::Bt2020:
            return cvvdp_data::rgb2020_to_dkl;
        case ColorPrimaries::St432_1:
            return cvvdp_data::p3_to_dkl;
        default:
            throw std::invalid_argument("cvvdp: unsupported color primaries");
    }
}

struct PreparePush {
    std::uint32_t desc_base, output, width, height, transfer;
    float peak, black, reflected, exposure, gamma, background;
    std::array<float, 9> matrix;
    std::uint32_t status;
};
static_assert(sizeof(PreparePush) <= 128);
}  // namespace

void ValidateCvvdpDisplay(const CvvdpDisplay& display) {
    const std::array values{display.peak_luminance,      display.contrast,
                            display.ambient_illuminance, display.reflectivity,
                            display.pixels_per_degree,   display.exposure};
    if (!std::all_of(
            values.begin(), values.end(),
            [](double value) { return std::isfinite(value) && value <= 1e6; }
        ) ||
        display.peak_luminance < 0.005 || display.contrast < 1.0 ||
        display.ambient_illuminance < 0.0 || display.reflectivity < 0.0 ||
        display.reflectivity > 1.0 || display.pixels_per_degree < 1.0 ||
        display.exposure <= 0.0)
        throw std::invalid_argument(
            "cvvdp: invalid physical display parameters"
        );
}

void ValidateCvvdpFormat(const ImageFormat& format) {
    (void)TransferKind(format.transfer);
    (void)DklMatrix(format.primaries);
    if (format.alpha != AlphaType::None || SelectMode(format) != OpMode::Matrix)
        throw std::invalid_argument(
            "cvvdp: input must be opaque RGB, grey or NCL YUV"
        );
}

void ValidateCvvdpInput(const ImageFormat& format, const CvvdpDisplay& display) {
    ValidateCvvdpFormat(format);
    if (format.transfer == TransferCharacteristics::AribB67 &&
        display.peak_luminance > 1000.0 && display.ambient_illuminance == 0.0)
        throw std::invalid_argument(
            "cvvdp: HDR HLG requires positive ambient illuminance above 1000 "
            "nits"
        );
}

CvvdpPrepare::CvvdpPrepare(
    VulkanRuntime& runtime, const CvvdpDisplay& display, ChromaFilter filter
)
    : runtime_(runtime),
      display_(display),
      filter_(filter),
      pipelines_(runtime.CreatePipelines(
          std::array{"cvvdp_prepare_0", "cvvdp_prepare_1", "cvvdp_prepare_2"}
      )) {}
CvvdpPrepare::~CvvdpPrepare() { runtime_.DestroyPipelines(pipelines_); }

void CvvdpPrepare::Record(
    std::size_t image, std::uint32_t descriptor, std::uint32_t output, Size size,
    const ImageFormat& format, std::uint32_t status
) {
    ValidateCvvdpInput(format, display_);
    const double gamma =
        display_.peak_luminance > 1000.0 && display_.ambient_illuminance > 0
            ? 1.2 + .42 * std::log10(display_.peak_luminance / 1000.0) -
                  .07623 * std::log10(display_.ambient_illuminance / 5.0)
            : 1.2;
    const PreparePush push{
        descriptor,
        output,
        static_cast<std::uint32_t>(size.width),
        static_cast<std::uint32_t>(size.height),
        TransferKind(format.transfer),
        static_cast<float>(display_.peak_luminance),
        static_cast<float>(display_.peak_luminance / display_.contrast),
        static_cast<float>(
            display_.ambient_illuminance * display_.reflectivity /
            std::acos(-1.0)
        ),
        static_cast<float>(display_.exposure),
        static_cast<float>(gamma),
        0.0f,
        DklMatrix(format.primaries),
        status};
    runtime_.BindSource(image);
    runtime_.Dispatch(
        pipelines_[static_cast<std::size_t>(filter_)], &push, sizeof(push),
        Groups(size.width, 32), Groups(size.height, 16), 1
    );
}
}  // namespace vqm::internal
