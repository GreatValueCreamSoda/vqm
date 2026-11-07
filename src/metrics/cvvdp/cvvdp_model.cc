// Original pooling and temporal filter model: Graphics and Displays Group,
// 2024, MIT. See pycvvdp/cvvdp_metric.py at the pinned v0.5.7 revision.
#include "cvvdp_internal.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "cvvdp_model_data.inc"

namespace vqm::internal {
std::size_t CvvdpTemporalTaps(const CvvdpConfig& config) {
    if (config.fps_numerator == 0 || config.fps_denominator == 0)
        throw std::invalid_argument("cvvdp: frame rate must be positive");
    const double fps =
        static_cast<double>(config.fps_numerator) / config.fps_denominator;
    // Bound setup work and history addressing before a quadratic inverse DFT.
    if (fps < 1.0 || fps > 240.0)
        throw std::invalid_argument("cvvdp: supported frame rate is 1..240 fps");
    return static_cast<std::size_t>(2.0 * std::ceil(.25 * fps / 2.0) + 1.0);
}

std::vector<float> CvvdpTemporalFilters(const CvvdpConfig& config) {
    const auto count = CvvdpTemporalTaps(config);
    const auto frequencies = count / 2 + 1;
    const float fps = static_cast<float>(
        static_cast<double>(config.fps_numerator) / config.fps_denominator
    );
    std::vector<float> filters(4 * count);
    // Original inclusive linspace(0,fps/2,M), odd-length irfft(backward),
    // fftshift. Initialization only; no FFT library is part of the core.
    // The initialization-only inverse DFT mechanism is also present in the
    // read-only Vship src/Vulkan/cvvdp/temporalFilter.hpp:7-28,47-79 at
    // 91d03ec2. This is our implementation of the original's inclusive grid
    // and odd-length transform; no peer implementation code is copied.
    for (std::size_t ch = 0; ch < 4; ++ch) {
        std::vector<float> response(frequencies);
        for (std::size_t k = 0; k < frequencies; ++k) {
            const float omega = (fps / 2) * static_cast<float>(k) /
                                static_cast<float>(frequencies - 1);
            const float power = std::pow(omega, cvvdp_data::beta_tf[ch]);
            response[k] =
                ch == 3
                    ? std::exp(
                          -std::pow(
                              power - std::pow(5.0f, cvvdp_data::beta_tf[ch]),
                              2.0f
                          ) /
                          cvvdp_data::sigma_tf[ch]
                      )
                    : std::exp(-power / cvvdp_data::sigma_tf[ch]);
        }
        for (std::size_t n = 0; n < count; ++n) {
            double value = response[0];
            for (std::size_t k = 1; k < frequencies; ++k)
                value += 2.0 * response[k] *
                         std::cos(
                             2.0 * std::acos(-1.0) * static_cast<double>(k * n) /
                             static_cast<double>(count)
                         );
            filters[ch * count + (n + count / 2) % count] =
                static_cast<float>(value / static_cast<double>(count));
        }
    }
    return filters;
}

double CvvdpSafePow(double value, double exponent) {
    return std::pow(value + 1e-5, exponent) - std::pow(1e-5, exponent);
}

double CvvdpFrameDistortion(std::span<const float> bands, std::size_t channels) {
    if ((channels != 3 && channels != 4) || bands.empty() ||
        bands.size() % channels != 0)
        throw std::invalid_argument(
            "cvvdp: invalid channel/band distortion array"
        );
    const auto count = bands.size() / channels;
    double channel_power = 0.0;
    for (std::size_t ch = 0; ch < channels; ++ch) {
        double band_power = 0.0;
        for (std::size_t band = 0; band < count; ++band) {
            const double input = bands[ch * count + band];
            if (!std::isfinite(input) || input < 0)
                throw std::runtime_error(
                    "cvvdp: non-finite or negative GPU distortion"
                );
            const double channel_weight = ch == 3 ? cvvdp_data::ch_trans_w : 1.0;
            const double band_weight =
                band + 1 == count ? cvvdp_data::baseband_weight[ch] : 1.0;
            band_power +=
                CvvdpSafePow(input * channel_weight * band_weight, 4.0);
        }
        channel_power += CvvdpSafePow(CvvdpSafePow(band_power, .25), 4.0);
    }
    return CvvdpSafePow(channel_power, .25);
}

double CvvdpJod(double distortion) {
    const double exponent = cvvdp_data::jod_exp;
    return 10.0 -
           cvvdp_data::jod_a * (distortion <= .1
                                    ? std::pow(.1, exponent - 1.0) * distortion
                                    : std::pow(distortion, exponent));
}
double CvvdpImageJod(std::span<const float> bands) {
    return CvvdpJod(CvvdpFrameDistortion(bands, 3) * cvvdp_data::image_int);
}
}  // namespace vqm::internal
