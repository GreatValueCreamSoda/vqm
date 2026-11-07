// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <cmath>
#include <limits>
#include <stdexcept>

#include "support.h"
#include "vqm.h"

VQM_TEST(TestScoreRejectsNonfiniteNorms) {
    constexpr double infinity = std::numeric_limits<double>::infinity();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    for (double invalid : {infinity, -infinity, nan}) {
        vqm::Ssimulacra2Result result;
        result.scales.resize(6);
        // Every public norm must be checked, including zero-weight entries:
        // they must not turn invalid readback into a perfect score.
        for (auto& scale : result.scales) {
            for (auto& norm : scale.avg_ssim) {
                norm = invalid;
                CHECK_THROWS_AS(result.Score(), std::invalid_argument);
                norm = 0.0;
            }
            for (auto& norm : scale.avg_edgediff) {
                norm = invalid;
                CHECK_THROWS_AS(result.Score(), std::invalid_argument);
                norm = 0.0;
            }
        }
    }
}

VQM_TEST(TestScoreOverflowAndRange) {
    vqm::Ssimulacra2Result result;
    CHECK(result.Score() == 100.0);
    result.scales.resize(6);
    CHECK(result.Score() == 100.0);
    result.scales[0].avg_ssim[3] = std::numeric_limits<double>::max();
    CHECK_THROWS_AS(result.Score(), std::invalid_argument);
    result.scales[0].avg_ssim[3] = 1e200;
    CHECK_THROWS_AS(result.Score(), std::invalid_argument);
    result.scales[0].avg_ssim[3] = 100.0;
    CHECK(std::isfinite(result.Score()));
    CHECK(result.Score() < 0.0);
}
