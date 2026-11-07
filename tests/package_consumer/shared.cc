// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include <cstddef>

#include "vqm_image.h"

extern "C" std::size_t VqmSharedConsumerProbe() {
    return vqm::RequiredBytes(vqm::ImageFormat{}, {8, 8});
}
