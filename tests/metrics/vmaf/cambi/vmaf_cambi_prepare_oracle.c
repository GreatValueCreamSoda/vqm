// Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Optional fixture generator; includes the pinned libvmaf source in this test
// translation unit to call its static preprocessing and mask functions.
// clang-format off
// gcc -O2 -Ibuild/vmaf-build/src -Ibuild/vmaf-src/libvmaf/include \
//   -Ibuild/vmaf-src/libvmaf/src tests/metrics/vmaf/cambi/vmaf_cambi_prepare_oracle.c \
//   build/vmaf-build/src/libvmaf.a -lstdc++ -lpthread -lm -o /tmp/vmaf-cambi-oracle
// /tmp/vmaf-cambi-oracle
// clang-format on
#include "../../../../build/vmaf-src/libvmaf/src/feature/cambi.c"
#include <inttypes.h>
#include <stdlib.h>

static uint32_t input_value(unsigned x, unsigned y, unsigned depth) {
    uint32_t v = ((x / 9) * 31 + (y / 11) * 41 + (x / 57) * 7) % 256;
    if (depth > 8) {
        unsigned shift = depth - 8;
        uint32_t low =
            ((x / 48 + y / 48) % 2) ? (x * 3 + y * 5) & ((1u << shift) - 1u) : 0;
        v = (v << shift) | low;
    }
    return v;
}
static uint64_t hash_values(const uint16_t* data, unsigned n) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned i = 0; i < n; ++i)
        hash = (hash ^ data[i]) * UINT64_C(1099511628211);
    return hash;
}
int main(void) {
    const unsigned w = 216, h = 217, n = w * h;
    const unsigned depths[][2] = {{8, 8}, {10, 10}, {12, 12}, {16, 16}, {10, 8}};
    uint16_t* processed = calloc(n, sizeof(uint16_t));
    uint16_t* mask = calloc(n, sizeof(uint16_t));
    uint16_t* derivative = calloc(w, sizeof(uint16_t));
    uint32_t* dp = calloc((w + 7) * 8, sizeof(uint32_t));
    for (unsigned k = 0; k < sizeof(depths) / sizeof(depths[0]); ++k) {
        unsigned depth = depths[k][0], encoded = depths[k][1];
        size_t bytes = depth == 8 ? 1 : 2;
        uint8_t* source = calloc(n, bytes);
        VmafPicture in = {0}, out = {0}, m = {0};
        in.bpc = depth;
        in.w[0] = w;
        in.h[0] = h;
        in.stride[0] = w * bytes;
        in.data[0] = source;
        out.w[0] = w;
        out.h[0] = h;
        out.stride[0] = w * 2;
        out.data[0] = processed;
        m.w[0] = w;
        m.h[0] = h;
        m.stride[0] = w * 2;
        m.data[0] = mask;
        for (unsigned y = 0; y < h; ++y)
            for (unsigned x = 0; x < w; ++x) {
                uint32_t value = input_value(x, y, depth);
                if (bytes == 1)
                    source[y * w + x] = value;
                else
                    ((uint16_t*)source)[y * w + x] = value;
            }
        if (cambi_preprocessing(&in, &out, w, h, encoded)) return 1;
        get_spatial_mask(
            &out, &m, dp, derivative, w, h, get_derivative_data_for_row,
            compute_dp_row, compute_mask_row
        );
        uint64_t sum = 0;
        for (unsigned i = 0; i < n; ++i) sum += mask[i];
        printf(
            "depth=%u encoded=%u threshold=%u processed=%" PRIu64
            " mask=%" PRIu64 " mask_sum=%" PRIu64 "\n",
            depth, encoded, get_mask_index(w, h, MASK_FILTER_SIZE),
            hash_values(processed, n), hash_values(mask, n), sum
        );
        free(source);
    }
    free(processed);
    free(mask);
    free(derivative);
    free(dp);
}
