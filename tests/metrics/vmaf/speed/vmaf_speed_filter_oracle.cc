// Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Optional reference fixture generator for
// tests/metrics/vmaf/speed/vmaf_speed_filter_test.cc. The called functions are
// exported by the pinned libvmaf static reference.
// clang-format off
// c++ -std=c++20 tests/metrics/vmaf/speed/vmaf_speed_filter_oracle.cc build/vmaf-build/src/libvmaf.a \
//   -lpthread -lm -o /tmp/vmaf-speed-filter-oracle
// /tmp/vmaf-speed-filter-oracle
// clang-format on
#include <cstdio>
#include <vector>
extern "C" {
void speed_get_antialias_filter(float*, int, float);
void vif_get_filter(float*, int, float);
void vif_filter1d_dec16_scalar_s(
    const float*, const float*, float*, float*, int, int, int, int, int
);
void vif_filter1d_s(
    const float*, const float*, float*, float*, int, int, int, int, int
);
}
int main() {
    constexpr int w = 93, h = 115, dw = w / 16, dh = h / 16;
    std::vector<float> src(w * h), tmp(w * h), down(dw * dh), blur(dw * dh);
    float filter[9], mean[3];
    speed_get_antialias_filter(filter, 4, 1);
    vif_get_filter(mean, 4, 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            src[y * w + x] =
                float((7 * x + 13 * y + (x / 7) * 23 + (y / 11) * 31) % 251) -
                120.0f + float((x + y) % 4) * .25f;
    vif_filter1d_dec16_scalar_s(
        filter, src.data(), down.data(), tmp.data(), w, h, w * sizeof(float),
        dw * sizeof(float), 9
    );
    vif_filter1d_s(
        mean, down.data(), blur.data(), tmp.data(), dw, dh, dw * sizeof(float),
        dw * sizeof(float), 3
    );
    std::puts("down:");
    for (int i = 0; i < dw * dh; ++i)
        std::printf("%.10gf,%s", down[i], i % 5 == 4 ? "\n" : " ");
    std::puts("result:");
    for (int i = 0; i < dw * dh; ++i)
        std::printf("%.10gf,%s", down[i] - blur[i], i % 5 == 4 ? "\n" : " ");
}
