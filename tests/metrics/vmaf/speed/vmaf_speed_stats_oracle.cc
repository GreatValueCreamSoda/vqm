// Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Full SpEED-feature oracle via pinned libvmaf's public API. Produces the
// fixture values in vmaf_speed_stats_test.cc for standard v1 options.
// clang-format off
// c++ -std=c++20 -I.reference/vmaf/libvmaf/include tests/metrics/vmaf/speed/vmaf_speed_stats_oracle.cc \
//   build/vmaf-build/src/libvmaf.a -lpthread -lm -o /tmp/vmaf-speed-stats-oracle
// /tmp/vmaf-speed-stats-oracle
// clang-format on
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <libvmaf/libvmaf.h>

int Run(unsigned depth, unsigned w, unsigned h) {
    VmafConfiguration c{};
    c.n_threads = 0;
    c.log_level = VMAF_LOG_LEVEL_WARNING;
    VmafContext* v = nullptr;
    if (vmaf_init(&v, c)) return 1;
    VmafFeatureDictionary* d = nullptr;
    vmaf_feature_dictionary_set(&d, "speed_kernelscale", "1.0");
    vmaf_feature_dictionary_set(&d, "speed_nn_floor", "0.1");
    vmaf_feature_dictionary_set(&d, "speed_sigma_nn", "0.19");
    vmaf_feature_dictionary_set(&d, "speed_weight_var_mode", "5");
    vmaf_feature_dictionary_set(&d, "speed_max_val", "45");
    if (vmaf_use_feature(v, "speed_chroma", d)) return 2;
    for (unsigned frame = 0; frame < 4; frame++) {
        VmafPicture a{}, b{};
        if (vmaf_picture_alloc(&a, VMAF_PIX_FMT_YUV444P, depth, w, h)) return 3;
        if (vmaf_picture_alloc(&b, VMAF_PIX_FMT_YUV444P, depth, w, h)) return 3;
        for (unsigned pl = 0; pl < 3; pl++)
            for (unsigned y = 0; y < h; y++)
                for (unsigned x = 0; x < w; x++) {
                    auto val =
                        16 + ((x * (7 + frame * 2) + y * 13 + (x / 7) * 23 +
                               (y / 11) * 31 + pl * 19 + frame * 33) %
                              220);
                    auto dist = (val + ((x * 13 + y * (3 + frame) + pl * 17) %
                                        (41 + frame * 7))) %
                                256;
                    if (depth == 8) {
                        static_cast<unsigned char*>(
                            a.data[pl]
                        )[y * a.stride[pl] + x] =
                            static_cast<unsigned char>(val);
                        static_cast<unsigned char*>(
                            b.data[pl]
                        )[y * b.stride[pl] + x] =
                            static_cast<unsigned char>(dist);
                    } else {
                        auto* ar = reinterpret_cast<std::uint16_t*>(
                            static_cast<unsigned char*>(a.data[pl]) +
                            y * a.stride[pl]
                        );
                        auto* br = reinterpret_cast<std::uint16_t*>(
                            static_cast<unsigned char*>(b.data[pl]) +
                            y * b.stride[pl]
                        );
                        ar[x] = static_cast<std::uint16_t>(
                            4 * val + ((x + y + pl) % 4)
                        );
                        br[x] = static_cast<std::uint16_t>(
                            4 * dist + ((x * 3 + y + frame) % 4)
                        );
                    }
                }
        if (vmaf_read_pictures(v, &a, &b, frame)) return 3;
    }
    if (vmaf_read_pictures(v, nullptr, nullptr, 0)) return 4;
    for (unsigned i = 0; i < 4; i++) {
        double u, vv, uv;
        for (auto p :
             {std::pair{"speed_chroma_u_mxv_45_nnf_0.1_snn_0.19_wvm_5", &u},
              std::pair{"speed_chroma_v_mxv_45_nnf_0.1_snn_0.19_wvm_5", &vv},
              std::pair{"speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5", &uv}}) {
            if (vmaf_feature_score_at_index(v, p.first, p.second, i)) return 5;
        }
        std::printf(
            "%ux%u %ubit frame %u u %.17g v %.17g uv %.17g\n", w, h, depth, i, u,
            vv, uv
        );
    }
    vmaf_close(v);
    return 0;
}

int main() {
    for (const auto [w, h] : {std::pair{320u, 320u}, std::pair{1920u, 1080u}}) {
        for (const unsigned depth : {8u, 10u}) {
            const int error = Run(depth, w, h);
            if (error) return error;
        }
    }
    return 0;
}
