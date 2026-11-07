// Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Public libvmaf full CAMBI feature oracle for vmaf_cambi_test.cc.
// clang-format off
// c++ -std=c++20 -I.reference/vmaf/libvmaf/include tests/metrics/vmaf/cambi/vmaf_cambi_oracle.cc \
//   build/vmaf-build/src/libvmaf.a -lpthread -lm -o /tmp/vmaf-cambi-oracle
// /tmp/vmaf-cambi-oracle
// clang-format on
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>
#include <libvmaf/libvmaf.h>

int Run(unsigned w, unsigned h, unsigned depth, bool override8) {
    VmafConfiguration c{};
    c.n_threads = 0;
    c.log_level = VMAF_LOG_LEVEL_DEBUG;
    VmafContext* v = nullptr;
    if (vmaf_init(&v, c)) return 1;
    VmafFeatureDictionary* options = nullptr;
    vmaf_feature_dictionary_set(&options, "cambi_high_res_speedup", "1080");
    vmaf_feature_dictionary_set(&options, "cambi_vis_lum_threshold", "0.06");
    vmaf_feature_dictionary_set(&options, "cambi_max_val", "17");
    if (override8) vmaf_feature_dictionary_set(&options, "enc_bitdepth", "8");
    if (vmaf_use_feature(v, "cambi", options)) return 2;
    for (unsigned frame = 0; frame < 4; frame++) {
        VmafPicture ref{}, dis{};
        if (vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV444P, depth, w, h) ||
            vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV444P, depth, w, h))
            return 3;
        for (unsigned pl = 0; pl < 3; pl++)
            for (unsigned y = 0; y < h; y++)
                for (unsigned x = 0; x < w; x++) {
                    // Gradient/bands and a separately patterned flat-region
                    // case.
                    unsigned val =
                        frame == 0 ? (16 + (x * 219) / (w - 1))
                        : frame == 1
                            ? (16 + (((x / 24) + (y / 32) * 3) % 11) * 19)
                        : frame == 2 ? (64 + ((x / 16) % 24))
                                     : (128 + ((x / 4) % 16));
                    if (depth == 8) {
                        static_cast<unsigned char*>(
                            ref.data[pl]
                        )[y * ref.stride[pl] + x] =
                            static_cast<unsigned char>(val);
                        static_cast<unsigned char*>(
                            dis.data[pl]
                        )[y * dis.stride[pl] + x] =
                            static_cast<unsigned char>(val);
                    } else {
                        auto* a = reinterpret_cast<std::uint16_t*>(
                            static_cast<unsigned char*>(ref.data[pl]) +
                            y * ref.stride[pl]
                        );
                        auto* b = reinterpret_cast<std::uint16_t*>(
                            static_cast<unsigned char*>(dis.data[pl]) +
                            y * dis.stride[pl]
                        );
                        const unsigned scale = 1u << (depth - 8);
                        const auto value = static_cast<std::uint16_t>(
                            val * scale + ((x + y + frame) % scale)
                        );
                        a[x] = value;
                        b[x] = value;
                    }
                }
        if (vmaf_read_pictures(v, &ref, &dis, frame)) return 4;
    }
    if (vmaf_read_pictures(v, nullptr, nullptr, 0)) return 5;
    const auto name = std::string("cambi_hrs_1080_cmxv_17_vlt_0.06") +
                      (override8 ? "_encbd_8" : "");
    for (unsigned frame = 0; frame < 4; frame++) {
        double score = 0;
        if (vmaf_feature_score_at_index(v, name.c_str(), &score, frame)) {
            std::fprintf(stderr, "missing %s\n", name.c_str());
            return 6;
        }
        std::printf(
            "%ux%u depth=%u enc=%u frame=%u %.17g\n", w, h, depth,
            override8 ? 8u : depth, frame, score
        );
    }
    vmaf_close(v);
    return 0;
}
int main() {
    for (auto [w, h] :
         {std::pair{320u, 320u}, std::pair{321u, 319u}, std::pair{1920u, 1080u}})
        for (auto [depth, enc8] :
             {std::pair{8u, false}, std::pair{10u, false}, std::pair{10u, true},
              std::pair{12u, false}, std::pair{12u, true}}) {
            auto result = Run(w, h, depth, enc8);
            if (result) return result;
        }
}
