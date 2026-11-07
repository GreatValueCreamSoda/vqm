// Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Pinned libvmaf oracle for near-singular 16-bit chroma covariance.
// clang-format off
// c++ -std=c++20 -I.reference/vmaf/libvmaf/include tests/metrics/vmaf/speed/vmaf_speed_singular_oracle.cc \
//   build/vmaf-build/src/libvmaf.a -lpthread -lm -o /tmp/vmaf-speed-singular-oracle
// /tmp/vmaf-speed-singular-oracle
// clang-format on
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <libvmaf/libvmaf.h>

uint32_t hash(unsigned x, unsigned y, unsigned pl, unsigned salt) {
    uint32_t z = x * 0x9e3779b9u ^ y * 0x85ebca6bu ^ pl * 0xc2b2ae35u ^
                 salt * 0x27d4eb2du;
    z ^= z >> 16;
    z *= 0x7feb352du;
    z ^= z >> 15;
    z *= 0x846ca68bu;
    z ^= z >> 16;
    return z;
}
int main() {
    for (unsigned amp : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 640u,
                         768u, 896u, 1024u, 2048u, 4096u, 8192u, 16000u}) {
        VmafConfiguration c{};
        c.log_level = VMAF_LOG_LEVEL_NONE;
        VmafContext* v = nullptr;
        if (vmaf_init(&v, c)) return 1;
        VmafFeatureDictionary* d = nullptr;
        vmaf_feature_dictionary_set(&d, "speed_kernelscale", "1.0");
        vmaf_feature_dictionary_set(&d, "speed_nn_floor", "0.1");
        vmaf_feature_dictionary_set(&d, "speed_sigma_nn", "0.19");
        vmaf_feature_dictionary_set(&d, "speed_weight_var_mode", "5");
        vmaf_feature_dictionary_set(&d, "speed_max_val", "45");
        if (vmaf_use_feature(v, "speed_chroma", d)) return 2;
        VmafPicture a{}, b{};
        if (vmaf_picture_alloc(&a, VMAF_PIX_FMT_YUV444P, 16, 320, 320) ||
            vmaf_picture_alloc(&b, VMAF_PIX_FMT_YUV444P, 16, 320, 320))
            return 3;
        for (unsigned p = 0; p < 3; p++)
            for (unsigned y = 0; y < 320; y++)
                for (unsigned x = 0; x < 320; x++) {
                    auto* ar = reinterpret_cast<uint16_t*>(
                        static_cast<uint8_t*>(a.data[p]) + y * a.stride[p]
                    );
                    auto* br = reinterpret_cast<uint16_t*>(
                        static_cast<uint8_t*>(b.data[p]) + y * b.stride[p]
                    );
                    int noise = int(hash(x, y, p, 0) % (2 * amp + 1)) - int(amp);
                    int difference =
                        int(hash(x, y, p, 1) % (amp + 1)) - int(amp / 2);
                    ar[x] = uint16_t(32768 + noise);
                    br[x] = uint16_t(
                        std::clamp(32768 + noise + difference, 0, 65535)
                    );
                }
        int err = vmaf_read_pictures(v, &a, &b, 0);
        if (err) return 4;
        if (vmaf_read_pictures(v, nullptr, nullptr, 0)) return 5;
        double u = 0, vv = 0, uv = 0;
        for (auto pair :
             {std::pair{"speed_chroma_u_mxv_45_nnf_0.1_snn_0.19_wvm_5", &u},
              std::pair{"speed_chroma_v_mxv_45_nnf_0.1_snn_0.19_wvm_5", &vv},
              std::pair{"speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5", &uv}})
            if (vmaf_feature_score_at_index(v, pair.first, pair.second, 0))
                return 6;
        std::printf("amp %u u %.12g v %.12g uv %.12g\n", amp, u, vv, uv);
        vmaf_close(v);
    }
}
