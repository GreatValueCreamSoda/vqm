// Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Reproduces the multi-scale decimation checks in vmaf_cambi_decimate_test.cc.
// Calls the exported scalar reference implementation in the pinned libvmaf.
// clang-format off
// c++ -std=c++20 -I.reference/vmaf/libvmaf/include tests/metrics/vmaf/cambi/vmaf_cambi_decimate_oracle.cc \
//   build/vmaf-build/src/libvmaf.a -lpthread -lm -o /tmp/vmaf-cambi-decimate-oracle
// /tmp/vmaf-cambi-decimate-oracle
// clang-format on
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <utility>
#include <libvmaf/picture.h>
extern "C" void decimate(VmafPicture*, unsigned, unsigned);

std::uint64_t Hash(const VmafPicture& pic, unsigned width, unsigned height) {
    std::uint64_t hash = 14695981039346656037ull;
    for (unsigned y = 0; y < height; y++) {
        const auto* row = reinterpret_cast<const std::uint16_t*>(
            static_cast<const unsigned char*>(pic.data[0]) + y * pic.stride[0]
        );
        for (unsigned x = 0; x < width; x++) {
            hash ^= row[x];
            hash *= 1099511628211ull;
        }
    }
    return hash;
}
int main() {
    for (auto [w, h] :
         {std::pair{17u, 19u}, std::pair{32u, 24u}, std::pair{33u, 17u}}) {
        VmafPicture a{}, b{};
        if (vmaf_picture_alloc(&a, VMAF_PIX_FMT_YUV400P, 10, w, h) ||
            vmaf_picture_alloc(&b, VMAF_PIX_FMT_YUV400P, 10, w, h))
            return 1;
        for (unsigned y = 0; y < h; y++) {
            auto* ar = reinterpret_cast<std::uint16_t*>(
                static_cast<unsigned char*>(a.data[0]) + y * a.stride[0]
            );
            auto* br = reinterpret_cast<std::uint16_t*>(
                static_cast<unsigned char*>(b.data[0]) + y * b.stride[0]
            );
            for (unsigned x = 0; x < w; x++) {
                ar[x] = static_cast<std::uint16_t>(
                    (x * 31 + y * 47 + (x / 3) * 11) % 1024
                );
                br[x] =
                    static_cast<std::uint16_t>((x * 3 + y * 5 + (x / 7)) % 2);
            }
        }
        std::printf("%ux%u\n", w, h);
        for (int scale = 0; scale < 4; scale++) {
            w = (w + 1) / 2;
            h = (h + 1) / 2;
            decimate(&a, w, h);
            decimate(&b, w, h);
            std::printf(
                "%llu %llu\n", static_cast<unsigned long long>(Hash(a, w, h)),
                static_cast<unsigned long long>(Hash(b, w, h))
            );
        }
        vmaf_picture_unref(&a);
        vmaf_picture_unref(&b);
    }
}
