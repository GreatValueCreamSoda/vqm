// Copyright (c) 2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Optional pinned-libvmaf reference for
// tests/metrics/vmaf/model/vmaf_v1_model_test.cc. Build with (from the
// repository root):
// clang-format off
// c++ -std=c++20 -I.reference/vmaf/libvmaf/include tests/metrics/vmaf/model/vmaf_v1_model_oracle.cc \
//   build/vmaf-build/src/libvmaf.a -lpthread -lm -o /tmp/vmaf-v1-model-oracle
// /tmp/vmaf-v1-model-oracle \
//   .reference/vmaf/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json
// clang-format on

#include <array>
#include <cstddef>
#include <cstdio>

#include <libvmaf/libvmaf.h>
#include <libvmaf/model.h>

namespace {

constexpr std::array<const char*, 4> kFeatureNames{
    "cambi_hrs_1080_cmxv_17_vlt_0.06",
    "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
    "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02",
    "integer_motion3_mmxv_18"};
constexpr std::array<std::array<double, 4>, 7> kFeatures{{
    {0.0, 0.0, 1.0, 0.0},
    {1.2, 3.3, 0.9, 5.0},
    {17.0, 45.0, 0.5, 18.0},
    {0.1, 0.0, 0.72, 10.0},
    {0.1, 1e-8, 0.72, 10.0},
    {0.0, 1.0, 1.0, 0.0},
    {10.0, 30.0, 0.55, 0.0},
}};

int Evaluate(const char* path, unsigned flags) {
    VmafModelConfig model_config{};
    model_config.name = "vmaf_v1";
    model_config.flags = flags;
    VmafModel* model = nullptr;
    int error = vmaf_model_load_from_path(&model, &model_config, path);
    if (error) return error;
    VmafConfiguration config{};
    config.n_subsample = 1;
    config.log_level = VMAF_LOG_LEVEL_DEBUG;
    VmafContext* context = nullptr;
    error = vmaf_init(&context, config);
    if (!error) {
        for (std::size_t i = 0; i < kFeatures.size() && !error; ++i) {
            for (std::size_t f = 0; f < kFeatureNames.size() && !error; ++f) {
                error = vmaf_import_feature_score(
                    context, kFeatureNames[f], kFeatures[i][f],
                    static_cast<unsigned>(i)
                );
            }
            double score = 0.0;
            if (!error)
                error = vmaf_score_at_index(
                    context, model, &score, static_cast<unsigned>(i)
                );
            if (!error)
                std::printf("flags=%u index=%zu score=%.17g\n", flags, i, score);
        }
        vmaf_close(context);
    }
    vmaf_model_destroy(model);
    return error;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.json\n", argv[0]);
        return 2;
    }
    for (unsigned flags :
         {0u, unsigned(VMAF_MODEL_FLAG_DISABLE_TRANSFORM),
          unsigned(VMAF_MODEL_FLAG_DISABLE_CLIP),
          unsigned(
              VMAF_MODEL_FLAG_DISABLE_TRANSFORM | VMAF_MODEL_FLAG_DISABLE_CLIP
          )}) {
        int error = Evaluate(argv[1], flags);
        if (error) {
            std::fprintf(stderr, "libvmaf error: %d\n", error);
            return 1;
        }
    }
    return 0;
}
