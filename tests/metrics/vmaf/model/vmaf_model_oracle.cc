// Copyright (c) 2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

// Optional reference program for tests/metrics/vmaf/model/vmaf_model_test.cc. It
// only uses libvmaf's public API; it is not linked into VQM or its normal test
// suite. Build the pinned reference with tools/build_references.py, then run:
// clang-format off
//   VMAF_FLAGS="-std=c++20 -I.reference/vmaf/libvmaf/include"
//   VMAF_LIB=build/vmaf-build/src/libvmaf.a
//   c++ $VMAF_FLAGS tests/metrics/vmaf/model/vmaf_model_oracle.cc "$VMAF_LIB" \
//     -lpthread -lm -o /tmp/vmaf-model-oracle
//   /tmp/vmaf-model-oracle .reference/vmaf/model/vmaf_float_v0.6.1.json
//   /tmp/vmaf-model-oracle .reference/vmaf/model/vmaf_float_4k_v0.6.1.json
// clang-format on
// Each row is printed in the model test's case order for flags 0, 2, 1, 3.
// The 4K model declares no score transform, so flags 2 and 3 are a no-op there;
// VQM's 4K phone transform is a derived extension, not this reference output.

#include <array>
#include <cstddef>
#include <cstdio>

#include <libvmaf/libvmaf.h>
#include <libvmaf/model.h>

namespace {

constexpr std::array<const char*, 6> kFeatureNames{
    "VMAF_feature_adm2_score",       "VMAF_feature_motion2_score",
    "VMAF_feature_vif_scale0_score", "VMAF_feature_vif_scale1_score",
    "VMAF_feature_vif_scale2_score", "VMAF_feature_vif_scale3_score"};
constexpr std::array<std::array<double, 6>, 6> kFeatures{{
    {1.0, 0.0, 1.0, 1.0, 1.0, 1.0},
    {0.962084, 3.894359729166667, 0.505712, 0.879061, 0.937873, 0.964301},
    {0.4, 30.0, 0.1, 0.2, 0.3, 0.4},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0, 100.0, 1.0, 1.0, 1.0, 1.0},
    {0.2, 0.5, 0.3, 0.55, 0.6, 0.81},
}};

int Evaluate(const char* path, unsigned flags) {
    VmafModelConfig model_config{};
    model_config.name = "vmaf_float";
    model_config.flags = flags;
    VmafModel* model = nullptr;
    int error = vmaf_model_load_from_path(&model, &model_config, path);
    if (error) return error;
    VmafConfiguration config{};
    config.n_subsample = 1;
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
            if (!error) {
                error = vmaf_score_at_index(
                    context, model, &score, static_cast<unsigned>(i)
                );
            }
            if (!error) {
                std::printf("flags=%u index=%zu score=%.17g\n", flags, i, score);
            }
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
         {0u, unsigned(VMAF_MODEL_FLAG_ENABLE_TRANSFORM),
          unsigned(VMAF_MODEL_FLAG_DISABLE_CLIP),
          unsigned(
              VMAF_MODEL_FLAG_ENABLE_TRANSFORM | VMAF_MODEL_FLAG_DISABLE_CLIP
          )}) {
        const int error = Evaluate(argv[1], flags);
        if (error) {
            std::fprintf(stderr, "libvmaf error: %d\n", error);
            return 1;
        }
    }
    return 0;
}
