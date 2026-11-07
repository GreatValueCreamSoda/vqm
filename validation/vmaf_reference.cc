// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
//
// Test-only driver for Netflix libvmaf's public API. It accepts a pair of raw
// little-endian planar YUV420 sequences, runs a selected model or the ADM
// debug feature, and emits reference values at double precision. No libvmaf
// source is copied here; the driver links the separately built reference.

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <libvmaf/libvmaf.h>

namespace {

void Check(int result, const char* operation) {
    if (result != 0)
        throw std::runtime_error(
            std::string(operation) + ": " + std::to_string(result)
        );
}

unsigned ParseUnsigned(const char* text) {
    std::string s(text);
    std::size_t parsed = 0;
    const auto value = std::stoull(s, &parsed);
    if (parsed != s.size() || value > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("invalid unsigned integer: " + s);
    return static_cast<unsigned>(value);
}

struct Picture {
    VmafPicture picture{};
    ~Picture() {
        if (picture.ref) vmaf_picture_unref(&picture);
    }
};

void ReadPicture(std::ifstream& stream, VmafPicture& picture, unsigned bytes) {
    for (unsigned plane = 0; plane < 3; ++plane) {
        const auto row_bytes =
            static_cast<std::streamsize>(picture.w[plane]) * bytes;
        auto* row = static_cast<char*>(picture.data[plane]);
        for (unsigned y = 0; y < picture.h[plane]; ++y) {
            if (!stream.read(row, row_bytes))
                throw std::runtime_error("truncated YUV frame");
            row += picture.stride[plane];
        }
    }
}

int Run(int argc, char** argv) {
    if (argc != 8 && argc != 9) {
        std::cerr << "usage: vmaf_reference float|integer|adm-debug "
                     "MODEL.json|- REFERENCE.yuv "
                     "DISTORTED.yuv WIDTH HEIGHT BITDEPTH\n"
                     "       vmaf_reference models V0.json REFERENCE.yuv "
                     "DISTORTED.yuv WIDTH HEIGHT BITDEPTH V1.json\n";
        return 2;
    }
    const std::string kind = argv[1];
    if ((kind == "models") != (argc == 9))
        throw std::invalid_argument("models mode needs two model paths");
    if (kind != "float" && kind != "integer" && kind != "adm-debug" &&
        kind != "models")
        throw std::invalid_argument("unknown model kind");
    if (kind == "adm-debug" && std::string(argv[2]) != "-")
        throw std::invalid_argument("adm-debug uses '-' in place of a model");
    const unsigned width = ParseUnsigned(argv[5]);
    const unsigned height = ParseUnsigned(argv[6]);
    const unsigned depth = ParseUnsigned(argv[7]);
    if (width < 16 || height < 16 ||
        (kind != "models" && (width % 2 || height % 2)) ||
        (depth != 8 && depth != 10 && depth != 12 && depth != 16))
        throw std::invalid_argument(
            "YUV420 needs even width/height >= 16 and depth 8/10/12/16"
        );
    if constexpr (std::endian::native != std::endian::little) {
        if (depth > 8)
            throw std::runtime_error(
                "high-bit-depth fixtures require little-endian host"
            );
    }
    if (std::uint64_t(width) * height >
        std::numeric_limits<std::uint64_t>::max() / 6)
        throw std::invalid_argument("YUV frame size overflows");
    const std::uint64_t frame_bytes = std::uint64_t(width) * height * 3 /
                                      (kind == "models" ? 1 : 2) *
                                      (depth > 8 ? 2 : 1);
    const auto ref_size = std::filesystem::file_size(argv[3]);
    const auto dist_size = std::filesystem::file_size(argv[4]);
    if (ref_size == 0 || ref_size != dist_size || ref_size % frame_bytes != 0 ||
        ref_size / frame_bytes > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument(
            "sequences must contain the same nonzero number of complete frames"
        );
    const unsigned count = static_cast<unsigned>(ref_size / frame_bytes);
    std::ifstream reference(argv[3], std::ios::binary),
        distorted(argv[4], std::ios::binary);
    if (!reference || !distorted)
        throw std::runtime_error("cannot open input sequences");

    VmafContext* raw_context = nullptr;
    VmafConfiguration config{};
    config.log_level = VMAF_LOG_LEVEL_NONE;
    config.n_threads = 0;  // Synchronous, scalar build.
    Check(vmaf_init(&raw_context, config), "vmaf_init");
    const auto close_context = [](VmafContext* context) { vmaf_close(context); };
    std::unique_ptr<VmafContext, decltype(close_context)> context(
        raw_context, close_context
    );
    std::unique_ptr<VmafModel, decltype(&vmaf_model_destroy)> model(
        nullptr, vmaf_model_destroy
    );
    std::unique_ptr<VmafModel, decltype(&vmaf_model_destroy)> model2(
        nullptr, vmaf_model_destroy
    );
    if (kind == "adm-debug") {
        VmafFeatureDictionary* options = nullptr;
        Check(
            vmaf_feature_dictionary_set(&options, "debug", "true"), "adm debug"
        );
        const int status = vmaf_use_feature(context.get(), "float_adm", options);
        if (status != 0) vmaf_feature_dictionary_free(&options);
        Check(status, "vmaf_use_feature(float_adm)");
    } else {
        VmafModel* raw_model = nullptr;
        VmafModelConfig model_config{};
        model_config.name = "reference";
        Check(
            vmaf_model_load_from_path(&raw_model, &model_config, argv[2]),
            "vmaf_model_load_from_path"
        );
        model.reset(raw_model);
        Check(
            vmaf_use_features_from_model(context.get(), model.get()),
            "vmaf_use_features_from_model"
        );
        if (kind == "models") {
            raw_model = nullptr;
            model_config.name = "v1";
            Check(
                vmaf_model_load_from_path(&raw_model, &model_config, argv[8]),
                "vmaf_model_load_from_path(v1)"
            );
            model2.reset(raw_model);
            Check(
                vmaf_use_features_from_model(context.get(), model2.get()),
                "vmaf_use_features_from_model(v1)"
            );
        }
    }
    for (unsigned index = 0; index < count; ++index) {
        Picture ref, dist;
        Check(
            vmaf_picture_alloc(
                &ref.picture,
                kind == "models" ? VMAF_PIX_FMT_YUV444P : VMAF_PIX_FMT_YUV420P,
                depth, width, height
            ),
            "vmaf_picture_alloc(ref)"
        );
        Check(
            vmaf_picture_alloc(
                &dist.picture,
                kind == "models" ? VMAF_PIX_FMT_YUV444P : VMAF_PIX_FMT_YUV420P,
                depth, width, height
            ),
            "vmaf_picture_alloc(dist)"
        );
        ReadPicture(reference, ref.picture, depth > 8 ? 2 : 1);
        ReadPicture(distorted, dist.picture, depth > 8 ? 2 : 1);
        Check(
            vmaf_read_pictures(
                context.get(), &ref.picture, &dist.picture, index
            ),
            "vmaf_read_pictures"
        );
    }
    Check(vmaf_read_pictures(context.get(), nullptr, nullptr, 0), "vmaf_flush");
    if (kind == "models") {
        std::cout << "frame,v0,v1\n" << std::setprecision(17);
        for (unsigned index = 0; index < count; ++index) {
            double first = 0, second = 0;
            Check(
                vmaf_score_at_index(context.get(), model.get(), &first, index),
                "v0 score"
            );
            Check(
                vmaf_score_at_index(context.get(), model2.get(), &second, index),
                "v1 score"
            );
            std::cout << index << ',' << first << ',' << second << '\n';
        }
        return 0;
    }
    if (kind == "adm-debug") {
        const std::array<const char*, 9> features = {
            "adm_num_scale0", "adm_den_scale0", "adm_num_scale1",
            "adm_den_scale1", "adm_num_scale2", "adm_den_scale2",
            "adm_num_scale3", "adm_den_scale3", "VMAF_feature_adm2_score"};
        std::cout << "frame,num0,den0,num1,den1,num2,den2,num3,den3,adm2\n";
        std::cout << std::setprecision(17);
        for (unsigned index = 0; index < count; ++index) {
            std::cout << index;
            for (const char* feature : features) {
                double value = 0;
                Check(
                    vmaf_feature_score_at_index(
                        context.get(), feature, &value, index
                    ),
                    feature
                );
                std::cout << ',' << value;
            }
            std::cout << '\n';
        }
        return 0;
    }
    const std::string prefix =
        kind == "float" ? "VMAF_feature_" : "VMAF_integer_feature_";
    const std::array<std::string, 6> names = {
        prefix + "adm2_score",       prefix + "motion2_score",
        prefix + "vif_scale0_score", prefix + "vif_scale1_score",
        prefix + "vif_scale2_score", prefix + "vif_scale3_score"};
    std::cout << "frame,adm2,motion2,vif_scale0,vif_scale1,vif_scale2,vif_"
                 "scale3,vmaf\n";
    std::cout << std::setprecision(17);
    for (unsigned index = 0; index < count; ++index) {
        std::cout << index;
        for (const auto& name : names) {
            double value = 0;
            Check(
                vmaf_feature_score_at_index(
                    context.get(), name.c_str(), &value, index
                ),
                name.c_str()
            );
            std::cout << ',' << value;
        }
        double score = 0;
        Check(
            vmaf_score_at_index(context.get(), model.get(), &score, index),
            "vmaf_score_at_index"
        );
        std::cout << ',' << score << '\n';
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "vmaf reference: " << error.what() << '\n';
        return 1;
    }
}
