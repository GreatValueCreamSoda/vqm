// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Research-only public VMAF v1 feature probe. No performance timers.
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "io.h"
#include "vmaf_v1.h"
#include "vulkan/vulkan_support.h"

namespace {

struct Options {
    std::string_view reference, distorted;
    vqm::VmafChromaSampling chroma = vqm::VmafChromaSampling::Native;
    std::string_view chroma_name = "native";
    bool clip = true;
};

constexpr std::string_view kUsage =
    "usage: feature_probe REF DIST [--chroma native|420|444] [--unclipped]\n";

Options Parse(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument("expected two image paths");
    Options options{argv[1], argv[2]};
    for (int i = 3; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--unclipped") {
            options.clip = false;
        } else if (flag == "--chroma" && i + 1 < argc) {
            options.chroma_name = argv[++i];
            if (options.chroma_name == "native")
                options.chroma = vqm::VmafChromaSampling::Native;
            else if (options.chroma_name == "420")
                options.chroma = vqm::VmafChromaSampling::Yuv420;
            else if (options.chroma_name == "444")
                options.chroma = vqm::VmafChromaSampling::Yuv444;
            else
                throw std::invalid_argument("chroma must be native, 420 or 444");
        } else {
            throw std::invalid_argument("unknown or incomplete option");
        }
    }
    return options;
}

vqm::Image Load(std::string_view path) {
    std::string error;
    auto image = vqm::image_io::LoadImage(path, error);
    if (!image) throw std::runtime_error(std::string(path) + ": " + error);
    return std::move(*image);
}

void Probe(const Options& options) {
    const auto reference = Load(options.reference);
    const auto distorted = Load(options.distorted);
    if (reference.size != distorted.size)
        throw std::invalid_argument("reference and distorted sizes differ");
    vqm::VmafV1Config config;
    config.size = reference.size;
    config.input_format = reference.format;
    config.chroma_sampling = options.chroma;
    config.gpu_resident_images = 0;
    config.clip = options.clip;
    vqm::VmafV1Pipeline pipeline(config);
    if (pipeline.Push(reference, distorted))
        throw std::runtime_error("single-frame sequence emitted before Flush");
    const auto frame = pipeline.Flush();
    if (!frame) throw std::runtime_error("Flush emitted no single-frame result");
    std::cout
        << "width,height,chroma,clip,adm3,speed_chroma_uv,cambi,motion3,score\n"
        << reference.size.width << ',' << reference.size.height << ','
        << options.chroma_name << ',' << options.clip << ','
        << std::setprecision(17) << frame->adm3 << ',' << frame->speed_chroma_uv
        << ',' << frame->cambi << ',' << frame->motion3 << ',' << frame->score
        << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << kUsage;
        return 0;
    }
    int status = 0;
    try {
        Probe(Parse(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "feature_probe: " << error.what() << '\n' << kUsage;
        status = 1;
    }
    // No pipeline remains alive. Explicit shutdown avoids driver atexit-order
    // issues in headless research runs; metric evaluation uses the public API.
    vqm::internal::ShutdownDefaultVulkanContext();
    return status;
}
