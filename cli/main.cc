// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <cstdio>
#include <exception>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "butteraugli.h"
#include "io.h"
#include "options.h"
#include "report.h"
#include "video.h"
#include "vqm.h"
#include "vqm_alpha.h"
#include "vulkan/vulkan_support.h"

namespace {

int RunButteraugli(
    const vqm::cli::Options& options, const vqm::Image& original,
    const vqm::Image& distorted
) {
    if (options.half_planes) {
        std::fprintf(stderr, "--half-planes is ssimulacra2 only\n");
        return 2;
    }
    const auto config = vqm::cli::MakeButteraugliConfig(options, original.size);
    vqm::ButteraugliPipeline pipeline(config);
    const vqm::ButteraugliScore result =
        vqm::ComputeWorstBackground(pipeline, original, distorted);
    if (options.raw) std::printf("%.8f\n", result.distance_p3);
    std::printf("%.8f\n", result.score);
    return 0;
}

int RunSsimulacra2(
    const vqm::cli::Options& options, const vqm::Image& original,
    const vqm::Image& distorted
) {
    const auto config = vqm::cli::MakeSsimulacra2Config(options, original.size);
    vqm::Ssimulacra2Pipeline pipeline(config);
    const vqm::Ssimulacra2Result result =
        vqm::ComputeWorstBackground(pipeline, original, distorted);
    if (options.raw) vqm::cli::PrintRawNorms(result);
    std::printf("%.8f\n", result.Score());
    return 0;
}

std::string_view DeviceTypeName(vqm::DeviceType type) {
    switch (type) {
        case vqm::DeviceType::IntegratedGpu:
            return "integrated GPU";
        case vqm::DeviceType::DiscreteGpu:
            return "discrete GPU";
        case vqm::DeviceType::VirtualGpu:
            return "virtual GPU";
        case vqm::DeviceType::Cpu:
            return "CPU";
        case vqm::DeviceType::Other:
            return "other";
    }
    return "other";
}

void PrintDevices() {
    const auto devices = vqm::ListDevices();
    if (devices.empty()) {
        std::puts("No compatible Vulkan compute devices found.");
        return;
    }
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& [uuid, name, type] = devices[i];
        std::printf(
            "%zu  %s  (%s)  ", i, name.c_str(), DeviceTypeName(type).data()
        );
        for (std::size_t byte = 0; byte < uuid.size(); ++byte) {
            if (byte == 4 || byte == 6 || byte == 8 || byte == 10)
                std::putchar('-');
            std::printf("%02x", static_cast<unsigned>(uuid[byte]));
        }
        std::putchar('\n');
    }
}

std::optional<vqm::DeviceUuid> SelectDevice(std::optional<unsigned> index) {
    if (!index) return std::nullopt;
    const auto devices = vqm::ListDevices();
    if (*index >= devices.size())
        throw std::out_of_range(
            std::format(
                "--device {} is unavailable. Run --list-devices to see "
                "available "
                "indexes.",
                *index
            )
        );
    const auto uuid = devices[*index].uuid;
    vqm::SetDefaultDevice(uuid);
    return uuid;
}

int RunCliMain(int argc, char** argv) try {
    const std::string_view program = argc > 0 ? argv[0] : "vqm";
    std::vector<std::string_view> args;
    args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

    const vqm::cli::ParseResult result = vqm::cli::ParseOptions(args);
    if (std::holds_alternative<vqm::cli::HelpRequested>(result)) {
        std::fputs(vqm::cli::Usage(program).c_str(), stdout);
        return 0;
    }
    if (std::holds_alternative<vqm::cli::DeviceListRequested>(result)) {
        PrintDevices();
        return 0;
    }
    if (const auto* parse_error = std::get_if<vqm::cli::ParseError>(&result)) {
        std::fprintf(
            stderr, "%s\n%s", parse_error->message.c_str(),
            vqm::cli::Usage(program).c_str()
        );
        return 2;
    }
    const vqm::cli::Options& options = std::get<vqm::cli::Options>(result);
    const auto metric = options.metrics.front();
    const auto device_uuid = SelectDevice(options.device_index);

    if (options.metrics.size() > 1 || options.start_frame ||
        options.frame_count || options.json || options.verbose ||
        metric == vqm::cli::Metric::Cvvdp ||
        metric == vqm::cli::Metric::PsnrHvs ||
        metric == vqm::cli::Metric::VmafV0 ||
        metric == vqm::cli::Metric::VmafV1 || options.iterations != 1 ||
        options.warmup != 0)
        return vqm::cli::RunSequences(options, device_uuid);

    if (vqm::cli::TryRunBestSourcePair(options, device_uuid)) return 0;

    std::string error;
    std::optional<vqm::Image> original =
        vqm::image_io::LoadImage(options.original, error);
    if (!original) {
        std::fprintf(
            stderr, "Could not load original image: %s\n%s\n",
            options.original.c_str(), error.c_str()
        );
        return 1;
    }
    // No 8x8 pre-check here: Ssimulacra2Pipeline's constructor enforces
    // kMinDimension and throws with the same message this used to print.
    std::optional<vqm::Image> distorted =
        vqm::image_io::LoadImage(options.distorted, error);
    if (!distorted) {
        std::fprintf(
            stderr, "Could not load distorted image: %s\n%s\n",
            options.distorted.c_str(), error.c_str()
        );
        return 1;
    }
    if (original->size != distorted->size) {
        std::fprintf(stderr, "Image size mismatch\n");
        return 1;
    }

    options.input.ApplyTo(original->format);
    options.input.ApplyTo(distorted->format);

    if (vqm::cli::HasY4mInput(options))
        return vqm::cli::RunSequences(options, device_uuid);

    if (metric == vqm::cli::Metric::Butteraugli) {
        return RunButteraugli(options, *original, *distorted);
    }
    return RunSsimulacra2(options, *original, *distorted);
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    const int status = RunCliMain(argc, argv);
    // Destroy automatic pipelines before releasing the retained device, while
    // driver exit handlers are still idle. Long video batches can initialize
    // additional driver components after the fallback exit handler is installed.
    try {
        vqm::internal::ShutdownDefaultVulkanContext();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CLI shutdown: %s\n", error.what());
        return 1;
    }
    return status;
}
