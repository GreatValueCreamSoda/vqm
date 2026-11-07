// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Promoted from the 10,000-matrix stress-provenance driver. This is a
// correctness tool, with no timers or independent metric arithmetic.
// Historical artifacts: artifacts/reports/vmaf-v1-pooling-parallelism/
// stress-provenance/{eigen_stress.cc,eigen_stress_legacy_guard.cc}.
// Legacy mode uses the unchanged current forced-QR blob with 2604 words;
// it does not emulate the archived 2602-word legacy-guard layout.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "metrics/vmaf/speed/vmaf_speed.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

namespace {
using namespace vqm::internal;
using Matrix = std::array<float, 49>;
enum class Solver { Original, Phases, Legacy };
struct Options {
    std::filesystem::path input, output;
    Solver solver = Solver::Original;
    std::string solver_name = "original";
    std::uint32_t width = 0;
};
constexpr std::string_view kUsage =
    "usage: eigen_stress --input MATRICES.bin [--out RESULTS.csv]\n"
    "                    [--solver original|phases|legacy] [--width "
    "auto|128|512]\n";

Options Parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("option needs a value");
        const std::string_view value = argv[++i];
        if (flag == "--input")
            options.input = value;
        else if (flag == "--out")
            options.output = value;
        else if (flag == "--solver" || flag == "--mode") {
            options.solver_name = value;
            if (value == "original")
                options.solver = Solver::Original;
            else if (value == "phases")
                options.solver = Solver::Phases;
            else if (value == "legacy")
                options.solver = Solver::Legacy;
            else
                throw std::invalid_argument("unknown solver");
        } else if (flag == "--width") {
            if (value == "auto")
                options.width = 0;
            else if (value == "128")
                options.width = 128;
            else if (value == "512")
                options.width = 512;
            else
                throw std::invalid_argument("width must be auto, 128 or 512");
        } else
            throw std::invalid_argument("unknown option");
    }
    if (options.input.empty())
        throw std::invalid_argument("--input is required");
    if (options.output.empty()) {
        options.output =
            std::filesystem::path(
                "artifacts/reports/vmaf-speed-parallelism/stress"
            ) /
            (options.solver_name + "-" +
             (options.width == 0 ? "auto" : std::to_string(options.width)) +
             ".csv");
    }
    if (std::filesystem::weakly_canonical(options.input) ==
        std::filesystem::weakly_canonical(options.output))
        throw std::invalid_argument("input and output must differ");
    return options;
}

std::vector<Matrix> Load(const std::filesystem::path& path) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(sizeof(Matrix) == 196);
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error(
            "VQMTRI01 driver requires a little-endian host"
        );
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open dataset");
    const auto bytes = input.tellg();
    input.seekg(0);
    std::array<char, 8> magic{};
    std::uint32_t count = 0;
    input.read(magic.data(), magic.size());
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input || std::string_view(magic.data(), magic.size()) != "VQMTRI01" ||
        count == 0 || bytes != std::streamoff(12) + std::streamoff(count) * 196)
        throw std::runtime_error("invalid VQMTRI01 header or length");
    std::vector<Matrix> data(count);
    input.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(Matrix))
    );
    if (!input) throw std::runtime_error("error reading dataset");
    return data;
}

class StressDevice {
public:
    StressDevice(std::shared_ptr<VulkanContext> context, const Options& options)
        : runtime_(context, Config()),
          phases_(options.solver == Solver::Phases),
          words_(phases_ ? 2604u + 2u * 56u : 2604u) {
        const auto& limits = context->limits;
        const bool supports_wide =
            limits.maxComputeWorkGroupInvocations >= 512 &&
            limits.maxComputeWorkGroupSize[0] >= 512;
        if (options.width == 512 && !supports_wide)
            throw std::runtime_error("device does not support width 512");
        if (options.solver == Solver::Legacy && options.width == 512)
            throw std::invalid_argument(
                "legacy QR oracle uses its fixed 128-lane blob"
            );
        wide_ = options.solver != Solver::Legacy &&
                (options.width == 512 || (options.width == 0 && supports_wide));
        runtime_.CreateArena(Bytes(words_));
        runtime_.CreateStaging(Bytes(2 * words_));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(words_));
        if (phases_) {
            const std::array names{wide_ ? "vmaf_speed_stats_solver_prepare_wide"
                                         : "vmaf_speed_stats_solver_prepare",
                                   "vmaf_speed_stats_solver_bisect",
                                   "vmaf_speed_stats_solver_finish"};
            pipelines_ = runtime_.CreatePipelines(names);
        } else {
            pipelines_[0] = runtime_.CreatePipeline(
                options.solver == Solver::Legacy
                    ? "vmaf_speed_stats_eigen_legacy"
                : wide_ ? "vmaf_speed_stats_eigen_wide"
                        : "vmaf_speed_stats_eigen"
            );
        }
    }
    ~StressDevice() { runtime_.DestroyPipelines(pipelines_); }
    const float* Compute(std::span<const Matrix> pair) {
        std::fill_n(runtime_.staging_map(), words_, 0.0f);
        for (std::size_t ch = 0; ch < pair.size(); ++ch) {
            auto* matrix = runtime_.staging_map() + 50 + ch * 625;
            for (std::size_t i = 0; i < 25; ++i) {
                matrix[i * 25 + i] = pair[ch][i];
                if (i < 24)
                    matrix[i * 25 + i + 1] = matrix[(i + 1) * 25 + i] =
                        pair[ch][25 + i];
            }
        }
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{VulkanRuntime::CopyRange{0, 0, words_}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        std::array<std::uint32_t, 9> pc{};
        static_assert(sizeof(pc) == VmafSpeedStats::PushBytes());
        if (phases_) pc[8] = 1;
        runtime_.ComputeBarrier();
        runtime_.Dispatch(pipelines_[0], pc.data(), sizeof(pc), 2, 2, 1);
        runtime_.ComputeBarrier();
        if (phases_) {
            runtime_.Dispatch(pipelines_[1], pc.data(), sizeof(pc), 25, 2, 1);
            runtime_.ComputeBarrier();
            runtime_.Dispatch(pipelines_[2], pc.data(), sizeof(pc), 2, 1, 1);
            runtime_.ComputeBarrier();
        }
        runtime_.RecordReadback(0, words_, words_);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        return runtime_.staging_map() + words_;
    }
    std::uint32_t width() const { return wide_ ? 512u : 128u; }

private:
    static VulkanRuntime::Config Config() {
        VulkanRuntime::Config config;
        config.base_size = {25, 25};
        config.base_pixels = 625;
        config.push_constant_size = VmafSpeedStats::PushBytes();
        config.streaming = false;
        return config;
    }
    VulkanRuntime runtime_;
    bool phases_, wide_ = false;
    std::uint32_t words_;
    std::array<VkPipeline, 3> pipelines_{};
};

void WriteRow(
    std::ostream& output, std::size_t id, const Matrix& input,
    const float* result, std::size_t ch
) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(input.data());
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < sizeof(input); ++i)
        hash = (hash ^ bytes[i]) * 1099511628211ull;
    const float spectral = result[2600 + ch], factor = result[2602 + ch];
    output << std::dec << id << ',' << std::hex << std::setw(16)
           << std::setfill('0') << hash << std::dec << ',' << spectral << ','
           << factor << ',' << (spectral == 0.0f ? factor : spectral);
    for (std::size_t i = 0; i < 25; ++i)
        output << ',' << std::hex << std::setw(8) << std::setfill('0')
               << std::bit_cast<std::uint32_t>(result[2550 + ch * 25 + i]);
    output << std::dec << '\n';
}

void Run(const Options& options) {
    const auto data = Load(options.input);
    auto context = AcquireVulkanContext();
    if (!context) throw std::runtime_error("no usable Vulkan device");
    StressDevice device(context, options);
    if (!options.output.parent_path().empty())
        std::filesystem::create_directories(options.output.parent_path());
    std::ofstream output(options.output);
    if (!output) throw std::runtime_error("cannot open output CSV");
    output << "id,input_fnv1a64,spectral_status,factor_status,combined_status";
    for (int i = 0; i < 25; ++i) output << ",e" << i;
    output << '\n';
    for (std::size_t first = 0; first < data.size(); first += 2) {
        const auto count = std::min<std::size_t>(2, data.size() - first);
        const auto* result =
            device.Compute(std::span(data).subspan(first, count));
        for (std::size_t ch = 0; ch < count; ++ch)
            WriteRow(output, first + ch, data[first + ch], result, ch);
    }
    if (!output) throw std::runtime_error("error writing output CSV");
    std::cout << "cases=" << data.size() << " output=" << options.output
              << " solver=" << options.solver_name << " width=" << device.width()
              << " eigen_bits=fp32_hex\n";
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << kUsage;
        return 0;
    }
    int status = 0;
    try {
        Run(Parse(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "eigen_stress: " << error.what() << '\n' << kUsage;
        status = 1;
    }
    ShutdownDefaultVulkanContext();
    return status;
}
