// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "bench.h"
#include "input.h"
#include "io.h"
#include "vulkan/vulkan_support.h"

namespace bench_driver {

// One timed load of both files, for the decode-ahead queue. The metric's
// timer starts only after this returns, so overlap with the previous pair's
// scoring does not enter core_ms.
bool LoadOnce(const Pair& pair, DecodedPair& out, std::string& error) {
    const Clock::time_point start = Clock::now();
    std::optional<vqm::Image> original =
        vqm::image_io::LoadImage(pair.original, error);
    if (!original) return false;
    std::optional<vqm::Image> distorted =
        vqm::image_io::LoadImage(pair.distorted, error);
    if (!distorted) return false;
    out.decode_ms = 1000.0 * Seconds(Clock::now() - start);
    out.original = std::move(*original);
    out.distorted = std::move(*distorted);
    out.width = static_cast<int>(out.original.size.width);
    out.height = static_cast<int>(out.original.size.height);
    return true;
}

// Times `attempts` loads of both files and returns the median of the sums.
// The images from the last load are kept for the metric. A scoring run asks
// for one load: decode_ms is not reported there, and three PNG decodes per
// pair would triple a corpus run's wall time for nothing.
bool MeasureDecode(
    const Pair& pair, int attempts, double* decode_ms, vqm::Image* original,
    vqm::Image* distorted, std::string* error
) {
    std::vector<double> load_times;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        const Clock::time_point start = Clock::now();
        std::optional<vqm::Image> loaded_original =
            vqm::image_io::LoadImage(pair.original, *error);
        std::optional<vqm::Image> loaded_distorted;
        if (loaded_original) {
            loaded_distorted = vqm::image_io::LoadImage(pair.distorted, *error);
        }
        if (loaded_original && loaded_distorted) {
            *original = std::move(*loaded_original);
            *distorted = std::move(*loaded_distorted);
        }
        load_times.push_back(Seconds(Clock::now() - start));
        if (!loaded_distorted) return false;
    }
    *decode_ms = 1000.0 * Median(load_times);
    return true;
}

PairDecoder::PairDecoder(const BenchOptions& options)
    : attempts_(options.score_only ? 1 : 3) {
    int threads = options.decode_threads;
    if (threads == 0) {
        threads = static_cast<int>(
            std::min(4u, std::max(1u, std::thread::hardware_concurrency()))
        );
    }
    if (threads > 1) prefetch_.emplace(options.pairs, threads, LoadOnce);
}

bool PairDecoder::Decode(
    const Pair& pair, DecodedPair& out, std::string& error
) {
    if (prefetch_) return prefetch_->Take(out, &error);
    if (!MeasureDecode(
            pair, attempts_, &out.decode_ms, &out.original, &out.distorted,
            &error
        ))
        return false;
    out.width = static_cast<int>(out.original.size.width);
    out.height = static_cast<int>(out.original.size.height);
    return true;
}

bool WritePairsCsv(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream file(path);
    if (!file) return false;
    file << "original,distorted,width,height,decode_ms,core_ms,score\n";
    for (const Row& row : rows) {
        file << std::format(
            "{},{},{},{},{:.6f},{:.6f},{:.6f}\n",
            inmem::CsvField(row.pair.original),
            inmem::CsvField(row.pair.distorted), row.width, row.height,
            row.decode_ms, row.core_ms, row.score
        );
    }
    file.close();
    return static_cast<bool>(file);
}

bool WriteStagesCsv(
    const std::string& path,
    const std::array<double, vqm::internal::kStageCount>& totals,
    double total_iterations, double total_core_seconds, int first_stage,
    int last_stage, bool concurrent
) {
    std::ofstream file(path);
    if (!file) return false;
    file << "stage,ms_per_iter,share\n";
    for (int s = first_stage; s <= last_stage; ++s) {
        const std::size_t index = static_cast<std::size_t>(s);
        const double seconds = totals[index];
        const double share =
            total_core_seconds > 0.0 ? seconds / total_core_seconds : 0.0;
        const std::string_view name =
            vqm::internal::StageName(static_cast<vqm::internal::Stage>(s));
        if (concurrent) {
            file << std::format(
                "{},{:.6f},\n", name, seconds / total_iterations * 1000.0
            );
        } else {
            file << std::format(
                "{},{:.6f},{:.6f}\n", name, seconds / total_iterations * 1000.0,
                share
            );
        }
    }
    if (concurrent) file << "untimed,,\n";
    file.close();
    return static_cast<bool>(file);
}

void PrintStages(
    const std::array<double, vqm::internal::kStageCount>& totals,
    double total_iterations, double total_core_seconds, int first_stage,
    int last_stage, bool concurrent
) {
    if (concurrent) {
        std::puts(
            "  Stage times are cumulative per call and may overlap; "
            "wall-clock shares and untimed are unavailable."
        );
    }
    double stage_sum = 0.0;
    for (int stage = first_stage; stage <= last_stage; ++stage) {
        const double seconds = totals[static_cast<std::size_t>(stage)];
        stage_sum += seconds;
        const std::string_view name =
            vqm::internal::StageName(static_cast<vqm::internal::Stage>(stage));
        if (concurrent) {
            std::printf(
                "  %-10.*s %8.2f ms/iter (cumulative)\n",
                static_cast<int>(name.size()), name.data(),
                seconds / total_iterations * 1000.0
            );
        } else {
            const double share = total_core_seconds > 0.0
                                     ? 100.0 * seconds / total_core_seconds
                                     : 0.0;
            std::printf(
                "  %-10.*s %6.1f%% %8.2f ms/iter\n",
                static_cast<int>(name.size()), name.data(), share,
                seconds / total_iterations * 1000.0
            );
        }
    }
    if (!concurrent) {
        const double untimed = std::max(total_core_seconds - stage_sum, 0.0);
        std::printf(
            "  %-10s %6.1f%% %8.2f ms/iter\n", "untimed",
            total_core_seconds > 0.0 ? 100.0 * untimed / total_core_seconds
                                     : 0.0,
            untimed / total_iterations * 1000.0
        );
    }
}

// The same overrides the CLI applies: replace what the decoder declared, on
// both sides of the pair.
void ApplyOverrides(
    const BenchOptions& options, vqm::Image* original, vqm::Image* distorted
) {
    options.input.ApplyTo(original->format);
    options.input.ApplyTo(distorted->format);
}

// Build borrowed views once, outside the timer, from the decoded planar images.
vqm::ImageView BorrowImage(const vqm::Image& image) {
    vqm::ImageView view;
    view.size = image.size;
    view.format = image.format;
    const auto planes = vqm::internal::PlaneLayout(image.format, image.size);
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        if (!planes[plane].present()) continue;
        view.planes[plane] = {
            reinterpret_cast<const std::byte*>(image.data.data()) +
                planes[plane].offset,
            planes[plane].width * planes[plane].sample_bytes};
    }
    return view;
}

// Report the actual device chosen by the process-wide context before timing.
// The raw driver version uses Vulkan's vendor-specific encoding.
int PrintDeviceInfo() {
    const auto context = vqm::internal::AcquireVulkanContext();
    if (!context) {
        std::fprintf(stderr, "vulkan: no usable compute device\n");
        return 1;
    }
    VkPhysicalDeviceIDProperties id{};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id;
    vkGetPhysicalDeviceProperties2(context->physical_device, &properties);
    std::string name;
    for (const char character :
         std::string_view(properties.properties.deviceName)) {
        const auto c = static_cast<unsigned char>(character);
        if (c == '"' || c == '\\') {
            name.push_back('\\');
            name.push_back(static_cast<char>(c));
        } else if (c < 32) {
            name += std::format("\\u{:04x}", c);
        } else {
            name.push_back(static_cast<char>(c));
        }
    }
    std::string uuid;
    for (const std::uint8_t byte : id.deviceUUID)
        uuid += std::format("{:02x}", byte);
    std::printf(
        "{\"name\":\"%s\",\"uuid\":\"%s\",\"driver_version\":%u}\n",
        name.c_str(), uuid.c_str(), properties.properties.driverVersion
    );
    return 0;
}

}  // namespace bench_driver
