// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <cstddef>
#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include "bench.h"
#include "vulkan/vulkan_support.h"

namespace bench_driver {

static int RunBenchMain(int argc, char** argv) {
    const std::string_view program = argc > 0 ? argv[0] : "vqm_bench";
    std::vector<std::string_view> args;
    args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

    BenchOptions options;
    std::string error;
    if (!ParseArgs(args, options, error)) {
        std::fprintf(stderr, "%s\n%s", error.c_str(), Usage(program).c_str());
        return 2;
    }
    if (options.help) {
        std::fputs(Usage(program).c_str(), stdout);
        return 0;
    }
    if (options.device_info) return PrintDeviceInfo();
    if (!options.pairs_file.empty()) {
        if (!inmem::ReadPairsFile(
                options.pairs_file, options.pairs, &error, false
            )) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 2;
        }
    }
    if (options.pairs.empty()) {
        std::fprintf(stderr, "%s", Usage(program).c_str());
        return 2;
    }

    if (!options.metric_pair.empty()) return RunMetricPair(options);
    if (options.metric == Metric::Cvvdp) return RunCvvdp(options);
    if (options.metric == Metric::PsnrHvs) return RunPsnrHvs(options);
    if (options.IsVmaf()) {
        return RunVmafForMetric(options);
    }
    if (options.resident_ring > 0) return RunResidentRing(options);
    if (options.pool) return RunPool(options);

    // The throughput window and the concurrent batch come first: both exist so
    // one process can keep the device busy, and neither prints the latency
    // table. The stage timers are a process-wide accumulator, so they cannot
    // be attributed to one pair once several pipelines share the device; the
    // window reports the one-worker figure instead of a per-pair CSV.
    const bool concurrent = options.throughput_seconds > 0.0 ||
                            (options.score_only && options.workers > 1);
    if (concurrent && !options.stages_csv_path.empty()) {
        std::fprintf(
            stderr,
            "--stages-csv needs the latency path: the stage timers are one "
            "process-wide accumulator and cannot be split per pair across "
            "concurrent workers\n"
        );
        return 2;
    }
    if (options.throughput_seconds > 0.0) {
        return RunWindowForMetric(options);
    }
    if (options.score_only && options.workers > 1) {
        return RunBatchWorkersForMetric(options);
    }
    if (options.metric == Metric::Butteraugli) return RunButteraugli(options);

    return RunSsimulacra2(options);
}

}  // namespace bench_driver

int main(int argc, char** argv) {
    const int status = bench_driver::RunBenchMain(argc, argv);
    // RunBenchMain's automatic pipelines and worker pools are destroyed before
    // releasing the retained device, while driver exit handlers are still idle.
    try {
        vqm::internal::ShutdownDefaultVulkanContext();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "benchmark shutdown: %s\n", e.what());
        return 1;
    }
    return status;
}
