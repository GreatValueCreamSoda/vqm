// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>

#include "cvvdp.h"
#include "vqm_memory.h"

namespace {
bool SameAllocation(
    const vqm::MemoryAllocation& plan, const vqm::MemoryAllocation& actual
) {
    return plan.buffer_bytes == actual.buffer_bytes &&
           plan.allocation_bytes == actual.allocation_bytes &&
           plan.memory_type == actual.memory_type && plan.heap == actual.heap;
}
std::optional<vqm::MetricMemoryRequirements> Query(
    const vqm::CvvdpConfig& config
) {
    try {
        return vqm::GetMemoryRequirements(config);
    } catch (const std::invalid_argument& error) {
        std::printf(
            "%zux%zu %u fps: unsupported: %s\n", config.size.width,
            config.size.height, config.fps_numerator, error.what()
        );
        return std::nullopt;
    }
}
bool Probe(vqm::Size size, std::uint32_t fps) {
    vqm::CvvdpConfig config;
    config.size = size;
    config.fps_numerator = fps;
    config.gpu_resident_images = 0;
    const auto plan = Query(config);
    if (!plan) return true;  // Device-limit rejection, before allocation.
    const vqm::CvvdpPipeline pipeline(config);
    const auto usage = pipeline.GetMemoryUsage();
    const auto& scenario = usage.uses_staging ? plan->staged : plan->direct;
    bool match = scenario.supported &&
                 SameAllocation(scenario.arena, usage.arena) &&
                 SameAllocation(scenario.staging, usage.staging) &&
                 scenario.auxiliary.size() == usage.auxiliary.size();
    std::uint64_t history = 0;
    for (std::size_t index = 0; index < usage.auxiliary.size(); ++index) {
        history += usage.auxiliary[index].allocation_bytes;
        match =
            match &&
            SameAllocation(scenario.auxiliary.at(index), usage.auxiliary[index]);
    }
    std::printf(
        "%zux%zu %u fps: arena=%llu staging=%llu history=%llu "
        "query_matches=%s\n",
        size.width, size.height, fps,
        static_cast<unsigned long long>(usage.arena.allocation_bytes),
        static_cast<unsigned long long>(usage.staging.allocation_bytes),
        static_cast<unsigned long long>(history), match ? "true" : "false"
    );
    return match;
}
int Run() {
    for (const auto size : {vqm::Size{1920, 1080}, vqm::Size{3840, 2160}}) {
        for (const auto fps : {24u, 30u, 60u, 120u}) {
            if (!Probe(size, fps)) return 1;
        }
    }
    return 0;
}
}  // namespace
int main() {
    try {
        return Run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "allocation/query failed: %s\n", error.what());
        return 1;
    }
}
