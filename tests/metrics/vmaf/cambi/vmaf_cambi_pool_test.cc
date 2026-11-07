// CAMBI: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>
#include "support.h"
#include "metrics/vmaf/cambi/vmaf_cambi.h"
#include "vulkan/vulkan_runtime.h"

namespace {
using namespace vqm::internal;
class PoolDevice {
public:
    explicit PoolDevice(std::size_t capacity)
        : runtime_(AcquireVulkanContext(), Config()), capacity_(capacity) {
        std::uint32_t cursor = static_cast<std::uint32_t>(capacity);
        buffers_.histogram = cursor;
        cursor += 5 * 256;
        buffers_.state = cursor;
        cursor += 5 * 2;
        buffers_.partials = cursor;
        cursor +=
            static_cast<std::uint32_t>(VmafCambiPool::PartialCount(capacity)) +
            4;
        buffers_.result = cursor;
        cursor += 5;
        runtime_.CreateArena(Bytes(cursor));
        runtime_.CreateStaging(Bytes(std::max<std::size_t>(cursor, capacity)));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, Bytes(cursor));
        pool_.emplace(runtime_);
    }
    float Compute(const std::vector<float>& values, double ratio) {
        CHECK(values.size() <= capacity_);
        std::copy(values.begin(), values.end(), runtime_.staging_map());
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{VulkanRuntime::CopyRange{0, 0, values.size()}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        pool_->Record(values.size(), buffers_, ratio);
        runtime_.RecordReadback(buffers_.result, 0, 1);
        runtime_.RecordReadback(buffers_.state, 1, 2);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        // Verify the selected set independently of floating-point summation.
        // This also catches wrong ordering for tiny values whose mean could
        // otherwise fit inside the numeric oracle's absolute tolerance.
        std::array<std::uint32_t, 2> selection{};
        std::memcpy(
            selection.data(), runtime_.staging_map() + 1, sizeof(selection)
        );
        std::size_t above = 0, equal = 0, zero = 0;
        for (float value : values) {
            std::uint32_t key = 0;
            std::memcpy(&key, &value, sizeof(key));
            key &= 0x7fffffffu;
            above += key > selection[0];
            equal += key == selection[0];
            zero += key == 0;
        }
        const auto selected = VmafCambiPool::SelectedCount(values.size(), ratio);
        if (selection[1] != 0) {
            CHECK(above + selection[1] == selected);
            CHECK(selection[1] <= equal);
        } else if (selection[0] == 0) {
            // Exact zero contributes nothing and need not enter the sum.
            CHECK(above <= selected);
            CHECK(selected <= above + zero);
        } else {
            CHECK(above == selected);
        }
        return runtime_.staging_map()[0];
    }

    std::array<float, 5> ComputeBatch(
        const std::array<std::vector<float>, 5>& maps, double ratio,
        bool allow_packed
    ) {
        std::array<std::size_t, 5> counts{};
        std::size_t total = 0;
        for (std::size_t i = 0; i < maps.size(); ++i) {
            counts[i] = maps[i].size();
            CHECK(total + counts[i] <= capacity_);
            std::copy(
                maps[i].begin(), maps[i].end(), runtime_.staging_map() + total
            );
            total += counts[i];
        }
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        const std::array copies{VulkanRuntime::CopyRange{0, 0, total}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        pool_->RecordBatch(counts, buffers_, ratio, allow_packed);
        runtime_.RecordReadback(buffers_.result, 0, 5);
        runtime_.RecordReadback(buffers_.state, 5, 10);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        std::array<float, 5> result{};
        std::copy_n(runtime_.staging_map(), 5, result.begin());
        std::array<std::uint32_t, 10> selection{};
        std::memcpy(
            selection.data(), runtime_.staging_map() + 5, sizeof(selection)
        );
        for (std::size_t scale = 0; scale < maps.size(); ++scale) {
            std::size_t above = 0, equal = 0, zero = 0;
            for (float value : maps[scale]) {
                std::uint32_t key = 0;
                std::memcpy(&key, &value, sizeof(key));
                key &= 0x7fffffffu;
                above += key > selection[2 * scale];
                equal += key == selection[2 * scale];
                zero += key == 0;
            }
            const auto selected =
                VmafCambiPool::SelectedCount(counts[scale], ratio);
            const auto threshold = selection[2 * scale];
            const auto rank = selection[2 * scale + 1];
            if (rank != 0) {
                CHECK(above + rank == selected);
                CHECK(rank <= equal);
            } else if (threshold == 0) {
                CHECK(above <= selected);
                CHECK(selected <= above + zero);
            } else {
                CHECK(above == selected);
            }
        }
        return result;
    }

private:
    static VulkanRuntime::Config Config() {
        VulkanRuntime::Config config;
        config.streaming = false;
        config.push_constant_size = VmafCambiPool::PushBytes();
        return config;
    }
    VulkanRuntime runtime_;
    std::size_t capacity_;
    VmafCambiPoolBuffers buffers_{};
    std::optional<VmafCambiPool> pool_;
};

double CpuPool(std::vector<float> values, double ratio) {
    const auto n = VmafCambiPool::SelectedCount(values.size(), ratio);
    std::sort(values.begin(), values.end(), std::greater<float>());
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) sum += static_cast<double>(values[i]);
    return sum / n;
}
}  // namespace

VQM_TEST(TestVmafCambiPoolCount) {
    CHECK(VmafCambiPool::SelectedCount(1) == 1);
    CHECK(VmafCambiPool::SelectedCount(2) == 1);
    CHECK(VmafCambiPool::SelectedCount(5) == 3);
    CHECK(VmafCambiPool::SelectedCount(7) == 4);
    CHECK(VmafCambiPool::SelectedCount(7, 1.0) == 7);
    CHECK_THROWS_AS(VmafCambiPool::SelectedCount(0), std::invalid_argument);
    CHECK_THROWS_AS(VmafCambiPool::SelectedCount(5, 0), std::invalid_argument);
    CHECK_THROWS_AS(VmafCambiPool::SelectedCount(5, 1.1), std::invalid_argument);
    CHECK_THROWS_AS(
        VmafCambiPool::SelectedCount(
            5, std::numeric_limits<double>::quiet_NaN()
        ),
        std::invalid_argument
    );
}

VQM_TEST(TestVmafCambiPoolExactTopSet) {
    if (!vqm::test::GpuAvailable()) return;
    PoolDevice device(8193);
    for (std::size_t n : {1u, 2u, 3u, 5u, 7u, 255u, 256u, 257u, 1024u, 8193u}) {
        for (int pattern = 0; pattern < 10; ++pattern) {
            std::vector<float> values(n);
            for (std::size_t i = 0; i < n; ++i) {
                if (pattern == 0) values[i] = 0.0f;
                if (pattern == 1) values[i] = i % 2 ? -0.0f : 0.0f;
                if (pattern == 2) values[i] = static_cast<float>((i * 37) % 11);
                if (pattern == 3) values[i] = i % 3 ? 5.0f : 100.0f;
                if (pattern == 4)
                    values[i] =
                        i % 4 ? 1e-20f : 1e6f + static_cast<float>(i % 13);
                // Normal positive values in the first radix byte's zero
                // bucket must not be confused with exact zero.
                if (pattern == 5)
                    values[i] =
                        i % 4 == 0
                            ? 0.0f
                            : std::numeric_limits<float>::min() *
                                  (1.0f + static_cast<float>(i % 7) / 8.0f);
                if (pattern == 6)
                    values[i] =
                        1.0f +
                        std::ldexp(static_cast<float>((i * 43) % 2048), -23);
                if (pattern == 7) values[i] = i % 5 ? 1.0f : 256.0f;
                if (pattern >= 8) {
                    // Scatter deterministic finite keys through all four
                    // radix bytes, with or without repeated exact ties.
                    std::uint32_t key = static_cast<std::uint32_t>(
                        (pattern == 9 ? i % 17 : i) * 2654435761u
                    );
                    key = 0x3e000000u | (key & 0x01ffffffu);
                    std::memcpy(&values[i], &key, sizeof(key));
                }
            }
            for (double ratio : {0.01, 0.2, 0.6, 1.0}) {
                const auto expected = CpuPool(values, ratio);
                CHECK_NEAR(
                    device.Compute(values, ratio), expected,
                    std::max(1e-7, std::fabs(expected) * 2e-6)
                );
            }
        }
    }
}

VQM_TEST(TestVmafCambiPoolBatchIndependentStates) {
    if (!vqm::test::GpuAvailable()) return;
    PoolDevice device(2048);
    constexpr std::array<std::size_t, 5> sizes{257, 256, 255, 33, 17};
    std::array<std::vector<float>, 5> maps;
    for (std::size_t scale = 0; scale < maps.size(); ++scale) {
        maps[scale].resize(sizes[scale]);
        for (std::size_t i = 0; i < maps[scale].size(); ++i) {
            switch (scale) {
                case 0:
                    maps[scale][i] = i % 2 ? -0.0f : 0.0f;
                    break;
                case 1:
                    maps[scale][i] = i % 3 ? 5.0f : 100.0f;
                    break;
                case 2:
                    maps[scale][i] =
                        1.0f + std::ldexp(float((i * 43) % 2048), -23);
                    break;
                case 3:
                    maps[scale][i] = i % 4 ? std::numeric_limits<float>::min() *
                                                 (1.0f + float(i % 7) / 8.0f)
                                           : 0.0f;
                    break;
                case 4:
                    maps[scale][i] = float((i * 37) % 11);
                    break;
            }
        }
    }
    for (double ratio : {0.01, 0.2, 0.6, 1.0}) {
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            const auto packed = device.ComputeBatch(maps, ratio, true);
            const auto rectangular = device.ComputeBatch(maps, ratio, false);
            for (std::size_t scale = 0; scale < maps.size(); ++scale) {
                const auto expected = CpuPool(maps[scale], ratio);
                CHECK_NEAR(
                    packed[scale], expected,
                    std::max(1e-7, std::fabs(expected) * 2e-6)
                );
                CHECK(packed[scale] == rectangular[scale]);
                std::reverse(maps[scale].begin(), maps[scale].end());
            }
            // Exercise packed boundaries even when the map sizes are not in
            // descending order, and reuse the same scratch across executions.
            std::rotate(maps.begin(), maps.begin() + 1, maps.end());
        }
    }
}
