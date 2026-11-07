// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include "support.h"
#include "metrics/vmaf/speed/vmaf_speed.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

VQM_TEST(TestVmafSpeedFilterGeometry) {
    using vqm::internal::VmafSpeedFilter;
    CHECK(VmafSpeedFilter::OutputPixels({93, 115}) == 35);
    CHECK(VmafSpeedFilter::ScratchWords({93, 115}) == 651);
    CHECK_THROWS_AS(
        VmafSpeedFilter::OutputPixels({79, 80}), std::invalid_argument
    );
    CHECK_THROWS_AS(
        VmafSpeedFilter::OutputPixels({80, 79}), std::invalid_argument
    );
    CHECK_THROWS_AS(
        VmafSpeedFilter::OutputPixels(
            {std::numeric_limits<std::size_t>::max(), 80}
        ),
        std::invalid_argument
    );
}

VQM_TEST(TestVmafSpeedFilterAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::internal;
    constexpr vqm::Size size{93, 115};
    constexpr auto pixels = size.width * size.height;
    constexpr std::array<float, 35> expected_down{
        -80.61790466f, 59.70933914f, -25.25205612f, -1.751713276f, 51.94499207f,
        -0.762294054f, 22.35453415f, -47.81531525f, 43.50088501f,  14.59018898f,
        21.12699509f,  23.68074989f, -31.92807388f, 31.31268692f,  15.91640091f,
        -9.975277901f, 29.35453415f, -48.12480164f, 37.93721008f,  21.59018898f,
        21.71676826f,  25.06130409f, -38.01951218f, 29.5010128f,   18.73336601f,
        -22.96282005f, 34.00941849f, -35.28229141f, 25.01649666f,  26.24507523f,
        6.655025482f,  26.45368195f, -42.68013f,    28.57398415f,  20.57109833f};
    constexpr std::array<float, 35> expected{
        -60.28900528f, 42.60515976f,  -13.47323608f,  -11.75416946f,
        21.15114212f,  -3.39868331f,  11.76696396f,   -30.6170845f,
        24.24481773f,  -10.73427391f, 4.792377472f,   11.54940414f,
        -16.85333824f, 11.76281738f,  -6.647407532f,  -19.50656509f,
        18.0746727f,   -29.43687057f, 18.21868515f,   -3.658742905f,
        6.632383347f,  11.89983368f,  -21.38722801f,  12.50371552f,
        -4.952852249f, -26.29232025f, 21.06365204f,   -20.20169258f,
        9.580792427f,  1.42739296f,   -0.8487234116f, 14.34903526f,
        -25.22143364f, 13.2528286f,   -3.528898239f};
    const auto scratch = VmafSpeedFilter::ScratchWords(size);
    VmafSpeedFilterBuffers b{0, static_cast<std::uint32_t>(pixels),
                             static_cast<std::uint32_t>(pixels + scratch),
                             static_cast<std::uint32_t>(pixels + scratch + 35)};
    VulkanRuntime::Config config;
    config.base_size = size;
    config.base_pixels = pixels;
    config.push_constant_size = VmafSpeedFilter::PushBytes();
    config.streaming = false;
    VulkanRuntime runtime(AcquireVulkanContext(), config);
    runtime.CreateArena(Bytes(pixels + scratch + 70));
    runtime.CreateStaging(Bytes(pixels + 70));
    runtime.CreateCommandObjects();
    runtime.CreateDescriptors(0, Bytes(pixels + scratch + 70));
    VmafSpeedFilter filter(runtime);
    for (std::size_t y = 0; y < size.height; y++)
        for (std::size_t x = 0; x < size.width; x++)
            runtime.staging_map()[y * size.width + x] =
                float((7 * x + 13 * y + (x / 7) * 23 + (y / 11) * 31) % 251) -
                120.0f + float((x + y) % 4) * .25f;
    runtime.ResetCommandPool();
    runtime.BeginCommandBuffer();
    const std::array copies{VulkanRuntime::CopyRange{0, b.input, pixels}};
    runtime.RecordStagedCopies(copies);
    runtime.BindDescriptorSet();
    filter.Record(size, b);
    runtime.RecordReadback(b.down, pixels, 70);
    runtime.EndCommandBuffer();
    runtime.SubmitAndWait();
    for (std::size_t i = 0; i < expected.size(); i++) {
        CHECK_NEAR(runtime.staging_map()[pixels + i], expected_down[i], 1e-4);
        CHECK_NEAR(runtime.staging_map()[pixels + 35 + i], expected[i], 1e-4);
    }
}

VQM_TEST(TestVmafSpeedFilterBatchMatchesIndependentPlanes) {
    if (!vqm::test::GpuAvailable()) return;
    using namespace vqm::internal;
    for (const auto size : {vqm::Size{93, 115}, vqm::Size{160, 81}}) {
        const auto pixels = static_cast<std::uint32_t>(size.width * size.height);
        const auto scratch =
            static_cast<std::uint32_t>(VmafSpeedFilter::ScratchWords(size));
        const auto output =
            static_cast<std::uint32_t>(VmafSpeedFilter::OutputPixels(size));
        const std::uint32_t single_scratch = 4 * pixels;
        const std::uint32_t single_down = single_scratch + scratch;
        const std::uint32_t single_result = single_down + 4 * output;
        const std::uint32_t batch_scratch = single_result + 4 * output;
        const std::uint32_t batch_down = batch_scratch + 4 * scratch;
        const std::uint32_t batch_result = batch_down + 4 * output;
        const std::uint32_t words = batch_result + 4 * output;
        VulkanRuntime::Config config;
        config.base_size = size;
        config.base_pixels = pixels;
        config.push_constant_size = VmafSpeedFilter::PushBytes();
        config.streaming = false;
        VulkanRuntime runtime(AcquireVulkanContext(), config);
        runtime.CreateArena(Bytes(words));
        runtime.CreateStaging(Bytes(4 * pixels + 16 * output));
        runtime.CreateCommandObjects();
        runtime.CreateDescriptors(0, Bytes(words));
        VmafSpeedFilter filter(runtime);
        for (std::uint32_t plane = 0; plane < 4; ++plane)
            for (std::uint32_t y = 0; y < size.height; ++y)
                for (std::uint32_t x = 0; x < size.width; ++x)
                    runtime.staging_map()[plane * pixels + y * size.width + x] =
                        plane == 3
                            ? 0.25f
                            : static_cast<float>(
                                  (7 * x + 13 * y + 47 * plane + (x / 7) * 23) %
                                  251
                              ) - 125.0f +
                                  0.25f * float((x + y + plane) % 4);
        runtime.BeginCommandBuffer();
        const std::array copies{VulkanRuntime::CopyRange{0, 0, 4 * pixels}};
        runtime.RecordStagedCopies(copies);
        runtime.BindDescriptorSet();
        std::array<VmafSpeedFilterBuffers, 4> batch{};
        for (std::uint32_t plane = 0; plane < 4; ++plane) {
            filter.Record(
                size,
                {plane * pixels, single_scratch, single_down + plane * output,
                 single_result + plane * output}
            );
            batch[plane] = {plane * pixels, batch_scratch + plane * scratch,
                            batch_down + plane * output,
                            batch_result + plane * output};
        }
        filter.RecordBatch(size, batch);
        runtime.RecordReadback(single_down, 4 * pixels, 8 * output);
        runtime.RecordReadback(batch_down, 4 * pixels + 8 * output, 8 * output);
        runtime.EndCommandBuffer();
        runtime.SubmitAndWait();
        for (std::uint32_t i = 0; i < 8 * output; ++i)
            CHECK(
                runtime.staging_map()[4 * pixels + i] ==
                runtime.staging_map()[4 * pixels + 8 * output + i]
            );
    }
}
