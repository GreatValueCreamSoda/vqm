// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <cstddef>
#include <stdexcept>
#include "support.h"
#include "vqm_memory.h"
#include "vqm_shared.h"
#include "vqm_views.h"

namespace {
vqm::ImageView View(const vqm::Image& image) {
    vqm::ImageView out;
    out.size = image.size;
    out.format = image.format;
    const auto layout = vqm::internal::PlaneLayout(image.format, image.size);
    for (std::size_t i = 0; i < layout.size(); ++i) {
        if (!layout[i].present()) continue;
        out.planes[i] = {reinterpret_cast<const std::byte*>(
                             image.data.data() + layout[i].offset
                         ),
                         layout[i].width * layout[i].sample_bytes};
    }
    return out;
}
}  // namespace

VQM_TEST(TestSharedNonTemporalWorkspace) {
    if (!vqm::test::GpuAvailable()) {
        vqm::test::MarkSkip("no usable Vulkan device", true);
        return;
    }
    const auto original = vqm::test::MakeGradient(0);
    const auto distorted = vqm::test::MakeGradient(1);
    const auto other = vqm::test::MakeGradient(2);
    vqm::Ssimulacra2Config ssimu_config{.size = original.size};
    ssimu_config.gpu_resident_images = 0;
    vqm::ButteraugliConfig butter_config{.size = original.size};
    butter_config.gpu_resident_images = 0;
    vqm::Ssimulacra2Pipeline ssimu(ssimu_config);
    vqm::ButteraugliPipeline butter(butter_config);
    const double identity_expected = ssimu.Compute(original, original).Score();
    const double ssimu_expected = ssimu.Compute(original, distorted).Score();
    const auto butter_expected = butter.Compute(original, distorted);
    const double other_expected = ssimu.Compute(original, other).Score();
    vqm::SharedMetricWorkspace workspace({.size = original.size});

    const auto memory = workspace.GetMemoryUsage();
    CHECK(memory.arena.buffer_bytes > 0);
    CHECK(memory.arena.allocation_bytes >= memory.arena.buffer_bytes);
    CHECK(memory.uses_staging == (memory.staging.buffer_bytes != 0));
    if (memory.uses_staging)
        CHECK(memory.staging.allocation_bytes >= memory.staging.buffer_bytes);

    // Switching clobbers each backend's tables, and must republish them. The
    // second SSIMU call uses a different image to catch wrong-source bindings.
    CHECK_NEAR(
        workspace.ComputeSsimulacra2(original, distorted).Score(),
        ssimu_expected, 1e-9
    );
    const auto got = workspace.ComputeButteraugli(original, distorted);
    CHECK_NEAR(got.score, butter_expected.score, 1e-9);
    CHECK_NEAR(got.distance_p3, butter_expected.distance_p3, 1e-9);
    CHECK_NEAR(
        workspace.ComputeSsimulacra2(original, other).Score(), other_expected,
        1e-9
    );
    CHECK_NEAR(workspace.ComputeButteraugli(original, original).score, 0.0, 0.0);
    CHECK_NEAR(
        workspace.ComputeSsimulacra2(original, original).Score(),
        identity_expected, 1e-9
    );

    const auto reference_view = View(original);
    const auto distorted_view = View(distorted);
    CHECK_NEAR(
        workspace.ComputeButteraugli(reference_view, distorted_view).score,
        butter_expected.score, 1e-9
    );
    CHECK_NEAR(
        workspace.ComputeSsimulacra2(reference_view, distorted_view).Score(),
        ssimu_expected, 1e-9
    );

    auto invalid = distorted;
    invalid.data.pop_back();
    CHECK_THROWS_AS(
        workspace.ComputeButteraugli(original, invalid), std::invalid_argument
    );
    CHECK_NEAR(
        workspace.ComputeSsimulacra2(original, distorted).Score(),
        ssimu_expected, 1e-9
    );
    CHECK_NEAR(
        workspace.ComputeButteraugli(original, distorted).score,
        butter_expected.score, 1e-9
    );
    const auto after = workspace.GetMemoryUsage();
    CHECK(after.arena.allocation_bytes == memory.arena.allocation_bytes);
    CHECK(after.arena.heap == memory.arena.heap);
    CHECK(after.staging.allocation_bytes == memory.staging.allocation_bytes);
    CHECK(after.staging.heap == memory.staging.heap);
}

VQM_TEST(TestSharedNonTemporalParameters) {
    using vqm::SharedMetricConfig;
    using vqm::SharedMetricWorkspace;
    CHECK_THROWS_AS(
        SharedMetricWorkspace(SharedMetricConfig{.size = {1, 1}}),
        std::invalid_argument
    );
    SharedMetricConfig config{.size = {32, 32}};
    config.xmul = -1;
    CHECK_THROWS_AS(SharedMetricWorkspace(config), std::invalid_argument);
}

VQM_TEST(TestSharedStagedRecoveryAfterRejectedInput) {
    if (!vqm::test::GpuAvailable()) {
        vqm::test::MarkSkip("no usable Vulkan device", true);
        return;
    }
    const auto original = vqm::test::MakeGradient(0);
    const auto distorted = vqm::test::MakeGradient(1);
    auto invalid = distorted;
    // Unknown transfers must be rejected before source writes or recording.
    // Switching metrics after rejection must still publish the right tables.
    invalid.format.transfer = static_cast<vqm::TransferCharacteristics>(255);
    vqm::SharedMetricConfig config{.size = original.size};
    config.memory_policy = vqm::MemoryPolicy::Staged;
    if (!vqm::GetMemoryRequirements(config).staged.supported) {
        vqm::test::MarkSkip("no compatible staged memory type");
        return;
    }
    vqm::Ssimulacra2Pipeline independent({.size = original.size});
    // Compare implementations on this device. The core identity gate separately
    // enforces the documented software-driver band and discrete-GPU result.
    const double identity_expected =
        independent.Compute(original, original).Score();
    vqm::SharedMetricWorkspace workspace(config);
    CHECK_THROWS_AS(
        workspace.ComputeSsimulacra2(original, invalid), std::invalid_argument
    );
    CHECK_NEAR(workspace.ComputeButteraugli(original, original).score, 0.0, 0.0);
    CHECK_NEAR(
        workspace.ComputeSsimulacra2(original, original).Score(),
        identity_expected, 1e-9
    );
    CHECK_THROWS_AS(
        workspace.ComputeButteraugli(original, invalid), std::invalid_argument
    );
    CHECK_NEAR(
        workspace.ComputeSsimulacra2(original, original).Score(),
        identity_expected, 1e-9
    );
    CHECK_NEAR(workspace.ComputeButteraugli(original, original).score, 0.0, 0.0);
}
