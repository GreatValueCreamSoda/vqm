// VMAF motion: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "support.h"
#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_shared_storage.h"
#include "vulkan/vulkan_support.h"

namespace {

using vqm::Size;
using vqm::internal::VmafMotion;
using vqm::internal::VmafMotionBuffers;
using vqm::internal::VulkanRuntime;

// A direct two-dimensional convolution is independent of the separable GPU
// passes. The kernel and edge rule come from Netflix motion_tools.h and
// common/convolution_internal.h, revision
// 4611e47a7330cd342498dcc130e1008efbeab6e4.
std::vector<float> ReferenceBlur(const std::vector<float>& source, Size size) {
    constexpr std::array<double, 5> kWeights{
        0.054488685, 0.244201342, 0.402619947, 0.244201342, 0.054488685};
    const auto mirror = [](int coordinate, int limit) {
        if (coordinate < 0) return -coordinate;
        if (coordinate >= limit) return 2 * limit - coordinate - 2;
        return coordinate;
    };
    std::vector<float> result(source.size());
    for (std::size_t y = 0; y < size.height; ++y) {
        for (std::size_t x = 0; x < size.width; ++x) {
            double value = 0.0;
            for (int j = 0; j < 5; ++j) {
                const auto yy = static_cast<std::size_t>(mirror(
                    static_cast<int>(y) + j - 2, static_cast<int>(size.height)
                ));
                for (int i = 0; i < 5; ++i) {
                    const auto xx = static_cast<std::size_t>(mirror(
                        static_cast<int>(x) + i - 2, static_cast<int>(size.width)
                    ));
                    value += kWeights[static_cast<std::size_t>(j)] *
                             kWeights[static_cast<std::size_t>(i)] *
                             source[yy * size.width + xx];
                }
            }
            result[y * size.width + x] = static_cast<float>(value);
        }
    }
    return result;
}

double ReferenceDifference(
    const std::vector<float>& first, const std::vector<float>& second, Size size
) {
    const auto a = ReferenceBlur(first, size);
    const auto b = ReferenceBlur(second, size);
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        sum += std::fabs(static_cast<double>(a[i]) - b[i]);
    return sum / static_cast<double>(a.size());
}

class MotionDevice {
public:
    explicit MotionDevice(Size size, bool private_history = false)
        : size_(size),
          pixels_(size.width * size.height),
          partials_(VmafMotion::PartialCount(size)),
          scratch_(Scratch(size, private_history)),
          runtime_(
              scratch_ ? scratch_->backing().shared_context()
                       : vqm::internal::AcquireVulkanContext(),
              Config(size, private_history), scratch_
          ),
          buffers_{0, static_cast<std::uint32_t>(pixels_),
                   static_cast<std::uint32_t>(2 * pixels_),
                   static_cast<std::uint32_t>(3 * pixels_),
                   static_cast<std::uint32_t>(3 * pixels_ + partials_)} {
        const std::size_t arena_floats = 3 * pixels_ + partials_ + 1;
        runtime_.CreateArena(vqm::internal::Bytes(arena_floats));
        runtime_.CreateStaging(vqm::internal::Bytes(pixels_ + 1));
        runtime_.CreateCommandObjects();
        runtime_.CreateDescriptors(0, vqm::internal::Bytes(arena_floats));
        runtime_.InitializeSharedState();
        motion_.emplace(runtime_, runtime_.BindPersistentPlanes());
    }

    double Push(const std::vector<float>& luma) {
        if (luma.size() != pixels_) throw std::invalid_argument("test geometry");
        std::copy(luma.begin(), luma.end(), runtime_.staging_map());
        runtime_.ResetCommandPool();
        runtime_.BeginCommandBuffer();
        if (scratch_)
            runtime_.FillZero(static_cast<std::uint32_t>(pixels_), 2 * pixels_);
        const std::array copies{
            VulkanRuntime::CopyRange{0, buffers_.luma, pixels_}};
        runtime_.RecordStagedCopies(copies);
        runtime_.BindDescriptorSet();
        motion_->Record(size_, buffers_, have_previous_);
        runtime_.RecordReadback(buffers_.score, pixels_, 1);
        runtime_.EndCommandBuffer();
        runtime_.SubmitAndWait();
        const double result = runtime_.staging_map()[pixels_];
        std::swap(buffers_.previous, buffers_.blurred);
        have_previous_ = true;
        return result;
    }

private:
    static VulkanRuntime::Config Config(Size size, bool private_history) {
        VulkanRuntime::Config config;
        config.base_size = size;
        config.base_pixels = size.width * size.height;
        config.push_constant_size = VmafMotion::PushBytes();
        config.streaming = false;
        if (private_history) {
            config.memory_policy = vqm::MemoryPolicy::Staged;
            config.persistent_ranges.push_back(
                {vqm::internal::Bytes(config.base_pixels),
                 vqm::internal::Bytes(2 * config.base_pixels), false}
            );
        }
        return config;
    }

    static std::shared_ptr<vqm::internal::SharedMetricStorage> Scratch(
        Size size, bool private_history
    ) {
        if (!private_history) return {};
        const std::size_t pixels = size.width * size.height;
        const auto words = 3 * pixels + VmafMotion::PartialCount(size) + 1;
        return std::make_shared<vqm::internal::SharedMetricStorage>(
            vqm::internal::AcquireVulkanContext(), vqm::internal::Bytes(words),
            vqm::internal::Bytes(pixels + 1), vqm::MemoryPolicy::Staged
        );
    }

    Size size_;
    std::size_t pixels_;
    std::size_t partials_;
    std::shared_ptr<vqm::internal::SharedMetricStorage> scratch_;
    VulkanRuntime runtime_;
    VmafMotionBuffers buffers_;
    std::optional<VmafMotion> motion_;
    bool have_previous_ = false;
};

}  // namespace

VQM_TEST(TestVmafMotionTimeline) {
    using vqm::internal::VmafMotionTimeline;
    VmafMotionTimeline timeline;
    CHECK(!timeline.Flush());
    CHECK(!timeline.Flush());
    CHECK_THROWS_AS(timeline.Push(0.0), std::logic_error);
    timeline.Reset();
    CHECK_THROWS_AS(timeline.Push(1.0), std::invalid_argument);
    auto frame = timeline.Push(0.0);
    CHECK(frame && frame->index == 0 && frame->motion2 == 0.0);
    CHECK(!timeline.Flush());
    timeline.Reset();
    frame = timeline.Push(0.0);
    CHECK(frame && frame->index == 0 && frame->motion2 == 0.0);
    CHECK_THROWS_AS(
        timeline.Push(std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument
    );
    CHECK_THROWS_AS(timeline.Push(-1.0), std::invalid_argument);
    CHECK(!timeline.Push(4.0));
    frame = timeline.Push(2.0);
    CHECK(frame && frame->index == 1 && frame->motion2 == 2.0);
    frame = timeline.Push(9.0);
    CHECK(frame && frame->index == 2 && frame->motion2 == 2.0);
    frame = timeline.Flush();
    CHECK(frame && frame->index == 3 && frame->motion2 == 9.0);
    CHECK(!timeline.Flush());
    CHECK_THROWS_AS(timeline.Push(0.0), std::logic_error);
    timeline.Reset();
    CHECK(timeline.Push(0.0));
    CHECK(!timeline.Push(20000.0));
    frame = timeline.Flush();
    CHECK(frame && frame->index == 1 && frame->motion2 == 10000.0);
}

VQM_TEST(TestVmafMotionGeometry) {
    CHECK(VmafMotion::PartialCount({3, 3}) == 1);
    CHECK(VmafMotion::PartialCount({17, 19}) == 3);
    CHECK_THROWS_AS(VmafMotion::PartialCount({2, 8}), std::invalid_argument);
    CHECK_THROWS_AS(
        VmafMotion::PartialCount({std::numeric_limits<std::size_t>::max(), 3}),
        std::invalid_argument
    );
}

VQM_TEST(TestVmafMotionVulkan) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{17, 19};
    std::vector<float> frame_a(size.width * size.height);
    std::vector<float> frame_b(frame_a.size());
    for (std::size_t y = 0; y < size.height; ++y) {
        for (std::size_t x = 0; x < size.width; ++x) {
            const std::size_t i = y * size.width + x;
            frame_a[i] = static_cast<float>((x * 13 + y * 7) % 93) - 50.0f;
            frame_b[i] = static_cast<float>((x * 3 + y * 11 + 5) % 101) - 50.0f;
        }
    }
    frame_a[0] = 100.0f;
    frame_b[frame_b.size() - 1] = 100.0f;
    MotionDevice device(size);
    CHECK_NEAR(device.Push(frame_a), 0.0, 0.0);
    CHECK_NEAR(
        device.Push(frame_b), ReferenceDifference(frame_a, frame_b, size), 1e-4
    );
    auto shifted = frame_b;
    for (float& value : shifted) value += 0.25f;
    // A 10-bit one-code difference is 0.25 in this FP32 domain, not zero.
    CHECK_NEAR(device.Push(shifted), 0.25, 1e-4);
    CHECK_NEAR(device.Push(shifted), 0.0, 0.0);
}

VQM_TEST(TestVmafMotionPreservesHighBitDepthTransitions) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{17, 19};
    const std::vector<float> original(size.width * size.height, 127.0f);
    // Centered native 12/16-bit one-code changes remain meaningful near the
    // positive endpoint. FP16 input storage would erase the 16-bit change
    // before even an FP32 blur or reduction could observe it.
    for (float delta : {1.0f / 16.0f, 1.0f / 256.0f}) {
        const std::vector<float> changed(original.size(), 127.0f + delta);
        const double transition = ReferenceDifference(original, changed, size);
        CHECK(transition > 0.0);
        MotionDevice device(size);
        CHECK_NEAR(device.Push(original), 0.0, 0.0);
        CHECK_NEAR(device.Push(changed), transition, 1e-4);
        CHECK_NEAR(device.Push(changed), 0.0, 0.0);
        CHECK_NEAR(
            device.Push(original), ReferenceDifference(changed, original, size),
            1e-4
        );
        CHECK_NEAR(device.Push(original), 0.0, 0.0);
    }
}

VQM_TEST(TestVmafMotionPrivateHistorySurvivesScratchOverwrite) {
    if (!vqm::test::GpuAvailable()) return;
    constexpr Size size{17, 19};
    MotionDevice device(size, true);
    MotionDevice standalone(size);
    std::vector<float> previous(size.width * size.height, 0.0f);
    CHECK_NEAR(device.Push(previous), 0.0, 0.0);
    CHECK_NEAR(standalone.Push(previous), 0.0, 0.0);
    for (std::size_t frame = 1; frame < 6; ++frame) {
        std::vector<float> current(previous.size());
        for (std::size_t index = 0; index < current.size(); ++index)
            current[index] =
                static_cast<float>((index * 13 + frame * 29) % 192) - 96.0f;
        const auto actual = device.Push(current);
        CHECK_NEAR(actual, standalone.Push(current), 0.0);
        CHECK_NEAR(actual, ReferenceDifference(previous, current, size), 1e-4);
        previous = std::move(current);
    }
}

VQM_TEST(TestVmafMotionTileBoundaries) {
    if (!vqm::test::GpuAvailable()) return;
    for (Size size : {Size{3, 3}, Size{3, 17}, Size{17, 3}, Size{15, 16},
                      Size{16, 15}, Size{16, 16}, Size{31, 33}}) {
        std::vector<float> a(size.width * size.height), b(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<float>((i * 19) % 201) - 100.0f;
            b[i] = static_cast<float>((i * 31 + 7) % 199) - 99.0f;
        }
        MotionDevice device(size);
        CHECK_NEAR(device.Push(a), 0.0, 0.0);
        CHECK_NEAR(device.Push(b), ReferenceDifference(a, b, size), 1e-4);
    }
}

VQM_TEST(TestVmafMotionAgainstLibvmaf) {
    if (!vqm::test::GpuAvailable()) return;
    // Reference values generated by pinned libvmaf's floating-point v0.6.1
    // extractor from tests/make_vmaf_fixtures.py's 10bit-low-bits sequence:
    // frame 0..3 `motion` values in /tmp/vmaf-float-test.json. These are
    // transition scores, before the motion2 look-ahead minimum.
    constexpr Size size{64, 48};
    constexpr std::array expected_motion{0.0, 33.434216, 29.808189, 33.892773};
    constexpr std::array expected_motion2{0.0, 29.808189, 29.808189, 33.892773};
    MotionDevice device(size);
    vqm::internal::VmafMotionTimeline timeline;
    for (std::size_t frame = 0; frame < expected_motion.size(); ++frame) {
        std::vector<float> luma(size.width * size.height);
        for (std::size_t y = 0; y < size.height; ++y) {
            for (std::size_t x = 0; x < size.width; ++x) {
                const std::size_t base =
                    16 +
                    ((7 * x + 13 * y + 17 * frame + 29 * ((x / 7 + frame) % 2)) %
                     220);
                luma[y * size.width + x] = static_cast<float>(base) - 128.0f;
            }
        }
        const double transition = device.Push(luma);
        CHECK_NEAR(transition, expected_motion[frame], 1e-4);
        if (const auto ready = timeline.Push(transition)) {
            CHECK_NEAR(ready->motion2, expected_motion2[ready->index], 1e-4);
        }
    }
    const auto last = timeline.Flush();
    CHECK(last && last->index == 3);
    if (last) CHECK_NEAR(last->motion2, expected_motion2[last->index], 1e-4);
}
