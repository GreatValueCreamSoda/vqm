// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "support.h"
#include "temporal_test_support.h"
#include "vmaf.h"

namespace {
using namespace vqm;
using namespace vqm::test::temporal;
constexpr Size kSize{32, 32};

void CheckSpatial(const VmafFrame& actual, const VmafFrame& expected) {
    CHECK_NEAR(actual.adm2, expected.adm2, 1e-6);
    for (std::size_t scale = 0; scale < actual.vif.size(); ++scale)
        CHECK_NEAR(actual.vif[scale], expected.vif[scale], 1e-6);
}

void CheckFrames(
    std::span<const VmafFrame> actual, std::span<const VmafFrame> expected
) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
        CHECK(actual[i].index == expected[i].index);
        CheckSpatial(actual[i], expected[i]);
        CHECK_NEAR(actual[i].motion2, expected[i].motion2, 1e-6);
        CHECK_NEAR(actual[i].score, expected[i].score, 1e-6);
    }
}
}  // namespace

VQM_TEST(TestVmafTemporalPublicBoundariesAndFeatureAssociation) {
    if (!vqm::test::GpuAvailable()) return;
    const auto pairs = MakeClip(kSize, kOffsets.size());
    VmafPipeline still({.size = kSize, .temporal_weighting = false});
    std::vector<VmafFrame> spatial;
    for (const auto& pair : pairs) {
        const auto frame = still.Push(pair.original, pair.distorted);
        CHECK(frame.has_value());
        if (frame) spatial.push_back(*frame);
    }
    CHECK(spatial.size() == pairs.size());
    if (spatial.size() != pairs.size()) return;
    CHECK(std::abs(spatial[0].adm2 - spatial[3].adm2) > 1e-5);
    VmafPipeline pipeline({.size = kSize});
    for (const std::size_t count : {0u, 1u, 2u, 3u, 9u}) {
        pipeline.Reset();
        CHECK_THROWS_AS(pipeline.PooledScore(), std::logic_error);
        std::vector<VmafFrame> frames;
        for (std::size_t i = 0; i < count; ++i) {
            const auto ready =
                pipeline.Push(pairs[i].original, pairs[i].distorted);
            CHECK(ready.has_value() == (i != 1));
            if (!ready) continue;
            CHECK(ready->index == (i == 0 ? 0 : i - 1));
            frames.push_back(*ready);
        }
        const auto last = pipeline.Flush();
        CHECK(last.has_value() == (count > 1));
        if (last) frames.push_back(*last);
        CHECK(!pipeline.Flush());
        CHECK(frames.size() == count);
        for (std::size_t i = 0; i < frames.size(); ++i) {
            CHECK(frames[i].index == i);
            CheckSpatial(frames[i], spatial[i]);
            CHECK_NEAR(frames[i].motion2, ExpectedMotion(i, count, false), 1e-4);
        }
        CheckPooled<VmafFrame>(pipeline, frames);
        CHECK_THROWS_AS(
            pipeline.Push(pairs[0].original, pairs[0].distorted),
            std::logic_error
        );
    }
}

VQM_TEST(TestVmafTemporalMotionUsesOnlyReferenceHistory) {
    if (!vqm::test::GpuAvailable()) return;
    auto pairs = MakeClip(kSize, 4);
    for (auto& pair : pairs) pair.original = pairs[0].original;
    VmafPipeline pipeline({.size = kSize});
    auto frames = RunClip<VmafFrame>(pipeline, pairs);
    CHECK(frames.size() == pairs.size());
    for (const auto& frame : frames) CHECK(frame.motion2 == 0.0);
    pairs = MakeClip(kSize, 4);
    for (auto& pair : pairs) pair.distorted = pairs[0].distorted;
    pipeline.Reset();
    frames = RunClip<VmafFrame>(pipeline, pairs);
    CHECK(frames.size() == pairs.size());
    for (std::size_t i = 0; i < frames.size(); ++i)
        CHECK_NEAR(
            frames[i].motion2, ExpectedMotion(i, pairs.size(), false), 1e-4
        );
}

VQM_TEST(TestVmafTemporalReusedHostStorageAndResetIsolation) {
    if (!vqm::test::GpuAvailable()) return;
    const auto pairs = MakeClip(kSize, 6);
    VmafPipeline fresh({.size = kSize});
    const auto expected = RunClip<VmafFrame>(fresh, pairs);
    VmafPipeline reused({.size = kSize});
    auto original = MakeImage(kSize), distorted = MakeImage(kSize);
    const auto* original_storage = original.data.data();
    const auto* distorted_storage = distorted.data.data();
    std::vector<VmafFrame> frames;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        FillImage(original, i, false);
        FillImage(distorted, i, true);
        if (const auto frame = reused.Push(original, distorted))
            frames.push_back(*frame);
        std::fill(original.data.begin(), original.data.end(), 0);
        std::fill(distorted.data.begin(), distorted.data.end(), 0xff);
    }
    CHECK(original_storage == original.data.data());
    CHECK(distorted_storage == distorted.data.data());
    if (const auto last = reused.Flush()) frames.push_back(*last);
    CheckFrames(frames, expected);
    for (const bool pending : {false, true}) {
        reused.Reset();
        CHECK(reused.Push(pairs[3].original, pairs[3].distorted).has_value());
        if (pending) CHECK(!reused.Push(pairs[4].original, pairs[4].distorted));
        if (!pending) CHECK(!reused.Flush());
        reused.Reset();
        CheckFrames(RunClip<VmafFrame>(reused, pairs), expected);
        CheckPooled<VmafFrame>(reused, expected);
    }
}

VQM_TEST(TestVmafTemporalInterleavedPipelinesKeepIndependentHistory) {
    if (!vqm::test::GpuAvailable()) return;
    const auto a = MakeClip(kSize, 6);
    auto b = MakeClip(kSize, 6);
    std::reverse(b.begin(), b.end());
    VmafPipeline isolated_a({.size = kSize}), isolated_b({.size = kSize});
    const auto expected_a = RunClip<VmafFrame>(isolated_a, a);
    const auto expected_b = RunClip<VmafFrame>(isolated_b, b);
    VmafPipeline interleaved_a({.size = kSize}), interleaved_b({.size = kSize});
    std::vector<VmafFrame> actual_a, actual_b;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (const auto frame = interleaved_a.Push(a[i].original, a[i].distorted))
            actual_a.push_back(*frame);
        if (const auto frame = interleaved_b.Push(b[i].original, b[i].distorted))
            actual_b.push_back(*frame);
    }
    if (const auto frame = interleaved_b.Flush()) actual_b.push_back(*frame);
    if (const auto frame = interleaved_a.Flush()) actual_a.push_back(*frame);
    CheckFrames(actual_a, expected_a);
    CheckFrames(actual_b, expected_b);
}
