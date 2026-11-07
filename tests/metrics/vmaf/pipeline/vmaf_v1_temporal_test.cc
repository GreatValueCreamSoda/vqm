// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "support.h"
#include "temporal_test_support.h"
#include "vmaf_v1.h"

namespace {
using namespace vqm;
using namespace vqm::test::temporal;
// Native 4:4:4 keeps CAMBI's minimum 216-pixel dimension and SpEED's
// minimum 80-pixel chroma dimensions without a large video fixture.
constexpr Size kSize{216, 80};

VmafV1Config Config() {
    VmafV1Config config;
    config.size = kSize;
    config.input_format = MakeImage(kSize).format;
    return config;
}

void CheckSpatial(const VmafV1Frame& actual, const VmafV1Frame& expected) {
    CHECK_NEAR(actual.cambi, expected.cambi, 1e-6);
    CHECK_NEAR(actual.speed_chroma_uv, expected.speed_chroma_uv, 1e-6);
    CHECK_NEAR(actual.adm3, expected.adm3, 1e-6);
}

void CheckFrames(
    std::span<const VmafV1Frame> actual, std::span<const VmafV1Frame> expected
) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
        CHECK(actual[i].index == expected[i].index);
        CheckSpatial(actual[i], expected[i]);
        CHECK_NEAR(actual[i].motion3, expected[i].motion3, 1e-6);
        CHECK_NEAR(actual[i].score, expected[i].score, 1e-6);
    }
}
}  // namespace

VQM_TEST(TestVmafV1TemporalPublicBoundariesAndFeatureAssociation) {
    if (!vqm::test::GpuAvailable()) return;
    const auto pairs = MakeClip(kSize, kOffsets.size());
    VmafV1Pipeline still(Config());
    std::vector<VmafV1Frame> spatial;
    for (const auto& pair : pairs) {
        still.Reset();
        CHECK(!still.Push(pair.original, pair.distorted));
        const auto frame = still.Flush();
        CHECK(frame.has_value());
        if (frame) spatial.push_back(*frame);
    }
    CHECK(spatial.size() == pairs.size());
    if (spatial.size() != pairs.size()) return;
    CHECK(std::abs(spatial[0].adm3 - spatial[3].adm3) > 1e-5);
    VmafV1Pipeline pipeline(Config());
    for (const std::size_t count : {0u, 1u, 2u, 3u, 9u}) {
        pipeline.Reset();
        CHECK_THROWS_AS(pipeline.PooledScore(), std::logic_error);
        std::vector<VmafV1Frame> frames;
        for (std::size_t i = 0; i < count; ++i) {
            const auto ready =
                pipeline.Push(pairs[i].original, pairs[i].distorted);
            CHECK(ready.has_value() == (i != 0));
            if (!ready) continue;
            CHECK(ready->index == i - 1);
            frames.push_back(*ready);
        }
        const auto last = pipeline.Flush();
        CHECK(last.has_value() == (count != 0));
        if (last) frames.push_back(*last);
        CHECK(!pipeline.Flush());
        CHECK(frames.size() == count);
        for (std::size_t i = 0; i < frames.size(); ++i) {
            CHECK(frames[i].index == i);
            CheckSpatial(frames[i], spatial[i]);
            CHECK_NEAR(frames[i].motion3, ExpectedMotion(i, count, true), 1e-4);
        }
        CheckPooled<VmafV1Frame>(pipeline, frames);
        CHECK_THROWS_AS(
            pipeline.Push(pairs[0].original, pairs[0].distorted),
            std::logic_error
        );
    }
}

VQM_TEST(TestVmafV1TemporalMotionUsesOnlyReferenceHistory) {
    if (!vqm::test::GpuAvailable()) return;
    auto pairs = MakeClip(kSize, 4);
    for (auto& pair : pairs) pair.original = pairs[0].original;
    VmafV1Pipeline pipeline(Config());
    auto frames = RunClip<VmafV1Frame>(pipeline, pairs);
    CHECK(frames.size() == pairs.size());
    for (const auto& frame : frames) CHECK(frame.motion3 == 0.0);
    pairs = MakeClip(kSize, 4);
    for (auto& pair : pairs) pair.distorted = pairs[0].distorted;
    pipeline.Reset();
    frames = RunClip<VmafV1Frame>(pipeline, pairs);
    CHECK(frames.size() == pairs.size());
    for (std::size_t i = 0; i < frames.size(); ++i)
        CHECK_NEAR(
            frames[i].motion3, ExpectedMotion(i, pairs.size(), true), 1e-4
        );
}

VQM_TEST(TestVmafV1TemporalReusedHostStorageAndResetIsolation) {
    if (!vqm::test::GpuAvailable()) return;
    const auto pairs = MakeClip(kSize, 6);
    VmafV1Pipeline fresh(Config());
    const auto expected = RunClip<VmafV1Frame>(fresh, pairs);
    VmafV1Pipeline reused(Config());
    auto original = MakeImage(kSize), distorted = MakeImage(kSize);
    const auto* original_storage = original.data.data();
    const auto* distorted_storage = distorted.data.data();
    std::vector<VmafV1Frame> frames;
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
        CHECK(!reused.Push(pairs[3].original, pairs[3].distorted));
        if (!pending) CHECK(reused.Flush().has_value());
        reused.Reset();
        CheckFrames(RunClip<VmafV1Frame>(reused, pairs), expected);
        CheckPooled<VmafV1Frame>(reused, expected);
    }
}

VQM_TEST(TestVmafV1TemporalInterleavedPipelinesKeepIndependentHistory) {
    if (!vqm::test::GpuAvailable()) return;
    const auto a = MakeClip(kSize, 6);
    auto b = MakeClip(kSize, 6);
    std::reverse(b.begin(), b.end());
    VmafV1Pipeline isolated_a(Config()), isolated_b(Config());
    const auto expected_a = RunClip<VmafV1Frame>(isolated_a, a);
    const auto expected_b = RunClip<VmafV1Frame>(isolated_b, b);
    VmafV1Pipeline interleaved_a(Config()), interleaved_b(Config());
    std::vector<VmafV1Frame> actual_a, actual_b;
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
