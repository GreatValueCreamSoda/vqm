// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <array>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "input.h"
#include "vqm_pool.h"
#include "vqm_views.h"
#include "support.h"

VQM_TEST(TestPoolZeroWorkers) {
    using namespace vqm;
    Ssimulacra2Config config;
    config.size = {32, 32};
    CHECK_THROWS_AS(Ssimulacra2Pool(config, 0), std::invalid_argument);
    ButteraugliConfig butter;
    butter.size = config.size;
    CHECK_THROWS_AS(ButteraugliPool(butter, 0), std::invalid_argument);
}

VQM_TEST(TestPoolOrderedHostPairs) {
    using namespace vqm;
    if (!test::GpuAvailable()) return;
    const auto a = test::MakeGradient(0);
    const auto b = test::MakeGradient(1);
    const auto c = test::MakeGradient(2);
    const std::array<ImagePair, 5> pairs{
        {{a, a}, {a, b}, {b, c}, {c, a}, {a, c}}};
    Ssimulacra2Config config;
    config.size = a.size;
    Ssimulacra2Pipeline reference(config);
    std::array<double, 5> expected{};
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        expected[i] =
            reference.Compute(pairs[i].original, pairs[i].distorted).Score();
    }
    Ssimulacra2Pool pool(config, 2);
    CHECK(pool.Workers() == 2);
    CHECK(pool.ComputeMany(std::span<const ImagePair>{}).empty());
    const auto results = pool.ComputeMany(pairs);
    const auto scores = pool.ScoreMany(pairs);
    CHECK(results.size() == pairs.size());
    CHECK(scores.size() == pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        CHECK_NEAR(results[i].Score(), expected[i], 1e-9);
        CHECK_NEAR(scores[i], expected[i], 1e-9);
    }
    const std::span<const ImagePair> one(pairs.data(), 1);
    CHECK_NEAR(pool.ScoreMany(one)[0], expected[0], 1e-9);

    Ssimulacra2Pool moved(std::move(pool));
    CHECK(pool.Workers() == 0);
    CHECK_THROWS_AS(pool.ScoreMany(one), std::logic_error);
    CHECK(moved.Workers() == 2);
    CHECK_NEAR(moved.ScoreMany(one)[0], expected[0], 1e-9);
}

VQM_TEST(TestPoolDrainsOnInvalidHostPair) {
    using namespace vqm;
    if (!test::GpuAvailable()) return;
    const auto a = test::MakeGradient(0);
    const auto b = test::MakeGradient(1);
    auto invalid = b;
    ++invalid.size.width;
    Ssimulacra2Config config;
    config.size = a.size;
    Ssimulacra2Pool pool(config, 2);
    const std::array<ImagePair, 3> pairs{{{a, b}, {b, a}, {a, invalid}}};
    // Pair 1 is still in flight when the invalid third pair is submitted.
    CHECK_THROWS_AS(pool.ComputeMany(pairs), std::invalid_argument);
    const std::span<const ImagePair> one(pairs.data(), 1);
    CHECK(pool.ComputeMany(one).size() == 1);
}

VQM_TEST(TestButterPoolOrderedHostPairsAndDrain) {
    using namespace vqm;
    if (!test::GpuAvailable()) return;
    const auto a = test::MakeGradient(0);
    const auto b = test::MakeGradient(1);
    const auto c = test::MakeGradient(2);
    const std::array<ImagePair, 4> pairs{{{a, a}, {a, b}, {b, c}, {c, a}}};
    ButteraugliConfig config;
    config.size = a.size;
    ButteraugliPipeline reference(config);
    std::array<ButteraugliScore, 4> expected{};
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        expected[i] = reference.Compute(pairs[i].original, pairs[i].distorted);
    }
    ButteraugliPool pool(config, 2);
    CHECK(pool.Workers() == 2);
    CHECK(pool.ComputeMany(std::span<const ImagePair>{}).empty());
    const auto results = pool.ComputeMany(pairs);
    CHECK(results.size() == pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        CHECK_NEAR(results[i].score, expected[i].score, 1e-9);
        CHECK_NEAR(results[i].distance_p3, expected[i].distance_p3, 1e-9);
    }
    auto invalid = b;
    ++invalid.size.height;
    const std::array<ImagePair, 3> bad_pairs{{{a, b}, {b, a}, {a, invalid}}};
    CHECK_THROWS_AS(pool.ComputeMany(bad_pairs), std::invalid_argument);
    const std::span<const ImagePair> one(pairs.data(), 1);
    CHECK_NEAR(pool.ComputeMany(one)[0].score, expected[0].score, 1e-9);
}

VQM_TEST(TestPoolPaddedImageViews) {
    using namespace vqm;
    if (!test::GpuAvailable()) return;
    const Image original = test::MakeRampImage(33, 17, true, PixelType::Word);
    Image distorted = original;
    distorted.data[100] ^= 0x7f;
    const auto make_view = [](const Image& image,
                              std::array<std::vector<std::byte>, 4>& storage) {
        ImageView view;
        view.size = image.size;
        view.format = image.format;
        const auto layout = internal::PlaneLayout(image.format, image.size);
        for (std::size_t p = 0; p < layout.size(); ++p) {
            if (!layout[p].present()) continue;
            const std::size_t row_bytes =
                layout[p].width * layout[p].sample_bytes;
            const std::size_t stride = row_bytes + 7;
            storage[p].resize(stride * layout[p].height);
            for (std::size_t row = 0; row < layout[p].height; ++row) {
                std::memcpy(
                    storage[p].data() + row * stride,
                    image.data.data() + layout[p].offset + row * row_bytes,
                    row_bytes
                );
            }
            view.planes[p] = {storage[p].data(), stride};
        }
        return view;
    };
    std::array<std::vector<std::byte>, 4> a_storage, b_storage;
    const ImageView a = make_view(original, a_storage);
    const ImageView b = make_view(distorted, b_storage);
    const std::array<ImageViewPair, 4> pairs{{{a, b}, {b, a}, {a, a}, {b, b}}};
    const std::array<ImagePair, 4> packed{{{original, distorted},
                                           {distorted, original},
                                           {original, original},
                                           {distorted, distorted}}};

    Ssimulacra2Pipeline reference({original.size});
    Ssimulacra2Pool pool({original.size}, 2);
    const auto actual = pool.ComputeMany(pairs);
    const auto scores = pool.ScoreMany(pairs);
    CHECK(actual.size() == pairs.size());
    CHECK(scores.size() == pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const double expected =
            reference.Compute(packed[i].original, packed[i].distorted).Score();
        CHECK_NEAR(actual[i].Score(), expected, 1e-9);
        CHECK_NEAR(scores[i], expected, 1e-9);
    }

    ButteraugliPipeline butter_reference({original.size});
    ButteraugliPool butter({original.size}, 2);
    const auto butter_actual = butter.ComputeMany(pairs);
    CHECK(butter_actual.size() == pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const auto expected =
            butter_reference.Compute(packed[i].original, packed[i].distorted);
        CHECK_NEAR(butter_actual[i].score, expected.score, 1e-9);
        CHECK_NEAR(butter_actual[i].distance_p3, expected.distance_p3, 1e-9);
    }

    ImageView invalid = b;
    invalid.planes[0].stride_bytes = 1;
    const std::array<ImageViewPair, 3> bad{{{a, b}, {b, a}, {a, invalid}}};
    CHECK_THROWS_AS(pool.ComputeMany(bad), std::invalid_argument);
    CHECK_NEAR(pool.ScoreMany(std::span(pairs).first(1))[0], scores[0], 1e-9);
    CHECK_THROWS_AS(butter.ComputeMany(bad), std::invalid_argument);
    const auto after_error = butter.ComputeMany(std::span(pairs).first(1))[0];
    CHECK_NEAR(after_error.score, butter_actual[0].score, 1e-9);
    CHECK_NEAR(after_error.distance_p3, butter_actual[0].distance_p3, 1e-9);
}
