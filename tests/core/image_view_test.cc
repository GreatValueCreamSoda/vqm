// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "butteraugli.h"
#include "image_view.h"
#include "input.h"
#include "vqm_views.h"
#include "support.h"

VQM_TEST(TestCopyImageViewPaddedPlanes) {
    using namespace vqm;
    std::array<std::array<std::byte, 10>, 3> planes{};
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        for (std::size_t y = 0; y < 2; ++y) {
            for (std::size_t x = 0; x < 3; ++x) {
                planes[plane][y * 5 + x] =
                    static_cast<std::byte>(plane * 20 + y * 3 + x + 1);
            }
        }
    }
    ImageView view;
    view.size = {3, 2};
    view.format = test::PinnedFormat(AlphaType::None, PixelType::Byte);
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        view.planes[plane] = {planes[plane].data(), 5};
    }
    const Image image = CopyImage(view);
    CHECK(image.size == view.size);
    CHECK(
        image.data ==
        std::vector<std::uint8_t>({1,  2,  3, 4, 5,  6,  0,  0,  21, 22, 23, 24,
                                   25, 26, 0, 0, 41, 42, 43, 44, 45, 46, 0,  0})
    );
}

VQM_TEST(TestCopyImageViewSubsampledAlpha) {
    using namespace vqm;
    // 4x4 Y, 2x2 U/V and 4x4 alpha, each with padding between rows.
    std::array<std::byte, 24> y{};
    std::array<std::byte, 8> u{};
    std::array<std::byte, 8> v{};
    std::array<std::byte, 24> alpha{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            y[row * 6 + col] = static_cast<std::byte>(row * 4 + col);
            alpha[row * 6 + col] = static_cast<std::byte>(200 + row * 4 + col);
        }
    }
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t col = 0; col < 2; ++col) {
            u[row * 4 + col] = static_cast<std::byte>(30 + row * 2 + col);
            v[row * 4 + col] = static_cast<std::byte>(60 + row * 2 + col);
        }
    }
    ImageView view;
    view.size = {4, 4};
    view.format.type = PixelType::Byte;
    view.format.family = ColorFamily::Yuv;
    view.format.alpha = AlphaType::Straight;
    view.format.matrix = MatrixCoefficients::Bt709;
    view.format.subsample_w = 1;
    view.format.subsample_h = 1;
    view.planes = {
        {{y.data(), 6}, {u.data(), 4}, {v.data(), 4}, {alpha.data(), 6}}};
    const Image image = CopyImage(view);
    CHECK(image.data == std::vector<std::uint8_t>({0,   1,   2,   3,   4,   5,
                                                   6,   7,   8,   9,   10,  11,
                                                   12,  13,  14,  15,  30,  31,
                                                   32,  33,  60,  61,  62,  63,
                                                   200, 201, 202, 203, 204, 205,
                                                   206, 207, 208, 209, 210, 211,
                                                   212, 213, 214, 215}));
}

VQM_TEST(TestCopyImageViewWordSamples) {
    using namespace vqm;
    std::array<std::array<std::byte, 12>, 3> planes{};
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t byte = 0; byte < 4; ++byte) {
                planes[plane][row * 6 + byte] =
                    static_cast<std::byte>(1 + plane * 20 + row * 4 + byte);
            }
        }
    }
    ImageView view;
    view.size = {2, 2};
    view.format = test::PinnedFormat(AlphaType::None, PixelType::Word);
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        view.planes[plane] = {planes[plane].data(), 6};
    }
    CHECK(CopyImage(view).data == std::vector<std::uint8_t>({1,  2,  3,  4,  5,
                                                             6,  7,  8,  21, 22,
                                                             23, 24, 25, 26, 27,
                                                             28, 41, 42, 43, 44,
                                                             45, 46, 47, 48}));
}

VQM_TEST(TestPackImageViewSelectedPlanes) {
    using namespace vqm;
    std::array<std::array<std::byte, 4>, 3> planes{};
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        for (std::size_t i = 0; i < planes[plane].size(); ++i) {
            planes[plane][i] =
                static_cast<std::byte>(1 + plane * planes[plane].size() + i);
        }
    }
    ImageView view;
    view.size = {2, 2};
    view.format = test::PinnedFormat(AlphaType::None, PixelType::Byte);
    for (std::size_t plane = 0; plane < planes.size(); ++plane)
        view.planes[plane] = {planes[plane].data(), 2};

    const auto layout = internal::DescribeImageView(view);
    std::vector<std::byte> destination(layout.bytes, std::byte{0xa5});
    internal::PackImageView(destination.data(), view, layout, 0x01);
    CHECK(
        std::equal(
            destination.begin(), destination.begin() + 4, planes[0].begin()
        )
    );
    CHECK(
        std::all_of(
            destination.begin() + 4, destination.end(),
            [](std::byte value) { return value == std::byte{0xa5}; }
        )
    );
}

VQM_TEST(TestImageViewValidation) {
    using namespace vqm;
    std::array<std::byte, 16> data{};
    ImageView view;
    view.size = {2, 2};
    view.format = test::PinnedFormat(AlphaType::None, PixelType::Byte);
    view.planes = {{{data.data(), 2}, {data.data(), 2}, {data.data(), 2}, {}}};
    CHECK(CopyImage(view).data.size() == 12);
    auto bad = view;
    bad.planes[1].data = nullptr;
    CHECK_THROWS_AS(CopyImage(bad), std::invalid_argument);
    bad = view;
    bad.planes[2].stride_bytes = 1;
    CHECK_THROWS_AS(CopyImage(bad), std::invalid_argument);
    bad = view;
    bad.planes[0].stride_bytes = std::numeric_limits<std::size_t>::max();
    CHECK_THROWS_AS(CopyImage(bad), std::overflow_error);
    bad = view;
    bad.size.width = 0;
    CHECK_THROWS_AS(CopyImage(bad), std::invalid_argument);
    bad = view;
    bad.format.depth = 16;
    CHECK_THROWS_AS(CopyImage(bad), std::invalid_argument);
}

VQM_TEST(TestImageViewScore) {
    using namespace vqm;
    if (!test::GpuAvailable()) return;
    Image original = test::MakeRampImage(32, 32, false, PixelType::Byte);
    Image distorted = original;
    distorted.data[100] ^= 0x7fu;
    const auto view = [](const Image& image) {
        ImageView result;
        result.size = image.size;
        result.format = image.format;
        for (std::size_t plane = 0; plane < 3; ++plane) {
            result.planes[plane] = {reinterpret_cast<const std::byte*>(
                                        image.data.data() + plane * 1024
                                    ),
                                    32};
        }
        return result;
    };
    Ssimulacra2Pipeline pipeline({{32, 32}});
    CHECK_NEAR(
        Score(pipeline, view(original), view(distorted)),
        pipeline.Compute(original, distorted).Score(), 1e-9
    );
    ImageView bad = view(distorted);
    bad.size.width = 16;
    CHECK_THROWS_AS(Score(pipeline, view(original), bad), std::invalid_argument);
}

VQM_TEST(TestImageViewStreamingBothMetrics) {
    using namespace vqm;
    if (!test::GpuAvailable()) return;
    const Image original = test::MakeRampImage(33, 17, true, PixelType::Word);
    Image distorted = original;
    distorted.data[100] ^= 0x7fu;
    const auto make_view = [](const Image& image,
                              std::array<std::vector<std::byte>, 4>& storage) {
        ImageView result;
        result.size = image.size;
        result.format = image.format;
        const auto layout = internal::PlaneLayout(image.format, image.size);
        for (std::size_t p = 0; p < layout.size(); ++p) {
            if (!layout[p].present()) continue;
            const std::size_t bytes = layout[p].width * layout[p].sample_bytes;
            const std::size_t stride = bytes + 7;
            storage[p].resize(stride * layout[p].height);
            for (std::size_t row = 0; row < layout[p].height; ++row) {
                std::memcpy(
                    storage[p].data() + row * stride,
                    image.data.data() + layout[p].offset + row * bytes, bytes
                );
            }
            result.planes[p] = {storage[p].data(), stride};
        }
        return result;
    };
    std::array<std::vector<std::byte>, 4> a_planes;
    std::array<std::vector<std::byte>, 4> b_planes;
    const ImageView a = make_view(original, a_planes);
    const ImageView b = make_view(distorted, b_planes);

    Ssimulacra2Pipeline pipeline({original.size});
    const double expected = pipeline.Compute(original, distorted).Score();
    CHECK_NEAR(pipeline.Compute(a, b).Score(), expected, 1e-9);
    const auto token = pipeline.Submit(a, b);
    CHECK_THROWS_AS(pipeline.Submit(a, b), std::logic_error);
    a_planes[0][0] ^= std::byte{0xff};
    CHECK_NEAR(pipeline.Finish(token).Score(), expected, 1e-9);
    a_planes[0][0] ^= std::byte{0xff};

    ButteraugliPipeline butter({original.size});
    const auto butter_expected = butter.Compute(original, distorted);
    const auto butter_actual = butter.Compute(a, b);
    CHECK_NEAR(butter_actual.score, butter_expected.score, 1e-9);
    CHECK_NEAR(butter_actual.distance_p3, butter_expected.distance_p3, 1e-9);
    const auto butter_token = butter.Submit(a, b);
    CHECK_THROWS_AS(butter.Submit(a, b), std::logic_error);
    a_planes[0][0] ^= std::byte{0xff};
    const auto butter_submitted = butter.Finish(butter_token);
    CHECK_NEAR(butter_submitted.score, butter_expected.score, 1e-9);
    CHECK_NEAR(butter_submitted.distance_p3, butter_expected.distance_p3, 1e-9);
    a_planes[0][0] ^= std::byte{0xff};

    ImageView bad = b;
    bad.planes[1].stride_bytes = 1;
    CHECK_THROWS_AS(pipeline.Compute(a, bad), std::invalid_argument);
    CHECK_THROWS_AS(butter.Compute(a, bad), std::invalid_argument);
    bad = b;
    ++bad.size.width;
    CHECK_THROWS_AS(pipeline.Compute(a, bad), std::invalid_argument);
    CHECK_THROWS_AS(butter.Compute(a, bad), std::invalid_argument);

    Ssimulacra2Config no_stream;
    no_stream.size = original.size;
    no_stream.streaming_inputs = false;
    Ssimulacra2Pipeline resident_only(no_stream);
    CHECK_THROWS_AS(resident_only.Compute(a, b), std::invalid_argument);
    ButteraugliConfig butter_no_stream;
    butter_no_stream.size = original.size;
    butter_no_stream.streaming_inputs = false;
    ButteraugliPipeline butter_resident_only(butter_no_stream);
    CHECK_THROWS_AS(butter_resident_only.Compute(a, b), std::invalid_argument);
}

VQM_TEST(TestImageViewPaddedYuvAlphaStreaming) {
    using namespace vqm;
    if (!test::GpuAvailable()) return;
    constexpr Size size{34, 18};
    ImageFormat format;
    format.type = PixelType::Word;
    format.family = ColorFamily::Yuv;
    format.alpha = AlphaType::Straight;
    format.matrix = MatrixCoefficients::Bt709;
    format.transfer = TransferCharacteristics::Bt709;
    format.primaries = ColorPrimaries::Bt709;
    format.subsample_w = 1;
    format.subsample_h = 1;
    format.depth = 10;
    const auto layout = internal::PlaneLayout(format, size);
    const auto make_view = [&](std::array<std::vector<std::byte>, 4>& storage,
                               std::uint16_t delta) {
        ImageView result;
        result.size = size;
        result.format = format;
        for (std::size_t p = 0; p < layout.size(); ++p) {
            const std::size_t stride =
                layout[p].width * sizeof(std::uint16_t) + 11;
            storage[p].resize(stride * layout[p].height);
            for (std::size_t y = 0; y < layout[p].height; ++y) {
                for (std::size_t x = 0; x < layout[p].width; ++x) {
                    const std::uint16_t value = static_cast<std::uint16_t>(
                        p == 0   ? 80 + ((x * 7 + y * 5 + delta) % 700)
                        : p == 3 ? 450 + ((x + y) % 500)
                                 : 480 + ((x * 3 + y * 2) % 64)
                    );
                    std::memcpy(
                        storage[p].data() + y * stride + x * sizeof(value),
                        &value, sizeof(value)
                    );
                }
            }
            result.planes[p] = {storage[p].data(), stride};
        }
        return result;
    };
    std::array<std::vector<std::byte>, 4> a_planes;
    std::array<std::vector<std::byte>, 4> b_planes;
    const ImageView original = make_view(a_planes, 0);
    const ImageView distorted = make_view(b_planes, 29);
    const Image a = CopyImage(original);
    const Image b = CopyImage(distorted);
    constexpr float background = 0.1f;

    Ssimulacra2Pipeline pipeline({size});
    const double expected = pipeline.Compute(a, b, background).Score();
    CHECK_NEAR(
        pipeline.Compute(original, distorted, background).Score(), expected, 1e-9
    );
    ButteraugliPipeline butter({size});
    const auto butter_expected = butter.Compute(a, b, background);
    const auto butter_actual = butter.Compute(original, distorted, background);
    CHECK_NEAR(butter_actual.score, butter_expected.score, 1e-9);
    CHECK_NEAR(butter_actual.distance_p3, butter_expected.distance_p3, 1e-9);
}
