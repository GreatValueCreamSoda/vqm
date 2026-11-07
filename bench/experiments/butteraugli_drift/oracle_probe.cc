// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.
//
// Standalone diagnostic for the unchanged double-precision test oracle.
// Decode through the project's image_io and invert sRGB as butter_test.cc
// does, isolating GPU arithmetic from the comparator's algorithm. Alpha
// planes are accepted only when every sample is opaque. Colour metadata is
// interpreted according to image_io's documented defaults.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io.h"
#include "input.h"
#include "butter_ref.h"

namespace {
using Planes = std::array<std::vector<float>, 3>;

double Sample(const vqm::Image& image, std::size_t offset, std::size_t index) {
    if (image.format.type == vqm::PixelType::Byte)
        return image.data[offset + index] / 255.0;
    std::uint16_t value;
    std::memcpy(&value, image.data.data() + offset + 2 * index, sizeof(value));
    return value / 65535.0;
}

Planes Linear(const vqm::Image& image) {
    if ((image.format.type != vqm::PixelType::Byte &&
         image.format.type != vqm::PixelType::Word) ||
        image.format.family == vqm::ColorFamily::Yuv) {
        throw std::runtime_error(
            "probe requires integer RGB or grayscale input"
        );
    }
    const auto layout = vqm::internal::PlaneLayout(image.format, image.size);
    const std::size_t pixels = image.size.width * image.size.height;
    if (image.HasAlpha()) {
        for (std::size_t i = 0; i < pixels; ++i) {
            if (Sample(image, layout[3].offset, i) != 1.0)
                throw std::runtime_error("probe requires opaque alpha values");
        }
    }
    Planes planes;
    for (std::size_t c = 0; c < 3; ++c) {
        planes[c].resize(pixels);
        for (std::size_t i = 0; i < pixels; ++i) {
            const std::size_t plane =
                image.format.family == vqm::ColorFamily::Grey ? 0 : c;
            const double x = Sample(image, layout[plane].offset, i);
            planes[c][i] = static_cast<float>(
                x <= 0.04045 ? x / 12.92 : std::pow((x + 0.055) / 1.055, 2.4)
            );
        }
    }
    return planes;
}

void Pair(const std::string& original_path, const std::string& distorted_path) {
    std::string error;
    const auto original = vqm::image_io::LoadImage(original_path, error);
    if (!original) throw std::runtime_error(error);
    const auto distorted = vqm::image_io::LoadImage(distorted_path, error);
    if (!distorted) throw std::runtime_error(error);
    if (original->size != distorted->size)
        throw std::runtime_error("size mismatch");
    const auto a = Linear(*original);
    const auto b = Linear(*distorted);
    const auto result = vqm::test::ButteraugliCpuReference(
        a[0].data(), a[1].data(), a[2].data(), b[0].data(), b[1].data(),
        b[2].data(), static_cast<int>(original->size.width),
        static_cast<int>(original->size.height)
    );
    std::printf(
        "%.12g,%.12g,%u,%u,%s,%s\n", result.score, result.distance_p3,
        original->size.width, original->size.height, original_path.c_str(),
        distorted_path.c_str()
    );
    std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::puts("score,distance_p3,width,height,original,distorted");
        if (argc == 3 && std::string(argv[1]) == "--pairs-file") {
            std::ifstream input(argv[2]);
            if (!input) throw std::runtime_error("could not open pair list");
            std::string line;
            while (std::getline(input, line)) {
                const auto tab = line.find('\t');
                if (tab == std::string::npos)
                    throw std::runtime_error("expected tab-separated paths");
                Pair(line.substr(0, tab), line.substr(tab + 1));
            }
        } else if (argc == 3) {
            Pair(argv[1], argv[2]);
        } else {
            throw std::runtime_error(
                "usage: oracle_probe ORIG DIST | --pairs-file FILE"
            );
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
