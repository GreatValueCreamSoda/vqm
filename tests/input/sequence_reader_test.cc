// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include <chrono>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "io.h"

namespace {

void Require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("vqm-sequence-io-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()
         ));

    TemporaryDirectory() { std::filesystem::create_directory(path); }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void Write(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("could not write test fixture");
}

void TestSequence(const std::filesystem::path& path) {
    using namespace vqm::image_io;
    const std::string first(24, static_cast<char>(32));
    const std::string second(24, static_cast<char>(192));
    Write(
        path, "YUV4MPEG2 W4 H2 F60000:2002 Ip C444\nFRAME\n" + first +
                  "FRAME Xfoo=bar\n" + second
    );
    SequenceReader reader(path.string());
    Require(reader.size() == vqm::Size{4, 2}, "sequence dimensions");
    Require(reader.frame_rate() == FrameRate{30000, 1001}, "exact reduced FPS");
    Require(reader.is_video(), "Y4M must be a video source");
    std::string error;
    vqm::Image a, b;
    Require(reader.Next(a, error) == SequenceReadResult::Frame, "first frame");
    Require(reader.Next(b, error) == SequenceReadResult::Frame, "second frame");
    Require(
        a.data[0] == 32 && b.data[0] == 192,
        "frames retain independent ownership"
    );
    Require(reader.Next(b, error) == SequenceReadResult::End, "clean EOF");
    Require(reader.Next(b, error) == SequenceReadResult::End, "stable EOF");
    const auto first_only = LoadImage(path.string(), error);
    Require(
        first_only && first_only->data == a.data,
        "LoadImage keeps first-frame semantics"
    );
}

void TestMalformedFrames(const std::filesystem::path& path) {
    using namespace vqm::image_io;
    Write(
        path, "YUV4MPEG2 W4 H2 F30:1 Ip C444\nFRAME\n" + std::string(24, '\0') +
                  "FRAME\n" + std::string(23, '\0')
    );
    SequenceReader reader(path.string());
    vqm::Image frame;
    std::string error;
    Require(
        reader.Next(frame, error) == SequenceReadResult::Frame, "complete frame"
    );
    Require(
        reader.Next(frame, error) == SequenceReadResult::Error,
        "truncated frame is not EOF"
    );
    Require(error == "truncated y4m frame", "truncated frame diagnostic");
    Write(path, "YUV4MPEG2 W4 H2 F30:1 Ip C444\nFRAMEoops\n");
    SequenceReader invalid_marker(path.string());
    Require(
        invalid_marker.Next(frame, error) == SequenceReadResult::Error,
        "invalid frame marker"
    );
}

void TestMetadata(const std::filesystem::path& path) {
    for (const std::string header :
         {"YUV4MPEG2 W4 H2 F0:1 Ip C444\n", "YUV4MPEG2 W4 H2 F30:0 Ip C444\n",
          "YUV4MPEG2 W4 H2 Ip C444\n", "YUV4MPEG2 W4 H2 F30:1 It C444\n",
          "YUV4MPEG2 W4 H2 F30:1 Im C444\n"}) {
        Write(path, header + "FRAME\n" + std::string(24, '\0'));
        bool rejected = false;
        try {
            vqm::image_io::SequenceReader reader(path.string());
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        Require(rejected, "invalid timing or interlace rejected");
        // The still-image loader intentionally does not interpret timing.
        std::string error;
        Require(
            vqm::image_io::LoadImage(path.string(), error).has_value(),
            "LoadImage retains temporal-metadata independence"
        );
    }
}

void TestWordSamples(const std::filesystem::path& path) {
    std::string pixels;
    for (int index = 0; index < 8; ++index) {
        pixels.push_back('\x34');
        pixels.push_back('\x12');
    }
    Write(path, "YUV4MPEG2 W4 H2 F24:1 Ip Cmonop16\nFRAME\n" + pixels);
    vqm::image_io::SequenceReader reader(path.string());
    vqm::Image frame;
    std::string error;
    Require(
        reader.Next(frame, error) == vqm::image_io::SequenceReadResult::Frame,
        "high-bit-depth frame"
    );
    std::uint16_t word = 0;
    std::memcpy(&word, frame.data.data(), sizeof(word));
    Require(
        word == 0x1234 && frame.format.depth == 16, "Y4M little-endian words"
    );
}

void TestChromaDefaults(const std::filesystem::path& path) {
    using namespace vqm::image_io;
    constexpr std::array<std::pair<std::string_view, vqm::ChromaLocation>, 4>
        cases{{{"", vqm::ChromaLocation::Center},
               {" C420jpeg", vqm::ChromaLocation::Center},
               {" C420mpeg2", vqm::ChromaLocation::Left},
               {" C420", vqm::ChromaLocation::Left}}};
    for (const auto& [tag, location] : cases) {
        Write(
            path, "YUV4MPEG2 W4 H2 F24:1 Ip" + std::string(tag) + "\nFRAME\n" +
                      std::string(12, '\x80')
        );
        SequenceReader reader(path.string());
        Require(
            reader.format().chroma_location == location,
            "sequence reader chroma siting"
        );
        vqm::Image frame;
        std::string error;
        Require(
            reader.Next(frame, error) == SequenceReadResult::Frame,
            "chroma default frame"
        );
        const auto still = LoadImage(path.string(), error);
        Require(
            still && still->format.chroma_location == location &&
                still->data == frame.data,
            "still-image loader preserves chroma siting and pixels"
        );
    }
}

}  // namespace

int main() {
    try {
        TemporaryDirectory temporary;
        const auto path = temporary.path / "sequence.y4m";
        TestSequence(path);
        TestMalformedFrames(path);
        TestMetadata(path);
        TestWordSamples(path);
        TestChromaDefaults(path);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
