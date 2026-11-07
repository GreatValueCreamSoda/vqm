// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.
#include "bestsource.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/mathematics.h>
}

#include "input.h"
#include "tracklist.h"
#include "videosource.h"

namespace vqm::cli {
namespace {

std::filesystem::path LocalFile(const std::string& path) {
    const auto absolute = std::filesystem::absolute(path);
    if (!std::filesystem::is_regular_file(absolute))
        throw std::invalid_argument(
            "BestSource requires a regular local video file"
        );
    return absolute;
}

AVRational Ratio(const BSRational& value, const char* name) {
    constexpr auto limit = std::numeric_limits<int>::max();
    if (value.Num <= 0 || value.Num > limit || value.Den <= 0 ||
        value.Den > limit)
        throw std::runtime_error(std::string("BestSource: invalid ") + name);
    const auto gcd = std::gcd(value.Num, value.Den);
    return {static_cast<int>(value.Num / gcd),
            static_cast<int>(value.Den / gcd)};
}

struct FrameSpec {
    Size size;
    ImageFormat format;

    [[nodiscard]] bool operator==(const FrameSpec& other) const {
        return size == other.size &&
               internal::SameImageFormat(format, other.format);
    }
};

FrameSpec Describe(const BestVideoFrame& input) {
    if (!input.IsGPUResident() || input.HasPendingFieldMerge())
        throw std::runtime_error(
            "BestSource: expected one GPU-resident video frame"
        );
    const auto& av = *input.GetAVFrame();
    if ((av.flags & AV_FRAME_FLAG_CORRUPT) || av.decode_error_flags)
        throw std::runtime_error("BestSource: corrupt video frame");
    if (input.InterlacedFrame || av.crop_left || av.crop_top ||
        input.Width <= 0 || input.Height <= 0 ||
        input.Width != input.SSModWidth || input.Height != input.SSModHeight)
        throw std::invalid_argument(
            "BestSource requires progressive, chroma-aligned frames without "
            "left or top crop"
        );
    const auto& vf = input.VF;
    if (vf.ColorFamily != 3 || vf.Alpha || vf.Float ||
        (vf.Bits != 8 && vf.Bits != 10 && vf.Bits != 12) ||
        vf.SubSamplingW != 1 || (vf.SubSamplingH != 0 && vf.SubSamplingH != 1))
        throw std::invalid_argument(
            "BestSource requires 8-, 10- or 12-bit YUV 4:2:0 or 4:2:2"
        );

    ImageFormat format;
    format.type = vf.Bits == 8 ? PixelType::Byte : PixelType::Word;
    format.family = ColorFamily::Yuv;
    format.range = input.ColorRange == AVCOL_RANGE_JPEG ? PixelRange::Full
                                                        : PixelRange::Limited;
    format.matrix = input.Matrix == AVCOL_SPC_UNSPECIFIED
                        ? MatrixCoefficients::Bt470Bg
                        : static_cast<MatrixCoefficients>(input.Matrix);
    format.transfer = input.Transfer == AVCOL_TRC_UNSPECIFIED
                          ? TransferCharacteristics::Bt709
                          : static_cast<TransferCharacteristics>(input.Transfer);
    format.primaries = input.Primaries == AVCOL_PRI_UNSPECIFIED
                           ? ColorPrimaries::Bt709
                           : static_cast<ColorPrimaries>(input.Primaries);
    if (input.ChromaLocation >= AVCHROMA_LOC_LEFT &&
        input.ChromaLocation <= AVCHROMA_LOC_BOTTOM)
        format.chroma_location = static_cast<ChromaLocation>(
            input.ChromaLocation - AVCHROMA_LOC_LEFT
        );
    format.depth = static_cast<std::uint8_t>(vf.Bits);
    format.subsample_w = static_cast<std::uint8_t>(vf.SubSamplingW);
    format.subsample_h = static_cast<std::uint8_t>(vf.SubSamplingH);
    const Size size{static_cast<std::size_t>(input.Width),
                    static_cast<std::size_t>(input.Height)};
    internal::ValidateImage(format, size, "BestSource frame");
    return {size, format};
}

}  // namespace

std::optional<int> FindBestSourceTrack(const std::string& path) {
    std::error_code error;
    const auto absolute = std::filesystem::absolute(path, error);
    if (error || !std::filesystem::is_regular_file(absolute, error))
        return std::nullopt;
    std::ifstream file(absolute, std::ios::binary);
    if (!file) return std::nullopt;
    std::array<char, 9> bytes{};
    file.read(bytes.data(), bytes.size());
    const std::string_view header(
        bytes.data(), static_cast<std::size_t>(file.gcount())
    );
    if (header.starts_with("YUV4MPEG2") ||
        header.starts_with("\x89PNG\r\n\x1a\n") ||
        header.starts_with("\xff\xd8"))
        return std::nullopt;
    file.close();

    try {
        const BestTrackList tracks(absolute, nullptr);
        for (int i = 0; i < tracks.GetNumTracks(); ++i) {
            const auto& track = tracks.GetTrackInfo(i);
            if (track.MediaType == AVMEDIA_TYPE_VIDEO &&
                !(track.Disposition & AV_DISPOSITION_ATTACHED_PIC))
                return i;
        }
    } catch (const BestSourceException&) {
        // The native input loader reports unrecognized files.
    }
    return std::nullopt;
}

struct BestSourceReader::Impl {
    BestVideoSource source;
    std::optional<BestSourceInterop> interop;
    std::unique_ptr<BestVideoFrame> first;
    FrameSpec spec{};
    AVRational rate{};
    AVRational time_base{};
    std::int64_t next = 0;
    std::int64_t end = 0;
    std::int64_t first_index = 0;
    std::int64_t first_timestamp = AV_NOPTS_VALUE;
    bool failed = false;

    Impl(
        const std::string& path, int track, BestSourceDevice& device,
        std::int64_t start, std::optional<std::int64_t> count
    )
        : source(LocalFile(path), true, device.uuid(), track, 0, 1, 1, {}, {}),
          next(start) {
        auto* decoder = source.GetHWDeviceContext();
        if (!source.IsGPU() || !decoder || !source.GetGpuHasher())
            throw std::runtime_error(
                "BestSource requires Vulkan Video hardware decoding"
            );
        source.SetMaxDecoderInstances(1);
        const auto& properties = source.GetVideoProperties();
        if (start < 0 || start >= properties.NumFrames)
            throw std::out_of_range(
                "BestSource: start frame is outside the clip"
            );
        if (count && (*count <= 0 || *count > properties.NumFrames - start))
            throw std::out_of_range(
                "BestSource: requested frame range exceeds the clip"
            );
        end = count ? start + *count : properties.NumFrames;
        rate = Ratio(properties.FPS, "frame rate");
        time_base = Ratio(properties.TimeBase, "time base");
        first = ReadFrame();
        spec = Describe(*first);
        interop.emplace(device, *decoder, spec.size, spec.format);
    }

    [[nodiscard]] std::unique_ptr<BestVideoFrame> ReadFrame() {
        std::unique_ptr<BestVideoFrame> frame(source.GetFrame(next));
        if (!frame)
            throw std::runtime_error(
                "BestSource: could not decode frame " + std::to_string(next)
            );
        return frame;
    }

    void CheckTiming(std::int64_t pts) {
        if (pts == AV_NOPTS_VALUE) return;
        if (first_timestamp == AV_NOPTS_VALUE) {
            first_timestamp = pts;
            first_index = next;
            return;
        }
        const auto expected =
            av_rescale_q(next - first_index, av_inv_q(rate), time_base);
        const auto error =
            av_sat_sub64(av_sat_sub64(pts, first_timestamp), expected);
        if (error < -1 || error > 1)
            throw std::runtime_error(
                "BestSource input must have a constant frame rate"
            );
    }
};

BestSourceReader::BestSourceReader(
    const std::string& path, int track, BestSourceDevice& device,
    std::int64_t start, std::optional<std::int64_t> count
) try
    : impl_(std::make_unique<Impl>(path, track, device, start, count)) {
} catch (const std::exception& error) {
    throw std::runtime_error(path + ": " + error.what());
}

BestSourceReader::~BestSourceReader() = default;

Size BestSourceReader::size() const noexcept { return impl_->spec.size; }
const ImageFormat& BestSourceReader::format() const noexcept {
    return impl_->spec.format;
}
image_io::FrameRate BestSourceReader::frame_rate() const noexcept {
    return {static_cast<std::uint32_t>(impl_->rate.num),
            static_cast<std::uint32_t>(impl_->rate.den)};
}

image_io::SequenceReadResult BestSourceReader::Next(
    VulkanBufferView& view, std::string& error
) {
    using enum image_io::SequenceReadResult;
    if (impl_->failed) {
        error = "BestSource reader failed earlier";
        return Error;
    }
    if (impl_->next == impl_->end) return End;
    try {
        auto frame = impl_->first ? std::move(impl_->first) : impl_->ReadFrame();
        if (!(Describe(*frame) == impl_->spec))
            throw std::runtime_error(
                "BestSource frame format changed within the selected range"
            );
        impl_->CheckTiming(frame->PTS);
        view = impl_->interop->Pack(
            *frame->GetAVFrame(), *impl_->source.GetGpuHasher()
        );
        ++impl_->next;
        return Frame;
    } catch (const std::exception& failure) {
        error = failure.what();
        impl_->failed = true;
        return Error;
    }
}

}  // namespace vqm::cli
