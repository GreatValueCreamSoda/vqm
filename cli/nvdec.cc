// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#include "nvdec.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <tuple>

#include "nvdec_interop.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/common.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/log.h>
#include <libavutil/pixdesc.h>
}

namespace vqm::cli {
namespace {

void CheckAv(int status, std::string_view operation) {
    if (status >= 0) return;
    char error[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(status, error, sizeof(error));
    throw std::runtime_error(std::string(operation) + ": " + error);
}

template <typename T, void (*Free)(T**)>
using AvOwner = std::unique_ptr<T, decltype([](T* pointer) { Free(&pointer); })>;

template <typename T>
T* RequireAllocation(T* pointer) {
    if (!pointer) throw std::bad_alloc();
    return pointer;
}

AVPixelFormat SelectCuda(AVCodecContext*, const AVPixelFormat* formats) {
    for (; *formats != AV_PIX_FMT_NONE; ++formats) {
        if (*formats == AV_PIX_FMT_CUDA) return *formats;
    }
    return AV_PIX_FMT_NONE;
}

bool HasCudaDecoder(const AVCodec& codec) {
    for (int i = 0; const auto* config = avcodec_get_hw_config(&codec, i); ++i) {
        if (config->device_type == AV_HWDEVICE_TYPE_CUDA &&
            config->pix_fmt == AV_PIX_FMT_CUDA &&
            (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX))
            return true;
    }
    return false;
}

const AVCodec* FindCudaDecoder(AVCodecID id) {
    // The default decoder can be software-only, as with libdav1d for AV1.
    void* iterator = nullptr;
    while (const auto* codec = av_codec_iterate(&iterator)) {
        if (codec->id == id && av_codec_is_decoder(codec) &&
            HasCudaDecoder(*codec))
            return codec;
    }
    return nullptr;
}

ImageFormat ColorFormat(const AVFrame& frame) {
    // FFmpeg and vqm use the H.273 values for matrix, transfer and primaries.
    // Use the Y4M defaults for unspecified metadata.
    ImageFormat format;
    format.family = ColorFamily::Yuv;
    format.range = frame.color_range == AVCOL_RANGE_JPEG ? PixelRange::Full
                                                         : PixelRange::Limited;
    format.matrix = frame.colorspace == AVCOL_SPC_UNSPECIFIED
                        ? MatrixCoefficients::Bt470Bg
                        : static_cast<MatrixCoefficients>(frame.colorspace);
    format.transfer =
        frame.color_trc == AVCOL_TRC_UNSPECIFIED
            ? TransferCharacteristics::Bt709
            : static_cast<TransferCharacteristics>(frame.color_trc);
    format.primaries = frame.color_primaries == AVCOL_PRI_UNSPECIFIED
                           ? ColorPrimaries::Bt709
                           : static_cast<ColorPrimaries>(frame.color_primaries);
    if (frame.chroma_location >= AVCHROMA_LOC_LEFT &&
        frame.chroma_location <= AVCHROMA_LOC_BOTTOM)
        format.chroma_location = static_cast<ChromaLocation>(
            frame.chroma_location - AVCHROMA_LOC_LEFT
        );
    return format;
}

auto ColorKey(const ImageFormat& format) {
    return std::tie(
        format.range, format.matrix, format.transfer, format.primaries,
        format.chroma_location
    );
}

struct Layout {
    AVPixelFormat surface;
    bool semiplanar;
    unsigned shift;
    int depth;

    bool operator==(const Layout&) const = default;
};

bool ComponentMatches(
    const AVComponentDescriptor& component, int plane, int step, int offset,
    const AVComponentDescriptor& luma
) {
    return component.plane == plane && component.step == step &&
           component.offset == offset && component.depth == luma.depth &&
           component.shift == luma.shift;
}

Layout DescribeLayout(AVPixelFormat surface, AVPixelFormat coded) {
    const auto* desc = av_pix_fmt_desc_get(surface);
    const auto* coded_desc = av_pix_fmt_desc_get(coded);
    if (!desc || !coded_desc || desc->nb_components != 3 ||
        coded_desc->nb_components != 3 ||
        (desc->flags & (AV_PIX_FMT_FLAG_BE | AV_PIX_FMT_FLAG_RGB)) ||
        desc->log2_chroma_w > 1 || desc->log2_chroma_h > desc->log2_chroma_w ||
        desc->log2_chroma_w != coded_desc->log2_chroma_w ||
        desc->log2_chroma_h != coded_desc->log2_chroma_h)
        throw std::runtime_error(
            "NVDEC requires an 8-, 10- or 12-bit YUV surface"
        );

    const int depth = coded_desc->comp[0].depth;
    if (depth != 8 && depth != 10 && depth != 12)
        throw std::runtime_error("unsupported NVDEC coded bit depth");
    const auto& y = desc->comp[0];
    const auto& u = desc->comp[1];
    const auto& v = desc->comp[2];
    const int bytes = depth == 8 ? 1 : 2;
    const bool semiplanar = u.plane == v.plane;
    const int step = bytes * (semiplanar ? 2 : 1);
    if ((y.depth != depth && !(bytes == 2 && y.depth == 16)) ||
        !ComponentMatches(y, 0, bytes, 0, y) ||
        !ComponentMatches(u, 1, step, 0, y) ||
        !ComponentMatches(
            v, semiplanar ? 1 : 2, step, semiplanar ? bytes : 0, y
        ))
        throw std::runtime_error("unsupported NVDEC CUDA surface layout");
    // Older FFmpeg exposes 12-bit NVDEC surfaces as P016, and 10/12-bit
    // 4:4:4 as YUV444P16. The decoder still writes the high bits of each word.
    const auto shift =
        static_cast<unsigned>(y.depth == 16 ? 16 - depth : y.shift);
    return {surface, semiplanar, shift, depth};
}

struct FrameSpec {
    Size size;
    ImageFormat format;
    Layout layout;

    bool operator==(const FrameSpec& other) const {
        return size == other.size && layout == other.layout &&
               ColorKey(format) == ColorKey(other.format);
    }
};

const AVHWFramesContext& CudaFrames(const AVFrame& frame) {
    if (frame.format != AV_PIX_FMT_CUDA || !frame.hw_frames_ctx)
        throw std::runtime_error("NVDEC decoder did not return a CUDA frame");
    return *reinterpret_cast<const AVHWFramesContext*>(
        frame.hw_frames_ctx->data
    );
}

FrameSpec DescribeFrame(const AVFrame& frame, AVPixelFormat coded) {
    const auto& frames = CudaFrames(frame);
    if ((frame.flags & AV_FRAME_FLAG_CORRUPT) || frame.decode_error_flags)
        throw std::runtime_error("corrupt decoded video frame");
    if (frame.flags & AV_FRAME_FLAG_INTERLACED)
        throw std::runtime_error("NVDEC input must be progressive");
    if (frame.width <= 0 || frame.height <= 0 || frame.crop_left ||
        frame.crop_right || frame.crop_top || frame.crop_bottom)
        throw std::runtime_error("unsupported NVDEC frame geometry or crop");

    const auto layout = DescribeLayout(frames.sw_format, coded);
    const auto& desc = *av_pix_fmt_desc_get(layout.surface);
    if (frame.width % (1 << desc.log2_chroma_w) ||
        frame.height % (1 << desc.log2_chroma_h))
        throw std::runtime_error(
            "NVDEC frame size must align to its chroma grid"
        );
    auto format = ColorFormat(frame);
    format.type = layout.depth == 8 ? PixelType::Byte : PixelType::Word;
    format.depth = static_cast<std::uint8_t>(layout.depth);
    format.subsample_w = desc.log2_chroma_w;
    format.subsample_h = desc.log2_chroma_h;
    return {{static_cast<std::size_t>(frame.width),
             static_cast<std::size_t>(frame.height)},
            format,
            layout};
}

NvdecSource Source(const AVFrame& frame, const FrameSpec& spec) {
    const auto& frames = CudaFrames(frame);
    const auto* cuda =
        static_cast<const AVCUDADeviceContext*>(frames.device_ctx->hwctx);
    NvdecSource source{.stream = cuda->stream,
                       .semiplanar = spec.layout.semiplanar,
                       .shift = spec.layout.shift};
    for (unsigned plane = 0; plane < (source.semiplanar ? 2u : 3u); ++plane) {
        if (frame.linesize[plane] <= 0)
            throw std::runtime_error("invalid pitch in NVDEC CUDA frame");
        source.data[plane] = reinterpret_cast<CUdeviceptr>(frame.data[plane]);
        source.stride[plane] = static_cast<unsigned>(frame.linesize[plane]);
    }
    return source;
}

}  // namespace

bool IsCompressedVideo(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::array<char, 9> bytes{};
    file.read(bytes.data(), bytes.size());
    const std::string_view header(
        bytes.data(), static_cast<std::size_t>(file.gcount())
    );
    if (header.starts_with("YUV4MPEG2") ||
        header.starts_with("\x89PNG\r\n\x1a\n") ||
        header.starts_with("\xff\xd8"))
        return false;
    file.close();

    AVFormatContext* context = nullptr;
    const std::string filename = "file:" + path;
    const auto log_level = av_log_get_level();
    av_log_set_level(AV_LOG_QUIET);
    const int status =
        avformat_open_input(&context, filename.c_str(), nullptr, nullptr);
    av_log_set_level(log_level);
    AvOwner<AVFormatContext, avformat_close_input> input(context);
    if (status < 0) return false;
    const std::string_view format = input->iformat->name;
    if (format == "image2" || format == "image2pipe" ||
        format.ends_with("_pipe"))
        return false;
    const int stream =
        av_find_best_stream(input.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    return stream >= 0 &&
           !(input->streams[stream]->disposition & AV_DISPOSITION_ATTACHED_PIC);
}

struct NvdecDevice::Impl {
    NvdecInterop interop;
    AvOwner<AVBufferRef, av_buffer_unref> av_device;

    Impl() {
        av_log_set_level(AV_LOG_WARNING);
        // FFmpeg uses the matched ordinal for hardware-frame alignment and
        // borrows the CUDA context used for the Vulkan mappings.
        const CurrentCudaContext current(interop.context());
        const auto ordinal = std::to_string(interop.cuda_ordinal());
        AVBufferRef* device = nullptr;
        const int status = av_hwdevice_ctx_create(
            &device, AV_HWDEVICE_TYPE_CUDA, ordinal.c_str(), nullptr,
            AV_CUDA_USE_CURRENT_CONTEXT
        );
        av_device.reset(device);
        CheckAv(status, "initialize FFmpeg CUDA device");
    }
};

NvdecDevice::NvdecDevice() : impl_(std::make_unique<Impl>()) {}
NvdecDevice::~NvdecDevice() = default;

const VulkanDevice& NvdecDevice::vulkan() const noexcept {
    return impl_->interop.vulkan();
}

struct NvdecReader::Impl {
    NvdecInterop& interop;
    NvdecBuffer buffer;
    AvOwner<AVFormatContext, avformat_close_input> input;
    AvOwner<AVCodecContext, avcodec_free_context> decoder;
    AvOwner<AVPacket, av_packet_free> packet{
        RequireAllocation(av_packet_alloc())};
    AvOwner<AVFrame, av_frame_free> frame{RequireAllocation(av_frame_alloc())};
    AVStream* stream = nullptr;
    FrameSpec spec{};
    AVRational rate{};
    std::int64_t index = 0;
    std::int64_t first_index = 0;
    std::int64_t first_timestamp = AV_NOPTS_VALUE;
    bool pending = false;
    bool draining = false;
    bool ended = false;
    bool failed = false;

    Impl(const std::string& path, NvdecDevice::Impl& device)
        : interop(device.interop) {
        AVFormatContext* demux = nullptr;
        const std::string filename = "file:" + path;
        CheckAv(
            avformat_open_input(&demux, filename.c_str(), nullptr, nullptr),
            "open video"
        );
        input.reset(demux);
        OpenDecoder(device.av_device.get());
        if (!Receive()) throw std::runtime_error("input sequence has no frames");
        pending = true;
        spec = DescribeFrame(*frame, decoder->sw_pix_fmt);
        rate = stream->avg_frame_rate;
        if (rate.num <= 0 || rate.den <= 0)
            rate = av_guess_frame_rate(input.get(), stream, frame.get());
        if (rate.num <= 0 || rate.den <= 0) rate = decoder->framerate;
        if (rate.num <= 0 || rate.den <= 0 || stream->time_base.num <= 0 ||
            stream->time_base.den <= 0)
            throw std::runtime_error("NVDEC input has no usable frame rate");
        const auto gcd = std::gcd(rate.num, rate.den);
        rate = {rate.num / gcd, rate.den / gcd};
    }

    void OpenDecoder(AVBufferRef* cuda_device) {
        // Open the stream from demuxer metadata. FFmpeg's stream-info probe can
        // open a software decoder, so let the hardware decoder parse headers.
        const int selected = av_find_best_stream(
            input.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0
        );
        CheckAv(selected, "find video stream");
        stream = input->streams[selected];
        if (stream->disposition & AV_DISPOSITION_ATTACHED_PIC)
            throw std::runtime_error(
                "input has an attached picture instead of video"
            );
        const AVCodec* codec = FindCudaDecoder(stream->codecpar->codec_id);
        if (!codec)
            throw std::runtime_error(
                "FFmpeg has no NVDEC decoder for this codec"
            );
        const auto order = stream->codecpar->field_order;
        if (order != AV_FIELD_UNKNOWN && order != AV_FIELD_PROGRESSIVE)
            throw std::runtime_error("NVDEC input must be progressive");
        decoder.reset(RequireAllocation(avcodec_alloc_context3(codec)));
        CheckAv(
            avcodec_parameters_to_context(decoder.get(), stream->codecpar),
            "configure video decoder"
        );
        decoder->pkt_timebase = stream->time_base;
        decoder->hw_device_ctx = RequireAllocation(av_buffer_ref(cuda_device));
        decoder->get_format = SelectCuda;
        decoder->thread_count = 1;
        decoder->err_recognition = AV_EF_EXPLODE;
        CheckAv(
            avcodec_open2(decoder.get(), codec, nullptr), "open NVDEC decoder"
        );
    }

    void SendNextPacket() {
        for (;;) {
            const int read = av_read_frame(input.get(), packet.get());
            if (read == AVERROR_EOF) {
                if (input->pb) CheckAv(input->pb->error, "read video");
                CheckAv(
                    avcodec_send_packet(decoder.get(), nullptr), "drain NVDEC"
                );
                draining = true;
                return;
            }
            CheckAv(read, "read video packet");
            if (packet->stream_index == stream->index) break;
            av_packet_unref(packet.get());
        }
        if (packet->flags & AV_PKT_FLAG_CORRUPT)
            throw std::runtime_error("corrupt video packet");
        const int status = avcodec_send_packet(decoder.get(), packet.get());
        av_packet_unref(packet.get());
        CheckAv(status, "submit NVDEC packet");
    }

    bool Receive() {
        for (;;) {
            const int status = avcodec_receive_frame(decoder.get(), frame.get());
            if (status >= 0) return true;
            if (status == AVERROR_EOF) return false;
            if (status != AVERROR(EAGAIN) || draining)
                CheckAv(status, "receive NVDEC frame");
            SendNextPacket();
        }
    }

    void CheckTiming() {
        const auto timestamp = frame->best_effort_timestamp;
        if (timestamp == AV_NOPTS_VALUE) return;
        if (first_timestamp == AV_NOPTS_VALUE) {
            first_timestamp = timestamp;
            first_index = index;
            return;
        }
        const auto expected =
            av_rescale_q(index - first_index, av_inv_q(rate), stream->time_base);
        const auto actual = av_sat_sub64(timestamp, first_timestamp);
        const auto error = av_sat_sub64(actual, expected);
        // Container timestamps may round each endpoint to the nearest tick.
        if (error < -1 || error > 1)
            throw std::runtime_error(
                "NVDEC input must have a constant frame rate"
            );
    }

    image_io::SequenceReadResult Next(
        VulkanBufferView& view, std::string& error
    ) {
        using enum image_io::SequenceReadResult;
        if (ended) return End;
        if (failed) {
            error = "NVDEC reader failed earlier";
            return Error;
        }
        try {
            if (!pending && !Receive()) {
                ended = true;
                return End;
            }
            pending = false;
            if (DescribeFrame(*frame, decoder->sw_pix_fmt) != spec)
                throw std::runtime_error(
                    "NVDEC frame format changed within the clip"
                );
            CheckTiming();
            if (index == std::numeric_limits<std::int64_t>::max())
                throw std::overflow_error("too many NVDEC frames");
            view = interop.Pack(
                buffer, spec.size, spec.format, Source(*frame, spec)
            );
            av_frame_unref(frame.get());
            ++index;
            return Frame;
        } catch (const std::exception& ex) {
            error = ex.what();
            failed = true;
            return Error;
        }
    }
};

NvdecReader::NvdecReader(const std::string& path, NvdecDevice& device) try
    : impl_(std::make_unique<Impl>(path, *device.impl_)) {
} catch (const std::exception& ex) {
    throw std::runtime_error(path + ": " + ex.what());
}

NvdecReader::~NvdecReader() = default;

Size NvdecReader::size() const noexcept { return impl_->spec.size; }

const ImageFormat& NvdecReader::format() const noexcept {
    return impl_->spec.format;
}

image_io::FrameRate NvdecReader::frame_rate() const noexcept {
    return {static_cast<std::uint32_t>(impl_->rate.num),
            static_cast<std::uint32_t>(impl_->rate.den)};
}

image_io::SequenceReadResult NvdecReader::Next(
    VulkanBufferView& view, std::string& error
) {
    return impl_->Next(view, error);
}

}  // namespace vqm::cli
