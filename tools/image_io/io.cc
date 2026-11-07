// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <jpeglib.h>
#include <jerror.h>
#include <memory>
#include <limits>
#include <optional>
#include <png.h>
#include <span>
#include <stdexcept>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "input.h"
#include "io.h"

namespace vqm::image_io {

// File-local: the decoders are only reached through LoadImage below.
static bool ReadPng(std::FILE* file, Image& image, std::string& error);
static bool ReadJpeg(std::FILE* file, Image& image, std::string& error);
static bool ReadY4m(std::FILE* file, Image& image, std::string& error);

}  // namespace vqm::image_io

namespace vqm::image_io {
namespace {

struct FileCloser {
    void operator()(std::FILE* file) const noexcept {
        if (file != nullptr) std::fclose(file);
    }
};
using File = std::unique_ptr<std::FILE, FileCloser>;

constexpr std::string_view kPngMagic{"\x89PNG\r\n\x1a\n", 8};
constexpr std::string_view kY4mMagic{"YUV4MPEG2"};
constexpr unsigned char kJpegMagic0 = 0xFF;
constexpr unsigned char kJpegMagic1 = 0xD8;

}  // namespace

std::optional<Image> LoadImage(std::string_view path, std::string& error) {
    const std::string path_string(path);
    File file(std::fopen(path_string.c_str(), "rb"));
    if (!file) {
        error = "could not open file";
        return std::nullopt;
    }
    std::array<unsigned char, 10> magic = {};
    const std::size_t got =
        std::fread(magic.data(), 1, magic.size(), file.get());
    std::rewind(file.get());
    Image image;
    if (got >= kY4mMagic.size() &&
        std::string_view(
            reinterpret_cast<const char*>(magic.data()), kY4mMagic.size()
        ) == kY4mMagic) {
        if (!ReadY4m(file.get(), image, error)) return std::nullopt;
        return image;
    }
    if (got >= kPngMagic.size() &&
        std::string_view(
            reinterpret_cast<const char*>(magic.data()), kPngMagic.size()
        ) == kPngMagic) {
        if (!ReadPng(file.get(), image, error)) return std::nullopt;
        return image;
    }
    if (got >= 2 && magic[0] == kJpegMagic0 && magic[1] == kJpegMagic1) {
        if (!ReadJpeg(file.get(), image, error)) return std::nullopt;
        return image;
    }
    error = "unsupported image format (PNG, JPEG and YUV4MPEG2 are supported)";
    return std::nullopt;
}

}  // namespace vqm::image_io

namespace vqm::image_io {
namespace {

void PngErrorHandler(png_structp png, png_const_charp message) {
    auto* error = static_cast<std::string*>(png_get_error_ptr(png));
    if (error != nullptr) {
        *error = message != nullptr ? message : "PNG read error";
    }
    png_longjmp(png, 1);
}

void PngWarningHandler(png_structp, png_const_charp) {}

// Heap-allocated so that no automatic object is modified between setjmp() and
// a potential longjmp().
struct PngReadState {
    png_structp png = nullptr;
    png_infop info = nullptr;
    std::vector<png_bytep> rows;
    std::vector<unsigned char> pixels;
    std::size_t row_bytes = 0;

    ~PngReadState() {
        if (png != nullptr) png_destroy_read_struct(&png, &info, nullptr);
    }
};

// The image plane each interleaved channel of a decoded row feeds, indexed by
// channels - 1. Planes are zimg's: Y/R, U/G, V/B, alpha.
constexpr std::array<std::array<int, 4>, 4> kChannelPlanes{{
    {0, -1, -1, -1},  // grey
    {0, 3, -1, -1},   // grey + alpha
    {0, 1, 2, -1},    // RGB
    {0, 1, 2, 3},     // RGBA
}};

// Copies the decoded rows into the packed plane layout, byte for byte: an
// 8-bit PNG becomes Byte samples, a 16-bit one Word samples in host order.
template <std::size_t SampleBytes>
void StoreRows(const PngReadState& state, int channels, Image& image) {
    const auto layout = internal::PlaneLayout(image.format, image.size);
    const auto& planes = kChannelPlanes[static_cast<std::size_t>(channels - 1)];
    const std::size_t width = image.size.width;
    const std::size_t channel_count = static_cast<std::size_t>(channels);
    for (std::size_t y = 0; y < image.size.height; ++y) {
        const unsigned char* row = state.rows[y];
        for (std::size_t channel = 0; channel < channel_count; ++channel) {
            const auto plane = static_cast<std::size_t>(planes[channel]);
            unsigned char* out = image.data.data() + layout[plane].offset +
                                 y * width * SampleBytes;
            for (std::size_t x = 0; x < width; ++x) {
                std::memcpy(
                    out + x * SampleBytes,
                    row + (channel + x * channel_count) * SampleBytes,
                    SampleBytes
                );
            }
        }
    }
}

}  // namespace

bool ReadPng(std::FILE* file, Image& image, std::string& error) {
    // All mutable decoder state lives on the heap. Its owner is established
    // before setjmp, so both libpng errors and C++ exceptions release it.
    const auto state = std::make_unique<PngReadState>();
    state->png = png_create_read_struct(
        PNG_LIBPNG_VER_STRING, &error, PngErrorHandler, PngWarningHandler
    );
    png_structp png = state->png;
    if (png == nullptr) {
        error = "could not allocate PNG read struct";
        return false;
    }
    state->info = png_create_info_struct(png);
    png_infop info = state->info;
    if (info == nullptr) {
        error = "could not allocate PNG info struct";
        return false;
    }
    if (setjmp(png_jmpbuf(png))) {
        return false;
    }

    png_init_io(png, file);
    png_read_info(png, info);

    png_set_expand(png);
    png_set_palette_to_rgb(png);
    png_set_tRNS_to_alpha(png);
    // The arena holds Word samples in host byte order.
    if constexpr (std::endian::native == std::endian::little) {
        png_set_swap(png);
    }
    (void)png_set_interlace_handling(png);
    png_read_update_info(png, info);

    const png_uint_32 width = png_get_image_width(png, info);
    const png_uint_32 height = png_get_image_height(png, info);
    const int channels = png_get_channels(png, info);
    const int bit_depth = png_get_bit_depth(png, info);
    if (bit_depth != 8 && bit_depth != 16) {
        error = "unsupported PNG bit depth (only 8 and 16 are supported)";
        return false;
    }
    if (channels < 1 || channels > 4) {
        error = "unsupported PNG channel count";
        return false;
    }

    state->row_bytes = png_get_rowbytes(png, info);
    if (state->row_bytes > std::numeric_limits<std::size_t>::max() / height) {
        error = "PNG image layout exceeds size_t";
        return false;
    }
    state->pixels.resize(state->row_bytes * height);
    state->rows.resize(height);
    for (png_uint_32 y = 0; y < height; ++y) {
        state->rows[y] = state->pixels.data() + state->row_bytes * y;
    }
    png_read_image(png, state->rows.data());
    png_read_end(png, nullptr);

    image.size = {width, height};
    image.format = ImageFormat{};
    image.format.type = bit_depth == 16 ? PixelType::Word : PixelType::Byte;
    image.format.family = channels <= 2 ? ColorFamily::Grey : ColorFamily::Rgb;
    image.format.alpha =
        channels == 2 || channels == 4 ? AlphaType::Straight : AlphaType::None;
    // A greyscale plane has no matrix, but the enum still has to stay out of
    // the RGB family's hands.
    image.format.matrix = image.format.family == ColorFamily::Grey
                              ? MatrixCoefficients::Bt709
                              : MatrixCoefficients::Rgb;
    image.format.transfer = TransferCharacteristics::Srgb;
    image.format.primaries = ColorPrimaries::Bt709;
    image.format.depth = bit_depth == 16 ? 16 : 0;
    image.data.assign(RequiredBytes(image.format, image.size), 0u);
    if (bit_depth == 16)
        StoreRows<2>(*state, channels, image);
    else
        StoreRows<1>(*state, channels, image);
    return true;
}

}  // namespace vqm::image_io

namespace vqm::image_io {
namespace {

struct JpegErrorState {
    jpeg_error_mgr pub;
    jmp_buf jump;
    std::array<char, JMSG_LENGTH_MAX> message{};
    void (*default_emit_message)(j_common_ptr, int) = nullptr;
};

struct JpegReadState {
    jpeg_decompress_struct decoder{};
    JpegErrorState error;
    std::vector<unsigned char> row;

    ~JpegReadState() { jpeg_destroy_decompress(&decoder); }
};

void JpegErrorHandler(j_common_ptr cinfo) {
    auto* state = reinterpret_cast<JpegErrorState*>(cinfo->err);
    state->pub.format_message(cinfo, state->message.data());
    longjmp(state->jump, 1);
}

void JpegMessageHandler(j_common_ptr cinfo, int level) {
    // libjpeg can recover missing scan samples by synthesizing pixels. Do not
    // score that recovery as an image. EOF-only and metadata warnings retain
    // their normal behavior when no compressed samples are missing.
    if (level < 0 && cinfo->err->msg_code == JWRN_HIT_MARKER) {
        JpegErrorHandler(cinfo);
    }
    auto* state = reinterpret_cast<JpegErrorState*>(cinfo->err);
    state->default_emit_message(cinfo, level);
}

}  // namespace

bool ReadJpeg(std::FILE* file, Image& image, std::string& error) {
    // As with PNG, longjmp changes only heap state and does not bypass the
    // owner's lifetime. A zero-initialized decoder can also be destroyed if
    // jpeg_create_decompress itself fails.
    const auto state = std::make_unique<JpegReadState>();
    jpeg_decompress_struct& cinfo = state->decoder;
    cinfo.err = jpeg_std_error(&state->error.pub);
    state->error.pub.error_exit = JpegErrorHandler;
    state->error.default_emit_message = state->error.pub.emit_message;
    state->error.pub.emit_message = JpegMessageHandler;
    if (setjmp(state->error.jump)) {
        error = state->error.message[0] != '\0' ? state->error.message.data()
                                                : "JPEG read error";
        return false;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);

    if (cinfo.num_components != 1 && cinfo.num_components != 3) {
        error = "unsupported number of components in JPEG";
        return false;
    }

    jpeg_start_decompress(&cinfo);
    const int components = cinfo.output_components;
    if (components != cinfo.num_components) {
        error = "unsupported JPEG color transform";
        return false;
    }

    const JDIMENSION width = cinfo.output_width;
    const JDIMENSION height = cinfo.output_height;
    image.size = {width, height};
    image.format = ImageFormat{};
    image.format.type = PixelType::Byte;
    image.format.family = components == 1 ? ColorFamily::Grey : ColorFamily::Rgb;
    image.format.matrix = image.format.family == ColorFamily::Grey
                              ? MatrixCoefficients::Bt709
                              : MatrixCoefficients::Rgb;
    image.format.transfer = TransferCharacteristics::Srgb;
    image.format.primaries = ColorPrimaries::Bt709;
    image.data.assign(RequiredBytes(image.format, image.size), 0u);

    // libjpeg yields one interleaved row at a time; scatter it into the
    // planar buffer.
    const std::array<internal::PlaneGeometry, 4> layout =
        internal::PlaneLayout(image.format, image.size);
    unsigned char* const data = image.data.data();
    state->row.resize(image.size.width * static_cast<std::size_t>(components));
    const std::span<unsigned char> row_span(state->row);
    for (JDIMENSION y = 0; y < height; ++y) {
        JSAMPROW row_pointer = row_span.data();
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
        const std::size_t base = static_cast<std::size_t>(y) * image.size.width;
        if (components == 1) {
            std::copy(
                row_span.begin(),
                row_span.begin() + static_cast<std::ptrdiff_t>(width),
                data + layout[0].offset + base
            );
            continue;
        }
        unsigned char* r = data + layout[0].offset + base;
        unsigned char* g = data + layout[1].offset + base;
        unsigned char* b = data + layout[2].offset + base;
        for (JDIMENSION x = 0; x < width; ++x) {
            const unsigned char* pixel =
                row_span.data() + static_cast<std::size_t>(x) * 3u;
            r[x] = pixel[0];
            g[x] = pixel[1];
            b[x] = pixel[2];
        }
    }

    jpeg_finish_decompress(&cinfo);
    return true;
}

}  // namespace vqm::image_io

namespace vqm::image_io {
namespace {

// Bound on one y4m header line: real headers are a few dozen bytes, and this
// only stops an unterminated one from reading the whole file as its header.
constexpr std::size_t kMaxHeaderLength = 1024;

struct ChromaTag {
    std::string_view name;
    ColorFamily family;
    std::uint8_t subsample_w;
    std::uint8_t subsample_h;
    ChromaLocation location;
    bool alpha;
};

// The chroma tags this reader accepts (yuv4mpeg2's C tag).
constexpr ChromaTag kChromaTags[] = {
    {"mono", ColorFamily::Grey, 0, 0, ChromaLocation::Left, false},
    {"420", ColorFamily::Yuv, 1, 1, ChromaLocation::Left, false},
    {"420mpeg2", ColorFamily::Yuv, 1, 1, ChromaLocation::Left, false},
    {"420jpeg", ColorFamily::Yuv, 1, 1, ChromaLocation::Center, false},
    {"420paldv", ColorFamily::Yuv, 1, 1, ChromaLocation::TopLeft, false},
    {"422", ColorFamily::Yuv, 1, 0, ChromaLocation::Left, false},
    {"444", ColorFamily::Yuv, 0, 0, ChromaLocation::Left, false},
    {"411", ColorFamily::Yuv, 2, 0, ChromaLocation::Left, false},
    {"444alpha", ColorFamily::Yuv, 0, 0, ChromaLocation::Left, true},
};

// Reads one line, without the newline, up to `limit` characters.
bool ReadLine(std::FILE* file, std::size_t limit, std::string& line) {
    line.clear();
    for (;;) {
        const int c = std::fgetc(file);
        if (c == EOF) return !line.empty();
        if (c == '\n') return true;
        if (line.size() >= limit) return false;
        line.push_back(static_cast<char>(c));
    }
}

bool ParseUnsigned(std::string_view text, std::size_t& out) {
    if (text.empty()) return false;
    std::size_t value = 0;
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        return false;
    out = value;
    return true;
}

// Splits "C420p10" into the base tag and an optional bit depth.
bool ParseChromaTag(
    std::string_view tag, const ChromaTag*& out_tag, std::uint8_t& out_depth
) {
    out_depth = 0;
    std::string_view base = tag;
    const std::size_t p = tag.rfind('p');
    if (p != std::string_view::npos && p + 1 < tag.size()) {
        const std::string_view digits = tag.substr(p + 1);
        std::size_t depth = 0;
        // The pN depths above 8 a 16-bit word sample can carry and y4m
        // streams actually exist for (10/12 common, 9 and 14 rare; 8 is the
        // unsuffixed default). Anything else -- 11, 13, 15, 17 -- leaves the
        // whole tag as the base, which then fails the table below.
        if (ParseUnsigned(digits, depth) &&
            (depth == 9 || depth == 10 || depth == 12 || depth == 14 ||
             depth == 16)) {
            base = tag.substr(0, p);
            out_depth = static_cast<std::uint8_t>(depth);
        }
    }
    for (const ChromaTag& candidate : kChromaTags) {
        if (candidate.name == base) {
            out_tag = &candidate;
            return true;
        }
    }
    return false;
}

}  // namespace

namespace {

struct Y4mHeader {
    Size size;
    ImageFormat format;
    FrameRate fps;
    bool valid_timing = true;
    bool progressive = true;
};

bool ParseFrameRate(std::string_view text, FrameRate& fps) {
    const std::size_t colon = text.find(':');
    std::size_t numerator = 0;
    std::size_t denominator = 0;
    if (colon == std::string_view::npos ||
        !ParseUnsigned(text.substr(0, colon), numerator) ||
        !ParseUnsigned(text.substr(colon + 1), denominator) || numerator == 0 ||
        denominator == 0 || numerator > UINT32_MAX || denominator > UINT32_MAX)
        return false;
    const std::size_t divisor = std::gcd(numerator, denominator);
    fps = {static_cast<std::uint32_t>(numerator / divisor),
           static_cast<std::uint32_t>(denominator / divisor)};
    return true;
}

bool ReadY4mHeader(std::FILE* file, Y4mHeader& out, std::string& error) {
    // The same magic LoadImage sniffs, re-checked here because it also
    // validates the stream's own header, not just the format guess.
    std::vector<char> magic(kY4mMagic.size());
    if (std::fread(magic.data(), 1, magic.size(), file) != magic.size() ||
        std::string_view(magic.data(), magic.size()) != kY4mMagic) {
        error = "not a YUV4MPEG2 file";
        return false;
    }

    std::string header;
    if (!ReadLine(file, kMaxHeaderLength, header)) {
        error = "truncated y4m header";
        return false;
    }

    std::size_t width = 0;
    std::size_t height = 0;
    bool have_width = false;
    bool have_height = false;
    // MJPEGtools utils/yuv4mpeg.h specifies 420jpeg for an omitted C tag:
    // https://svn.code.sf.net/p/mjpeg/Code/trunk/mjpeg_play/utils/yuv4mpeg.h
    std::string chroma = "420jpeg";
    for (std::size_t i = 0; i < header.size();) {
        if (header[i] == ' ') {
            ++i;
            continue;
        }
        const std::size_t start = i;
        while (i < header.size() && header[i] != ' ') ++i;
        const std::string_view token(header.data() + start, i - start);
        if (token.size() < 2) continue;
        const std::string_view value = token.substr(1);
        switch (token[0]) {
            case 'W':
                have_width = ParseUnsigned(value, width);
                if (!have_width) {
                    error = "invalid y4m width";
                    return false;
                }
                break;
            case 'H':
                have_height = ParseUnsigned(value, height);
                if (!have_height) {
                    error = "invalid y4m height";
                    return false;
                }
                break;
            case 'C':
                chroma = std::string(value);
                break;
            case 'F':
                out.valid_timing = ParseFrameRate(value, out.fps);
                break;
            case 'I':
                out.progressive = value == "p";
                break;
            default:
                // A (aspect) and X (comment) are not image sample metadata.
                break;
        }
    }
    if (!have_width || !have_height || width == 0 || height == 0) {
        error = "y4m header has no frame size";
        return false;
    }

    const ChromaTag* tag = nullptr;
    std::uint8_t depth = 0;
    if (!ParseChromaTag(chroma, tag, depth)) {
        error = "unsupported y4m chroma tag";
        return false;
    }
    if (width % (std::size_t{1} << tag->subsample_w) != 0 ||
        height % (std::size_t{1} << tag->subsample_h) != 0) {
        error = "y4m dimensions are not divisible by the chroma subsampling";
        return false;
    }

    out.size = {width, height};
    out.format = ImageFormat{};
    out.format.type = depth == 0 ? PixelType::Byte : PixelType::Word;
    out.format.depth = depth;
    out.format.family = tag->family;
    out.format.alpha = tag->alpha ? AlphaType::Straight : AlphaType::None;
    out.format.chroma_location = tag->location;
    out.format.subsample_w = tag->subsample_w;
    out.format.subsample_h = tag->subsample_h;
    out.format.range = PixelRange::Limited;
    // y4m carries no colours. Rec.601 is the format's de-facto default; the
    // CLI's --input-* flags override it.
    out.format.matrix = MatrixCoefficients::Bt470Bg;
    out.format.transfer = TransferCharacteristics::Bt709;
    out.format.primaries = ColorPrimaries::Bt709;
    return true;
}

SequenceReadResult ReadY4mFrame(
    std::FILE* file, const Y4mHeader& header, Image& image, std::string& error
) {
    const int first = std::fgetc(file);
    if (first == EOF) {
        if (!std::ferror(file)) return SequenceReadResult::End;
        error = "could not read y4m frame";
        return SequenceReadResult::Error;
    }
    std::ungetc(first, file);
    std::string frame;
    if (!ReadLine(file, kMaxHeaderLength, frame) ||
        (frame != "FRAME" && !std::string_view(frame).starts_with("FRAME "))) {
        error = "y4m file has no frame";
        return SequenceReadResult::Error;
    }
    image.size = header.size;
    image.format = header.format;
    image.data.assign(RequiredBytes(image.format, image.size), 0u);
    const std::array<internal::PlaneGeometry, 4> layout =
        internal::PlaneLayout(image.format, image.size);
    for (std::size_t plane = 0; plane < layout.size(); ++plane) {
        if (!layout[plane].present()) continue;
        const std::size_t samples = layout[plane].width * layout[plane].height;
        const std::size_t bytes = samples * layout[plane].sample_bytes;
        unsigned char* base = image.data.data() + layout[plane].offset;
        if (std::fread(base, 1, bytes, file) != bytes) {
            error = "truncated y4m frame";
            return SequenceReadResult::Error;
        }
        // FFmpeg's high-bit-depth Y4M extension stores little-endian words;
        // the image and arena use host byte order.
        if (layout[plane].sample_bytes == 2 &&
            std::endian::native == std::endian::big) {
            for (std::size_t i = 0; i < samples; ++i) {
                std::swap(base[i * 2], base[i * 2 + 1]);
            }
        }
    }
    return SequenceReadResult::Frame;
}

}  // namespace

bool ReadY4m(std::FILE* file, Image& image, std::string& error) {
    Y4mHeader header;
    if (!ReadY4mHeader(file, header, error)) return false;
    const SequenceReadResult result = ReadY4mFrame(file, header, image, error);
    if (result == SequenceReadResult::End) error = "y4m file has no frame";
    return result == SequenceReadResult::Frame;
}

struct SequenceReader::Impl {
    File file;
    Y4mHeader header;
    std::optional<Image> still;
    bool video = false;
    bool ended = false;
};

SequenceReader::SequenceReader(std::string_view path)
    : impl_(std::make_unique<Impl>()) {
    const std::string filename(path);
    impl_->file.reset(std::fopen(filename.c_str(), "rb"));
    if (!impl_->file) throw std::runtime_error("could not open " + filename);
    std::array<char, 9> magic{};
    const std::size_t bytes =
        std::fread(magic.data(), 1, magic.size(), impl_->file.get());
    std::rewind(impl_->file.get());
    std::string error;
    if (bytes != magic.size() ||
        std::string_view(magic.data(), bytes) != kY4mMagic) {
        impl_->still = LoadImage(path, error);
        if (!impl_->still) throw std::runtime_error(filename + ": " + error);
        impl_->header.size = impl_->still->size;
        impl_->header.format = impl_->still->format;
        impl_->file.reset();
        return;
    }
    impl_->video = true;
    if (!ReadY4mHeader(impl_->file.get(), impl_->header, error))
        throw std::runtime_error(filename + ": " + error);
    if (!impl_->header.valid_timing || impl_->header.fps.numerator == 0)
        throw std::runtime_error(filename + ": y4m requires a valid frame rate");
    if (!impl_->header.progressive)
        throw std::runtime_error(
            filename + ": only progressive y4m is supported"
        );
}

SequenceReader::~SequenceReader() = default;
SequenceReader::SequenceReader(SequenceReader&&) noexcept = default;
SequenceReader& SequenceReader::operator=(SequenceReader&&) noexcept = default;
Size SequenceReader::size() const noexcept { return impl_->header.size; }
const ImageFormat& SequenceReader::format() const noexcept {
    return impl_->header.format;
}
FrameRate SequenceReader::frame_rate() const noexcept {
    return impl_->header.fps;
}
bool SequenceReader::is_video() const noexcept { return impl_->video; }

SequenceReadResult SequenceReader::Next(Image& image, std::string& error) {
    error.clear();
    if (impl_->ended) return SequenceReadResult::End;
    if (impl_->still) {
        image = std::move(*impl_->still);
        impl_->still.reset();
        impl_->ended = true;
        return SequenceReadResult::Frame;
    }
    const auto result =
        ReadY4mFrame(impl_->file.get(), impl_->header, image, error);
    impl_->ended = result != SequenceReadResult::Frame;
    return result;
}

}  // namespace vqm::image_io

// Shared command-line colour declarations

namespace vqm::image_io {
namespace {

template <typename T, std::size_t N>
bool ParseNamed(
    const std::array<std::pair<std::string_view, T>, N>& table,
    std::string_view text, T& out, std::string& accepted
) {
    accepted.clear();
    for (const auto& [name, value] : table) {
        if (name == text) {
            out = value;
            return true;
        }
        if (!accepted.empty()) accepted += '|';
        accepted += name;
    }
    return false;
}

inline constexpr std::array<std::pair<std::string_view, MatrixCoefficients>, 12>
    kMatrixNames{{
        {"rgb", MatrixCoefficients::Rgb},
        {"bt709", MatrixCoefficients::Bt709},
        {"fcc", MatrixCoefficients::Fcc},
        {"bt470bg", MatrixCoefficients::Bt470Bg},
        {"st170m", MatrixCoefficients::St170M},
        {"st240m", MatrixCoefficients::St240M},
        {"ycgco", MatrixCoefficients::YCgCo},
        {"bt2020ncl", MatrixCoefficients::Bt2020Ncl},
        {"bt2020cl", MatrixCoefficients::Bt2020Cl},
        {"chromaticity-derived-ncl", MatrixCoefficients::ChromaticityDerivedNcl},
        {"chromaticity-derived-cl", MatrixCoefficients::ChromaticityDerivedCl},
        {"ictcp", MatrixCoefficients::Bt2100ICtCp},
    }};

inline constexpr std::array<
    std::pair<std::string_view, TransferCharacteristics>, 16>
    kTransferNames{{
        {"bt709", TransferCharacteristics::Bt709},
        {"bt601", TransferCharacteristics::Bt601},
        {"bt470m", TransferCharacteristics::Bt470M},
        {"bt470bg", TransferCharacteristics::Bt470Bg},
        {"st240m", TransferCharacteristics::St240M},
        {"linear", TransferCharacteristics::Linear},
        {"log100", TransferCharacteristics::Log100},
        {"log316", TransferCharacteristics::Log316},
        {"xvycc", TransferCharacteristics::Iec61966_2_4},
        {"bt1361", TransferCharacteristics::Bt1361},
        {"srgb", TransferCharacteristics::Srgb},
        {"bt2020-10", TransferCharacteristics::Bt2020_10},
        {"bt2020-12", TransferCharacteristics::Bt2020_12},
        {"st2084", TransferCharacteristics::St2084},
        {"st428", TransferCharacteristics::St428},
        {"arib-b67", TransferCharacteristics::AribB67},
    }};

inline constexpr std::array<std::pair<std::string_view, ColorPrimaries>, 11>
    kPrimariesNames{{
        {"bt709", ColorPrimaries::Bt709},
        {"bt470m", ColorPrimaries::Bt470M},
        {"bt470bg", ColorPrimaries::Bt470Bg},
        {"st170m", ColorPrimaries::St170M},
        {"st240m", ColorPrimaries::St240M},
        {"film", ColorPrimaries::Film},
        {"bt2020", ColorPrimaries::Bt2020},
        {"st428", ColorPrimaries::St428},
        {"st431-2", ColorPrimaries::St431_2},
        {"st432-1", ColorPrimaries::St432_1},
        {"ebu3213e", ColorPrimaries::Ebu3213E},
    }};

inline constexpr std::array<std::pair<std::string_view, PixelRange>, 2>
    kRangeNames{{
        {"limited", PixelRange::Limited},
        {"full", PixelRange::Full},
    }};

inline constexpr std::array<std::pair<std::string_view, ChromaFilter>, 3>
    kFilterNames{{
        {"bilinear", ChromaFilter::Bilinear},
        {"bicubic", ChromaFilter::Bicubic},
        {"lanczos", ChromaFilter::Lanczos},
    }};

constexpr std::array<std::pair<std::string_view, InputColorOption>, 5>
    kInputColorFlags{{
        {"--input-matrix", InputColorOption::Matrix},
        {"--input-transfer", InputColorOption::Transfer},
        {"--input-primaries", InputColorOption::Primaries},
        {"--input-range", InputColorOption::Range},
        {"--chroma-filter", InputColorOption::ChromaFilter},
    }};

template <typename T, std::size_t N>
bool ParseOverride(
    const std::array<std::pair<std::string_view, T>, N>& names,
    std::string_view text, std::optional<T>& out, std::string& accepted
) {
    T parsed{};
    const bool valid = ParseNamed(names, text, parsed, accepted);
    out = parsed;
    return valid;
}

}  // namespace

void InputColorOptions::ApplyTo(ImageFormat& format) const {
    if (matrix) format.matrix = *matrix;
    if (transfer) format.transfer = *transfer;
    if (primaries) format.primaries = *primaries;
    if (range) format.range = *range;
}

std::optional<InputColorOption> IdentifyInputColorOption(std::string_view flag) {
    for (const auto& [name, option] : kInputColorFlags) {
        if (flag == name) return option;
    }
    return std::nullopt;
}

bool ParseInputColorOption(
    InputColorOption option, std::string_view value, InputColorOptions& options,
    std::string& accepted
) {
    switch (option) {
        case InputColorOption::Matrix:
            return ParseOverride(kMatrixNames, value, options.matrix, accepted);
        case InputColorOption::Transfer:
            return ParseOverride(
                kTransferNames, value, options.transfer, accepted
            );
        case InputColorOption::Primaries:
            return ParseOverride(
                kPrimariesNames, value, options.primaries, accepted
            );
        case InputColorOption::Range:
            return ParseOverride(kRangeNames, value, options.range, accepted);
        case InputColorOption::ChromaFilter: {
            ChromaFilter parsed{};
            const bool valid = ParseNamed(kFilterNames, value, parsed, accepted);
            options.chroma_filter = parsed;
            return valid;
        }
    }
    accepted.clear();
    return false;
}

}  // namespace vqm::image_io
