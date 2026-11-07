// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

// One invocation writes one word. Each plane starts on a word boundary, so
// its final word can be padded without racing another invocation or plane.
extern "C" __global__ void PackNvdecFrame(
    unsigned* dst, const unsigned char* y, const unsigned char* u,
    const unsigned char* v, unsigned y_stride, unsigned u_stride,
    unsigned v_stride, unsigned width, unsigned height, unsigned subsample_w,
    unsigned subsample_h, unsigned sample_bytes, unsigned semiplanar,
    unsigned shift
) {
    const unsigned plane = blockIdx.y;
    const bool chroma = plane != 0;
    const unsigned plane_width = chroma ? width >> subsample_w : width;
    const unsigned plane_height = chroma ? height >> subsample_h : height;
    const unsigned count = plane_width * plane_height;
    const unsigned samples_per_word = 4 / sample_bytes;
    const unsigned word = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned first = word * samples_per_word;
    if (first >= count) return;

    const bool interleaved = chroma && semiplanar;
    const auto* source = !chroma ? y : (interleaved || plane == 1 ? u : v);
    const unsigned stride =
        !chroma ? y_stride : (interleaved || plane == 1 ? u_stride : v_stride);
    const unsigned step = sample_bytes * (interleaved ? 2u : 1u);
    if (interleaved && plane == 2) source += sample_bytes;

    unsigned packed = 0;
    for (unsigned lane = 0; lane < samples_per_word && first + lane < count;
         ++lane) {
        const unsigned sample = first + lane;
        const auto* code =
            source +
            static_cast<unsigned long long>(sample / plane_width) * stride +
            (sample % plane_width) * step;
        unsigned value = code[0];
        if (sample_bytes == 2) value |= static_cast<unsigned>(code[1]) << 8;
        packed |= (value >> shift) << (lane * sample_bytes * 8);
    }
    const unsigned y_words =
        (width * height + samples_per_word - 1) / samples_per_word;
    const unsigned uv_words = (count + samples_per_word - 1) / samples_per_word;
    const unsigned offset = chroma ? y_words + (plane - 1) * uv_words : 0;
    dst[offset + word] = packed;
}
