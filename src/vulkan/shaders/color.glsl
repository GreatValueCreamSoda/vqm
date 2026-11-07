// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// The input colour pipeline of the prepare dispatch, from a raw sample to
// linear Rec.709 RGB. Every formula is a transliteration of the vendored zimg
// (see src/input.cc for the host-side constants and the reference
// locations); the arithmetic is float32 throughout and integer samples are
// only ever unpacked from a 32-bit word, never arithmetically shifted.
//
// The chain, per sample: decode (unpack + the depth module's scale/offset),
// chroma upsampling, the matrix/CL/ICtCp op, alpha, the transfer function and
// the primaries matrix. The compile-time class selects which of those cost
// anything:
//
//   VQM_SRC_CHROMA    0 = 4:4:4/GREY, 1 = subsampled (the chroma filter)
//   VQM_SRC_TRANSFER  0 = descriptor chain, 1 = sRGB, 2 = linear (identity)
//   VQM_SRC_FILTER    0 = bilinear, 1 = bicubic, 2 = lanczos (chroma = 1 only)
//
// A class that is not selected compiles no code and holds no registers for
// the path it does not use.

#ifndef VQM_SRC_CHROMA
#define VQM_SRC_CHROMA 0
#endif
#ifndef VQM_SRC_TRANSFER
#define VQM_SRC_TRANSFER 0
#endif
#ifndef VQM_SRC_FILTER
#define VQM_SRC_FILTER 0
#endif

// --- descriptor field offsets; see src/vulkan/vulkan_support.h
// -------------------

const uint kDescPlaneWord = 0u;  // 4 words: source word offset of each plane
const uint kDescEnums0 = 4u;
const uint kDescEnums1 = 5u;
const uint kDescScale = 6u;   // 3 words: luma, chroma, alpha
const uint kDescOffset = 9u;  // 3 words: luma, chroma, alpha
const uint kDescMatrix1 = 12u;
const uint kDescMatrix2 = 21u;
const uint kDescKr = 30u;
const uint kDescKg = 31u;
const uint kDescKb = 32u;
const uint kDescToLinearScale = 33u;
const uint kDescShift = 34u;  // shift_x, shift_y
const uint kDescClNb = 36u;   // cl_nb, cl_pb, cl_nr, cl_pr
const uint kDescRowSamples = 40u;
const uint kDescIdentity1 = 1u;
const uint kDescIdentity2 = 2u;

// Descriptor fields, relative to this image's block.
uint desc_word(uint index) { return desc[pc.desc_base + index]; }
float desc_f(uint index) { return uintBitsToFloat(desc[pc.desc_base + index]); }
// Plane offsets are relative to the source view (binding 3). Internal images
// bind the whole arena and retain absolute offsets; an external image binds its
// caller-owned buffer at an aligned base and uses offsets relative to that base.
uint arena_word(uint index) { return source_words[index]; }
uint desc_enums0() { return desc_word(kDescEnums0); }

uint src_type() { return desc_enums0() & 3u; }
uint src_family() { return (desc_enums0() >> 2) & 3u; }
uint src_alpha() { return (desc_enums0() >> 4) & 3u; }
uint src_sub_w() { return (desc_enums0() >> 6) & 3u; }
uint src_sub_h() { return (desc_enums0() >> 8) & 3u; }
uint src_transfer() { return (desc_enums0() >> 10) & 31u; }
uint src_mode() { return (desc_enums0() >> 17) & 3u; }
bool src_identity1() { return (desc_word(kDescEnums1) & kDescIdentity1) != 0u; }
bool src_identity2() { return (desc_word(kDescEnums1) & kDescIdentity2) != 0u; }

// --- sample decoding -------------------------------------------------------

// Luma and alpha are full resolution; the two chroma planes are not.
uint plane_width(uint plane) {
    const bool chroma = plane == 1u || plane == 2u;
    return source_width() >> (chroma ? src_sub_w() : 0u);
}
uint plane_height(uint plane) {
    const bool chroma = plane == 1u || plane == 2u;
    return source_height() >> (chroma ? src_sub_h() : 0u);
}

// 0 for luma/R, 1 for a YUV image's chroma planes, 2 for alpha. Only a YUV
// image has chroma: an RGB image's G and B planes carry the luma plane's
// depth-module mapping even in limited range.
uint plane_class(uint plane) {
    if (plane == 3u) return 2u;
    return plane != 0u && src_family() == 2u ? 1u : 0u;
}

// One raw sample, decoded to a normalised float. The plane base is word
// aligned (Image's layout guarantee), so the word is the sample index shifted
// and the byte lane is its remainder.
float raw_sample(uint plane, uint x, uint y) {
    const uint index = y * desc_word(kDescRowSamples + plane) + x;
    const uint base = desc_word(kDescPlaneWord + plane);
    const uint type = src_type();
    float value;
    if (type == 0u) {
        const uint word = arena_word(base + (index >> 2));
        value = float((word >> (8u * (index & 3u))) & 255u);
    } else if (type == 1u) {
        const uint word = arena_word(base + (index >> 1));
        value = float((word >> (16u * (index & 1u))) & 65535u);
    } else if (type == 2u) {
        const vec2 pair = unpackHalf2x16(arena_word(base + (index >> 1)));
        value = (index & 1u) != 0u ? pair.y : pair.x;
    } else {
        value = uintBitsToFloat(arena_word(base + index));
    }
    const uint cls = plane_class(plane);
    return value * desc_f(kDescScale + cls) + desc_f(kDescOffset + cls);
}

// --- transfer functions ----------------------------------------------------

const float kRec709Alpha = 1.09929682680944;
const float kRec709Beta = 0.018053968510807;

float rec709_oetf(float x) {
    x = max(x, 0.0);
    return x < kRec709Beta ? x * 4.5
                           : kRec709Alpha * pow(x, 0.45) - (kRec709Alpha - 1.0);
}

float rec709_inverse_oetf(float x) {
    x = max(x, 0.0);
    return x < 4.5 * kRec709Beta
               ? x / 4.5
               : pow((x + (kRec709Alpha - 1.0)) / kRec709Alpha, 1.0 / 0.45);
}

// This implementation's IEC 61966-2-1 pair, which is what every recorded
// score uses (zimg derives derivative-continuous constants instead).
float srgb_eotf(float v) {
    return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4);
}

float st2084_eotf(float x) {
    if (x <= 0.0) return 0.0;
    const float xpow = pow(x, 1.0 / 78.84375);
    const float num = max(xpow - 0.8359375, 0.0);
    const float den = max(18.8515625 - 18.6875 * xpow, 1.17549435e-38);
    return pow(num / den, 1.0 / 0.1593017578125);
}

float arib_b67_inverse_oetf(float x) {
    x = max(x, 0.0);
    return x <= 0.5 ? (x * x) * (1.0 / 3.0)
                    : (exp((x - 0.55991073) / 0.17883277) + 0.28466892) / 12.0;
}

// Scalar decode for the transfer ids that act per channel. HLG needs the
// whole triple and is handled in transfer_decode_rgb.
float transfer_decode(uint id, float v) {
    if (id == 0u) return v;
    if (id == 1u) return v < 0.0 ? 0.0 : pow(v, 2.4);
    if (id == 2u) return v < 0.0 ? 0.0 : pow(v, 2.2);
    if (id == 3u) return v < 0.0 ? 0.0 : pow(v, 2.8);
    if (id == 4u) return srgb_eotf(v);
    if (id == 5u) return v <= 0.0 ? 0.01 : pow(10.0, 2.0 * (v - 1.0));
    if (id == 6u) return v <= 0.0 ? 0.00316227766 : pow(10.0, 2.5 * (v - 1.0));
    if (id == 7u) {  // IEC 61966-2-4 (xvYCC)
        return v >= 0.0 && v <= 1.0 ? pow(v, 2.4)
                                    : (v < 0.0 ? -rec709_inverse_oetf(-v)
                                               : rec709_inverse_oetf(v));
    }
    if (id == 8u) {  // BT.1361
        if (v >= 0.0 && v <= 1.0) return pow(v, 2.4);
        return v >= 0.0 ? rec709_inverse_oetf(v)
                        : -rec709_inverse_oetf(-4.0 * v) / 4.0;
    }
    if (id == 9u) return desc_f(kDescToLinearScale) * st2084_eotf(v);
    if (id == 10u) {
        return v < 0.0 ? 0.0 : (52.37 / 48.0) * pow(v, 2.6);
    }
    // id 11 (ARIB B67) needs the luminance OOTF; see transfer_decode_rgb.
    return v;
}

vec3 transfer_decode_rgb(uint id, vec3 c) {
#if VQM_SRC_TRANSFER == 2
    return c;
#elif VQM_SRC_TRANSFER == 1
    return vec3(srgb_eotf(c.x), srgb_eotf(c.y), srgb_eotf(c.z));
#else
    if (id == 11u) {
        // zimg's AribB67InverseOperationC: per-channel inverse OETF, then the
        // luminance OOTF and the 1000/peak factor.
        c = vec3(
            arib_b67_inverse_oetf(c.x), arib_b67_inverse_oetf(c.y),
            arib_b67_inverse_oetf(c.z)
        );
        const float luma =
            max(desc_f(kDescKr) * c.x + desc_f(kDescKg) * c.y +
                    desc_f(kDescKb) * c.z,
                1.17549435e-38);
        return c * (pow(luma, 0.2) * desc_f(kDescToLinearScale));
    }
    return vec3(
        transfer_decode(id, c.x), transfer_decode(id, c.y),
        transfer_decode(id, c.z)
    );
#endif
}

// --- matrix ops ------------------------------------------------------------

vec3 apply_matrix(uint base, vec3 v) {
    return vec3(
        desc_f(base + 0u) * v.x + desc_f(base + 1u) * v.y +
            desc_f(base + 2u) * v.z,
        desc_f(base + 3u) * v.x + desc_f(base + 4u) * v.y +
            desc_f(base + 5u) * v.z,
        desc_f(base + 6u) * v.x + desc_f(base + 7u) * v.y +
            desc_f(base + 8u) * v.z
    );
}

// zimg's CLToRGBOperationC: the constant-luminance op evaluated with the
// BT.709 OETF pair, returning linear RGB. Always scene-referred, so its
// scale is 1.
vec3 cl_to_rgb(vec3 e) {
    const float u = e.y;
    const float v = e.z;
    const float b_minus_y =
        u * 2.0 * (u < 0.0 ? desc_f(kDescClNb) : desc_f(kDescClNb + 1u));
    const float r_minus_y =
        v * 2.0 * (v < 0.0 ? desc_f(kDescClNb + 2u) : desc_f(kDescClNb + 3u));
    const float b = rec709_inverse_oetf(b_minus_y + e.x);
    const float r = rec709_inverse_oetf(r_minus_y + e.x);
    const float luma = rec709_inverse_oetf(e.x);
    const float g =
        (luma - desc_f(kDescKr) * r - desc_f(kDescKb) * b) / desc_f(kDescKg);
    return vec3(r, g, b);
}

// --- chroma upsampling -----------------------------------------------------

#if VQM_SRC_CHROMA

#if VQM_SRC_FILTER == 0
#define VQM_TAPS 2
#elif VQM_SRC_FILTER == 1
#define VQM_TAPS 4
#else
#define VQM_TAPS 6
#endif

float sinc_(float x) {
    return x == 0.0
               ? 1.0
               : sin(3.14159265358979323846 * x) / (3.14159265358979323846 * x);
}

// zimg's resize filter kernels at their defaults: triangle, Catmull-Rom
// (B = 0, C = 0.5) and Lanczos with 3 taps.
float filter_kernel(float x) {
    x = abs(x);
#if VQM_SRC_FILTER == 0
    return max(1.0 - x, 0.0);
#elif VQM_SRC_FILTER == 1
    if (x < 1.0) return 1.0 + x * x * (-2.5 + x * 1.5);
    if (x < 2.0) return 2.0 + x * (-4.0 + x * (2.5 - 0.5 * x));
    return 0.0;
#else
    return x < 3.0 ? sinc_(x) * sinc_(x / 3.0) : 0.0;
#endif
}

// zimg's round_halfup, which keeps round(x - 1) == round(x) - 1.
float round_halfup(float x) {
    return x < 0.0 ? floor(x + 0.5) : floor(x + 0.49999997);
}

// A tap position mirrored once into the plane, then clamped, exactly as
// zimg's resampler resolves out-of-range taps.
int mirror_index(float xpos, uint dim) {
    const float extent = float(dim);
    const float mirrored =
        xpos < 0.0 ? -xpos : (xpos >= extent ? 2.0 * extent - xpos : xpos);
    return int(clamp(mirrored, 0.0, extent - 1.0));
}

#if VQM_SRC_FILTER == 0
// The two bilinear taps of one output position, as window slots (0..2, the
// 3x3 neighbourhood shifted by one) and weights.
void bilinear_phase(
    uint out_pos, float scale, float shift, int centre, out int slot0,
    out int slot1, out float weight0, out float weight1
) {
    const float pos = (float(out_pos) + 0.5) / scale + shift;
    const float base = round_halfup(pos - 1.0) + 0.5;
    const float t = pos - base;
    slot0 = int(floor(base)) - centre + 1;
    slot1 = slot0 + 1;
    weight0 = 1.0 - t;
    weight1 = t;
}

float bilinear_window(
    float window[9], int xa, int xb, int ya, int yb, float xwa, float xwb,
    float ywa, float ywb
) {
    const float w00 = xwa * ywa;
    const float w10 = xwb * ywa;
    const float w01 = xwa * ywb;
    const float w11 = xwb * ywb;
    // No normalisation: for the triangle kernel the two taps of a phase are
    // 1 - t and t, whose sum is exactly 1 in IEEE round-to-nearest for every
    // t in [0, 1], so zimg's division by the tap sum is the identity.
    return w00 * window[ya * 3 + xa] + w10 * window[ya * 3 + xb] +
           w01 * window[yb * 3 + xa] + w11 * window[yb * 3 + xb];
}

// Filtered chroma for all four samples of this invocation's 2x2 block. The
// 3x3 neighbourhood the four bilinear footprints cover is loaded once per
// plane and each axis phase gets one tap pair, so the block costs 9 loads and
// 8 kernel evaluations instead of 16 loads and 32 evaluations.
//
// Window slot i holds the tap at position centre + i + 0.5: the mirror has to
// see the tap's position, not its index, because a tap at -0.5 mirrors to
// +0.5 and so resolves to sample 0 rather than sample 1.
vec4 chroma_block(uint plane, uvec4 px, uvec4 py) {
    const uint dim_x = plane_width(plane);
    const uint dim_y = plane_height(plane);
    const int centre_x = int(px.x >> src_sub_w());
    const int centre_y = int(py.x >> src_sub_h());
    float window[9];
    if (centre_x > 0 && centre_x + 1 < int(dim_x) && centre_y > 0 &&
        centre_y + 1 < int(dim_y)) {
        // Interior block: no tap leaves the plane, so skip the mirror.
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                window[(j + 1) * 3 + (i + 1)] =
                    raw_sample(plane, uint(centre_x + i), uint(centre_y + j));
            }
        }
    } else {
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                window[(j + 1) * 3 + (i + 1)] = raw_sample(
                    plane, uint(mirror_index(float(centre_x + i) + 0.5, dim_x)),
                    uint(mirror_index(float(centre_y + j) + 0.5, dim_y))
                );
            }
        }
    }

    const float scale_x = float(1u << src_sub_w());
    const float scale_y = float(1u << src_sub_h());
    const float shift_x = desc_f(kDescShift);
    const float shift_y = desc_f(kDescShift + 1u);
    int xa0, xa1, xb0, xb1, ya0, ya1, yb0, yb1;
    float xwa0, xwa1, xwb0, xwb1, ywa0, ywa1, ywb0, ywb1;
    bilinear_phase(px.x, scale_x, shift_x, centre_x, xa0, xa1, xwa0, xwa1);
    bilinear_phase(px.y, scale_x, shift_x, centre_x, xb0, xb1, xwb0, xwb1);
    bilinear_phase(py.x, scale_y, shift_y, centre_y, ya0, ya1, ywa0, ywa1);
    bilinear_phase(py.z, scale_y, shift_y, centre_y, yb0, yb1, ywb0, ywb1);

    return vec4(
        bilinear_window(window, xa0, xa1, ya0, ya1, xwa0, xwa1, ywa0, ywa1),
        bilinear_window(window, xb0, xb1, ya0, ya1, xwb0, xwb1, ywa0, ywa1),
        bilinear_window(window, xa0, xa1, yb0, yb1, xwa0, xwa1, ywb0, ywb1),
        bilinear_window(window, xb0, xb1, yb0, yb1, xwb0, xwb1, ywb0, ywb1)
    );
}
#else
// One axis of the resampler: the mirrored index and normalised weight of every
// tap of the output position `out_pos`, in a plane of `dim` samples.
void axis_taps(
    uint out_pos, float scale, float shift, uint dim,
    out float weights[VQM_TAPS], out int indices[VQM_TAPS], out float total
) {
    const float pos = (float(out_pos) + 0.5) / scale + shift;
    const float base = round_halfup(pos - float(VQM_TAPS) / 2.0) + 0.5;
    total = 0.0;
    for (int j = 0; j < VQM_TAPS; ++j) {
        total += filter_kernel(base + float(j) - pos);
    }
    for (int j = 0; j < VQM_TAPS; ++j) {
        weights[j] = filter_kernel(base + float(j) - pos) / total;
        indices[j] = mirror_index(base + float(j), dim);
    }
}

// Separable accumulation of one output position from two tap tables.
float taps_2d(
    uint plane, float wx[VQM_TAPS], int ix[VQM_TAPS], float wy[VQM_TAPS],
    int iy[VQM_TAPS]
) {
    float sum = 0.0;
    for (int j = 0; j < VQM_TAPS; ++j) {
        float row = 0.0;
        for (int k = 0; k < VQM_TAPS; ++k) {
            row += wx[k] * raw_sample(plane, uint(ix[k]), uint(iy[j]));
        }
        sum += wy[j] * row;
    }
    return sum;
}

// Filtered chroma for all four samples of the 2x2 block. Each axis has two
// distinct phases (x and x+1, y and y+1), so each tap table is built once per
// invocation and shared by the two positions that use it.
vec4 chroma_block(uint plane, uvec4 px, uvec4 py) {
    const uint dim_x = plane_width(plane);
    const uint dim_y = plane_height(plane);
    const float scale_x = float(1u << src_sub_w());
    const float scale_y = float(1u << src_sub_h());
    const float shift_x = desc_f(kDescShift);
    const float shift_y = desc_f(kDescShift + 1u);

    float wx0[VQM_TAPS];
    float wx1[VQM_TAPS];
    float wy0[VQM_TAPS];
    float wy1[VQM_TAPS];
    int ix0[VQM_TAPS];
    int ix1[VQM_TAPS];
    int iy0[VQM_TAPS];
    int iy1[VQM_TAPS];
    float tx0, tx1, ty0, ty1;
    axis_taps(px.x, scale_x, shift_x, dim_x, wx0, ix0, tx0);
    axis_taps(px.y, scale_x, shift_x, dim_x, wx1, ix1, tx1);
    axis_taps(py.x, scale_y, shift_y, dim_y, wy0, iy0, ty0);
    axis_taps(py.z, scale_y, shift_y, dim_y, wy1, iy1, ty1);

    return vec4(
        taps_2d(plane, wx0, ix0, wy0, iy0), taps_2d(plane, wx1, ix1, wy0, iy0),
        taps_2d(plane, wx0, ix0, wy1, iy1), taps_2d(plane, wx1, ix1, wy1, iy1)
    );
}
#endif

#endif  // VQM_SRC_CHROMA

// --- the block ------------------------------------------------------------

// The four samples of this invocation's 2x2 block, decoded all the way to
// linear Rec.709 and alpha blended, in (0,0), (1,0), (0,1), (1,1) order.
struct Rgb4 {
    vec4 r;
    vec4 g;
    vec4 b;
};

Rgb4 block_linear_rgb(uint bx, uint by) {
    const uint last_x = source_width() - 1u;
    const uint last_y = source_height() - 1u;
    const uvec4 px = uvec4(
        min(2u * bx, last_x), min(2u * bx + 1u, last_x), min(2u * bx, last_x),
        min(2u * bx + 1u, last_x)
    );
    const uvec4 py = uvec4(
        min(2u * by, last_y), min(2u * by, last_y), min(2u * by + 1u, last_y),
        min(2u * by + 1u, last_y)
    );

    const uint family = src_family();
    const uint mode = src_mode();
    const bool has_alpha = src_alpha() != 0u;
    const bool premultiplied = src_alpha() == 2u;

    // Chroma for the whole block first, so a subsampled plane is filtered out
    // of one loaded neighbourhood rather than per sample.
    vec4 chroma_u = vec4(0.0);
    vec4 chroma_v = vec4(0.0);
#if VQM_SRC_CHROMA
    if (family == 2u) {
        chroma_u = chroma_block(1u, px, py);
        chroma_v = chroma_block(2u, px, py);
    }
#endif

    float r[4];
    float g[4];
    float b[4];
    for (uint i = 0u; i < 4u; ++i) {
        const float alpha = has_alpha ? raw_sample(3u, px[i], py[i]) : 1.0;

        vec3 encoded = vec3(raw_sample(0u, px[i], py[i]));
        // Subsampled YUV must use chroma-plane coordinates. Do not read its
        // planes at full-resolution coordinates before applying the filter.
        if (family != 0u) {
#if VQM_SRC_CHROMA
            if (family == 2u) {
                encoded.y = chroma_u[i];
                encoded.z = chroma_v[i];
            } else {
                encoded.y = raw_sample(1u, px[i], py[i]);
                encoded.z = raw_sample(2u, px[i], py[i]);
            }
#else
            encoded.y = raw_sample(1u, px[i], py[i]);
            encoded.z = raw_sample(2u, px[i], py[i]);
#endif
        }

        // Step 2: the matrix/special op. RGB and GREY carry identity1; CL
        // consumes Y/U/V directly; ICtCp is ICtCp -> LMS, which the transfer
        // then decodes per channel.
        vec3 rgb;
        if (mode == 1u) {
            rgb = cl_to_rgb(encoded);
        } else {
            rgb =
                src_identity1() ? encoded : apply_matrix(kDescMatrix1, encoded);
        }

        // Step 3: alpha, in RGB space -- linear for CL and ICtCp.
        if (premultiplied) {
            const float clamped = clamp(alpha, 0.0, 1.0);
            rgb = clamped == 0.0 ? vec3(0.0) : rgb / clamped;
        }
        if (has_alpha) {
            // Scalar per channel, with the background term hoisted out of the
            // three: the blend happens in encoded RGB space, before the
            // transfer decode.
            const float target = (1.0 - alpha) * pc.background;
            rgb = vec3(
                alpha * rgb.x + target, alpha * rgb.y + target,
                alpha * rgb.z + target
            );
        }

        // Step 4: the transfer function. CL already produced linear values.
        vec3 linear =
            mode == 1u ? rgb : transfer_decode_rgb(src_transfer(), rgb);

        // Step 5: primaries -> linear Rec.709.
        linear = src_identity2() ? linear : apply_matrix(kDescMatrix2, linear);

        r[i] = linear.x;
        g[i] = linear.y;
        b[i] = linear.z;
    }
    return Rgb4(
        vec4(r[0], r[1], r[2], r[3]), vec4(g[0], g[1], g[2], g[3]),
        vec4(b[0], b[1], b[2], b[3])
    );
}
