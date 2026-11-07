// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// SSIMULACRA 2's own shader interface: its push-constant block, the plane
// region as f32 or float16_t, the plane accessors built on it, and the level
// table the scale and reduce passes read. Butteraugli has the symmetric file,
// butter_common.glsl; the bindings and the input decode (color.glsl) are
// shared with it. The C++ mirror of the push block is PushParams in
// src/metrics/ssimulacra2/vulkan.cc (static_assert'ed), and the level table's
// C++ side is BuildLevelTable there.

#include "bindings.glsl"

layout(push_constant, std430) uniform Params {
    // Word offset of the level table in binding 2; see level_u below.
    uint level_base;
    // Levels this geometry runs, 1 .. 6. The pyramid's size is fixed by
    // src/metrics/ssimulacra2/pipeline.cc, so this is a constant of the image
    // size, not of a call.
    uint scale_count;
    // prepare: word offset of this image's descriptor in binding 2.
    uint desc_base;
    // prepare: which side of the pair this dispatch converts, 0 or 1. It
    // selects the image's plane offsets out of the level table.
    uint image;
    float background;  // prepare: alpha blend target
    // scale: one level's planes, geometry and norm block, so the tile loop's
    // addresses never wait on a level-table load.
    uint plane[2];  // image 0, image 1 plane offsets
    uint stride;    // plane channel stride: the level's area
    uint width;     // the level's size
    uint height;
    uint partials;  // this level's norm partials
    uint groups;    // scale workgroups of this level
    uint groups_x;  // of those, the count along x
    // scale: this level's three channel bits, bit c = X, Y, B, so a channel
    // whose maps are pruned can return unread. prepare and reduce: 18 bits,
    // bit level * 3 + channel, so one word covers the whole pyramid. See
    // Ssimulacra2ChannelPruned in src/metrics/ssimulacra2/pipeline.cc.
    uint channel_mask;
    float weights[16];  // Gaussian taps, 13 are used
}
pc;

// The plane region is f32 by default and float16_t for the half-plane build
// (CMakeLists.txt compiles a second SPIR-V per plane shader with
// SS2_HALF_PLANES). Only loads, stores and conversions touch the 16-bit type:
// no arithmetic runs at half precision, so the only cost of the mode is the
// rounding of the plane values themselves.
#ifdef SS2_HALF_PLANES
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define SS2_PLANE float16_t
#else
#define SS2_PLANE float
#endif
layout(std430, binding = 1) buffer Planes { SS2_PLANE planes[]; };

// The level table: one row of eight words per level, addressed through
// binding 2 so no shader needs the arena's float offsets.
//
//   0, 1  plane offset of image 0, 1 (plane elements, channel 0)
//   2     this level's norm partials (arena floats)
//   3     this level's norms (arena floats)
//   4, 5  width and height in pixels
//   6     scale workgroups of this level
//   7     reserved: the row is padded to eight words so the shift is exact
//
// A level's plane set is three channel planes of width*height, so its channel
// stride is width*height and its pitch is its width.
const uint kLevelShift = 3u;

uint level_u(uint level, uint field) {
    return desc[pc.level_base + (level << kLevelShift) + field];
}

uint level_stride(uint level) { return level_u(level, 4u) * level_u(level, 5u); }

// The source image's own geometry, which is the base level's.
uint source_width() { return level_u(0u, 4u); }
uint source_height() { return level_u(0u, 5u); }

// Every plane access goes through these two, so the shaders above stay
// precision agnostic. plane_round is the same conversion without a store: the
// linear sample the pyramid hands from one level to the next used to reach the
// next level through the plane region, so the half-plane build rounded it and
// has to keep doing so to stay bit-identical.
#ifdef SS2_HALF_PLANES
float plane_load(uint index) { return float(planes[index]); }
void plane_store(uint index, float value) { planes[index] = float16_t(value); }
float plane_round(float value) { return float(float16_t(value)); }
#else
float plane_load(uint index) { return planes[index]; }
void plane_store(uint index, float value) { planes[index] = value; }
float plane_round(float value) { return value; }
#endif

// Channel c of the plane set at `offset`; `stride` is the plane set's capacity
// in pixels (its channel stride) and `pitch` is the width of the plane set
// being indexed -- the current level for the plane set in use, the previous
// level for the samples a downsample reads.
uint plane_index_pitch(
    uint offset, uint stride, uint c, uint x, uint y, uint pitch
) {
    return offset + c * stride + y * pitch + x;
}
