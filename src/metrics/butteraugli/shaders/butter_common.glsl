// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.

// Shared interface of the Butteraugli compute shaders: binding 0 is the arena
// as floats, binding 1 its plane region, binding 2 the arena as words, and
// binding 3 the input source. The push block fits the 128-byte minimum.
//
// Butteraugli has its own push block. It carries the two fields color.glsl reads
// (`desc_base`, `background`) and the two geometry helpers it calls, so the
// decode chain is reused verbatim by butter_opsin.comp.
//
// Every Butteraugli plane is one channel of one level, so a dispatch that
// covers several channels of one chain (scale.comp's z trick) walks the chain
// by `area` floats per channel. The region table lives in binding 2, exactly
// like SSIMULACRA 2's level table, and holds absolute arena float offsets.

layout(push_constant, std430) uniform Params {
    uint level_base;   // word offset of the region table (16 words per level)
    uint level;        // this dispatch's level: 0 base, 1 the 2x subsample
    uint op;           // shader-specific selector
    uint taps;         // blur tap count
    uint kernel;       // arena float offset of the normalized taps
    uint image;        // 0 = reference chain, 1 = distorted chain
    uint z_count;      // channels (or images) this dispatch covers
    uint desc_base;    // color.glsl: this image's descriptor, in binding 2
    uint a, b, c, d;   // arena float plane offsets the dispatch names
    float background;  // color.glsl: alpha blend target
    float p0, p1, p2, p3;  // band prefactors and scalars
    float q0, q1, q2, q3;  // asymmetry, xmul, intensity target, fold weight
}
pc;

#include "bindings.glsl"

// Region table fields, one 16-word row per level.
const uint R_CHAIN0 = 0u;     // reference chain base
const uint R_CHAIN1 = 1u;     // distorted chain base
const uint R_SCRATCH_A = 2u;  // xyb, then mask0, mask1, diffs
const uint R_SCRATCH_B = 3u;  // mf before the HF blur, then blur0/1, eroded
const uint R_AC = 4u;         // three accumulated channel planes
const uint R_DIFFMAP = 5u;
const uint R_SUB0 = 6u;  // subsampled rgb the next level reads
const uint R_SUB1 = 7u;
const uint R_WIDTH = 8u;
const uint R_HEIGHT = 9u;
const uint R_AREA = 10u;  // width * height: the channel stride of a chain
const uint R_SUBW = 11u;
const uint R_SUBH = 12u;
const uint R_PARTIALS = 13u;  // reduce output, level 0's row only

const uint kRowShift = 4u;

uint butter_row(uint level, uint field) {
    return desc[pc.level_base + (level << kRowShift) + field];
}

uint butter_here(uint field) { return butter_row(pc.level, field); }
uint butter_area() { return butter_here(R_AREA); }
uint butter_width() { return butter_here(R_WIDTH); }
uint butter_height() { return butter_here(R_HEIGHT); }
uint butter_chain(uint which) {
    return butter_here(which == 0u ? R_CHAIN0 : R_CHAIN1);
}

// color.glsl reads the source geometry through these two; the base level's row
// carries it, so they need no push-constant field of their own.
uint source_width() { return butter_row(0u, R_WIDTH); }
uint source_height() { return butter_row(0u, R_HEIGHT); }

// Chain layout, contiguous: lf 3 planes, mf 3, hf 2 (the UHF input), hff 2
// (the post-UHF high frequency), uhf 2.
uint chain_lf(uint chain) { return chain; }
uint chain_mf(uint chain) { return chain + 3u * butter_area(); }
uint chain_hf(uint chain) { return chain + 6u * butter_area(); }
uint chain_hff(uint chain) { return chain + 8u * butter_area(); }
uint chain_uhf(uint chain) { return chain + 10u * butter_area(); }

// Scratch A: one plane each, reused across phases (xyb -> mask0, mask1).
uint scratch_a(uint index) {
    return butter_here(R_SCRATCH_A) + index * butter_area();
}

// Scratch B: one plane each (mf before the HF blur -> blur0, blur1, eroded).
uint scratch_b(uint index) {
    return butter_here(R_SCRATCH_B) + index * butter_area();
}

// The two range helpers of the separation, butteraugli.cc:309-326.
float remove_range(float x, float w) {
    if (x > w) return x - w;
    if (x < -w) return x + w;
    return 0.0;
}

float amplify_range(float x, float w) {
    if (x > w) return x + w;
    if (x < -w) return x - w;
    return 2.0 * x;
}

// MaximumClamp, butteraugli.cc:296-306: values past the clamp are compressed
// by kMul instead of clipped.
float maximum_clamp(float v, float max_value) {
    const float kMul = 0.724216145665;
    if (v >= max_value) return max_value + (v - max_value) * kMul;
    if (v <= -max_value) return -max_value + (v + max_value) * kMul;
    return v;
}

// XybLowFreqToVals, butteraugli.cc:327-342, applied to one pixel's lf triple.
// The b channel folds the original y in, which is why the transform needs all
// three planes at once.
vec3 xyb_lf_to_vals(vec3 lf) {
    const float kXmul = 33.832837186260;
    const float kYmul = 14.458268100570;
    const float kBmul = 49.87984651440;
    const float kYToB = -0.362267051518;
    const float b = kYToB * lf.y + lf.z;
    return vec3(lf.x * kXmul, lf.y * kYmul, b * kBmul);
}
