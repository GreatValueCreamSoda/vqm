// ColorVideoVDP: Copyright (c) 2024, Graphics and Displays group.
// MIT; see licenses/COLORVIDEOVDP-LICENSE.
// Base model arithmetic follows the pinned original v0.5.7, commit 2a268bc.

#include "bindings.glsl"

layout(push_constant, std430) uniform Params {
    uint width;
    uint height;
    uint next_width;
    uint next_height;
    uint reference;
    uint distorted;
    uint destination;
    uint auxiliary;
    uint scratch;
    uint tables;
    uint partials;
    uint results;
    uint channels;
    uint bands;
    uint band;
    uint op;
    uint input_stride;
    uint destination_stride;
    uint groups;
    uint parity;
}
pc;

float model(uint index) { return arena[pc.tables + index]; }

float safe_pow(float value, float exponent) {
    // The original tensor operations cancel exactly at zero. Preserve that
    // identity when a driver folds only the constant epsilon power.
    if (value <= 0.0) return 0.0;
    return max(pow(value + 1e-5, exponent) - pow(1e-5, exponent), 0.0);
}

int reflect(int coordinate, int extent) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= extent) return 2 * extent - coordinate - 2;
    return coordinate;
}
