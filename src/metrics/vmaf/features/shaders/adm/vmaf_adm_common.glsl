// Copyright 2016-2020 Netflix, Inc. BSD-2-Clause-Patent.
// See licenses/NETFLIX-VMAF-LICENSE.
#include "bindings.glsl"
layout(push_constant) uniform Params {
    uint width;
    uint height;
    uint reference;
    uint first_scale;
    uint restored;
    uint additive;
    uint stride;
    uint partials;
    uint results;
    uint scale;
    uint groups;
    uint original_pixels;
    float axial;
    float diagonal;
}
pc;
