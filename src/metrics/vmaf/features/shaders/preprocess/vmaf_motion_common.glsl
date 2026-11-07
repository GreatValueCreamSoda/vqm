// VMAF motion: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// The constants and mirror borders follow Netflix libvmaf float_motion.c,
// motion_tools.h and common/convolution_internal.h at
// 4611e47a7330cd342498dcc130e1008efbeab6e4.

#include "bindings.glsl"

layout(push_constant, std430) uniform Params {
    uint width;
    uint height;
    uint luma;
    uint previous;
    uint blurred;
    uint scratch;
    uint partials;
    uint output_score;
    uint has_previous;
    uint groups;
}
pc;

const float kMotionFilter[5] =
    float[5](0.054488685, 0.244201342, 0.402619947, 0.244201342, 0.054488685);

int motion_mirror(int coordinate, int extent) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= extent) return 2 * extent - coordinate - 2;
    return coordinate;
}
