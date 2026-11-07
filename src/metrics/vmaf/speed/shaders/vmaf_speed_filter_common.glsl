// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Filtering follows vif_tools.c and feature/speed.c at
// Netflix/vmaf 4611e47a7330cd342498dcc130e1008efbeab6e4.
#include "bindings.glsl"

layout(push_constant, std430) uniform Params {
    uint width;
    uint height;
    uint op;
    uint input_offset[4];
    uint scratch_offset[4];
    uint down_offset[4];
    uint result_offset[4];
}
pc;

// `speed_get_antialias_filter` uses a 9-tap Gaussian with sigma 3.6.
const float kAntialias[9] = float[9](
    0.07564882934, 0.09910339117, 0.1201889366, 0.1349366903, 0.1402443051,
    0.1349366903, 0.1201889366, 0.09910339117, 0.07564882934
);
const float kMean[3] = float[3](0.166378513, 0.6672429442, 0.166378513);

int speed_mirror(int x, int extent) {
    return x < 0 ? -x : (x >= extent ? 2 * extent - x - 2 : x);
}
