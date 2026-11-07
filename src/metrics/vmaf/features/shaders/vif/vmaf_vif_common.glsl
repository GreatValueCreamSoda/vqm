// VMAF VIF: Copyright 2016-2020 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Default kernels and mirror rules follow libvmaf's vif_tools.c and
// common/convolution_internal.h at 4611e47a7330cd342498dcc130e1008efbeab6e4.

#include "bindings.glsl"

layout(push_constant, std430) uniform Params {
    uint width;
    uint height;
    uint reference;
    uint distorted;
    uint scratch;
    uint reference_out;
    uint distorted_out;
    uint partials;
    uint results;
    uint scale;
    uint op;
    uint groups;
    float gain_limit;
}
pc;

const float kVif17[17] = float[17](
    0.007456269115, 0.01426550094, 0.02503131889, 0.04028206691, 0.05945261195,
    0.08047509938, 0.09990410507, 0.1137460694, 0.1187738329, 0.1137460694,
    0.09990410507, 0.08047509938, 0.05945261195, 0.04028206691, 0.02503131889,
    0.01426550094, 0.007456269115
);
const float kVif9[9] = float[9](
    0.01897808164, 0.05589817092, 0.1209209114, 0.1921160668, 0.2241735607,
    0.1921160668, 0.1209209114, 0.05589817092, 0.01897808164
);
const float kVif5[5] = float[5](
    0.05448868126, 0.2442013323, 0.4026199281, 0.2442013323, 0.05448868126
);
const float kVif3[3] = float[3](0.166378513, 0.6672429442, 0.166378513);

#ifndef VIF_SCALE
#define VIF_SCALE pc.scale
#endif
int vif_width() { return (1 << (4 - int(VIF_SCALE))) + 1; }
float vif_weight(int tap) {
    if (VIF_SCALE == 0u) return kVif17[tap];
    if (VIF_SCALE == 1u) return kVif9[tap];
    if (VIF_SCALE == 2u) return kVif5[tap];
    return kVif3[tap];
}
int vif_mirror(int coordinate, int extent) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= extent) return 2 * extent - coordinate - 2;
    return coordinate;
}
