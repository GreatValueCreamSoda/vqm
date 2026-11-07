// Copyright 2016-2020 Netflix, Inc. BSD-2-Clause-Patent.
// See licenses/NETFLIX-VMAF-LICENSE.
layout(std430, binding = 0) buffer Data { uint data[]; };
layout(push_constant) uniform Params {
    uint values;
    uint histogram;
    uint state;
    uint partials;
    uint result;
    uint count;
    uint selected;
    uint groups;
    uint pass;
    uint counts[5];
    uint selected_counts[5];
}
pc;

// A zero scalar count selects the five-scale layout. In the rectangular
// grid, y selects the scale and x traverses its values/partials. The packed
// grid maps x across all scales' consecutive group ranges instead.
struct PoolArgs {
    uint values, histogram, state, partials, result;
    uint count, selected, groups, pass, group;
};
PoolArgs pool_args() {
    PoolArgs a;
    a.values = pc.values;
    a.histogram = pc.histogram;
    a.state = pc.state;
    a.partials = pc.partials;
    a.result = pc.result;
    a.count = pc.count;
    a.selected = pc.selected;
    a.groups = pc.groups;
    a.pass = pc.pass & 255u;
    a.group = gl_WorkGroupID.x;
    if (pc.count == 0u) {
        uint scale = gl_WorkGroupID.y;
        if ((pc.pass & 256u) != 0u) {
            scale = 0u;
            while (scale < 4u) {
                uint n = pc.counts[scale];
                uint groups = n / 256u + uint(n % 256u != 0u);
                if (a.group < groups) break;
                a.group -= groups;
                ++scale;
            }
        }
        a.count = pc.counts[scale];
        a.selected = pc.selected_counts[scale];
        a.groups = a.count / 256u + uint(a.count % 256u != 0u);
        a.histogram += 256u * scale;
        a.state += 2u * scale;
        a.result += scale;
        for (uint i = 0u; i < scale; ++i) {
            a.values += pc.counts[i];
            a.partials += pc.counts[i] / 256u + uint(pc.counts[i] % 256u != 0u);
        }
    }
    return a;
}
