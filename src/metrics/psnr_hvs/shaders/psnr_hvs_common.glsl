// Copyright 2001-2012 Xiph.Org and contributors.
// BSD-2-Clause; see licenses/XIPH-PSNR-HVS-LICENSE.

#include "bindings.glsl"

layout(push_constant) uniform Push {
    uint desc_base;
    uint plane;
    uint width;
    uint height;
    uint reference;
    uint distorted;
    uint destination;
    uint blocks_x;
    uint blocks;
    uint tables;
}
pc;
