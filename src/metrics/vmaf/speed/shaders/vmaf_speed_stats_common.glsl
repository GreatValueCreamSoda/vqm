// VMAF SpEED: Copyright 2016-2025 Netflix, Inc.
// BSD-2-Clause-Patent; see licenses/NETFLIX-VMAF-LICENSE.
// Mean, covariance, singular handling, and entropy follow est_params and
// get_speed_score in Netflix/vmaf feature/speed.c at
// 4611e47a7330cd342498dcc130e1008efbeab6e4.
#include "bindings.glsl"
layout(push_constant, std430) uniform Params {
    uint width;
    uint height;
    uint reference;
    uint distorted;
    uint scratch;
    uint partials;
    uint result;
    uint groups;
    uint op;
}
pc;
const uint kMeans = 0u;
const uint kCov = 50u;
const uint kCholesky = 1300u;
const uint kEigen = 2550u;
const uint kSingular = 2600u;
// U and V banks have independent means/covariance/eigensystem/status regions.
uint bank(uint ch) { return pc.scratch + (ch / 2u) * 2604u; }
uint means(uint ch, uint i) { return bank(ch) + kMeans + 25u * (ch % 2u) + i; }
uint cov(uint ch, uint i, uint j) {
    return bank(ch) + kCov + 625u * (ch % 2u) + 25u * i + j;
}
uint cholesky(uint ch, uint i, uint j) {
    return bank(ch) + kCholesky + 625u * (ch % 2u) + 25u * i + j;
}
uint eigen(uint ch, uint i) { return bank(ch) + kEigen + 25u * (ch % 2u) + i; }
uint singular(uint ch) { return bank(ch) + kSingular + ch % 2u; }
uint factor_status(uint ch) { return bank(ch) + 2602u + ch % 2u; }
float status(uint ch) {
    float spectral = arena[singular(ch)];
    return spectral == 0.0 ? arena[factor_status(ch)] : spectral;
}
uint source(uint ch) {
    return (ch % 2u == 0u ? pc.reference : pc.distorted) +
           (ch / 2u) * pc.width * pc.height;
}
