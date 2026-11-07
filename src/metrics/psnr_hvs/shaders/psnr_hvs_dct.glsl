// Copyright 2001-2012 Xiph.Org and contributors.
// BSD-2-Clause; see licenses/XIPH-PSNR-HVS-LICENSE.
// Read-only FP32 metadata is published once by the host. Keeping these
// indexed tables in the arena avoids per-invocation private constant arrays.
// The matching values and provenance live in metrics/psnr_hvs/tables.h.

float Csf(uint index) { return uintBitsToFloat(desc[pc.tables + index]); }

float DctBasis(uint index) {
    return uintBitsToFloat(desc[pc.tables + 192u + index]);
}
