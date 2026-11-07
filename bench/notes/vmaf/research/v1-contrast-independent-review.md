# Independent review of compact CAMBI contrast storage

Reviewed on 2026-11-05, independently of the implementation review in
[contrast headroom](v1-contrast-headroom.md). Scope: the experimental
`contrast_variants.py --reuse-tail` and `--radius8` arms, separately and
combined, with the normal 16×8 contrast tile. No production files were edited
and no GPU work or timing was run for this review. No actionable correctness
or portability defect was found in these changes.

Reviewed SHA-256 fingerprints:

```text
contrast_variants.py           32c1d193ba1ca97c1097d8e1796c38a0840710502cfded9680a6e9c9e933a6f9
vmaf_cambi_contrast.comp       32dd3e5fd379da21ee59b021cf01a79edbc9914077831bba2e1027453f3d1933
vmaf_cambi.cc                  4e17efdffa5ccfbc70b55e11f99859435a9386d1046a0234bc005c4593b1b297
vmaf_cambi.h                   fc9fa00bd2c8ff32710d31f71df2b1f409fc2a467ceebac7bf94f4c13cc14d58
```

## Tail ownership and synchronization

The accepted sample range is 67–563 inclusive. `packed_sample` consequently
returns zero or 1–497; `update_bin` subtracts one and addresses packed histogram
words 0–248. `bin_count` independently rejects requested bins outside 0–496.
Words 249–252 are therefore available for the four per-row contrast terms,
and word 255 of the last row is available for the shared active-row mask.
Neither tail region aliases histogram counts or the other tail region.

The flag's lifetime is safe in both sparse and dense paths:

1. Invocation zero initializes the flag. The existing barrier precedes all
   center probes and their `atomicOr` operations.
2. The probe barrier precedes every invocation's cached `active_mask` read.
   The empty-group branch is uniform and writes zero to every owned valid
   output coordinate before returning.
3. Histogram initialization deliberately skips the flag word. It cannot race
   another invocation's cached read, even when different subgroups advance at
   different rates.
4. Every invocation reaches the halo barrier after caching the mask. Dense
   construction starts after this barrier and may eventually overwrite the
   last row's full 256-word histogram, including the dead flag. All later
   decisions use private cached masks. Sparse construction leaves the unused
   flag word alone; it is reset by the next workgroup's initial probe phase.

Only four flag references remain in the transformed source: its alias,
initialization, atomic probe update, and cached read. The four contrast terms
are published after histogram cloning has finished. The existing per-column
barrier orders their writes before lane zero reads them; the end-of-column
barrier prevents the next column's term writers from overtaking that read.
The final column exits uniformly, after its term publication/read. Packed
histogram updates continue to address only words 0–248. No new subgroup-width
assumption or cross-workgroup synchronization is introduced.

Float terms pass through `floatBitsToUint` and `uintBitsToFloat` without
changing their expressions or the order of the four-way maximum. Eligible
sample symbols remain exactly representable in FP16, and histogram counts
remain integers. The largest 33×33 window has 1,089 samples, safely below a
16-bit counter's capacity; tail reuse does not alter that existing bound.

## Radius selection, fallback and edge bounds

`VmafCambiContrast::Pixels` validates a nonzero odd window no larger than 33,
dimensions no larger than 1920×1080, and dimensions at least as large as the
window. The experimental host selects the compact pipeline only when
`window <= 17`; this proves `radius = window / 2 <= 8`. Larger accepted
windows use the original radius-16 capacity. Mode-filter phases do not read
these shared arrays and keep their existing 16×16 dispatch geometry.

The compact source array contains `(16 + 2R) * (8 + 2R)` elements for capacity
radius `R`. At runtime radius `r <= R`, center and halo writes occupy at most
`(16 + 2r) * (8 + 2r)` elements. The last entering horizontal sample is at
column `15 + 2r`; the last clone entering row is `7 + 2r`. Both are within
the same extent. Image-edge samples are checked before global reads and are
stored as zero. Output stores check both dimensions. This also covers radius
zero, partial edge tiles and completely inactive rows.

`half_source_` still requires both caller permission and an enabled logical
device `shaderFloat16` feature. With `allow_half=false`, or a borrowed device
that does not advertise enabled Float16, both capacity choices select integer
modules. Integer modules retain 16×8 geometry. No additional shader capability
is emitted for their tail reuse. Pipeline creation is transactional through
`CreatePipelines`, and all handles are destroyed by the owning contrast object.

## Independent checks

The four transformed modules were compiled from stdin using glslc 2026.4 with
`-O --target-env=vulkan1.1 -fshader-stage=compute`, then passed directly through
`spirv-val --target-env vulkan1.1`. Their disassembled Workgroup variable sizes
were independently totaled:

| Tail reuse capacity | Integer bytes | Half bytes |
| --- | ---: | ---: |
| Radius 8 | 11,264 | 9,728 |
| Radius 16 | 15,872 | 12,032 |

All four fit Vulkan's 16 KiB minimum shared-memory capacity. Integer modules
do not declare `OpCapability Float16`; half modules do. Existing local size
and invocation-count requirements are unchanged. An exhaustive integer address
check across all radii 0–16 verified 49,950 center/window/slide/clone indices.

Existing GPU test source covers windows 1, 17 and 33, odd dimensions, sentinel
overwrite for zero maps, sparse/dense masks, exact integer/half comparison and
the borrowed-device fallback. GPU execution remains the measuring agent's
responsibility. Explicit windows 15 and 19 would add coverage immediately on
each side of the compact-pipeline boundary; this is a coverage improvement,
not evidence of a defect. Compilation and index checks do not establish GPU
memory behavior or a performance benefit.
