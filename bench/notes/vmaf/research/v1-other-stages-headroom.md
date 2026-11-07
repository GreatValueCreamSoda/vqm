# Remaining VMAF v1 luma, ADM3 and motion headroom

Read-only source/history review on 2026-11-05; no GPU execution or production
edits. The active investigation reported approximately 59 µs native conversion,
105 µs ADM3 and 70 µs motion at 1080p. Those are parent investigation observations,
not independently reproduced measurements in this review. Together they account
for 234 µs; removing a hypothetical 10% from all three would save only 23.4 µs.
Actual gains require the repository's interleaved correctness/performance gates.

## Existing work already removes several obvious costs

Current ownership is
[`vmaf_preprocess.cc`](../../../../src/metrics/vmaf/features/vmaf_preprocess.cc),
[`vmaf_adm.cc`](../../../../src/metrics/vmaf/features/vmaf_adm.cc) and their
[metric-owned shaders](../../../../src/metrics/vmaf/features/shaders/).
The [VMAF speed investigation](../vmaf-speed-investigation.md) already adopted
32×8 motion tiles, skipped wholly excluded ADM groups, and overlapped each ADM
reduction with the next independent wavelet. The native converter already
processes four 8-bit or two 16-bit samples per invocation. These opportunities
are implemented and should not be proposed as new work.

History rules out several tempting repetitions:

| Idea | Existing evidence |
| --- | --- |
| Pack all native Y/U/V conversion into two dispatches | Exact outputs, but conversion regressed 7.2% at 1080p and 1.1% at 4K in [the speed investigation](../vmaf-speed-investigation.md). The subtraction-only mapping alternative also missed its gate. |
| Subgroup motion reduction | Preserved the add tree and passed checks, but missed 2% and slightly regressed at 4K in the same report. |
| Preweight ADM absolute magnitudes | Exact checks, but ADM timings stayed within 0.2% in the same report. |
| FP16 motion source tile | [FP16 results](../vmaf-fp16.md) measured +0.05% for v1 despite smaller shared storage. Rounding blur/history/products also introduces temporal errors and did not pay. |
| Paired FP32 ADM source arithmetic | The same FP16 study's precision control regressed v1 ADM 2.61%. FP16 arithmetic additionally moved scores. |
| Stage ADM additive neighborhoods, fuse decouple/masking or use half intermediate planes | [The v0 optimization record](../vmaf-v0-optimization.md#wider-vif-windows-packed-luma-and-tiled-adm) says each was neutral or slower. A new proposal must explain a concrete difference. |
| Reorder motion before spatial features | The same record measured a small 1080p improvement and a 4K regression and reverted it. |

## Native conversion: one modest exact load-reuse control remains

The current converter records six independent plane dispatches for native v1
YUV: both Y planes, both U planes and both V planes. It applies exactly
`float(code) * 2^(8-depth) - 128`. Binding 3 is selected for each source, and
metadata stays in binding 2. These bindings and external-buffer behavior are
load-bearing; combining sources requires more than changing a dispatch count.

Inside each invocation, the existing shader loops over `pc.step` and loads
`source_words[base + (index >> shift)]` separately for every output sample.
Aligned rows therefore address the same packed source word two/four times.
A focused control could explicitly load that word once and extract its
components, while retaining the exact conversion expression and original
per-row/per-plane output addresses. This differs from the rejected packed
multi-plane grid: it changes neither launch count nor plane mapping.

It needs a correct slow path for odd sample strides, partial right-edge packs
and unaligned row starts. Plane `base` is in source words, but `y*stride` can
still put a row's first sample inside a word. The common aligned, complete pack
can use one load; the existing loop remains the fallback. Compile-time byte/
word variants could also make the loop bound and shifts explicit, at the cost
of additional pipelines. Input depths 8/10/12/16, independent source formats,
odd row strides and borrowed buffers must remain covered.

This reduces issued load/address work, not necessarily physical memory
traffic: repeated packed-word reads can already hit cache, and the driver may
optimize some cases. At 1080p YUV420, both images together contain 6,220,800
coded samples. With 16-bit inputs and FP32 output, mandatory source/output
payload is about 37.3 MB before caches; the six launches and destination stores
remain. This is a small measurement candidate, not evidence of a large win.

Fusing native conversion into ADM would not simply remove the luma bank:
CAMBI and motion also consume that bank. V1's fused DWT needs both sources at
once, while the current input contract exposes one selected source binding.
That is a broader source-interface project with extra synchronization and
provenance work, not an easy launch deletion.

## Motion: dead shared storage can be reused, but the grid is already large

The current 32×8 blur/difference shader allocates 432 source floats, 288 vertical
intermediates and 256 reduction floats: 3,904 bytes. At 1920×1080 it launches
60×135 = 8,100 groups, followed by one 256-invocation reducer. This stage has
ample grid-level SM parallelism; its large-grid blur is not structurally
limited to a handful of SMs like SpEED's small-matrix solver.

One exact resource-lifetime control is to reuse the first 256 source-tile
locations for the group reduction. After the vertical-results barrier, all
source-tile reads are complete. Horizontal filtering reads only `vertical`,
so each invocation can write its difference into `tile[lane]`, then keep the
same barrier, reduction indices, addition order and final partial store.
Eliminating the separate `partial[256]` allocation changes shared capacity
3,904→2,880 bytes without changing arithmetic, history banks or dispatches.
The earlier half-input tile saved 864 bytes with no gain; the existing shared
block is already small. This new alias saves 1,024 bytes but may also be neutral.
The vertical array cannot simply alias the source array while it is being
produced: other invocations still read the source tile during that phase.

The main blur group executes two producer/consumer barriers plus the partial
publication barrier and eight halving-tree barriers. The final reducer uses a
publication barrier and eight halving-tree barriers. Removing those through
subgroups was already tested unsuccessfully; a changed sum grouping must also
pass moving-frame and low-bit checks. A repeated-frame benchmark's zero motion
cannot qualify numerical behavior on transitions.

The first frame offers a separate latency-only shortcut: its difference is
known to be zero, so the group/final sum trees need not discover that zero.
The blurred history still must be produced for the next frame. Implementing an
explicit first-frame zero store could skip reduction work safely, but it does
not improve warmed continuous-video timings. Skipping the blur or deciding
that two caller-provided buffers contain identical pixels from their handles
would violate history/mutability contracts.

## ADM3: source tiling is a control; larger changes face prior failures

Current v1 ADM3 dispatches fused DWT/decouple, accumulation and reduction for
each of four scales, then a one-invocation finish: 13 dispatches. The DWT uses
precise FP32 ordered four-tap vertical/horizontal sums. The accumulator computes
both additive and restored 3×3 masks plus nine FP32 cubic norm sums; its
9×256-float reduction block is 9,216 bytes. Fully excluded groups already write
positive-zero partials and return. Per-scale reduction can overlap the next
wavelet because the buffers are disjoint; adding broad barriers would undo
that measured improvement.

V0 already enables a 34×34 shared source tile per image for DWT/decouple.
V1 ADM3 deliberately retains the untiled variant in current CMake/constructor
selection. The historical note records that distinction, but this review did
not find an isolated numerical/timing explanation for an ADM3 tiling rejection.
Using the existing tiled source with enhancement gain one is therefore a
cheap control if it was not already tested in the active measurement session.
It preserves the precise transform operations, adds 9,248 shared bytes and a
load barrier, and reduces redundant source loads; extra shared resources and
synchronization can still lose. It should be tried as an isolated control,
without assuming the v0 result transfers to v1.

A new accumulator halo should not be sold as an untried easy win: staging ADM
additive neighborhoods was explicitly rejected previously. ADM3 additionally
needs restored neighborhoods for AIM, increasing a full six-plane 18×18 tile
to 7,776 bytes before the existing 9,216-byte reduction arrays. Partial aliasing
could avoid simultaneous lifetimes, but requires a concrete new schedule and
careful synchronization; it is a larger investigation with an unfavorable
historical control. Likewise, batching scale reducers needs retained partials
for every scale and must outperform the already allowed overlap, rather than
merely reducing command count. VIF's analogous batched reduction was rejected.

## Practical conclusion

There is no source argument that these stages have reached hardware limits,
but the obvious precision, packing, neighborhood and subgroup substitutions
have already been measured. The remaining small exact controls are explicit
packed-word reuse in native conversion, dead motion-source-tile reuse, and the
existing ADM tiled-source variant as an isolated v1 control. None is claimed
faster here. Larger gains would need a new dataflow or workload design, while
preserving native low-bit samples, temporal history and FP32 reductions.
