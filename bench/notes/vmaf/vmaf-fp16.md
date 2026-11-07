# VMAF FP16 practicality

Investigated on 2026-11-05 using an RTX 5070, driver 615.78.08, GCC 16.2.1
and the source after the [speed investigation](vmaf-speed-investigation.md).
Both pipelines' spatial features are predominantly FP32; model prediction is
host double precision. V0 already stores its native 8–12-bit VIF source tile
in FP16 without rounding those centered samples.

## Conclusion

A pure FP16 replacement cannot preserve the current metric contracts. FP32 is
required before accumulation in several places: rounding products, stored
inputs or thresholds to FP16 already discards information that a wider sum
cannot recover. Eleven isolated storage/arithmetic/control prototypes were
built and measured. None of the lossy scalar FP16 arithmetic candidates passed the
accuracy and 2% targeted-stage speed gates.

One exact storage optimization was retained for v1: CAMBI's shared source
symbols use FP16 when the logical device has enabled `shaderFloat16`. Symbols
are integers 0–497, so their values are unchanged. Histogram counters and
atomics stay integer, contrast/pooling stay FP32, and global arena layout stays
unchanged. The existing shader remains the portable fallback. No approximate
precision mode or public API option was introduced.

On the primary 1080p/4K inputs, the integrated change reduced CAMBI stage time
about 9.0%/7.7% and resident frame time 3.5%/3.8%. The ducks frame was neutral;
dense synthetic inputs benefited more. All 4,292 CID22 scores match the
captured baseline exactly at the benchmark's eight printed decimals.

## Why FP32 accumulation alone is insufficient

| component | constraint | practical precision |
| --- | --- | --- |
| Native input preparation | Centered 8/10/12-bit coded luma fits FP16 exactly. Native 16-bit and canonical SDR retain steps of 1/256, which FP16 can lose. Centering must precede any half cast. | Integer/FP32 conversion; source-specific exact half storage only. |
| VIF | Variance is `E[x²] - E[x]²`, with a branch at 2. Rounded squared/cross products can create or erase variance before the FP32 sum. Epsilon `1e-10` becomes zero in half. | FP32 moment/product-result precision, cancellation, statistics, logarithms and reductions. |
| ADM2/ADM3 | Wavelet product errors feed detail cancellation and enhancement decisions. Decoupling uses `1e-30`; weighted per-pixel cubes can exceed 65,504 before reduction. | FP32 wavelet/decoupling/norm path for the current accuracy contract. |
| Motion | Rounding source/history or blur products can erase small transitions. Full-frame totals also exceed half range. | FP32 arithmetic, history and reductions; exact native source storage is possible but did not pay here. |
| SpEED | Filter rounding can introduce a nonzero highpass signal in a flat plane. Covariance rounding affects nearly singular matrices; the solver uses `1e-6` decisions and `1e-37` pivot guards. | FP32 filtering, covariance, factorization/eigensolver, variance solve and pooling. |
| CAMBI | Exact histogram state and rank/tie semantics matter. Even a 33×33 contrast numerator can overflow half before division. | Integer histograms/counts and FP32 contrast/pooling. Bounded shared symbols can use FP16 exactly. |
| Predictor | The reference uses double precision. Its measured cost is a small fraction of a frame. | Keep the host double predictor. |

For a concrete valid VIF input, native 10-bit codes 914/913 become centered
values 100.5/100.25. The true constant-window variance is zero. Half rounds
`100.5² = 10100.25` to 10104, creating variance 3.75 and crossing the branch at
2 even with FP32 accumulation. The GPU prototype changed this flat-window VIF
from approximately 1 to 0. A workgroup-wide recentering can fix uniform tiles
but fails for an allowed heterogeneous tile whose center is inappropriate for
distant flat windows. Per-window central-moment reconstruction or rescaling
would be a different, more expensive algorithm requiring new error analysis.

For native 16-bit motion, `127 → 127 + 1/256` collapses to the same half value.
The input-storage-only prototype passed the old 201-test suite, yet targeted
moving sequences exposed score drift of 0.01634. A new regression test now
protects those low-order transitions.

Pure half range is also insufficient independently of accumulation:
`64³ = 262144`, beyond binary16's maximum finite 65,504. Valid 10-bit stripe
and checkerboard inputs in the CPU range diagnostic produce ADM weighted
cubes above 500,000. In CAMBI, `4 * 544 * 545` overflows half although the
subsequent normalized contrast is finite. Algebraic reordering could avoid
some overflow, but would not restore lost precision or establish reference
compatibility.

## FP16 inputs with a wide multiply-accumulate are a different option

The scalar mixed-product prototypes explicitly round `OpFMul` to FP16 before
promoting the result into an FP32 sum. This is different from FP16 **operands**
feeding a wide-product/FP32 matrix accumulator. NVIDIA documents that Tensor
Cores can preserve the full product of FP16 inputs while accumulating in FP32;
see its [GPU performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html).
The VIF product-rounding counterexample does not disprove that wider operation
when its inputs are already exactly representable.

A read-only query on this GPU advertises cooperative-matrix FP16 A/B with
FP32 C/result shapes 16×16×16, 16×8×16 and 16×8×8 at subgroup scope. The
library does not enable `VK_KHR_cooperative_matrix`, and this study does not
implement or benchmark a matrix-unit backend. No Tensor Core speed or numerical
equivalence is claimed from the scalar experiments.

Such a backend is a plausible next arithmetic investigation, particularly for
SpEED's covariance Gram matrix, with its means, factorization, singular tests
and output kept FP32. However, its filtered/de-meaned inputs are continuous
FP32 values, so rounding those operands to half can still change conditioning.
VIF's weighted moments also do not map directly to a single unweighted matrix
multiply. Padding the 25-component matrix, packing/transfers, negotiated device
features, precision guards/fallback and the relatively small covariance share
of total frame time would all need measurement. This remains distinct from a
drop-in scalar precision switch or a pure FP16 pipeline.

The subsequent [cooperative-matrix investigation](vmaf-cooperative-matrix.md)
now executes the wider operation and measures isolated SpEED covariance
backends. Its results and limitations are separate from the scalar study below.

## Experimental method

The entire current uncommitted source state was captured before the study and
built independently. Each prototype was applied only to a separate mutable
source/build directory, with fresh file timestamps when restoring earlier
sources. Arithmetic candidates use explicit Float16 operations; SPIR-V
disassembly distinguishes real half multiplication/addition from storage-only
conversions. A paired FP32 ADM control avoids treating vector pairing itself
as an FP16 benefit. No experimental arithmetic remains enabled in production.

Each arm ran the frozen 201-test GPU suite and a diagnostic of 48 six-frame
sequences: 288 unique emitted frames, native 8/10/12/16-bit YUV444, low-bit
motion, flat high luma, near-black, texture/chroma and extreme transitions.
Raw features and scores were compared, including replay through the unchanged
double predictor with score clipping disabled. These are targeted falsification
cases at 320×192, not a general error bound. Every arm completed all frames;
final aggregation checks duplicate, missing and extra keys.

Five alternating A/B benchmark runs per content/size used the repository's
20-iteration/3-warmup resident protocol. Stages come from GPU timestamps; timing
claims do not come from the diagnostic program. Inputs include 10-bit 1080p
and 4K and the 8-bit ducks frame. Both versions use their standard 1080p viewing
model. Full-preparation resident calls repeat pixels and become zero-motion
after warmup; they do not establish temporal precision or complete-video
throughput. Native, RGB and dense streaming/resident arms separately checked
the retained change.

## Storage, platform and memory scope

FP16 arithmetic and 16-bit buffer interfaces have separate Vulkan requirements.
`shaderFloat16` permits the local/workgroup types used here; true 16-bit SSBO
members additionally need the appropriate storage feature. Physical support
alone does not prove a borrowed logical device enabled a feature. The retained
constructor respects the existing enabled-feature declaration and tests the
promise-false fallback. See the primary [Khronos shader-feature guide](https://docs.vulkan.org/guide/latest/extensions/shader_features.html).

This GPU reports Float16, storage16, Float16 denormal-preserve and RTE support.
Those queried capabilities are distinct from requesting float-control execution
modes. CPU binary16 illustrations do not prescribe every GPU contraction or
denormal behavior. Nor does scalar half arithmetic automatically use tensor
cores; a cooperative-matrix implementation would be a separate device-feature,
tiling and numerical project. The [Khronos arithmetic sample](https://docs.vulkan.org/samples/latest/samples/performance/16bit_arithmetic/README.html)
also separates arithmetic throughput, buffer storage and packed operations.

The retained shared tile changes its nominal language-layout footprint from
16,004 to 12,164 bytes, a 3,840-byte reduction. Driver allocation padding and
workgroup residency were not measured. This does **not** halve the arena,
resident image slots or upload traffic. Stage timings establish the benefit
on this GPU, not its stall cause or performance on AMD/Intel hardware.

There is a larger exact-storage opportunity beyond this change: CAMBI's
prepared/decimated/mode-filtered codes are integers 0–1024, and its derivative
and mask banks hold booleans. Packed `uint16` is a better representation for
those global banks than approximate half arithmetic. For the native 420
benchmark's two-input layouts (resident 2 with streaming off, or streaming with
no resident slots), packing the eight eligible banks would analytically
save 17,625,600 bytes at 1080p and 70,502,400 bytes at 4K, about 10.3% of the
whole arena before allocation rounding. This is not implemented or a speed
claim: it needs new layouts, odd-edge/write ownership, fallback and memory-query
checks. Phase-lifetime reuse could save memory independently of precision and
also requires explicit alias contracts. Contrast maps, histogram/state counts
and pooling totals must remain wider.

Raw artifacts, compiler/module evidence, source/binary hashes, patches,
sequences and per-run tables are under `artifacts/reports/vmaf-fp16/`.

## Retained-code validation

- Release and ASAN/UBSAN GPU suites: 204 tests, zero failed checks.
- Hardware and Mesa software suites with Vulkan synchronization validation:
  204 tests each, no validation errors.
- Release and sanitizer CTest: 12/12 each, including package consumers and
  benchmark integration.
- Recursive formatting: 199 maintained files; lint: 41 production C++ files.
- All 114 active SPIR-V modules independently recompiled byte-identically and
  passed `spirv-val`. The original portable CAMBI module is byte-identical to
  the pre-study module; the retained half module matches the measured prototype.
- Current integrated v1: all 4,292 corpus scores exactly match the captured
  baseline. Public headers, model tables and global arena layouts are unchanged.

New guards cover high-luma low-variance VIF through fused/packed/fallback
paths, native 12/16-bit low-order temporal changes, and exact CAMBI maps through
automatic, forced-portable and enabled-feature-promise-false paths. Existing
independent oracles and tolerances were preserved.

## Measured precision results

Failures below are intentional experimental regressions against the frozen 201-test baseline, not failures in the retained production code. The targeted score column uses the unchanged predictor with clipping disabled; this avoids hiding errors behind scores saturated at 0 or 100. Values are observed maxima, not error bounds.

| isolated arm | failed checks | targeted sequence score drift |
| --- | ---: | ---: |
| motion-input-shared | 0 | 0.01634262 |
| motion-all-shared | 20 | 0.01634262 |
| motion-half-blur | 21 | 0.17299663 |
| motion-half-products | 21 | 0.13033754 |
| speed-half-filter | 91 | 0.84774237 |
| speed-half-covariance | 43 | 0.00060696 |
| vif-half-products | 165 | 65.72712601 |
| adm-half-products | 108 | 4.11610335 |
| adm-paired-fp32 | 0 | 0.00000000 |
| adm-paired-half | 108 | 4.11610335 |
| cambi-half-shared | 0 | 0.00000000 |

Every arm emitted all 288 distinct frames with no errors or missing/extra keys. In the SpEED arms, the default-clipped sequence scores hid errors: filtering changed the raw SpEED feature by up to 0.91038 even though those synthetic final scores clipped to the same value.

| full 4,292-pair still-image comparison | maximum score drift | mean score drift | exact pairs |
| --- | ---: | ---: | ---: |
| cambi-half-shared | 0.00000000 | 0.00000000 | 4292 |
| speed-half-filter | 0.14661516 | 0.00656294 | 0 |
| speed-half-covariance | 0.00981773 | 0.00041308 | 0 |
| vif-half-products | 8.18732925 | 3.36348920 | 0 |
| adm-half-products-v0 | 0.20437981 | 0.01186121 | 4 |
| adm-half-products-v1 | 0.07037640 | 0.00936419 | 2 |
| motion-half-blur-v0 | 0.00000000 | 0.00000000 | 4292 |
| motion-half-blur-v1 | 0.00000000 | 0.00000000 | 4292 |

The half-blur motion arm passes every still-image score exactly while failing moving-frame tests. A still-image corpus or a repeated-frame timing run alone is therefore insufficient to qualify temporal precision. The CAMBI arm is exact in the corpus, targeted moving sequences and direct contrast-map comparisons.


## Arithmetic performance

The following 4K GPU-stage changes compare each prototype against the original baseline in its own interleaved sitting. Negative is faster; the percentages are not summed or subtracted across experiments. The paired FP32 control and paired half arm isolate the representation from the idea of pairing reference/distortion work. No lossy arm reached a useful, accuracy-qualified stage gain.

| arm | v0 targeted stage change | v1 targeted stage change |
| --- | ---: | ---: |
| motion-input-shared | +0.09% | +0.05% |
| motion-all-shared | +0.30% | +0.31% |
| motion-half-blur | +0.03% | +0.10% |
| motion-half-products | +0.77% | +0.71% |
| speed-half-filter | — | +0.07% |
| speed-half-covariance | — | +1.08% |
| vif-half-products | +7.51% | — |
| adm-half-products | -0.47% | +0.51% |
| adm-paired-fp32 | -0.30% | +2.61% |
| adm-paired-half | -0.55% | +2.76% |
| cambi-half-shared | — | -7.68% |

## Retained production comparison

These compare the integrated feature-gated change against the independently built pre-study source. Each value is the median of five per-run frame medians in milliseconds, with the five-run minimum–maximum range. Stage medians and all commands/score/checksum values are in the corresponding `production-*/results.json`. Resident uses 20 timed calls/3 warmups; streaming uses 8/2.

| workload | before ms [range] | after ms [range] | frame time change |
| --- | ---: | ---: | ---: |
| vmaf-v0-1080 native-resident | 0.376500 [0.376269–0.377264] | 0.376694 [0.375825–0.389789] | +0.05% |
| vmaf-v0-4k native-resident | 1.597255 [1.596491–1.598976] | 1.598680 [1.598140–1.601856] | +0.09% |
| vmaf-v0-ducks native-resident | 0.384104 [0.383289–0.385504] | 0.384224 [0.383194–0.385219] | +0.03% |
| vmaf-v1-1080 native-resident | 0.684770 [0.682289–0.691704] | 0.660925 [0.656160–0.662735] | -3.48% |
| vmaf-v1-4k native-resident | 2.614984 [2.614455–2.735088] | 2.514802 [2.506826–2.520446] | -3.83% |
| vmaf-v1-ducks native-resident | 0.558551 [0.557112–0.560366] | 0.558036 [0.556201–0.558626] | -0.09% |
| vmaf-v0-1080 native-streaming | 0.752134 [0.740989–0.754118] | 0.749644 [0.742189–0.754039] | -0.33% |
| vmaf-v0-4k native-streaming | 3.075968 [3.072278–3.082217] | 3.078198 [3.074968–3.083738] | +0.07% |
| vmaf-v0-ducks native-streaming | 0.568862 [0.568206–0.571381] | 0.570446 [0.568461–0.580711] | +0.28% |
| vmaf-v1-1080 native-streaming | 1.240611 [1.236961–1.248055] | 1.207302 [1.205891–1.219406] | -2.68% |
| vmaf-v1-4k native-streaming | 4.803107 [4.792416–4.820626] | 4.690463 [4.689728–4.704093] | -2.35% |
| vmaf-v1-ducks native-streaming | 0.846392 [0.845477–0.848532] | 0.838727 [0.835152–0.847612] | -0.91% |
| vmaf-v1-rgb1080 rgb-resident | 0.705425 [0.704239–0.725519] | 0.677289 [0.676745–0.678005] | -3.99% |
| vmaf-v1-rgb4k rgb-resident | 2.631690 [2.617405–2.652305] | 2.534961 [2.529351–2.556736] | -3.68% |
| vmaf-v1-rgb1080 rgb-streaming | 1.256196 [1.255135–1.258356] | 1.229216 [1.227506–1.237781] | -2.15% |
| vmaf-v1-rgb4k rgb-streaming | 4.829491 [4.823416–4.840426] | 4.736812 [4.724963–4.759937] | -1.92% |
| vmaf-v1-dense1080 dense-resident | 0.778393 [0.776228–0.784493] | 0.719549 [0.718899–0.722624] | -7.56% |
| vmaf-v1-dense4k dense-resident | 3.169606 [3.166611–3.174722] | 2.977054 [2.976269–2.978974] | -6.07% |

All 180 production comparison runs have exact corresponding reported scores/checksums and stable per-arm results. V0 has no new computation path; its observed medians stay within 0.4% of baseline with overlapping ranges. The ducks contrast workload is mostly unaffected; the gain is content-dependent rather than a blanket frame-speed multiplier.
