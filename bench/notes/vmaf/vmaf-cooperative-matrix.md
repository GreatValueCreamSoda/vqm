# VMAF FP16-input / FP32 cooperative-matrix investigation

Investigated on 2026-11-05, RTX 5070, NVIDIA driver 615.78.08, GCC 16.2.1.
**The wider operation works, but the tested backend is not worth adopting as a
production default.** Plain FP16 covariance operands fail existing numerical
checks. High/residual reconstruction greatly reduces drift, but the best tuned
4K arm saves about 3 microseconds in SpEED, leaves whole-frame time effectively
unchanged, regresses 1080p, and increases scratch memory enough to fail an
existing layout budget check. Production shaders, layouts, runtime and public
API remain unchanged by this investigation.

This continues the [scalar FP16 study](vmaf-fp16.md), whose half-rounded product
failures did not establish the behavior of wider matrix multiplication.
[Compact measured results](../measurements/vmaf-cooperative-results.json) retain
all 170 benchmark runs over 17 workloads, every score/checksum/stage, input and
binary hashes, test failures and sequence comparisons. Supporting studies are
[the Vulkan contract](research/cooperative-platform.md),
[SpEED numerical analysis](research/cooperative-speed-numerics.md), and
[other VMAF feature mappings](research/cooperative-feature-mapping.md).

## What was established on the GPU

A private device enabled `VK_KHR_cooperative_matrix`, FP16 shader arithmetic,
the Vulkan memory model and full compute subgroups. All three advertised
FP16 A/B, FP32 accumulator/result shapes ran successfully: 16×16×16, 16×8×16
and 16×8×8, at subgroup scope with 32 lanes. All 18 platform correctness cases
passed with Vulkan and synchronization validation enabled. No half storage
buffer feature was needed: operands were converted from FP32 storage into
shared FP16 arrays.

For example, sixteen `100.5²` terms produced exactly `161604`, rather than
`161664` after rounding the individual products to half. Products beyond half
range remained finite, and small cancellation residuals were retained.
Discarded operand bits remained discarded: an input step of `1/256` near 127
collapsed before multiplication. The
[platform snapshot](../measurements/vmaf-cooperative-platform.json) preserves
every case.

The [Vulkan SPIR-V contract](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)
leaves matrix evaluation order and internal precision implementation-dependent.
An FP32 accumulator/result type is not a portable promise of sequential IEEE
FP32 accumulation. The wider-product observations above apply to this GPU and
driver. The pipeline also explicitly requires full subgroups; launch size alone
does not satisfy the current cooperative-matrix requirements for SPIR-V 1.3.

## Integrated SpEED prototype

SpEED is the strongest direct matrix mapping in these implementations. After
filtering, each reference/distorted U/V plane produces an N×25 matrix of
mean-subtracted overlapping 5×5 patches, with covariance `XᵀX/N`. Means,
filtering, eigensolver, singular handling, Cholesky solve, pooling and predictor
remain FP32 or their existing wider host types. The experiment changes only
covariance construction and its temporary storage.

Each matrix is padded to 32 components and K to a multiple of 16, with explicit
zeros. The original crop and true sample count are retained. Workgroups gather
centered operands, perform 16×16×16 cooperative operations over a partition of
K, and write partial matrices. A separate pass merges partials and mirrors the
upper triangle before the unchanged solver. Both U/V channels and both images
have disjoint scratch regions. The three-tile variant omits unused lower tile
2. The two-MMA variant needs that tile for its transposed cross term.

For the residual arms, `H = half(X)`, `L = half((X−H)*2048)`. These reconstruct

`G ≈ [HᵀH + (HᵀL + LᵀH)/2048] / N`.

Three MMAs compute the three terms separately. The two-MMA variant computes
`HᵀH` and `HᵀL`, then transposes the latter during merge; this doubles its
partial streams. A four-MMA form also includes `LᵀL/2048²`. All four forms
compile and pass SPIR-V validation, with exactly 1/2/3/4 cooperative multiply-add
instructions respectively. The four-MMA form was compiler-checked, not GPU
benchmarked. The compact numerical study's two/three-Gram labels count its
symmetry-based Gram expressions, not the GPU three/four-MMA instruction count.

The constant residual scale helps preserve small residuals. It is not an
all-exponent FP32 representation or an accuracy guarantee. Summation order also
changes. CPU conditioning examples still cross the solver's strict `1e-6`
classification boundary even after residual reconstruction. The direct-matrix
stress examples illustrate conditioning; they are not claimed to be realizable
filtered image patches. An existing QR fallback sees the already perturbed
covariance and cannot recover information lost there. A robust backend would
need a justified guard and FP32 covariance recomputation.

## Interleaved measurements

The complete current source state was frozen and built independently for the
baseline. Its source hash manifest and source archive were preserved; `HEAD`
alone would not describe that already-modified working tree. Candidate edits
were applied only to a separate source copy. The production checkout was never
used as an experimental target.

Every arm used five interleaved A,B rounds in one sitting: resident 2, 20 calls,
3 warmups; the upload arm separately used 8 calls and 2 warmups. All timers are
from `vqm_bench`; no standalone probe has timers. Values below are changes of
per-stage medians, negative meaning faster. Arms have their own baselines and
must not be compared by subtracting timings across sittings.

| Resident arm | SpEED 1080p | SpEED 4K | SpEED ducks 1080p | Full frame 4K |
| --- | ---: | ---: | ---: | ---: |
| FP16 operands, 8 partitions, 4 tiles | +17.47% | +47.19% | +17.95% | +2.91% |
| Two MMAs + transpose, 32 partitions | +4.05% | +6.04% | +4.07% | +0.21% |
| Three MMAs, 128 partitions, 3 tiles | +4.25% | -2.42% | +4.62% | -0.08% |
| Three MMAs, 32 partitions, 3 tiles | +2.36% | +2.61% | +2.20% | -0.15% |
| Three MMAs, 8 partitions, 4 tiles | +20.31% | +54.94% | Not measured | +2.31% |

The best 4K resident comparison had these five-run spreads:

| Best 4K arm | SpEED ms, median [min–max] | Frame ms, median [min–max] |
| --- | ---: | ---: |
| FP32 baseline | 0.131243 [0.130994–0.131398] | 2.515797 [2.489983–2.520453] |
| Residual matrix prototype | 0.128070 [0.127886–0.128181] | 2.513813 [2.508213–2.515837] |

The same 128-partition arm's streaming SpEED stage was 2.33% faster at 4K, but
full-frame time changed by +0.05%. At 1080p, resident SpEED was 4.25% slower and
full-frame time +0.80%; streaming SpEED was 3.86% slower and frame time +1.04%.
Ducks 1080p also regressed. The 4K stage passes the nominal 2% median speed
threshold; the backend does not pass the complete adoption gates.

All 170 runs had finite scores, checksums and stage tables, and stable
score/checksum pairs within each arm. Corresponding candidate scores differ
from baseline. For the best resident arm, observed per-call score drift was
0.00000802 at 1080p, 0.00000050 at 4K and 0.00000220 on ducks. These are three
measured pairs, not a corpus accuracy qualification.

This path pays for operand gathering/conversion, padded matrix work, split-K
partials, merging and another dispatch. The measurements include those costs;
they do not isolate a particular hardware stall or establish a fundamental
matrix-unit limit. Prepacked operands, better data reuse or phase-lifetime
scratch reuse would be separate changes requiring their own measurements.

## Correctness and memory gates

| Arm | Required-GPU suite | Validation | Sequence evidence |
| --- | --- | --- | --- |
| Frozen baseline | 204 tests, zero failed checks | Production behavior | 288 distinct frames |
| Plain half, K split 8 | Two failed checks in existing v1 reference/temporal tests | No Vulkan/sync errors | SpEED drift up to 0.00045811 in 288 frames; clipped scores concealed this drift |
| Three MMAs, K split 8 | 204 tests, zero failed checks | No Vulkan/sync errors | SpEED drift up to 0.00000898 in 288 frames |
| Two MMAs, K split 32 | One failed layout-budget check | No Vulkan/sync errors | SpEED drift up to 0.00000606 in 288 frames |
| Three MMAs, K split 32/128, three tiles | One failed layout-budget check each | No Vulkan/sync errors | 288 unclipped frames each; maximum score drift 0.00000838, SpEED drift 0.00000898 |

The plain-half failure is concrete: an existing SpEED feature changed to
`3.84819173813` where the oracle expects `3.930904 ± 0.08`. Original tolerances
were preserved. The larger residual arms' failure is instead
`TestVmafV1MemoryFitsWithoutUnusedDistortedDetailBands`: its 512×384 layout no
longer fits the pinned 14,000,000-byte buffer budget. The two-channel prototype
adds 128 KiB for split 8, 512 KiB for three-MMA split 32, 1 MiB for two-MMA split
32, and 2 MiB for three-MMA split 128. Production memory planning remains
unchanged.

The unclipped sequence probe covers both implementations at 8/10/12/16 bits.
ADM, motion, CAMBI and VIF features remained exact in those comparisons; v0 was
not routed through the new SpEED path. The newly available `--unclipped` probe
option prevents final score saturation from hiding arithmetic errors.

The final maintained release build and all 12 CTest groups passed. Recursive
formatting checked 204 maintained files. Peer reviews found no current shader
indexing, synchronization, merge, protocol or documentation discrepancy.

No candidate was adopted, so this study did not run a new full CID22 corpus or
qualify ASAN/software-Vulkan portability for the matrix variants. Passing the
small numerical and sequence samples would not establish complete metric
parity. A production backend would also need feature/shape negotiation,
portable and borrowed-device fallbacks, dispatch-limit validation, revised
memory contracts and all applicable corpus/correctness/performance gates.

## Both VMAF implementations

V0 has no corresponding dense SpEED Gram stage. The source/CPU review found
that native centered 8/10/12-bit codes fit FP16 exactly, but VIF coefficients,
weighted moments and filtered intermediates generally do not. Specific
Toeplitz and weighted-Gram formulations still cross VIF's variance branch.
High/residual DWT examples reduce ADM error substantially, without qualifying
an end-to-end GPU implementation. Short filters also have packing and padding
costs. CAMBI is histogram/rank work, and the host predictor is too small a share
to motivate a matrix offload from this evidence. See the
[feature mapping review](research/cooperative-feature-mapping.md) for the
examples, useful-operation fractions and historical stage ceilings. No V0 GPU
matrix replacement was timed in this study.

## Reproduction and retained knowledge

Use [`bench/experiments/cooperative_matrix/`](../../experiments/cooperative_matrix/README.md)
for the private-device platform probe, deterministic CPU examples, isolated
variant installer, shaders and interleaved benchmark driver. The installer
requires a fresh isolated source copy and explicitly refuses the maintained
checkout. The integrated prototype assumes this local device's negotiated
capabilities; it is not a portable production backend.

The current shader source reproduces both embedded SPIR-V modules byte for
byte in every measured candidate binary, for its recorded flags. Module
checks, source/binary hashes, raw stages, test logs, synthetic sequences and
compiler evidence are under ignored `artifacts/reports/vmaf-cooperative-matrix/`.
The frozen baseline source archive is under `artifacts/snapshots/`. Shared
conclusions, reproducible tools and compact evidence remain in maintained
repository paths; the build and temporary source copies are disposable.
