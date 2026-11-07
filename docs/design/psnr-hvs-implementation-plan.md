# Plan: add libvmaf-compatible PSNR-HVS

2026-11-06. The requested target is
**ranking agreement with libvmaf's `psnr_hvs`**, using the repository's new
direct agreement testing. Identical returned scores are not required. This
plan uses the current metric-local working tree, including its uncommitted
architecture and CVVDP changes. The checkout's HEAD is
`a949b6177efb88890dd44860cc30dff45486a25d`; HEAD alone does
not describe that implementation baseline.

Implementation was subsequently authorized with test/qualification integration
explicitly deferred. The processing requirement is FP32 for every input storage
type, including the DCT, masking, reduction, normalization and score conversion.
Use an orthonormal FP32 DCT rather than the reference's rounded integer DCT.
The implementation and manual verification record accompanies this plan.

## Recommendation

Add a frame-local Vulkan metric with a standalone `PsnrHvsPipeline`, an optional
entry in `Metrics`, memory queries, CLI selection and benchmark support. Begin
with native coded YUV and a parallel reduction, then qualify direct ranking
agreement before adding RGB convenience conversion or further optimization.

Use the existing libvmaf pin
`4611e47a7330cd342498dcc130e1008efbeab6e4` in `tools/references.lock` as the
oracle. No new external reference checkout is needed. Production must remain
independent of libvmaf; reference tools use copied build trees under `build/`,
and `.reference/` remains read-only.

## Reference contract and provenance

The canonical implementation for this addition is
[Xiph's implementation carried by the pinned libvmaf](https://github.com/Netflix/vmaf/blob/4611e47a7330cd342498dcc130e1008efbeab6e4/libvmaf/src/feature/third_party/xiph/psnr_hvs.c),
locally `.reference/vmaf/libvmaf/src/feature/third_party/xiph/psnr_hvs.c`.
Its definition differs from the authors' original PSNR-HVS. Document the
libvmaf/Xiph variant explicitly; original PSNR-HVS and PSNR-HVS-M are outside
this addition's scope.

| Contract | Reference algorithm and initial implementation baseline |
| --- | --- |
| Transform | Reference uses rounded integer 8x8 DCTs; implementation uses separate orthonormal FP32 transforms for reference and distortion. |
| Sampling | Complete 8x8 windows, stride 7; uncovered trailing pixels are ignored. |
| Perception | Xiph CSF tables, global/quadrant variance, contrast masking and unmasked DC. |
| Inputs | Raw YUV code values, without range expansion or transfer conversion; 8-12 bits. The initial supported subset here is 8/10/12 bits. |
| Plane errors | Coefficient average divided by squared maximum code value. The reference's serial FP32 accumulation and rounding are diagnostic, not final-score acceptance requirements. |
| Outputs | Y, Cb, Cr and combined dB scores; combine normalized errors with weights 0.8/0.1/0.1 before logarithm. |
| Zero error | Positive infinity, including fully masked differences. |

Source anchors: lines 159-201 contain tables, 203-339 the plane calculation,
342-345 dB conversion, and 348-408 validation and extraction. The file carries
a permissive Xiph redistribution notice; preserve it in derived files and add
an installed `licenses/XIPH-PSNR-HVS-LICENSE`. Record source pin and function
anchors in production and oracle provenance. Do not substitute Netflix's
license notice for Xiph's notice.

The [authors' download page](https://ponomarenko.info/psnrhvsm.htm) explains the
original variants. Their MATLAB implementation is not the production porting
source for this compatibility target.

## First milestone: freeze direct ranking qualification

Use the existing [agreement framework](../../validation/agreement.py) and
[validation policy](../validation.md). Final-score acceptance is
direct candidate-versus-reference ranking agreement, including meaningful
within-source ordering. Similar correlations with MOS do not establish this
agreement, and absolute score error is not an acceptance gate.

Build a small reference adapter against the existing pinned libvmaf and report
all four features plus exact input descriptors, compiler and build flags. The
candidate uses the normal `vqm` CLI for file-based still/video qualification;
do not introduce a separate video-scoring executable.

Before candidate scoring, freeze a PSNR-HVS-specific policy and original-only
reports with sample/source/configuration IDs, exact input hashes, reference
revision, required domains and coverage. Reuse the source-balanced selection
method where useful, but generate new converted-YUV hashes, original scores
and ambiguity margins. Existing SSIMULACRA 2/Butteraugli policies qualify only
their own metrics and inputs. Use reference-only pilot data to set meaningful
dB ambiguity margins; do not tune margins or sample membership to the candidate.

Adopt the new suite's limits for the qualified PSNR-HVS domain:

- Spearman SRCC >= 0.999 and Kendall tau-b >= 0.99.
- Source-balanced clearly ordered failures <= 0.5%, distortion-family failures
  <= 1% and worst-source failures <= 5%. Candidate ties on clearly ordered
  reference pairs count as failures.
- Required depth/layout/resolution strata and sufficient distinct source groups,
  samples and eligible within-source comparisons, frozen in the policy. A
  missing or constant required group cannot pass through undefined correlation.

Run the evaluator with ranking explicitly selected:

```sh
python3 -m validation.agreement original.json candidate.json \
  --policy psnr-hvs-policy.json --require ranking --output agreement.json
```

The framework's default `both` also requests numerical acceptance.
Retain maximum error, MAE, RMSE, Pearson correlation, offset and slope as
diagnostics. Do not add absolute-error or dB-error limits to the new PSNR-HVS
final-score gate. Preserve paired old/new reports when replacing an existing
numerical rule for an already-qualified metric/domain.

The agreement schema currently requires finite scores. Freeze a finite-score
ranking corpus using reference-only classification, and retain identity,
fully masked and other zero-error cases as separate hard invariant tests.
Record those cases and their coverage explicitly; never remove a sample
because the candidate fails or returns infinity. A candidate nonfinite score
on a required finite-score sample fails qualification.

When test integration resumes, keep independent component tests under
`tests/metrics/psnr_hvs/`: FP32 DCT fixtures, numerical variance/masking
oracles, sampling/normalization tests and zero-error anchors. These verify the
metric's mathematical components and remain separate from the final-score
ranking gate. Keep scalar oracles independent of production helpers/shaders.

The reference's serial FP32 coefficient sum can differ from a parallel GPU
tree. Use the parallel tree as the default; characterize any score drift and
test its effect on ranking. Reproducing the serial fold is not a prerequisite.
Preserve the reference's mathematical pooling definition while allowing a
different floating-point evaluation order. Document any changed score scale
or public thresholds rather than claiming original dB calibration from rank
agreement alone.

## Input and result API

Propose `include/psnr_hvs.h` with `PsnrHvsConfig`, `PsnrHvsScore` and
`PsnrHvsPipeline`; expose it through `include/vqm.h` and install the leaf header.
The score returns combined `score` plus named `y`, `cb` and `cr` values in dB.
Retain normalized plane errors internally for validation and diagnostics.
Scores are higher-is-better, are not limited to 100, and may be negative or
positive infinity. NaN and negative infinity are failures.

Initial input support:

- Opaque planar YUV420, YUV422 and YUV444, using bytes at 8 bits or words at
  10/12 bits. Both inputs have matching dimensions, bit depth, subsampling
  and coded-color interpretation. Qualify all three layouts deliberately,
  retaining the reference tables even where their names mention 420.
- Every scored plane is at least 8x8: minimum image sizes are 16x16 for 420,
  16x8 for 422 and 8x8 for 444. Keep the existing divisibility requirement for
  subsampled dimensions. Larger dimensions need not be multiples of 8 or 7.
- Reject RGB, grey, alpha, floating-point samples, unsupported depths and
  mismatched native formats explicitly. Raw range/transfer metadata does not
  trigger a conversion. Require valid sample values within the declared depth.

Support the established image/view, uploaded image, allocated input-buffer and
borrowed Vulkan-buffer paths. Follow the existing move/ownership, one-in-flight,
`Submit`/`Finish`, stale-handle, queue and semaphore contracts through
`VulkanRuntime` and `ResidentTable`. Config includes geometry, source storage
options and `MemoryPolicy`; omit unrelated chroma filtering or half-plane knobs.

RGB/PNG/JPEG support is a separate explicit adapter milestone. Define matrix,
transfer, range, output depth, chroma sampling/filter, clipping, quantization
and alpha policy, then feed the exact same converted integer planes to both
implementations. This makes conversion qualification reproducible. VMAF's
preprocessing has different normalization and 16-bit conversion contracts;
do not make this metric depend on those classes or silently reuse their output.
Native YUV support can ship independently, with its limits documented.

## GPU implementation and memory

Own implementation, layout and shaders under `src/metrics/psnr_hvs/`. A small
set of cohesive files can cover the facade/pipeline, layout, dispatch and
constants; keep shader files separate. Reuse metric-independent Vulkan
machinery without changing the arithmetic of other metrics.

1. Prepare each source separately through binding 3 into scratch planes
   containing unchanged code values represented exactly as FP32. Metadata
   stays in binding 2. This handles distinct external buffers without adding
   another source binding or reading pixels through the metadata binding.
2. Run a block pass that loads corresponding windows from both prepared
   images, calculates variances and both FP32 DCTs, applies masking/CSF and
   emits compact partials. Use separable transforms and workgroup-local
   storage; begin with a portable workgroup shape and no subgroup dependency.
3. Reduce compact partials with a deterministic parallel tree. Keep full
   coefficient images out of the default path. Select accumulation/normalization
   precision for numerical stability and measured performance, qualifying
   final-score behavior through the frozen ranking policy.
4. Read back compact errors and perform the mixture/logarithm in FP32 on the host.
   Keep shader completion barriers, scratch lifetimes and score conversion
   together with the dispatch sequence.

Implement `PlanPsnrHvsMemory` returning `BufferMemoryPlan`, shared by
construction and memory queries. Include raw source slots/pools, both prepared
images, partials, descriptors, readback and staging. Enforce checked block/count
arithmetic, offsets, storage-buffer limits and dispatch limits. There is no
temporal history or private temporal allocation. Query both direct and staged
paths and include this component in worker budgets and actual usage reports.

## Integration

| Surface | Planned changes |
| --- | --- |
| Combined evaluation | Add optional config/result/memory fields in `include/vqm_metrics.h`; update selection, construction, evaluation and budget aggregation in `src/metrics/metrics.cc`. Keep scores aligned with their frame when VMAF delays output. Preserve wait-on-first/signal-on-last behavior. |
| Memory query | Add overloads in `include/vqm_memory.h` and `src/vulkan/vulkan_memory_query.cc`, using the metric-owned planner. |
| CLI | Add `--metric psnr-hvs` to the existing still/video CLI routing, help, named channel diagnostics and explicit input/option validation. Reuse progressive Y4M, JSON, pairs-file, iteration and warmup support. |
| Benchmark | Add `bench/driver/psnr_hvs.cc`; extend metric parsing/routing and worker handling in the driver. Add prepare, block/error, reduce and host-score timing in `src/timing.{h,cc}`. |
| Score transport | Update `bench/run_bench.py`'s finite-only checks and relevant test/report parsers. Define JSON infinity encoding explicitly; keep NaN invalid. A finite plane or aggregate score must not be required when error is zero. |
| Checksums | Preserve score-checksum meaning; add an error diagnostic checksum if needed for zero-error runs. Use finite-score pairs for timing qualification. |
| References and agreement | Reuse the vmaf lock entry/build; add a PSNR-HVS reference adapter, canonical reports, frozen ranking policy and relevant manifests without changing the pin. Use `--require ranking` / `--gate-mode ranking` as supported by the adapter. |
| Build/package | Register host, test and benchmark sources and metric-local shaders in `CMakeLists.txt`; install header/license and cover standalone leaf-header consumption. |
| Documentation | Add `docs/psnr-hvs.md`; update metric/architecture/reference/input documentation and preserve numerical/benchmark evidence beside this plan. |

Initially give PSNR-HVS an independent allocation in `Metrics`.
`SharedMetricWorkspace` remains responsible for SSIMULACRA 2 and Butteraugli.
A shared-arena extension, reference-feature cache and public worker-pool class
can follow separate memory/performance evidence. No temporal lookahead or clip
finalization API is needed; frames are independent.

## Verification and delivery gates

The numerical matrix includes identity, flat/DC differences, chroma-only
differences, texture/noise, masking thresholds, different reference/distortion
variances, all depths/layouts and extreme valid codes. Exercise plane dimensions
8, 14, 15, 16, 21 and 22 to expose stride coverage. Test undersized rejection,
uncovered-border differences and fully masked differences. Positive infinity
must be accepted for distinct inputs when measured error is zero.

Compare component arithmetic against the independent oracle and require
equivalent results across every supported input path. Include padded views,
distinct external source buffers, resident handle
ownership, input pools, in-flight failures, both memory policies and semaphore
ordering. Test combined results with VMAF lookahead and existing spatial metrics,
CLI/report parsing, package consumers and memory estimates versus allocations.

Use the repository's fresh release build, `ctest --preset release`,
`./build/vqm_test --require-gpu`, sanitizer checks and
`python3 tools/check_quality.py --lint` for implementation qualification.
Run Ruff and YAPF on changed Python files. Existing metrics retain their
identity/golden/oracle checks; a CPU-only pass does not establish GPU correctness.

Extend corpus validation to score identical decoded/converted pairs with ours
and pinned libvmaf. Require the frozen direct ranking/ordering gates across
qualified configurations and strata; retain per-channel numerical differences
as diagnostics. CID22 can supply a source-balanced still-image corpus once its
YUV conversion and input hashes are frozen; correlation to CID22 MOS alone
does not replace direct agreement with libvmaf. High-bit-depth, alternate
layout and resolution domains need corresponding qualification coverage and
component oracles. For file-based video workloads use complete clips through
the normal `vqm` CLI and explicitly define frame-to-clip pooling for both arms.
Keep nonfinite-score invariants separate from the finite agreement reports.
The initial CLI summary used libvmaf's arithmetic mean of frame dB scores,
where one infinite frame makes the mean infinite. When evaluating that
reference mean, freeze reference-finite video variants before candidate
scoring, including a base distortion on every frame where needed.
The [current CLI summary](../psnr-hvs.md#combined-metrics-and-video) uses the
mean frame error before converting to dB.

Benchmark native 420/444 at 1080p and 4K, separately for resident, streaming and
borrowed Vulkan inputs, with all input/CPU/GPU/build provenance. Follow the
interleaved protocol for later optimization or existing hot-path changes and
record stage medians/spreads, score checks, allocations and fixed-count wall
time. Compare the reference on identical coded samples. Claim a speedup only
after the applicable measured gate.

Delivery order: frozen reference reports/policy and component oracles;
standalone GPU and memory planning; input/resource coverage; combined API and
tools; component, ranking and performance qualification. Add convenience
conversion after the native contract is established.

## Review evidence and deferred approaches

Method: inspect the existing working-tree APIs, source decoder, runtime,
planners, CLI, benchmark/reporting and pinned reference; independently review
integration and numerical parity. Revise acceptance against the new agreement
framework, frozen source-balanced policies and the user's ranking-only
clarification. No reference execution, build, GPU probe,
test or benchmark was performed. There are no measured accuracy, speed or
memory-allocation results from this planning review, and no hardware-dependent
conclusions. Input provenance for subsequent results must include exact coded
planes and conversion commands, rather than filenames alone.

Excluded approaches: porting the original MATLAB metric under the same name;
replacing the two masking inputs with a DCT of the difference; applying
the standard linear-RGB decoder; capping identity at 60 or 100; requiring serial
FP32 final-score reproduction; treating similar MOS correlations as direct
reference agreement; silently padding undersized/edge windows; broadening
the shared workspace during the initial addition. Each changes the selected
contract or expands scope before ranking qualification is established.

Conclusion: existing resource abstractions support this metric cleanly. Use a
parallel GPU implementation, qualify direct ranking agreement with pinned
libvmaf on a frozen domain, and retain numerical component/invariant tests.
Final-score differences are diagnostic and do not require serial accumulation.
