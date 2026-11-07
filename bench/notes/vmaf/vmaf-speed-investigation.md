# VMAF v0 and v1 speed investigation

Measured on 2026-11-05 with an RTX 5070, driver 615.78.08 and GCC 16.2.1.
Eight changes were retained after testing fourteen approaches. They preserve
the public API, model tables, feature definitions, input precision and temporal
policies. No new maintained implementation files or shader variants remain.

The baseline includes the preceding, uncommitted audit and DRY work. Comparing
against `HEAD` would also measure those earlier changes. Its complete source
state was therefore captured before this investigation, fingerprinted and
built independently under `/tmp/vqm-vmaf-speed-base/build` with the same
Release, Ninja and warnings-as-errors settings. The original pre-change
benchmark was also preserved; all 113 independently rebuilt baseline shaders
match blobs embedded in that binary.

## Retained changes

The percentages below are independent, incremental GPU-stage comparisons,
except where a narrower target is identified. They are not additive estimates
of whole-frame savings. Five A/B runs were interleaved for each content/size.

| change | responsibility and invariants | measured target |
| --- | --- | --- |
| Early low-variance VIF result | The existing `s1 < 2` result is returned before divisions and polynomial logarithms that it previously replaced. Variances and the selected result expression keep their arithmetic. The now-unreachable `s1 < eps` branch was removed. All fused and fallback statistics paths agree. | VIF −3.5% at 1080p, −4.4% at 4K on the primary inputs. |
| Compact first SpEED filter | Only the nine horizontal taps consumed at each sixteen-column center receive vertical filtering. Scratch capacity stays unchanged; mirrored coordinates, tap order and subsequent mean filtering stay explicit. | SpEED −2.7% / −4.2%; ducks −2.7%. |
| Skip excluded ADM groups | Workgroups wholly outside the scored interior publish exact positive-zero partials and return before shared reductions. Group numbering, partial layout, active-pixel arithmetic and later reduction order remain unchanged. Applies to ADM2 and ADM3. | ADM2 −3.0% / −2.4%; ADM3 −3.2% / −2.6%. |
| Motion tile 32×8 | The same 256 invocations and 3,904 shared bytes cover full contiguous warp rows and calculate fewer vertical intermediates. Pixel blur arithmetic is unchanged; moving-difference grouping changes and is checked against temporal oracles. Memory queries and dispatch-limit checks use the same geometry. | 4K motion −3.6% for v0 and −3.5% for v1. V1 1080p −3.1%; v0 1080p −1.4%, below a separate 2% claim. |
| Shared SpEED factors | Each pooling group stages its two Cholesky factors and spectra once. Edge lanes participate in publication; singular status behavior and each forward solve's operation order are retained. | SpEED −4.1% / −2.1%; ducks −4.3%. |
| Local VIF/ADM dependencies | Remove three VIF and six ADM barriers per call. A reduction can overlap the following downsample/wavelet because their banks are disjoint. The following existing barrier completes reduction reads before partials are overwritten. Entry, final-result and caller synchronization contracts remain. | VIF −3.4% / −2.3%; ADM2 −5.9% / −7.0%; ADM3 −11.0% / −8.4%. |
| Caller synchronization for SDR conversion | Original and distorted conversion write disjoint planes. Their consumer or fit/pad pass already supplies the dependency. The converter documents caller-owned output synchronization. | RGB preparation at 1080p: v0 −3.5%, v1 −3.9%. At 4K the isolated preparation gain is below 2%; no separate 4K gain is claimed. |
| Triangular cooperative covariance | Compute 325 entries per 25×25 matrix and mirror off-diagonal values. The means, per-lane sample order and halving sum tree are unchanged. Integer row mapping covers every matrix entry. | SpEED −4.8% / −12.5%; ducks −5.3%. |

The covariance experiment revisits a materially changed implementation. The
old [symmetric schedule](vmaf-v1-optimization.md#rejected-symmetric-covariance-schedule)
assigned a serial invocation per entry and found no improvement. Each current
entry has a 64-invocation workgroup, and eigensolver costs have fallen since
that measurement. The new measurements establish the benefit of the current
schedule, rather than replacing the earlier result.

## Alternatives not retained

| experiment | result |
| --- | --- |
| Packed native Y/U/V conversion, six dispatches to two | Exact outputs, but preparation regressed 7.2% at 1080p and 1.1% at 4K. Fewer dispatches did not pay for the new mapping. |
| Subtraction-only plane strips | Removed variable division from the packed mapping. Exploratory arms still missed the preparation gate. These arms contained a subsequently identified stale CAMBI artifact, so their whole-frame results are not used for an isolated claim. The original plane path and push layout were restored. |
| Balanced CAMBI histogram clones | Six barriers instead of fourteen, but twelve row updates instead of seven on normal windows. Exact scores; CAMBI regressed 0.8% / 1.9%. The serial clone chain remains. |
| Subgroup motion reduction | Capability-gated 32-lane shuffle tail, an actual active-lane ballot and a portable fallback preserved the original add tree. Exact portable/automatic tests passed, but clean stage comparisons missed 2% and 4K slightly regressed. Removed the variant, capability query and test-only selector. The [Khronos subgroup rules](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-subgroup-size) explain why assuming full subgroups from workgroup size alone would have been unsafe. |
| Preweight ADM magnitudes | Moved repeated CSF multiplication/absolute values into fused decoupling, with explicit rounding boundaries and no extra memory traffic. Tests and printed scores matched; ADM timing stayed within 0.2%. Raw private detail banks remain unchanged. |
| Forced staged streaming memory | Tested separately from the default-policy optimization comparison. Scores/checksums matched, but whole-frame times increased about 19–36% across native and RGB inputs. Some preparation timestamps fell while upload cost moved elsewhere; whole-frame measurement rejects the change. `Automatic` remains the default. |

No shader-stall, occupancy or bandwidth cause is inferred from stage timings
alone. Existing rejected filter/tile/fusion variants in the preceding reports
were reviewed before selecting these experiments.

Restoring source files with old modification times initially left stale CAMBI
and luma SPIR-V in some exploratory binaries. Those comparisons are retained
as diagnostic history, not final evidence. Restored sources were rebuilt, and
all 113 final active SPIR-V artifacts were independently recompiled and checked
byte-for-byte against current source. The final comparison uses fresh builds.

## Workloads and interpretation

Both models use their standard 1080p viewing model, including at 4K input
geometry. V0 uses temporal weighting, without NEG or the optional transform.
Resident calls fully prepare both images; inputs repeat in hot GPU memory and
motion becomes zero after the first call. They exclude upload and decoding.
Streaming includes upload but excludes decoding. This does not establish
unique-frame, complete-video, multi-stream or cross-device throughput.

The primary 4K pair is the existing 10-bit YUV420 fixture. The 1080p pair was
derived from it with FFmpeg's Lanczos scaler, preserving 10-bit YUV420; it is
not the older `/tmp/vmaf1080_*` pair, which now clips to zero model scores.
Additional inputs are the existing 8-bit ducks frame, RGB PNG pairs and dense
CAMBI stress pairs. Every result JSON records both input SHA-256 values.

Resident uses `--iterations 20 --warmup 3 --resident 2`; streaming uses
`--iterations 8 --warmup 2`. Five alternating A/B runs form each comparison.
Whole-frame values are the median of five per-run frame medians, with their
minimum–maximum range. Stage values are the median of five per-run GPU-stage
averages. No results from separate sittings are combined into one median.

## Evidence

Raw commands, logs, per-run stage tables, score/checksum checks, input and
binary hashes, experimental patches, source archives and validation results
are retained under `artifacts/reports/vmaf-speed-investigation/`. The
`verified-*/results.json` files contain the final independently built A/B
comparison; `final-*` contains its earlier repeat using the preserved original
baseline executable. `final-provenance.json` identifies the measured build.

The implementation remains specific about ownership: private reduction banks
are reused only after their readers complete, image preparation retains each
image's source binding, and sequence history remains private to its pipeline.
Model predictors and generated data were not modified. Independent CPU and
libvmaf oracles were preserved.

## Final frame times

These are the final independent-build comparisons. Values are milliseconds; brackets show the five-run minimum–maximum. All 240 runs were checked, and each arm had stable scores/checksums across its five runs. Small between-arm differences are quantified below.

| workload | before ms, median [range] | after ms, median [range] | frame time change |
| --- | ---: | ---: | ---: |
| vmaf-v0-1080 native resident | 0.399315 [0.397195–0.408094] | 0.377810 [0.376285–0.379669] | -5.39% |
| vmaf-v0-4k native resident | 1.707036 [1.704056–1.730676] | 1.599372 [1.598442–1.602072] | -6.31% |
| vmaf-v0-ducks native resident | 0.402449 [0.400635–0.430449] | 0.386655 [0.384885–0.387815] | -3.92% |
| vmaf-v1-1080 native resident | 0.713249 [0.711860–0.714960] | 0.683736 [0.682766–0.684535] | -4.14% |
| vmaf-v1-4k native resident | 2.722421 [2.718946–2.723812] | 2.616188 [2.588778–2.619298] | -3.90% |
| vmaf-v1-ducks native resident | 0.589612 [0.588992–0.591241] | 0.558952 [0.558427–0.560642] | -5.20% |
| vmaf-v0-1080 native streaming | 0.771549 [0.766734–0.775114] | 0.751559 [0.745435–0.758479] | -2.59% |
| vmaf-v0-4k native streaming | 3.184170 [3.180820–3.186514] | 3.077421 [3.073797–3.079022] | -3.35% |
| vmaf-v0-ducks native streaming | 0.596157 [0.588082–0.600442] | 0.580472 [0.567677–0.582087] | -2.63% |
| vmaf-v1-1080 native streaming | 1.275072 [1.265772–1.276942] | 1.243448 [1.241597–1.244577] | -2.48% |
| vmaf-v1-4k native streaming | 4.910951 [4.907840–4.925930] | 4.798652 [4.793127–4.814536] | -2.29% |
| vmaf-v1-ducks native streaming | 0.874993 [0.865067–0.878862] | 0.839283 [0.834038–0.849253] | -4.08% |
| vmaf-v0-rgb1080 RGB resident | 0.446744 [0.445783–0.459929] | 0.423464 [0.423064–0.424639] | -5.21% |
| vmaf-v0-rgb4k RGB resident | 1.914217 [1.912987–1.926793] | 1.811844 [1.810660–1.829004] | -5.35% |
| vmaf-v1-rgb1080 RGB resident | 0.740194 [0.738274–0.743964] | 0.706889 [0.704735–0.725049] | -4.50% |
| vmaf-v1-rgb4k RGB resident | 2.767035 [2.749826–2.773016] | 2.656218 [2.649132–2.657217] | -4.00% |
| vmaf-v0-rgb1080 RGB streaming | 1.008196 [1.006355–1.011456] | 0.984266 [0.974997–0.985980] | -2.37% |
| vmaf-v0-rgb4k RGB streaming | 4.117171 [4.111261–4.161631] | 4.017893 [4.013788–4.020863] | -2.41% |
| vmaf-v1-rgb1080 RGB streaming | 1.299376 [1.295832–1.299826] | 1.266157 [1.258847–1.267132] | -2.56% |
| vmaf-v1-rgb4k RGB streaming | 4.962079 [4.936995–4.977449] | 4.850382 [4.836276–4.861856] | -2.25% |
| vmaf-v0-dense1080 dense resident | 0.399364 [0.398894–0.400439] | 0.376005 [0.375399–0.392440] | -5.85% |
| vmaf-v0-dense4k dense resident | 1.711541 [1.705322–1.723205] | 1.598227 [1.594867–1.617692] | -6.62% |
| vmaf-v1-dense1080 dense resident | 0.813643 [0.805743–0.826648] | 0.780564 [0.778989–0.789614] | -4.07% |
| vmaf-v1-dense4k dense resident | 3.271604 [3.263859–3.272988] | 3.174660 [3.171710–3.176990] | -2.96% |

Resident frame-time reductions span 3.9–6.6% for v0 and 3.0–5.2% for v1 on these contents. Streaming reductions span 2.3–4.1% across the two versions. These percentages describe elapsed time, not an equal percentage increase in throughput.

The final native 4K GPU-stage reductions are about 6.6% for VIF, 9.1% for ADM2, 10.8% for ADM3, 17.9% for SpEED and 3.4–3.5% for motion. CAMBI remains unchanged and is still the largest v1 stage. Stage totals and frame times differ because recording, upload, submission and readback also take time.


## Correctness and source checks

- Release warnings-as-errors build and ASAN/UBSAN build passed.
- Hardware and Mesa software Vulkan suites each passed all 201 tests with synchronization validation and no validation errors.
- ASAN/UBSAN passed all 201 GPU tests with the repository's `detect_leaks=0` / `halt_on_error=1` settings.
- Release and sanitizer CTest each passed 12/12, including installed/embedded package consumers and benchmark integration.
- Recursive clang-format 23.1.4 checked 199 maintained files; clang-tidy checked 41 compiled production C++ files.
- All 113 active shaders passed `spirv-val --target-env vulkan1.1` and fresh-compilation byte checks.
- All 14 installed public headers and both generated model tables remain byte-identical to the captured baseline.
- SSIMULACRA 2 identity is `100.00000000`; Butteraugli identity is `0.00000000`.
- The 300-pair SSIMULACRA 2 MCOS correlation gate passed at tolerance 0.005.
  Its separate absolute-score parity diagnostic still reports the existing
  Cloudinary divergence (maximum 0.524941, mean 0.202585); no SSIMULACRA 2
  implementation or shader changed in this investigation.

Two new ADM tests poison the largest partial arrays with NaNs at 513×385 and 512×256, repeat identity/distorted/identity calls, and check that every slot is rewritten. Fully excluded groups beyond the later scales' reused prefix must publish exact positive zero. Existing independent feature, odd-edge, fallback, NEG, external-buffer, low-bit, singularity and moving-sequence oracles retain their original tolerances.


## Full CID22 score checks

Both models scored all 4,292 independent still pairs. The reference validation checked the source/YUV manifest; an additional pass verified the actual payload bytes of all 4,341 Y4M wrappers consumed by both arms. The hashed score comparisons verify identical inputs, not just matching pair names. Motion is zero for this still-image corpus; temporal tests are separate.

| introduced score drift | v0 | v1 |
| --- | ---: | ---: |
| maximum | 0.0000360100 | 0.0000032000 |
| mean | 0.0000031720 | 0.0000004254 |
| p99 | 0.0000149218 | 0.0000018000 |

The additional engineering drift guard was maximum ≤0.001 and mean ≤0.0001; both pass comfortably. No test or reference tolerance was relaxed. Although branch/sample arithmetic is preserved, shader control flow and load placement can change compiler contraction/scheduling, so the retained code is not claimed bit-identical.

| absolute score difference versus pinned libvmaf | v0 | v1 |
| --- | ---: | ---: |
| maximum | 0.0171078278 | 0.0046387765 |
| mean | 0.0001956489 | 0.0016747896 |

The v0 absolute-score difference is the existing float-reference divergence, not a new ≤0.005 absolute parity claim. KRCC/SRCC/PCC remain within 0.00002 of the corresponding reference correlations. Full per-model statistics and provenance are in `corpus/summary.json` and `drift-v{0,1}.json`.


## Remaining opportunities

The next large v1 target is active-content CAMBI: its histogram/contrast work remains roughly half of the resident 4K frame on the primary input. A larger vertical tile would need an explicitly selected shared-memory variant and a portable fallback; a subgroup redesign needs careful active-lane and synchronization handling. Neither is claimed faster without a new prototype and content-dependent measurements.

For small images, command recording/submission and the limited parallelism of four SpEED matrices deserve separate measurement. Descriptor or recorded-command reuse would have to preserve source changes, history swaps and one-submission-in-flight ownership. Startup pipeline caching and independent-video concurrency are distinct workloads from this hot single-sequence benchmark. Reusing a repeated resident reference's features would likewise not establish faster unique-frame processing.

AMD/Intel hardware performance and complete changing-frame video throughput were not measured. The software-device and moving-frame oracle checks provide correctness coverage, not those performance claims.
