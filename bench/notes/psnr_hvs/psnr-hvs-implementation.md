# FP32 PSNR-HVS implementation

2026-11-06. PSNR-HVS is implemented as a frame-local Vulkan metric. All signal
arithmetic is FP32, including the orthonormal 8x8 DCT, masking, accumulation,
normalization, channel mixture and logarithm. Public double fields hold
promoted completed FP32 values. The target is ranking agreement with pinned
libvmaf; integration with the evolving test and qualification tooling was
explicitly deferred.

## Implementation and input contract

`include/psnr_hvs.h` exposes `PsnrHvsConfig`, `PsnrHvsScore` and
`PsnrHvsPipeline`. Implementation and shaders live under
`src/metrics/psnr_hvs/`. Construction and memory queries share a layout planner.
The combined `Metrics` API exposes independent PSNR-HVS configuration, aligned
frame results and memory requirements/usage/budgeting.

Native inputs are opaque YUV420/422/444, Byte at 8 bits or Word at 10/12 bits,
with matching format/geometry and complete 8x8 planes. Samples enter FP32 as
unchanged code values. Integer operations only unpack storage and describe
addresses/geometry. RGB, alpha, float/half storage, other depths and automatic
color conversion are outside the initial input contract. See the
[metric guide](../../../docs/psnr-hvs.md).

The implementation retains Xiph's CSF tables, variance/masking model, stride-7
sampling and normalized-error plane weights. Its FP32 DCT replaces the
reference's rounded integer transform; parallel sums replace the serial
coefficient fold. Reproducing those original arithmetic choices was rejected
because the user requires FP32 processing and direct ranking acceptance.

The normal CLI supports still Y4M, complete clips, JSON and batch pairs. Clip
scores use an FP32 mean of frame dB scores. Zero measured error yields positive
infinity, serialized as `"inf"`. Benchmarks support streaming, resident,
mixed-reference, host-view, external-buffer and fixed-count worker paths.
Both images are prepared on every call. Score-only benchmark batches require
one worker; pool, resident-ring, throughput-window and metric-pair modes are
explicitly unsupported.

## Build and provenance

The implementation extends the existing uncommitted metric-local refactor;
HEAD is `a949b6177efb88890dd44860cc30dff45486a25d`. Existing work was retained
and no commit was created. A fresh build used GNU 16.2.1, Release, warnings as
errors and test targets disabled:

```sh
cmake -S . -B build-psnr-hvs -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DVQM_WARNINGS_AS_ERRORS=ON -DVQM_BUILD_TESTS=OFF
cmake --build build-psnr-hvs --target vqm vqm_bench --parallel 8
python3 tools/check_quality.py --lint --build-dir build-psnr-hvs --jobs 4
```

Both executables built successfully. Quality checks passed for 256 maintained
formatted files and 52 compiled production translation units. All three
shaders also compiled with standalone `glslc`. Disassembled built SPIR-V has
FP32 floating types and no FP16/FP64 floating types. Manual GPU checks ran on
NVIDIA GeForce RTX 5070, driver 615.78.08.

The reference executable was the existing `build/vmaf-build/tools/vmaf`, with
the pinned revision `4611e47a7330cd342498dcc130e1008efbeab6e4`. Its copied
PSNR-HVS source matches the pinned `.reference/` source by SHA-256. Source,
executable and input hashes are retained with compact results in
[`implementation-smoke.json`](implementation-smoke.json).

## Manual checks and results

Comparison inputs were three deterministic 96x64, three-frame Y4M pairs at
24 fps: 420/8-bit, 422/10-bit and 444/12-bit, containing ramps, seeded noise
and positive per-plane DC offsets. Both implementations consumed the same
stored codes. Candidate CLI runs used two complete measured iterations and
one warmup. The reference extracted its four PSNR-HVS features:

```sh
build/vmaf-build/tools/vmaf \
  -r .cache/bench/psnr-hvs-smoke/420-8-original.y4m \
  -d .cache/bench/psnr-hvs-smoke/420-8-distorted.y4m \
  --feature psnr_hvs --no_prediction --json \
  -o artifacts/reports/psnr-hvs-implementation/reference-420-8.json
```

| Native input | Our mean frame score, dB | Maximum frame-score difference from reference, dB |
| --- | ---: | ---: |
| YUV420, 8-bit | 16.1921215057 | 0.00755958 |
| YUV422, 10-bit | 16.2176322937 | 0.00124741 |
| YUV444, 12-bit | 16.2239971161 | 0.00019223 |

The reference JSON prints six decimals. These few constructed cases establish
neither ranking qualification nor a universal numerical error bound.

The first 420/8-bit frame scored `17.3338775635` in all six benchmark input/
worker arms. Four timed calls and one warmup checked stage tables/checksums.
Workers warm separately, so the two-worker checksum includes an extra warmup.
These timings were not interleaved against a baseline and support no speedup
claim. Three-frame 444/12-bit identity produced infinite frame/clip scores
with exactly zero normalized errors.

A separate 16x16 420/8-bit public API probe changed constant Y from 100 to 105
with unchanged 128-valued chroma. Direct and staged policies both scored
`30.9334907532` dB, with Y `29.964389801` and infinite Cb/Cr. Streaming,
resident and available allocated-buffer results agreed. All returned values
round-tripped through float exactly. Memory queries matched actual allocations:
arena 23,936 bytes; staging 0 bytes direct and 8,560 bytes staged. Identity,
in-flight mutation guards, unknown/consumed tokens and released handles passed
manual checks. Allocated-image mapping correctly required the direct path.

A 32x32 unified API probe checked identity, immediate PSNR-only frames,
empty/idempotent flush, reset, top-level configuration overrides and memory
budgeting. PSNR-only arena use was 57,984 bytes with no staging; its configured
64 MiB heap budgets permitted 1,157 workers arithmetically. SSIMULACRA 2 plus
PSNR-HVS returned both scores with separate memory entries. With VMAF v0
lookahead, PSNR scores remained aligned to frames 0, 1 and 2, matching their
standalone values `32.8717002869`, `26.8510913849` and `23.3292655945`.

Installation into `build-psnr-hvs/install` succeeded. An external CMake consumer
found `vqm`, linked `vqm::core`, compiled the leaf and umbrella headers,
resolved memory-query/constructor symbols and ran successfully. Installed
PSNR-HVS header and Xiph license matched the maintained files byte-for-byte.

Raw outputs, temporary probes and reproduction commands are preserved under
`artifacts/reports/psnr-hvs-implementation/`; generated Y4M inputs are under
`.cache/bench/psnr-hvs-smoke/`. They are not prerequisites for a core build.

## Deferred qualification and conclusion

No PSNR-HVS tests, reference fixtures, agreement policies, corpus manifests or
test registrations were added. No `ctest`, maintained GPU test suite, sanitizer
run, ranking corpus or interleaved performance gate was run for this addition.
The ongoing testing/qualification work remains separate as requested.

Remaining qualification must cover frozen source-balanced ranking domains,
broader distortions/resolutions, resource/synchronization edge cases and any
future input conversion. Keep component oracles and identity anchors while
accepting final scores through the agreed ranking policy.

The API, CLI and benchmark implementation is available with FP32 signal
processing. Fresh build, quality, packaging and manual GPU checks passed;
corpus-wide reference agreement and performance remain unqualified.

Subsequently, the requested resident throughput was measured under the
five-run protocol; see the [dated resident report](resident-throughput-20261106.md).
It establishes a measured resident baseline, with no before/after speedup claim.
