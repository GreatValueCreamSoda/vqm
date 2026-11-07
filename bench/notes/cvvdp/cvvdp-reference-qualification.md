# CVVDP reference and numerical qualification

2026-11-06. The implemented base ColorVideoVDP target passes the complete
**4,292-pair CID22** comparison against the pinned original CPU FP32 model.
Maximum absolute JOD error is **9.6611084e-7**, against the fixed **1e-3** gate;
mean absolute error is **2.4369149e-7**. These are numerical qualification
results. Performance measurements and the broader API/resource gates are in
[the implementation report](cvvdp-implementation.md).

## Target, inputs and execution

The canonical target is original ColorVideoVDP **v0.5.7**, commit
`2a268bce8d56e2f3abde46df3927d8a633707a24`, base `cvvdp_mult_mutual` model,
`weber_g1` contrast and bundled `weber_fixed_size` CSF. The original MATLAB
entry point wraps this Python implementation; no native MATLAB metric was
substituted. Vship is a separate comparison reference and supplied no copied
implementation code or model data.

Original execution used CPython 3.12.14, PyTorch **2.15.0+cpu**, one CPU thread,
FP32 tensors and `highest` float32 matrix precision. The complete optional
runtime is pinned in
[`tools/cvvdp-reference-requirements.txt`](../../../tools/cvvdp-reference-requirements.txt).
The resolved display was `standard_4k`: 200 nits, contrast 1000, ambient 250 lux,
reflectivity 0.005, exposure 1 and 75.402449345163049 pixels/degree. Geometry
was preserved. Source interpretation was explicitly sRGB.

Pillow **13.0.0** decoded each corpus input to opaque RGB8. Canonical lossless
PNGs were written under `.cache/validation/cvvdp-rgb8/`; both arms consumed
those exact samples. The original took normalized raw `[B,C,F,H,W]` FP32
tensors, bypassing its image loaders. The candidate read the canonical PNGs.
This separates metric error from JPEG decoder differences. Each raw result
retains source-file and decoded-pixel SHA-256, dimensions and both scores.

The final candidate was the frozen `vqm_bench` snapshot:

```text
51e144d5a07a082a9ef8dccf3eaf21ebf40ca5268bbeff2238374bc799ac5568
```

It ran on **NVIDIA GeForce RTX 5070**, driver **615.78.08**, Vulkan device UUID
`7293d6cd964492010338cb259cfc9cf2`. The implementation includes the existing
uncommitted audit/reorganization work; repository HEAD
`a949b6177efb88890dd44860cc30dff45486a25d` alone does not identify the measured
source. Both the supplied final snapshot and the qualification's own copied
executable are retained under `artifacts/reports/cvvdp/`.

## Results and retained evidence

| Check | Result |
| --- | ---: |
| Full CID22 distortion pairs | 4,292 / 4,292 pass |
| Absolute JOD gate | 0.001 |
| Maximum absolute JOD error | 0.0000009661108393 |
| Mean absolute JOD error | 0.0000002436914921 |
| Largest MCOS correlation difference | 0.0000006026576911 (KRCC) |

The largest error is `original/2079234.png` versus
`compressed/2079234/cld_webp/q90.png`: original **9.94027042388916**, candidate
**9.94027139**. Candidate output is rounded to eight decimal places. This gate
establishes bounded numerical error, not bit identity.

For supplementary subjective-score context, original/candidate correlations
against MCOS are KRCC **0.6237678657 / 0.6237672630**, SRCC
**0.8213785914 / 0.8213785373**, and PCC **0.7849762452 / 0.7849762816**.
These correlations are not a substitute for the pairwise numerical gate.

Maintained supporting data:

- [Compact qualification summary](../measurements/cvvdp-qualification-results.json)
  contains the executable/device identity, corpus CSV hash, ordered source and
  decoded-input manifest hash, fixed gate and supporting-data checksum.
- [All 4,292 score comparisons](../measurements/cvvdp-cid22-results.csv)
  preserve pair identity, both scores and absolute error in corpus order.
- [Independent oracle characterization](oracle-characterization.json)
  retains deterministic image/sequence stage data, input hashes, temporal
  coefficients, repeated original scores and odd-boundary evidence.
- Raw logs, complete per-file provenance and retained executables are under
  `artifacts/reports/cvvdp/cid22-final/`; initial exploration is preserved
  separately under `cid22-20/` and `cid22-full/`.

The final run reused the completed, fully keyed original-score cache and
reported zero seconds for that replay. It did not re-execute all original
predictions. The original process was paused during separate performance
capture; neither its total elapsed time nor retained per-pair `seconds`
fields should be used as a throughput benchmark.

## Independent fixtures and compatibility decisions

The checked-in fixture generator produced **29** complete image/clip cases,
**10** direct spatial DKL cases and coefficient arrays for **7** integral or
rational frame rates. Every complete original score was repeated three times;
maximum observed CPU repeat spread was **0**. Inputs use explicit stepwise
binary32 arithmetic, matching the C++ fixture contract.

Coverage includes identity at 10, tiny errors, opponent colors, negative JOD,
small/mixed-parity rectangles, FHD/4K presets, P3, PQ, HLG, absolute linear
input, one-frame image equivalence, two-frame clips, 1/120/240 fps limits,
short history, ring wrap, fractional double wrap, flicker and frame counts
N-1/N/N+1 around the filter length. The direct spatial fixtures isolate the
pyramid, adaptation, CSF, masking and pooling from display conversion and FIR.
Routine GPU checks consume these independently generated values; see the
implementation report for their actual hardware/software gate results.

The suspected original pyramid parity issue was reproduced numerically.
For height 5, width 4, an `arange(20)` plane reduces with a right-column
error up to **4.300001** compared with mathematically symmetric padding;
height 4, width 5 differs by **3.387501**. Matched-parity planes agree within
1.91e-6. The implementation preserves the pinned original's height-parity
branch for compatibility. Silently correcting that branch was rejected as a
change to the numerical target.

Uniform achromatic black versus white scores 10 in the original's contrast
model; tiny uniform intensity changes can do the same. The fixtures preserve
that observed behavior. Opponent-color and flickering sequences independently
exercise large distortions, including a **-4.788311004638672 JOD** reference
case. No clamp or rescale was added to make scores appear more conventional.

## Reproduction and limits

Follow [the development workflow](../../../docs/cvvdp-development.md) for the
read-only original pin, reference environment, table `--check`, oracle
regeneration and validation commands. The final corpus comparison used:

```sh
build/cvvdp-reference-env/bin/python validation/cvvdp.py --limit 0 \
  --binary artifacts/reports/cvvdp/vqm_bench-final \
  --out-dir artifacts/reports/cvvdp/cid22-final
```

The table generator's pinned-asset checks pass, reference-fetch selection tests
pass (7 tests), the original checkout remains clean, and Ruff/YAPF checks pass
for the changed reference, validation and quality tools. The evidence does not
establish parity with every PyTorch release or GPU driver. CID22 is an SDR
still-image corpus; temporal and HDR coverage here comes from deterministic
fixtures. Large natural HDR/video corpora remain extended validation work.
Supported public scope and rejected input/configuration combinations are
specified in [the CVVDP guide](../../../docs/cvvdp.md).
