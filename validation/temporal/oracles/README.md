# Independent hosted video scores for the five established metrics

These compact reports contain original scores for the exact downloaded and
prepared corpus described in [sources.json](../sources.json) and
[the workflow guide](../README.md). They contain no
source video, candidate output or executable.

- `smoke/`: three source groups, 13 variants, 12 frames; 39 complete clip pairs
  per metric, 195 metric/clip comparisons across this five-metric matrix.
- `qualification/`: 30 source groups, 13 variants, 40 frames; 390 complete
  clip pairs per metric, 1,950 metric/clip comparisons across this five-metric
  matrix.

Each directory contains one canonical score report, original-only frozen
policy and runtime provenance record per metric: SSIMULACRA 2, Butteraugli,
VMAF v0, VMAF v1 and CVVDP. These are matrix sizes rather than claims about a
candidate's pass/fail outcome. The runner's aggregate `summary.json` reports
the completed selection and its actual correlations and gate results. Policy
hashes bind the exact prepared sample manifest and resolved configurations.
Margins were computed from original scores before candidate evaluation.
Neither numerical nor ranking thresholds were widened during qualification.

After the paired old/new audit, ranking became the default for these registered
original/policy pairs on 2026-11-06. Raw numerical errors remain diagnostic;
`--gate-mode both` explicitly requires their numerical thresholds as well.
The [qualified-domain registry](../../qualified_domains.json) binds the original
scores, input/configuration identities, reference pin and frozen policy.
Unregistered or modified pairs retain the combined default. This changes
final corpus-score acceptance while identity and numerical component tests
remain required. Historical reports retain their measured gate modes.
[Qualified score gates](../../README.md#qualified-score-gates)

Originals are Cloudinary SSIMULACRA 2 at
`81feacf6a5b0bbd7ec11b4a881a4615033ab0b10`, libjxl Butteraugli at
`e47d2a2b5bb418f2947ce07f36cb048861db3820`, Netflix libvmaf at
`4611e47a7330cd342498dcc130e1008efbeab6e4` and base ColorVideoVDP v0.5.7 at
`2a268bce8d56e2f3abde46df3927d8a633707a24`. VMAF uses its public scalar CLI
with native YUV444 8-bit samples, one thread, the float v0.6.1 and SDR v1 3H
models. The executable statically contains libvmaf; executable/model hashes
are recorded. Its per-frame JSON scores have the original CLI's six-digit
precision before arithmetic-mean clip pooling.

CVVDP uses the original Python implementation on CPU FP32, one thread per
clip, highest float32 matrix precision, Standard4K photometry and replicate
temporal padding. Input RGB is an explicit conversion of the same canonical
YUV444 bytes consumed by the public CLI. Whole-clip JOD is retained without
averaging frame JOD. Runtime versions and generator fingerprints are recorded.
See the [Netflix notice](../../../licenses/NETFLIX-VMAF-LICENSE) and
[ColorVideoVDP notice](../../../licenses/COLORVIDEOVDP-LICENSE).

SSIMULACRA 2 and Butteraugli use a separate spatial canonicalization of the
same source clips. Clipped source RGB is compressed into `[0.02, 0.98]`, encoded
as limited-range BT.601 YUV444 8-bit, then decoded without clipping from those
exact bytes. The decoded values are safely within `[0, 1]`. The public `vqm`
CLI scores this spatial Y4M with `--input-transfer srgb`. Each original image
tool receives the decoded FP32 sRGB-encoded values in a PFM per frame, and the
complete-clip score is the arithmetic mean of final frame scores. The original
spatial harness serializes each frame score to eight decimal places before
pooling, adding at most `5e-9` score units of decimal rounding per frame. Separate
spatial paths and hashes bind this representation; its pixels are not claimed
to equal the native VMAF/CVVDP representation. Attribution is retained in the
[Cloudinary notice](../../../LICENSE) and
[JPEG XL notice](../../../licenses/JPEG-XL-LICENSE).

Regenerate using `--reference generate --reference-only` into
`artifacts/reports/`, as documented in the workflow guide. Review its complete
input/runtime evidence, coverage and mutation audit before promoting
`<metric>-original.json` to `<metric>.json`, plus its matching policy and
provenance record. Ordinary validation consumes these fixtures without
installing the original runtimes.

These fixtures measure agreement on a fixed SDR corpus. Describe new runs
by their complete metric selection and aggregate results.
