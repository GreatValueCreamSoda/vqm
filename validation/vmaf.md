# VMAF reference and validation

Run these commands from the repository root after the local build and Python
setup. [Build and test](../CONTRIBUTING.md#build-and-test)

### Pinned libvmaf and sequence fixtures

`tools/references.lock` pins Netflix libvmaf 3.2.0 at commit
`4611e47a7330cd342498dcc130e1008efbeab6e4`. The reference build needs Meson,
Ninja, and C/C++ compilers. The build script copies the checkout to
`build/vmaf-src` and compiles it in `build/vmaf-build` with SIMD assembly and
embedded models disabled. The scalar build supports float and integer feature
extractors and loads model files by path. Use it as a correctness oracle.

```sh
python3 tools/fetch_references.py --only vmaf
python3 tools/build_references.py --only vmaf --jobs 8
python3 tests/make_vmaf_fixtures.py
.venv/bin/python -m unittest discover -s tests -p test_vmaf_oracle.py -v
```

The generator writes four synthetic 64×48 frames per input in planar YUV420 at
8, 10 and 12 bits. The 10- and 12-bit `low-bits` pairs differ in luma samples
while their upper eight luma bits match. The fixture manifest at
`build/fixtures/vmaf/manifest.json` records geometry, depth, frame count and
SHA-256 hashes.

The `vmaf_reference` driver calls libvmaf's public C API. Its `float` and
`integer` modes each report a score and six features per frame in CSV at
17-digit precision. Use the model file for the selected feature extractor.

```sh
build/vmaf-build/vmaf_reference float \
  build/vmaf-src/model/vmaf_float_v0.6.1.json \
  build/fixtures/vmaf/10bit-low-bits-ref.yuv \
  build/fixtures/vmaf/10bit-low-bits-dist.yuv 64 48 10

build/vmaf-build/vmaf_reference integer \
  build/vmaf-src/model/vmaf_v0.6.1.json \
  build/fixtures/vmaf/10bit-low-bits-ref.yuv \
  build/fixtures/vmaf/10bit-low-bits-dist.yuv 64 48 10
```

Both modes take raw little-endian YUV420 at 8, 10, 12 or 16 bits, with matching
even dimensions of at least 16×16 and equal, nonzero counts of complete frames.
The driver processes frames in order and flushes libvmaf before reading results
so the output includes the end-of-stream motion behavior. The native libvmaf
CLI at `build/vmaf-build/tools/vmaf` reports JSON values to six decimal places.
Use the CSV driver for tighter feature comparisons. Its separate `adm-debug`
mode reports the float ADM numerator and denominator at four scales and the
aggregate ADM2. Pass `-` in place of a model.

```sh
build/vmaf-build/vmaf_reference adm-debug - \
  build/fixtures/vmaf/8bit-moving-ref.yuv \
  build/fixtures/vmaf/8bit-moving-dist.yuv 64 48 8
```

VQM's C++ tests retain independent feature and frame-score oracles. Coverage
includes 4:4:4 and 4:2:0, low-order high-bit-depth differences, sequence
boundaries, CAMBI encoded-depth overrides, and borrowed buffers.

```sh
cmake --build --preset release --parallel 8
./build/vqm_test --require-gpu
ctest --preset release
```

### HDR sequence oracle

To regenerate the HDR oracle, build the pinned libvmaf reference with zimg
available to Meson, then run the fixture generator.

```sh
python3 tools/fetch_references.py --only vmaf
python3 tools/build_references.py --only vmaf
meson setup build/vmaf-hdr-build build/vmaf-src/libvmaf --buildtype=release \
  -Ddefault_library=static -Dbuilt_in_models=false -Denable_asm=false \
  -Denable_tests=false -Denable_docs=false -Denable_tools=true \
  -Denable_float=true -Denable_zimg=true
meson compile -C build/vmaf-hdr-build
.venv/bin/python tests/make_vmaf_v1_oracles.py --hdr > tests/metrics/vmaf/pipeline/vmaf_v1_hdr_reference.inc
```

### CID22 still-image comparison

`validation/vmaf_cid22.py` requires NumPy and Pillow from the local Python
setup. It converts the 4,292 annotated 512×512 pairs and scores each pair as an
independent one-frame sequence with zero motion. The reference uses the
v0.6.1 float model and the standard v1.0.16 SDR 3H model. Use `--limit 300`
for a shorter run, with the same limit on both commands.

```sh
python3 tools/fetch_references.py --only cid22
.venv/bin/python validation/vmaf_cid22.py --jobs 8
.venv/bin/python validation/vmaf_cid22.py --jobs 8 --vqm-v0 --vqm-v1 \
  --reference-scores artifacts/reports/vmaf-cid22/scores.csv \
  --out-dir artifacts/reports/vmaf-cid22-vqm
```

The candidate run requires `build/vqm_bench` and a working Vulkan device. It
runs one process per VQM model with `--score-only`. Use just
`--vqm-v0` or `--vqm-v1` to compare a single model. With `--reference-scores`,
the tool checks the previous run's input manifest, models, driver, annotations,
conversion and pair order before reusing its reference scores.

The conversion applies the BT.709 Y′CbCr matrix directly to sRGB-encoded 8-bit
RGB samples. It rounds to nearest, ties to even, in 10-bit limited range with
offset/scale 64/876 for Y and 512/896 for Cb and Cr. Chroma stays 4:4:4.
The tool ignores opaque alpha and rejects non-opaque alpha.

`inputs.csv` records geometry, paths and source/YUV SHA-256 hashes for the exact
bytes cached in `.cache/validation/vmaf-cid22-yuv/`. The full cache uses about
6.4 GiB. The VQM arms wrap those bytes in single-frame Y4M files in
`.cache/validation/vmaf-cid22-y4m/`, using about another 6.4 GiB. Use
`--prepare-only` to generate the input manifest without scoring.
`scores.csv` contains the per-pair scores. `summary.json` records correlations,
score differences and model, driver and input fingerprints.

To compare an implementation on the same inputs, pass `--vqm-scores PATH` with
one row per selected pair and `original`, `distorted`, and `v0` and/or `v1`
columns. To compare two optimization runs pair by pair, pass their score files
to the comparison tool. The default column is `vqm_v0`. Use
`--column vqm_v1` for v1 or specify separate reference and candidate columns.

```sh
.venv/bin/python validation/compare_vmaf_scores.py \
  BASELINE/scores.csv CANDIDATE/scores.csv
```

When both reports contain input hashes, the comparison requires them to match.
Optional `--max-absolute` and `--mean-absolute` thresholds fail the comparison
when drift exceeds the specified limits.

### Complete clips

The CLI accepts `--metric vmaf-v0` and `--metric vmaf-v1` for independent images
and complete progressive Y4M sequences:

```sh
./build/vqm --metric vmaf-v0 --vmaf-transform original.y4m distorted.y4m
./build/vqm --metric vmaf-v0,vmaf-v1 --vmaf-model 4k --vmaf-neg \
  --vmaf-scale --vmaf-pad original.y4m distorted.y4m
./build/vqm --metric vmaf-v0 --json --pairs-file pairs.tsv \
  --iterations 5 --warmup 1
```

Each TSV row is `sample ID<TAB>original path<TAB>distorted path`. JSON contains the
arithmetic-mean clip score and every emitted frame in index order, with its score
and feature array. The v0 feature order is ADM2, motion2 and VIF scales
0–3. V1 uses CAMBI, SpEED chroma UV, ADM3 and motion3. The CLI drains pending
frames before reporting a clip.

For v0, `--vmaf-model 1080|4k` selects the predictor, `--vmaf-neg` disables
enhancement gain, and `--vmaf-transform` applies the phone transform. The
options compose and apply to the v0 results when both versions are selected.

`--vmaf-scale` scales each selected SDR VMAF metric to fit its own model
canvas. `--vmaf-pad` adds bars to make the frame 16:9. For a 960×720 input
and a 1080p model, scaling gives 1440×1080, padding gives 1280×720, and both
flags give 1920×1080. A 4K model with both flags uses 3840×2160. The HDR v1
preset controls its own geometry. The shared `--input-*` color overrides
apply in all modes.

`--iterations` and `--warmup` repeat whole clips with reset history. Core timing
includes the final flush and pooling; decode is reported separately. Hosted
video validation uses this same executable:

```sh
.venv/bin/python -m validation.temporal.run --profile smoke --metrics vmaf-v0,vmaf-v1
```

The runner scores at the prepared input dimensions to match the pinned
original-implementation records. The registered profile requires ranking
agreement and reports numerical differences. Add `--gate-mode both` to require
the numerical limits too.
[Frozen corpus and full qualification workflow](temporal/README.md)

## Repeated-pair benchmark

`vqm_bench` retains the repeated-pair VMAF microbenchmark: it decodes one pair and
advances the sequence by reusing that pair, including the first frame of a Y4M
file. Its resident and streaming stage measurements describe that workload.
Use its `--vmaf-chroma native`, `--vmaf-chroma 420`, or `--vmaf-chroma 444` options
to select the input path described above. The repeated-pair benchmark defaults
to native geometry. Use `--vmaf-fit pad` to time the fitted path for either
VMAF version.

```sh
./build/vqm_bench --metric vmaf-v1 --vmaf-chroma 420 \
  --pair original.png distorted.png --resident 2 --iterations 20 --warmup 3
```
