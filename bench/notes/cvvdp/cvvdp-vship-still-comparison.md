# CVVDP first implementation: Vship still comparison

2026-11-06. Five alternating rounds on an RTX 5070, driver 615.78.08, compare
VQM's initial FP32 CVVDP implementation with the pinned Vship Vulkan C API.
Both process the same decoded RGB8 sRGB PNGs, explicitly using the original
`standard_4k` physical display, with no resizing or heatmap. These are
independent still-image workloads, not repeated frames of a video.

| Geometry | Implementation | Median ms/image | Five-round range ms | Images/s |
| --- | --- | ---: | ---: | ---: |
| 1920×1080 | VQM streaming | 2.636373 | 2.539283–2.842454 | 379.3 |
| 1920×1080 | Vship Vulkan | 5.834371 | 5.042198–6.915781 | 171.4 |
| 3840×2160 | VQM streaming | 10.888119 | 10.432766–11.682422 | 91.8 |
| 3840×2160 | Vship Vulkan | 21.093563 | 19.078982–22.387239 | 47.4 |

Each round uses eight timed calls after two warm-ups, with decode and pipeline
construction outside the measured calls. VQM uploads both inputs on every
call. Vship's public C API also consumes host planes. The Vship driver resets
both history and accumulated score before each call, sets `fps=1`, and takes
its one-image result. Reset API overhead is included in the peer's core time;
VQM's reset is outside its timer. This boundary difference is retained and
reported, rather than silently equating the two timers.

The results establish an initial streaming baseline on these two pairs.
They do not measure resident images, ordered video, setup latency or a corpus.
Vship has known numerical differences, so this comparison does not establish
numerical interchangeability or a stage-level optimization claim.

## Correctness and provenance

The numerical oracle is original ColorVideoVDP v0.5.7 at
`2a268bce8d56e2f3abde46df3927d8a633707a24`, CPU FP32, one thread,
PyTorch `2.15.0+cpu`. Its two PNG input scores were computed separately;
those oracle durations were not qualified CPU performance measurements.

| Geometry | Original CPU JOD | VQM JOD | Vship Vulkan JOD |
| --- | ---: | ---: | ---: |
| 1920×1080 | 9.589558601379395 | 9.58955901 | 9.58127880 |
| 3840×2160 | 9.572444915771484 | 9.57244480 | 9.56727982 |

VQM's absolute errors are below `4.1e-7` JOD on these inputs. The peer's errors
are approximately `0.00828` and `0.00517` JOD. It remains a comparison arm;
see the [source review](../../../docs/design/cvvdp-vship-review.md) for the known differences
in power semantics, Gaussian support/borders, temporal sampling and startup.

The candidate executable SHA-256 was unchanged before and after the sitting:
`51e144d5a07a082a9ef8dccf3eaf21ebf40ca5268bbeff2238374bc799ac5568`.
The retained frozen binary is
`artifacts/reports/cvvdp/vqm_bench-final`. The source tree includes the prior
uncommitted audit and this implementation; HEAD alone cannot reproduce it.

Vship is read-only reference commit
`91d03ec2f2e03e4817b64655a5ea58ab7ab3d3ba`. Its library was built from a copy
under `build/` using the pinned checkout's bundled SPIR-V, as upstream
`make build` specifies. No peer implementation was copied into VQM.
`bench/vship_bench.cc` uses only the public C API, with explicit
`standard_4k`, no display resize and no distortion-map readback.

The display is 200 cd/m² peak, contrast 1000, ambient 250 lux, diffuse
reflectivity 0.005, exposure 1 and 75.402449345163049 pixels/degree. Input
geometry does not select or resize this display. Device matching found a
unique RTX 5070 name at peer device 0; Vship's API exposes no UUID, so this
is a name match rather than independent physical identity proof.

CPU reference work and software Vulkan validation were paused during the
sitting. All ten logs' checksums match their ten warm-up-plus-timed image
calls per pair. Each candidate run's score and stage table was read and
checked. Stage tables combine the two geometries, so they cannot be used as
per-resolution stage timings. All raw logs, CSVs, command lines and full
environment fingerprints remain under
`artifacts/reports/cvvdp/peer-benchmark/`; the original scores and input
hashes are under `artifacts/reports/cvvdp/peer-original/`.

[Compact supporting data](../measurements/cvvdp-vship-still-results.json)
retains every round's timings, scores, checksums, displayed stage tables,
input hashes, binary hashes, protocol and limits.

## Reproduction

Build the core and optional pinned peer through the maintained workflows:

```sh
cmake --preset release
cmake --build --preset release --parallel 8
python3 tools/fetch_references.py --only Vship
python3 tools/build_references.py --only vship-vulkan --jobs 8
python3 bench/make_bench_pairs.py --synthetic-only --sizes 1920x1080,3840x2160
```

Create a two-line tab-separated pair list for the generated `orig.png` and
`dist.png` files in `.cache/bench/pairs/synthetic-{1920x1080,3840x2160}/`.
Run the two arms with the streaming protocol:

```sh
python3 bench/run_inmemory.py --metric cvvdp --mode latency \
  --pairs-file artifacts/reports/cvvdp/peer-benchmark/pairs.tsv \
  --only ours-streaming,vship-vulkan --rounds 5 --iterations 8 --warmup 2 \
  --out-dir artifacts/reports/cvvdp/peer-benchmark
```

`bench/benchmark.py --metric cvvdp --references vship-vulkan` is a convenience
workflow for one selected pair/size. Explicit `--iterations 8 --warmup 2`
selects the upload-path protocol. `--metric all` in the general runner keeps
its existing SSIMULACRA 2/Butteraugli scope. CVVDP comparisons require explicit
selection and latency mode; ordered video uses `vqm_bench --cvvdp-video`.

## Tooling qualification

The optional peer driver passes C++ syntax checks, clang-format 23 and
clang-tidy 23. Its shared item-description and timed-call helpers preserve
existing SSIMULACRA 2 and Butteraugli contracts while removing duplicated
latency/batch code. The maintained benchmark helper suite passes all 34 tests.

The two existing Python benchmark runners were formatted with required YAPF.
AST comparison against the preserved pre-change source plus the explicit
CVVDP edits verified no formatting-induced semantic changes. YAPF reports no
diff. Ruff reports the same 214 existing diagnostics before and after these
runner edits, with zero new diagnostics; existing annotation, docstring,
complexity and import-order findings were not broadened into this metric task.
The new CVVDP tool tests pass Ruff and YAPF cleanly.
