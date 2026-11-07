# GitHub readiness audit, 2026-11-06

This archived audit reviewed the 2026-11-06 working tree for correctness, clone/build
reproducibility and publication hygiene. It does not certify the absence of all
bugs. The starting revision was `a949b6177efb88890dd44860cc30dff45486a25d`,
with substantial pre-existing uncommitted reorganization, CVVDP/PSNR-HVS work,
validation and documentation changes. Those changes were preserved. No commits,
remote publication or CI workflows were created.

## Scope and method

Independent reviews covered shared Vulkan input/resource lifetime and public
APIs; the six metric implementations and numerical contracts; Python/shell
benchmark and validation ingestion; image/video decoding and CLI errors;
CMake installation/embedding; documentation, licenses, ignored files, candidate
source sizes and common secret patterns. Testing remained local.

Initial candidate-source hashes, status and tracked patch, complete check logs,
source-only build commands and raw correlation results are retained under
`artifacts/reports/repo-audit-2026-11-06/`. Maintained source was copied into
fresh temporary source trees without `.git`, `.reference`, `.cache`, build
outputs, local artifacts or the virtual environment. The full test build used
the existing managed Python environment as its interpreter; the standalone core
build disabled Python, PNG and JPEG discovery.

Hardware: NVIDIA GeForce RTX 5070, driver 615.78.08; Linux, GCC 16.2.1,
CMake/Ninja Release with warnings as errors and a separate ASan/UBSan Debug
build. Python 3.12.14 uses the preserved `.venv/`. CPU reference/model inputs
remain pinned, external and read-only.

## Confirmed issues repaired

1. **Disabled CVVDP streaming could overwrite unreserved storage.** Host
   Image/ImageView submissions reached shared upload code even when source
   regions had not been reserved. The shared streaming destination now rejects
   the disabled path before calculating an address or copying pixels. A GPU
   regression covers still/sequence calls, direct/staged allocation and
   subsequent resident reuse.
2. **Validation and video benchmarks accepted incomplete frame evidence.** A
   matching reported clip frame count and finite final score could conceal
   omitted/reordered frame results, malformed VMAF features, inconsistent
   pooling or invalid timings. Both consumers now share a standard-library
   validator for sample/metric identity, exact ordered frame coverage, finite
   numbers, model feature counts and arithmetic means. CVVDP explicitly requires
   its clip-only result with an empty frame array. Six local Python regressions
   cover the contract and its ingestion path. Existing qualified JSONL records
   for all 390 clips in each of five metrics remain accepted.
3. **An omitted Y4M chroma tag used the wrong location.** The decoder selected
   left-sited `420` rather than the format default, centered `420jpeg`.
   The omitted-tag default now matches MJPEGtools' original format declaration;
   explicit `C420`, `C420mpeg2` and `C420jpeg` retain their existing meanings.
   CPU reader tests cover SequenceReader and LoadImage. See
   [the input contract](../reference.md#input-formats) and the
   [original format header](https://svn.code.sf.net/p/mjpeg/Code/trunk/mjpeg_play/utils/yuv4mpeg.h).
4. **PSNR-HVS had no tests in the maintained C++ suite.** Eight registered
   tests now cover independent closed-form DC and frequency-4 masking oracles,
   8/10/12-bit 420/422/444 combinations, exact zero-error identity, uncovered
   borders, symmetry, single/two-stage reductions, staged/resident/mapped input
   paths, token/handle/in-flight contracts, invalid inputs and delayed unified
   frame alignment. The oracles derive their expected errors analytically;
   they do not use the production DCT/table helpers. Upstream CSF constants
   retain their provenance and notice. These tests do not qualify corpus ranking.
5. **Sample-aspect documentation overstated reader validation.** The temporal
   guide now distinguishes coded pixel geometry in the general sequence reader
   from the hosted qualification preparation's explicit square-pixel contract.
6. **The local validation recipe could conceal a missing layer.** The loader
   can continue successfully without an environment-requested validation layer.
   The contributing recipe now captures loader errors as well as VUID and
   synchronization errors, so that this cannot be mistaken for a passing check.
7. **An optional libvmaf comparison could accept missing upstream frames.**
   Its `zip()` compared only the common prefix and checked the adapter's frame
   count. It now checks the upstream CLI frame count before comparing values.

## Publication checks

All reviewed local Markdown links and heading anchors resolved. All 16 public
headers independently compiled under C++20 with warnings as errors. Upstream
license notices are present and installed; Cloudinary, JPEG XL, VMAF and CVVDP
notices compared byte-for-byte with the pinned originals, and the Xiph notice
was retained.

The initial 588 candidate files totalled about 9.9 MB; the largest was about
538 KB. The largest of 2,305 scanned reachable historical blobs was about
1.5 MB. Common private-key/token/credential-URL scans found only an intentional
credential-rejection fixture. These pattern scans do not establish that every
possible secret is absent. Historical reference/source snapshots in Git history
were not rewritten. No current CI workflow files exist.

## Remaining qualification and style limits

PSNR-HVS remains outside the five-metric hosted ranking qualification matrix.
Its independent component/API tests and prior smoke comparisons do not establish
corpus-wide reference ranking agreement. Hosted SDR qualification does not
replace independent HDR, alternate-display/model or extreme-rate oracles.

The initial repository-wide user-default Ruff sweep reported **2,478 existing
findings**: predominantly missing annotations (1,069), quotes (648) and
docstrings (392), plus nesting (187), complexity (39), statement count (36),
imports (40) and smaller categories. All eight loop-closure warnings were
reviewed: their callbacks are consumed within their loop iteration, so no
current late-binding failure was demonstrated. No undefined-name failures were
reported. The initial YAPF comparison found differences in **51 files**;
formatting the changed legacy oracle test leaves **50 files** outstanding.
This is outstanding Python maintainability/formatting debt, not a clean global
Python quality gate. The validator, its consumers and associated updated mock
tests pass Ruff and YAPF individually. The legacy libvmaf oracle test retains
185 pre-existing Ruff findings; its added length assertion introduces none and
the file passes YAPF. No repository-wide style rewrite or lint suppression was
introduced.

No performance improvement is claimed. Fixes affect invalid-input validation
and input metadata interpretation; metric arithmetic and shader kernels were
not changed. The interleaved performance protocol was not run.

The Khronos validation layer is unavailable in this environment. A loader-debug
probe confirmed `VK_LAYER_KHRONOS_validation` was not found, despite a successful
device probe exit code. Vulkan API/synchronization validation is therefore
unverified in this audit; ordinary required-GPU and sanitizer results remain
separate checks. The raw loader diagnostic is retained with the other logs.

## Final local results

Release CTest passed **27/27** with no Python reference skips. The required-GPU
suite and ASan/UBSan suite each passed **249 tests**, zero failed checks and no
skips. The fresh source-only full build also passed 27/27 CTest groups and all
249 required-GPU tests; five optional reference cases skipped because that
snapshot deliberately excluded external binaries. The original workspace ran
those cases, including all six libvmaf oracle tests after the count assertion.

The standalone core build, installed consumer and embedded consumer passed.
Formatting checked 258 maintained C++/GLSL files; production clang-tidy checked
52 translation units. All three generated model tables match their pins.
CLI identity values were SSIMULACRA 2 `100.00000000` and Butteraugli
`0.00000000`.

The final rebuilt CLI ran the complete default hosted qualification matrix
offline, using 390 complete 40-frame clips per metric and the frozen originals
and policies. No gate thresholds changed.

| Metric | SRCC | Kendall tau-b | Registered ranking | Old numerical |
| --- | ---: | ---: | --- | --- |
| SSIMULACRA 2 | 0.999971274 | 0.997495221 | pass | fail |
| Butteraugli | 1.000000000 | 1.000000000 | pass | pass |
| VMAF v0 | 0.999985030 | 0.999551653 | pass | pass |
| VMAF v1 | 0.999999588 | 0.999943238 | pass | pass |
| CVVDP | 1.000000000 | 1.000000000 | pass | pass |

SSIMULACRA 2 retains its old numerical failure: maximum absolute error
5.262514485 and MAE 0.295218340, versus old limits 0.01 and 0.001. Its frozen
registered ranking policy passes. This is not a numerical parity claim or a
threshold relaxation. Subjective comparison was not requested in this hosted
matrix. PSNR-HVS was not included in that five-metric qualification.

Exact check counts, per-metric numerical diagnostics, coverage, hashes and raw
report locations are recorded in
[the compact results](../../bench/notes/system/github-readiness-audit-20261106.json).

## Publication cleanup follow-up

The follow-up compared VQM's source layout, packaging, local tests, Vulkan
capabilities, media handling, benchmark reports, and public documentation
with the upstream projects below. It started from the local publication
commit `f9d5024` and its subsequent workspace cleanup.

| Area | Outcome | Upstream references |
| --- | --- | --- |
| CMake integration | The installed static core links into shared consumers. An embedded build leaves the parent's `BUILD_TESTING` alone. Package discovery rejects unsupported required components. The installed and embedded consumer checks cover all three. | [Highway integration](https://github.com/google/highway/blob/master/g3doc/faq.md), [CMake package helpers](https://cmake.org/cmake/help/v3.25/module/CMakePackageConfigHelpers.html) |
| Vulkan portability | SSIMULACRA 2 and Butteraugli validate fixed workgroup requirements before allocation. Butteraugli also checks its opsin shared-memory requirement. VQM keeps its documented per-heap budgets, borrowed-queue contracts, and manifest-driven shader validation. | [Vulkan shader requirements](https://docs.vulkan.org/refpages/latest/refpages/source/RuntimeSpirv.html), [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools#validator-tool), [Vulkan Memory Allocator budgets](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/staying_within_budget.html) |
| Local testing | Redundant benchmark, CLI, and constructor checks were removed. Independent score oracles and compatibility tests remain. CLI-only builds run the standard-library video tests even without NumPy. | [FFmpeg FATE](https://www.ffmpeg.org/fate.html), [Catch2 test selection](https://catch2-temp.readthedocs.io/en/latest/command-line.html) |
| Examples and measurements | The video guide has a runnable three-frame example. The VMAF developer guide owns current oracle commands. The benchmark guide links to per-round inputs, outputs, and timing definitions. | [Netflix VMAF FFmpeg guide](https://github.com/Netflix/vmaf/blob/master/resource/doc/ffmpeg.md), [libjxl benchmark guide](https://github.com/libjxl/libjxl/blob/main/doc/benchmarking.md), [Google Benchmark reporting](https://google.github.io/benchmark/user_guide.html#json) |
| Publication | The contributor guide asks for a minimal reproducible input and toolchain details. Source and installed packages carry the JPEG XL license and upstream patent grant. | [FFmpeg developer guide](https://www.ffmpeg.org/developer.html), [Pinned libjxl patent grant](https://github.com/libjxl/libjxl/blob/e47d2a2b5bb418f2947ce07f36cb048861db3820/PATENTS) |

### Open findings

- The Y4M reader ignores FFmpeg's `XCOLORRANGE` extension. It also accepts
  `Cmonop10` while FFmpeg writes `Cmono10`. The help and format guide document
  the supported tags and full-range override. A reader change should qualify
  all five established file-based metrics. [FFmpeg Y4M writer](https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/yuv4mpegenc.c)
- A truncated 44-byte Y4M file declaring 32768×32768 4:4:4 requests about 3 GiB
  before validating the frame payload. The CPU reproducer returned
  `std::bad_alloc` under a 256 MiB address-space limit. A resource-bound policy
  and tiny malformed-header test belong in the decoder suite.
  [OpenImageIO limits](https://openimageio.readthedocs.io/en/stable/imageioapi.html) ·
  [OpenEXR bounded validation](https://openexr.com/en/latest/bin/exrcheck.html)
- The C++ harness registers one CTest group for 249 cases. Listing and
  selecting a named case would help isolate GPU or process failures.
  [Catch2 CMake discovery](https://catch2-temp.readthedocs.io/en/latest/cmake-integration.html)
- A versioned release, a CLI version string, and a private reporting channel
  would make published builds easier to identify and support. The private
  route depends on the eventual repository's maintainer settings.
  [Highway versioning](https://github.com/google/highway/blob/master/README.md#versioning) ·
  [libjxl security policy](https://github.com/libjxl/libjxl/blob/main/SECURITY.md)

### Local verification

The Release build passed all 27 CTest groups, including the installed and
embedded consumer checks. On an RTX 5070 with driver 615.78.08, all 249 C++
tests passed with a GPU required and passed again with Khronos synchronization
validation enabled. The loader and validation log contained no matching errors.
All 127 generated shader modules passed Vulkan 1.1 SPIR-V validation.

Clang-format checked 259 maintained files, clang-tidy checked 52 production
translation units, and Ruff and YAPF checked 92 Python files. The VMAF reference
examples ran against a freshly built pinned CPU reference. An eight-pair
CID22 smoke run reused the original scores with matching input hashes and
zero score drift. The three-frame video example returned 100 for each frame
and the complete clip.
