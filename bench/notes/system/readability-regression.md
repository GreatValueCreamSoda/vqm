# Readability refactor regression check

Baseline `f46d8d7`, NVIDIA GeForce RTX 5070, driver 615.78.08. The changes
simplify CLI and API control flow, Vulkan input and memory planning, and
VMAF result layout names. They do not intentionally change metric arithmetic.

Each result is the median and min–max pair time in milliseconds over five
interleaved baseline/candidate runs on 8-bit YUV420 fixtures. Resident uses
`--resident 2 --iterations 20 --warmup 3`; streaming uses `--iterations 8
--warmup 2`. SSIMULACRA 2 prepares both images for every resident call;
Butteraugli caches its resident reference. VMAF benchmark modes follow the
same flags. Pixels are reused, so these resident measurements represent a
hot working set.

| Metric | Workload | Baseline ms (min–max) | Refactor ms (min–max) |
| --- | --- | ---: | ---: |
| ssimulacra2 | 1080-resident | 0.4904 (0.4902–0.4995) | 0.4912 (0.4903–0.4923) |
| ssimulacra2 | 1080-streaming | 1.0420 (1.0351–1.0462) | 1.0471 (1.0360–1.0479) |
| ssimulacra2 | 4k-resident | 1.8629 (1.8572–1.8751) | 1.8584 (1.8526–1.8603) |
| ssimulacra2 | 4k-streaming | 4.0633 (4.0611–4.0796) | 4.0673 (4.0646–4.0845) |
| butteraugli | 1080-resident | 2.5414 (2.5406–2.5513) | 2.5410 (2.5334–2.5513) |
| butteraugli | 1080-streaming | 4.3848 (4.3647–4.4087) | 4.4135 (4.3866–4.5287) |
| butteraugli | 4k-resident | 10.9964 (10.9881–11.0295) | 10.9935 (10.9510–11.0281) |
| butteraugli | 4k-streaming | 18.4948 (18.4659–18.4991) | 18.4995 (18.4520–18.5107) |
| vmaf-v0 | 1080-resident | 0.5686 (0.5675–0.5693) | 0.5684 (0.5673–0.5692) |
| vmaf-v0 | 1080-streaming | 1.1209 (1.1190–1.1294) | 1.1205 (1.1204–1.1223) |
| vmaf-v0 | 4k-resident | 2.3669 (2.3524–2.3754) | 2.3655 (2.3569–2.3748) |
| vmaf-v0 | 4k-streaming | 4.5546 (4.5428–4.5623) | 4.5529 (4.5395–4.5641) |
| vmaf-v1 | 1080-resident | 0.7298 (0.7290–0.7372) | 0.7293 (0.7283–0.7374) |
| vmaf-v1 | 1080-streaming | 1.2901 (1.2784–1.2952) | 1.2859 (1.2781–1.2902) |
| vmaf-v1 | 4k-resident | 2.7239 (2.7061–2.7352) | 2.7381 (2.7113–2.7430) |
| vmaf-v1 | 4k-streaming | 4.9187 (4.9109–4.9356) | 4.9050 (4.8911–4.9390) |

Scores and checksums matched between arms for every workload. These tests
do not establish a performance improvement or material regression.
[Detailed runs, stage tables and input hashes](readability-results.json)
record the individual measurements.

The release and ASan/UBSan GPU suites passed 153 tests, all 12 CTest checks
passed, and the 300-pair SSIMULACRA 2 CID22 correlation gate passed at the
0.005 tolerance. The SSIMULACRA 2 and Butteraugli identity scores were
`100.00000000` and `0.00000000`, respectively.
