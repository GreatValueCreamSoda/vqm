# External Vulkan-buffer measurement and rejected metadata experiment

On 2026-11-02 an RTX 5070 with driver 615.78.08 scored RGB8 image pairs with
`--vulkan-buffers`. This arm creates two caller-owned device-local storage
buffers on the same Vulkan device and queue used by the metric, uploads the
decoded pixels once through staging before warm-up, and then reads both buffers
directly on every timed call. Both images are prepared on every call; their
pixels and GPU addresses repeat. Device setup, decoding and the one-time
upload are outside the timer. This is a hot-memory, single-pipeline benchmark,
not a measurement over a sequence of distinct video frames.

The experiment attempted to validate and resolve each external buffer once
per submission, then reuse its descriptor binding and plane layout during
recording. It still validated each submission and rewrote the external
descriptors, preserving safety if the caller recycled a buffer handle. The
baseline was commit `0199050`; the same external-buffer harness was applied to
both builds. The candidate changed only metadata resolution. Scores and
checksums matched in every run.

The tables show per-pair untimed milliseconds: the per-call elapsed time less
the reported GPU and score stages. Each value is the median of five alternating
baseline/candidate runs, with the observed range in parentheses. The value for
each run is calculated from the six-decimal stage CSV. The initial protocol
used `--iterations 20 --warmup 3` for every size:

| Metric and size | Baseline | Candidate |
| --- | ---: | ---: |
| SSIMULACRA 2, 32x32 | 0.064570 (0.062046–0.067200) | 0.066140 (0.063941–0.067014) |
| SSIMULACRA 2, 1080p | 0.086568 (0.084402–0.087900) | 0.085482 (0.082344–0.088374) |
| SSIMULACRA 2, 4K | 0.096819 (0.087929–0.113166) | 0.086973 (0.084445–0.094407) |
| Butteraugli, 32x32 | 0.051908 (0.050785–0.071495) | 0.051527 (0.050417–0.052701) |
| Butteraugli, 1080p | 0.172801 (0.134892–0.213955) | 0.110885 (0.102362–0.138600) |
| Butteraugli, 4K | 0.335225 (0.257878–0.371650) | 0.328980 (0.285938–0.391225) |

Because the first result varied widely and differed across sizes, a second
interleaved sitting used 200 iterations at 32x32, 100 at 1080p and 20 at 4K,
each with three warm-ups:

| Metric and size | Baseline | Candidate |
| --- | ---: | ---: |
| SSIMULACRA 2, 32x32 | 0.062690 (0.061763–0.065329) | 0.063445 (0.061372–0.066092) |
| SSIMULACRA 2, 1080p | 0.085692 (0.083281–0.087574) | 0.086643 (0.084653–0.087039) |
| SSIMULACRA 2, 4K | 0.092387 (0.086564–0.112860) | 0.099306 (0.085717–0.110898) |
| Butteraugli, 32x32 | 0.049376 (0.049070–0.049669) | 0.049770 (0.048977–0.051048) |
| Butteraugli, 1080p | 0.151507 (0.135670–0.189187) | 0.181837 (0.145797–0.197848) |
| Butteraugli, 4K | 0.323002 (0.299651–0.364328) | 0.349624 (0.321745–0.389363) |

The apparent first-sitting gain did not repeat, and the second sitting's
medians did not improve at any tested size. This did not meet the 2% speed
gate reliably, so the metadata change was reverted. The benchmark arm remains.
The untimed figure also includes submission, waits and host scheduling, so it
does not isolate metadata processing.

In the initial sitting the per-call scores were 93.175057, 91.716994 and
92.437150 for SSIMULACRA 2 at 32x32, 1080p and 4K; their 23-call checksums
were 2143.026310, 2109.490866 and 2126.054439. Butteraugli scores were
3.751055, 7.279592 and 7.279592; their checksums were 86.274254, 167.430606
and 167.430606. In the follow-up the 203-call checksums at 32x32 were
18914.536563 and 761.464069; the 103-call checksums at 1080p were 9446.850400
and 749.797931. Per-call scores and 4K checksums were unchanged.

For context, in the second sitting SSIMULACRA 2's 4K baseline median core
latency was 1.201949 ms (1.196584–1.206584) with caller-owned Vulkan buffers,
1.202329 ms (1.196529–1.204494) with resident images, and 3.418055 ms
(3.416080–3.420260) streaming. This is a comparison on the same synthetic
pair in that sitting, not a separate claim about general application
throughput. Butteraugli's resident arm reuses its reference while the external
arm rebuilds it, so those two arms do not have identical workloads.
