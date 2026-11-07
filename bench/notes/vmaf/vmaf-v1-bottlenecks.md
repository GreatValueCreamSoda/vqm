# VMAF v1 bottleneck investigation

For current workgroup counts, solver experiments and fresh 1080p measurements,
see the [2026-11-05 parallelism investigation](vmaf-speed-parallelism.md).

For the resulting pooling and SpEED changes, see
[the follow-up measurements](vmaf-v1-pooling-parallelism.md).

Measurements on 2026-11-02 used the RTX 5070 (48 SMs), driver 615.78.08,
standard SDR v1.0.16 3H model and commit `1824793`. Both source images are
prepared on each resident call; inputs are preuploaded and pixels repeat, so
this measures hot GPU memory and excludes upload and decode. Motion becomes
zero after the first call. The 1080p and 4K pairs are 10-bit YUV420; ducks is
one 8-bit 1080p source frame distorted with x264 CRF 28, not Netflix's complete
video benchmark. Pair contents and hashes are recorded in
[`vmaf-v1-algorithm-optimization.md`](vmaf-v1-algorithm-optimization.md) and
`artifacts/reports/vmaf-v1-chart/provenance.json`.

## Where time goes

Five interleaved runs per arm used `--iterations 20 --warmup 3 --resident 2`.
The first arm was the unmodified benchmark, and the second was an otherwise
identical temporary build with extra GPU timestamp queries. The diagnostic
queries perturb execution and do not yield independent additive optimization
savings. Scores and checksums were identical between arms. Medians and
min–max spans below are milliseconds per pair:

| workload | normal pair | normal CAMBI | normal SpEED | diagnostic contrast | diagnostic radix selection | diagnostic eigen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p | 0.9854 (0.9734–0.9993) | 0.4638 (0.4631–0.4646) | 0.1563 (0.1562–0.1566) | 0.2313 (0.2309–0.2320) | 0.1146 (0.1146–0.1151) | 0.0935 (0.0935–0.0937) |
| 4K | 3.4610 (3.4184–3.5193) | 1.9333 (1.9329–1.9336) | 0.2301 (0.2292–0.2302) | 1.1595 (1.1594–1.1597) | 0.1432 (0.1430–0.1435) | 0.1010 (0.1010–0.1011) |
| ducks 1080p | 0.8293 (0.8190–0.8340) | 0.2936 (0.2932–0.2939) | 0.1780 (0.1775–0.1781) | 0.0583 (0.0583–0.0584) | 0.1152 (0.1151–0.1153) | 0.1151 (0.1151–0.1151) |

Diagnostic pair medians were 0.9959, 3.4956 and 0.8430 ms respectively;
the extra timestamp overhead and ordinary timing spread prevent treating
the substage numbers as exact fractions of the normal arm. CAMBI also spends
0.0614/0.4658/0.0651 ms on full-resolution preparation, and 0.0246/0.0559/
0.0250 ms on final pooling sums. The detailed logs, per-run stage tables,
diagnostic patch and Nsight capture are in the ignored local directory
`artifacts/reports/vmaf-v1-bottleneck/`.

The GPU-timed outer stages sum to about 0.888 ms in the normal 1080p arm,
versus 0.985 ms elapsed; they include the GPU dispatches and dependencies,
not just active arithmetic. Host score conversion takes about 0.0009 ms.
They do not directly attribute the remaining elapsed time to any one cause.

## What the implementation tells us

- **SpEED leaves most SMs without work during its eigensolver dispatch.**
  `VmafSpeedStats::Record` dispatches four workgroups, one each for reference
  and distorted U and V covariance matrices. On the 48-SM test GPU, no more
  than four SMs can run those groups at once. Within each group, the 25×25
  matrix admits limited parallel work; portions of the QR and Cholesky
  algorithms run on lane 0 and synchronize the workgroup. This is a structural
  bound on that dispatch, not a measured average SM occupancy for the pair.
- **CAMBI contrast has many groups, but also workgroup synchronization and
  shared-memory atomics.** The first scale launches 8,100 groups at 1080p and
  32,400 at 4K, so a shortage of groups cannot explain the large contrast
  dispatch. Each active 256-invocation group computes 16×4 outputs with four
  512-bin shared histograms; four lanes calculate each output column and the
  group executes 34 barriers, including 30 between column updates. Many lanes
  cooperatively build and update the histograms with shared atomics. This
  structure makes serial work, barriers and atomic contention plausible
  bottlenecks; their individual costs have not been measured. A fully empty
  group exits early, which explains why the work is content dependent.
- **CAMBI pooling has a long chain of small dispatches.** Five scales each
  record 11 dispatches, including four radix histogram/selection rounds.
  The selectors use one workgroup and lane 0 scans the histogram. The measured
  radix stage covers histogram dispatches, selectors and GPU dependencies;
  it does not identify how much time is attributable to each.
- **4K CAMBI has more work per pixel.** The contrast window grows from
  17×17 to 33×33. This increases each active tile's halo and histogram setup
  in addition to quadrupling pixel count; it is not evidence by itself of
  external memory saturation.

The single-worker benchmark holds a host mutex only around `vkQueueSubmit`;
it releases it before `vkWaitForFences`. In a 500-iteration Nsight Systems
capture of the ducks pair, the 505 waits (including warm-up) had a 0.756 ms
median, while the GPU command-buffer workload had a 0.736 ms median. This is
consistent with the host waiting for the GPU, not evidence that a host mutex
is the dominant cost. The profiling layer changes execution time, so these
numbers are timeline evidence, not throughput figures. The trace records
62,115 pipeline binds for 505 calls, or 123 per call; 80 are CAMBI dispatches.
The Vulkan layer exposes one GPU workload per command buffer rather than a
per-dispatch timing. CPU scheduling and backtraces were unavailable because
of local profiling permissions, so the trace cannot rule out all lock costs.

## What remains unproven

We cannot currently say whether CAMBI is limited primarily by DRAM bandwidth,
cache/shared-memory behavior, atomic contention, barriers, instruction issue,
or their interaction. GPU timestamps and group counts do not measure any of
those stall reasons. Nsight Systems captured GPU metrics, but its supported
kernel-level handoff requires CUDA kernel events, which are absent for these
Vulkan shaders. Previous Nsight Compute attempts reported no Vulkan kernels;
no per-kernel occupancy, memory bandwidth, or stall breakdown is claimed.
The earlier utilization counters describe the SSIMULACRA 2 workload.
[SSIMULACRA 2 utilization measurements](../system/vship-notes.md#measured-utilization)

The comparison with Netflix's CPU chart is also not apples-to-apples: its
exact distortion, v0 feature mode and model are not established by the chart.
Our own same-sitting ducks comparison in
[`vmaf-v1-algorithm-optimization.md`](vmaf-v1-algorithm-optimization.md)
measured v0 VIF at 0.309 ms and v1 SpEED plus CAMBI at 0.471 ms; this
replacement accounts for much of the observed 0.254 ms pair-time gap, with
other stage and overhead differences. The CPU reference's sliding histograms
and SIMD implementation map differently from this Vulkan implementation.

Promising work is therefore to reduce the dependent pooling passes and to
increase useful work in the four-matrix SpEED solver, while separately
redesigning CAMBI histogram reuse for active images. Benchmark each independently
on multiple contents, with correctness checks: the ducks and 10-bit fixtures
have very different contrast costs.
