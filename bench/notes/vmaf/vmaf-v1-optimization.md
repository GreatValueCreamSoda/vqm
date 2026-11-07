# VMAF v1 optimization measurements

These are the first optimization pass. See
[`vmaf-v1-pipeline-optimization.md`](vmaf-v1-pipeline-optimization.md) for the
later histogram, pooling and parallel eigensolver measurements.

All measurements here use the standard SDR v1.0.16 3H model on an RTX 5070,
driver 615.78.08. Arms were interleaved five times using the repository's
timed benchmark, 10-bit YUV420 inputs, 20 iterations / 3 warm-ups / 2 resident
slots for the resident arms and 8 / 2 for streaming. Both input images are
prepared on every call. Resident inputs reuse pixels and approximate a
unique-frame stream without PCIe transfers.

## Combined result

Final five-run interleaved comparison against original commit `33e8da0`:

| workload | original ms/pair | optimized ms/pair | throughput original → optimized |
| --- | ---: | ---: | ---: |
| 1080p resident | 7.358878 (7.312594–7.377928) | 2.098369 (2.086050–2.136364) | 136 → 477 pairs/s |
| 4K resident | 14.250613 (14.225368–14.267053) | 8.074348 (8.015098–8.086097) | 70 → 124 pairs/s |
| 1080p streaming | 7.917940 (7.859876–7.953315) | 2.644517 (2.624622–2.652497) | 126 → 378 pairs/s |
| 4K streaming | 16.420222 (16.407342–16.496970) | 10.235741 (10.211261–10.342340) | 61 → 98 pairs/s |

At 1080p resident, SpEED moved 5.987845 → 0.903675 ms, CAMBI
0.880560 → 0.789954 ms and ADM3 0.137722 → 0.119102 ms. The score/checksum
changed 95.973505 / 2207.390606 → 95.973465 / 2207.389685. At 4K the
score/checksum changed 96.065097 / 2209.497237 → 96.065015 / 2209.495343.
The covariance reduction is responsible for the small score differences.

V0 uses the same fused wavelet implementation. A separate interleaved
regression check in the same sitting measured v0 1080p resident
0.568082 → 0.568357 ms and streaming 1.122634 → 1.127198 ms, with overlapping
timing ranges and identical scores/checksums.

The combined build passed the full 4,292-pair CID22 comparison with verified
input hashes. Versus the pre-optimization VQM scores, maximum/mean/p99 drift
was 0.00006355 / 0.000005446 / 0.00002941. Against pinned libvmaf, maximum/mean
absolute error was 0.004653 / 0.001699. KRCC/SRCC/PCC were
0.56395148 / 0.76184503 / 0.71779591 versus the reference's
0.56396690 / 0.76186042 / 0.71780187. This is a still-image, zero-motion
corpus check; the GPU suite separately exercises moving sequences.

Exploratory GPU timestamps on the original 1080p build apportioned about
5.80 ms to SpEED's 25×25 eigensolvers, 0.51 ms to CAMBI contrast and 0.30 ms
to CAMBI pooling. At 4K, SpEED eigen and CAMBI contrast each took about 5.5 ms.
These extra timestamp ranges are diagnostic, not the source of speed claims.

## Cooperative SpEED eigensolver

The original ordered Jacobi solver performs each rotation on one invocation,
repeatedly accessing its matrices in device memory. The replacement retains
the rotation order and failure guard, stages matrices in workgroup memory,
and uses 32 cooperating invocations to update the matrix and eigenvectors.
It follows the shared-memory and workgroup synchronization patterns used by
our VIF and motion kernels. Five interleaved baseline → candidate runs:

| workload | pair ms | SpEED ms |
| --- | ---: | ---: |
| 1080p resident | 7.389831 (7.379636–7.423366) → 2.679794 (2.662774–2.708859) | 5.988610 (5.987885–5.988789) → 1.361709 (1.361619–1.362000) |
| 4K resident | 14.235926 (14.229522–14.266131) → 9.858694 (9.821964–9.887633) | 6.137499 (6.135240–6.138006) → 1.753058 (1.752070–1.753344) |
| 1080p streaming | 7.934868 → 3.223391 | 5.988784 → 1.361972 |

The 1080p score/checksum was 95.973505 / 2207.390606 and the 4K value was
96.065097 / 2209.497237 in both arms. All 121 GPU tests passed, including
near-singular cases. Across all 4,292 CID22 pairs, VQM scores matched the
baseline to eight decimal places with verified input hashes.

## Stop unchanged Jacobi sweeps

A complete sweep that makes no rotations leaves the matrix unchanged;
subsequent sweeps would therefore make the same decisions. Stop after that
sweep, retaining the final singular/nonconvergence checks. A uniform barrier
ensures every invocation reads the convergence flag before it is reset.
On top of the cooperative solver, five interleaved runs measured:

| workload | pair ms | SpEED ms |
| --- | ---: | ---: |
| 1080p resident | 2.675714 (2.665199–2.702938) → 2.295865 (2.286015–2.306630) | 1.361522 (1.361363–1.361877) → 0.984720 (0.984691–0.984834) |
| 4K resident | 9.847024 (9.820135–9.995483) → 9.503774 (9.415077–9.514741) | 1.752456 (1.751592–1.752851) → 1.374936 (1.374739–1.376077) |
| 1080p streaming | 3.231780 → 2.848802 | 1.362452 → 0.985660 |

All 121 GPU tests and all 4,292 CID22 pairs passed with the same printed
scores/checksums as before, including the near-singular fixtures.

## Rejected symmetric covariance schedule

Computing only one triangle of the symmetric covariance matrix and mirroring
the values preserves each element's accumulation order. On the cooperative
solver baseline it did not reduce the SpEED stage: five interleaved resident
runs measured 1080p 1.361275 ms (1.361234–1.361974) → 1.361590 ms
(1.361472–1.361763) and 4K 1.752742 ms (1.751650–1.753410) → 1.751667 ms
(1.751414–1.752002). Scores/checksums were identical at printed precision,
and all 121 GPU tests passed, but the candidate missed the 2% gate.

Additional diagnostic timestamps gave covariance alone 0.100936 → 0.100989 ms
at 1080p and 0.481085 → 0.480627 ms at 4K. The change halves dispatched
matrix entries but leaves the serial sample loop per entry unchanged; these
timings do not establish the underlying GPU stall reason. It was reverted.

## Parallel covariance accumulation

Instead of assigning the entire sample region to one invocation per matrix
element, assign a 64-invocation workgroup to each mean/covariance element and
reduce its partial sums using our existing shared-reduction pattern. The
strided loop has a guarded final increment to prevent unsigned wraparound.
This changes FP32 summation order, so feature, singularity and corpus checks
are required. Five interleaved baseline → parallel runs:

| workload | pair ms | SpEED ms |
| --- | ---: | ---: |
| 1080p resident | 2.203079 (2.198499–2.244693) → 2.138214 (2.117755–2.142889) | 0.984979 (0.984930–0.985062) → 0.904234 (0.904098–0.904342) |
| 4K resident | 8.617232 (8.596407–8.631771) → 8.204333 (8.168859–8.231073) | 1.374227 (1.373563–1.374346) → 0.955610 (0.955322–0.955846) |
| 1080p streaming | 2.760290 → 2.680436 | 0.985256 → 0.904284 |

The 1080p resident score/checksum changed 95.973505 / 2207.390606 →
95.973465 / 2207.389685; at 4K 96.065097 / 2209.497237 → 96.065015 /
2209.495343. Across all 4,292 CID22 pairs, maximum/mean/p99 score drift was
0.00006355 / 0.000005446 / 0.00002941. All 122 GPU tests passed, including
near-singular and low-entropy cases.

## Shared CAMBI contrast tile

The CAMBI contrast kernel stages its input window and mask in a packed shared
tile, following our v0 VIF shared-halo pattern. The worst tile uses 9,216 bytes,
below the Vulkan minimum supported by VQM. Five interleaved baseline → tile
runs, independent of the SpEED change:

| workload | pair ms | CAMBI ms |
| --- | ---: | ---: |
| 1080p resident | 7.385250 (7.334876–7.413641) → 7.303272 (7.262782–7.321201) | 0.879982 (0.879318–0.880514) → 0.794485 (0.793614–0.795427) |
| 4K resident | 14.238949 (14.225494–14.281164) → 13.374223 (13.351093–13.430147) | 6.539432 (6.531726–6.546011) → 5.681854 (5.672067–5.688258) |
| 1080p streaming | 7.924747 → 7.860563 | 0.879888 → 0.794788 |

The 1080p resident score/checksum was 95.973505 / 2207.390606, 4K was
96.065097 / 2209.497237, and streaming checksum was 959.735046 in both arms.
All 4,292 CID22 scores matched baseline to eight decimal places. A new
per-pixel reference fixture covers the 33-pixel window and odd image edges.

## Rejected local pooling histograms

Local shared-memory histogram aggregation reduced CAMBI at 1080p from
0.880008 ms (0.879483–0.880446) to 0.859171 ms (0.859074–0.859826), but
at 4K from 6.545053 ms (6.544139–6.546365) to only 6.473853 ms
(6.473112–6.475357), below the 2% gate. Streaming was 0.880084 → 0.859888 ms.
Scores and checksums matched to printed precision; 120 GPU tests passed. The
experiment was reverted.

## Reuse the v0 fused ADM wavelet

Build the same fused DWT/decoupling shader used by v0 with ADM3's enhancement
gain limit of 1, replacing the two wavelet dispatches and decoupling dispatch
with one per scale. V1's downstream CSF and masking stay the same; the default
v0 shader binary was verified byte-identical after introducing the parameter.
Five interleaved baseline → fused measurements:

| workload | pair ms | ADM3 ms |
| --- | ---: | ---: |
| 1080p resident | 2.225172 (2.211617–2.240741) → 2.185137 (2.167257–2.201707) | 0.137669 (0.137346–0.137870) → 0.119101 (0.119058–0.119147) |
| 4K resident | 8.628218 (8.608554–8.660828) → 8.476607 (8.447112–8.514876) | 0.706618 (0.706309–0.707064) → 0.573219 (0.572949–0.573266) |
| 1080p streaming | 2.757723 → 2.720694 | 0.138036 → 0.119148 |
| 4K streaming | 10.817440 → 10.664283 | 0.706728 → 0.573120 |

Scores/checksums matched to printed precision in every run, and all 122 GPU
tests passed. All 4,292 CID22 scores matched baseline to eight decimal places,
with input payload hashes checked against the original manifest.
