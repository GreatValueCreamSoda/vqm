# VMAF research notes

These versioned notes preserve the reasoning, numerical examples, platform
constraints and reproduction instructions from earlier investigations. They
were written at different stages of the work, often before GPU measurements or
the final integration. Statements about proposed experiments and unchanged
production code describe that historical stage. Current implementation details
can differ from the snapshots reviewed here.

Use the final measured reports for the retained implementation, performance
results and completed validation. The research notes explain how those results
were investigated and retain details that would otherwise exist only in local
run output.

| research note | original report-relative path | final measured report |
| --- | --- | --- |
| [VIF and ADM FP16](fp16-vif-adm.md) | `vmaf-fp16/experiments/vif-adm-review.md` | [FP16 practicality](../vmaf-fp16.md) |
| [Platform and motion FP16](fp16-platform-motion.md) | `vmaf-fp16/motion/platform-and-motion-review.md` | [FP16 practicality](../vmaf-fp16.md) |
| [SpEED and CAMBI precision](fp16-speed-cambi.md) | `vmaf-fp16/v1-precision-review.md` | [FP16 practicality](../vmaf-fp16.md) |
| [SDR synchronization](sdr-synchronization.md) | `vmaf-speed-investigation/experiments/sdr-caller-barrier-review.md` | [VMAF speed investigation](../vmaf-speed-investigation.md) |
| [Eigensolver stress provenance](eigensolver-stress.md) | `vmaf-v1-pooling-parallelism/stress-provenance/README.md` | [Pooling and parallelism](../vmaf-v1-pooling-parallelism.md) |

The original paths in the table are relative to the research report tree that
was captured from the build directory. Complete local records, including
patches, driver sources, datasets and raw results, are preserved separately
under `artifacts/reports/` using those same
relative paths. Those generated or historical records are ignored by Git; the
notes and final reports remain versioned source documentation.

Historical command blocks retain their original paths, commit identifiers and
temporary-directory assumptions. Reproduce them against the named snapshot or
adapt the paths deliberately. Research records are not inputs to the normal
library, CLI or test builds.

## Cooperative-matrix investigation

The [integrated investigation](../vmaf-cooperative-matrix.md) records the
FP16-input/FP32-accumulation decision and benchmark results. Supporting reviews:

- [Vulkan platform and precision contract](cooperative-platform.md).
- [SpEED covariance rounding, residual reconstruction and conditioning](cooperative-speed-numerics.md).
- [VIF, ADM, motion and other feature mappings](cooperative-feature-mapping.md).

Current reproduction tools live in
[`bench/experiments/cooperative_matrix/`](../../../experiments/cooperative_matrix/README.md).

## SpEED parallelism investigation

The [2026-11-05 SpEED investigation and 1080p results](../vmaf-speed-parallelism.md)
combine [dispatch/source review](speed-parallelism-review.md),
[pooling work distribution](speed-pooling-prototype.md),
[solver phase and subgroup alternatives](speed-solver-parallelism.md), and
[certified early QR selection](speed-early-qr.md). Reproduction tools are under
[`bench/experiments/speed_parallelism/`](../../../experiments/speed_parallelism/README.md).

## VMAF v1 remaining headroom

The [measured investigation](../vmaf-v1-headroom.md) combines
[contrast geometry and storage](v1-contrast-headroom.md),
[independent histogram-tail/radius review](v1-contrast-independent-review.md),
[pooling shortcuts](v1-pooling-headroom.md),
[storage and row synchronization](v1-storage-dispatch-headroom.md), and
[luma, ADM3 and motion review](v1-other-stages-headroom.md).
Current reproduction tools and the retained patch are under
[`bench/experiments/v1_headroom/`](../../../experiments/v1_headroom/README.md).
