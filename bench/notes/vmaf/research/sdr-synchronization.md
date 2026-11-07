# SDR conversion dependency review

> Historical research note. The [VMAF speed investigation](../vmaf-speed-investigation.md) records the final
> measured result and retained implementation. Current source may differ from
> the reviewed snapshots. Relative experiment files and commands below refer
> to the archived research directory
> `artifacts/reports/vmaf-speed-investigation/experiments/`.

The standalone patch removes only VmafSdrConverter::Record's trailing full
compute-to-compute dependency. It does not change the shader or arithmetic.
The caller publishes and synchronizes the source before Record, allocates
disjoint output planes, and establishes a dependency before reading or
overwriting those outputs. The internal header states that ownership.

Current call sites:

- VMAF v0 records reference/distortion conversions into disjoint luma planes.
  Fit/pad immediately calls ComputeBarrier before either resizer reads them.
  Native geometry instead reaches VmafVif::Record, whose initial
  ComputeBarrier makes both conversions visible before feature reads.
- VMAF v1 records disjoint Y/U/V output banks for the two images.
  VmafAdm3::Record reaches RecordAdmPyramid, whose first ComputeBarrier
  publishes both conversions. Later features remain ordered by existing
  dependencies.
- SdrHarness records one conversion and follows it with ComputeToHostBarrier
  for visible arenas, or RecordReadback's compute-to-transfer dependency and
  transfer-to-host dependency for staged arenas, then waits for submission.

Changing the bound source descriptor between recorded dispatches does not
change the preceding dispatch's binding; vkCmdBindDescriptorSets captures
the selected descriptor set in command-buffer order. Both sets have stable
binding 3 throughout the in-flight submission. No descriptor update is moved.

Root should gate this separately on RGB/RGBA resident and streaming workloads
for both VMAF versions, plus fit/pad for v0. Compare exact scores/checksums and
run sync validation. The benefit is empirical; removing two dependencies does
not establish concurrent execution or guarantee a speedup.
