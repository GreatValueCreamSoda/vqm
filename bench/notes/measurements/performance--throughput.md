> Historical generated measurement, preserved during the 2026-11-05 build cleanup.
> Original report: `build/reports/performance/throughput.md`. Raw data now lives under
> `artifacts/reports/performance/`. Paths and numbers below retain their original provenance.

# Throughput (FPS)

generated: 2026-10-27T01:29:08.473424+00:00
protocol: --throughput-seconds 3, median of 3 runs, one-thread and N-thread runs alternating; decode is outside the window; one process per implementation

The threads column is per size (left to right): a CPU arm is capped where the frame is large enough that the workers would not fit in this machine's memory.

| implementation | metric | threads | 1920x1080 FPS | 3840x2160 FPS | 1920x1080 FPS @1 | 3840x2160 FPS @1 | 1920x1080 ceiling | 3840x2160 ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours-resident | ssimu2 | 3/3 | 3575 | 854 | 2921 | 796 | 4084 (gpu) | 876 (gpu) |
| ours-streaming | ssimu2 | 3/3 | 1810 | 456 | 1113 | 289 | 4075 (gpu) | 885 (gpu) |
| ours-half-planes | ssimu2 | 3/3 | 3693 | 928 | 2949 | 857 | 4078 (gpu) | 934 (gpu) |
| cloudinary | ssimu2 | 16/9 | 22 | 4 | 4 | 1 | 57 (ideal) | 8 (ideal) |
| libjxl | ssimu2 | 16/9 | 25 | 6 | 8 | 2 | 127 (ideal) | 17 (ideal) |
| fssimu2 | ssimu2 | 16/9 | 56 | 13 | 11 | 3 | 182 (ideal) | 25 (ideal) |
| vship-cuda | ssimu2 | 3/3 | 832 | 193 | 605 | 157 | n/a | n/a |
| vship-vulkan | ssimu2 | 3/3 | 468 | 103 | 407 | 98 | n/a | n/a |
| vship-fast-vulkan | ssimu2 | 3/3 | 585 | 152 | 399 | 95 | n/a | n/a |
| ours-resident | butteraugli | 3/3 | 421 | 93 | 407 | 92 | 428 (gpu) | 95 (gpu) |
| ours-streaming | butteraugli | 3/3 | 286 | 61 | 242 | 57 | 290 (gpu) | 66 (gpu) |
| libjxl | butteraugli | 16/9 | 14 | 3 | 3 | 1 | 50 (ideal) | 7 (ideal) |
| vship-cuda | butteraugli | 3/3 | 165 | 37 | 164 | 35 | n/a | n/a |
| vship-vulkan | butteraugli | 3/3 | 146 | 33 | 144 | 32 | n/a | n/a |
| vship-fast-vulkan | butteraugli | 3/3 | 312 | 76 | 251 | 58 | n/a | n/a |

The ceiling is the stage timer's GPU time from the one-thread run (`prepare` + `scale` for SSIMULACRA 2, every Butteraugli stage), i.e. what the device could reach with a perfect host; `ideal` for a CPU arm is `threads * FPS@1`, which assumes perfect scaling and is an upper bound, not a measurement.

## Vship comparison


### butteraugli

| size | ours-resident FPS | vship-cuda FPS | vship-vulkan FPS | vship-fast-vulkan FPS | ours/vship-cuda | ours/vship-fast |
| --- | --- | --- | --- | --- | --- | --- |
| 1920x1080 | 421 | 165 | 146 | 312 | 2.54x | 1.35x |
| 3840x2160 | 93 | 37 | 33 | 76 | 2.53x | 1.22x |

### ssimu2

| size | ours-resident FPS | vship-cuda FPS | vship-vulkan FPS | vship-fast-vulkan FPS | ours/vship-cuda | ours/vship-fast |
| --- | --- | --- | --- | --- | --- | --- |
| 1920x1080 | 3575 | 832 | 468 | 585 | 4.30x | 6.11x |
| 3840x2160 | 854 | 193 | 103 | 152 | 4.43x | 5.61x |
