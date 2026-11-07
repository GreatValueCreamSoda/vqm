> Historical generated measurement, preserved during the 2026-11-05 build cleanup.
> Original report: `build/reports/perf-smoke/summary.md`. Raw data now lives under
> `artifacts/reports/perf-smoke/`. Paths and numbers below retain their original provenance.

# In-memory benchmark

generated: 2026-10-27T01:29:37.441877+00:00
pairs: build/bench/pairs/cid22-10.txt (10 pairs)
protocol: decode once per pair; median of 2 timed calls after 1 warmups; 1 interleaved rounds; FPS is 1000 / median ms/pair

| implementation | metric | size | pairs | median ms/pair | FPS | median score | max |Δscore| vs ours |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ours-resident | ssimu2 | 512x512 | 10 | 0.129 | 7726.9 | 68.5595 | 0.0000 |
| ours-streaming | ssimu2 | 512x512 | 10 | 0.215 | 4650.6 | 68.5595 | 0.0000 |
| ours-half-planes | ssimu2 | 512x512 | 10 | 0.128 | 7809.3 | 68.5216 | 0.0984 |
| cloudinary | ssimu2 | 512x512 | 10 | 37.232 | 26.9 | 68.4354 | 0.3594 |
| libjxl | ssimu2 | 512x512 | 10 | 15.684 | 63.8 | 68.4212 | 0.3570 |
| fssimu2 | ssimu2 | 512x512 | 10 | 10.180 | 98.2 | 67.0017 | 4.6942 |
| vship-cuda | ssimu2 | 512x512 | 10 | 0.350 | 2856.0 | 67.1335 | 4.4799 |
| vship-vulkan | ssimu2 | 512x512 | 10 | 0.556 | 1798.5 | 67.1302 | 4.4774 |
| vship-fast-vulkan | ssimu2 | 512x512 | 10 | 0.519 | 1925.4 | 68.4328 | 0.3635 |
| ours-resident | butteraugli | 512x512 | 10 | 0.344 | 2909.2 | 3.6326 | 0.0000 |
| ours-streaming | butteraugli | 512x512 | 10 | 0.572 | 1749.2 | 3.6326 | 0.0000 |
| libjxl | butteraugli | 512x512 | 10 | 98.144 | 10.2 | 3.6747 | 0.1740 |
| vship-cuda | butteraugli | 512x512 | 10 | 0.993 | 1007.4 | 3.5561 | 0.3506 |
| vship-vulkan | butteraugli | 512x512 | 10 | 1.616 | 618.9 | 3.5561 | 0.3506 |
| vship-fast-vulkan | butteraugli | 512x512 | 10 | 0.854 | 1171.5 | 3.6747 | 0.1740 |
