> Historical generated measurement, preserved during the 2026-11-05 build cleanup.
> Original report: `build/reports/performance/summary.md`. Raw data now lives under
> `artifacts/reports/performance/`. Paths and numbers below retain their original provenance.

# In-memory benchmark

generated: 2026-10-27T01:43:48.601089+00:00
pairs: build/bench/pairs/all.txt (103 pairs)
protocol: decode once per pair; median of 20 timed calls after 3 warmups; 3 interleaved rounds; FPS is 1000 / median ms/pair

| implementation | metric | size | pairs | median ms/pair | FPS | median score | max |Δscore| vs ours |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ours-resident | ssimu2 | 512x512 | 101 | 0.122 | 8181.4 | 67.2978 | 0.0000 |
| ours-resident | ssimu2 | 1920x1080 | 1 | 0.328 | 3045.1 | 30.7424 | 0.0000 |
| ours-resident | ssimu2 | 3840x2160 | 1 | 1.195 | 836.6 | 27.3045 | 0.0000 |
| ours-streaming | ssimu2 | 512x512 | 101 | 0.200 | 4989.0 | 67.2978 | 0.0000 |
| ours-streaming | ssimu2 | 1920x1080 | 1 | 0.880 | 1135.8 | 30.7424 | 0.0000 |
| ours-streaming | ssimu2 | 3840x2160 | 1 | 3.400 | 294.1 | 27.3045 | 0.0000 |
| ours-half-planes | ssimu2 | 512x512 | 101 | 0.122 | 8221.4 | 67.2423 | 0.1752 |
| ours-half-planes | ssimu2 | 1920x1080 | 1 | 0.327 | 3059.4 | 30.6901 | 0.0523 |
| ours-half-planes | ssimu2 | 3840x2160 | 1 | 1.125 | 889.1 | 27.2913 | 0.0132 |
| cloudinary | ssimu2 | 512x512 | 101 | 37.648 | 26.6 | 67.1725 | 0.4213 |
| cloudinary | ssimu2 | 1920x1080 | 1 | 280.681 | 3.6 | 30.0196 | 0.7228 |
| cloudinary | ssimu2 | 3840x2160 | 1 | 1158.285 | 0.9 | 26.3290 | 0.9756 |
| libjxl | ssimu2 | 512x512 | 101 | 15.882 | 63.0 | 67.1748 | 0.4646 |
| libjxl | ssimu2 | 1920x1080 | 1 | 126.820 | 7.9 | 30.0423 | 0.7001 |
| libjxl | ssimu2 | 3840x2160 | 1 | 526.528 | 1.9 | 26.1941 | 1.1104 |
