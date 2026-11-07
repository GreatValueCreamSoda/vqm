> Historical generated measurement, preserved during the 2026-11-05 build cleanup.
> Original report: `build/reports/validation/ssimulacra2-summary.md`. Raw data now lives under
> `artifacts/reports/validation/`. Paths and numbers below retain their original provenance.

# ssimulacra2 validation

generated: 2026-10-27T01:31:06.425668+00:00
pairs: 4292
MCOS: validation/data/CID22_validation_set/CID22_validation_set.csv
command: python3 validation/run_corpus.py --metric ssimulacra2 --out-dir build/reports/validation

## Scoring time

| implementation | pairs | seconds | pairs/s |
| --- | --- | --- | --- |
| ours | 4292 | 3.21 | 1339.08 |
| ours-half-planes | 4292 | 2.97 | 1445.46 |
| cloudinary | 4292 | 27.82 | 154.27 |
| libjxl | 4292 | 20.43 | 210.07 |
| fssimu2 | 4292 | 9.69 | 442.96 |
| vship-cuda | 4292 | 3.11 | 1378.62 |
| vship-vulkan | 4292 | 3.67 | 1168.49 |
| vship-fast-vulkan | 4292 | 3.39 | 1265.51 |

One process per implementation, all pairs scored once; the clock covers decode and metric together.

## Parity vs cloudinary

| implementation | pairs | max |diff| | mean |diff| | p99 |diff| | RMSE | >0.001 | >0.01 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 4292 | 3.000931 | 0.312592 | 1.434458 | 0.408375 | 4292 | 4292 |
| ours-half-planes | 4292 | 2.684856 | 0.252225 | 1.210078 | 0.347355 | 4289 | 4265 |
| libjxl | 4292 | 1.218272 | 0.078039 | 0.647757 | 0.149010 | 4184 | 3277 |
| fssimu2 | 4292 | 5.545774 | 1.443643 | 4.117493 | 1.713228 | 4289 | 4273 |
| vship-cuda | 4292 | 5.185680 | 1.282903 | 3.875791 | 1.547252 | 4291 | 4281 |
| vship-vulkan | 4292 | 5.174778 | 1.277969 | 3.861648 | 1.541746 | 4290 | 4283 |
| vship-fast-vulkan | 4292 | 1.224478 | 0.079191 | 0.643196 | 0.151046 | 4172 | 3233 |

## MCOS correlation (KRCC, SRCC, PCC)

| implementation | KRCC | SRCC | PCC |
| --- | --- | --- | --- |
| ours | 0.7046 | 0.8885 | 0.8765 |
| ours-half-planes | 0.7051 | 0.8888 | 0.8768 |
| cloudinary | 0.7072 | 0.8901 | 0.8784 |
| libjxl | 0.7062 | 0.8895 | 0.8778 |
| fssimu2 | 0.7051 | 0.8889 | 0.8769 |
| vship-cuda | 0.7050 | 0.8889 | 0.8768 |
| vship-vulkan | 0.7049 | 0.8888 | 0.8767 |
| vship-fast-vulkan | 0.7060 | 0.8893 | 0.8777 |
