> Historical generated measurement, preserved during the 2026-11-05 build cleanup.
> Original report: `build/reports/validation/butteraugli-summary.md`. Raw data now lives under
> `artifacts/reports/validation/`. Paths and numbers below retain their original provenance.

# butteraugli validation

generated: 2026-10-27T01:32:13.410915+00:00
pairs: 4292
MCOS: validation/data/CID22_validation_set/CID22_validation_set.csv
command: python3 validation/run_corpus.py --metric butteraugli --out-dir build/reports/validation

## Scoring time

| implementation | pairs | seconds | pairs/s |
| --- | --- | --- | --- |
| ours | 4292 | 2.98 | 1442.29 |
| libjxl | 4292 | 49.15 | 87.33 |
| vship-cuda | 4292 | 3.80 | 1128.90 |
| vship-vulkan | 4292 | 7.05 | 608.66 |
| vship-fast-vulkan | 4292 | 3.47 | 1237.81 |

One process per implementation, all pairs scored once; the clock covers decode and metric together.

## Parity vs libjxl

| implementation | pairs | max |diff| | mean |diff| | p99 |diff| | RMSE | >0.001 | >0.01 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 4292 | 2.730601 | 0.075824 | 0.607551 | 0.151838 | 4205 | 3481 |
| vship-cuda | 4292 | 1.188496 | 0.122342 | 0.594212 | 0.172952 | 4267 | 4003 |
| vship-vulkan | 4292 | 1.188498 | 0.122342 | 0.594214 | 0.172952 | 4267 | 4003 |
| vship-fast-vulkan | 4292 | 0.032576 | 0.000272 | 0.005912 | 0.001278 | 307 | 8 |

## MCOS correlation (KRCC, SRCC, PCC)

| implementation | KRCC | SRCC | PCC |
| --- | --- | --- | --- |
| ours | 0.5445 | 0.7335 | 0.6701 |
| libjxl | 0.5511 | 0.7413 | 0.6843 |
| vship-cuda | 0.5431 | 0.7332 | 0.6785 |
| vship-vulkan | 0.5431 | 0.7332 | 0.6785 |
| vship-fast-vulkan | 0.5511 | 0.7413 | 0.6843 |

Butteraugli is lower-is-better; the scores are negated before correlating, so a larger value always means better quality.
