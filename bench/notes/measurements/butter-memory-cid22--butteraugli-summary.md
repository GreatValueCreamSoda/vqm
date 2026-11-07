> Historical generated measurement, preserved during the 2026-11-05 build cleanup.
> Original report: `build/reports/butter-memory-cid22/butteraugli-summary.md`. Raw data now lives under
> `artifacts/reports/butter-memory-cid22/`. Paths and numbers below retain their original provenance.

# butteraugli validation

generated: 2026-11-03T13:34:41.471706+00:00
pairs: 4292
MCOS: validation/data/CID22_validation_set/CID22_validation_set.csv
command: /usr/bin/python3 validation/run_corpus.py --metric butteraugli --only ours,libjxl --accuracy-tolerance 0.02 --out-dir build/reports/butter-memory-cid22

## Scoring time

| implementation | pairs | seconds | pairs/s |
| --- | --- | --- | --- |
| ours | 4292 | 2.95 | 1452.96 |
| libjxl | 4292 | 47.13 | 91.06 |

Wall clock covers process startup, decode and scoring; pair counts include successfully scored pairs only.

## Parity vs libjxl

| implementation | pairs | max |diff| | mean |diff| | p99 |diff| | RMSE | >0.001 | >0.01 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 4292 | 2.730601 | 0.075824 | 0.607551 | 0.151838 | 4205 | 3481 |

## MCOS correlation (KRCC, SRCC, PCC)

| implementation | KRCC | SRCC | PCC |
| --- | --- | --- | --- |
| ours | 0.5445 | 0.7335 | 0.6701 |
| libjxl | 0.5511 | 0.7413 | 0.6843 |

Butteraugli is lower-is-better; the scores are negated before correlating, so a larger value always means better quality.
