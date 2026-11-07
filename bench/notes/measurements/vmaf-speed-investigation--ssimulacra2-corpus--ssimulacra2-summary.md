> Historical generated measurement, preserved during the 2026-11-05 build cleanup.
> Original report: `build/reports/vmaf-speed-investigation/ssimulacra2-corpus/ssimulacra2-summary.md`. Raw data now lives under
> `artifacts/reports/vmaf-speed-investigation/ssimulacra2-corpus/`. Paths and numbers below retain their original provenance.

# ssimulacra2 validation

generated: 2026-11-05T11:22:24.064505+00:00
pairs: 300
MCOS: validation/data/CID22_validation_set/CID22_validation_set.csv
command: /usr/bin/python3 validation/run_corpus.py --metric ssimulacra2 --only ours,cloudinary --limit 300 --accuracy-tolerance 0.005 --out-dir build/reports/vmaf-speed-investigation/ssimulacra2-corpus

## Scoring time

| implementation | pairs | seconds | pairs/s |
| --- | --- | --- | --- |
| ours | 300 | 0.45 | 668.72 |
| cloudinary | 300 | 2.52 | 119.21 |

Wall clock covers process startup, decode and scoring; pair counts include successfully scored pairs only.

## Parity vs cloudinary

| implementation | pairs | max |diff| | mean |diff| | p99 |diff| | RMSE | >0.001 | >0.01 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 300 | 0.524941 | 0.202585 | 0.446414 | 0.220544 | 300 | 300 |

## MCOS correlation (KRCC, SRCC, PCC)

| implementation | KRCC | SRCC | PCC |
| --- | --- | --- | --- |
| ours | 0.8145 | 0.9554 | 0.9433 |
| cloudinary | 0.8144 | 0.9554 | 0.9436 |
