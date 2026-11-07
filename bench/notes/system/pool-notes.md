# Pipeline pool: streaming fixed-count benchmark

2026-11-02, RTX 5070 (driver 615.78.08), default exact planes. The public
`PipelinePool::ScoreMany` and `ButteraugliPool::ComputeMany` were run via
`ssimulacra2_bench --pool`. Each round alternated worker-count order; 1, 2,
and 3 workers ran five times each at each size and metric in one sitting.
The wall clock brackets the actual public pool call, including initial
submissions, score/result conversion, and final drains. The pipeline creation
and decoding occur outside the timer. Each SSIMULACRA 2 run timed 300 pairs,
and each Butteraugli run timed 50 pairs; both used three warm-up calls per
worker. The same decoded RGB8 testsrc2 pair is repeated for each call; upload
is included and both images are prepared each call. This is not a diverse
video workload or a resident-buffer benchmark.

| metric | size | workers | median pairs/s | full range | score | checksum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ssimulacra2 | 1080p | 1 | 1141.5 | 1138.3–1143.2 | 91.716994 | 27790.249236 |
| ssimulacra2 | 1080p | 2 | 1640.9 | 1639.1–1642.3 | 91.716994 | 28065.400219 |
| ssimulacra2 | 1080p | 3 | 1641.0 | 1637.9–1642.4 | 91.716994 | 28340.551201 |
| ssimulacra2 | 4k | 1 | 292.7 | 292.1–293.3 | 92.437150 | 28008.456299 |
| ssimulacra2 | 4k | 2 | 441.3 | 440.3–441.5 | 92.437150 | 28285.767748 |
| ssimulacra2 | 4k | 3 | 440.5 | 439.9–440.8 | 92.437150 | 28563.079196 |
| butteraugli | 1080p | 1 | 285.3 | 283.4–286.5 | 7.279592 | 385.818353 |
| butteraugli | 1080p | 2 | 349.8 | 349.3–350.5 | 7.279592 | 407.657127 |
| butteraugli | 1080p | 3 | 349.2 | 349.0–350.0 | 7.279592 | 429.495902 |
| butteraugli | 4k | 1 | 67.0 | 66.8–67.1 | 7.279592 | 385.818353 |
| butteraugli | 4k | 2 | 79.0 | 79.0–79.1 | 7.279592 | 407.657127 |
| butteraugli | 4k | 3 | 73.7 | 73.7–73.8 | 7.279592 | 429.495902 |

Checksums include `iterations + 3 * workers` calls. At 1080p SSIMULACRA 2,
two workers exceeded one by about 44%; at 4K by about 51%. Three did not
improve on two in this workload. Butteraugli improved with two workers, while
three were slower than two at 4K. These are measurements of this pool and
input, not a universal worker-count recommendation.

## GPU stages

The stage values below are per-call medians and the five-run range, in ms.
For multiple workers these are cumulative device-stage durations per call,
which can overlap; they are not shares of the measured wall clock. The
`score` stage is host-side conversion. Empty `untimed` rows are omitted.

| metric | size | workers | stage | median ms | full range ms |
| --- | --- | ---: | --- | ---: | ---: |
| ssimulacra2 | 1080p | 1 | prepare | 0.097950 | 0.097539–0.098041 |
| ssimulacra2 | 1080p | 1 | scale | 0.145417 | 0.144822–0.145454 |
| ssimulacra2 | 1080p | 1 | score | 0.000117 | 0.000113–0.000159 |
| ssimulacra2 | 1080p | 2 | prepare | 0.112900 | 0.112772–0.112975 |
| ssimulacra2 | 1080p | 2 | scale | 0.145809 | 0.145407–0.145856 |
| ssimulacra2 | 1080p | 2 | score | 0.000122 | 0.000112–0.000124 |
| ssimulacra2 | 1080p | 3 | prepare | 0.113175 | 0.112914–0.113252 |
| ssimulacra2 | 1080p | 3 | scale | 0.146004 | 0.145475–0.146068 |
| ssimulacra2 | 1080p | 3 | score | 0.000117 | 0.000112–0.000123 |
| ssimulacra2 | 4k | 1 | prepare | 0.432616 | 0.428728–0.434031 |
| ssimulacra2 | 4k | 1 | scale | 0.683188 | 0.680957–0.683501 |
| ssimulacra2 | 4k | 1 | score | 0.000387 | 0.000354–0.000394 |
| ssimulacra2 | 4k | 2 | prepare | 0.433587 | 0.431409–0.433958 |
| ssimulacra2 | 4k | 2 | scale | 0.694692 | 0.693844–0.695925 |
| ssimulacra2 | 4k | 2 | score | 0.000362 | 0.000345–0.000394 |
| ssimulacra2 | 4k | 3 | prepare | 0.432522 | 0.431342–0.435019 |
| ssimulacra2 | 4k | 3 | scale | 0.694269 | 0.693958–0.696486 |
| ssimulacra2 | 4k | 3 | score | 0.000355 | 0.000325–0.000382 |
| butteraugli | 1080p | 1 | opsin | 0.512577 | 0.511580–0.514905 |
| butteraugli | 1080p | 1 | separate | 1.135648 | 1.132789–1.137324 |
| butteraugli | 1080p | 1 | malta | 0.578381 | 0.574605–0.579436 |
| butteraugli | 1080p | 1 | diff | 0.258449 | 0.257350–0.258603 |
| butteraugli | 1080p | 1 | mask | 0.085534 | 0.084920–0.085719 |
| butteraugli | 1080p | 1 | combine | 0.176472 | 0.175958–0.176632 |
| butteraugli | 1080p | 1 | reduce | 0.043079 | 0.043041–0.043245 |
| butteraugli | 1080p | 2 | opsin | 0.520902 | 0.520420–0.521606 |
| butteraugli | 1080p | 2 | separate | 1.143554 | 1.141069–1.144220 |
| butteraugli | 1080p | 2 | malta | 0.581516 | 0.579004–0.581980 |
| butteraugli | 1080p | 2 | diff | 0.256140 | 0.256004–0.257021 |
| butteraugli | 1080p | 2 | mask | 0.085220 | 0.085075–0.085306 |
| butteraugli | 1080p | 2 | combine | 0.176635 | 0.176199–0.176780 |
| butteraugli | 1080p | 2 | reduce | 0.043213 | 0.043067–0.043253 |
| butteraugli | 1080p | 3 | opsin | 0.521991 | 0.520799–0.522222 |
| butteraugli | 1080p | 3 | separate | 1.145793 | 1.143655–1.146555 |
| butteraugli | 1080p | 3 | malta | 0.581948 | 0.578842–0.582403 |
| butteraugli | 1080p | 3 | diff | 0.256789 | 0.256535–0.257139 |
| butteraugli | 1080p | 3 | mask | 0.085508 | 0.085390–0.085678 |
| butteraugli | 1080p | 3 | combine | 0.177800 | 0.177433–0.178056 |
| butteraugli | 1080p | 3 | reduce | 0.043189 | 0.043080–0.043306 |
| butteraugli | 4k | 1 | opsin | 2.001727 | 1.998371–2.002971 |
| butteraugli | 4k | 1 | separate | 4.945743 | 4.940722–4.949580 |
| butteraugli | 4k | 1 | malta | 2.691146 | 2.680474–2.692458 |
| butteraugli | 4k | 1 | diff | 1.413788 | 1.413436–1.415464 |
| butteraugli | 4k | 1 | mask | 0.392388 | 0.391762–0.392644 |
| butteraugli | 4k | 1 | combine | 0.781536 | 0.781185–0.783097 |
| butteraugli | 4k | 1 | reduce | 0.183976 | 0.183903–0.184253 |
| butteraugli | 4k | 2 | opsin | 2.005633 | 2.005062–2.006011 |
| butteraugli | 4k | 2 | separate | 4.971080 | 4.967197–4.971959 |
| butteraugli | 4k | 2 | malta | 2.690669 | 2.685789–2.691184 |
| butteraugli | 4k | 2 | diff | 1.412918 | 1.412830–1.413099 |
| butteraugli | 4k | 2 | mask | 0.391316 | 0.391226–0.392008 |
| butteraugli | 4k | 2 | combine | 0.782839 | 0.782627–0.786892 |
| butteraugli | 4k | 2 | reduce | 0.184273 | 0.183925–0.184343 |
| butteraugli | 4k | 3 | opsin | 2.002790 | 2.001498–2.003144 |
| butteraugli | 4k | 3 | separate | 4.958461 | 4.955308–4.959172 |
| butteraugli | 4k | 3 | malta | 2.686833 | 2.684688–2.687338 |
| butteraugli | 4k | 3 | diff | 1.411724 | 1.411499–1.413577 |
| butteraugli | 4k | 3 | mask | 0.391618 | 0.391276–0.392072 |
| butteraugli | 4k | 3 | combine | 0.783439 | 0.783220–0.787921 |
| butteraugli | 4k | 3 | reduce | 0.184214 | 0.183816–0.184257 |

Representative commands, with `ORIG`/`DIST` the 1080p or 4K fixture:

```sh
./build/ssimulacra2_bench --pool --metric ssimulacra2 --pair "$ORIG" "$DIST" --workers 2 --iterations 300 --warmup 3 --csv /tmp/pool-pair.csv --stages-csv /tmp/pool-stages.csv
./build/ssimulacra2_bench --pool --metric butteraugli --pair "$ORIG" "$DIST" --workers 2 --iterations 50 --warmup 3 --csv /tmp/pool-pair.csv --stages-csv /tmp/pool-stages.csv
```

All individual logs, scores, checksums, and stage tables were saved under
`/tmp/ss2-pool-results/` (temporary).
