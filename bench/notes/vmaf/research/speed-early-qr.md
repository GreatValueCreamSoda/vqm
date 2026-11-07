# Selecting the existing SpEED QR fallback earlier

Source and CPU numerical investigation, 2026-11-05. This note accompanies
[`early_qr.py`](../../../experiments/speed_parallelism/early_qr.py) and
[`early_qr_numerics.py`](../../../experiments/speed_parallelism/early_qr_numerics.py).
It records the selection argument and CPU diagnostics. GPU speed, feature
equality and adoption gates are measured separately by the parent investigation.

## Work that can be discarded

The current SpEED spectral workgroup first forms a 25×25 symmetric
tridiagonal representation using Householder reduction. It then computes
25 eigenvalues in parallel by FP32 Sturm bisection. Afterward it checks

`failed == 2 || minimum_eigen <= 1e-6 + 1e-3*scale`,

where `scale` is the maximum absolute padded Gershgorin endpoint. The safety
band retains the earlier QR behavior near a singular covariance. When the
guard selects QR, that path restores the tridiagonal from the saved matrix,
resets the bisection failure state and overwrites all 25 eigenvalue outputs.
The bisection results have no remaining consumer.

This creates an opportunity to avoid work rather than distribute more of it
across SMs. It does not make one QR system span multiple SMs; it can shorten
the single-matrix critical path when fallback is inevitable. The benefit
depends on actual content and requires profiling.

## Cheap Rayleigh gate

For the exact real symmetric matrix defined by the stored FP32 tridiagonal,
Rayleigh's principle gives `lambda_min <= min(diagonal)`: selecting a coordinate
basis vector produces its diagonal entry as a Rayleigh quotient. Thus a diagonal
entry well below the existing guard is a cheap indication that QR will be
needed.

The experiment uses `min(diagonal) < guard - 1e-6*scale` as a gate. The scale
margin is an engineering choice for deciding whether a short search is likely
to pay. It is not a complete FP32 Sturm error bound. In particular, the
Rayleigh statement concerns the exact spectrum of the stored tridiagonal;
it alone does not prove that the current rounded, pivot-clamped Sturm
recurrence will return a minimum inside the guard. Pivot cancellation,
overflow and nonmonotonic rounded counts remain possible.

The installer's `--mode rayleigh` bypasses the search certificate and selects
QR directly after this gate. It is retained as a heuristic control, not as a
qualified behavioral-equivalence claim.

## Prefix certificate

The default `--mode prefix` uses the diagonal gate only to decide whether to
try up to 16 steps of the existing rank-zero bisection on lane zero. It uses
the same initial endpoints, midpoint expression, `sturm_count` function and
rank-zero endpoint update as the unchanged full search. It selects QR early
only in either of these cases:

1. The rank-zero upper endpoint is at or below the existing guard. Every
   continuation keeps its interval inside that upper endpoint, so its final
   midpoint cannot exceed the guard. If it fails instead, the original
   `failed == 2` condition also selects QR.
2. That same rank-zero prefix encounters an invalid Sturm count. The unchanged
   full search would leave its convergence flag false and select QR because
   of failure state 2.

An adjacent-endpoint convergence during the prefix is also accepted when its
midpoint is inside the guard. Otherwise, exhausting the short prefix changes
no selection: the original full 25-lane search runs again from its original
endpoints. No new approximate spectrum is accepted. Sturm monotonicity is not
needed for the endpoint certificate: each update chooses a midpoint inside
the current finite ordered interval.

`--gate sturm` instead tries the same prefix only when a cheap count at the
existing guard lies in 1..25. This count is only a gate, never a fallback
certificate by itself. It can catch a small eigenvalue hidden by comparatively
large diagonal entries. A count above 25 leaves the ordinary path in place.
`--gate always` tries the prefix for every valid set of bounds as a pricing
control. Both retain the same endpoint/failure certificate, and can add
unproductive work when the original path would remain on bisection.

The argument assumes the prefix and continuation evaluate that recurrence in
the same FP32 way. A compiler can change contraction or instruction lowering
when surrounding control flow changes, so same-device comparisons against the
unmodified shader and independent QR oracle remain necessary. The certificate
is stronger than the diagonal-only heuristic; compiler checks alone are not a
metric parity qualification.

## State and synchronization

The installer adds one workgroup-shared flag and initializes it on lane zero
after the enclosing bounds and their validity are published. It reuses the
existing following workgroup barrier. Only the bisection condition changes;
Householder, Cholesky and legacy QR arithmetic remain textually unchanged.

`minimum_eigen` is already initialized to zero. When a prefix certificate
skips bisection, that sentinel lies inside the positive existing guard and
therefore selects the unchanged QR fallback. The fallback resets failure
state and republishes all eigenvalues before the original final barrier.

Invalid or nonfinite bounds do not enter the new probe. They retain the
existing failure-to-QR behavior. Finite negative intervals are valid; their
upper endpoint may certify fallback immediately. NaN covariance inputs that
return earlier keep their existing path. The probe reads tridiagonal values
but modifies neither them nor the saved matrix, so QR starts from the same
stored inputs.

There are no new arena words, descriptors, API features, dispatches or public
layout changes. The existing 128/512-invocation shaders and test-only forced-QR
variant use the same source. This differs from the subgroup-pooling experiment,
which needs explicit subgroup capability handling.

## CPU evidence and limits

The maintained numerical probe ran 270 deterministic direct-tridiagonal cases:
14 structured zero, identity, negative, threshold-adjacent and scale cases,
plus 256 seeded random bidiagonal-product and arbitrary symmetric tridiagonal
stress cases. Scale ranges from `1e-30` to `1e30`; the seed is `20261105`.

The explicitly noncontracted FP32 model found:

| Observation | Cases |
| --- | ---: |
| Finite valid enclosing bounds | 270 |
| Unmodified rank-zero search would select QR | 265 |
| Short-prefix certificates | 262 |
| Certificates contradicting the unchanged rank-zero selection | 0 |
| QR cases missed by the cheap diagonal gate | 3 |
| Diagonal-gate versus rank-zero disagreements in this sample | 0 |

The last zero is sampled evidence, not a proof for the heuristic mode. The
rank-zero check is sufficient for the certificates; failures in the other
24 searches can make the complete original shader select QR in additional
cases. The probe does not model those extra fallback causes.

With either the Sturm or always gate, the same 270 cases produced 265
certificates and zero contradictions. They recovered the three fallback cases
missed by the diagonal gate. The snapshot stores the default full case table
and compact differences for these alternate gates.

The CPU arithmetic uses explicit NumPy FP32 operations with separate multiply
and subtract. Driver FMA contraction, denormal handling and GPU code generation
are not emulated. The arbitrary tridiagonal cases are conditioning diagnostics,
not claimed realizable covariance matrices from image inputs. These counts
are not an estimate of fallback frequency in videos.

The [compact snapshot](../../measurements/vmaf-speed-earlyqr-numerics.json)
keeps the input hashes, parameters, source hash and every case result. The raw
result is under `artifacts/reports/vmaf-speed-parallelism/earlyqr/numerics.json`.

## Reproduction

Apply to a fresh isolated source copy; the installer refuses the maintained
checkout, its ancestors and its descendants:

```sh
python3 bench/experiments/speed_parallelism/early_qr.py /tmp/source-copy
```

`--mode rayleigh` selects the heuristic control. `--margin-scale` changes the
diagonal gate, and `--probe-iterations` accepts 1..320. The default generated
shader uses prefix mode, the diagonal gate, margin `1e-6` and 16 probe steps.
`--gate sturm` and `--gate always` are available only with prefix mode. The
default generated shader remained byte-identical when these options were added.
All anchors are checked
before its single source-file write; create a fresh copy for each arm.

Both modes, both 128/512 lane counts and the ordinary/forced-legacy variants
compile with `glslc -O --target-env=vulkan1.1` and pass Vulkan 1.1 SPIR-V
validation: eight variants in total. The Sturm/always gates additionally
compile and validate at 128/512 lanes in the ordinary path. Commands, generated source and module
hashes are retained under
`artifacts/reports/vmaf-speed-parallelism/earlyqr/compiler/`. The original
compiler commands retain their historical report paths after relocation.

Reproduce the CPU diagnostics with Python and NumPy:

```sh
python3 bench/experiments/speed_parallelism/early_qr_numerics.py
```

The CLI supports `--cases`, `--seed`, `--probe-iterations`, `--margin-scale`, `--gate` and
`--out`. Importing the module performs no work. Rerunning after the raw-output
relocation reproduced all substantive case data exactly.
