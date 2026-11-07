# Ranking defaults for qualified final-score domains

2026-11-06. Following the paired audits, ranking agreement is now the default
final-score gate for the five established hosted SDR video metrics and the
balanced CID22 SSIMULACRA 2/Butteraugli profiles. Raw numerical errors remain
in every agreement report. Identity and numerical component tests remain hard
requirements. No metric arithmetic, ambiguity margin or acceptance threshold
changed in this transition.

The [five-metric audit](../../bench/notes/system/all-metric-reference-correlation.md) and
[still/temporal audit](../../bench/notes/system/ranking-temporal-validation-results.md) retain their
actual historical commands, defaults and numerical failures. This transition
does not rewrite those measurements or turn their combined-gate failures into
historical passes.

## Scope and selection

[qualified_domains.json](../../validation/qualified_domains.json) registers
the ten hosted metric/profile combinations (five metrics, smoke and full
qualification) and the two balanced still profiles. Each entry binds the
canonical frozen policy and its resolved defaults, metric, pinned original revision, input manifest and
complete original sample identities, scores and MOS labels. Temporal entries
also bind the resolved configuration hash. CID22's global configuration hash
includes candidate executable fingerprints and can change after a rebuild;
its original samples and complete policy remain bound.

The shared resolver chooses `ranking` only for an exact registered identity.
Changed policies, original scores, sample membership, input hashes or unknown
domains retain `both`. JSON key order, sample presentation order and copying
a policy file to another path do not alter its identity. Registration is an
explicit maintained-source change supported by independent qualification;
passing an arbitrary policy once does not register a domain.

The standalone agreement CLI, hosted runner and corpus adapters share this
selection. Explicit `--require` or `--gate-mode` takes precedence. `ranking`
requires reference agreement, coverage and any required subjective
noninferiority; it does not let raw numerical failure change the exit status.
`both` requires numerical and ranking acceptance, and `numerical` reproduces
the numerical gate. Legacy adapter runs without an agreement policy retain
their existing numerical acceptance behavior. Unsupported HDR, high-bit-depth,
alternate-model/display and other component oracles have not been retired.

## Verification

The ordinary command now selects ranking without an override:

```sh
python3 -m validation.temporal.run --profile smoke
python3 -m validation.temporal.run --profile qualification
```

The actual five-metric smoke run exited 0 and recorded resolved `ranking` in
both aggregate and per-metric reports. SSIMULACRA 2's numerical failure stayed
visible. Repeating its smoke arm with `--gate-mode both` exited 1. The five
retained 390-pair qualification triples were re-evaluated without an explicit
mode and each exited 0. Both retained CID22 triples also defaulted to ranking,
including their required paired subjective checks, with numerical failures
preserved. Live corpus-adapter reproductions use the frozen CID22 subset and
the same default selection.

Fifteen new scoped-default regression tests cover all twelve registry entries,
policy/sample mutations, binary fingerprint changes, standalone default exit
behavior, explicit strict overrides and rejected score reversal. Adapter tests
cover registered and unregistered defaults, retained diagnostics and resolved
CVVDP aggregate mode. The existing identity, numerical oracle and component
tests remain in CTest. All 24 release suites pass at this transition; modified
new Python passes Ruff/YAPF, while the legacy corpus adapter retains its prior
78 Ruff findings without introducing new ones.

Raw commands, reports and logs are retained under
`artifacts/reports/ranking-default-gates/`; the reference registry and historical
compact score reports remain maintained inputs. This is a change in final-score
acceptance semantics for the documented domains, not a claim about unseen
content or universal perceptual fidelity.
