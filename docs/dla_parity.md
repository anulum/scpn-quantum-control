<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# DLA-parity validation pathway

The `scpn_quantum_control.dla_parity` subpackage is the open-data
validation surface for the DLA-parity campaign on
[`ibm_kingston`](https://quantum.ibm.com/services/resources?backend=ibm_kingston).
It loads the 342 published circuits, recomputes every published
scalar from the raw counts, cross-checks each one against the
published summary within a documented tolerance bundle, and builds
a noiseless classical reference showing that the observed
asymmetry is a hardware-origin effect (the Hamiltonian conserves
parity exactly).

Four responsibility-scoped modules, one public facade, one CLI.

## Install

```bash
pip install 'scpn-quantum-control[dla-parity]'
```

The optional `[dla-parity]` extra adds QuTiP for the cross-backend
classical reference. The numpy backend is always available, so the
reproducer runs even without the extra.

## End-to-end — one call

```python
from scpn_quantum_control.dla_parity import run_full_harness

result = run_full_harness()
print(
    f"{result.dataset.n_circuits_total} circuits, "
    f"peak asym {100 * result.reproduction.peak_asymmetry_relative:+.2f}% "
    f"@ depth {result.reproduction.peak_asymmetry_depth}"
)
```

`run_full_harness()` raises `AssertionError` on any breach and
returns a `FullHarnessResult` with the loaded `DlaParityDataset`,
the `ReproductionResult` (including an audit trail of every
claim checked), and the `ClassicalLeakageReference`.

## CLI

```bash
python scripts/run_dla_parity_suite.py
python scripts/run_dla_parity_suite.py --verify-integrity
python scripts/run_dla_parity_suite.py --backend qutip --json > result.json
```

The CLI exits non-zero on any tolerance or invariant breach —
suitable for CI.

## Four modules, four responsibilities

### `schema` — typed dataclasses

```python
from scpn_quantum_control.dla_parity import (
    DlaParityCircuit,
    DlaParityCircuitMeta,
    DlaParityDataset,
    DlaParityRun,
    StatisticalSummary,
)
```

Every dataclass is `frozen=True, slots=True`. No I/O, no
computation — just the shape that the JSON and the statistical
pipeline agree on.

### `dataset` — JSON loader

```python
from scpn_quantum_control.dla_parity import load_dla_parity_dataset

ds = load_dla_parity_dataset(verify_integrity=True)
```

Strict schema validation (missing key → `ValueError` with the
offending JSON path). Optional SHA-256 integrity check against an
embedded digest table. Extra top-level keys land in each run's
`extra` dict for forward compatibility.

### `reproduce` — statistical re-computation

```python
from scpn_quantum_control.dla_parity import (
    ReproductionTolerance,
    reproduce_statistics,
)

# Tighten beyond the default 1e-9 / 1e-6 bundle.
result = reproduce_statistics(ds, tolerance=ReproductionTolerance(welch_p_rel=1e-9))
for name, published, actual, diff in result.claims_checked:
    print(f"{name:<40s} published={published:.6g} actual={actual:.6g} diff={diff:.3e}")
```

Ignores the pre-computed `stats` blob inside each circuit record —
walks the `counts` directly — so the full count-to-scalar path is
exercised on every run. The default tolerance bundle passes
bit-exact IEEE-754 drift but catches any real statistical
divergence.

### `baselines` — classical reference

```python
from scpn_quantum_control.dla_parity import (
    available_baselines,
    compute_classical_leakage_reference,
)

print(available_baselines())  # {'numpy': True, 'qutip': True/False}

ref = compute_classical_leakage_reference(backend="auto")
assert ref.is_zero_within_tolerance   # < 1e-10 at every (depth, sector)
```

Builds the same Hamiltonian the hardware circuits implement
(`H = Σ ω_i Z_i + Σ_{i,i+1} K_{i,i+1}(X_i X_{i+1} + Y_i Y_{i+1})`,
ω = linspace(0.8, 1.2, n), `K_{ij} = 0.45 · exp(-0.3 |i-j|)`,
Lie-Trotter at t_step = 0.3) and evolves both sectors across the
published depth sweep. Every Trotter term commutes with the
total-parity operator, so the noiseless leakage is identically
zero up to floating-point noise — the `is_zero_within_tolerance`
predicate asserts that invariant directly.

Two backends: `numpy` (always available, via
`scipy.linalg.expm`) and `qutip` (optional, via QuTiP). The
`backend="auto"` selector prefers qutip if importable. When both
run on the same inputs, they agree per-point to 1e-12.

## What the validation actually proves

| Claim (published) | Reproduced value | Tolerance |
| --- | --- | --- |
| Fisher χ² across 8 depths | 123.40 | rel ≤ 1e-6 |
| # depths significant at p < 0.05 | 7 / 8 | exact |
| Peak asymmetry | +17.48 % at depth 6 | abs ≤ 1e-9 |
| Mean asymmetry | +9.25 % (computed from `depth_summaries`) | abs ≤ 1e-9 |
| Per-depth leakage / SEM / Welch t, p | 8 depths × 2 sectors × 7 scalars | bundle |
| Classical leakage (noiseless) | 0 at every (depth, sector) | abs ≤ 1e-10 |

The reproducer recomputes every number from `counts` directly,
walks the published-summary JSON, and fails loudly on the first
mismatch. 57 scalars checked per run; every one passes on the
current dataset under the default tolerance bundle.

## Reproducing the paper number

```python
from scpn_quantum_control.dla_parity import run_full_harness

r = run_full_harness()
assert r.reproduction.fisher.chi2 > 100
assert r.reproduction.fisher.n_depths_significant_at_0_05 == 7
assert r.reproduction.peak_asymmetry_depth == 6
assert 0.17 < r.reproduction.peak_asymmetry_relative < 0.18
```

## Related

* Raw dataset:
  [`data/phase1_dla_parity/`](https://github.com/anulum/scpn-quantum-control/tree/main/data/phase1_dla_parity)
* Published summary JSON:
  [`figures/phase1/phase1_dla_parity_summary.json`](https://github.com/anulum/scpn-quantum-control/blob/main/figures/phase1/phase1_dla_parity_summary.json)
* Analysis reference script:
  [`scripts/analyse_phase1_dla_parity.py`](https://github.com/anulum/scpn-quantum-control/blob/main/scripts/analyse_phase1_dla_parity.py)
* Short paper LaTeX:
  [`paper/phase1_dla_parity.tex`](https://github.com/anulum/scpn-quantum-control/blob/main/paper/phase1_dla_parity.tex)
