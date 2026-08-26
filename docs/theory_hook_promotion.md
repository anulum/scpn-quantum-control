<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Theory-Hook Promotion Matrix

The promotion matrix distinguishes a useful research routine from an admitted
product or control capability. Importability is not promotion. Every reviewed
hook has a machine-readable tier, one permitted role, explicit forbidden claims,
and a small local fixture that proves only the bounded software path.

The current matrix is fail-closed: no hook is admitted for actuation,
publication claims, hardware performance, or differentiable execution. A
passing fixture does not change those permissions.

## Current decisions

| Hook | Tier | Status | Permitted role | Current evidence |
|---|---:|---|---|---|
| Quantum speed limit | B | `bounded_candidate` | Optional offline control constraint candidate | Two-qubit closed-system threshold fixture |
| Hamiltonian learning | B | `bounded_candidate` | Synthetic inverse problem | Exact two-qubit correlators, initialized at the generating coupling |
| Koopman closure | B | `diagnostic_only` | Classical local baseline | Finite two-oscillator observable matrix and Hermitian projection |
| Minimum bipartite QMI | D | `research_only` | Mutual-information diagnostic | Bell-state QMI identity plus tier-D policy check |
| Stabilizer Rényi entropy | B | `diagnostic_only` | Resource-theory diagnostic | Stabilizer/T-state contrast |
| Spectral form factor | B | `diagnostic_only` | Finite-size spectral diagnostic | Four-qubit exact spectrum with magnetisation-sector spacing ratio |

The canonical records are returned by
`list_theory_hook_promotions()`. The committed result is regenerated from code,
not edited by hand:

```bash
python scripts/run_theory_hook_promotion_evidence.py --check
```

The byte-checked outputs are `data/theory_hook_promotion/evidence.json` and
`data/theory_hook_promotion/evidence.md` in the repository.

## Reading the tiers

Tier B means the implementation has a bounded, testable mathematical role. It
does not mean production-ready, differentiable, experimentally validated, or
publication-ready. Tier D means research-only: the present semantics are not
sufficient for promotion, regardless of whether the code executes correctly.

Statuses add a second axis:

- `bounded_candidate` permits future evaluation against named promotion gates;
- `diagnostic_only` permits the current observable or baseline but not its use
  as a certification;
- `research_only` prevents promotion until the semantic gap itself is closed.

All six records expose `admitted_for_control = false`,
`admitted_for_publication_claim = false`, and `differentiable = false`.

## Quantum speed limit

`compute_qsl()` evolves a finite closed Kuramoto-XY model and reports a sampled
local-phase-order threshold time. `tau_MT` is the Mandelstam–Tamm lower bound for
the actual initial/target overlap in that simulation.

The legacy `tau_ML` field is more limited: it evaluates the traditional
Margolus–Levitin orthogonalization-time expression. If the final state is not
orthogonal to the initial state, that value is not an arbitrary-fidelity bound
for the target. It is retained for compatibility and labelled as an
orthogonalization reference. The distinction matters because extensions of the
Margolus–Levitin result to arbitrary fidelity require additional semantics.

Allowed now:

- finite closed-system MT and legacy ML diagnostic calculation;
- offline evaluation as a possible lower-bound feature after an independent
  control contract is defined.

Not allowed now:

- measured synchronization time;
- a critical exponent or BKT certificate;
- automatic rejection or actuation inside a controller;
- arbitrary-fidelity interpretation of `tau_ML`.

Primary mathematical context: Levitin and Toffoli,
[Phys. Rev. Lett. 103, 160502 (2009)](https://doi.org/10.1103/PhysRevLett.103.160502).

## Hamiltonian learning

`learn_hamiltonian()` fits a symmetric non-negative coupling matrix by matching
ground-state `XX+YY` correlators. The current objective repeatedly performs
dense exact diagonalization and reports an in-sample residual.

The bounded fixture deliberately starts the optimizer at the generating
two-qubit coupling. It proves end-to-end wiring and self-consistency, not
identifiability. Distinct couplings may yield similar observables, and the
current routine has no shot-noise model, calibration model, posterior,
confidence region, held-out objective, or misspecification test.

Before promotion, this route needs:

1. identifiability and uncertainty analysis;
2. held-out noisy and misspecified synthetic systems;
3. measured-data calibration with no fit/evaluation leakage.

Hamiltonian-learning context: Wiebe, Granade, Ferrie, and Cory,
[Phys. Rev. A 89, 042314 (2014)](https://doi.org/10.1103/PhysRevA.89.042314).

## Koopman local closure

The exact Koopman operator is linear but generally infinite-dimensional. The
repository routine uses a finite basis containing phase identities and pairwise
sine/cosine observables. Its matrix is reference-point-dependent and truncates
higher-order terms. It is therefore described as a finite local
Koopman-*style* closure.

`koopman_to_hamiltonian()` constructs `i(L-L†)/2`. This is a Hermitian projection:
it discards the symmetric part of `L`. It is useful for bounded matrix and
spectral experiments but is not dynamically equivalent to the complete
nonlinear Kuramoto flow.

Allowed now:

- a classical local observable-space baseline;
- closure-matrix and projected-matrix diagnostics.

Explicitly forbidden:

- an exact finite Koopman invariant subspace;
- full nonlinear dynamics;
- BQP-completeness or quantum advantage.

Finite approximation context: Williams, Kevrekidis, and Rowley,
[J. Nonlinear Sci. 25, 1307–1346 (2015)](https://doi.org/10.1007/s00332-015-9258-5).

## Minimum bipartite mutual information is not IIT Φ

The legacy `quantum_phi` module computes

\[
I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})
\]

for every non-trivial bipartition and returns the minimum and maximum. This is
quantum mutual information. It is not Integrated Information Theory Φ.

No causal model, intervention repertoire, cause-effect structure, exclusion
postulate, or IIT composition calculation is implemented. Consequently:

- the route is permanently labelled tier D under its current semantics;
- `IntegratedInformationPhi` fails closed unless a caller explicitly requests
  a labelled entropy or mutual-information diagnostic;
- proxy results set `phi_available = 0.0` and
  `is_integrated_information = 0.0`;
- proxy results never use the key `phi`;
- no consciousness, sentience, cognition, or clinical interpretation is
  allowed.

Legacy `PhiResult`, `compute_quantum_phi()`, and `phi_*` field names remain only
for import and serialization compatibility. Their API documentation identifies
the values as QMI.

IIT 3.0 illustrates the causal-structure semantics absent from this code:
Oizumi, Albantakis, and Tononi,
[PLOS Computational Biology 10, e1003588 (2014)](https://doi.org/10.1371/journal.pcbi.1003588).
That citation documents the semantic mismatch; it is not endorsement or
validation of IIT claims by this package.

## Stabilizer Rényi entropy

`magic_nonstabilizerness` enumerates all `4**n` Pauli strings for a pure state
and evaluates the documented stabilizer Rényi-2 quantity. This is a bounded
resource-theory diagnostic. The exact fixture distinguishes a computational
stabilizer state from a T state.

A maximum over a finite coupling grid is only the grid argmax. It is not a
critical-point estimate and does not establish fault-tolerant resource cost,
classical hardness, or quantum advantage. Criticality work requires a
preregistered finite-size protocol and uncertainty-aware measurement route.

Primary definition: Leone, Oliviero, and Hamma,
[Phys. Rev. Lett. 128, 050402 (2022)](https://doi.org/10.1103/PhysRevLett.128.050402).

## Spectral form factor and adjacent-gap ratio

`compute_sff()` evaluates the normalized finite-spectrum form factor. The
reported adjacent-gap ratio defaults to a magnetisation sector because mixing
independent symmetry sectors can distort level statistics. Full-spectrum and
selected-sector ratios remain separate in the result.

The compatibility field `chaos_onset_K` is a fixed-threshold crossing on the
provided grid. It is not a statistical certificate of quantum chaos, a
Poisson-to-random-matrix transition, a BKT transition, or a coincidence between
chaos and synchronization.

Promotion requires a preregistered ensemble, energy window, symmetry policy,
null distribution, finite-size scaling plan, and held-out replication. For the
adjacent-gap-ratio statistic, see Atas *et al.*,
[Phys. Rev. Lett. 110, 084101 (2013)](https://doi.org/10.1103/PhysRevLett.110.084101).

## Programmatic use

Inspect policy without running numerical fixtures:

```python
from scpn_quantum_control.analysis import get_theory_hook_promotion

policy = get_theory_hook_promotion("koopman_local_closure")
assert policy.status.value == "diagnostic_only"
assert policy.admitted_for_control is False
assert "BQP-completeness" in policy.forbidden_claims
```

Run all local fixtures and build a digest-locked report:

```python
from scpn_quantum_control.analysis import build_theory_hook_promotion_report

report = build_theory_hook_promotion_report()
assert report.passed
assert all(not row.admitted_for_control for row in report.records)
print(report.content_digest)
```

`report.passed` means only that the six named fixtures passed. Always evaluate
the corresponding policy record before using a metric downstream.

## Promotion procedure

A future change to a row must provide all of the following:

1. a new versioned schema when semantics change;
2. an exact claim and review of the affected source, tests, docs, and evidence;
3. held-out evidence satisfying every listed `promotion_requirement`;
4. an updated forbidden-claim audit;
5. independent review before any publication or control admission;
6. explicit authority for provider, hardware, actuation, or publication work.

Local evidence never supplies those external authorities.
