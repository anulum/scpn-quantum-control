<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- symmetry-sector mitigation compiler -->

# Symmetry- and Sector-Aware Mitigation Compiler

This page defines the bounded first contract for the mitigation compiler lane.
The current implementation is a planner, not a circuit transformer. It decides
whether existing primitives are eligible for a Kuramoto/XY experiment descriptor
and returns explicit blockers when required evidence is missing.

Public API:

```python
from scpn_quantum_control.mitigation import (
    SymmetrySectorProblem,
    plan_symmetry_sector_mitigation,
)
```

## Inputs

`SymmetrySectorProblem` records:

- `n_qubits`
- symmetric finite `coupling_matrix`
- finite `omega`
- computational-basis `initial_state`
- measurement basis: `z`, `x`, `y`, `xyz`, or `counts`
- whether raw counts are available
- whether noise-scaled symmetry observables are available for GUESS

## Outputs

`SymmetrySectorPlan` records:

- `status`: `eligible` or `blocked`
- `expected_parity`
- eligible primitives: `parity_postselection`, `symmetry_expansion`, `guess_symmetry_decay`
- blockers
- required evidence
- benchmark gates
- claim boundary

## Failure modes

The planner blocks rather than guessing when:

- the coupling matrix is not square, finite, and symmetric;
- `omega` length or finiteness is invalid;
- `initial_state` is not a computational-basis bitstring of length `n_qubits`;
- raw counts are absent;
- GUESS is requested without noise-scaled symmetry observables.

## Benchmark gates

Before this planner is wired into execution paths, every integration must pass:

- `scpn-bench symmetry-sector-mitigation-gate`
- `scpn-bench sync-benchmark-gate`
- hardware result-pack verifier for any raw-count replay claim
- focused mitigation regression tests for the chosen primitive

## Claim boundary

Planner output is an eligibility contract only. It does not mutate circuits,
submit hardware jobs, prove improved hardware performance, or broaden DLA/GUESS
claims without benchmark and raw-count evidence.
