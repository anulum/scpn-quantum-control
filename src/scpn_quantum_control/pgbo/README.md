<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# `pgbo` — PGBO quantum bridge (phase-geometry bridge operator)

## What it is

The quantum counterpart of the SCPN framework's **Phase-Geometry Bridge
Operator**. The classical PGBO builds a metric tensor `h_μν` that maps
phase differences to geometric distances; this package extracts the same
geometry directly from the quantum state via the **quantum geometric
tensor** (QGT)

```
Q_μν = <∂_μ ψ|∂_ν ψ> − <∂_μ ψ|ψ><ψ|∂_ν ψ>,   ∂_μ = ∂/∂K_μ
```

whose real part is the Fubini–Study metric (quantum distance) and whose
imaginary part is the Berry curvature (geometric phase). Derivatives are
taken by parameter-shift on the coupling matrix `K_ij`.

Public surface (`compute_pgbo_tensor`, `PGBOResult`): the metric tensor
`h_μν = Re Q_μν`, the Berry curvature `F_μν = −2 Im Q_μν`, the metric
determinant (volume element), and the total curvature.

## Which paper

SCPN framework, PGBO layer (phase-geometry bridge). The classical operator
is the metric-tensor bridge between phase dynamics and geometry; this is
its quantum lift over the Kuramoto–XY Hamiltonian parameterised by `K_ij`.

## Wiring status — ACTIVE

- `analysis/qfi_geometric_crosscheck.py` calls `compute_pgbo_tensor` and
  checks the QFI relation `F = 4·Re(Q)` for pure states — a live
  cross-check between the geometric and Fisher-information paths.
- Cross-referenced by `tcbo/` and `l16/` and exercised by
  `tests/test_pgbo_bridge.py`.

## Claim boundary

Exact-diagonalisation ground state at small `n` (`hardware.classical.
classical_exact_diag`); derivatives are finite-difference parameter-shift,
not analytic. No hardware execution and no large-`n` claim: this is a
bounded geometric diagnostic, not a scalable QGT estimator.
