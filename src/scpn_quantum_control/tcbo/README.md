<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# `tcbo` — TCBO quantum observer (topological coherence bridge observer)

## What it is

The small-system proxy surface for the SCPN framework's **Topological Coherence
Bridge Observer**. It exact-diagonalises the XY Kuramoto Hamiltonian and
aggregates state and gauge diagnostics. It does not compute persistent homology
or measure topological content directly. The separate coupling-weighted
persistent-homology reconstruction lives in
`analysis/tcbo_weighted_complex.py`.

Public surface (`compute_tcbo_observables`, `TCBOResult`):

- `p_h1` — gauge vortex density as an `H₁`-labelled proxy (empirical/open
  threshold target `p_h1 = 0.72`);
- `tee` — a seven-term entropy inclusion-exclusion proxy in bits,
  `S_topo = S_A + S_B + S_C − S_AB − S_BC − S_AC + S_ABC`
  over three contiguous qubit-index regions; this is inspired by the
  Kitaev–Preskill construction but does not certify topological order;
- `string_order` — the Pauli-string expectation
  `O_string = <Z_i · Π_k X_k · Z_j>`;
- `β₀`/`β₁`-labelled proxies, which are independently defined and need not sum
  to one.

## Which paper

SCPN framework, TCBO layer (topological coherence bridge). The entropy
inclusion-exclusion form is adapted from Kitaev–Preskill and Levin–Wen (2006);
vortex density comes from the gauge `vortex_detector`.

## Wiring status — CALLABLE LIBRARY SURFACE

- `tcbo/__init__.py` exports the two public symbols, and the project root exports
  the `tcbo` package module.
- `compute_tcbo_observables` depends on
  `hardware.classical.classical_exact_diag`,
  `gauge.vortex_detector.measure_vortex_density`, and the entropy/partial-trace
  helpers in `analysis.quantum_phi`.
- No production control or analysis pipeline imports this observer. Callers opt
  into it explicitly; the weighted-complex analysis is a separate surface.
- The observer is exercised by `tests/test_tcbo_observer.py`,
  `tests/test_tcbo_observer_edge_cases.py`, and
  `tests/test_gauge_topology_contracts.py`.

## Claim boundary

Small-`n` exact-diagonalisation state; Betti-labelled values and the entropy
inclusion-exclusion value are **proxies**, not computed persistent-homology or
certified topological-order quantities. `p_h1 = 0.72` is an open empirical
target, not a proven bound. No hardware execution.
