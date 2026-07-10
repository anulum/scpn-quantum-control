<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# `tcbo` — TCBO quantum observer (topological coherence bridge observer)

## What it is

The quantum counterpart of the SCPN framework's **Topological Coherence
Bridge Observer**. The classical TCBO reads persistent-homology Betti
numbers (`β₀`, `β₁`) off the oscillator phase configuration; this package
measures the topological content directly from the quantum state.

Public surface (`compute_tcbo_observables`, `TCBOResult`):

- `p_h1` — vortex density as a persistent-`H₁` proxy (empirical/open
  threshold target `p_h1 = 0.72`);
- `tee` — topological entanglement entropy
  `S_topo = S_A + S_B + S_C − S_AB − S_BC − S_AC + S_ABC`
  (Kitaev–Preskill / Levin–Wen 2006), which distinguishes trivial
  (`TEE = 0`) from topological (`TEE ≠ 0`) phases at the XY/BKT
  transition;
- `string_order` — the SPT string order parameter
  `O_string = <Z_i · Π_k X_k · Z_j>`;
- `β₀`/`β₁` proxies.

## Which paper

SCPN framework, TCBO layer (topological coherence bridge). TEE definition
after Kitaev–Preskill and Levin–Wen (2006); vortex density via the gauge
`vortex_detector`.

## Wiring status — ACTIVE

- `analysis/qfi_geometric_crosscheck.py` and the `analysis` package import
  the observer; `gauge/vortex_detector.measure_vortex_density` feeds the
  `p_h1` proxy.
- Cross-referenced by `pgbo/` and `l16/`; exercised by
  `tests/test_tcbo_observer_edge_cases.py`,
  `tests/test_tcbo_weighted_complex_validation.py`, and
  `tests/test_gauge_topology_contracts.py`.

## Claim boundary

Small-`n` exact-diagonalisation state; Betti numbers are **proxies**
(vortex density, connected-components estimate), not computed persistent
homology, and `p_h1 = 0.72` is an open empirical target, not a proven
bound. No hardware execution.
