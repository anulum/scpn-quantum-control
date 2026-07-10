<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# `l16` — L16 quantum director (Lyapunov monitoring for cybernetic closure)

## What it is

The quantum counterpart of the SCPN framework's **Layer-16 director**,
which provides cybernetic closure by monitoring system stability. The
quantum L16 extracts stability indicators from the quantum-evolved state
and turns them into a control decision.

Public surface (`compute_l16_lyapunov`, `L16Result`):

- **Loschmidt echo** `L(t) = |<ψ(0)|ψ(t)>|²` — state-return probability,
  whose decay bounds the quantum Lyapunov exponent;
- **fidelity susceptibility** `χ_F = −∂²F/∂ε²` — peaks at quantum phase
  transitions;
- **energy variance** `ΔE² = <H²> − <H>²` — low = near-eigenstate/stable;
- **order-parameter rate** `dR/dt` from consecutive snapshots;
- a composite `stability_score` and an `action` ∈ {`continue`, `adjust`,
  `halt`}.

## Which paper

SCPN framework, Layer 16 (cybernetic closure / stability governor). The
`L1–L16` layer hierarchy is the same one the coupling matrix
`build_knm_paper27` encodes (see the `L1–L16` cross-hierarchy boost in the
README K_nm description).

## Wiring status — ACTIVE

- `bridge/orchestrator_feedback.compute_orchestrator_feedback` calls
  `compute_l16_lyapunov` and maps its `order_parameter`/`stability_score`/
  `action` onto the orchestrator's advance / hold / rollback decision —
  this is the package's live consumer.
- Cross-referenced by `pgbo/` and `tcbo/`; exercised by
  `tests/test_l16_director.py`, `tests/test_orchestrator_feedback.py`, and
  `tests/test_control_module_contracts.py`.

## Claim boundary

Small-`n` exact-diagonalisation / dense `expm` evolution
(`hardware.classical`, `hardware.gpu_accel.expm`); the quantum Lyapunov
exponent is a **bound from Loschmidt decay**, not a measured spectrum. The
`action` is a heuristic threshold decision, not a certified controller. No
hardware execution.
