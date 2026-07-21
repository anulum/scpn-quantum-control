<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IQM Square-Lattice Layout-Transfer Preregistration -->

# IQM Square-Lattice Layout-Transfer Preregistration

Date: 2026-07-21

This preregistration commits the design, decision rule, and budget for the
cross-topology layout-transfer campaign promised in our 2026-07-16 IQM
research-credits request (extended Resonance access granted 2026-07-20,
ticket IQMCS-1691). It does not submit an IQM job and does not authorise
credit spend; execution requires a separate explicit per-submit approval and
a completed harness readiness gate.

## Scientific Question

Our fidelity-aware layout tooling — the DynQ community-detection analysis
pass (`hardware/dynq_layout_pass.py`), the Kuramoto-XY-aware discrete layout
cost model (`hardware/kuramoto_layout_cost.py`), and the discrete Kuramoto
layout optimiser (`hardware/kuramoto_layout_optimiser.py`) — was built and
benchmarked on IBM heavy-hex topologies. IQM Garnet is a 20-qubit square
lattice.

> Do the fidelity gains of calibration-aware Kuramoto layout optimisation
> transfer from IBM heavy-hex to IQM square-lattice topology, or are they
> topology-specific?

This is an open, publishable question either way, and it is the "why IQM
specifically" lane of the vetted credits request: square lattice is the
natural second data point for topology-transfer evidence.

## Design

Circuit family: Kuramoto-XY Trotter chains built by the committed campaign
builders, scheduled with the two-edge-colour width-2 schedule
(`analysis/two_colour_schedule.py`, classical evidence
`docs/campaigns/two_colour_width2_schedule_2026-07-21.md`) so the two-qubit
depth per Trotter step is constant in `n` and the physics (excitation-number
conservation) is preserved exactly.

| Field | Value |
|-------|-------|
| Device | IQM Garnet (20 qubits, square lattice) via Resonance |
| Sizes | chains `n = 8, 12, 16` |
| Depths | fixed preregistered rep count per size, identical across arms |
| Arms per size | 3 (see below) |
| Main shots | `2048` per (size, arm) |
| Readout states | `0…0`, `1…1` per size, `1024` shots each |

Arms (identical circuits, different qubit placement):

1. `optimised` — layout chosen by the discrete Kuramoto layout optimiser
   with the cost model fed by live Garnet calibration data;
2. `default` — Qiskit-on-IQM automatic transpiler placement;
3. `naive` — preregistered linear chain along consecutive physical indices.

Circuit count: `3 sizes × 3 arms = 9` main + `6` readout = `15` circuits.
Shot count: `18,432` main + `6,144` readout = `24,576` shots.

Classical reference: exact statevector order parameter for every (size,
depth) — free, computed and committed before submission.

## Primary Endpoint and Decision Rule (frozen)

Observable: readout-corrected order-parameter error
`err(arm, n) = |R_hw(arm, n) − R_exact(n)|`.

- Primary: paired one-sided comparison `err(optimised) < err(default)`
  across the three sizes — the optimiser arm must win at all three sizes AND
  the bootstrap 90 % confidence interval of the pooled error difference
  (10,000 resamples of the shot-level counts, committed script) must exclude
  zero.
- Secondary: `err(default)` versus `err(naive)` (does automatic placement
  matter at all on square lattice); per-size Wilson intervals; two-qubit
  gate-count and transpiled-depth per arm (the optimiser must not win by
  simply compiling shallower circuits — depth parity within 10 % is a
  validity gate, not an outcome).
- Transfer claim granted only if the primary rule passes. If it fails,
  publish the bounded negative: calibration-aware layout gains measured on
  heavy-hex do not transfer to this square-lattice device at this power —
  with the same prominence a positive result would receive.

## Harness Readiness Gate (blocks submission)

This lane needs one new committed component before any spend: a Garnet
square-lattice adapter for the layout cost model (topology graph + live
calibration ingestion via the Resonance metadata endpoint — metadata queries
only, no QPU spend). The adapter and the full 15-circuit matrix must pass:

- unit tests to the repository coverage standard;
- an `IQMFakeGarnet` dry run of all arms from committed code only;
- a committed exact-baseline artefact for all sizes;
- the depth-parity validity gate across arms on the fake backend.

## Budget and Stop Rules

- Submit the `n = 8` block (3 main + 2 readout circuits) alone first; read
  the actual credit burn from the Resonance dashboard before continuing.
- Abort if the `n = 8` block consumes more than one quarter of the currently
  visible credit allowance.
- No submissions beyond this matrix without a fresh preregistration.

## Claim Boundary

Blocked regardless of outcome: quantum advantage (all circuits are
classically simulable at these sizes — that is what makes the exact
reference possible); any coherence-protection claim; extrapolation beyond
the sampled device, calibration window, chain lengths, and schedule; any
modification of the frozen submissions under `paper/submissions/` (results
feed a new manuscript only). IQM and IQM Resonance are credited in every
resulting output.

## Output Artefacts

- `data/iqm_layout_transfer/iqm_layout_transfer_<timestamp>_{plan,executed}.json`;
- `data/iqm_layout_transfer/iqm_layout_transfer_analysis_<date>.{json,md}`;
- exact-baseline artefact `data/iqm_layout_transfer/exact_reference_<date>.json`;
- raw job identifiers public; access token never committed;
- campaign manifest `docs/campaigns/iqm_layout_transfer_manifest_<date>.md`.

## Submission Boundary

This preregistration is complete once committed. QPU execution remains
blocked until the harness readiness gate passes and the owner grants a
separate explicit per-submit GO. Submission order relative to the powered
DLA backend-sensitivity block
(`iqm_dla_backend_sensitivity_powered_prereg_2026-07-21.md`) is decided
after the first executed block's measured credit burn.
