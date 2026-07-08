# Control Scope Boundary

SPDX-License-Identifier: AGPL-3.0-or-later

The `scpn_quantum_control.control` package is scoped to synchronisation-control
workflows that already exist in this repository:

- Kuramoto-XY feedback and set-point tracking.
- FRC pulsed-shot schedule scoring and QAOA sampling.
- Software-in-the-loop closed-loop response analysis.
- Petri-net, disruption, topology, and VQLS proxy surfaces that feed the same
  synchronisation and campaign-governance path.

It does not provide generic pulse-shape optimisation, provider-native pulse
calibration, hardware drift compensation, or lab-instrument control. Those
routes require backend calibration data, explicit live-execution approval,
provider-native control-flow or pulse APIs, and hardware evidence ledgers before
any public claim can be promoted.

## In Scope

| Surface | Role | Claim boundary |
|---|---|---|
| `control.realtime_feedback` | Software controller for finite-shot Kuramoto-XY feedback loops | Local simulation and exportable circuit templates; no hardware latency or pulse-control claim. |
| `control.closed_loop_analysis` | Response classification, latency budgets, and publication-package scaffolds | Software-in-the-loop evidence unless a live ticket and raw-count replay artefacts exist. |
| `control.qaoa_pulsed_cost` and `control.frc_pulsed_qaoa` | FRC schedule scoring and QAOA sampling against a control-grade surrogate | Schedule-selection support only; high-fidelity FRC physics and actuator calibration stay outside this package. |
| `control.qaoa_mpc`, `control.qpetri`, `control.q_disruption`, and topology-control adapters | Synchronisation, campaign, and supervisory control experiments | Simulator or no-submit campaign evidence unless a hardware-result pack promotes a specific row. |

## Out of Scope

The package must not be described as any of the following:

- A generic pulse-control or pulse-calibration product.
- A replacement for provider-native pulse compilation or drift calibration.
- A lab-control stack for arbitrary instruments.
- Evidence that pulse-level hardware control has been executed.
- Evidence that an actuator schedule improves plasma, chip, sensor, or device
  performance without the matching domain solver and hardware ledger row.

The pulse-related surfaces in this repository are intentionally separated:

- `phase.pulse_shaping` provides deterministic mathematical pulse envelopes and
  Rust/Python parity for those envelope formulas.
- `codegen.ultrascale_hls` emits manifest-bound HLS source artifacts for
  downstream FPGA review; it does not run synthesis or hardware.
- `hardware.openpulse_control` builds no-submit calibration-workflow dossiers;
  it does not submit provider pulse jobs or fit live backend drift.
- `control.*` consumes schedules, feedback histories, and surrogate objectives
  for synchronisation-control analysis.

## Promotion Rule

A control result can be promoted only when the evidence class matches the claim:

1. Simulator or software-in-loop claims need deterministic seeds, replayable
   parameters, and focused tests.
2. Provider-prepared dynamic-circuit claims need backend capability evidence and
   a provider-preparation artifact.
3. Live hardware claims need explicit approval, job IDs, raw counts,
   calibration snapshots, replay validation, and a hardware-result pack.
4. Pulse-level or instrument-control claims stay closed until a provider-native
   or lab-native execution artifact exists.

This boundary keeps `control/` focused on synchronisation and campaign
governance while leaving generic pulse-control work outside the project scope.
