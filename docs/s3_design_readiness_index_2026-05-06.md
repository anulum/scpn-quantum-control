<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — S3 Design Readiness Index -->

# S3 Design Readiness Index

S3 starts as a no-QPU readiness gate for ML-augmented pulse and ansatz design.
The current slice does not train a model and does not submit hardware jobs. It
ranks deterministic candidate families so that later ML surrogate training has a
validated artefact schema and a strict claim boundary.

Canonical command:

```bash
scpn-bench s3-design-ready
```

Generated artefacts:

- `data/s3_pulse_ansatz_design/s3_design_readiness_2026-05-06.json`
- `data/s3_pulse_ansatz_design/s3_design_readiness_2026-05-06.md`

Current candidate families:

- `ansatz`: Kuramoto-XY structured circuits scored by depth, size, and two-qubit gate proxy.
- `pulse`: hypergeometric Trotter-step pulse schedules scored by analytic infidelity and pulse-count proxy.

Allowed claims:

- Candidate rows are reproducible no-QPU design proxies.
- The artefact schema is ready for later ML surrogate training and held-out validation.
- No provider-specific pulse feasibility or hardware improvement has been established.

Forbidden claims:

- No learned optimiser has been demonstrated yet.
- No pulse-level hardware improvement is established without backend calibration data.
- No quantum advantage or backend-independent performance claim is permitted from this readiness gate.

Next S3 work:

1. Add a held-out surrogate-training harness over generated candidate rows.
2. Compare promoted ansatz candidates against VQE or observable targets.
3. Add provider-specific pulse feasibility probes before any pulse-level submission.
4. Attach hardware-job dossiers before QPU or pulse-level execution.

## Surrogate rehearsal

The first no-QPU surrogate rehearsal is regenerated with:

```bash
scpn-bench s3-design-surrogate
```

It expands the deterministic candidate grid across several small system sizes,
fits a closed-form ridge linear surrogate to proxy scores, and reports held-out
and per-family metrics. This is deliberately a rehearsal over proxy scores, not
evidence of a hardware pulse improvement or VQE improvement.

## Ansatz observable validation

Promoted ansatz candidates are checked against exact no-QPU observables with:

```bash
scpn-bench s3-ansatz-observables
```

The validation reports exact statevector energy expectation, exact dense ground
energy, energy error, and a simple synchronisation proxy for the lowest-resource
ansatz candidates. It is an observable sanity check, not VQE optimisation and
not a hardware result.

## Pulse feasibility probe

Provider metadata can be checked against the S3 hypergeometric pulse schedule
without opening provider sessions:

```bash
scpn-bench s3-pulse-feasibility
```

The probe reports ready, blocked, manual-review, or unknown decisions from
metadata only. It does not calibrate pulses, submit jobs, or establish hardware
performance.
