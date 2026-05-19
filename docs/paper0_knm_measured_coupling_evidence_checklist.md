<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
(c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
(c) Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
scpn-quantum-control -- Paper 0 K_nm measured-coupling evidence checklist
-->

# GOTM-SCPN Paper 0 K_nm measured-coupling evidence checklist

This checklist defines the minimum evidence package required before the first
Paper 0 K_nm replay can be reconsidered for measured-system promotion. It is
deliberately stricter than the current replay output because the current EEG
candidate is dimensionless PLV and does not contain per-edge uncertainty.

## Current status

- Current replay status: `blocked_non_closing_preregistered_replay`
- Current promotion decision: `do_not_promote`
- Hardware submission: blocked
- Claim promotion: blocked
- Primary blocker: EEG alpha PLV is a synchronisation observable, not a
  calibrated coupling magnitude.
- Secondary blocker: the primary candidate lacks per-edge uncertainty.

## Required evidence before reconsideration

| Evidence item | Acceptance condition | Failure mode |
|---|---|---|
| Named measured system | Source dataset identifies the biological or physical system, channel map, acquisition context, and preprocessing provenance. | Anonymous or partially described matrix cannot be promoted. |
| Coupling units | Each K_nm edge has source units or a documented transformation into a calibrated coupling scale. | Dimensionless association matrix remains a candidate observable only. |
| Unit-class gate | The measured-system promotion audit classifies the unit as a calibrated or model-derived coupling magnitude. | PLV, coherence, correlation, mutual information, transfer entropy, or unknown unit classes remain non-promotional observables. |
| Per-edge uncertainty | Each retained edge has uncertainty, standard error, posterior interval, or equivalent uncertainty metadata. | Point estimates without uncertainty keep the measured-system gate closed. |
| Normalisation lock | Normalisation is specified before replay and cannot be adjusted after seeing diagnostics. | Post-hoc normalisation invalidates the preregistration. |
| Negative controls | Sparse and dense controls are defined with the same analysis code path and frozen thresholds. | Topology-only agreement can be mistaken for physical validation. |
| Null-model battery | Permutation or resampling tests are frozen with seeds, counts, and interpretation thresholds. | Diagnostic significance cannot be audited. |
| Frozen manifest | Dataset paths, digests, seeds, environment assumptions, and scripts are recorded before promotion review. | Input drift or hidden replay changes invalidate comparison. |
| Claim boundary | Promotion text states exactly what is validated and what remains unvalidated. | Broad K_nm or hardware claims remain blocked. |

## Minimum promotion review packet

A future promotion review must include:

- Updated preregistration document.
- Machine-readable replay JSON with digest-locked inputs.
- Human-readable replay report.
- Contract-check output for the replay schema version.
- Comparator output showing no drift.
- Null-model summary for primary and control systems.
- Explicit falsifier evaluation.
- Signed decision preserving the no-QPU boundary unless hardware gates are
  separately authorised.

The contract-check command for the current replay is:

```bash
PYTHONPATH=src ./.venv-linux/bin/python scripts/export_paper0_knm_replay_contract.py \
  --check-replay data/paper0_knm_preregistered_replay.json
```

The measured-candidate release gate for the committed EEG and power-grid audit
artefacts is:

```bash
scpn-bench knm-measured-candidate-gate
```

It fails if any committed candidate is promoted, changes its expected edge
count, drops the strict unit-class decision, or records K_nm physical
validation as closed.

## Non-promotional outputs

The following outputs are useful but not sufficient for measured-system
promotion:

- Matrix correlation against the built-in 16-layer candidate.
- A stable SHA-256 input manifest.
- A passing no-QPU replay gate.
- A dimensionless PLV matrix.
- A power-grid negative control with uncertainties.
- A report that reproduces exactly from committed artefacts.

These outputs improve reproducibility. They do not close the measured-system
gate by themselves.
