# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 K_nm preregistered replay report

# GOTM-SCPN Paper 0 K_nm preregistered replay

- Paper: GOTM-SCPN Paper 0: The Foundational Framework
- Schema: `paper0_knm_preregistered_replay_v1`
- Status: `blocked_non_closing_preregistered_replay`
- Execution surface: offline deterministic replay; no QPU submission; no network dependency.

## Claim boundary

This replay is a deterministic, no-QPU preregistration artefact. It does not validate K_nm as a measured physical coupling law and does not authorise hardware submission.

## Inputs

- `primary_candidate`: `data/public_application_benchmarks/eeg_alpha_plv_8ch.json`
- `negative_control`: `data/public_application_benchmarks/ieee5bus_power_grid.json`
- `negative_measured_couplings`: `data/knm_physical_validation/measured_couplings_power_grid_ieee5bus.json`
- `preregistration`: `docs/paper0_first_preregistered_downstream_experiment.md`
- `pathway`: `docs/paper0_experimental_pathway.md`
- `lane_registry`: `docs/paper0_lane_registry.md`

## Reproducibility manifest

- Generator: `scripts/run_paper0_knm_preregistered_replay.py`
- Comparator: `scripts/compare_paper0_knm_preregistered_replay.py`
- Gate: `scpn-bench paper0-knm-preregistered-replay-gate`
- Floating-point policy: numpy float64 deterministic matrix diagnostics
- Randomness policy: fixed local permutation-null seeds; no global RNG state

Input digests:
- `primary_candidate`: `data/public_application_benchmarks/eeg_alpha_plv_8ch.json` sha256 `1a6e4e4369902eccaebd29a7eb960252e3aca851310a2e08f56922d78312db9f`
- `negative_control`: `data/public_application_benchmarks/ieee5bus_power_grid.json` sha256 `189c5db6dfc42faed529256ed4f6b7d879968900ad0e42a50d5f12dfb59af75c`
- `negative_measured_couplings`: `data/knm_physical_validation/measured_couplings_power_grid_ieee5bus.json` sha256 `c2df21d988b3d3bee763f80ecc76e3fe863fed147d34df647904feeb34d591fd`

## Primary candidate: EEG alpha PLV

- Source: `eeg_alpha_plv_8ch`
- Domain: `eeg`
- Matrix shape: `[8, 8]`
- Pearson upper-triangle correlation: `0.898892484379`
- Spearman upper-triangle correlation: `0.916002410296`
- Frobenius relative error: `0.860101256997`
- Candidate density: `1.0`
- Permutation-null seed: `2701`
- Permutation-null count: `512`
- Empirical two-sided null p-value: `0.001949317739`
- Observed-vs-null z-score: `4.598696045381`

Primary blockers:
- PLV values are dimensionless synchronisation observables, not calibrated coupling magnitudes.
- No per-edge uncertainty model is present in the public EEG candidate artefact.

## Negative control: IEEE 5-bus power grid

- Source: `ieee5bus_power_grid`
- Domain: `power-grid`
- Matrix shape: `[5, 5]`
- Pearson upper-triangle correlation: `0.226143607811`
- Spearman upper-triangle correlation: `0.190394327647`
- Frobenius relative error: `0.998081120431`
- Candidate density: `0.5`
- Permutation-null seed: `2702`
- Permutation-null count: `512`
- Empirical two-sided null p-value: `0.518518518519`
- Observed-vs-null z-score: `0.673781840801`
- Measured pairwise entries: `10`
- Entries with uncertainty: `10`
- Normalisation locked: `True`

The sparse power-grid control remains non-closing and prevents topology-only success from being treated as measured-system validation.

## Gates

- `named_system`: `pass`
- `units_and_normalisation`: `blocked_primary_dimensionless_plv`
- `pairwise_uncertainty`: `blocked_primary_missing_per_edge_uncertainty`
- `negative_control`: `pass_non_promotional_sparse_control_remains_non_closing`
- `qpu_submission`: `blocked_no_qpu_preregistration_lane`
- `claim_promotion`: `blocked_measured_system_gate_open`

## Promotion decision

- Decision: `do_not_promote`
- Hardware submission authorised: `False`
- Claim promotion authorised: `False`

Required evidence before reconsideration:
- calibrated EEG coupling magnitudes with source units
- per-edge uncertainty model for the primary measured-system candidate
- matched null-model battery across dense and sparse controls
- frozen analysis manifest reviewed before any QPU submission

Falsifiers:
- primary candidate remains dimensionless PLV-only after source audit
- negative control becomes indistinguishable from primary under the locked null battery
- input digest drift occurs without a new preregistration revision
- any hardware submission path is requested before measured-system gates close

## Next required artefacts

- measured EEG coupling-magnitude dataset with source units and per-edge uncertainty
- frozen analysis manifest for the selected measured-system replay
- null-model battery over matched sparse and dense candidate graphs
