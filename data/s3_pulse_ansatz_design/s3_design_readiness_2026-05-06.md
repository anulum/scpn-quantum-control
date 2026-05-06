# S3 Pulse / Ansatz Design Readiness

Protocol ID: `s3_ml_augmented_pulse_ansatz_design_2026-05-06`

Submission state: no hardware submission; deterministic candidate scoring only.

## Objective
Rank small Kuramoto-XY pulse and ansatz candidates by deterministic resource and analytic-error proxies before any ML training or QPU use.

## Candidate Scores

- `ansatz_shallow_knm` (ansatz): score=66.0; metrics={"depth": 12, "hardware_submission": false, "n_qubits": 4, "parameters": 0, "size": 24, "two_qubit_gates": 12}
- `ansatz_mid_knm` (ansatz): score=129.0; metrics={"depth": 22, "hardware_submission": false, "n_qubits": 4, "parameters": 0, "size": 44, "two_qubit_gates": 24}
- `pulse_stirap_balanced` (pulse): score=208.08085712700696; metrics={"hardware_submission": false, "infidelity_bound": 208.00085712700695, "max_points_per_pulse": 200, "n_qubits": 4, "pulse_count": 6, "total_time": 0.2}
- `pulse_demkov_kunike` (pulse): score=218.31276115302487; metrics={"hardware_submission": false, "infidelity_bound": 218.23276115302485, "max_points_per_pulse": 200, "n_qubits": 4, "pulse_count": 6, "total_time": 0.2}

## Forbidden Claims
- No learned optimiser is demonstrated by this readiness gate.
- No pulse-level hardware improvement is established without provider calibration data.
- No quantum advantage or backend-independent performance claim is permitted.

## Required Follow-ups
- Train or evaluate the ML surrogate on generated candidate rows with held-out checks.
- Compare promoted ansatz candidates against VQE or observable targets.
- Run provider-specific pulse feasibility checks before any pulse submission.
- Attach hardware-job dossiers before QPU or pulse-level execution.
