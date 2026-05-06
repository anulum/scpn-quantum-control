# S3 promoted ansatz observable follow-up run

Job ID: `s3_ansatz_observable_followup`

## Purpose
Test whether the promoted structured Kuramoto-XY ansatz candidate retains its no-QPU observable behaviour under real backend transpilation, layout, and noise.

## Hypothesis
If the structured candidate is hardware-usable, its measured observable proxies should remain directionally consistent with the exact statevector baseline under a matched simulator and backend layout package.

## Falsification Condition
The candidate is falsified for hardware follow-up if transpilation depth, layout noise, or readout effects destroy the observed energy/synchronisation ordering relative to matched baseline candidates.

## Expected Observables
- energy expectation estimate
- synchronisation proxy
- transpiled depth and two-qubit gate count
- layout and readout metadata
- matched simulator comparison

## Circuit / Package Summary
- `candidate_label`: ansatz_d1_c0p75
- `n_qubits`: 3
- `depth`: 5
- `size`: 9
- `two_qubit_gates`: 3
- `parameters`: {'trotter_depth': 1, 'time_step': 0.1, 'coupling_scale': 0.75}

## QPU Budget
- `status`: not_requested
- `recommended_initial_scope`: transpile-only plus optional small shot-count observable run after approval
- `estimated_execution_seconds`: 0.0
- `hardware_submission`: False

## Platform Fit
- `gate_based`: manual_review_required
- `pulse_level`: not_required_for_this_ansatz_dossier
- `analogue`: separate native formulation required

## Risks and Confounds
- The current observable row is not VQE optimisation.
- Energy estimation may require many Pauli measurements if promoted beyond simulation.
- The shallow resource winner may not be the best accuracy winner after noise.
- Backend layout and readout can dominate the observable proxy.

## Decision Tree
- `positive`: Promote to a small hardware observable comparison against matched ansatz baselines.
- `null`: Keep as software-only design evidence and improve candidate generation.
- `negative`: Reject this resource-ranked candidate for hardware follow-up and use accuracy-ranked candidates instead.

## Paper Impact
Would support the S3 design-methods narrative by connecting no-QPU candidate selection to a bounded hardware-readiness package.

## Follow-up Avenue
Provider-specific transpilation probes, then small observable runs only after budget approval.

## Possibilities Opened
- budget-justified ansatz hardware screening
- layout-aware candidate promotion
- observable-driven candidate rejection before expensive QPU use

## Claim Boundary
This dossier does not authorise submission and does not claim VQE improvement, pulse-level performance, quantum advantage, or backend-independent behaviour.

## Reproducibility Package
- `ansatz_validation`: data/s3_pulse_ansatz_design/s3_ansatz_observable_validation_2026-05-06.json
- `dossier_script`: scripts/export_s3_hardware_dossiers.py
- `bench_command`: scpn-bench s3-hardware-dossiers

## Prerequisites
- run transpile-only backend feasibility checks
- attach exact measurement grouping before hardware execution
- obtain explicit QPU-budget approval


# S3 pulse-schedule feasibility follow-up run

Job ID: `s3_pulse_feasibility_followup`

## Purpose
Test whether the hypergeometric Kuramoto-XY pulse schedule can be mapped to a provider-supported pulse-control or native-XY execution path after calibration review.

## Hypothesis
If provider timing, pulse-count, and native interaction constraints are satisfied, a small calibrated pulse follow-up can test whether pulse shaping reduces digital Trotter overhead for the same observable family.

## Falsification Condition
The pulse route is falsified for near-term hardware if provider calibration, timing granularity, pulse duration, or native interaction constraints cannot realise the schedule within the documented budget and error boundary.

## Expected Observables
- provider pulse feasibility status
- calibrated duration and sample spacing
- pulse-count and qubit-count fit
- matched digital ansatz observable comparison
- post-calibration error budget

## Circuit / Package Summary
- `schedule_family`: hypergeometric_trotter_step
- `n_qubits`: 4
- `pulse_count`: 6
- `total_time`: 0.2
- `min_sample_spacing`: 0.0010050251256281326
- `ready_targets`: ibm_pulse_metadata_template, neutral_atom_xy_review_template

## QPU Budget
- `status`: not_requested
- `recommended_initial_scope`: provider calibration review only; no pulse job before approval
- `estimated_execution_seconds`: 0.0
- `hardware_submission`: False

## Platform Fit
- `ibm_pulse_metadata_template`: ready
- `neutral_atom_xy_review_template`: ready

## Risks and Confounds
- Metadata readiness is not calibration readiness.
- Pulse schedules may require provider-specific waveform constraints not captured in metadata.
- Native-XY analogue execution is not equivalent to a gate-level pulse schedule without a separate mapping.
- The current analytic infidelity proxy is not a measured hardware error model.

## Decision Tree
- `positive`: Prepare a provider-specific calibrated pulse preregistration package.
- `manual_review`: Request platform-specific waveform constraints before any execution.
- `negative`: Keep S3 pulse work as simulation-only and prioritise gate-based ansatz validation.

## Paper Impact
Would define the bridge from S3 design methods to S4 multi-hardware pulse-level control.

## Follow-up Avenue
S4 provider-specific pulse adapter and calibrated no-submit waveform package.

## Possibilities Opened
- analogue/native XY follow-up planning
- pulse-vs-digital Trotter comparison
- provider-specific QPU budget justification

## Claim Boundary
This dossier is metadata-only and does not calibrate pulses, open provider sessions, submit jobs, or establish hardware performance.

## Reproducibility Package
- `pulse_feasibility`: data/s3_pulse_ansatz_design/s3_pulse_feasibility_summary_2026-05-06.json
- `dossier_script`: scripts/export_s3_hardware_dossiers.py
- `bench_command`: scpn-bench s3-hardware-dossiers

## Prerequisites
- obtain provider-specific waveform and calibration constraints
- run no-submit waveform compilation
- obtain explicit QPU-budget approval
