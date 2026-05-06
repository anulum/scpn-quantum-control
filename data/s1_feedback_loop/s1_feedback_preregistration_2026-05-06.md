# S1 Feedback Preregistration Manifest

Experiment ID: `s1_dynamic_feedback_preregistration_2026-05-06`

Submission state: prepared, no hardware submission.

## Circuit Summary
- `n_qubits`: 4
- `n_clbits`: 6
- `depth`: 41
- `operation_counts`: {'cx': 18, 'ry': 12, 'measure': 6, 'if_else': 6, 'PauliEvolution': 3}
- `has_mid_circuit_measurement`: True
- `has_conditional_control`: True
- `has_conditional_reset`: True
- `n_rounds`: 3

## Budget
- `circuits`: 2
- `shots_per_circuit`: 1024
- `repetitions`: 12
- `estimated_execution_seconds`: 24.0
- `queue_seconds`: 0.0
- `calibration_seconds`: 0.0
- `total_reserved_seconds`: 24.0

## Platform Readiness
- `IBM Heron dynamic-circuit backend`: ready (declared capabilities satisfy the dynamic-circuit payload)
- `Generic dynamic-circuit gate backend`: ready (declared capabilities satisfy the dynamic-circuit payload)
- `Neutral-atom analogue XY target`: manual_review (payload requires mid-circuit measurement; payload requires conditional rotations; payload requires conditional reset)
- `Continuous-variable analogue target`: manual_review (payload requires mid-circuit measurement; payload requires conditional rotations; payload requires conditional reset)
- `Local statevector simulator`: ready (declared capabilities satisfy the dynamic-circuit payload)

## Provider Dry-runs
- `ibm_runtime`: submission_enabled=False
- `openqasm3_gate`: submission_enabled=False
- `analog_native_review`: submission_enabled=False

## Capability Probe Examples
- `ibm_dynamic_metadata_template`: ready (backend metadata satisfies S1 dynamic-circuit requirements)
- `analog_native_review_template`: blocked (missing required feature: mid_circuit_measurement; missing required feature: conditional_control; missing required feature: conditional_reset)

## Hardware Job Dossier

Purpose: Test whether a monitored cross-shot feedback policy can steer the Kuramoto-XY synchronisation observable on hardware under a bounded dynamic-circuit payload.

Hypothesis: If the feedback loop survives hardware noise and provider-side dynamic-circuit execution, the observed live order parameter should move toward the preregistered target more often than a matched open-loop control at the same circuit family, shots, and layout.

Falsification condition: The feedback arm fails if it does not improve the target-order-parameter error relative to the matched open-loop control, or if transpilation/readout/latency overhead dominates so strongly that the feedback action is statistically indistinguishable from noise.

Claim boundary: This job cannot prove sub-microsecond real-time control unless feedback is implemented provider-side, cannot establish quantum advantage, and cannot generalise beyond the tested backend, layout, circuit family, and calibration window.

## Reproducibility Package
- `preregistration_manifest`: data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.json
- `raw_counts_path`: data/s1_feedback_loop/raw_counts/
- `analysis_script`: scripts/analyse_s1_feedback_hardware.py
- `latency_benchmark`: data/s1_feedback_loop/s1_feedback_loop_latency_summary_2026-05-06.json
- `claim_boundary_doc`: docs/hybrid_feedback_loop_s1_2026-05-06.md
