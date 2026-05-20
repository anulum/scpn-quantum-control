# S4 IBM pulse-level Kuramoto-XY calibration-review preregistration

Job ID: `s4_ibm_pulse_calibration_review`

## Purpose
Prepare the IBM pulse-level route for human calibration review before any pulse job, backend session, or QPU spend.

## Hypothesis
If the exported circuit-QED pulse design matches a backend-supported IBM pulse control path after calibration review, it can become a bounded candidate for testing pulse-shaped Kuramoto-XY evolution against the digital Trotter route.

## Falsification Condition
The IBM pulse route is rejected for near-term execution if the selected backend does not expose compatible pulse controls, calibrated channels, timing units, or safe amplitude/duration bounds for the exported schedule.

## Expected Observables
- backend pulse-control capability status
- drive/control channel mapping for each coupled pair
- dt or duration-unit compatibility
- amplitude and duration bounds for the exported envelope
- transpiled digital comparator depth and two-qubit gate count
- post-review QPU time estimate before any execution approval

## Circuit / Package Summary
- `provider`: ibm_pulse
- `platform`: circuit_qed
- `native_schema`: exchange_resonator_v1
- `n_oscillators`: 4
- `n_couplers`: 6
- `n_drives`: 4
- `duration`: 0.2
- `sdk_module`: qiskit.pulse
- `sdk_available`: False

## QPU Budget
- `status`: not_requested
- `hardware_submission`: False
- `cloud_contact`: False
- `recommended_initial_scope`: calibration metadata review only
- `estimated_execution_seconds`: 0.0
- `future_optional_ceiling_seconds`: 300.0

## Platform Fit
- `ibm_pulse`: blocked_until_sdk_calibration_approval
- `gate_based_comparator`: required_before_execution
- `neutral_atom`: separate_S4_route

## Risks and Confounds
- Modern IBM Runtime access may not expose OpenPulse-style low-level controls on every backend.
- A design payload is not a calibrated Qiskit pulse Schedule or provider-accepted instruction.
- Pulse-shaped evolution and digital Trotter circuits are not equivalent without a matched observable protocol.
- Calibration drift can invalidate a pulse review before execution.
- Backend channel constraints can dominate any expected reduction in Trotter overhead.

## Decision Tree
- `accepted`: Promote to live backend metadata capture and a no-submit channel-map artefact.
- `manual_review`: Request backend-specific pulse constraints before choosing a hardware scope.
- `fail`: Reject IBM pulse execution for this route and prioritise digital or neutral-atom S4 paths.

## Paper Impact
A passed calibration review would support the S4 methods narrative by showing how the software stack moves from analogue design payloads to provider-specific hardware readiness.

## Follow-up Avenue
After review, generate a backend-specific pulse channel map, then compare a tiny pulse-shaped candidate against a matched digital comparator only after explicit QPU-budget approval.

## Possibilities Opened
- pulse-vs-digital Trotter overhead comparison
- provider-specific calibration package for future QPU credit requests
- bounded IBM pulse-level route before non-IBM S4 hardware replication

## Claim Boundary
This preregistration does not create a pulse Schedule, contact IBM services, submit QPU jobs, or claim pulse-level performance. It only defines the calibration-review gate.

## Reproducibility Package
- `s4_readiness`: data/s4_multi_hardware_control/s4_multi_hardware_readiness_2026-05-06.json
- `preregistration_script`: scripts/export_s4_provider_preregistration.py
- `bench_command`: scpn-bench s4-provider-preregistration
- `readiness_doc`: docs/campaigns/s4_multi_hardware_readiness_2026-05-06.md

## Prerequisites
- select a concrete IBM backend that exposes compatible pulse metadata
- capture backend calibration metadata without submitting jobs
- record channel map, duration unit, amplitude bounds, and timing granularity
- estimate QPU time after the review and before any hardware approval
