# S4 neutral-atom Kuramoto-XY provider-object preregistration

Job ID: `s4_neutral_atom_provider_object_review`

## Purpose
Prepare Pulser and Bloqade neutral-atom object-construction routes for local or approved-emulator review before any cloud provider session or hardware spend.

## Hypothesis
If the neutral-atom payload can be converted into provider SDK objects under local unit and geometry constraints, it becomes the most plausible S4 path for testing native XY-like evolution without digital Trotter overhead.

## Falsification Condition
The route is rejected for near-term execution if the register geometry, Rydberg interaction constraints, SDK object construction, or emulator-only resource estimate cannot realise the n=4 Kuramoto-XY payload without changing the preregistered observable family.

## Expected Observables
- provider SDK availability and version
- register geometry and minimum spacing validity
- Rydberg or AHS interaction coefficient compatibility
- local provider-object construction status
- emulator-only resource estimate where an approved emulator exists
- matched digital comparator observable definition

## Circuit / Package Summary
- `providers`: pulser,bloqade
- `platform`: neutral_atoms
- `pulser_schema`: native_ahs_v1
- `bloqade_schema`: native_ahs_v1
- `n_oscillators`: 4
- `n_couplers`: 6
- `duration`: 0.2
- `pulser_sdk_available`: False
- `bloqade_sdk_available`: False

## QPU Budget
- `status`: not_requested
- `hardware_submission`: False
- `cloud_contact`: False
- `recommended_initial_scope`: local SDK-object construction or approved emulator only
- `estimated_execution_seconds`: 0.0
- `future_optional_ceiling_seconds`: 600.0

## Platform Fit
- `pulser`: blocked_until_sdk_calibration_approval
- `bloqade`: blocked_until_sdk_calibration_approval
- `ibm_pulse`: separate_S4_route
- `gate_based_comparator`: required_before_execution

## Risks and Confounds
- Neutral-atom native interactions are not identical to the gate-model XY Hamiltonian without a mapped observable protocol.
- Register geometry and interaction signs can force a different effective coupling matrix.
- Provider SDK object construction is not cloud execution and cannot establish hardware performance.
- Local emulator success can be classically spoofable and must be separated from QPU evidence.
- Provider cost and credit access can dominate practical execution feasibility.

## Decision Tree
- `accepted`: Promote to local provider-object construction with versioned SDK metadata.
- `manual_review`: Request provider-specific geometry, units, and emulator constraints before promotion.
- `fail`: Reject neutral-atom S4 execution for this payload and keep the route as design-only evidence.

## Paper Impact
A passed provider-object review would support a future S4 hardware-methods section by showing that the analogue Kuramoto compiler can target neutral-atom SDKs before hardware allocation.

## Follow-up Avenue
Construct local Pulser and Bloqade objects in emulator-only mode, then prepare a budgeted cloud provider request only if geometry, units, and comparator observables pass review.

## Possibilities Opened
- native neutral-atom Kuramoto-XY execution planning
- digital-vs-analogue Trotter-overhead comparison
- non-IBM S4 credit application evidence
- provider-neutral artefact package for cross-vendor reproducibility

## Claim Boundary
This preregistration does not import provider SDK constructors, run emulators, contact cloud providers, submit jobs, or claim neutral-atom hardware performance.

## Reproducibility Package
- `s4_readiness`: data/s4_multi_hardware_control/s4_multi_hardware_readiness_2026-05-06.json
- `preregistration_script`: scripts/export_s4_neutral_atom_preregistration.py
- `bench_command`: scpn-bench s4-neutral-atom-preregistration
- `readiness_doc`: docs/campaigns/s4_multi_hardware_readiness_2026-05-06.md

## Prerequisites
- install or verify provider SDK versions in an isolated environment
- construct provider objects locally without cloud credentials
- record register geometry and unit-conversion assumptions
- define matched digital comparator observables before hardware execution
- obtain provider credit and explicit QPU-budget approval before cloud submission
