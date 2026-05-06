# S4 Multi-hardware Readiness

This is a no-submit readiness artefact. It does not contact cloud providers, reserve QPU time, or authorise hardware execution.

## Scientific Question
Can the Kuramoto-XY pulse/analogue programme family be represented as provider-specific no-submit payloads before any cross-vendor hardware run?

## Falsification Boundary
S4 provider promotion is blocked unless the selected provider SDK is available, calibration metadata are attached, live resource checks pass, and explicit budget approval is recorded.

## Provider Plans

### pulser
- Platform: `neutral_atoms`
- SDK module: `pulser`
- SDK available: `False`
- Native schema: `native_ahs_v1`
- Oscillators / couplers: `4` / `6`
- Readiness: `blocked_until_sdk_calibration_approval`
- Can submit: `False`
- Can execute: `False`
- Reason: `blocked_until_explicit_execution_approval`

### bloqade
- Platform: `neutral_atoms`
- SDK module: `bloqade`
- SDK available: `False`
- Native schema: `native_ahs_v1`
- Oscillators / couplers: `4` / `6`
- Readiness: `blocked_until_sdk_calibration_approval`
- Can submit: `False`
- Can execute: `False`
- Reason: `blocked_until_explicit_execution_approval`

### ibm_pulse
- Platform: `circuit_qed`
- SDK module: `qiskit.pulse`
- SDK available: `False`
- Native schema: `exchange_resonator_v1`
- Oscillators / couplers: `4` / `6`
- Readiness: `blocked_until_sdk_calibration_approval`
- Can submit: `False`
- Can execute: `False`
- Reason: `blocked_until_explicit_execution_approval`

## Promotion Gates
- provider account or research-credit route exists
- provider SDK object construction succeeds locally or in an approved emulator
- provider calibration metadata are recorded
- live transpilation or provider resource estimate is below the preregistered ceiling
- QPU budget is approved for a named provider and backend
- raw data, job identifiers, and analysis scripts are assigned before submission

## Blocked Claims
- no non-IBM hardware result
- no pulse-level calibration result
- no cross-vendor replication claim
- no hardware performance improvement claim
- no quantum-advantage claim

## Next Actions
- select one provider route with available credits
- construct provider SDK object in emulator-only mode after approval
- prepare a provider-specific preregistration dossier
- run budget-gated hardware only after live readiness checks
