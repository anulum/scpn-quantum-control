# S6 Quantum-Kuramoto API Contract

This contract validates the proposed `quantum_kuramoto` export surface before any package skeleton is created.

## Summary
- Contract passed: `True`
- Package skeleton allowed: `False`
- Proposed exports: `16`
- Importable source modules: `16`
- Immediately promotable exports: `14`
- Warnings: `2`
- Errors: `0`

## Export Rows
- `quantum_kuramoto.phase.xy_kuramoto` from `scpn_quantum_control.phase.xy_kuramoto`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`14`
- `quantum_kuramoto.phase.xy_compiler` from `scpn_quantum_control.phase.xy_compiler`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`6`
- `quantum_kuramoto.phase.trotter_error` from `scpn_quantum_control.phase.trotter_error`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`11`
- `quantum_kuramoto.phase.structured_ansatz` from `scpn_quantum_control.phase.structured_ansatz`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`5`
- `quantum_kuramoto.phase.phase_vqe` from `scpn_quantum_control.phase.phase_vqe`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`8`
- `quantum_kuramoto.phase.pulse_shaping` from `scpn_quantum_control.phase.pulse_shaping`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`16`
- `quantum_kuramoto.phase.lindblad` from `scpn_quantum_control.phase.lindblad`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`5`
- `quantum_kuramoto.phase.mps_evolution` from `scpn_quantum_control.phase.mps_evolution`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`8`
- `quantum_kuramoto.hardware.backends` from `scpn_quantum_control.hardware.backends`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`22`
- `quantum_kuramoto.hardware.async_runner` from `scpn_quantum_control.hardware.async_runner`: importable=`True`, status=`scpn_specific`, promotable=`False`, public_members=`11`
- `quantum_kuramoto.hardware.qubit_mapper` from `scpn_quantum_control.hardware.qubit_mapper`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`12`
- `quantum_kuramoto.hardware.qasm_export` from `scpn_quantum_control.hardware.qasm_export`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`13`
- `quantum_kuramoto.hardware.circuit_export` from `scpn_quantum_control.hardware.circuit_export`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`12`
- `quantum_kuramoto.hardware.analog_kuramoto` from `scpn_quantum_control.hardware.analog_kuramoto`: importable=`True`, status=`scpn_specific`, promotable=`False`, public_members=`29`
- `quantum_kuramoto.hardware.plugin_registry` from `scpn_quantum_control.hardware.plugin_registry`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`4`
- `quantum_kuramoto.accel.rust_import` from `scpn_quantum_control.accel.rust_import`: importable=`True`, status=`reusable`, promotable=`True`, public_members=`3`

## Warnings
- scpn_quantum_control.hardware.async_runner has status scpn_specific; require facade or isolation before promotion
- scpn_quantum_control.hardware.analog_kuramoto has status scpn_specific; require facade or isolation before promotion

## Next Steps
- extract package-local facades for non-reusable proposed rows
- add import-compatibility tests before skeleton creation
- create the skeleton only after contract_passed remains true and boundary blockers are closed
