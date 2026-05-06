# S6 Quantum-Kuramoto Boundary Review

This review converts the import-graph audit into a conservative package-boundary decision. It does not create a `quantum_kuramoto` package skeleton.

## Decision
- Package skeleton allowed: `False`
- Reason: manual boundary review still requires refactors for config/provenance/analysis-dependent rows

## Proposed Public API Surface
- `quantum_kuramoto.phase.xy_kuramoto` from `scpn_quantum_control.phase.xy_kuramoto` (`reusable`)
- `quantum_kuramoto.phase.xy_compiler` from `scpn_quantum_control.phase.xy_compiler` (`reusable`)
- `quantum_kuramoto.phase.trotter_error` from `scpn_quantum_control.phase.trotter_error` (`reusable`)
- `quantum_kuramoto.phase.structured_ansatz` from `scpn_quantum_control.phase.structured_ansatz` (`reusable`)
- `quantum_kuramoto.phase.phase_vqe` from `scpn_quantum_control.phase.phase_vqe` (`reusable`)
- `quantum_kuramoto.phase.pulse_shaping` from `scpn_quantum_control.phase.pulse_shaping` (`reusable`)
- `quantum_kuramoto.phase.lindblad` from `scpn_quantum_control.phase.lindblad` (`reusable`)
- `quantum_kuramoto.phase.mps_evolution` from `scpn_quantum_control.phase.mps_evolution` (`reusable`)
- `quantum_kuramoto.hardware.backends` from `scpn_quantum_control.hardware.backends` (`reusable`)
- `quantum_kuramoto.hardware.async_runner` from `scpn_quantum_control.hardware.async_runner` (`scpn_specific`)
- `quantum_kuramoto.hardware.qubit_mapper` from `scpn_quantum_control.hardware.qubit_mapper` (`reusable`)
- `quantum_kuramoto.hardware.qasm_export` from `scpn_quantum_control.hardware.qasm_export` (`reusable`)
- `quantum_kuramoto.hardware.circuit_export` from `scpn_quantum_control.hardware.circuit_export` (`reusable`)
- `quantum_kuramoto.hardware.analog_kuramoto` from `scpn_quantum_control.hardware.analog_kuramoto` (`scpn_specific`)
- `quantum_kuramoto.hardware.plugin_registry` from `scpn_quantum_control.hardware.plugin_registry` (`reusable`)
- `quantum_kuramoto.accel.rust_import` from `scpn_quantum_control.accel.rust_import` (`reusable`)

## Needs-Review Decisions
- `scpn_quantum_control.phase.backend_selector` — `defer` — depends on analysis-sector helpers outside the proposed lightweight package — refactor: move sector summaries behind optional adapter or duplicate minimal selector logic
- `scpn_quantum_control.bridge.sparse_hamiltonian` — `defer` — depends on analysis.magnetisation_sectors — refactor: extract magnetisation-sector primitive into the reusable boundary
- `scpn_quantum_control.hardware.hybrid_digital_analog` — `promote_after_facade` — scientifically reusable, but imports kuramoto_core facade from the parent package — refactor: route through quantum_kuramoto public facade after skeleton exists
- `scpn_quantum_control.hardware.provenance` — `defer` — depends on parent package configuration semantics — refactor: define package-local provenance config or keep provenance in parent package
- `scpn_quantum_control.hardware.runner` — `defer` — depends on parent config, logging setup, provenance, and mitigation modules — refactor: start with AsyncHardwareRunner and add full runner only after config/provenance split

## Compatibility Requirements
- existing scpn_quantum_control imports must remain unchanged
- quantum_kuramoto must not import SCPN-specific SSGF/SNN/orchestrator modules
- runner skeleton must start with async/backend abstractions before parent config is split
- separate pyproject is allowed only after import-compatibility tests pass

## Next Steps
- add API-surface contract tests for proposed exports
- extract or wrap magnetisation-sector primitive if sparse_hamiltonian is promoted
- define package-local config/provenance policy before promoting hardware.runner
- create skeleton only after the above blockers are closed
