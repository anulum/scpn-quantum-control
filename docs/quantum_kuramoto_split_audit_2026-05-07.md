# S6 Quantum-Kuramoto Split Audit

This is a first-pass import-graph and marker audit for a future decoupled `quantum-kuramoto` package. It does not create or publish a second package.

## Status Counts
- Reusable: `44`
- Needs review: `5`
- SCPN-specific: `36`

## Boundary
- Safe to publish now: `False`
- Reason: first-pass import audit only; no package skeleton or publish workflow yet.

## Reusable Candidates
- `scpn_quantum_control.phase.__init__` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.adapt_vqe` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.adiabatic_preparation` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.ancilla_lindblad` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.avqds` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.contraction_optimiser` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.floquet_kuramoto` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.gpu_batch_vqe` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.jax_nqs` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.lindblad` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.lindblad_engine` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.mps_evolution` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.nqs_ansatz` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.param_shift` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.phase_vqe` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.pulse_shaping` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.qsvt_evolution` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.structured_ansatz` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.tensor_jump` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.trotter_error` — core_kuramoto_candidate
- `scpn_quantum_control.phase.varqite` — no_scpn_specific_marker_detected
- `scpn_quantum_control.phase.xy_compiler` — core_kuramoto_candidate
- `scpn_quantum_control.phase.xy_kuramoto` — core_kuramoto_candidate
- `scpn_quantum_control.bridge.phase_artifact` — no_scpn_specific_marker_detected
- `scpn_quantum_control.bridge.qpu_data_artifact` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware._experiment_helpers` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.backends` — hardware_core_candidate
- `scpn_quantum_control.hardware.circuit_export` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.cirq_adapter` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.experiments` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.fast_classical` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.gpu_accel` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.jax_accel` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.noise_model` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.pennylane_adapter` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.plugin_registry` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.pulse_feasibility` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.qasm_export` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.qcvv` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.qiskit_compat` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.qubit_mapper` — no_scpn_specific_marker_detected
- `scpn_quantum_control.hardware.trapped_ion` — no_scpn_specific_marker_detected
- `scpn_quantum_control.accel.julia.__init__` — no_scpn_specific_marker_detected
- `scpn_quantum_control.accel.rust_import` — acceleration_candidate

## Review or Exclusion Rows
- `scpn_quantum_control.phase.ansatz_bench` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.phase.ansatz_methodology` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.phase.backend_selector` — `needs_review` — imports_non_foundation_scpn_module
- `scpn_quantum_control.phase.cross_domain_transfer` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.phase.kuramoto_variants` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.phase.trotter_upde` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.bridge.__init__` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.bridge.control_plasma_knm` — `scpn_specific` — module_name_contains_scpn_specific_marker
- `scpn_quantum_control.bridge.knm_hamiltonian` — `scpn_specific` — source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.bridge.orchestrator_adapter` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker
- `scpn_quantum_control.bridge.orchestrator_feedback` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.bridge.sc_to_quantum` — `scpn_specific` — module_name_contains_scpn_specific_marker
- `scpn_quantum_control.bridge.snn_adapter` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.bridge.snn_backward` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.bridge.sparse_hamiltonian` — `needs_review` — imports_non_foundation_scpn_module
- `scpn_quantum_control.bridge.spn_to_qcircuit` — `scpn_specific` — source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.bridge.ssgf_adapter` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker
- `scpn_quantum_control.bridge.ssgf_w_adapter` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.__init__` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.analog_kuramoto` — `scpn_specific` — source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.hardware.async_runner` — `scpn_specific` — source_contains_scpn_specific_marker, imports_non_foundation_scpn_module, hardware_core_candidate
- `scpn_quantum_control.hardware.circuit_cutting` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.classical` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.cutting_runner` — `scpn_specific` — source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.hardware.experiment_control` — `scpn_specific` — source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.hardware.experiment_dynamics` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.experiment_mitigation` — `scpn_specific` — source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.hardware.experiment_vqe` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.feedback_capability_probe` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.feedback_dryrun` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.feedback_hardware_scheduler` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.feedback_loop` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.hardware.feedback_provider_metadata` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.feedback_submission` — `scpn_specific` — module_name_contains_scpn_specific_marker, source_contains_scpn_specific_marker, imports_non_foundation_scpn_module
- `scpn_quantum_control.hardware.hybrid_digital_analog` — `needs_review` — imports_non_foundation_scpn_module
- `scpn_quantum_control.hardware.job_dossier` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.hardware.provenance` — `needs_review` — imports_non_foundation_scpn_module
- `scpn_quantum_control.hardware.runner` — `needs_review` — imports_non_foundation_scpn_module, hardware_core_candidate
- `scpn_quantum_control.accel.__init__` — `scpn_specific` — source_contains_scpn_specific_marker
- `scpn_quantum_control.accel.dispatcher` — `scpn_specific` — source_contains_scpn_specific_marker, acceleration_candidate
- `scpn_quantum_control.accel.rust_kuramoto_classical` — `scpn_specific` — source_contains_scpn_specific_marker, acceleration_candidate
