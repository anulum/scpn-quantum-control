# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Auto-Generated API Reference

# Auto-Generated API Reference

Generated from source docstrings via mkdocstrings.

This page is an advanced module index. It is useful for maintainers and
subsystem authors who need direct access to implementation modules. First-path
user workflows should start with [Stable Facades API](stable_facades_api.md)
and [Kuramoto Core Facade](kuramoto_core_facade.md).

## Documentation Surface TODO

The repository documentation-surface audit is tracked by
`tools/audit_documentation_surface.py`. As of 2026-05-18, after the Paper 0
generated-builder burn-down through generated builder 10-batch 25, the
remaining `scripts` audit inventory is:

- Total remaining findings: `734`
- Function findings: `697`
- Module findings: `37`
- Class findings: `0`

Continue from the next audit head:

- `scripts/build_paper0_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_specs.py`
- `scripts/build_paper0_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_specs.py`
- `scripts/build_paper0_section_2_specialised_sensory_systems_specs.py`
- `scripts/build_paper0_section_2_the_central_void_the_source_specs.py`
- `scripts/build_paper0_section_2_the_endocrine_system_and_hpa_axis_stress_response_specs.py`

Production continuation rule: process generated Paper 0 builder CLI entrypoints
in deterministic audit order, run focused documentation-surface audit, Ruff,
mypy, diff hygiene, public-token hygiene, and freeze checks for each batch
before staging or committing.

## Stable Facades

::: scpn_quantum_control.kuramoto_core
    options:
      members: [KuramotoProblem, build_kuramoto_problem, validate_kuramoto_inputs, compile_hamiltonian, compile_dense_hamiltonian, compile_trotter_circuit, measure_order_parameter, simulate_variant_trajectory]
      show_root_heading: true

## Advanced Module Reference

The sections below expose lower-level packages directly. Use them when extending
or debugging a subsystem, not as the default path for tutorial code.

## Bridge

::: scpn_quantum_control.bridge.knm_hamiltonian
    options:
      members: [OMEGA_N_16, build_knm_paper27, build_kuramoto_ring, knm_to_hamiltonian, knm_to_ansatz]

::: scpn_quantum_control.bridge.snn_adapter
    options:
      members: [spike_train_to_rotations, quantum_measurement_to_current, SNNQuantumBridge, ArcaneNeuronBridge]

::: scpn_quantum_control.bridge.ssgf_adapter
    options:
      members: [ssgf_w_to_hamiltonian, ssgf_state_to_quantum, quantum_to_ssgf_state, SSGFQuantumLoop]

## Phase

::: scpn_quantum_control.phase.xy_kuramoto
    options:
      members: [QuantumKuramotoSolver]

::: scpn_quantum_control.phase.kuramoto_variants
    options:
      members: [KuramotoVariant, KuramotoVariantResult, HigherOrderKuramotoSpec, MonitoredKuramotoSpec, PTSymmetricKuramotoSpec, build_triadic_ring_terms, simulate_higher_order_kuramoto, simulate_monitored_kuramoto, simulate_pt_symmetric_kuramoto]

::: scpn_quantum_control.phase.phase_vqe
    options:
      members: [PhaseVQE]

::: scpn_quantum_control.phase.trotter_upde
    options:
      members: [QuantumUPDESolver]

## Control

::: scpn_quantum_control.control.qaoa_mpc
    options:
      members: [QAOA_MPC]

::: scpn_quantum_control.control.q_disruption_iter
    options:
      members: [ITERFeatureSpec, normalize_iter_features, generate_synthetic_iter_data, from_fusion_core_shot, DisruptionBenchmark]

## QSNN

::: scpn_quantum_control.qsnn.qlif
    options:
      members: [QuantumLIFNeuron]

::: scpn_quantum_control.qsnn.qlayer
    options:
      members: [QuantumDenseLayer]

::: scpn_quantum_control.qsnn.training
    options:
      members: [QSNNTrainer]

## Differentiable Programming

::: scpn_quantum_control.differentiable
    options:
      members: [Parameter, ParameterBounds, ParameterShiftRule, DualNumber, ReverseNode, GradientResult, StochasticGradientResult, ShotAllocationResult, SparseMatrixResult, ImplicitSensitivityResult, FixedPointSensitivityResult, CustomDerivativeRule, CustomDerivativeCheckResult, OptimizationResult, ArmijoLineSearchResult, GradientCheckResult, JacobianResult, JVPResult, VJPResult, HessianResult, HVPResult, LeastSquaresCovarianceResult, FisherVectorProductResult, FisherConjugateGradientResult, NaturalGradientResult, NaturalGradientOptimizationResult, NaturalGradientOptimizer, LevenbergMarquardtDampingUpdate, LevenbergMarquardtOptimizer, LevenbergMarquardtResult, LevenbergMarquardtStep, LevenbergMarquardtTrial, WeightedGradientResult, DifferentiableOptimizer, dual_sin, dual_cos, dual_exp, dual_log, reverse_sin, reverse_cos, reverse_exp, reverse_log, armijo_backtracking_line_search, parameter_shift_gradient, value_and_parameter_shift_grad, parameter_shift_gradient_with_uncertainty, allocate_parameter_shift_shots, forward_mode_gradient, value_and_forward_mode_grad, reverse_mode_gradient, value_and_reverse_mode_grad, grad, value_and_grad, batch_parameter_shift_gradient, batch_value_and_parameter_shift_grad, finite_difference_gradient, value_and_finite_difference_grad, batch_value_and_finite_difference_grad, complex_step_gradient, value_and_complex_step_grad, batch_complex_step_gradient, batch_custom_jvp, batch_value_and_custom_jvp, batch_custom_vjp, batch_value_and_custom_vjp, batch_custom_jacobian, batch_value_and_custom_jacobian, batch_value_and_complex_step_grad, finite_difference_jacobian, value_and_finite_difference_jacobian, jacobian, value_and_jacobian, dense_to_sparse_matrix, sparse_jacobian, sparse_hessian, sparse_empirical_fisher_metric, finite_difference_jvp, value_and_finite_difference_jvp, batch_finite_difference_jvp, batch_value_and_finite_difference_jvp, finite_difference_vjp, batch_finite_difference_vjp, batch_value_and_finite_difference_vjp, vector_jacobian_product, batch_vector_jacobian_product, finite_difference_hessian, value_and_finite_difference_hessian, hessian, value_and_hessian, implicit_stationary_sensitivity, implicit_fixed_point_sensitivity, finite_difference_hvp, value_and_finite_difference_hvp, batch_finite_difference_hvp, batch_value_and_finite_difference_hvp, empirical_fisher_metric, empirical_fisher_vector_product, empirical_fisher_conjugate_gradient, evaluate_levenberg_marquardt_step, gauss_newton_gradient, huber_residual_weights, least_squares_covariance, levenberg_marquardt_step, natural_gradient, soft_l1_residual_weights, update_levenberg_marquardt_damping, weighted_gradient_sum, check_parameter_shift_consistency, check_custom_derivative_consistency, custom_jacobian, value_and_custom_jacobian, custom_jvp, value_and_custom_jvp, custom_vjp, value_and_custom_vjp, is_jax_autodiff_available, jax_value_and_grad]

## MLIR Compiler

::: scpn_quantum_control.compiler.mlir
    options:
      members: [MLIRCompileConfig, MLIRModule, compile_kuramoto_to_mlir]

## Real-Time Runtime

::: scpn_quantum_control.control.realtime_runtime
    options:
      members: [RealtimeRuntimeConfig, RealtimeTickRecord, RealtimeRunResult, VirtualRealtimeClock, MonotonicRealtimeClock, run_realtime_control_loop]

## Cloud-Native Deployment

::: scpn_quantum_control.deployment.cloud_native
    options:
      members: [ContainerResources, CloudDeploymentSpec, CloudManifestBundle, generate_cloud_manifests]

## Hardware Abstraction Layer

::: scpn_quantum_control.hardware.backends
    options:
      members: [QuantumBackendDescriptor, describe_hal_backend_profile, list_hal_backend_descriptors, describe_backend, list_quantum_backends]

::: scpn_quantum_control.hardware.provider_smoke
    options:
      members: [AggregatorProviderOptionalDependencyRow, ProviderOptionalDependencyRow, aggregator_provider_optional_dependency_matrix, main, provider_optional_dependency_matrix]

::: scpn_quantum_control.hardware.provider_capability_discovery
    options:
      members: [ProviderCapabilitySnapshot, ProviderCapabilityDecision, assess_provider_capability_snapshot, probe_aggregator_provider_capability, snapshot_from_azure_target, snapshot_from_braket_device, snapshot_from_dwave_solver, snapshot_from_iqm_backend, snapshot_from_ionq_backend, snapshot_from_oqc_target, snapshot_from_pasqal_target, snapshot_from_qiskit_runtime_backend, snapshot_from_qbraid_device, snapshot_from_quandela_processor, snapshot_from_quantinuum_backend, snapshot_from_quera_bloqade, snapshot_from_rigetti_qcs, snapshot_from_strangeworks_backend]

::: scpn_quantum_control.hardware.aggregators
    options:
      members: [AggregatorProviderRoute, ResolvedAggregatorProviderRoute, aggregator_provider_routes_for, built_in_aggregator_provider_routes, resolve_aggregator_provider_route]

::: scpn_quantum_control.hardware.hal
    options:
      members: [BackendCapabilities, BackendProfile, QuantumWorkload, QuantumJobRef, QuantumJobResult, QuantumBackend, LocalDeterministicSimulator, HardwareAbstractionLayer, built_in_backend_profiles]

::: scpn_quantum_control.hardware.hal_qiskit
    options:
      members: [QiskitAerHALAdapter, QiskitRuntimeHALAdapter, qiskit_circuit_to_workload, qiskit_circuit_to_qasm3_workload]

::: scpn_quantum_control.hardware.hal_braket
    options:
      members: [BraketLocalHALAdapter, BraketAwsHALAdapter, braket_circuit_to_workload]

::: scpn_quantum_control.hardware.hal_cirq
    options:
      members: [CirqLocalHALAdapter, cirq_circuit_workload]

::: scpn_quantum_control.hardware.hal_dwave
    options:
      members: [DWaveLeapHALAdapter, dwave_bqm_workload]

::: scpn_quantum_control.hardware.hal_azure
    options:
      members: [AzureQuantumHALAdapter, azure_openqasm3_to_workload]

::: scpn_quantum_control.hardware.hal_ionq
    options:
      members: [IonQCloudHALAdapter, ionq_qis_workload]

::: scpn_quantum_control.hardware.hal_iqm
    options:
      members: [IQMHALAdapter, iqm_qiskit_workload]

::: scpn_quantum_control.hardware.hal_oqc
    options:
      members: [OQCHALAdapter, oqc_openqasm3_workload]

::: scpn_quantum_control.hardware.hal_pasqal
    options:
      members: [PasqalPulserHALAdapter, pulser_sequence_workload]

::: scpn_quantum_control.hardware.hal_pennylane
    options:
      members: [PennyLaneDeviceHALAdapter, pennylane_gate_workload]

::: scpn_quantum_control.hardware.hal_qbraid
    options:
      members: [QbraidRuntimeHALAdapter, qbraid_program_to_workload]

::: scpn_quantum_control.hardware.hal_strangeworks
    options:
      members: [StrangeworksComputeHALAdapter, strangeworks_program_to_workload]

::: scpn_quantum_control.hardware.hal_quandela
    options:
      members: [QuandelaPercevalHALAdapter, quandela_perceval_workload]

::: scpn_quantum_control.hardware.hal_quera_bloqade
    options:
      members: [QuEraBloqadeHALAdapter, bloqade_ahs_workload]

::: scpn_quantum_control.hardware.hal_quantinuum
    options:
      members: [QuantinuumCloudHALAdapter, quantinuum_tket_workload]

::: scpn_quantum_control.hardware.hal_rigetti
    options:
      members: [RigettiQCSHALAdapter, rigetti_quil_workload]

## Identity

::: scpn_quantum_control.identity.ground_state
    options:
      members: [IdentityAttractor]

::: scpn_quantum_control.identity.coherence_budget
    options:
      members: [coherence_budget, fidelity_at_depth]

::: scpn_quantum_control.identity.entanglement_witness
    options:
      members: [chsh_from_statevector, disposition_entanglement_map]

::: scpn_quantum_control.identity.identity_key
    options:
      members: [identity_fingerprint, prove_identity, verify_identity]

::: scpn_quantum_control.identity.binding_spec
    options:
      members: [ARCANE_SAPIENCE_SPEC, ORCHESTRATOR_MAPPING, build_identity_attractor, solve_identity, quantum_to_orchestrator_phases, orchestrator_to_quantum_phases]

## Benchmarks

::: scpn_quantum_control.benchmarks.classical_baselines
    options:
      members: [ClassicalBaselineRun, available_baselines, scipy_ode_baseline, qutip_lindblad_baseline, mps_tebd_baseline, run_documented_classical_baselines]

::: scpn_quantum_control.benchmarks.quantum_advantage
    options:
      members: [AdvantageResult, classical_benchmark, quantum_benchmark, estimate_crossover, run_scaling_benchmark]

## QEC

::: scpn_quantum_control.qec.fault_tolerant
    options:
      members: [LogicalQubit, RepetitionCodeUPDE]

::: scpn_quantum_control.qec.surface_code_upde
    options:
      members: [SurfaceCodeSpec, SurfaceCodeUPDE]

## Mitigation

::: scpn_quantum_control.mitigation.pec
    options:
      members: [PECResult, pauli_twirl_decompose, pec_sample]

::: scpn_quantum_control.mitigation.zne
    options:
      members: [ZNEResult, gate_fold_circuit, zne_extrapolate]

## Hardware

::: scpn_quantum_control.hardware.trapped_ion
    options:
      members: [trapped_ion_noise_model, transpile_for_trapped_ion]

::: scpn_quantum_control.hardware.fast_classical
    options:
      members: [fast_sparse_evolution]

::: scpn_quantum_control.hardware.runner
    options:
      members: [HardwareRunner, JobResult]

## Analysis

::: scpn_quantum_control.analysis.sync_witness
    options:
      members: [WitnessResult, evaluate_all_witnesses]

::: scpn_quantum_control.analysis.witness_discovery
    options:
      members: [WitnessCandidate, WitnessDiscoverySpec, WitnessDiscoveryEvaluation, WitnessDiscoveryResult, WitnessSearchMode, discover_kuramoto_witnesses, score_witness_candidates]

::: scpn_quantum_control.analysis.quantum_persistent_homology
    options:
      members: [QuantumPHResult, quantum_persistent_homology, ph_sync_scan]

::: scpn_quantum_control.analysis.berry_phase
    options:
      members: [BerryPhaseResult, berry_phase_scan]

::: scpn_quantum_control.analysis.finite_size_scaling
    options:
      members: [FSSResult, finite_size_scaling]

::: scpn_quantum_control.analysis.krylov_complexity
    options:
      members: [KrylovResult, krylov_vs_coupling]

::: scpn_quantum_control.analysis.magic_nonstabilizerness
    options:
      members: [MagicResult, magic_vs_coupling]

## Crypto

::: scpn_quantum_control.crypto
    options:
      show_root_heading: true

## Applications

::: scpn_quantum_control.applications
    options:
      show_root_heading: true

::: scpn_quantum_control.applications.dataset_catalog
    options:
      members: [ApplicationBenchmarkDescriptor, list_application_benchmark_descriptors, get_application_benchmark_descriptor, load_application_benchmark_artifact, artifact_to_kuramoto_problem]

::: scpn_quantum_control.applications.app_plugins
    options:
      members: [ApplicationPluginBenchmark, ApplicationPluginRegistry, get_application_plugin, load_application_dataset, compile_application_problem, run_application_benchmark_suite]

## Gauge

::: scpn_quantum_control.gauge
    options:
      show_root_heading: true

## TCBO

::: scpn_quantum_control.tcbo
    options:
      show_root_heading: true

## PGBO

::: scpn_quantum_control.pgbo
    options:
      show_root_heading: true

## L16

::: scpn_quantum_control.l16
    options:
      show_root_heading: true
