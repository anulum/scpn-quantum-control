# Auto-Generated API Reference

Generated from source docstrings via mkdocstrings.

This page is an advanced module index. It is useful for maintainers and
subsystem authors who need direct access to implementation modules. First-path
user workflows should start with [Stable Facades API](stable_facades_api.md)
and [Kuramoto Core Facade](kuramoto_core_facade.md).

## Stable Facades

::: scpn_quantum_control.kuramoto_core
    options:
      members: [KuramotoProblem, build_kuramoto_problem, validate_kuramoto_inputs, compile_hamiltonian, compile_dense_hamiltonian, compile_trotter_circuit, compile_analog_program, compile_hybrid_program, measure_order_parameter, simulate_variant_trajectory]
      show_root_heading: true

## Advanced Module Reference

The sections below expose lower-level packages directly. Use them when extending
or debugging a subsystem, not as the default path for tutorial code.

## Bridge

::: scpn_quantum_control.bridge.knm_hamiltonian
    options:
      members: [OMEGA_N_16, omega_for_oscillators, build_knm_paper27, build_kuramoto_ring, knm_to_hamiltonian, knm_to_ansatz]

::: scpn_quantum_control.bridge.qpu_data_artifact
    options:
      members: [QPUDataArtifact, artifact_from_arrays, artifact_to_kuramoto_problem, validate_qpu_data_artifact, read_qpu_data_artifact, write_qpu_data_artifact]

::: scpn_quantum_control.bridge.scpn_upde_edge
    options:
      members: [SCPNUPDEEdge, build_paper27_scpn_upde_edge, build_scpn_upde_edge, validate_scpn_upde_edge_payload]

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

::: scpn_quantum_control.phase.coupling_learning
    options:
      members: [CouplingGradientVerificationResult, CouplingLearningResult, coupling_matrix_from_edge_vector, learn_couplings_from_observations, verify_coupling_parameter_shift_gradient]

::: scpn_quantum_control.phase.coupling_time_series_recovery
    options:
      members: [COUPLING_RECOVERY_CLAIM_BOUNDARY, COUPLING_RECOVERY_EVIDENCE_CLASS, CouplingRecoveryBoundaryRow, CouplingRecoveryCase, CouplingRecoveryRecord, CouplingRecoverySuiteResult, coupling_recovery_boundary_rows, default_coupling_recovery_cases, inject_time_series_noise_and_missing, recover_kuramoto_couplings_from_time_series, recover_xy_couplings_from_pair_energy_series, run_coupling_recovery_suite, simulate_kuramoto_phase_time_series, simulate_xy_pair_energy_time_series]

::: scpn_quantum_control.phase.synchronisation_witness
    options:
      members: [SYNC_WITNESS_CLAIM_BOUNDARY, SYNC_WITNESS_EVIDENCE_CLASS, PhaseCloudRegime, SyncWitnessBoundaryRow, SyncWitnessCase, SyncWitnessRecord, SyncWitnessSuiteResult, betti_curve, default_sync_witness_cases, geodesic_phase_distance_matrix, harmonic_order_parameter, phase_cloud_synchronisation_witness, run_sync_witness_suite, sync_witness_boundary_rows, vietoris_rips_persistence]

::: scpn_quantum_control.phase.differentiable_audit
    options:
      members: [DifferentiableQuantumAuditReport, DifferentiableWorkflowAuditSuiteResult, FiniteShotGradientAuditResult, MLFrameworkGradientAuditRecord, MLFrameworkGradientAuditSuiteResult, ParameterShiftAnalyticAgreement, PhaseGradientBenchmarkSuiteResult, run_differentiable_workflow_audit_suite, run_finite_shot_gradient_uncertainty_audit, run_known_phase_gradient_audit, run_ml_framework_gradient_audit, run_parameter_shift_audit_suite, run_phase_gradient_benchmark_suite, verify_parameter_shift_analytic_gradient]

::: scpn_quantum_control.phase.gradient_descent
    options:
      members: [ParameterShiftTrainingStep, ParameterShiftTrainingResult, ParameterShiftTrainingCertificate, parameter_shift_gradient_descent, validate_parameter_shift_training]

::: scpn_quantum_control.phase.natural_gradient
    options:
      members: [NaturalGradientRegularizationPolicy, NaturalGradientDirection, ParameterShiftNaturalGradientStep, ParameterShiftNaturalGradientResult, ParameterShiftNaturalGradientCertificate, solve_natural_gradient_direction, parameter_shift_natural_gradient_descent, validate_natural_gradient_training]

::: scpn_quantum_control.phase.trainability
    options:
      members: [TRAINABILITY_CLAIM_BOUNDARY, TrainabilityGradientSample, AdaptiveShotAllocationDryRun, BarrenPlateauTrainabilityReport, run_barren_plateau_trainability_report]

::: scpn_quantum_control.phase.optimizer_audit
    options:
      members: [OptimizerConvergenceRecord, OptimizerComparisonSuiteResult, run_parameter_shift_optimizer_comparison]

::: scpn_quantum_control.phase.optimizer_convergence_suite
    options:
      members: [GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY, GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS, KnownGroundStateObjective, GroundStateConvergenceCertificate, GroundStateOptimizerRunRecord, GroundStateOptimizerBoundaryRow, GroundStateOptimizerConvergenceSuiteResult, default_ground_state_optimizer_objectives, run_ground_state_optimizer_convergence_suite]

::: scpn_quantum_control.phase.open_system_objectives
    options:
      members: [OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY, OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS, BoundedOpenSystemObjectiveCase, DensityMatrixInvariantCertificate, MCWFReproducibilityCertificate, OpenSystemObjectiveRecord, OpenSystemObjectiveBoundaryRow, OpenSystemObjectiveSuiteResult, certify_density_matrix_invariants, certify_mcwf_reproducibility, default_open_system_objective_cases, evaluate_lindblad_objective, evaluate_mcwf_objective, open_system_objective_boundary_rows, run_open_system_objective_suite]

::: scpn_quantum_control.phase.objectives
    options:
      members: [ObjectiveTermValue, ObjectiveGradientEvaluation, ObjectiveTerm, ComposedPhaseObjective, ComposedObjectiveTrainingStep, ComposedObjectiveTrainingResult, ComposedObjectiveTrainingCertificate, phase_energy_term, phase_fidelity_target_term, periodic_regularization_term, phase_symmetry_penalty_term, smooth_box_safety_penalty_term, build_phase_control_objective, train_composed_phase_objective, validate_composed_objective_training, kuramoto_order_parameter, kuramoto_order_parameter_gradient, kuramoto_order_parameter_target_term, phase_locking_target_term, cluster_synchronisation_target_term, build_synchronisation_objective]

::: scpn_quantum_control.phase.objective_audit
    options:
      members: [ComposedObjectiveGradientAgreement, ComposedObjectiveAuditSuiteResult, verify_composed_objective_gradient, run_composed_objective_audit_suite]

::: scpn_quantum_control.phase.objective_planner
    options:
      members: [ComposedObjectiveExecutionPlan, ComposedObjectivePlannerAuditResult, plan_composed_objective_execution, assert_composed_objective_execution_supported, run_composed_objective_planner_audit]

::: scpn_quantum_control.phase.gradient_backend
    options:
      members: [QuantumGradientBackendCapability, QuantumGradientPlan, QuantumGradientRejectedMethod, QuantumGradientShotPolicy, QuantumGradientMethodExplanation, quantum_gradient_backend_capability, plan_quantum_gradient_backend, explain_quantum_gradient_method]

::: scpn_quantum_control.phase.gradient_support_matrix
    options:
      members: [GradientSupportCapability, GradientSupportPlan, GradientSupportMatrixAuditResult, gradient_support_capability, list_gradient_support_capabilities, plan_gradient_support, assert_gradient_support, run_gradient_support_matrix_audit]

::: scpn_quantum_control.phase.transform_nesting
    options:
      members: [GradientTransformNestingPlan, GradientTransformNestingAuditResult, plan_gradient_transform_nesting, assert_gradient_transform_nesting_supported, run_gradient_transform_nesting_audit]

::: scpn_quantum_control.phase.gradient_tape
    options:
      members: [GRADIENT_TAPE_CONTRACT_CLAIM_BOUNDARY, TapeGradientRecord, GradientTapeContractCheck, GradientTapeContractAuditResult, QuantumGradientTape, gradient_tape, run_gradient_tape_contract_audit]

::: scpn_quantum_control.phase.provider_gradient_audit
    options:
      members: [ProviderGradientReadinessScenario, ProviderGradientReadinessRecord, ProviderGradientReadinessAuditResult, default_provider_gradient_readiness_scenarios, run_provider_gradient_readiness_audit]

::: scpn_quantum_control.phase.trotter_upde
    options:
      members: [QuantumUPDESolver]

## Studio

::: scpn_quantum_control.studio.recompute_kernel
    options:
      members: [XYCompileRecomputeUnit, build_xy_compile_recompute_unit, canonical_xy_compile_input_bytes, verify_xy_compile_recompute_unit, xy_compile_digest_python]

## Control

::: scpn_quantum_control.control.qaoa_mpc
    options:
      members: [QAOA_MPC]

::: scpn_quantum_control.control.q_disruption_iter
    options:
      members: [ITERFeatureSpec, normalize_iter_features, generate_synthetic_iter_data, from_fusion_core_shot, scpn_control_bridge_dependency_contract, validate_scpn_control_bridge_dependency_contract, DisruptionBenchmark]

## QSNN

::: scpn_quantum_control.qsnn.qlif
    options:
      members: [QuantumLIFNeuron]

::: scpn_quantum_control.qsnn.qlayer
    options:
      members: [QuantumDenseLayer]

::: scpn_quantum_control.qsnn.training
    options:
      members: [QSNNTrainer, QSNNTrainingDiagnostics, QSNNTrainingRun, QSNNParameterShiftDescentRun]

## Differentiable Programming

::: scpn_quantum_control.diff
    options:
      members: [ShotPolicy, EstimatorProvenance, BackendCapabilityMetadata, DifferentiableCircuitDiagnostics, DifferentiableCircuit, JITExplanation, DifferentiableCircuitContractCheck, DifferentiableCircuitContractAuditResult, differentiable_circuit, jit_or_explain, run_differentiable_circuit_contract_audit, supported_transforms, namespace_metadata]

::: scpn_quantum_control.differentiable
    options:
      members: [Parameter, ParameterBounds, ParameterShiftRule, ParameterShiftSampleRecord, DualNumber, ReverseNode, GradientResult, StochasticGradientResult, ShotAllocationResult, SparseMatrixResult, ImplicitSensitivityResult, FixedPointSensitivityResult, CustomDerivativeRule, PrimitiveIdentity, PrimitiveTransformRule, CustomDerivativeRegistry, CustomDerivativeCheckResult, OptimizationResult, ArmijoLineSearchResult, GradientCheckResult, JacobianResult, JVPResult, VJPResult, HessianResult, HVPResult, LeastSquaresCovarianceResult, FisherVectorProductResult, FisherConjugateGradientResult, NaturalGradientResult, NaturalGradientOptimizationResult, NaturalGradientOptimizer, LevenbergMarquardtDampingUpdate, LevenbergMarquardtOptimizer, LevenbergMarquardtResult, LevenbergMarquardtStep, LevenbergMarquardtTrial, WeightedGradientResult, WholeProgramTraceEvent, WholeProgramIRNode, WholeProgramADResult, WholeProgramBytecodeInstruction, WholeProgramBytecodeBasicBlock, WholeProgramSourceIRFeature, WholeProgramSourceRegion, WholeProgramSourceBytecodeLineMap, WholeProgramSymbolScopeEntry, WholeProgramUnsupportedSemanticDiagnostic, WholeProgramSemanticsReport, WholeProgramCompilerFrontendReport, ProgramADAdjointStep, TraceADArray, TraceADScalar, DifferentiableOptimizer, dual_sin, dual_cos, dual_exp, dual_log, reverse_sin, reverse_cos, reverse_exp, reverse_log, armijo_backtracking_line_search, register_custom_derivative_rule, register_primitive_transform_rule, register_primitive_batching_rule, custom_derivative_rule_for, registered_custom_jvp, registered_custom_vjp, registered_custom_jacobian, multi_frequency_parameter_shift_rule, parameter_shift_gradient, value_and_parameter_shift_grad, parameter_shift_gradient_with_uncertainty, allocate_parameter_shift_shots, forward_mode_gradient, value_and_forward_mode_grad, reverse_mode_gradient, value_and_reverse_mode_grad, grad, value_and_grad, whole_program_grad, whole_program_value_and_grad, program_adjoint_result, program_adjoint_gradient, program_adjoint_grad, program_adjoint_value_and_grad, batch_parameter_shift_gradient, batch_value_and_parameter_shift_grad, finite_difference_gradient, value_and_finite_difference_grad, batch_value_and_finite_difference_grad, complex_step_gradient, value_and_complex_step_grad, batch_complex_step_gradient, batch_custom_jvp, batch_value_and_custom_jvp, batch_custom_vjp, batch_value_and_custom_vjp, batch_custom_jacobian, batch_value_and_custom_jacobian, batch_value_and_complex_step_grad, finite_difference_jacobian, value_and_finite_difference_jacobian, jacobian, value_and_jacobian, jacfwd, value_and_jacfwd, jacrev, value_and_jacrev, dense_to_sparse_matrix, sparse_jacobian, sparse_hessian, sparse_empirical_fisher_metric, finite_difference_jvp, value_and_finite_difference_jvp, batch_finite_difference_jvp, batch_value_and_finite_difference_jvp, finite_difference_vjp, batch_finite_difference_vjp, batch_value_and_finite_difference_vjp, vector_jacobian_product, batch_vector_jacobian_product, finite_difference_hessian, value_and_finite_difference_hessian, hessian, value_and_hessian, implicit_stationary_sensitivity, implicit_fixed_point_sensitivity, finite_difference_hvp, value_and_finite_difference_hvp, batch_finite_difference_hvp, batch_value_and_finite_difference_hvp, empirical_fisher_metric, empirical_fisher_vector_product, empirical_fisher_conjugate_gradient, evaluate_levenberg_marquardt_step, gauss_newton_gradient, huber_residual_weights, least_squares_covariance, levenberg_marquardt_step, natural_gradient, soft_l1_residual_weights, update_levenberg_marquardt_damping, weighted_gradient_sum, check_parameter_shift_consistency, check_custom_derivative_consistency, program_ad_linalg_trace_derivative_rule, program_ad_linalg_diag_derivative_rule, program_ad_linalg_diagflat_derivative_rule, program_ad_linalg_matrix_power_derivative_rule, program_ad_linalg_multi_dot_derivative_rule, program_ad_linalg_solve_derivative_rule, program_ad_linalg_eigvals_derivative_rule, program_ad_linalg_eigvalsh_derivative_rule, program_ad_linalg_svdvals_derivative_rule, program_ad_linalg_pinv_derivative_rule, custom_gauss_newton_gradient, custom_levenberg_marquardt_step, custom_jacobian, value_and_custom_jacobian, custom_jvp, value_and_custom_jvp, custom_vjp, value_and_custom_vjp, is_jax_autodiff_available, vmap, whole_program_grad, whole_program_value_and_grad, jax_value_and_grad]

::: scpn_quantum_control.differentiable_framework_overlay
    options:
      members: [FrameworkOverlayManifest, FrameworkOverlayVerification, build_framework_overlay_manifest, install_framework_overlay, verify_framework_overlay_path, main]

::: scpn_quantum_control.differentiable_module_hardening_audit
    options:
      members: [DifferentiableModuleHardeningAuditResult, DifferentiableModuleHardeningRecord, differentiable_module_hardening_registry, run_differentiable_module_hardening_audit]

::: scpn_quantum_control.differentiable_transform_algebra
    options:
      members: [TransformAlgebraAudit, TransformAlgebraCase, run_transform_algebra_audit, assert_transform_algebra_audit_passes]

::: scpn_quantum_control.differentiable_benchmark_report
    options:
      members: [DifferentiableBenchmarkReport, build_differentiable_benchmark_report]

::: scpn_quantum_control.benchmarks.differentiable_evidence
    options:
      members: [BenchmarkIsolationMetadata, DifferentiableBenchmarkEvidenceBundle, capture_host_load, infer_heavy_jobs_running, read_cpu_frequency_mhz, read_cpu_governor, write_differentiable_benchmark_evidence_bundle]

::: scpn_quantum_control.benchmarks.differentiable_hardening_gate
    options:
      members: [DifferentiableBenchmarkClassificationCase, DifferentiableHardeningGateCheck, DifferentiableHardeningSliceGateResult, run_differentiable_hardening_slice_gate]

::: scpn_quantum_control.phase.tensorflow_maintenance
    options:
      members: [PhaseTensorFlowMaintenanceReport, PhaseTensorFlowMaintenanceRoute, run_tensorflow_maintenance_decision]

::: scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan
    options:
      members: [DifferentiableIsolatedBenchmarkPlan, DifferentiableIsolatedBenchmarkPlanRow, DifferentiableIsolatedBenchmarkPlanValidation, render_differentiable_isolated_benchmark_plan_markdown, run_differentiable_isolated_benchmark_plan, validate_differentiable_isolated_benchmark_plan]

::: scpn_quantum_control.benchmarks.differentiable_optimizer_convergence
    options:
      members: [GROUND_STATE_OPTIMIZER_CONVERGENCE_SCHEMA, GroundStateOptimizerConvergenceArtifact, ground_state_optimizer_convergence_payload, render_ground_state_optimizer_convergence_markdown, write_ground_state_optimizer_convergence_artifact]

::: scpn_quantum_control.benchmarks.open_system_objective_evidence
    options:
      members: [OPEN_SYSTEM_OBJECTIVE_EVIDENCE_SCHEMA, OpenSystemObjectiveEvidenceArtifact, open_system_objective_evidence_payload, render_open_system_objective_evidence_markdown, write_open_system_objective_evidence_artifact]

::: scpn_quantum_control.benchmarks.differentiable_catalyst_comparison
    options:
      members: [CATALYST_UNSUPPORTED_PROVIDER_ROUTES, CatalystCompilerWorkflowComparison, catalyst_compiler_workflow_comparison]

::: scpn_quantum_control.benchmarks.differentiable_external_comparison
    options:
      members: [ExternalComparisonArtifact, ExternalComparisonRow, IdenticalCircuitGradientComparisonArtifact, IdenticalCircuitGradientComparisonRow, run_differentiable_external_comparison_suite, run_identical_circuit_gradient_comparison_suite, write_differentiable_external_comparison, write_identical_circuit_gradient_comparison]

::: scpn_quantum_control.phase.pennylane_provider_plugin
    options:
      members: [PennyLaneHardwarePluginExecutionArtifact, PennyLaneProviderEvidenceBundle, PennyLanePluginMatrixResult, PennyLanePluginMatrixRoute, PennyLaneProviderGradientParityArtifact, PennyLaneProviderPluginExecutionArtifact, run_pennylane_plugin_matrix]

::: scpn_quantum_control.differentiable_claim_ledger
    options:
      members: [ClaimLedger, ClaimLedgerRow, ClaimLedgerValidation, DifferentiableSupportSurfaceAlignment, load_differentiable_claim_ledger, load_differentiable_support_surface_alignment, render_claim_ledger_markdown, render_differentiable_support_surface_alignment_markdown, validate_claim_ledger, validate_differentiable_support_surface_alignment, validate_public_language_against_ledger]

::: scpn_quantum_control.differentiable_architecture_map
    options:
      members: [DifferentiableArchitectureMap, DifferentiableArchitectureMapLayer, DifferentiableArchitectureMapValidation, render_differentiable_architecture_map_markdown, run_differentiable_architecture_map, validate_differentiable_architecture_map]

::: scpn_quantum_control.differentiable_dependency_environment_map
    options:
      members: [DifferentiableDependencyEnvironmentMap, DifferentiableDependencyEnvironmentProfile, DifferentiableDependencyEnvironmentMapValidation, render_differentiable_dependency_environment_map_markdown, run_differentiable_dependency_environment_map, validate_differentiable_dependency_environment_map]

::: scpn_quantum_control.differentiable_competitive_baselines
    options:
      members: [CompetitiveBaselinePromotionGate, CompetitiveBaselineRefresh, CompetitiveBaselineRow, CompetitiveBaselineValidation, audit_competitive_baseline_promotion_gate, load_competitive_baseline_refresh, render_competitive_baseline_refresh_markdown, run_competitive_baseline_refresh, validate_competitive_baseline_refresh]

## MLIR Compiler

::: scpn_quantum_control.compiler.mlir
    options:
      members: [MLIRCompileConfig, DifferentiableMLIRCompileConfig, CompilerADExecutableConfig, CompilerADKernelVerification, ExecutableCompilerADKernel, ExecutableWholeProgramADBatchResult, ExecutableWholeProgramADKernel, NativeWholeProgramADKernel, MLIRModule, compile_kuramoto_to_mlir, compile_custom_derivative_rule_to_mlir, compile_custom_derivative_rule_to_executable, compile_registered_primitive_to_executable, compile_whole_program_ad_trace_to_executable, compile_whole_program_ad_trace_to_native_llvm_jit, compile_whole_program_ad_trace_to_mlir]

## Real-Time Runtime

::: scpn_quantum_control.control.realtime_runtime
    options:
      members: [CycleSample, RealtimeSLAConfig, RealtimeSLAReport, MonotonicRealtimeClock, RealtimeClock, RealtimeRunResult, RealtimeRuntimeConfig, RealtimeTickRecord, SubMicrosecondReport, SubMicrosecondTracker, VirtualRealtimeClock, enforce_realtime_sla, evaluate_realtime_sla, run_realtime_control_loop, summarise_cycle_samples]

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
      members: [FSSFitDiagnostics, FSSResult, finite_size_scaling]

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
