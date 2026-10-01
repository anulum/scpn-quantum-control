# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Dynamics Solvers
"""Phase dynamics solver public exports.

Includes Kuramoto-XY Trotterisation, VQE ground-state search, UPDE Trotter
integration, Trotter error analysis, and ansatz benchmarking.
"""

from __future__ import annotations

import sys as _sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapt_vqe import ADAPTResult, adapt_vqe
    from .adiabatic_preparation import AdiabaticResult, adiabatic_ramp
    from .ansatz_bench import AnsatzBenchmarkRow, benchmark_ansatz, run_ansatz_benchmark
    from .ansatz_methodology import AnsatzBenchmarkResult
    from .avqds import AVQDSResult, avqds_simulate
    from .coupling_learning import (
        CouplingGradientVerificationResult,
        CouplingLearningResult,
        coupling_matrix_from_edge_vector,
        learn_couplings_from_observations,
        verify_coupling_parameter_shift_gradient,
    )
    from .coupling_time_series_recovery import (
        COUPLING_RECOVERY_CLAIM_BOUNDARY,
        COUPLING_RECOVERY_EVIDENCE_CLASS,
        CouplingRecoveryBoundaryRow,
        CouplingRecoveryCase,
        CouplingRecoveryRecord,
        CouplingRecoverySuiteResult,
        coupling_recovery_boundary_rows,
        default_coupling_recovery_cases,
        inject_time_series_noise_and_missing,
        recover_kuramoto_couplings_from_time_series,
        recover_xy_couplings_from_pair_energy_series,
        run_coupling_recovery_suite,
        simulate_kuramoto_phase_time_series,
        simulate_xy_pair_energy_time_series,
    )
    from .cross_domain_transfer import TransferResult, build_systems, transfer_experiment
    from .differentiable_audit import (
        DifferentiableQuantumAuditReport,
        DifferentiableWorkflowAuditSuiteResult,
        FiniteShotGradientAuditResult,
        MLFrameworkGradientAuditRecord,
        MLFrameworkGradientAuditSuiteResult,
        ParameterShiftAnalyticAgreement,
        PhaseGradientBenchmarkSuiteResult,
        run_differentiable_workflow_audit_suite,
        run_finite_shot_gradient_uncertainty_audit,
        run_known_phase_gradient_audit,
        run_ml_framework_gradient_audit,
        run_parameter_shift_audit_suite,
        run_phase_gradient_benchmark_suite,
        verify_parameter_shift_analytic_gradient,
    )
    from .differentiable_readiness import (
        DifferentiableReadinessAuditRecord,
        DifferentiableReadinessAuditResult,
        DifferentiableReadinessSurface,
        default_differentiable_readiness_surfaces,
        run_differentiable_readiness_audit,
    )
    from .domain_benchmark_datasets import (
        DifferentiableDomainBenchmarkDatasetSuite,
        DifferentiableDomainBenchmarkValidationResult,
        DifferentiableDomainBenchmarkValidationSuite,
        DifferentiableKuramotoExactAnswerCase,
        DifferentiablePublishedDomainBenchmarkCase,
        DifferentiablePublishedDomainBenchmarkSuite,
        DifferentiablePublishedDomainBenchmarkValidationResult,
        DifferentiablePublishedDomainBenchmarkValidationSuite,
        DifferentiableQNNExactAnswerCase,
        load_differentiable_domain_benchmark_datasets,
        load_differentiable_published_domain_benchmark_cases,
        run_differentiable_domain_benchmark_dataset_validation,
        run_differentiable_published_domain_benchmark_validation,
    )
    from .floquet_kuramoto import (
        DTC_SUBHARMONIC_THRESHOLD,
        FloquetResult,
        floquet_evolve,
        scan_drive_amplitude,
    )
    from .general_unitary import build_u3_operations, su2_zyz_angles
    from .generalised_parameter_shift import (
        GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY,
        GeneralisedParameterShiftPlan,
        GeneralisedParameterShiftResult,
        GeneralisedParameterShiftTerm,
        GeneralisedStochasticParameterShiftResult,
        estimate_generalised_parameter_shift_shot_noise,
        generalised_parameter_shift_gradient,
        plan_generalised_parameter_shift,
        value_and_generalised_parameter_shift_grad,
    )
    from .gradient_backend import (
        QuantumGradientBackendCapability,
        QuantumGradientMethodExplanation,
        QuantumGradientPlan,
        QuantumGradientRejectedMethod,
        QuantumGradientShotPolicy,
        explain_quantum_gradient_method,
        plan_quantum_gradient_backend,
        quantum_gradient_backend_capability,
    )
    from .gradient_descent import (
        ParameterShiftTrainingCertificate,
        ParameterShiftTrainingResult,
        ParameterShiftTrainingStep,
        parameter_shift_gradient_descent,
        validate_parameter_shift_training,
    )
    from .gradient_support_matrix import (
        GradientSupportCapability,
        GradientSupportMatrixAuditResult,
        GradientSupportPlan,
        assert_gradient_support,
        gradient_support_capability,
        list_gradient_support_capabilities,
        plan_gradient_support,
        run_gradient_support_matrix_audit,
    )
    from .gradient_tape import (
        GRADIENT_TAPE_CONTRACT_CLAIM_BOUNDARY,
        GradientTapeContractAuditResult,
        GradientTapeContractCheck,
        QuantumGradientTape,
        TapeContractStatus,
        TapeGradientRecord,
        gradient_tape,
        run_gradient_tape_contract_audit,
    )
    from .hardware_gradient_campaign import (
        HardwareGradientCampaignPlan,
        HardwareGradientCampaignSpec,
        HardwareGradientCampaignSuite,
        HardwareGradientReplaySchema,
        default_hardware_gradient_campaign_specs,
        plan_hardware_gradient_campaign,
        run_hardware_gradient_campaign_readiness_suite,
    )
    from .hardware_gradient_policy import (
        HardwareGradientPolicy,
        HardwareGradientPolicyDecision,
        HardwareGradientReadinessSuiteResult,
        HardwareGradientRequest,
        assert_hardware_gradient_policy_approved,
        evaluate_hardware_gradient_policy,
        run_hardware_gradient_policy_readiness_suite,
    )
    from .hardware_gradient_publication import (
        HardwareGradientArtifactMapEntry,
        HardwareGradientBenchmarkPlaceholder,
        HardwareGradientClaimLedgerRow,
        HardwareGradientMethodSection,
        HardwareGradientPreregistration,
        HardwareGradientPublicationPackage,
        build_hardware_gradient_publication_package,
    )
    from .jax_bridge import (
        check_jax_parameter_shift_agreement,
        is_phase_jax_available,
        jax_custom_vjp_qnn_value_and_grad,
        jax_native_qnn_value_and_grad,
        jax_parameter_shift_value_and_grad,
        jax_phase_qnode_aot_export_audit,
        jax_phase_qnode_native_transform_audit,
        jax_phase_qnode_pytree_transform_audit,
        jax_phase_qnode_sharding_transform_audit,
        jax_phase_qnode_value_and_grad,
        plan_jax_cloud_validation_batch,
        run_jax_jit_compatibility_audit,
        run_jax_maturity_audit,
        run_jax_nested_transform_algebra_audit,
        run_jax_phase_qnode_lowering_matrix,
        run_jax_pytree_compatibility_audit,
        run_jax_sharding_compatibility_audit,
        run_jax_vmap_compatibility_audit,
    )
    from .jax_bridge_contracts import (
        PhaseJAXCloudValidationRunSpec,
        PhaseJAXCustomVJPQNNGradientResult,
        PhaseJAXGradientAgreementResult,
        PhaseJAXJITCompatibilityResult,
        PhaseJAXMaturityAuditResult,
        PhaseJAXNativeQNNGradientResult,
        PhaseJAXNestedTransformAlgebraResult,
        PhaseJAXNestedTransformRoute,
        PhaseJAXParameterShiftResult,
        PhaseJAXPhaseQNodeAOTExportResult,
        PhaseJAXPhaseQNodeLoweringMatrixResult,
        PhaseJAXPhaseQNodeLoweringRoute,
        PhaseJAXPhaseQNodeNativeTransformResult,
        PhaseJAXPhaseQNodePyTreeTransformResult,
        PhaseJAXPhaseQNodeShardingTransformResult,
        PhaseJAXPhaseQNodeStatevectorResult,
        PhaseJAXPyTreeCompatibilityResult,
        PhaseJAXShardingCompatibilityResult,
        PhaseJAXVMAPCompatibilityResult,
    )
    from .kuramoto_variants import (
        HigherOrderKuramotoSpec,
        KuramotoVariant,
        KuramotoVariantResult,
        MonitoredKuramotoSpec,
        PTSymmetricKuramotoSpec,
        build_triadic_ring_terms,
        simulate_higher_order_kuramoto,
        simulate_monitored_kuramoto,
        simulate_pt_symmetric_kuramoto,
    )
    from .lindblad_engine import LindbladSyncEngine
    from .model_training_evidence import (
        DifferentiableModelTrainingEvidenceSuite,
        DifferentiableModelTrainingRecord,
        RegisteredDifferentiableTrainingSuiteAuditResult,
        RegisteredDifferentiableTrainingSuiteRecord,
        run_differentiable_model_training_evidence_suite,
        run_registered_differentiable_training_suite_audit,
    )
    from .natural_gradient import (
        NaturalGradientDirection,
        NaturalGradientRegularizationPolicy,
        ParameterShiftNaturalGradientCertificate,
        ParameterShiftNaturalGradientResult,
        ParameterShiftNaturalGradientStep,
        parameter_shift_natural_gradient_descent,
        solve_natural_gradient_direction,
        validate_natural_gradient_training,
    )
    from .objective_audit import (
        ComposedObjectiveAuditSuiteResult,
        ComposedObjectiveGradientAgreement,
        run_composed_objective_audit_suite,
        verify_composed_objective_gradient,
    )
    from .objective_planner import (
        ComposedObjectiveExecutionPlan,
        ComposedObjectivePlannerAuditResult,
        assert_composed_objective_execution_supported,
        plan_composed_objective_execution,
        run_composed_objective_planner_audit,
    )
    from .objectives import (
        ComposedObjectiveTrainingCertificate,
        ComposedObjectiveTrainingResult,
        ComposedObjectiveTrainingStep,
        ComposedPhaseObjective,
        ObjectiveGradientEvaluation,
        ObjectiveTerm,
        ObjectiveTermValue,
        build_phase_control_objective,
        periodic_regularization_term,
        phase_energy_term,
        phase_fidelity_target_term,
        phase_symmetry_penalty_term,
        smooth_box_safety_penalty_term,
        train_composed_phase_objective,
        validate_composed_objective_training,
    )
    from .open_system_objectives import (
        OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY,
        OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS,
        BoundedOpenSystemObjectiveCase,
        DensityMatrixInvariantCertificate,
        MCWFReproducibilityCertificate,
        OpenSystemObjectiveBoundaryRow,
        OpenSystemObjectiveRecord,
        OpenSystemObjectiveSuiteResult,
        certify_density_matrix_invariants,
        certify_mcwf_reproducibility,
        default_open_system_objective_cases,
        evaluate_lindblad_objective,
        evaluate_mcwf_objective,
        open_system_objective_boundary_rows,
        run_open_system_objective_suite,
    )
    from .optimizer_audit import (
        OptimizerComparisonSuiteResult,
        OptimizerConvergenceRecord,
        run_parameter_shift_optimizer_comparison,
    )
    from .optimizer_convergence_suite import (
        GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY,
        GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS,
        GroundStateConvergenceCertificate,
        GroundStateOptimizerBoundaryRow,
        GroundStateOptimizerConvergenceSuiteResult,
        GroundStateOptimizerRunRecord,
        KnownGroundStateObjective,
        default_ground_state_optimizer_objectives,
        run_ground_state_optimizer_convergence_suite,
    )
    from .param_shift import (
        GenericParameterShiftEvaluationPlan,
        GradientVerificationResult,
        HessianVerificationResult,
        ParamShiftConvergenceDiagnostics,
        ParamShiftVQEResult,
        multi_frequency_parameter_shift_rule,
        parameter_shift_gradient,
        parameter_shift_gradient_with_uncertainty,
        parameter_shift_hessian,
        plan_generic_parameter_shift_evaluations,
        plan_parameter_shift_shots,
        validate_param_shift_convergence,
        value_and_parameter_shift_grad,
        value_and_vqe_grad,
        verify_parameter_shift_gradient,
        verify_parameter_shift_hessian,
        verify_vqe_parameter_shift_gradient,
        verify_vqe_parameter_shift_hessian,
        vqe_with_param_shift,
    )
    from .pennylane_bridge import (
        PennyLaneGradientAgreementResult,
        PennyLaneMaturityAuditResult,
        PennyLaneQNodeConversionResult,
        PennyLaneRoundTripResult,
        build_pennylane_qnode_from_phase_qnode,
        check_pennylane_parameter_shift_agreement,
        check_pennylane_phase_qnode_round_trip,
        check_pennylane_qnode_round_trip,
        is_phase_pennylane_available,
        run_pennylane_maturity_audit,
    )
    from .pennylane_import import (
        PennyLaneImportResult,
        PennyLaneImportRoundTripResult,
        check_pennylane_phase_qnode_import_round_trip,
        import_phase_qnode_from_pennylane,
        is_pennylane_import_available,
    )
    from .pennylane_provider_plugin import (
        PennyLaneHardwarePluginExecutionArtifact,
        PennyLanePluginMatrixResult,
        PennyLanePluginMatrixRoute,
        PennyLaneProviderEvidenceBundle,
        PennyLaneProviderGradientParityArtifact,
        PennyLaneProviderPluginExecutionArtifact,
        run_pennylane_plugin_matrix,
    )
    from .phase_vqe import PhaseVQE, PhaseVQEResult
    from .provider_gradient import (
        ProviderExpectationSample,
        ProviderGradientExecutionResult,
        ProviderHardwareGradientPreparationResult,
        ProviderParameterShiftRecord,
        execute_provider_parameter_shift_gradient,
        prepare_provider_hardware_parameter_shift_gradient,
    )
    from .provider_gradient_audit import (
        ProviderGradientReadinessAuditResult,
        ProviderGradientReadinessRecord,
        ProviderGradientReadinessScenario,
        default_provider_gradient_readiness_scenarios,
        run_provider_gradient_readiness_audit,
    )
    from .provider_hardware_gradient_audit import (
        ProviderHardwareGradientPreparationAuditResult,
        ProviderHardwareGradientPreparationRecord,
        ProviderHardwareGradientPreparationScenario,
        default_provider_hardware_gradient_preparation_scenarios,
        run_provider_hardware_gradient_preparation_audit,
    )
    from .provider_hardware_safety_audit import (
        DifferentiableProviderHardwareEvidenceChain,
        DifferentiableProviderHardwareSafetyAuditResult,
        DifferentiableProviderHardwareSafetySurface,
        run_differentiable_provider_hardware_safety_audit,
    )
    from .pulse_shaping import (
        HypergeometricPulse,
        ICIPulse,
        PulseSchedule,
        build_hypergeometric_pulse,
        build_ici_pulse,
        build_trotter_pulse_schedule,
        hypergeometric_envelope,
        ici_three_level_evolution,
        infidelity_bound,
    )
    from .qgnn import (
        KnmGraph,
        QGNNConfig,
        QGNNTrainingResult,
        synthetic_kuramoto_target,
    )
    from .qiskit_bridge import (
        QiskitCalibrationStatevectorComparisonArtifact,
        QiskitMaturityAuditResult,
        QiskitParameterShiftGradientResult,
        QiskitParameterShiftRecord,
        QiskitProviderGradientWorkflowArtifact,
        QiskitRawCountReplayArtifact,
        QiskitRuntimePrimitiveExecutionArtifact,
        QiskitRuntimeQPUExecutionArtifact,
        QiskitRuntimeQPUProviderEvidenceBundle,
        build_qiskit_provider_gradient_workflow_artifact,
        build_qiskit_runtime_qpu_execution_artifact,
        build_qiskit_runtime_qpu_provider_evidence_bundle,
        execute_qiskit_finite_shot_parameter_shift,
        execute_qiskit_statevector_parameter_shift,
        generate_qiskit_parameter_shift_circuits,
        run_qiskit_maturity_audit,
    )
    from .qnn_conformance import (
        ExternalGradientMap,
        ParameterShiftQNNConformanceCaseResult,
        ParameterShiftQNNConformanceSuiteResult,
        ParameterShiftQNNUnsupportedScenario,
        run_parameter_shift_qnn_conformance_suite,
        summarize_parameter_shift_qnn_unsuitable_scenarios,
    )
    from .qnn_convergence import (
        ParameterShiftQNNConvergenceCaseResult,
        ParameterShiftQNNConvergenceSuiteResult,
        ParameterShiftQNNConvergenceUnsuitableScenario,
        ParameterShiftQNNMultiSeedConvergenceCaseResult,
        ParameterShiftQNNMultiSeedConvergenceRunResult,
        ParameterShiftQNNMultiSeedConvergenceSuiteResult,
        run_parameter_shift_qnn_convergence_suite,
        run_parameter_shift_qnn_multi_seed_convergence_suite,
        summarize_parameter_shift_qnn_convergence_unsuitable_scenarios,
    )
    from .qnn_finite_shot import (
        ParameterShiftQNNFiniteShotConvergenceCaseResult,
        ParameterShiftQNNFiniteShotConvergenceSuiteResult,
        ParameterShiftQNNFiniteShotGradientResult,
        ParameterShiftQNNFiniteShotProbeRecord,
        ParameterShiftQNNFiniteShotUnsupportedScenario,
        estimate_parameter_shift_qnn_finite_shot_gradient,
        run_parameter_shift_qnn_finite_shot_convergence_suite,
        summarize_parameter_shift_qnn_finite_shot_unsuitable_scenarios,
    )
    from .qnn_framework_agreement import (
        FrameworkGradientCaseMap,
        FrameworkGradientMap,
        ParameterShiftQNNFrameworkAgreementResult,
        ParameterShiftQNNFrameworkAgreementSuiteResult,
        ParameterShiftQNNFrameworkGradientAgreement,
        run_parameter_shift_qnn_framework_agreement_suite,
        verify_parameter_shift_qnn_framework_agreement,
    )
    from .qnn_framework_bridge_matrix import (
        BoundedQNNFrameworkBridgeCapability,
        BoundedQNNFrameworkBridgeMatrixResult,
        assert_bounded_qnn_framework_bridge_supported,
        run_bounded_qnn_framework_bridge_matrix,
    )
    from .qnn_loss_landscape import (
        ParameterShiftQNNLossLandscapeCaseResult,
        ParameterShiftQNNLossLandscapePoint,
        ParameterShiftQNNLossLandscapeSuiteResult,
        run_parameter_shift_qnn_loss_landscape_suite,
    )
    from .qnn_optimizer_benchmark import (
        DerivativeFreeCandidateMap,
        ParameterShiftQNNOptimizerBenchmarkCaseResult,
        ParameterShiftQNNOptimizerBenchmarkSuiteResult,
        QNNOptimizerBaselineResult,
        run_parameter_shift_qnn_optimizer_benchmark_suite,
    )
    from .qnn_training import (
        ParameterShiftQNNExternalGradientAgreement,
        ParameterShiftQNNGradientVerificationResult,
        ParameterShiftQNNPredictionResult,
        ParameterShiftQNNTrainingResult,
        parameter_shift_qnn_classifier_gradient,
        parameter_shift_qnn_classifier_loss,
        predict_parameter_shift_qnn_classifier,
        train_parameter_shift_qnn_classifier,
        verify_parameter_shift_qnn_classifier_gradient,
    )
    from .qnode_affinity_benchmark import (
        PhaseQNodeAffinityArtifactValidation,
        PhaseQNodeAffinityBenchmarkMetadata,
        PhaseQNodeAffinityBenchmarkResult,
        classify_affinity_evidence,
        run_phase_qnode_affinity_benchmark,
        validate_phase_qnode_affinity_artifact,
    )
    from .qnode_circuit import (
        DenseHermitianObservable,
        PauliCovarianceObservable,
        PauliTerm,
        PhaseQNodeCircuit,
        PhaseQNodeClassicalFisherResult,
        PhaseQNodeDensityCircuit,
        PhaseQNodeDensityExecutionResult,
        PhaseQNodeDepthProfile,
        PhaseQNodeExecutionResult,
        PhaseQNodeGradientEvaluationGroup,
        PhaseQNodeGradientEvaluationPlan,
        PhaseQNodeGradientResult,
        PhaseQNodeMetricTensorResult,
        PhaseQNodeNoiseChannel,
        PhaseQNodeOperation,
        PhaseQNodeRegisteredCircuitSpec,
        PhaseQNodeSupportError,
        PhaseQNodeSupportReport,
        PhaseQNodeTemplateSpec,
        SparsePauliHamiltonian,
        build_phase_qnode_template,
        build_registered_phase_qnode_circuit,
        build_sparse_ising_chain_hamiltonian,
        decompose_phase_qnode_controlled_gate,
        execute_phase_qnode_circuit,
        execute_phase_qnode_density_matrix,
        parameter_shift_phase_qnode_gradient,
        phase_qnode_computational_basis_fisher_information,
        phase_qnode_computational_basis_fisher_support_report,
        phase_qnode_density_support_report,
        phase_qnode_depth_profile,
        phase_qnode_gradient_support_report,
        phase_qnode_metric_support_report,
        phase_qnode_natural_gradient_metric,
        phase_qnode_quantum_fisher_information,
        phase_qnode_support_report,
        plan_phase_qnode_parameter_shift_evaluations,
        registered_phase_qnode_decompositions,
        registered_phase_qnode_gates,
        registered_phase_qnode_noise_channels,
        registered_phase_qnode_observables,
        registered_phase_qnode_templates,
    )
    from .qnode_framework_parity import (
        ParityScenario,
        PhaseQNodeFrameworkParityRecord,
        PhaseQNodeFrameworkParitySuiteResult,
        run_phase_qnode_framework_parity_suite,
    )
    from .qnode_provider_transforms import (
        ProviderQNodeTransformReadinessSuiteResult,
        ProviderQNodeTransformResult,
        execute_provider_qnode_transform,
        execute_provider_qnode_vmap_grad,
        run_provider_qnode_transform_readiness_suite,
    )
    from .qnode_tape import (
        PhaseQNodeTape,
        PhaseQNodeTapeReadinessSuiteResult,
        PhaseQNodeTapeRecord,
        phase_qnode_tape,
        run_phase_qnode_tape_readiness_suite,
    )
    from .qnode_transforms import (
        PhaseQNodeComplexDerivativeContract,
        PhaseQNodeTransformReadinessSuiteResult,
        PhaseQNodeTransformResult,
        execute_phase_qnode_hessian_vector_product,
        execute_phase_qnode_transform,
        phase_qnode_complex_derivative_contract,
        run_phase_qnode_transform_readiness_suite,
    )
    from .qnode_vector_transforms import (
        PhaseQNodeVectorTransformReadinessSuiteResult,
        PhaseQNodeVectorTransformResult,
        execute_phase_qnode_vector_hessian,
        execute_phase_qnode_vector_jacobian,
        execute_phase_qnode_vector_jvp,
        execute_phase_qnode_vector_vjp,
        execute_phase_qnode_vmap_grad,
        run_phase_qnode_vector_transform_readiness_suite,
    )
    from .qsp_phases import (
        QSPPhaseFactors,
        QSPSynthesisError,
        complementary_polynomial,
        jacobi_anger_cosine_coefficients,
        jacobi_anger_sine_coefficients,
        qsp_response,
        qsp_unitary,
        synthesise_qsp_phases,
    )
    from .qsvt_evolution import QSVTResourceEstimate
    from .results import TrajectoryResult
    from .structured_ansatz import build_structured_ansatz
    from .synchronisation_objectives import (
        SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY,
        build_synchronisation_objective,
        cluster_synchronisation_target_term,
        kuramoto_order_parameter,
        kuramoto_order_parameter_gradient,
        kuramoto_order_parameter_target_term,
        phase_locking_target_term,
    )
    from .synchronisation_witness import (
        SYNC_WITNESS_CLAIM_BOUNDARY,
        SYNC_WITNESS_EVIDENCE_CLASS,
        PhaseCloudRegime,
        SyncWitnessBoundaryRow,
        SyncWitnessCase,
        SyncWitnessRecord,
        SyncWitnessSuiteResult,
        betti_curve,
        default_sync_witness_cases,
        geodesic_phase_distance_matrix,
        harmonic_order_parameter,
        phase_cloud_synchronisation_witness,
        run_sync_witness_suite,
        sync_witness_boundary_rows,
        vietoris_rips_persistence,
    )
    from .tensorflow_bridge import (
        PhaseTensorFlowFunctionCompatibilityResult,
        PhaseTensorFlowGradientTapeCompatibilityResult,
        PhaseTensorFlowKerasLayerWrapperAuditResult,
        PhaseTensorFlowMaturityAuditResult,
        PhaseTensorFlowParameterShiftResult,
        PhaseTensorFlowPhaseQNodeLoweringMatrixResult,
        PhaseTensorFlowPhaseQNodeLoweringRoute,
        PhaseTensorFlowQNNGradientResult,
        PhaseTensorFlowXLACompatibilityResult,
        is_phase_tensorflow_available,
        run_tensorflow_function_compatibility_audit,
        run_tensorflow_gradient_tape_compatibility_audit,
        run_tensorflow_keras_layer_wrapper_audit,
        run_tensorflow_maturity_audit,
        run_tensorflow_phase_qnode_lowering_matrix,
        run_tensorflow_xla_compatibility_audit,
        tensorflow_bounded_qnn_keras_layer,
        tensorflow_bounded_qnn_value_and_grad,
        tensorflow_parameter_shift_value_and_grad,
    )
    from .tensorflow_maintenance import (
        TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY,
        PhaseTensorFlowMaintenanceReport,
        PhaseTensorFlowMaintenanceRoute,
        TensorFlowMaintenanceDecision,
        TensorFlowMaintenanceStrategy,
        run_tensorflow_maintenance_decision,
    )
    from .torch_aot_autograd_export import (
        TORCH_AOT_AUTOGRAD_EXPORT_CLAIM_BOUNDARY,
        TORCH_AOT_AUTOGRAD_EXPORT_SCHEMA,
        PhaseTorchAOTAutogradExportResult,
        PhaseTorchAOTAutogradExportRoute,
        PhaseTorchAOTAutogradGraphRecord,
        run_torch_aot_autograd_export_audit,
    )
    from .torch_autograd_function import (
        TORCH_AUTOGRAD_FUNCTION_CLAIM_BOUNDARY,
        TORCH_AUTOGRAD_FUNCTION_SCHEMA,
        PhaseTorchAutogradFunctionResult,
        PhaseTorchAutogradFunctionRoute,
        run_torch_autograd_function_audit,
        torch_autograd_function_qnn_loss,
    )
    from .torch_bridge import (
        is_phase_torch_available,
        plan_torch_cloud_validation_batch,
        run_torch_compile_compatibility_audit,
        run_torch_ecosystem_maturity_audit,
        run_torch_func_compatibility_audit,
        run_torch_maturity_audit,
        run_torch_module_wrapper_audit,
        run_torch_phase_qnode_lowering_matrix,
        run_torch_training_loop_audit,
        torch_autograd_qnn_value_and_grad,
        torch_bounded_qnn_layer,
        torch_bounded_qnn_module,
        torch_bounded_qnn_value_and_grad,
        torch_parameter_shift_value_and_grad,
        torch_phase_qnode_compile_audit,
        torch_phase_qnode_compile_boundary_audit,
        torch_phase_qnode_transform_audit,
        torch_phase_qnode_value_and_grad,
    )
    from .torch_bridge_contracts import (
        PhaseTorchAutogradQNNGradientResult,
        PhaseTorchCloudValidationRunSpec,
        PhaseTorchCompileBoundaryAuditResult,
        PhaseTorchCompileBoundaryRoute,
        PhaseTorchCompileCompatibilityResult,
        PhaseTorchEcosystemMaturityAuditResult,
        PhaseTorchEcosystemMaturityRoute,
        PhaseTorchFuncCompatibilityResult,
        PhaseTorchLiveOverlayEvidence,
        PhaseTorchMaturityAuditResult,
        PhaseTorchModuleWrapperAuditResult,
        PhaseTorchParameterShiftResult,
        PhaseTorchPhaseQNodeCompileResult,
        PhaseTorchPhaseQNodeLoweringMatrixResult,
        PhaseTorchPhaseQNodeLoweringRoute,
        PhaseTorchPhaseQNodeStatevectorResult,
        PhaseTorchPhaseQNodeTransformResult,
        PhaseTorchQNNGradientResult,
        PhaseTorchTrainingLoopAuditResult,
    )
    from .torch_checkpoint import (
        TORCH_CHECKPOINT_CLAIM_BOUNDARY,
        TORCH_CHECKPOINT_SCHEMA,
        PhaseTorchCheckpointAuditResult,
        PhaseTorchCheckpointRoute,
        run_torch_module_checkpoint_audit,
    )
    from .torch_checkpoint_matrix import (
        TORCH_CHECKPOINT_MATRIX_CLAIM_BOUNDARY,
        TORCH_CHECKPOINT_MATRIX_SCHEMA,
        PhaseTorchCheckpointMatrixResult,
        PhaseTorchCheckpointMatrixRoute,
        PhaseTorchCheckpointMatrixTensorMetadata,
        PhaseTorchCheckpointRuntimeFingerprint,
        run_torch_long_lived_checkpoint_matrix,
    )
    from .torch_device_state import (
        TORCH_DEVICE_STATE_CLAIM_BOUNDARY,
        PhaseTorchDeviceStateAuditResult,
        PhaseTorchDeviceStateRoute,
        run_torch_module_device_state_audit,
    )
    from .torch_dynamic_shape_export import (
        TORCH_DYNAMIC_SHAPE_EXPORT_CLAIM_BOUNDARY,
        TORCH_DYNAMIC_SHAPE_EXPORT_SCHEMA,
        PhaseTorchDynamicShapeExportRecord,
        PhaseTorchDynamicShapeExportReplayCase,
        PhaseTorchDynamicShapeExportResult,
        PhaseTorchDynamicShapeExportRoute,
        default_torch_dynamic_shape_export_replay_cases,
        run_torch_dynamic_shape_export_audit,
    )
    from .torch_export import (
        TORCH_EXPORT_CLAIM_BOUNDARY,
        PhaseTorchExportAuditResult,
        PhaseTorchExportRoute,
        run_torch_module_export_audit,
    )
    from .torch_export_shape_matrix import (
        TORCH_EXPORT_SHAPE_MATRIX_CLAIM_BOUNDARY,
        TORCH_EXPORT_SHAPE_MATRIX_SCHEMA,
        PhaseTorchExportShapeMatrixRecord,
        PhaseTorchExportShapeMatrixResult,
        PhaseTorchExportShapeMatrixRoute,
        PhaseTorchExportShapeScenario,
        default_torch_export_shape_scenarios,
        run_torch_export_shape_matrix,
    )
    from .torch_module_state import (
        TORCH_MODULE_STATE_CLAIM_BOUNDARY,
        PhaseTorchModuleStateAuditResult,
        PhaseTorchModuleStateRoute,
        PhaseTorchModuleStateTensorMismatch,
        PhaseTorchModuleStateValidationResult,
        run_torch_module_state_audit,
        validate_torch_bounded_qnn_state_dict,
    )
    from .torch_training_loop_matrix import (
        TORCH_TRAINING_LOOP_MATRIX_CLAIM_BOUNDARY,
        TORCH_TRAINING_LOOP_MATRIX_SCHEMA,
        PhaseTorchTrainingLoopMatrixRecord,
        PhaseTorchTrainingLoopMatrixResult,
        PhaseTorchTrainingLoopMatrixRoute,
        PhaseTorchTrainingLoopScenario,
        default_torch_training_loop_scenarios,
        run_torch_training_loop_matrix,
    )
    from .trainability import (
        TRAINABILITY_CLAIM_BOUNDARY,
        AdaptiveShotAllocationDryRun,
        BarrenPlateauTrainabilityReport,
        TrainabilityGradientSample,
        TrainabilityStatus,
        run_barren_plateau_trainability_report,
    )
    from .transform_nesting import (
        GradientTransformNestingAuditResult,
        GradientTransformNestingPlan,
        assert_gradient_transform_nesting_supported,
        plan_gradient_transform_nesting,
        run_gradient_transform_nesting_audit,
    )
    from .trotter_error import trotter_error_norm, trotter_error_sweep
    from .trotter_upde import QuantumUPDESolver, UPDEStepResult, UPDETrajectoryResult
    from .variational_metric import (
        analytic_state_derivatives,
        assert_single_parameter_rotations,
        imaginary_time_force,
        mclachlan_metric,
        real_time_force,
    )
    from .varqite import VarQITEResult, varqite_ground_state
    from .xy_kuramoto import QuantumKuramotoSolver, TrotterEvolutionConfig

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ADAPTResult": ("scpn_quantum_control.phase.adapt_vqe", "ADAPTResult"),
    "adapt_vqe": ("scpn_quantum_control.phase.adapt_vqe", "adapt_vqe"),
    "AdiabaticResult": ("scpn_quantum_control.phase.adiabatic_preparation", "AdiabaticResult"),
    "adiabatic_ramp": ("scpn_quantum_control.phase.adiabatic_preparation", "adiabatic_ramp"),
    "AnsatzBenchmarkRow": ("scpn_quantum_control.phase.ansatz_bench", "AnsatzBenchmarkRow"),
    "benchmark_ansatz": ("scpn_quantum_control.phase.ansatz_bench", "benchmark_ansatz"),
    "run_ansatz_benchmark": ("scpn_quantum_control.phase.ansatz_bench", "run_ansatz_benchmark"),
    "AnsatzBenchmarkResult": (
        "scpn_quantum_control.phase.ansatz_methodology",
        "AnsatzBenchmarkResult",
    ),
    "AVQDSResult": ("scpn_quantum_control.phase.avqds", "AVQDSResult"),
    "avqds_simulate": ("scpn_quantum_control.phase.avqds", "avqds_simulate"),
    "CouplingGradientVerificationResult": (
        "scpn_quantum_control.phase.coupling_learning",
        "CouplingGradientVerificationResult",
    ),
    "CouplingLearningResult": (
        "scpn_quantum_control.phase.coupling_learning",
        "CouplingLearningResult",
    ),
    "coupling_matrix_from_edge_vector": (
        "scpn_quantum_control.phase.coupling_learning",
        "coupling_matrix_from_edge_vector",
    ),
    "learn_couplings_from_observations": (
        "scpn_quantum_control.phase.coupling_learning",
        "learn_couplings_from_observations",
    ),
    "verify_coupling_parameter_shift_gradient": (
        "scpn_quantum_control.phase.coupling_learning",
        "verify_coupling_parameter_shift_gradient",
    ),
    "COUPLING_RECOVERY_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "COUPLING_RECOVERY_CLAIM_BOUNDARY",
    ),
    "COUPLING_RECOVERY_EVIDENCE_CLASS": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "COUPLING_RECOVERY_EVIDENCE_CLASS",
    ),
    "CouplingRecoveryBoundaryRow": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "CouplingRecoveryBoundaryRow",
    ),
    "CouplingRecoveryCase": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "CouplingRecoveryCase",
    ),
    "CouplingRecoveryRecord": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "CouplingRecoveryRecord",
    ),
    "CouplingRecoverySuiteResult": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "CouplingRecoverySuiteResult",
    ),
    "coupling_recovery_boundary_rows": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "coupling_recovery_boundary_rows",
    ),
    "default_coupling_recovery_cases": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "default_coupling_recovery_cases",
    ),
    "inject_time_series_noise_and_missing": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "inject_time_series_noise_and_missing",
    ),
    "recover_kuramoto_couplings_from_time_series": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "recover_kuramoto_couplings_from_time_series",
    ),
    "recover_xy_couplings_from_pair_energy_series": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "recover_xy_couplings_from_pair_energy_series",
    ),
    "run_coupling_recovery_suite": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "run_coupling_recovery_suite",
    ),
    "simulate_kuramoto_phase_time_series": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "simulate_kuramoto_phase_time_series",
    ),
    "simulate_xy_pair_energy_time_series": (
        "scpn_quantum_control.phase.coupling_time_series_recovery",
        "simulate_xy_pair_energy_time_series",
    ),
    "TransferResult": ("scpn_quantum_control.phase.cross_domain_transfer", "TransferResult"),
    "build_systems": ("scpn_quantum_control.phase.cross_domain_transfer", "build_systems"),
    "transfer_experiment": (
        "scpn_quantum_control.phase.cross_domain_transfer",
        "transfer_experiment",
    ),
    "DifferentiableQuantumAuditReport": (
        "scpn_quantum_control.phase.differentiable_audit",
        "DifferentiableQuantumAuditReport",
    ),
    "DifferentiableWorkflowAuditSuiteResult": (
        "scpn_quantum_control.phase.differentiable_audit",
        "DifferentiableWorkflowAuditSuiteResult",
    ),
    "FiniteShotGradientAuditResult": (
        "scpn_quantum_control.phase.differentiable_audit",
        "FiniteShotGradientAuditResult",
    ),
    "MLFrameworkGradientAuditRecord": (
        "scpn_quantum_control.phase.differentiable_audit",
        "MLFrameworkGradientAuditRecord",
    ),
    "MLFrameworkGradientAuditSuiteResult": (
        "scpn_quantum_control.phase.differentiable_audit",
        "MLFrameworkGradientAuditSuiteResult",
    ),
    "ParameterShiftAnalyticAgreement": (
        "scpn_quantum_control.phase.differentiable_audit",
        "ParameterShiftAnalyticAgreement",
    ),
    "PhaseGradientBenchmarkSuiteResult": (
        "scpn_quantum_control.phase.differentiable_audit",
        "PhaseGradientBenchmarkSuiteResult",
    ),
    "run_differentiable_workflow_audit_suite": (
        "scpn_quantum_control.phase.differentiable_audit",
        "run_differentiable_workflow_audit_suite",
    ),
    "run_finite_shot_gradient_uncertainty_audit": (
        "scpn_quantum_control.phase.differentiable_audit",
        "run_finite_shot_gradient_uncertainty_audit",
    ),
    "run_known_phase_gradient_audit": (
        "scpn_quantum_control.phase.differentiable_audit",
        "run_known_phase_gradient_audit",
    ),
    "run_ml_framework_gradient_audit": (
        "scpn_quantum_control.phase.differentiable_audit",
        "run_ml_framework_gradient_audit",
    ),
    "run_parameter_shift_audit_suite": (
        "scpn_quantum_control.phase.differentiable_audit",
        "run_parameter_shift_audit_suite",
    ),
    "run_phase_gradient_benchmark_suite": (
        "scpn_quantum_control.phase.differentiable_audit",
        "run_phase_gradient_benchmark_suite",
    ),
    "verify_parameter_shift_analytic_gradient": (
        "scpn_quantum_control.phase.differentiable_audit",
        "verify_parameter_shift_analytic_gradient",
    ),
    "DifferentiableReadinessAuditRecord": (
        "scpn_quantum_control.phase.differentiable_readiness",
        "DifferentiableReadinessAuditRecord",
    ),
    "DifferentiableReadinessAuditResult": (
        "scpn_quantum_control.phase.differentiable_readiness",
        "DifferentiableReadinessAuditResult",
    ),
    "DifferentiableReadinessSurface": (
        "scpn_quantum_control.phase.differentiable_readiness",
        "DifferentiableReadinessSurface",
    ),
    "default_differentiable_readiness_surfaces": (
        "scpn_quantum_control.phase.differentiable_readiness",
        "default_differentiable_readiness_surfaces",
    ),
    "run_differentiable_readiness_audit": (
        "scpn_quantum_control.phase.differentiable_readiness",
        "run_differentiable_readiness_audit",
    ),
    "DifferentiableDomainBenchmarkDatasetSuite": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "DifferentiableDomainBenchmarkDatasetSuite",
    ),
    "DifferentiableDomainBenchmarkValidationResult": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "DifferentiableDomainBenchmarkValidationResult",
    ),
    "DifferentiableDomainBenchmarkValidationSuite": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "DifferentiableDomainBenchmarkValidationSuite",
    ),
    "DifferentiableKuramotoExactAnswerCase": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "DifferentiableKuramotoExactAnswerCase",
    ),
    "DifferentiablePublishedDomainBenchmarkCase": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "DifferentiablePublishedDomainBenchmarkCase",
    ),
    "DifferentiablePublishedDomainBenchmarkSuite": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "DifferentiablePublishedDomainBenchmarkSuite",
    ),
    "DifferentiablePublishedDomainBenchmarkValidationResult": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "DifferentiablePublishedDomainBenchmarkValidationResult",
    ),
    "DifferentiablePublishedDomainBenchmarkValidationSuite": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "DifferentiablePublishedDomainBenchmarkValidationSuite",
    ),
    "DifferentiableQNNExactAnswerCase": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "DifferentiableQNNExactAnswerCase",
    ),
    "load_differentiable_domain_benchmark_datasets": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "load_differentiable_domain_benchmark_datasets",
    ),
    "load_differentiable_published_domain_benchmark_cases": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "load_differentiable_published_domain_benchmark_cases",
    ),
    "run_differentiable_domain_benchmark_dataset_validation": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "run_differentiable_domain_benchmark_dataset_validation",
    ),
    "run_differentiable_published_domain_benchmark_validation": (
        "scpn_quantum_control.phase.domain_benchmark_datasets",
        "run_differentiable_published_domain_benchmark_validation",
    ),
    "DTC_SUBHARMONIC_THRESHOLD": (
        "scpn_quantum_control.phase.floquet_kuramoto",
        "DTC_SUBHARMONIC_THRESHOLD",
    ),
    "FloquetResult": ("scpn_quantum_control.phase.floquet_kuramoto", "FloquetResult"),
    "floquet_evolve": ("scpn_quantum_control.phase.floquet_kuramoto", "floquet_evolve"),
    "scan_drive_amplitude": (
        "scpn_quantum_control.phase.floquet_kuramoto",
        "scan_drive_amplitude",
    ),
    "build_u3_operations": ("scpn_quantum_control.phase.general_unitary", "build_u3_operations"),
    "su2_zyz_angles": ("scpn_quantum_control.phase.general_unitary", "su2_zyz_angles"),
    "GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.generalised_parameter_shift",
        "GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY",
    ),
    "GeneralisedParameterShiftPlan": (
        "scpn_quantum_control.phase.generalised_parameter_shift",
        "GeneralisedParameterShiftPlan",
    ),
    "GeneralisedParameterShiftResult": (
        "scpn_quantum_control.phase.generalised_parameter_shift",
        "GeneralisedParameterShiftResult",
    ),
    "GeneralisedParameterShiftTerm": (
        "scpn_quantum_control.phase.generalised_parameter_shift",
        "GeneralisedParameterShiftTerm",
    ),
    "GeneralisedStochasticParameterShiftResult": (
        "scpn_quantum_control.phase.generalised_parameter_shift",
        "GeneralisedStochasticParameterShiftResult",
    ),
    "estimate_generalised_parameter_shift_shot_noise": (
        "scpn_quantum_control.phase.generalised_parameter_shift",
        "estimate_generalised_parameter_shift_shot_noise",
    ),
    "generalised_parameter_shift_gradient": (
        "scpn_quantum_control.phase.generalised_parameter_shift",
        "generalised_parameter_shift_gradient",
    ),
    "plan_generalised_parameter_shift": (
        "scpn_quantum_control.phase.generalised_parameter_shift",
        "plan_generalised_parameter_shift",
    ),
    "value_and_generalised_parameter_shift_grad": (
        "scpn_quantum_control.phase.generalised_parameter_shift",
        "value_and_generalised_parameter_shift_grad",
    ),
    "QuantumGradientBackendCapability": (
        "scpn_quantum_control.phase.gradient_backend",
        "QuantumGradientBackendCapability",
    ),
    "QuantumGradientMethodExplanation": (
        "scpn_quantum_control.phase.gradient_backend",
        "QuantumGradientMethodExplanation",
    ),
    "QuantumGradientPlan": ("scpn_quantum_control.phase.gradient_backend", "QuantumGradientPlan"),
    "QuantumGradientRejectedMethod": (
        "scpn_quantum_control.phase.gradient_backend",
        "QuantumGradientRejectedMethod",
    ),
    "QuantumGradientShotPolicy": (
        "scpn_quantum_control.phase.gradient_backend",
        "QuantumGradientShotPolicy",
    ),
    "explain_quantum_gradient_method": (
        "scpn_quantum_control.phase.gradient_backend",
        "explain_quantum_gradient_method",
    ),
    "plan_quantum_gradient_backend": (
        "scpn_quantum_control.phase.gradient_backend",
        "plan_quantum_gradient_backend",
    ),
    "quantum_gradient_backend_capability": (
        "scpn_quantum_control.phase.gradient_backend",
        "quantum_gradient_backend_capability",
    ),
    "ParameterShiftTrainingCertificate": (
        "scpn_quantum_control.phase.gradient_descent",
        "ParameterShiftTrainingCertificate",
    ),
    "ParameterShiftTrainingResult": (
        "scpn_quantum_control.phase.gradient_descent",
        "ParameterShiftTrainingResult",
    ),
    "ParameterShiftTrainingStep": (
        "scpn_quantum_control.phase.gradient_descent",
        "ParameterShiftTrainingStep",
    ),
    "parameter_shift_gradient_descent": (
        "scpn_quantum_control.phase.gradient_descent",
        "parameter_shift_gradient_descent",
    ),
    "validate_parameter_shift_training": (
        "scpn_quantum_control.phase.gradient_descent",
        "validate_parameter_shift_training",
    ),
    "GradientSupportCapability": (
        "scpn_quantum_control.phase.gradient_support_matrix",
        "GradientSupportCapability",
    ),
    "GradientSupportMatrixAuditResult": (
        "scpn_quantum_control.phase.gradient_support_matrix",
        "GradientSupportMatrixAuditResult",
    ),
    "GradientSupportPlan": (
        "scpn_quantum_control.phase.gradient_support_matrix",
        "GradientSupportPlan",
    ),
    "assert_gradient_support": (
        "scpn_quantum_control.phase.gradient_support_matrix",
        "assert_gradient_support",
    ),
    "gradient_support_capability": (
        "scpn_quantum_control.phase.gradient_support_matrix",
        "gradient_support_capability",
    ),
    "list_gradient_support_capabilities": (
        "scpn_quantum_control.phase.gradient_support_matrix",
        "list_gradient_support_capabilities",
    ),
    "plan_gradient_support": (
        "scpn_quantum_control.phase.gradient_support_matrix",
        "plan_gradient_support",
    ),
    "run_gradient_support_matrix_audit": (
        "scpn_quantum_control.phase.gradient_support_matrix",
        "run_gradient_support_matrix_audit",
    ),
    "GRADIENT_TAPE_CONTRACT_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.gradient_tape",
        "GRADIENT_TAPE_CONTRACT_CLAIM_BOUNDARY",
    ),
    "GradientTapeContractAuditResult": (
        "scpn_quantum_control.phase.gradient_tape",
        "GradientTapeContractAuditResult",
    ),
    "GradientTapeContractCheck": (
        "scpn_quantum_control.phase.gradient_tape",
        "GradientTapeContractCheck",
    ),
    "QuantumGradientTape": ("scpn_quantum_control.phase.gradient_tape", "QuantumGradientTape"),
    "TapeContractStatus": ("scpn_quantum_control.phase.gradient_tape", "TapeContractStatus"),
    "TapeGradientRecord": ("scpn_quantum_control.phase.gradient_tape", "TapeGradientRecord"),
    "gradient_tape": ("scpn_quantum_control.phase.gradient_tape", "gradient_tape"),
    "run_gradient_tape_contract_audit": (
        "scpn_quantum_control.phase.gradient_tape",
        "run_gradient_tape_contract_audit",
    ),
    "HardwareGradientCampaignPlan": (
        "scpn_quantum_control.phase.hardware_gradient_campaign",
        "HardwareGradientCampaignPlan",
    ),
    "HardwareGradientCampaignSpec": (
        "scpn_quantum_control.phase.hardware_gradient_campaign",
        "HardwareGradientCampaignSpec",
    ),
    "HardwareGradientCampaignSuite": (
        "scpn_quantum_control.phase.hardware_gradient_campaign",
        "HardwareGradientCampaignSuite",
    ),
    "HardwareGradientReplaySchema": (
        "scpn_quantum_control.phase.hardware_gradient_campaign",
        "HardwareGradientReplaySchema",
    ),
    "default_hardware_gradient_campaign_specs": (
        "scpn_quantum_control.phase.hardware_gradient_campaign",
        "default_hardware_gradient_campaign_specs",
    ),
    "plan_hardware_gradient_campaign": (
        "scpn_quantum_control.phase.hardware_gradient_campaign",
        "plan_hardware_gradient_campaign",
    ),
    "run_hardware_gradient_campaign_readiness_suite": (
        "scpn_quantum_control.phase.hardware_gradient_campaign",
        "run_hardware_gradient_campaign_readiness_suite",
    ),
    "HardwareGradientPolicy": (
        "scpn_quantum_control.phase.hardware_gradient_policy",
        "HardwareGradientPolicy",
    ),
    "HardwareGradientPolicyDecision": (
        "scpn_quantum_control.phase.hardware_gradient_policy",
        "HardwareGradientPolicyDecision",
    ),
    "HardwareGradientReadinessSuiteResult": (
        "scpn_quantum_control.phase.hardware_gradient_policy",
        "HardwareGradientReadinessSuiteResult",
    ),
    "HardwareGradientRequest": (
        "scpn_quantum_control.phase.hardware_gradient_policy",
        "HardwareGradientRequest",
    ),
    "assert_hardware_gradient_policy_approved": (
        "scpn_quantum_control.phase.hardware_gradient_policy",
        "assert_hardware_gradient_policy_approved",
    ),
    "evaluate_hardware_gradient_policy": (
        "scpn_quantum_control.phase.hardware_gradient_policy",
        "evaluate_hardware_gradient_policy",
    ),
    "run_hardware_gradient_policy_readiness_suite": (
        "scpn_quantum_control.phase.hardware_gradient_policy",
        "run_hardware_gradient_policy_readiness_suite",
    ),
    "HardwareGradientArtifactMapEntry": (
        "scpn_quantum_control.phase.hardware_gradient_publication",
        "HardwareGradientArtifactMapEntry",
    ),
    "HardwareGradientBenchmarkPlaceholder": (
        "scpn_quantum_control.phase.hardware_gradient_publication",
        "HardwareGradientBenchmarkPlaceholder",
    ),
    "HardwareGradientClaimLedgerRow": (
        "scpn_quantum_control.phase.hardware_gradient_publication",
        "HardwareGradientClaimLedgerRow",
    ),
    "HardwareGradientMethodSection": (
        "scpn_quantum_control.phase.hardware_gradient_publication",
        "HardwareGradientMethodSection",
    ),
    "HardwareGradientPreregistration": (
        "scpn_quantum_control.phase.hardware_gradient_publication",
        "HardwareGradientPreregistration",
    ),
    "HardwareGradientPublicationPackage": (
        "scpn_quantum_control.phase.hardware_gradient_publication",
        "HardwareGradientPublicationPackage",
    ),
    "build_hardware_gradient_publication_package": (
        "scpn_quantum_control.phase.hardware_gradient_publication",
        "build_hardware_gradient_publication_package",
    ),
    "check_jax_parameter_shift_agreement": (
        "scpn_quantum_control.phase.jax_bridge",
        "check_jax_parameter_shift_agreement",
    ),
    "is_phase_jax_available": ("scpn_quantum_control.phase.jax_bridge", "is_phase_jax_available"),
    "jax_custom_vjp_qnn_value_and_grad": (
        "scpn_quantum_control.phase.jax_bridge",
        "jax_custom_vjp_qnn_value_and_grad",
    ),
    "jax_native_qnn_value_and_grad": (
        "scpn_quantum_control.phase.jax_bridge",
        "jax_native_qnn_value_and_grad",
    ),
    "jax_parameter_shift_value_and_grad": (
        "scpn_quantum_control.phase.jax_bridge",
        "jax_parameter_shift_value_and_grad",
    ),
    "jax_phase_qnode_aot_export_audit": (
        "scpn_quantum_control.phase.jax_bridge",
        "jax_phase_qnode_aot_export_audit",
    ),
    "jax_phase_qnode_native_transform_audit": (
        "scpn_quantum_control.phase.jax_bridge",
        "jax_phase_qnode_native_transform_audit",
    ),
    "jax_phase_qnode_pytree_transform_audit": (
        "scpn_quantum_control.phase.jax_bridge",
        "jax_phase_qnode_pytree_transform_audit",
    ),
    "jax_phase_qnode_sharding_transform_audit": (
        "scpn_quantum_control.phase.jax_bridge",
        "jax_phase_qnode_sharding_transform_audit",
    ),
    "jax_phase_qnode_value_and_grad": (
        "scpn_quantum_control.phase.jax_bridge",
        "jax_phase_qnode_value_and_grad",
    ),
    "plan_jax_cloud_validation_batch": (
        "scpn_quantum_control.phase.jax_bridge",
        "plan_jax_cloud_validation_batch",
    ),
    "run_jax_jit_compatibility_audit": (
        "scpn_quantum_control.phase.jax_bridge",
        "run_jax_jit_compatibility_audit",
    ),
    "run_jax_maturity_audit": ("scpn_quantum_control.phase.jax_bridge", "run_jax_maturity_audit"),
    "run_jax_nested_transform_algebra_audit": (
        "scpn_quantum_control.phase.jax_bridge",
        "run_jax_nested_transform_algebra_audit",
    ),
    "run_jax_phase_qnode_lowering_matrix": (
        "scpn_quantum_control.phase.jax_bridge",
        "run_jax_phase_qnode_lowering_matrix",
    ),
    "run_jax_pytree_compatibility_audit": (
        "scpn_quantum_control.phase.jax_bridge",
        "run_jax_pytree_compatibility_audit",
    ),
    "run_jax_sharding_compatibility_audit": (
        "scpn_quantum_control.phase.jax_bridge",
        "run_jax_sharding_compatibility_audit",
    ),
    "run_jax_vmap_compatibility_audit": (
        "scpn_quantum_control.phase.jax_bridge",
        "run_jax_vmap_compatibility_audit",
    ),
    "PhaseJAXCloudValidationRunSpec": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXCloudValidationRunSpec",
    ),
    "PhaseJAXCustomVJPQNNGradientResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXCustomVJPQNNGradientResult",
    ),
    "PhaseJAXGradientAgreementResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXGradientAgreementResult",
    ),
    "PhaseJAXJITCompatibilityResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXJITCompatibilityResult",
    ),
    "PhaseJAXMaturityAuditResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXMaturityAuditResult",
    ),
    "PhaseJAXNativeQNNGradientResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXNativeQNNGradientResult",
    ),
    "PhaseJAXNestedTransformAlgebraResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXNestedTransformAlgebraResult",
    ),
    "PhaseJAXNestedTransformRoute": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXNestedTransformRoute",
    ),
    "PhaseJAXParameterShiftResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXParameterShiftResult",
    ),
    "PhaseJAXPhaseQNodeAOTExportResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXPhaseQNodeAOTExportResult",
    ),
    "PhaseJAXPhaseQNodeLoweringMatrixResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXPhaseQNodeLoweringMatrixResult",
    ),
    "PhaseJAXPhaseQNodeLoweringRoute": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXPhaseQNodeLoweringRoute",
    ),
    "PhaseJAXPhaseQNodeNativeTransformResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXPhaseQNodeNativeTransformResult",
    ),
    "PhaseJAXPhaseQNodePyTreeTransformResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXPhaseQNodePyTreeTransformResult",
    ),
    "PhaseJAXPhaseQNodeShardingTransformResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXPhaseQNodeShardingTransformResult",
    ),
    "PhaseJAXPhaseQNodeStatevectorResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXPhaseQNodeStatevectorResult",
    ),
    "PhaseJAXPyTreeCompatibilityResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXPyTreeCompatibilityResult",
    ),
    "PhaseJAXShardingCompatibilityResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXShardingCompatibilityResult",
    ),
    "PhaseJAXVMAPCompatibilityResult": (
        "scpn_quantum_control.phase.jax_bridge_contracts",
        "PhaseJAXVMAPCompatibilityResult",
    ),
    "HigherOrderKuramotoSpec": (
        "scpn_quantum_control.phase.kuramoto_variants",
        "HigherOrderKuramotoSpec",
    ),
    "KuramotoVariant": ("scpn_quantum_control.phase.kuramoto_variants", "KuramotoVariant"),
    "KuramotoVariantResult": (
        "scpn_quantum_control.phase.kuramoto_variants",
        "KuramotoVariantResult",
    ),
    "MonitoredKuramotoSpec": (
        "scpn_quantum_control.phase.kuramoto_variants",
        "MonitoredKuramotoSpec",
    ),
    "PTSymmetricKuramotoSpec": (
        "scpn_quantum_control.phase.kuramoto_variants",
        "PTSymmetricKuramotoSpec",
    ),
    "build_triadic_ring_terms": (
        "scpn_quantum_control.phase.kuramoto_variants",
        "build_triadic_ring_terms",
    ),
    "simulate_higher_order_kuramoto": (
        "scpn_quantum_control.phase.kuramoto_variants",
        "simulate_higher_order_kuramoto",
    ),
    "simulate_monitored_kuramoto": (
        "scpn_quantum_control.phase.kuramoto_variants",
        "simulate_monitored_kuramoto",
    ),
    "simulate_pt_symmetric_kuramoto": (
        "scpn_quantum_control.phase.kuramoto_variants",
        "simulate_pt_symmetric_kuramoto",
    ),
    "LindbladSyncEngine": ("scpn_quantum_control.phase.lindblad_engine", "LindbladSyncEngine"),
    "DifferentiableModelTrainingEvidenceSuite": (
        "scpn_quantum_control.phase.model_training_evidence",
        "DifferentiableModelTrainingEvidenceSuite",
    ),
    "DifferentiableModelTrainingRecord": (
        "scpn_quantum_control.phase.model_training_evidence",
        "DifferentiableModelTrainingRecord",
    ),
    "RegisteredDifferentiableTrainingSuiteAuditResult": (
        "scpn_quantum_control.phase.model_training_evidence",
        "RegisteredDifferentiableTrainingSuiteAuditResult",
    ),
    "RegisteredDifferentiableTrainingSuiteRecord": (
        "scpn_quantum_control.phase.model_training_evidence",
        "RegisteredDifferentiableTrainingSuiteRecord",
    ),
    "run_differentiable_model_training_evidence_suite": (
        "scpn_quantum_control.phase.model_training_evidence",
        "run_differentiable_model_training_evidence_suite",
    ),
    "run_registered_differentiable_training_suite_audit": (
        "scpn_quantum_control.phase.model_training_evidence",
        "run_registered_differentiable_training_suite_audit",
    ),
    "NaturalGradientDirection": (
        "scpn_quantum_control.phase.natural_gradient",
        "NaturalGradientDirection",
    ),
    "NaturalGradientRegularizationPolicy": (
        "scpn_quantum_control.phase.natural_gradient",
        "NaturalGradientRegularizationPolicy",
    ),
    "ParameterShiftNaturalGradientCertificate": (
        "scpn_quantum_control.phase.natural_gradient",
        "ParameterShiftNaturalGradientCertificate",
    ),
    "ParameterShiftNaturalGradientResult": (
        "scpn_quantum_control.phase.natural_gradient",
        "ParameterShiftNaturalGradientResult",
    ),
    "ParameterShiftNaturalGradientStep": (
        "scpn_quantum_control.phase.natural_gradient",
        "ParameterShiftNaturalGradientStep",
    ),
    "parameter_shift_natural_gradient_descent": (
        "scpn_quantum_control.phase.natural_gradient",
        "parameter_shift_natural_gradient_descent",
    ),
    "solve_natural_gradient_direction": (
        "scpn_quantum_control.phase.natural_gradient",
        "solve_natural_gradient_direction",
    ),
    "validate_natural_gradient_training": (
        "scpn_quantum_control.phase.natural_gradient",
        "validate_natural_gradient_training",
    ),
    "ComposedObjectiveAuditSuiteResult": (
        "scpn_quantum_control.phase.objective_audit",
        "ComposedObjectiveAuditSuiteResult",
    ),
    "ComposedObjectiveGradientAgreement": (
        "scpn_quantum_control.phase.objective_audit",
        "ComposedObjectiveGradientAgreement",
    ),
    "run_composed_objective_audit_suite": (
        "scpn_quantum_control.phase.objective_audit",
        "run_composed_objective_audit_suite",
    ),
    "verify_composed_objective_gradient": (
        "scpn_quantum_control.phase.objective_audit",
        "verify_composed_objective_gradient",
    ),
    "ComposedObjectiveExecutionPlan": (
        "scpn_quantum_control.phase.objective_planner",
        "ComposedObjectiveExecutionPlan",
    ),
    "ComposedObjectivePlannerAuditResult": (
        "scpn_quantum_control.phase.objective_planner",
        "ComposedObjectivePlannerAuditResult",
    ),
    "assert_composed_objective_execution_supported": (
        "scpn_quantum_control.phase.objective_planner",
        "assert_composed_objective_execution_supported",
    ),
    "plan_composed_objective_execution": (
        "scpn_quantum_control.phase.objective_planner",
        "plan_composed_objective_execution",
    ),
    "run_composed_objective_planner_audit": (
        "scpn_quantum_control.phase.objective_planner",
        "run_composed_objective_planner_audit",
    ),
    "ComposedObjectiveTrainingCertificate": (
        "scpn_quantum_control.phase.objectives",
        "ComposedObjectiveTrainingCertificate",
    ),
    "ComposedObjectiveTrainingResult": (
        "scpn_quantum_control.phase.objectives",
        "ComposedObjectiveTrainingResult",
    ),
    "ComposedObjectiveTrainingStep": (
        "scpn_quantum_control.phase.objectives",
        "ComposedObjectiveTrainingStep",
    ),
    "ComposedPhaseObjective": ("scpn_quantum_control.phase.objectives", "ComposedPhaseObjective"),
    "ObjectiveGradientEvaluation": (
        "scpn_quantum_control.phase.objectives",
        "ObjectiveGradientEvaluation",
    ),
    "ObjectiveTerm": ("scpn_quantum_control.phase.objectives", "ObjectiveTerm"),
    "ObjectiveTermValue": ("scpn_quantum_control.phase.objectives", "ObjectiveTermValue"),
    "build_phase_control_objective": (
        "scpn_quantum_control.phase.objectives",
        "build_phase_control_objective",
    ),
    "periodic_regularization_term": (
        "scpn_quantum_control.phase.objectives",
        "periodic_regularization_term",
    ),
    "phase_energy_term": ("scpn_quantum_control.phase.objectives", "phase_energy_term"),
    "phase_fidelity_target_term": (
        "scpn_quantum_control.phase.objectives",
        "phase_fidelity_target_term",
    ),
    "phase_symmetry_penalty_term": (
        "scpn_quantum_control.phase.objectives",
        "phase_symmetry_penalty_term",
    ),
    "smooth_box_safety_penalty_term": (
        "scpn_quantum_control.phase.objectives",
        "smooth_box_safety_penalty_term",
    ),
    "train_composed_phase_objective": (
        "scpn_quantum_control.phase.objectives",
        "train_composed_phase_objective",
    ),
    "validate_composed_objective_training": (
        "scpn_quantum_control.phase.objectives",
        "validate_composed_objective_training",
    ),
    "OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.open_system_objectives",
        "OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY",
    ),
    "OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS": (
        "scpn_quantum_control.phase.open_system_objectives",
        "OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS",
    ),
    "BoundedOpenSystemObjectiveCase": (
        "scpn_quantum_control.phase.open_system_objectives",
        "BoundedOpenSystemObjectiveCase",
    ),
    "DensityMatrixInvariantCertificate": (
        "scpn_quantum_control.phase.open_system_objectives",
        "DensityMatrixInvariantCertificate",
    ),
    "MCWFReproducibilityCertificate": (
        "scpn_quantum_control.phase.open_system_objectives",
        "MCWFReproducibilityCertificate",
    ),
    "OpenSystemObjectiveBoundaryRow": (
        "scpn_quantum_control.phase.open_system_objectives",
        "OpenSystemObjectiveBoundaryRow",
    ),
    "OpenSystemObjectiveRecord": (
        "scpn_quantum_control.phase.open_system_objectives",
        "OpenSystemObjectiveRecord",
    ),
    "OpenSystemObjectiveSuiteResult": (
        "scpn_quantum_control.phase.open_system_objectives",
        "OpenSystemObjectiveSuiteResult",
    ),
    "certify_density_matrix_invariants": (
        "scpn_quantum_control.phase.open_system_objectives",
        "certify_density_matrix_invariants",
    ),
    "certify_mcwf_reproducibility": (
        "scpn_quantum_control.phase.open_system_objectives",
        "certify_mcwf_reproducibility",
    ),
    "default_open_system_objective_cases": (
        "scpn_quantum_control.phase.open_system_objectives",
        "default_open_system_objective_cases",
    ),
    "evaluate_lindblad_objective": (
        "scpn_quantum_control.phase.open_system_objectives",
        "evaluate_lindblad_objective",
    ),
    "evaluate_mcwf_objective": (
        "scpn_quantum_control.phase.open_system_objectives",
        "evaluate_mcwf_objective",
    ),
    "open_system_objective_boundary_rows": (
        "scpn_quantum_control.phase.open_system_objectives",
        "open_system_objective_boundary_rows",
    ),
    "run_open_system_objective_suite": (
        "scpn_quantum_control.phase.open_system_objectives",
        "run_open_system_objective_suite",
    ),
    "OptimizerComparisonSuiteResult": (
        "scpn_quantum_control.phase.optimizer_audit",
        "OptimizerComparisonSuiteResult",
    ),
    "OptimizerConvergenceRecord": (
        "scpn_quantum_control.phase.optimizer_audit",
        "OptimizerConvergenceRecord",
    ),
    "run_parameter_shift_optimizer_comparison": (
        "scpn_quantum_control.phase.optimizer_audit",
        "run_parameter_shift_optimizer_comparison",
    ),
    "GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.optimizer_convergence_suite",
        "GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY",
    ),
    "GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS": (
        "scpn_quantum_control.phase.optimizer_convergence_suite",
        "GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS",
    ),
    "GroundStateConvergenceCertificate": (
        "scpn_quantum_control.phase.optimizer_convergence_suite",
        "GroundStateConvergenceCertificate",
    ),
    "GroundStateOptimizerBoundaryRow": (
        "scpn_quantum_control.phase.optimizer_convergence_suite",
        "GroundStateOptimizerBoundaryRow",
    ),
    "GroundStateOptimizerConvergenceSuiteResult": (
        "scpn_quantum_control.phase.optimizer_convergence_suite",
        "GroundStateOptimizerConvergenceSuiteResult",
    ),
    "GroundStateOptimizerRunRecord": (
        "scpn_quantum_control.phase.optimizer_convergence_suite",
        "GroundStateOptimizerRunRecord",
    ),
    "KnownGroundStateObjective": (
        "scpn_quantum_control.phase.optimizer_convergence_suite",
        "KnownGroundStateObjective",
    ),
    "default_ground_state_optimizer_objectives": (
        "scpn_quantum_control.phase.optimizer_convergence_suite",
        "default_ground_state_optimizer_objectives",
    ),
    "run_ground_state_optimizer_convergence_suite": (
        "scpn_quantum_control.phase.optimizer_convergence_suite",
        "run_ground_state_optimizer_convergence_suite",
    ),
    "GenericParameterShiftEvaluationPlan": (
        "scpn_quantum_control.phase.param_shift",
        "GenericParameterShiftEvaluationPlan",
    ),
    "GradientVerificationResult": (
        "scpn_quantum_control.phase.param_shift",
        "GradientVerificationResult",
    ),
    "HessianVerificationResult": (
        "scpn_quantum_control.phase.param_shift",
        "HessianVerificationResult",
    ),
    "ParamShiftConvergenceDiagnostics": (
        "scpn_quantum_control.phase.param_shift",
        "ParamShiftConvergenceDiagnostics",
    ),
    "ParamShiftVQEResult": ("scpn_quantum_control.phase.param_shift", "ParamShiftVQEResult"),
    "multi_frequency_parameter_shift_rule": (
        "scpn_quantum_control.phase.param_shift",
        "multi_frequency_parameter_shift_rule",
    ),
    "parameter_shift_gradient": (
        "scpn_quantum_control.phase.param_shift",
        "parameter_shift_gradient",
    ),
    "parameter_shift_gradient_with_uncertainty": (
        "scpn_quantum_control.phase.param_shift",
        "parameter_shift_gradient_with_uncertainty",
    ),
    "parameter_shift_hessian": (
        "scpn_quantum_control.phase.param_shift",
        "parameter_shift_hessian",
    ),
    "plan_generic_parameter_shift_evaluations": (
        "scpn_quantum_control.phase.param_shift",
        "plan_generic_parameter_shift_evaluations",
    ),
    "plan_parameter_shift_shots": (
        "scpn_quantum_control.phase.param_shift",
        "plan_parameter_shift_shots",
    ),
    "validate_param_shift_convergence": (
        "scpn_quantum_control.phase.param_shift",
        "validate_param_shift_convergence",
    ),
    "value_and_parameter_shift_grad": (
        "scpn_quantum_control.phase.param_shift",
        "value_and_parameter_shift_grad",
    ),
    "value_and_vqe_grad": ("scpn_quantum_control.phase.param_shift", "value_and_vqe_grad"),
    "verify_parameter_shift_gradient": (
        "scpn_quantum_control.phase.param_shift",
        "verify_parameter_shift_gradient",
    ),
    "verify_parameter_shift_hessian": (
        "scpn_quantum_control.phase.param_shift",
        "verify_parameter_shift_hessian",
    ),
    "verify_vqe_parameter_shift_gradient": (
        "scpn_quantum_control.phase.param_shift",
        "verify_vqe_parameter_shift_gradient",
    ),
    "verify_vqe_parameter_shift_hessian": (
        "scpn_quantum_control.phase.param_shift",
        "verify_vqe_parameter_shift_hessian",
    ),
    "vqe_with_param_shift": ("scpn_quantum_control.phase.param_shift", "vqe_with_param_shift"),
    "PennyLaneGradientAgreementResult": (
        "scpn_quantum_control.phase.pennylane_bridge",
        "PennyLaneGradientAgreementResult",
    ),
    "PennyLaneMaturityAuditResult": (
        "scpn_quantum_control.phase.pennylane_bridge",
        "PennyLaneMaturityAuditResult",
    ),
    "PennyLaneQNodeConversionResult": (
        "scpn_quantum_control.phase.pennylane_bridge",
        "PennyLaneQNodeConversionResult",
    ),
    "PennyLaneRoundTripResult": (
        "scpn_quantum_control.phase.pennylane_bridge",
        "PennyLaneRoundTripResult",
    ),
    "build_pennylane_qnode_from_phase_qnode": (
        "scpn_quantum_control.phase.pennylane_bridge",
        "build_pennylane_qnode_from_phase_qnode",
    ),
    "check_pennylane_parameter_shift_agreement": (
        "scpn_quantum_control.phase.pennylane_bridge",
        "check_pennylane_parameter_shift_agreement",
    ),
    "check_pennylane_phase_qnode_round_trip": (
        "scpn_quantum_control.phase.pennylane_bridge",
        "check_pennylane_phase_qnode_round_trip",
    ),
    "check_pennylane_qnode_round_trip": (
        "scpn_quantum_control.phase.pennylane_bridge",
        "check_pennylane_qnode_round_trip",
    ),
    "is_phase_pennylane_available": (
        "scpn_quantum_control.phase.pennylane_bridge",
        "is_phase_pennylane_available",
    ),
    "run_pennylane_maturity_audit": (
        "scpn_quantum_control.phase.pennylane_bridge",
        "run_pennylane_maturity_audit",
    ),
    "PennyLaneImportResult": (
        "scpn_quantum_control.phase.pennylane_import",
        "PennyLaneImportResult",
    ),
    "PennyLaneImportRoundTripResult": (
        "scpn_quantum_control.phase.pennylane_import",
        "PennyLaneImportRoundTripResult",
    ),
    "check_pennylane_phase_qnode_import_round_trip": (
        "scpn_quantum_control.phase.pennylane_import",
        "check_pennylane_phase_qnode_import_round_trip",
    ),
    "import_phase_qnode_from_pennylane": (
        "scpn_quantum_control.phase.pennylane_import",
        "import_phase_qnode_from_pennylane",
    ),
    "is_pennylane_import_available": (
        "scpn_quantum_control.phase.pennylane_import",
        "is_pennylane_import_available",
    ),
    "PennyLaneHardwarePluginExecutionArtifact": (
        "scpn_quantum_control.phase.pennylane_provider_plugin",
        "PennyLaneHardwarePluginExecutionArtifact",
    ),
    "PennyLanePluginMatrixResult": (
        "scpn_quantum_control.phase.pennylane_provider_plugin",
        "PennyLanePluginMatrixResult",
    ),
    "PennyLanePluginMatrixRoute": (
        "scpn_quantum_control.phase.pennylane_provider_plugin",
        "PennyLanePluginMatrixRoute",
    ),
    "PennyLaneProviderEvidenceBundle": (
        "scpn_quantum_control.phase.pennylane_provider_plugin",
        "PennyLaneProviderEvidenceBundle",
    ),
    "PennyLaneProviderGradientParityArtifact": (
        "scpn_quantum_control.phase.pennylane_provider_plugin",
        "PennyLaneProviderGradientParityArtifact",
    ),
    "PennyLaneProviderPluginExecutionArtifact": (
        "scpn_quantum_control.phase.pennylane_provider_plugin",
        "PennyLaneProviderPluginExecutionArtifact",
    ),
    "run_pennylane_plugin_matrix": (
        "scpn_quantum_control.phase.pennylane_provider_plugin",
        "run_pennylane_plugin_matrix",
    ),
    "PhaseVQE": ("scpn_quantum_control.phase.phase_vqe", "PhaseVQE"),
    "PhaseVQEResult": ("scpn_quantum_control.phase.phase_vqe", "PhaseVQEResult"),
    "ProviderExpectationSample": (
        "scpn_quantum_control.phase.provider_gradient",
        "ProviderExpectationSample",
    ),
    "ProviderGradientExecutionResult": (
        "scpn_quantum_control.phase.provider_gradient",
        "ProviderGradientExecutionResult",
    ),
    "ProviderHardwareGradientPreparationResult": (
        "scpn_quantum_control.phase.provider_gradient",
        "ProviderHardwareGradientPreparationResult",
    ),
    "ProviderParameterShiftRecord": (
        "scpn_quantum_control.phase.provider_gradient",
        "ProviderParameterShiftRecord",
    ),
    "execute_provider_parameter_shift_gradient": (
        "scpn_quantum_control.phase.provider_gradient",
        "execute_provider_parameter_shift_gradient",
    ),
    "prepare_provider_hardware_parameter_shift_gradient": (
        "scpn_quantum_control.phase.provider_gradient",
        "prepare_provider_hardware_parameter_shift_gradient",
    ),
    "ProviderGradientReadinessAuditResult": (
        "scpn_quantum_control.phase.provider_gradient_audit",
        "ProviderGradientReadinessAuditResult",
    ),
    "ProviderGradientReadinessRecord": (
        "scpn_quantum_control.phase.provider_gradient_audit",
        "ProviderGradientReadinessRecord",
    ),
    "ProviderGradientReadinessScenario": (
        "scpn_quantum_control.phase.provider_gradient_audit",
        "ProviderGradientReadinessScenario",
    ),
    "default_provider_gradient_readiness_scenarios": (
        "scpn_quantum_control.phase.provider_gradient_audit",
        "default_provider_gradient_readiness_scenarios",
    ),
    "run_provider_gradient_readiness_audit": (
        "scpn_quantum_control.phase.provider_gradient_audit",
        "run_provider_gradient_readiness_audit",
    ),
    "ProviderHardwareGradientPreparationAuditResult": (
        "scpn_quantum_control.phase.provider_hardware_gradient_audit",
        "ProviderHardwareGradientPreparationAuditResult",
    ),
    "ProviderHardwareGradientPreparationRecord": (
        "scpn_quantum_control.phase.provider_hardware_gradient_audit",
        "ProviderHardwareGradientPreparationRecord",
    ),
    "ProviderHardwareGradientPreparationScenario": (
        "scpn_quantum_control.phase.provider_hardware_gradient_audit",
        "ProviderHardwareGradientPreparationScenario",
    ),
    "default_provider_hardware_gradient_preparation_scenarios": (
        "scpn_quantum_control.phase.provider_hardware_gradient_audit",
        "default_provider_hardware_gradient_preparation_scenarios",
    ),
    "run_provider_hardware_gradient_preparation_audit": (
        "scpn_quantum_control.phase.provider_hardware_gradient_audit",
        "run_provider_hardware_gradient_preparation_audit",
    ),
    "DifferentiableProviderHardwareEvidenceChain": (
        "scpn_quantum_control.phase.provider_hardware_safety_audit",
        "DifferentiableProviderHardwareEvidenceChain",
    ),
    "DifferentiableProviderHardwareSafetyAuditResult": (
        "scpn_quantum_control.phase.provider_hardware_safety_audit",
        "DifferentiableProviderHardwareSafetyAuditResult",
    ),
    "DifferentiableProviderHardwareSafetySurface": (
        "scpn_quantum_control.phase.provider_hardware_safety_audit",
        "DifferentiableProviderHardwareSafetySurface",
    ),
    "run_differentiable_provider_hardware_safety_audit": (
        "scpn_quantum_control.phase.provider_hardware_safety_audit",
        "run_differentiable_provider_hardware_safety_audit",
    ),
    "HypergeometricPulse": ("scpn_quantum_control.phase.pulse_shaping", "HypergeometricPulse"),
    "ICIPulse": ("scpn_quantum_control.phase.pulse_shaping", "ICIPulse"),
    "PulseSchedule": ("scpn_quantum_control.phase.pulse_shaping", "PulseSchedule"),
    "build_hypergeometric_pulse": (
        "scpn_quantum_control.phase.pulse_shaping",
        "build_hypergeometric_pulse",
    ),
    "build_ici_pulse": ("scpn_quantum_control.phase.pulse_shaping", "build_ici_pulse"),
    "build_trotter_pulse_schedule": (
        "scpn_quantum_control.phase.pulse_shaping",
        "build_trotter_pulse_schedule",
    ),
    "hypergeometric_envelope": (
        "scpn_quantum_control.phase.pulse_shaping",
        "hypergeometric_envelope",
    ),
    "ici_three_level_evolution": (
        "scpn_quantum_control.phase.pulse_shaping",
        "ici_three_level_evolution",
    ),
    "infidelity_bound": ("scpn_quantum_control.phase.pulse_shaping", "infidelity_bound"),
    "KnmGraph": ("scpn_quantum_control.phase.qgnn", "KnmGraph"),
    "QGNNConfig": ("scpn_quantum_control.phase.qgnn", "QGNNConfig"),
    "QGNNTrainingResult": ("scpn_quantum_control.phase.qgnn", "QGNNTrainingResult"),
    "synthetic_kuramoto_target": ("scpn_quantum_control.phase.qgnn", "synthetic_kuramoto_target"),
    "QiskitCalibrationStatevectorComparisonArtifact": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "QiskitCalibrationStatevectorComparisonArtifact",
    ),
    "QiskitMaturityAuditResult": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "QiskitMaturityAuditResult",
    ),
    "QiskitParameterShiftGradientResult": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "QiskitParameterShiftGradientResult",
    ),
    "QiskitParameterShiftRecord": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "QiskitParameterShiftRecord",
    ),
    "QiskitProviderGradientWorkflowArtifact": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "QiskitProviderGradientWorkflowArtifact",
    ),
    "QiskitRawCountReplayArtifact": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "QiskitRawCountReplayArtifact",
    ),
    "QiskitRuntimePrimitiveExecutionArtifact": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "QiskitRuntimePrimitiveExecutionArtifact",
    ),
    "QiskitRuntimeQPUExecutionArtifact": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "QiskitRuntimeQPUExecutionArtifact",
    ),
    "QiskitRuntimeQPUProviderEvidenceBundle": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "QiskitRuntimeQPUProviderEvidenceBundle",
    ),
    "build_qiskit_provider_gradient_workflow_artifact": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "build_qiskit_provider_gradient_workflow_artifact",
    ),
    "build_qiskit_runtime_qpu_execution_artifact": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "build_qiskit_runtime_qpu_execution_artifact",
    ),
    "build_qiskit_runtime_qpu_provider_evidence_bundle": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "build_qiskit_runtime_qpu_provider_evidence_bundle",
    ),
    "execute_qiskit_finite_shot_parameter_shift": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "execute_qiskit_finite_shot_parameter_shift",
    ),
    "execute_qiskit_statevector_parameter_shift": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "execute_qiskit_statevector_parameter_shift",
    ),
    "generate_qiskit_parameter_shift_circuits": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "generate_qiskit_parameter_shift_circuits",
    ),
    "run_qiskit_maturity_audit": (
        "scpn_quantum_control.phase.qiskit_bridge",
        "run_qiskit_maturity_audit",
    ),
    "ExternalGradientMap": ("scpn_quantum_control.phase.qnn_conformance", "ExternalGradientMap"),
    "ParameterShiftQNNConformanceCaseResult": (
        "scpn_quantum_control.phase.qnn_conformance",
        "ParameterShiftQNNConformanceCaseResult",
    ),
    "ParameterShiftQNNConformanceSuiteResult": (
        "scpn_quantum_control.phase.qnn_conformance",
        "ParameterShiftQNNConformanceSuiteResult",
    ),
    "ParameterShiftQNNUnsupportedScenario": (
        "scpn_quantum_control.phase.qnn_conformance",
        "ParameterShiftQNNUnsupportedScenario",
    ),
    "run_parameter_shift_qnn_conformance_suite": (
        "scpn_quantum_control.phase.qnn_conformance",
        "run_parameter_shift_qnn_conformance_suite",
    ),
    "summarize_parameter_shift_qnn_unsuitable_scenarios": (
        "scpn_quantum_control.phase.qnn_conformance",
        "summarize_parameter_shift_qnn_unsuitable_scenarios",
    ),
    "ParameterShiftQNNConvergenceCaseResult": (
        "scpn_quantum_control.phase.qnn_convergence",
        "ParameterShiftQNNConvergenceCaseResult",
    ),
    "ParameterShiftQNNConvergenceSuiteResult": (
        "scpn_quantum_control.phase.qnn_convergence",
        "ParameterShiftQNNConvergenceSuiteResult",
    ),
    "ParameterShiftQNNConvergenceUnsuitableScenario": (
        "scpn_quantum_control.phase.qnn_convergence",
        "ParameterShiftQNNConvergenceUnsuitableScenario",
    ),
    "ParameterShiftQNNMultiSeedConvergenceCaseResult": (
        "scpn_quantum_control.phase.qnn_convergence",
        "ParameterShiftQNNMultiSeedConvergenceCaseResult",
    ),
    "ParameterShiftQNNMultiSeedConvergenceRunResult": (
        "scpn_quantum_control.phase.qnn_convergence",
        "ParameterShiftQNNMultiSeedConvergenceRunResult",
    ),
    "ParameterShiftQNNMultiSeedConvergenceSuiteResult": (
        "scpn_quantum_control.phase.qnn_convergence",
        "ParameterShiftQNNMultiSeedConvergenceSuiteResult",
    ),
    "run_parameter_shift_qnn_convergence_suite": (
        "scpn_quantum_control.phase.qnn_convergence",
        "run_parameter_shift_qnn_convergence_suite",
    ),
    "run_parameter_shift_qnn_multi_seed_convergence_suite": (
        "scpn_quantum_control.phase.qnn_convergence",
        "run_parameter_shift_qnn_multi_seed_convergence_suite",
    ),
    "summarize_parameter_shift_qnn_convergence_unsuitable_scenarios": (
        "scpn_quantum_control.phase.qnn_convergence",
        "summarize_parameter_shift_qnn_convergence_unsuitable_scenarios",
    ),
    "ParameterShiftQNNFiniteShotConvergenceCaseResult": (
        "scpn_quantum_control.phase.qnn_finite_shot",
        "ParameterShiftQNNFiniteShotConvergenceCaseResult",
    ),
    "ParameterShiftQNNFiniteShotConvergenceSuiteResult": (
        "scpn_quantum_control.phase.qnn_finite_shot",
        "ParameterShiftQNNFiniteShotConvergenceSuiteResult",
    ),
    "ParameterShiftQNNFiniteShotGradientResult": (
        "scpn_quantum_control.phase.qnn_finite_shot",
        "ParameterShiftQNNFiniteShotGradientResult",
    ),
    "ParameterShiftQNNFiniteShotProbeRecord": (
        "scpn_quantum_control.phase.qnn_finite_shot",
        "ParameterShiftQNNFiniteShotProbeRecord",
    ),
    "ParameterShiftQNNFiniteShotUnsupportedScenario": (
        "scpn_quantum_control.phase.qnn_finite_shot",
        "ParameterShiftQNNFiniteShotUnsupportedScenario",
    ),
    "estimate_parameter_shift_qnn_finite_shot_gradient": (
        "scpn_quantum_control.phase.qnn_finite_shot",
        "estimate_parameter_shift_qnn_finite_shot_gradient",
    ),
    "run_parameter_shift_qnn_finite_shot_convergence_suite": (
        "scpn_quantum_control.phase.qnn_finite_shot",
        "run_parameter_shift_qnn_finite_shot_convergence_suite",
    ),
    "summarize_parameter_shift_qnn_finite_shot_unsuitable_scenarios": (
        "scpn_quantum_control.phase.qnn_finite_shot",
        "summarize_parameter_shift_qnn_finite_shot_unsuitable_scenarios",
    ),
    "FrameworkGradientCaseMap": (
        "scpn_quantum_control.phase.qnn_framework_agreement",
        "FrameworkGradientCaseMap",
    ),
    "FrameworkGradientMap": (
        "scpn_quantum_control.phase.qnn_framework_agreement",
        "FrameworkGradientMap",
    ),
    "ParameterShiftQNNFrameworkAgreementResult": (
        "scpn_quantum_control.phase.qnn_framework_agreement",
        "ParameterShiftQNNFrameworkAgreementResult",
    ),
    "ParameterShiftQNNFrameworkAgreementSuiteResult": (
        "scpn_quantum_control.phase.qnn_framework_agreement",
        "ParameterShiftQNNFrameworkAgreementSuiteResult",
    ),
    "ParameterShiftQNNFrameworkGradientAgreement": (
        "scpn_quantum_control.phase.qnn_framework_agreement",
        "ParameterShiftQNNFrameworkGradientAgreement",
    ),
    "run_parameter_shift_qnn_framework_agreement_suite": (
        "scpn_quantum_control.phase.qnn_framework_agreement",
        "run_parameter_shift_qnn_framework_agreement_suite",
    ),
    "verify_parameter_shift_qnn_framework_agreement": (
        "scpn_quantum_control.phase.qnn_framework_agreement",
        "verify_parameter_shift_qnn_framework_agreement",
    ),
    "BoundedQNNFrameworkBridgeCapability": (
        "scpn_quantum_control.phase.qnn_framework_bridge_matrix",
        "BoundedQNNFrameworkBridgeCapability",
    ),
    "BoundedQNNFrameworkBridgeMatrixResult": (
        "scpn_quantum_control.phase.qnn_framework_bridge_matrix",
        "BoundedQNNFrameworkBridgeMatrixResult",
    ),
    "assert_bounded_qnn_framework_bridge_supported": (
        "scpn_quantum_control.phase.qnn_framework_bridge_matrix",
        "assert_bounded_qnn_framework_bridge_supported",
    ),
    "run_bounded_qnn_framework_bridge_matrix": (
        "scpn_quantum_control.phase.qnn_framework_bridge_matrix",
        "run_bounded_qnn_framework_bridge_matrix",
    ),
    "ParameterShiftQNNLossLandscapeCaseResult": (
        "scpn_quantum_control.phase.qnn_loss_landscape",
        "ParameterShiftQNNLossLandscapeCaseResult",
    ),
    "ParameterShiftQNNLossLandscapePoint": (
        "scpn_quantum_control.phase.qnn_loss_landscape",
        "ParameterShiftQNNLossLandscapePoint",
    ),
    "ParameterShiftQNNLossLandscapeSuiteResult": (
        "scpn_quantum_control.phase.qnn_loss_landscape",
        "ParameterShiftQNNLossLandscapeSuiteResult",
    ),
    "run_parameter_shift_qnn_loss_landscape_suite": (
        "scpn_quantum_control.phase.qnn_loss_landscape",
        "run_parameter_shift_qnn_loss_landscape_suite",
    ),
    "DerivativeFreeCandidateMap": (
        "scpn_quantum_control.phase.qnn_optimizer_benchmark",
        "DerivativeFreeCandidateMap",
    ),
    "ParameterShiftQNNOptimizerBenchmarkCaseResult": (
        "scpn_quantum_control.phase.qnn_optimizer_benchmark",
        "ParameterShiftQNNOptimizerBenchmarkCaseResult",
    ),
    "ParameterShiftQNNOptimizerBenchmarkSuiteResult": (
        "scpn_quantum_control.phase.qnn_optimizer_benchmark",
        "ParameterShiftQNNOptimizerBenchmarkSuiteResult",
    ),
    "QNNOptimizerBaselineResult": (
        "scpn_quantum_control.phase.qnn_optimizer_benchmark",
        "QNNOptimizerBaselineResult",
    ),
    "run_parameter_shift_qnn_optimizer_benchmark_suite": (
        "scpn_quantum_control.phase.qnn_optimizer_benchmark",
        "run_parameter_shift_qnn_optimizer_benchmark_suite",
    ),
    "ParameterShiftQNNExternalGradientAgreement": (
        "scpn_quantum_control.phase.qnn_training",
        "ParameterShiftQNNExternalGradientAgreement",
    ),
    "ParameterShiftQNNGradientVerificationResult": (
        "scpn_quantum_control.phase.qnn_training",
        "ParameterShiftQNNGradientVerificationResult",
    ),
    "ParameterShiftQNNPredictionResult": (
        "scpn_quantum_control.phase.qnn_training",
        "ParameterShiftQNNPredictionResult",
    ),
    "ParameterShiftQNNTrainingResult": (
        "scpn_quantum_control.phase.qnn_training",
        "ParameterShiftQNNTrainingResult",
    ),
    "parameter_shift_qnn_classifier_gradient": (
        "scpn_quantum_control.phase.qnn_training",
        "parameter_shift_qnn_classifier_gradient",
    ),
    "parameter_shift_qnn_classifier_loss": (
        "scpn_quantum_control.phase.qnn_training",
        "parameter_shift_qnn_classifier_loss",
    ),
    "predict_parameter_shift_qnn_classifier": (
        "scpn_quantum_control.phase.qnn_training",
        "predict_parameter_shift_qnn_classifier",
    ),
    "train_parameter_shift_qnn_classifier": (
        "scpn_quantum_control.phase.qnn_training",
        "train_parameter_shift_qnn_classifier",
    ),
    "verify_parameter_shift_qnn_classifier_gradient": (
        "scpn_quantum_control.phase.qnn_training",
        "verify_parameter_shift_qnn_classifier_gradient",
    ),
    "PhaseQNodeAffinityArtifactValidation": (
        "scpn_quantum_control.phase.qnode_affinity_benchmark",
        "PhaseQNodeAffinityArtifactValidation",
    ),
    "PhaseQNodeAffinityBenchmarkMetadata": (
        "scpn_quantum_control.phase.qnode_affinity_benchmark",
        "PhaseQNodeAffinityBenchmarkMetadata",
    ),
    "PhaseQNodeAffinityBenchmarkResult": (
        "scpn_quantum_control.phase.qnode_affinity_benchmark",
        "PhaseQNodeAffinityBenchmarkResult",
    ),
    "classify_affinity_evidence": (
        "scpn_quantum_control.phase.qnode_affinity_benchmark",
        "classify_affinity_evidence",
    ),
    "run_phase_qnode_affinity_benchmark": (
        "scpn_quantum_control.phase.qnode_affinity_benchmark",
        "run_phase_qnode_affinity_benchmark",
    ),
    "validate_phase_qnode_affinity_artifact": (
        "scpn_quantum_control.phase.qnode_affinity_benchmark",
        "validate_phase_qnode_affinity_artifact",
    ),
    "DenseHermitianObservable": (
        "scpn_quantum_control.phase.qnode_circuit",
        "DenseHermitianObservable",
    ),
    "PauliCovarianceObservable": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PauliCovarianceObservable",
    ),
    "PauliTerm": ("scpn_quantum_control.phase.qnode_circuit", "PauliTerm"),
    "PhaseQNodeCircuit": ("scpn_quantum_control.phase.qnode_circuit", "PhaseQNodeCircuit"),
    "PhaseQNodeClassicalFisherResult": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeClassicalFisherResult",
    ),
    "PhaseQNodeDensityCircuit": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeDensityCircuit",
    ),
    "PhaseQNodeDensityExecutionResult": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeDensityExecutionResult",
    ),
    "PhaseQNodeDepthProfile": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeDepthProfile",
    ),
    "PhaseQNodeExecutionResult": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeExecutionResult",
    ),
    "PhaseQNodeGradientEvaluationGroup": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeGradientEvaluationGroup",
    ),
    "PhaseQNodeGradientEvaluationPlan": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeGradientEvaluationPlan",
    ),
    "PhaseQNodeGradientResult": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeGradientResult",
    ),
    "PhaseQNodeMetricTensorResult": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeMetricTensorResult",
    ),
    "PhaseQNodeNoiseChannel": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeNoiseChannel",
    ),
    "PhaseQNodeOperation": ("scpn_quantum_control.phase.qnode_circuit", "PhaseQNodeOperation"),
    "PhaseQNodeRegisteredCircuitSpec": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeRegisteredCircuitSpec",
    ),
    "PhaseQNodeSupportError": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeSupportError",
    ),
    "PhaseQNodeSupportReport": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeSupportReport",
    ),
    "PhaseQNodeTemplateSpec": (
        "scpn_quantum_control.phase.qnode_circuit",
        "PhaseQNodeTemplateSpec",
    ),
    "SparsePauliHamiltonian": (
        "scpn_quantum_control.phase.qnode_circuit",
        "SparsePauliHamiltonian",
    ),
    "build_phase_qnode_template": (
        "scpn_quantum_control.phase.qnode_circuit",
        "build_phase_qnode_template",
    ),
    "build_registered_phase_qnode_circuit": (
        "scpn_quantum_control.phase.qnode_circuit",
        "build_registered_phase_qnode_circuit",
    ),
    "build_sparse_ising_chain_hamiltonian": (
        "scpn_quantum_control.phase.qnode_circuit",
        "build_sparse_ising_chain_hamiltonian",
    ),
    "decompose_phase_qnode_controlled_gate": (
        "scpn_quantum_control.phase.qnode_circuit",
        "decompose_phase_qnode_controlled_gate",
    ),
    "execute_phase_qnode_circuit": (
        "scpn_quantum_control.phase.qnode_circuit",
        "execute_phase_qnode_circuit",
    ),
    "execute_phase_qnode_density_matrix": (
        "scpn_quantum_control.phase.qnode_circuit",
        "execute_phase_qnode_density_matrix",
    ),
    "parameter_shift_phase_qnode_gradient": (
        "scpn_quantum_control.phase.qnode_circuit",
        "parameter_shift_phase_qnode_gradient",
    ),
    "phase_qnode_computational_basis_fisher_information": (
        "scpn_quantum_control.phase.qnode_circuit",
        "phase_qnode_computational_basis_fisher_information",
    ),
    "phase_qnode_computational_basis_fisher_support_report": (
        "scpn_quantum_control.phase.qnode_circuit",
        "phase_qnode_computational_basis_fisher_support_report",
    ),
    "phase_qnode_density_support_report": (
        "scpn_quantum_control.phase.qnode_circuit",
        "phase_qnode_density_support_report",
    ),
    "phase_qnode_depth_profile": (
        "scpn_quantum_control.phase.qnode_circuit",
        "phase_qnode_depth_profile",
    ),
    "phase_qnode_gradient_support_report": (
        "scpn_quantum_control.phase.qnode_circuit",
        "phase_qnode_gradient_support_report",
    ),
    "phase_qnode_metric_support_report": (
        "scpn_quantum_control.phase.qnode_circuit",
        "phase_qnode_metric_support_report",
    ),
    "phase_qnode_natural_gradient_metric": (
        "scpn_quantum_control.phase.qnode_circuit",
        "phase_qnode_natural_gradient_metric",
    ),
    "phase_qnode_quantum_fisher_information": (
        "scpn_quantum_control.phase.qnode_circuit",
        "phase_qnode_quantum_fisher_information",
    ),
    "phase_qnode_support_report": (
        "scpn_quantum_control.phase.qnode_circuit",
        "phase_qnode_support_report",
    ),
    "plan_phase_qnode_parameter_shift_evaluations": (
        "scpn_quantum_control.phase.qnode_circuit",
        "plan_phase_qnode_parameter_shift_evaluations",
    ),
    "registered_phase_qnode_decompositions": (
        "scpn_quantum_control.phase.qnode_circuit",
        "registered_phase_qnode_decompositions",
    ),
    "registered_phase_qnode_gates": (
        "scpn_quantum_control.phase.qnode_circuit",
        "registered_phase_qnode_gates",
    ),
    "registered_phase_qnode_noise_channels": (
        "scpn_quantum_control.phase.qnode_circuit",
        "registered_phase_qnode_noise_channels",
    ),
    "registered_phase_qnode_observables": (
        "scpn_quantum_control.phase.qnode_circuit",
        "registered_phase_qnode_observables",
    ),
    "registered_phase_qnode_templates": (
        "scpn_quantum_control.phase.qnode_circuit",
        "registered_phase_qnode_templates",
    ),
    "ParityScenario": ("scpn_quantum_control.phase.qnode_framework_parity", "ParityScenario"),
    "PhaseQNodeFrameworkParityRecord": (
        "scpn_quantum_control.phase.qnode_framework_parity",
        "PhaseQNodeFrameworkParityRecord",
    ),
    "PhaseQNodeFrameworkParitySuiteResult": (
        "scpn_quantum_control.phase.qnode_framework_parity",
        "PhaseQNodeFrameworkParitySuiteResult",
    ),
    "run_phase_qnode_framework_parity_suite": (
        "scpn_quantum_control.phase.qnode_framework_parity",
        "run_phase_qnode_framework_parity_suite",
    ),
    "ProviderQNodeTransformReadinessSuiteResult": (
        "scpn_quantum_control.phase.qnode_provider_transforms",
        "ProviderQNodeTransformReadinessSuiteResult",
    ),
    "ProviderQNodeTransformResult": (
        "scpn_quantum_control.phase.qnode_provider_transforms",
        "ProviderQNodeTransformResult",
    ),
    "execute_provider_qnode_transform": (
        "scpn_quantum_control.phase.qnode_provider_transforms",
        "execute_provider_qnode_transform",
    ),
    "execute_provider_qnode_vmap_grad": (
        "scpn_quantum_control.phase.qnode_provider_transforms",
        "execute_provider_qnode_vmap_grad",
    ),
    "run_provider_qnode_transform_readiness_suite": (
        "scpn_quantum_control.phase.qnode_provider_transforms",
        "run_provider_qnode_transform_readiness_suite",
    ),
    "PhaseQNodeTape": ("scpn_quantum_control.phase.qnode_tape", "PhaseQNodeTape"),
    "PhaseQNodeTapeReadinessSuiteResult": (
        "scpn_quantum_control.phase.qnode_tape",
        "PhaseQNodeTapeReadinessSuiteResult",
    ),
    "PhaseQNodeTapeRecord": ("scpn_quantum_control.phase.qnode_tape", "PhaseQNodeTapeRecord"),
    "phase_qnode_tape": ("scpn_quantum_control.phase.qnode_tape", "phase_qnode_tape"),
    "run_phase_qnode_tape_readiness_suite": (
        "scpn_quantum_control.phase.qnode_tape",
        "run_phase_qnode_tape_readiness_suite",
    ),
    "PhaseQNodeComplexDerivativeContract": (
        "scpn_quantum_control.phase.qnode_transforms",
        "PhaseQNodeComplexDerivativeContract",
    ),
    "PhaseQNodeTransformReadinessSuiteResult": (
        "scpn_quantum_control.phase.qnode_transforms",
        "PhaseQNodeTransformReadinessSuiteResult",
    ),
    "PhaseQNodeTransformResult": (
        "scpn_quantum_control.phase.qnode_transforms",
        "PhaseQNodeTransformResult",
    ),
    "execute_phase_qnode_hessian_vector_product": (
        "scpn_quantum_control.phase.qnode_transforms",
        "execute_phase_qnode_hessian_vector_product",
    ),
    "execute_phase_qnode_transform": (
        "scpn_quantum_control.phase.qnode_transforms",
        "execute_phase_qnode_transform",
    ),
    "phase_qnode_complex_derivative_contract": (
        "scpn_quantum_control.phase.qnode_transforms",
        "phase_qnode_complex_derivative_contract",
    ),
    "run_phase_qnode_transform_readiness_suite": (
        "scpn_quantum_control.phase.qnode_transforms",
        "run_phase_qnode_transform_readiness_suite",
    ),
    "PhaseQNodeVectorTransformReadinessSuiteResult": (
        "scpn_quantum_control.phase.qnode_vector_transforms",
        "PhaseQNodeVectorTransformReadinessSuiteResult",
    ),
    "PhaseQNodeVectorTransformResult": (
        "scpn_quantum_control.phase.qnode_vector_transforms",
        "PhaseQNodeVectorTransformResult",
    ),
    "execute_phase_qnode_vector_hessian": (
        "scpn_quantum_control.phase.qnode_vector_transforms",
        "execute_phase_qnode_vector_hessian",
    ),
    "execute_phase_qnode_vector_jacobian": (
        "scpn_quantum_control.phase.qnode_vector_transforms",
        "execute_phase_qnode_vector_jacobian",
    ),
    "execute_phase_qnode_vector_jvp": (
        "scpn_quantum_control.phase.qnode_vector_transforms",
        "execute_phase_qnode_vector_jvp",
    ),
    "execute_phase_qnode_vector_vjp": (
        "scpn_quantum_control.phase.qnode_vector_transforms",
        "execute_phase_qnode_vector_vjp",
    ),
    "execute_phase_qnode_vmap_grad": (
        "scpn_quantum_control.phase.qnode_vector_transforms",
        "execute_phase_qnode_vmap_grad",
    ),
    "run_phase_qnode_vector_transform_readiness_suite": (
        "scpn_quantum_control.phase.qnode_vector_transforms",
        "run_phase_qnode_vector_transform_readiness_suite",
    ),
    "QSPPhaseFactors": ("scpn_quantum_control.phase.qsp_phases", "QSPPhaseFactors"),
    "QSPSynthesisError": ("scpn_quantum_control.phase.qsp_phases", "QSPSynthesisError"),
    "complementary_polynomial": (
        "scpn_quantum_control.phase.qsp_phases",
        "complementary_polynomial",
    ),
    "jacobi_anger_cosine_coefficients": (
        "scpn_quantum_control.phase.qsp_phases",
        "jacobi_anger_cosine_coefficients",
    ),
    "jacobi_anger_sine_coefficients": (
        "scpn_quantum_control.phase.qsp_phases",
        "jacobi_anger_sine_coefficients",
    ),
    "qsp_response": ("scpn_quantum_control.phase.qsp_phases", "qsp_response"),
    "qsp_unitary": ("scpn_quantum_control.phase.qsp_phases", "qsp_unitary"),
    "synthesise_qsp_phases": ("scpn_quantum_control.phase.qsp_phases", "synthesise_qsp_phases"),
    "QSVTResourceEstimate": ("scpn_quantum_control.phase.qsvt_evolution", "QSVTResourceEstimate"),
    "TrajectoryResult": ("scpn_quantum_control.phase.results", "TrajectoryResult"),
    "build_structured_ansatz": (
        "scpn_quantum_control.phase.structured_ansatz",
        "build_structured_ansatz",
    ),
    "SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.synchronisation_objectives",
        "SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY",
    ),
    "build_synchronisation_objective": (
        "scpn_quantum_control.phase.synchronisation_objectives",
        "build_synchronisation_objective",
    ),
    "cluster_synchronisation_target_term": (
        "scpn_quantum_control.phase.synchronisation_objectives",
        "cluster_synchronisation_target_term",
    ),
    "kuramoto_order_parameter": (
        "scpn_quantum_control.phase.synchronisation_objectives",
        "kuramoto_order_parameter",
    ),
    "kuramoto_order_parameter_gradient": (
        "scpn_quantum_control.phase.synchronisation_objectives",
        "kuramoto_order_parameter_gradient",
    ),
    "kuramoto_order_parameter_target_term": (
        "scpn_quantum_control.phase.synchronisation_objectives",
        "kuramoto_order_parameter_target_term",
    ),
    "phase_locking_target_term": (
        "scpn_quantum_control.phase.synchronisation_objectives",
        "phase_locking_target_term",
    ),
    "SYNC_WITNESS_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "SYNC_WITNESS_CLAIM_BOUNDARY",
    ),
    "SYNC_WITNESS_EVIDENCE_CLASS": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "SYNC_WITNESS_EVIDENCE_CLASS",
    ),
    "PhaseCloudRegime": ("scpn_quantum_control.phase.synchronisation_witness", "PhaseCloudRegime"),
    "SyncWitnessBoundaryRow": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "SyncWitnessBoundaryRow",
    ),
    "SyncWitnessCase": ("scpn_quantum_control.phase.synchronisation_witness", "SyncWitnessCase"),
    "SyncWitnessRecord": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "SyncWitnessRecord",
    ),
    "SyncWitnessSuiteResult": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "SyncWitnessSuiteResult",
    ),
    "betti_curve": ("scpn_quantum_control.phase.synchronisation_witness", "betti_curve"),
    "default_sync_witness_cases": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "default_sync_witness_cases",
    ),
    "geodesic_phase_distance_matrix": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "geodesic_phase_distance_matrix",
    ),
    "harmonic_order_parameter": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "harmonic_order_parameter",
    ),
    "phase_cloud_synchronisation_witness": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "phase_cloud_synchronisation_witness",
    ),
    "run_sync_witness_suite": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "run_sync_witness_suite",
    ),
    "sync_witness_boundary_rows": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "sync_witness_boundary_rows",
    ),
    "vietoris_rips_persistence": (
        "scpn_quantum_control.phase.synchronisation_witness",
        "vietoris_rips_persistence",
    ),
    "PhaseTensorFlowFunctionCompatibilityResult": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "PhaseTensorFlowFunctionCompatibilityResult",
    ),
    "PhaseTensorFlowGradientTapeCompatibilityResult": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "PhaseTensorFlowGradientTapeCompatibilityResult",
    ),
    "PhaseTensorFlowKerasLayerWrapperAuditResult": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "PhaseTensorFlowKerasLayerWrapperAuditResult",
    ),
    "PhaseTensorFlowMaturityAuditResult": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "PhaseTensorFlowMaturityAuditResult",
    ),
    "PhaseTensorFlowParameterShiftResult": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "PhaseTensorFlowParameterShiftResult",
    ),
    "PhaseTensorFlowPhaseQNodeLoweringMatrixResult": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "PhaseTensorFlowPhaseQNodeLoweringMatrixResult",
    ),
    "PhaseTensorFlowPhaseQNodeLoweringRoute": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "PhaseTensorFlowPhaseQNodeLoweringRoute",
    ),
    "PhaseTensorFlowQNNGradientResult": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "PhaseTensorFlowQNNGradientResult",
    ),
    "PhaseTensorFlowXLACompatibilityResult": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "PhaseTensorFlowXLACompatibilityResult",
    ),
    "is_phase_tensorflow_available": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "is_phase_tensorflow_available",
    ),
    "run_tensorflow_function_compatibility_audit": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "run_tensorflow_function_compatibility_audit",
    ),
    "run_tensorflow_gradient_tape_compatibility_audit": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "run_tensorflow_gradient_tape_compatibility_audit",
    ),
    "run_tensorflow_keras_layer_wrapper_audit": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "run_tensorflow_keras_layer_wrapper_audit",
    ),
    "run_tensorflow_maturity_audit": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "run_tensorflow_maturity_audit",
    ),
    "run_tensorflow_phase_qnode_lowering_matrix": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "run_tensorflow_phase_qnode_lowering_matrix",
    ),
    "run_tensorflow_xla_compatibility_audit": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "run_tensorflow_xla_compatibility_audit",
    ),
    "tensorflow_bounded_qnn_keras_layer": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "tensorflow_bounded_qnn_keras_layer",
    ),
    "tensorflow_bounded_qnn_value_and_grad": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "tensorflow_bounded_qnn_value_and_grad",
    ),
    "tensorflow_parameter_shift_value_and_grad": (
        "scpn_quantum_control.phase.tensorflow_bridge",
        "tensorflow_parameter_shift_value_and_grad",
    ),
    "TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.tensorflow_maintenance",
        "TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY",
    ),
    "PhaseTensorFlowMaintenanceReport": (
        "scpn_quantum_control.phase.tensorflow_maintenance",
        "PhaseTensorFlowMaintenanceReport",
    ),
    "PhaseTensorFlowMaintenanceRoute": (
        "scpn_quantum_control.phase.tensorflow_maintenance",
        "PhaseTensorFlowMaintenanceRoute",
    ),
    "TensorFlowMaintenanceDecision": (
        "scpn_quantum_control.phase.tensorflow_maintenance",
        "TensorFlowMaintenanceDecision",
    ),
    "TensorFlowMaintenanceStrategy": (
        "scpn_quantum_control.phase.tensorflow_maintenance",
        "TensorFlowMaintenanceStrategy",
    ),
    "run_tensorflow_maintenance_decision": (
        "scpn_quantum_control.phase.tensorflow_maintenance",
        "run_tensorflow_maintenance_decision",
    ),
    "TORCH_AOT_AUTOGRAD_EXPORT_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.torch_aot_autograd_export",
        "TORCH_AOT_AUTOGRAD_EXPORT_CLAIM_BOUNDARY",
    ),
    "TORCH_AOT_AUTOGRAD_EXPORT_SCHEMA": (
        "scpn_quantum_control.phase.torch_aot_autograd_export",
        "TORCH_AOT_AUTOGRAD_EXPORT_SCHEMA",
    ),
    "PhaseTorchAOTAutogradExportResult": (
        "scpn_quantum_control.phase.torch_aot_autograd_export",
        "PhaseTorchAOTAutogradExportResult",
    ),
    "PhaseTorchAOTAutogradExportRoute": (
        "scpn_quantum_control.phase.torch_aot_autograd_export",
        "PhaseTorchAOTAutogradExportRoute",
    ),
    "PhaseTorchAOTAutogradGraphRecord": (
        "scpn_quantum_control.phase.torch_aot_autograd_export",
        "PhaseTorchAOTAutogradGraphRecord",
    ),
    "run_torch_aot_autograd_export_audit": (
        "scpn_quantum_control.phase.torch_aot_autograd_export",
        "run_torch_aot_autograd_export_audit",
    ),
    "TORCH_AUTOGRAD_FUNCTION_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.torch_autograd_function",
        "TORCH_AUTOGRAD_FUNCTION_CLAIM_BOUNDARY",
    ),
    "TORCH_AUTOGRAD_FUNCTION_SCHEMA": (
        "scpn_quantum_control.phase.torch_autograd_function",
        "TORCH_AUTOGRAD_FUNCTION_SCHEMA",
    ),
    "PhaseTorchAutogradFunctionResult": (
        "scpn_quantum_control.phase.torch_autograd_function",
        "PhaseTorchAutogradFunctionResult",
    ),
    "PhaseTorchAutogradFunctionRoute": (
        "scpn_quantum_control.phase.torch_autograd_function",
        "PhaseTorchAutogradFunctionRoute",
    ),
    "run_torch_autograd_function_audit": (
        "scpn_quantum_control.phase.torch_autograd_function",
        "run_torch_autograd_function_audit",
    ),
    "torch_autograd_function_qnn_loss": (
        "scpn_quantum_control.phase.torch_autograd_function",
        "torch_autograd_function_qnn_loss",
    ),
    "is_phase_torch_available": (
        "scpn_quantum_control.phase.torch_bridge",
        "is_phase_torch_available",
    ),
    "plan_torch_cloud_validation_batch": (
        "scpn_quantum_control.phase.torch_bridge",
        "plan_torch_cloud_validation_batch",
    ),
    "run_torch_compile_compatibility_audit": (
        "scpn_quantum_control.phase.torch_bridge",
        "run_torch_compile_compatibility_audit",
    ),
    "run_torch_ecosystem_maturity_audit": (
        "scpn_quantum_control.phase.torch_bridge",
        "run_torch_ecosystem_maturity_audit",
    ),
    "run_torch_func_compatibility_audit": (
        "scpn_quantum_control.phase.torch_bridge",
        "run_torch_func_compatibility_audit",
    ),
    "run_torch_maturity_audit": (
        "scpn_quantum_control.phase.torch_bridge",
        "run_torch_maturity_audit",
    ),
    "run_torch_module_wrapper_audit": (
        "scpn_quantum_control.phase.torch_bridge",
        "run_torch_module_wrapper_audit",
    ),
    "run_torch_phase_qnode_lowering_matrix": (
        "scpn_quantum_control.phase.torch_bridge",
        "run_torch_phase_qnode_lowering_matrix",
    ),
    "run_torch_training_loop_audit": (
        "scpn_quantum_control.phase.torch_bridge",
        "run_torch_training_loop_audit",
    ),
    "torch_autograd_qnn_value_and_grad": (
        "scpn_quantum_control.phase.torch_bridge",
        "torch_autograd_qnn_value_and_grad",
    ),
    "torch_bounded_qnn_layer": (
        "scpn_quantum_control.phase.torch_bridge",
        "torch_bounded_qnn_layer",
    ),
    "torch_bounded_qnn_module": (
        "scpn_quantum_control.phase.torch_bridge",
        "torch_bounded_qnn_module",
    ),
    "torch_bounded_qnn_value_and_grad": (
        "scpn_quantum_control.phase.torch_bridge",
        "torch_bounded_qnn_value_and_grad",
    ),
    "torch_parameter_shift_value_and_grad": (
        "scpn_quantum_control.phase.torch_bridge",
        "torch_parameter_shift_value_and_grad",
    ),
    "torch_phase_qnode_compile_audit": (
        "scpn_quantum_control.phase.torch_bridge",
        "torch_phase_qnode_compile_audit",
    ),
    "torch_phase_qnode_compile_boundary_audit": (
        "scpn_quantum_control.phase.torch_bridge",
        "torch_phase_qnode_compile_boundary_audit",
    ),
    "torch_phase_qnode_transform_audit": (
        "scpn_quantum_control.phase.torch_bridge",
        "torch_phase_qnode_transform_audit",
    ),
    "torch_phase_qnode_value_and_grad": (
        "scpn_quantum_control.phase.torch_bridge",
        "torch_phase_qnode_value_and_grad",
    ),
    "PhaseTorchAutogradQNNGradientResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchAutogradQNNGradientResult",
    ),
    "PhaseTorchCloudValidationRunSpec": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchCloudValidationRunSpec",
    ),
    "PhaseTorchCompileBoundaryAuditResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchCompileBoundaryAuditResult",
    ),
    "PhaseTorchCompileBoundaryRoute": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchCompileBoundaryRoute",
    ),
    "PhaseTorchCompileCompatibilityResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchCompileCompatibilityResult",
    ),
    "PhaseTorchEcosystemMaturityAuditResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchEcosystemMaturityAuditResult",
    ),
    "PhaseTorchEcosystemMaturityRoute": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchEcosystemMaturityRoute",
    ),
    "PhaseTorchFuncCompatibilityResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchFuncCompatibilityResult",
    ),
    "PhaseTorchLiveOverlayEvidence": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchLiveOverlayEvidence",
    ),
    "PhaseTorchMaturityAuditResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchMaturityAuditResult",
    ),
    "PhaseTorchModuleWrapperAuditResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchModuleWrapperAuditResult",
    ),
    "PhaseTorchParameterShiftResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchParameterShiftResult",
    ),
    "PhaseTorchPhaseQNodeCompileResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchPhaseQNodeCompileResult",
    ),
    "PhaseTorchPhaseQNodeLoweringMatrixResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchPhaseQNodeLoweringMatrixResult",
    ),
    "PhaseTorchPhaseQNodeLoweringRoute": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchPhaseQNodeLoweringRoute",
    ),
    "PhaseTorchPhaseQNodeStatevectorResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchPhaseQNodeStatevectorResult",
    ),
    "PhaseTorchPhaseQNodeTransformResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchPhaseQNodeTransformResult",
    ),
    "PhaseTorchQNNGradientResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchQNNGradientResult",
    ),
    "PhaseTorchTrainingLoopAuditResult": (
        "scpn_quantum_control.phase.torch_bridge_contracts",
        "PhaseTorchTrainingLoopAuditResult",
    ),
    "TORCH_CHECKPOINT_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.torch_checkpoint",
        "TORCH_CHECKPOINT_CLAIM_BOUNDARY",
    ),
    "TORCH_CHECKPOINT_SCHEMA": (
        "scpn_quantum_control.phase.torch_checkpoint",
        "TORCH_CHECKPOINT_SCHEMA",
    ),
    "PhaseTorchCheckpointAuditResult": (
        "scpn_quantum_control.phase.torch_checkpoint",
        "PhaseTorchCheckpointAuditResult",
    ),
    "PhaseTorchCheckpointRoute": (
        "scpn_quantum_control.phase.torch_checkpoint",
        "PhaseTorchCheckpointRoute",
    ),
    "run_torch_module_checkpoint_audit": (
        "scpn_quantum_control.phase.torch_checkpoint",
        "run_torch_module_checkpoint_audit",
    ),
    "TORCH_CHECKPOINT_MATRIX_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.torch_checkpoint_matrix",
        "TORCH_CHECKPOINT_MATRIX_CLAIM_BOUNDARY",
    ),
    "TORCH_CHECKPOINT_MATRIX_SCHEMA": (
        "scpn_quantum_control.phase.torch_checkpoint_matrix",
        "TORCH_CHECKPOINT_MATRIX_SCHEMA",
    ),
    "PhaseTorchCheckpointMatrixResult": (
        "scpn_quantum_control.phase.torch_checkpoint_matrix",
        "PhaseTorchCheckpointMatrixResult",
    ),
    "PhaseTorchCheckpointMatrixRoute": (
        "scpn_quantum_control.phase.torch_checkpoint_matrix",
        "PhaseTorchCheckpointMatrixRoute",
    ),
    "PhaseTorchCheckpointMatrixTensorMetadata": (
        "scpn_quantum_control.phase.torch_checkpoint_matrix",
        "PhaseTorchCheckpointMatrixTensorMetadata",
    ),
    "PhaseTorchCheckpointRuntimeFingerprint": (
        "scpn_quantum_control.phase.torch_checkpoint_matrix",
        "PhaseTorchCheckpointRuntimeFingerprint",
    ),
    "run_torch_long_lived_checkpoint_matrix": (
        "scpn_quantum_control.phase.torch_checkpoint_matrix",
        "run_torch_long_lived_checkpoint_matrix",
    ),
    "TORCH_DEVICE_STATE_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.torch_device_state",
        "TORCH_DEVICE_STATE_CLAIM_BOUNDARY",
    ),
    "PhaseTorchDeviceStateAuditResult": (
        "scpn_quantum_control.phase.torch_device_state",
        "PhaseTorchDeviceStateAuditResult",
    ),
    "PhaseTorchDeviceStateRoute": (
        "scpn_quantum_control.phase.torch_device_state",
        "PhaseTorchDeviceStateRoute",
    ),
    "run_torch_module_device_state_audit": (
        "scpn_quantum_control.phase.torch_device_state",
        "run_torch_module_device_state_audit",
    ),
    "TORCH_DYNAMIC_SHAPE_EXPORT_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.torch_dynamic_shape_export",
        "TORCH_DYNAMIC_SHAPE_EXPORT_CLAIM_BOUNDARY",
    ),
    "TORCH_DYNAMIC_SHAPE_EXPORT_SCHEMA": (
        "scpn_quantum_control.phase.torch_dynamic_shape_export",
        "TORCH_DYNAMIC_SHAPE_EXPORT_SCHEMA",
    ),
    "PhaseTorchDynamicShapeExportRecord": (
        "scpn_quantum_control.phase.torch_dynamic_shape_export",
        "PhaseTorchDynamicShapeExportRecord",
    ),
    "PhaseTorchDynamicShapeExportReplayCase": (
        "scpn_quantum_control.phase.torch_dynamic_shape_export",
        "PhaseTorchDynamicShapeExportReplayCase",
    ),
    "PhaseTorchDynamicShapeExportResult": (
        "scpn_quantum_control.phase.torch_dynamic_shape_export",
        "PhaseTorchDynamicShapeExportResult",
    ),
    "PhaseTorchDynamicShapeExportRoute": (
        "scpn_quantum_control.phase.torch_dynamic_shape_export",
        "PhaseTorchDynamicShapeExportRoute",
    ),
    "default_torch_dynamic_shape_export_replay_cases": (
        "scpn_quantum_control.phase.torch_dynamic_shape_export",
        "default_torch_dynamic_shape_export_replay_cases",
    ),
    "run_torch_dynamic_shape_export_audit": (
        "scpn_quantum_control.phase.torch_dynamic_shape_export",
        "run_torch_dynamic_shape_export_audit",
    ),
    "TORCH_EXPORT_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.torch_export",
        "TORCH_EXPORT_CLAIM_BOUNDARY",
    ),
    "PhaseTorchExportAuditResult": (
        "scpn_quantum_control.phase.torch_export",
        "PhaseTorchExportAuditResult",
    ),
    "PhaseTorchExportRoute": ("scpn_quantum_control.phase.torch_export", "PhaseTorchExportRoute"),
    "run_torch_module_export_audit": (
        "scpn_quantum_control.phase.torch_export",
        "run_torch_module_export_audit",
    ),
    "TORCH_EXPORT_SHAPE_MATRIX_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.torch_export_shape_matrix",
        "TORCH_EXPORT_SHAPE_MATRIX_CLAIM_BOUNDARY",
    ),
    "TORCH_EXPORT_SHAPE_MATRIX_SCHEMA": (
        "scpn_quantum_control.phase.torch_export_shape_matrix",
        "TORCH_EXPORT_SHAPE_MATRIX_SCHEMA",
    ),
    "PhaseTorchExportShapeMatrixRecord": (
        "scpn_quantum_control.phase.torch_export_shape_matrix",
        "PhaseTorchExportShapeMatrixRecord",
    ),
    "PhaseTorchExportShapeMatrixResult": (
        "scpn_quantum_control.phase.torch_export_shape_matrix",
        "PhaseTorchExportShapeMatrixResult",
    ),
    "PhaseTorchExportShapeMatrixRoute": (
        "scpn_quantum_control.phase.torch_export_shape_matrix",
        "PhaseTorchExportShapeMatrixRoute",
    ),
    "PhaseTorchExportShapeScenario": (
        "scpn_quantum_control.phase.torch_export_shape_matrix",
        "PhaseTorchExportShapeScenario",
    ),
    "default_torch_export_shape_scenarios": (
        "scpn_quantum_control.phase.torch_export_shape_matrix",
        "default_torch_export_shape_scenarios",
    ),
    "run_torch_export_shape_matrix": (
        "scpn_quantum_control.phase.torch_export_shape_matrix",
        "run_torch_export_shape_matrix",
    ),
    "TORCH_MODULE_STATE_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.torch_module_state",
        "TORCH_MODULE_STATE_CLAIM_BOUNDARY",
    ),
    "PhaseTorchModuleStateAuditResult": (
        "scpn_quantum_control.phase.torch_module_state",
        "PhaseTorchModuleStateAuditResult",
    ),
    "PhaseTorchModuleStateRoute": (
        "scpn_quantum_control.phase.torch_module_state",
        "PhaseTorchModuleStateRoute",
    ),
    "PhaseTorchModuleStateTensorMismatch": (
        "scpn_quantum_control.phase.torch_module_state",
        "PhaseTorchModuleStateTensorMismatch",
    ),
    "PhaseTorchModuleStateValidationResult": (
        "scpn_quantum_control.phase.torch_module_state",
        "PhaseTorchModuleStateValidationResult",
    ),
    "run_torch_module_state_audit": (
        "scpn_quantum_control.phase.torch_module_state",
        "run_torch_module_state_audit",
    ),
    "validate_torch_bounded_qnn_state_dict": (
        "scpn_quantum_control.phase.torch_module_state",
        "validate_torch_bounded_qnn_state_dict",
    ),
    "TORCH_TRAINING_LOOP_MATRIX_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.torch_training_loop_matrix",
        "TORCH_TRAINING_LOOP_MATRIX_CLAIM_BOUNDARY",
    ),
    "TORCH_TRAINING_LOOP_MATRIX_SCHEMA": (
        "scpn_quantum_control.phase.torch_training_loop_matrix",
        "TORCH_TRAINING_LOOP_MATRIX_SCHEMA",
    ),
    "PhaseTorchTrainingLoopMatrixRecord": (
        "scpn_quantum_control.phase.torch_training_loop_matrix",
        "PhaseTorchTrainingLoopMatrixRecord",
    ),
    "PhaseTorchTrainingLoopMatrixResult": (
        "scpn_quantum_control.phase.torch_training_loop_matrix",
        "PhaseTorchTrainingLoopMatrixResult",
    ),
    "PhaseTorchTrainingLoopMatrixRoute": (
        "scpn_quantum_control.phase.torch_training_loop_matrix",
        "PhaseTorchTrainingLoopMatrixRoute",
    ),
    "PhaseTorchTrainingLoopScenario": (
        "scpn_quantum_control.phase.torch_training_loop_matrix",
        "PhaseTorchTrainingLoopScenario",
    ),
    "default_torch_training_loop_scenarios": (
        "scpn_quantum_control.phase.torch_training_loop_matrix",
        "default_torch_training_loop_scenarios",
    ),
    "run_torch_training_loop_matrix": (
        "scpn_quantum_control.phase.torch_training_loop_matrix",
        "run_torch_training_loop_matrix",
    ),
    "TRAINABILITY_CLAIM_BOUNDARY": (
        "scpn_quantum_control.phase.trainability",
        "TRAINABILITY_CLAIM_BOUNDARY",
    ),
    "AdaptiveShotAllocationDryRun": (
        "scpn_quantum_control.phase.trainability",
        "AdaptiveShotAllocationDryRun",
    ),
    "BarrenPlateauTrainabilityReport": (
        "scpn_quantum_control.phase.trainability",
        "BarrenPlateauTrainabilityReport",
    ),
    "TrainabilityGradientSample": (
        "scpn_quantum_control.phase.trainability",
        "TrainabilityGradientSample",
    ),
    "TrainabilityStatus": ("scpn_quantum_control.phase.trainability", "TrainabilityStatus"),
    "run_barren_plateau_trainability_report": (
        "scpn_quantum_control.phase.trainability",
        "run_barren_plateau_trainability_report",
    ),
    "GradientTransformNestingAuditResult": (
        "scpn_quantum_control.phase.transform_nesting",
        "GradientTransformNestingAuditResult",
    ),
    "GradientTransformNestingPlan": (
        "scpn_quantum_control.phase.transform_nesting",
        "GradientTransformNestingPlan",
    ),
    "assert_gradient_transform_nesting_supported": (
        "scpn_quantum_control.phase.transform_nesting",
        "assert_gradient_transform_nesting_supported",
    ),
    "plan_gradient_transform_nesting": (
        "scpn_quantum_control.phase.transform_nesting",
        "plan_gradient_transform_nesting",
    ),
    "run_gradient_transform_nesting_audit": (
        "scpn_quantum_control.phase.transform_nesting",
        "run_gradient_transform_nesting_audit",
    ),
    "trotter_error_norm": ("scpn_quantum_control.phase.trotter_error", "trotter_error_norm"),
    "trotter_error_sweep": ("scpn_quantum_control.phase.trotter_error", "trotter_error_sweep"),
    "QuantumUPDESolver": ("scpn_quantum_control.phase.trotter_upde", "QuantumUPDESolver"),
    "UPDEStepResult": ("scpn_quantum_control.phase.trotter_upde", "UPDEStepResult"),
    "UPDETrajectoryResult": ("scpn_quantum_control.phase.trotter_upde", "UPDETrajectoryResult"),
    "analytic_state_derivatives": (
        "scpn_quantum_control.phase.variational_metric",
        "analytic_state_derivatives",
    ),
    "assert_single_parameter_rotations": (
        "scpn_quantum_control.phase.variational_metric",
        "assert_single_parameter_rotations",
    ),
    "imaginary_time_force": (
        "scpn_quantum_control.phase.variational_metric",
        "imaginary_time_force",
    ),
    "mclachlan_metric": ("scpn_quantum_control.phase.variational_metric", "mclachlan_metric"),
    "real_time_force": ("scpn_quantum_control.phase.variational_metric", "real_time_force"),
    "VarQITEResult": ("scpn_quantum_control.phase.varqite", "VarQITEResult"),
    "varqite_ground_state": ("scpn_quantum_control.phase.varqite", "varqite_ground_state"),
    "QuantumKuramotoSolver": ("scpn_quantum_control.phase.xy_kuramoto", "QuantumKuramotoSolver"),
    "TrotterEvolutionConfig": ("scpn_quantum_control.phase.xy_kuramoto", "TrotterEvolutionConfig"),
}


class _ExportModule(ModuleType):
    """Keep declared object exports when Python publishes child modules."""

    def __setattr__(self, name: str, value: object) -> None:
        """Publish a module attribute while preserving same-named exports.

        Parameters
        ----------
        name
            Attribute assigned by the import system or a caller.
        value
            Value to publish in the package namespace.

        """
        target = _PUBLIC_EXPORTS.get(name)
        if (
            target is not None
            and target[1] is not None
            and isinstance(value, ModuleType)
            and value.__name__ in {target[0], f"{__name__}.{name}"}
        ):
            value = __getattr__(name)
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _ExportModule


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


__all__ = [
    "QuantumKuramotoSolver",
    "TrotterEvolutionConfig",
    "TrajectoryResult",
    "QuantumUPDESolver",
    "UPDEStepResult",
    "UPDETrajectoryResult",
    "PhaseVQE",
    "PhaseVQEResult",
    "trotter_error_norm",
    "trotter_error_sweep",
    "benchmark_ansatz",
    "run_ansatz_benchmark",
    "AnsatzBenchmarkRow",
    "adapt_vqe",
    "ADAPTResult",
    "adiabatic_ramp",
    "AdiabaticResult",
    "avqds_simulate",
    "AVQDSResult",
    "learn_couplings_from_observations",
    "verify_coupling_parameter_shift_gradient",
    "coupling_matrix_from_edge_vector",
    "CouplingGradientVerificationResult",
    "CouplingLearningResult",
    "COUPLING_RECOVERY_CLAIM_BOUNDARY",
    "COUPLING_RECOVERY_EVIDENCE_CLASS",
    "CouplingRecoveryBoundaryRow",
    "CouplingRecoveryCase",
    "CouplingRecoveryRecord",
    "CouplingRecoverySuiteResult",
    "coupling_recovery_boundary_rows",
    "default_coupling_recovery_cases",
    "inject_time_series_noise_and_missing",
    "recover_kuramoto_couplings_from_time_series",
    "recover_xy_couplings_from_pair_energy_series",
    "run_coupling_recovery_suite",
    "simulate_kuramoto_phase_time_series",
    "simulate_xy_pair_energy_time_series",
    "SYNC_WITNESS_CLAIM_BOUNDARY",
    "SYNC_WITNESS_EVIDENCE_CLASS",
    "PhaseCloudRegime",
    "SyncWitnessBoundaryRow",
    "SyncWitnessCase",
    "SyncWitnessRecord",
    "SyncWitnessSuiteResult",
    "betti_curve",
    "default_sync_witness_cases",
    "geodesic_phase_distance_matrix",
    "harmonic_order_parameter",
    "phase_cloud_synchronisation_witness",
    "run_sync_witness_suite",
    "sync_witness_boundary_rows",
    "vietoris_rips_persistence",
    "run_known_phase_gradient_audit",
    "run_differentiable_workflow_audit_suite",
    "run_finite_shot_gradient_uncertainty_audit",
    "run_parameter_shift_audit_suite",
    "run_phase_gradient_benchmark_suite",
    "run_ml_framework_gradient_audit",
    "verify_parameter_shift_analytic_gradient",
    "DifferentiableReadinessSurface",
    "DifferentiableReadinessAuditRecord",
    "DifferentiableReadinessAuditResult",
    "default_differentiable_readiness_surfaces",
    "run_differentiable_readiness_audit",
    "DifferentiableDomainBenchmarkDatasetSuite",
    "DifferentiableDomainBenchmarkValidationResult",
    "DifferentiableDomainBenchmarkValidationSuite",
    "DifferentiableKuramotoExactAnswerCase",
    "DifferentiablePublishedDomainBenchmarkCase",
    "DifferentiablePublishedDomainBenchmarkSuite",
    "DifferentiablePublishedDomainBenchmarkValidationResult",
    "DifferentiablePublishedDomainBenchmarkValidationSuite",
    "DifferentiableQNNExactAnswerCase",
    "load_differentiable_domain_benchmark_datasets",
    "load_differentiable_published_domain_benchmark_cases",
    "run_differentiable_domain_benchmark_dataset_validation",
    "run_differentiable_published_domain_benchmark_validation",
    "DifferentiableQuantumAuditReport",
    "DifferentiableWorkflowAuditSuiteResult",
    "FiniteShotGradientAuditResult",
    "MLFrameworkGradientAuditRecord",
    "MLFrameworkGradientAuditSuiteResult",
    "ParameterShiftAnalyticAgreement",
    "PhaseGradientBenchmarkSuiteResult",
    "varqite_ground_state",
    "VarQITEResult",
    "analytic_state_derivatives",
    "assert_single_parameter_rotations",
    "mclachlan_metric",
    "real_time_force",
    "imaginary_time_force",
    "floquet_evolve",
    "scan_drive_amplitude",
    "FloquetResult",
    "DTC_SUBHARMONIC_THRESHOLD",
    "quantum_gradient_backend_capability",
    "HardwareGradientPolicy",
    "HardwareGradientPolicyDecision",
    "HardwareGradientReadinessSuiteResult",
    "HardwareGradientRequest",
    "assert_hardware_gradient_policy_approved",
    "evaluate_hardware_gradient_policy",
    "run_hardware_gradient_policy_readiness_suite",
    "HardwareGradientCampaignPlan",
    "HardwareGradientCampaignSpec",
    "HardwareGradientCampaignSuite",
    "HardwareGradientReplaySchema",
    "default_hardware_gradient_campaign_specs",
    "plan_hardware_gradient_campaign",
    "run_hardware_gradient_campaign_readiness_suite",
    "HardwareGradientArtifactMapEntry",
    "HardwareGradientBenchmarkPlaceholder",
    "HardwareGradientClaimLedgerRow",
    "HardwareGradientMethodSection",
    "HardwareGradientPreregistration",
    "HardwareGradientPublicationPackage",
    "build_hardware_gradient_publication_package",
    "explain_quantum_gradient_method",
    "plan_quantum_gradient_backend",
    "QuantumGradientBackendCapability",
    "QuantumGradientMethodExplanation",
    "QuantumGradientPlan",
    "QuantumGradientRejectedMethod",
    "QuantumGradientShotPolicy",
    "GradientSupportCapability",
    "GradientSupportPlan",
    "GradientSupportMatrixAuditResult",
    "gradient_support_capability",
    "list_gradient_support_capabilities",
    "plan_gradient_support",
    "assert_gradient_support",
    "run_gradient_support_matrix_audit",
    "parameter_shift_gradient_descent",
    "validate_parameter_shift_training",
    "ParameterShiftTrainingCertificate",
    "ParameterShiftTrainingResult",
    "ParameterShiftTrainingStep",
    "parameter_shift_natural_gradient_descent",
    "validate_natural_gradient_training",
    "solve_natural_gradient_direction",
    "NaturalGradientDirection",
    "NaturalGradientRegularizationPolicy",
    "ParameterShiftNaturalGradientCertificate",
    "ParameterShiftNaturalGradientResult",
    "ParameterShiftNaturalGradientStep",
    "TRAINABILITY_CLAIM_BOUNDARY",
    "AdaptiveShotAllocationDryRun",
    "BarrenPlateauTrainabilityReport",
    "TrainabilityGradientSample",
    "TrainabilityStatus",
    "run_barren_plateau_trainability_report",
    "run_parameter_shift_optimizer_comparison",
    "OptimizerComparisonSuiteResult",
    "OptimizerConvergenceRecord",
    "GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY",
    "GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS",
    "GroundStateConvergenceCertificate",
    "GroundStateOptimizerBoundaryRow",
    "GroundStateOptimizerConvergenceSuiteResult",
    "GroundStateOptimizerRunRecord",
    "KnownGroundStateObjective",
    "default_ground_state_optimizer_objectives",
    "run_ground_state_optimizer_convergence_suite",
    "build_phase_control_objective",
    "train_composed_phase_objective",
    "validate_composed_objective_training",
    "phase_energy_term",
    "phase_fidelity_target_term",
    "periodic_regularization_term",
    "phase_symmetry_penalty_term",
    "smooth_box_safety_penalty_term",
    "ComposedPhaseObjective",
    "ObjectiveTerm",
    "ObjectiveTermValue",
    "ObjectiveGradientEvaluation",
    "ComposedObjectiveTrainingStep",
    "ComposedObjectiveTrainingResult",
    "ComposedObjectiveTrainingCertificate",
    "verify_composed_objective_gradient",
    "run_composed_objective_audit_suite",
    "ComposedObjectiveGradientAgreement",
    "ComposedObjectiveAuditSuiteResult",
    "plan_composed_objective_execution",
    "assert_composed_objective_execution_supported",
    "run_composed_objective_planner_audit",
    "ComposedObjectiveExecutionPlan",
    "ComposedObjectivePlannerAuditResult",
    "gradient_tape",
    "GRADIENT_TAPE_CONTRACT_CLAIM_BOUNDARY",
    "GradientTapeContractAuditResult",
    "GradientTapeContractCheck",
    "QuantumGradientTape",
    "TapeContractStatus",
    "TapeGradientRecord",
    "run_gradient_tape_contract_audit",
    "phase_qnode_tape",
    "PhaseQNodeTape",
    "PhaseQNodeTapeReadinessSuiteResult",
    "PhaseQNodeTapeRecord",
    "run_phase_qnode_tape_readiness_suite",
    "DenseHermitianObservable",
    "PauliTerm",
    "PauliCovarianceObservable",
    "PhaseQNodeClassicalFisherResult",
    "PhaseQNodeCircuit",
    "PhaseQNodeDepthProfile",
    "PhaseQNodeDensityCircuit",
    "PhaseQNodeDensityExecutionResult",
    "PhaseQNodeExecutionResult",
    "PhaseQNodeGradientEvaluationGroup",
    "PhaseQNodeGradientEvaluationPlan",
    "PhaseQNodeGradientResult",
    "PhaseQNodeMetricTensorResult",
    "PhaseQNodeNoiseChannel",
    "PhaseQNodeOperation",
    "PhaseQNodeRegisteredCircuitSpec",
    "PhaseQNodeSupportError",
    "PhaseQNodeSupportReport",
    "PhaseQNodeTemplateSpec",
    "SparsePauliHamiltonian",
    "build_registered_phase_qnode_circuit",
    "build_phase_qnode_template",
    "build_sparse_ising_chain_hamiltonian",
    "decompose_phase_qnode_controlled_gate",
    "build_u3_operations",
    "su2_zyz_angles",
    "KnmGraph",
    "QGNNConfig",
    "QGNNTrainingResult",
    "synthetic_kuramoto_target",
    "execute_phase_qnode_circuit",
    "execute_phase_qnode_density_matrix",
    "parameter_shift_phase_qnode_gradient",
    "plan_phase_qnode_parameter_shift_evaluations",
    "phase_qnode_computational_basis_fisher_information",
    "phase_qnode_computational_basis_fisher_support_report",
    "phase_qnode_density_support_report",
    "phase_qnode_depth_profile",
    "phase_qnode_gradient_support_report",
    "phase_qnode_metric_support_report",
    "phase_qnode_natural_gradient_metric",
    "phase_qnode_quantum_fisher_information",
    "phase_qnode_support_report",
    "registered_phase_qnode_gates",
    "registered_phase_qnode_observables",
    "registered_phase_qnode_decompositions",
    "registered_phase_qnode_noise_channels",
    "registered_phase_qnode_templates",
    "ParityScenario",
    "PhaseQNodeFrameworkParityRecord",
    "PhaseQNodeFrameworkParitySuiteResult",
    "run_phase_qnode_framework_parity_suite",
    "PhaseQNodeAffinityBenchmarkMetadata",
    "PhaseQNodeAffinityBenchmarkResult",
    "PhaseQNodeAffinityArtifactValidation",
    "classify_affinity_evidence",
    "run_phase_qnode_affinity_benchmark",
    "validate_phase_qnode_affinity_artifact",
    "DifferentiableModelTrainingEvidenceSuite",
    "DifferentiableModelTrainingRecord",
    "RegisteredDifferentiableTrainingSuiteAuditResult",
    "RegisteredDifferentiableTrainingSuiteRecord",
    "run_differentiable_model_training_evidence_suite",
    "run_registered_differentiable_training_suite_audit",
    "ProviderQNodeTransformReadinessSuiteResult",
    "ProviderQNodeTransformResult",
    "execute_provider_qnode_transform",
    "execute_provider_qnode_vmap_grad",
    "run_provider_qnode_transform_readiness_suite",
    "PhaseQNodeComplexDerivativeContract",
    "PhaseQNodeTransformReadinessSuiteResult",
    "PhaseQNodeTransformResult",
    "execute_phase_qnode_hessian_vector_product",
    "execute_phase_qnode_transform",
    "phase_qnode_complex_derivative_contract",
    "run_phase_qnode_transform_readiness_suite",
    "PhaseQNodeVectorTransformReadinessSuiteResult",
    "PhaseQNodeVectorTransformResult",
    "execute_phase_qnode_vector_hessian",
    "execute_phase_qnode_vector_jvp",
    "execute_phase_qnode_vector_jacobian",
    "execute_phase_qnode_vector_vjp",
    "execute_phase_qnode_vmap_grad",
    "run_phase_qnode_vector_transform_readiness_suite",
    "is_phase_jax_available",
    "check_jax_parameter_shift_agreement",
    "plan_jax_cloud_validation_batch",
    "jax_custom_vjp_qnn_value_and_grad",
    "jax_native_qnn_value_and_grad",
    "jax_parameter_shift_value_and_grad",
    "jax_phase_qnode_aot_export_audit",
    "jax_phase_qnode_native_transform_audit",
    "jax_phase_qnode_pytree_transform_audit",
    "jax_phase_qnode_sharding_transform_audit",
    "jax_phase_qnode_value_and_grad",
    "run_jax_jit_compatibility_audit",
    "run_jax_maturity_audit",
    "run_jax_nested_transform_algebra_audit",
    "run_jax_phase_qnode_lowering_matrix",
    "PhaseJAXCloudValidationRunSpec",
    "PhaseJAXCustomVJPQNNGradientResult",
    "PhaseJAXGradientAgreementResult",
    "PhaseJAXJITCompatibilityResult",
    "PhaseJAXMaturityAuditResult",
    "PhaseJAXNativeQNNGradientResult",
    "PhaseJAXNestedTransformAlgebraResult",
    "PhaseJAXNestedTransformRoute",
    "PhaseJAXParameterShiftResult",
    "PhaseJAXPhaseQNodeAOTExportResult",
    "PhaseJAXPhaseQNodeLoweringMatrixResult",
    "PhaseJAXPhaseQNodeLoweringRoute",
    "PhaseJAXPhaseQNodeNativeTransformResult",
    "PhaseJAXPhaseQNodePyTreeTransformResult",
    "PhaseJAXPhaseQNodeShardingTransformResult",
    "PhaseJAXPhaseQNodeStatevectorResult",
    "PhaseJAXPyTreeCompatibilityResult",
    "PhaseJAXShardingCompatibilityResult",
    "PhaseJAXVMAPCompatibilityResult",
    "is_phase_pennylane_available",
    "run_jax_pytree_compatibility_audit",
    "run_jax_vmap_compatibility_audit",
    "run_jax_sharding_compatibility_audit",
    "build_pennylane_qnode_from_phase_qnode",
    "check_pennylane_parameter_shift_agreement",
    "check_pennylane_phase_qnode_round_trip",
    "check_pennylane_qnode_round_trip",
    "run_pennylane_maturity_audit",
    "run_pennylane_plugin_matrix",
    "PennyLaneImportResult",
    "PennyLaneImportRoundTripResult",
    "import_phase_qnode_from_pennylane",
    "check_pennylane_phase_qnode_import_round_trip",
    "is_pennylane_import_available",
    "PennyLaneGradientAgreementResult",
    "PennyLaneHardwarePluginExecutionArtifact",
    "PennyLaneMaturityAuditResult",
    "PennyLanePluginMatrixResult",
    "PennyLanePluginMatrixRoute",
    "PennyLaneProviderEvidenceBundle",
    "PennyLaneProviderGradientParityArtifact",
    "PennyLaneProviderPluginExecutionArtifact",
    "PennyLaneQNodeConversionResult",
    "PennyLaneRoundTripResult",
    "ProviderExpectationSample",
    "ProviderGradientExecutionResult",
    "ProviderHardwareGradientPreparationResult",
    "ProviderParameterShiftRecord",
    "execute_provider_parameter_shift_gradient",
    "prepare_provider_hardware_parameter_shift_gradient",
    "ProviderHardwareGradientPreparationScenario",
    "ProviderHardwareGradientPreparationRecord",
    "ProviderHardwareGradientPreparationAuditResult",
    "default_provider_hardware_gradient_preparation_scenarios",
    "run_provider_hardware_gradient_preparation_audit",
    "DifferentiableProviderHardwareEvidenceChain",
    "DifferentiableProviderHardwareSafetyAuditResult",
    "DifferentiableProviderHardwareSafetySurface",
    "run_differentiable_provider_hardware_safety_audit",
    "ProviderGradientReadinessScenario",
    "ProviderGradientReadinessRecord",
    "ProviderGradientReadinessAuditResult",
    "default_provider_gradient_readiness_scenarios",
    "run_provider_gradient_readiness_audit",
    "ExternalGradientMap",
    "ParameterShiftQNNConformanceCaseResult",
    "ParameterShiftQNNConformanceSuiteResult",
    "ParameterShiftQNNUnsupportedScenario",
    "run_parameter_shift_qnn_conformance_suite",
    "summarize_parameter_shift_qnn_unsuitable_scenarios",
    "ParameterShiftQNNConvergenceCaseResult",
    "ParameterShiftQNNConvergenceSuiteResult",
    "ParameterShiftQNNConvergenceUnsuitableScenario",
    "ParameterShiftQNNMultiSeedConvergenceCaseResult",
    "ParameterShiftQNNMultiSeedConvergenceRunResult",
    "ParameterShiftQNNMultiSeedConvergenceSuiteResult",
    "run_parameter_shift_qnn_multi_seed_convergence_suite",
    "run_parameter_shift_qnn_convergence_suite",
    "summarize_parameter_shift_qnn_convergence_unsuitable_scenarios",
    "FrameworkGradientCaseMap",
    "FrameworkGradientMap",
    "ParameterShiftQNNFrameworkAgreementResult",
    "ParameterShiftQNNFrameworkAgreementSuiteResult",
    "ParameterShiftQNNFrameworkGradientAgreement",
    "BoundedQNNFrameworkBridgeCapability",
    "BoundedQNNFrameworkBridgeMatrixResult",
    "assert_bounded_qnn_framework_bridge_supported",
    "run_bounded_qnn_framework_bridge_matrix",
    "run_parameter_shift_qnn_framework_agreement_suite",
    "verify_parameter_shift_qnn_framework_agreement",
    "ParameterShiftQNNLossLandscapeCaseResult",
    "ParameterShiftQNNLossLandscapePoint",
    "ParameterShiftQNNLossLandscapeSuiteResult",
    "run_parameter_shift_qnn_loss_landscape_suite",
    "ParameterShiftQNNFiniteShotConvergenceCaseResult",
    "ParameterShiftQNNFiniteShotConvergenceSuiteResult",
    "ParameterShiftQNNFiniteShotGradientResult",
    "ParameterShiftQNNFiniteShotProbeRecord",
    "ParameterShiftQNNFiniteShotUnsupportedScenario",
    "estimate_parameter_shift_qnn_finite_shot_gradient",
    "run_parameter_shift_qnn_finite_shot_convergence_suite",
    "summarize_parameter_shift_qnn_finite_shot_unsuitable_scenarios",
    "DerivativeFreeCandidateMap",
    "ParameterShiftQNNOptimizerBenchmarkCaseResult",
    "ParameterShiftQNNOptimizerBenchmarkSuiteResult",
    "QNNOptimizerBaselineResult",
    "run_parameter_shift_qnn_optimizer_benchmark_suite",
    "ParameterShiftQNNPredictionResult",
    "ParameterShiftQNNTrainingResult",
    "ParameterShiftQNNExternalGradientAgreement",
    "ParameterShiftQNNGradientVerificationResult",
    "parameter_shift_qnn_classifier_loss",
    "parameter_shift_qnn_classifier_gradient",
    "predict_parameter_shift_qnn_classifier",
    "train_parameter_shift_qnn_classifier",
    "verify_parameter_shift_qnn_classifier_gradient",
    "QiskitParameterShiftGradientResult",
    "QiskitCalibrationStatevectorComparisonArtifact",
    "QiskitMaturityAuditResult",
    "QiskitParameterShiftRecord",
    "QiskitProviderGradientWorkflowArtifact",
    "QiskitRawCountReplayArtifact",
    "QiskitRuntimePrimitiveExecutionArtifact",
    "QiskitRuntimeQPUExecutionArtifact",
    "QiskitRuntimeQPUProviderEvidenceBundle",
    "build_qiskit_provider_gradient_workflow_artifact",
    "build_qiskit_runtime_qpu_execution_artifact",
    "build_qiskit_runtime_qpu_provider_evidence_bundle",
    "execute_qiskit_finite_shot_parameter_shift",
    "execute_qiskit_statevector_parameter_shift",
    "generate_qiskit_parameter_shift_circuits",
    "run_qiskit_maturity_audit",
    "is_phase_torch_available",
    "plan_torch_cloud_validation_batch",
    "run_torch_compile_compatibility_audit",
    "run_torch_ecosystem_maturity_audit",
    "run_torch_func_compatibility_audit",
    "run_torch_module_wrapper_audit",
    "torch_autograd_qnn_value_and_grad",
    "torch_bounded_qnn_value_and_grad",
    "torch_bounded_qnn_layer",
    "torch_bounded_qnn_module",
    "torch_parameter_shift_value_and_grad",
    "torch_phase_qnode_compile_boundary_audit",
    "torch_phase_qnode_compile_audit",
    "torch_phase_qnode_transform_audit",
    "PhaseTorchAutogradQNNGradientResult",
    "PhaseTorchCloudValidationRunSpec",
    "PhaseTorchCheckpointAuditResult",
    "PhaseTorchCheckpointRoute",
    "PhaseTorchCompileBoundaryAuditResult",
    "PhaseTorchCompileBoundaryRoute",
    "PhaseTorchCompileCompatibilityResult",
    "PhaseTorchDeviceStateAuditResult",
    "PhaseTorchDeviceStateRoute",
    "PhaseTorchAOTAutogradExportResult",
    "PhaseTorchAOTAutogradExportRoute",
    "PhaseTorchAOTAutogradGraphRecord",
    "PhaseTorchAutogradFunctionResult",
    "PhaseTorchAutogradFunctionRoute",
    "PhaseTorchDynamicShapeExportRecord",
    "PhaseTorchDynamicShapeExportReplayCase",
    "PhaseTorchDynamicShapeExportResult",
    "PhaseTorchDynamicShapeExportRoute",
    "PhaseTorchEcosystemMaturityAuditResult",
    "PhaseTorchEcosystemMaturityRoute",
    "PhaseTorchExportAuditResult",
    "PhaseTorchExportRoute",
    "PhaseTorchExportShapeMatrixRecord",
    "PhaseTorchExportShapeMatrixResult",
    "PhaseTorchExportShapeMatrixRoute",
    "PhaseTorchExportShapeScenario",
    "PhaseTorchFuncCompatibilityResult",
    "PhaseTorchLiveOverlayEvidence",
    "PhaseTorchMaturityAuditResult",
    "PhaseTorchCheckpointMatrixResult",
    "PhaseTorchCheckpointMatrixRoute",
    "PhaseTorchCheckpointMatrixTensorMetadata",
    "PhaseTorchCheckpointRuntimeFingerprint",
    "PhaseTorchModuleWrapperAuditResult",
    "PhaseTorchModuleStateAuditResult",
    "PhaseTorchModuleStateRoute",
    "PhaseTorchModuleStateTensorMismatch",
    "PhaseTorchModuleStateValidationResult",
    "PhaseTorchParameterShiftResult",
    "PhaseTorchPhaseQNodeCompileResult",
    "PhaseTorchPhaseQNodeLoweringMatrixResult",
    "PhaseTorchPhaseQNodeLoweringRoute",
    "PhaseTorchPhaseQNodeStatevectorResult",
    "PhaseTorchPhaseQNodeTransformResult",
    "PhaseTorchQNNGradientResult",
    "PhaseTorchTrainingLoopAuditResult",
    "PhaseTorchTrainingLoopMatrixRecord",
    "PhaseTorchTrainingLoopMatrixResult",
    "PhaseTorchTrainingLoopMatrixRoute",
    "PhaseTorchTrainingLoopScenario",
    "TORCH_CHECKPOINT_CLAIM_BOUNDARY",
    "TORCH_CHECKPOINT_MATRIX_CLAIM_BOUNDARY",
    "TORCH_CHECKPOINT_MATRIX_SCHEMA",
    "TORCH_CHECKPOINT_SCHEMA",
    "TORCH_DEVICE_STATE_CLAIM_BOUNDARY",
    "TORCH_AOT_AUTOGRAD_EXPORT_CLAIM_BOUNDARY",
    "TORCH_AOT_AUTOGRAD_EXPORT_SCHEMA",
    "TORCH_AUTOGRAD_FUNCTION_CLAIM_BOUNDARY",
    "TORCH_AUTOGRAD_FUNCTION_SCHEMA",
    "TORCH_DYNAMIC_SHAPE_EXPORT_CLAIM_BOUNDARY",
    "TORCH_DYNAMIC_SHAPE_EXPORT_SCHEMA",
    "TORCH_EXPORT_CLAIM_BOUNDARY",
    "TORCH_EXPORT_SHAPE_MATRIX_CLAIM_BOUNDARY",
    "TORCH_EXPORT_SHAPE_MATRIX_SCHEMA",
    "TORCH_MODULE_STATE_CLAIM_BOUNDARY",
    "TORCH_TRAINING_LOOP_MATRIX_CLAIM_BOUNDARY",
    "TORCH_TRAINING_LOOP_MATRIX_SCHEMA",
    "default_torch_dynamic_shape_export_replay_cases",
    "default_torch_export_shape_scenarios",
    "default_torch_training_loop_scenarios",
    "run_torch_maturity_audit",
    "run_torch_aot_autograd_export_audit",
    "run_torch_autograd_function_audit",
    "run_torch_dynamic_shape_export_audit",
    "run_torch_long_lived_checkpoint_matrix",
    "run_torch_module_checkpoint_audit",
    "run_torch_module_device_state_audit",
    "run_torch_module_export_audit",
    "run_torch_export_shape_matrix",
    "run_torch_module_state_audit",
    "run_torch_phase_qnode_lowering_matrix",
    "run_torch_training_loop_audit",
    "run_torch_training_loop_matrix",
    "torch_autograd_function_qnn_loss",
    "validate_torch_bounded_qnn_state_dict",
    "torch_phase_qnode_value_and_grad",
    "GradientTransformNestingPlan",
    "GradientTransformNestingAuditResult",
    "plan_gradient_transform_nesting",
    "assert_gradient_transform_nesting_supported",
    "run_gradient_transform_nesting_audit",
    "is_phase_tensorflow_available",
    "run_tensorflow_function_compatibility_audit",
    "run_tensorflow_gradient_tape_compatibility_audit",
    "run_tensorflow_keras_layer_wrapper_audit",
    "run_tensorflow_maturity_audit",
    "run_tensorflow_phase_qnode_lowering_matrix",
    "run_tensorflow_xla_compatibility_audit",
    "tensorflow_bounded_qnn_value_and_grad",
    "tensorflow_bounded_qnn_keras_layer",
    "tensorflow_parameter_shift_value_and_grad",
    "PhaseTensorFlowFunctionCompatibilityResult",
    "PhaseTensorFlowGradientTapeCompatibilityResult",
    "PhaseTensorFlowKerasLayerWrapperAuditResult",
    "PhaseTensorFlowMaturityAuditResult",
    "PhaseTensorFlowParameterShiftResult",
    "PhaseTensorFlowPhaseQNodeLoweringMatrixResult",
    "PhaseTensorFlowPhaseQNodeLoweringRoute",
    "PhaseTensorFlowQNNGradientResult",
    "PhaseTensorFlowXLACompatibilityResult",
    "TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY",
    "PhaseTensorFlowMaintenanceReport",
    "PhaseTensorFlowMaintenanceRoute",
    "TensorFlowMaintenanceDecision",
    "TensorFlowMaintenanceStrategy",
    "run_tensorflow_maintenance_decision",
    "build_structured_ansatz",
    "complementary_polynomial",
    "jacobi_anger_cosine_coefficients",
    "jacobi_anger_sine_coefficients",
    "qsp_response",
    "qsp_unitary",
    "synthesise_qsp_phases",
    "LindbladSyncEngine",
    "OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY",
    "OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS",
    "BoundedOpenSystemObjectiveCase",
    "DensityMatrixInvariantCertificate",
    "MCWFReproducibilityCertificate",
    "OpenSystemObjectiveBoundaryRow",
    "OpenSystemObjectiveRecord",
    "OpenSystemObjectiveSuiteResult",
    "certify_density_matrix_invariants",
    "certify_mcwf_reproducibility",
    "default_open_system_objective_cases",
    "evaluate_lindblad_objective",
    "evaluate_mcwf_objective",
    "open_system_objective_boundary_rows",
    "run_open_system_objective_suite",
    "multi_frequency_parameter_shift_rule",
    "GenericParameterShiftEvaluationPlan",
    "parameter_shift_gradient",
    "parameter_shift_hessian",
    "parameter_shift_gradient_with_uncertainty",
    "plan_generic_parameter_shift_evaluations",
    "plan_parameter_shift_shots",
    "validate_param_shift_convergence",
    "value_and_parameter_shift_grad",
    "value_and_vqe_grad",
    "verify_parameter_shift_gradient",
    "verify_parameter_shift_hessian",
    "verify_vqe_parameter_shift_gradient",
    "verify_vqe_parameter_shift_hessian",
    "GradientVerificationResult",
    "HessianVerificationResult",
    "vqe_with_param_shift",
    "ParamShiftConvergenceDiagnostics",
    "ParamShiftVQEResult",
    "KuramotoVariant",
    "KuramotoVariantResult",
    "HigherOrderKuramotoSpec",
    "MonitoredKuramotoSpec",
    "PTSymmetricKuramotoSpec",
    "build_triadic_ring_terms",
    "simulate_higher_order_kuramoto",
    "simulate_monitored_kuramoto",
    "simulate_pt_symmetric_kuramoto",
    "transfer_experiment",
    "build_systems",
    "TransferResult",
    "AnsatzBenchmarkResult",
    "GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY",
    "GeneralisedParameterShiftPlan",
    "GeneralisedParameterShiftResult",
    "GeneralisedParameterShiftTerm",
    "GeneralisedStochasticParameterShiftResult",
    "estimate_generalised_parameter_shift_shot_noise",
    "generalised_parameter_shift_gradient",
    "plan_generalised_parameter_shift",
    "value_and_generalised_parameter_shift_grad",
    "QSPPhaseFactors",
    "QSPSynthesisError",
    "QSVTResourceEstimate",
    "ICIPulse",
    "HypergeometricPulse",
    "PulseSchedule",
    "build_ici_pulse",
    "build_hypergeometric_pulse",
    "build_trotter_pulse_schedule",
    "hypergeometric_envelope",
    "ici_three_level_evolution",
    "infidelity_bound",
    "SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY",
    "build_synchronisation_objective",
    "cluster_synchronisation_target_term",
    "kuramoto_order_parameter",
    "kuramoto_order_parameter_gradient",
    "kuramoto_order_parameter_target_term",
    "phase_locking_target_term",
]
