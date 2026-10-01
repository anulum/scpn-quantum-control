# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hardware Backends
"""Expose hardware backends and execution contracts.

The package layer provides classical exact solvers, the IBM Quantum runner,
noise models, trapped-ion transpilation, and experiment definitions.
"""

from __future__ import annotations

import sys as _sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .aggregators import (
        AggregatorProviderRoute,
        ResolvedAggregatorProviderRoute,
        aggregator_provider_routes_for,
        built_in_aggregator_provider_routes,
        resolve_aggregator_provider_route,
    )
    from .analog_kuramoto import (
        AnalogBackendCapabilities,
        AnalogCouplingTerm,
        AnalogDriveTerm,
        AnalogFeedbackTerm,
        AnalogKuramotoBackend,
        AnalogKuramotoBackendProtocol,
        AnalogKuramotoPlatform,
        AnalogKuramotoProgram,
        AnalogProviderTarget,
        ProviderAnalogExecutionPlan,
        ProviderAnalogPayload,
        analog_kuramoto_factory,
        compile_analog_kuramoto,
        export_provider_payload,
        prepare_provider_execution_plan,
    )
    from .analog_native_readiness import (
        ANALOG_NATIVE_SCHEMA,
        AnalogNativePrimitiveComparison,
        AnalogNativeReadinessConfig,
        AnalogProviderReadinessRow,
        analog_native_markdown,
        analog_native_payload,
        compare_native_to_digital_primitives,
        provider_readiness_rows,
    )
    from .async_runner import AsyncHardwareRunner, AsyncJobHandle, BackendSubstitutionError
    from .backends import (
        BackendProtocol,
        BackendRegistrationError,
        BackendRegistry,
        QuantumBackendDescriptor,
        describe_backend,
        describe_hal_backend_profile,
        discover_backends,
        get_backend,
        get_registry,
        list_backends,
        list_hal_backend_descriptors,
        list_quantum_backends,
        register_backend,
        unregister_backend,
    )
    from .classical import (
        INTEGRATION_GRID_RELATIVE_TOLERANCE,
        bloch_vectors_from_json,
        classical_brute_mpc,
        classical_exact_diag,
        classical_exact_evolution,
        classical_kuramoto_reference,
        integration_step_count,
        integration_times,
    )
    from .dynq_layout_pass import DynQLayoutPass, calibration_from_target
    from .experiments import (
        ALL_EXPERIMENTS,
        ansatz_comparison_hw_experiment,
        bell_test_4q_experiment,
        correlator_4q_experiment,
        decoherence_scaling_experiment,
        kuramoto_4osc_experiment,
        kuramoto_4osc_trotter2_experiment,
        kuramoto_4osc_zne_experiment,
        kuramoto_8osc_experiment,
        kuramoto_8osc_zne_experiment,
        noise_baseline_experiment,
        qaoa_mpc_4_experiment,
        qkd_qber_4q_experiment,
        sync_threshold_experiment,
        upde_16_dd_experiment,
        upde_16_snapshot_experiment,
        vqe_4q_experiment,
        vqe_8q_experiment,
        vqe_8q_hardware_experiment,
        vqe_landscape_experiment,
        zne_higher_order_experiment,
    )
    from .feedback_capability_probe import (
        BackendCapabilitySnapshot,
        FeedbackCapabilityDecision,
        assess_feedback_backend_capability,
        assess_feedback_backend_fleet,
        required_s1_dynamic_features,
    )
    from .feedback_dryrun import (
        FeedbackDryRunPayload,
        build_analog_native_review_payload,
        build_ibm_runtime_dry_run,
        build_openqasm3_gate_dry_run,
        build_s1_feedback_dry_run_bundle,
    )
    from .feedback_hardware_scheduler import (
        ApprovalGatedFeedbackHardwareScheduler,
        HardwareApprovalRecord,
        HardwareSubmissionRecord,
        hash_package_manifest,
    )
    from .feedback_loop import (
        FeedbackCommand,
        FeedbackLoopConfig,
        FeedbackLoopLatencySLA,
        FeedbackObserver,
        FeedbackResult,
        FeedbackRunner,
        FeedbackScheduler,
        FeedbackStepRecord,
        ProportionalMetricObserver,
        RealtimeControllerScheduler,
    )
    from .feedback_provider_metadata import (
        snapshot_from_generic_metadata,
        snapshot_from_qiskit_backend,
    )
    from .feedback_submission import (
        FeedbackBudgetEstimate,
        FeedbackCircuitSummary,
        FeedbackPlatformCapability,
        FeedbackSubmissionPackage,
        PlatformReadiness,
        assess_platform_readiness,
        build_s1_feedback_submission_package,
        default_s1_platforms,
        summarise_feedback_circuit,
    )
    from .hal import (
        BackendCapabilities,
        BackendProfile,
        HardwareAbstractionLayer,
        LocalDeterministicSimulator,
        QuantumBackend,
        QuantumJobRef,
        QuantumJobResult,
        QuantumWorkload,
        built_in_backend_profiles,
    )
    from .hal_azure import AzureQuantumHALAdapter, azure_openqasm3_to_workload
    from .hal_braket import (
        BraketAwsHALAdapter,
        BraketLocalHALAdapter,
        braket_circuit_to_workload,
    )
    from .hal_cirq import CirqLocalHALAdapter, cirq_circuit_workload
    from .hal_dwave import DWaveLeapHALAdapter, dwave_bqm_workload
    from .hal_ionq import IonQCloudHALAdapter, ionq_qis_workload
    from .hal_iqm import IQMHALAdapter, iqm_qiskit_workload
    from .hal_oqc import OQCHALAdapter, oqc_openqasm3_workload
    from .hal_pasqal import PasqalPulserHALAdapter, pulser_sequence_workload
    from .hal_pennylane import PennyLaneDeviceHALAdapter, pennylane_gate_workload
    from .hal_qbraid import QbraidRuntimeHALAdapter, qbraid_program_to_workload
    from .hal_qiskit import (
        QiskitAerHALAdapter,
        QiskitRuntimeHALAdapter,
        qiskit_circuit_to_qasm3_workload,
        qiskit_circuit_to_workload,
    )
    from .hal_quandela import QuandelaPercevalHALAdapter, quandela_perceval_workload
    from .hal_quantinuum import QuantinuumCloudHALAdapter, quantinuum_tket_workload
    from .hal_quera_bloqade import QuEraBloqadeHALAdapter, bloqade_ahs_workload
    from .hal_rigetti import RigettiQCSHALAdapter, rigetti_quil_workload
    from .hal_strangeworks import (
        StrangeworksComputeHALAdapter,
        strangeworks_program_to_workload,
    )
    from .hybrid_digital_analog import (
        HybridCouplingAssignment,
        HybridCouplingPartition,
        HybridDigitalAnalogBackend,
        HybridDigitalAnalogBackendProtocol,
        HybridDigitalAnalogProgram,
        HybridRoute,
        compile_hybrid_digital_analog,
        hybrid_digital_analog_factory,
        partition_kuramoto_couplings,
    )
    from .iqm_backend import (
        IQMBackendConfig,
        IQMQuantumBackend,
        IQMRunResult,
        IQMTargetCompilationError,
        iqm_factory,
        is_iqm_available,
    )
    from .job_dossier import HardwareJobDossier, build_s1_feedback_job_dossier
    from .kuramoto_layout_cost import (
        CostWeights,
        LayoutCost,
        dynq_mean_gate_fidelity,
        kuramoto_layout_cost,
        routed_layout_depth,
    )
    from .kuramoto_layout_optimiser import (
        LayoutSearchConfig,
        LayoutSearchResult,
        optimise_kuramoto_layout,
    )
    from .kuramoto_layout_relaxation import (
        RelaxationSearchResult,
        SinkhornRelaxationConfig,
        coupling_graph_distances,
        relax_kuramoto_layout,
        sinkhorn_normalise,
        swap_distance_surrogate,
    )
    from .noise_model import heron_r2_noise_model
    from .openpulse_control import (
        OpenPulseCalibrationWorkflow,
        OpenPulseInstruction,
        OpenPulseSchedule,
        OpenPulseWaveform,
        RabiCalibrationPoint,
        RabiPiCalibrationEstimate,
        build_rabi_amplitude_calibration_workflow,
        compile_hypergeometric_openpulse_schedule,
        estimate_rabi_pi_amplitude,
        schedule_to_qiskit_pulse,
    )
    from .provider_capability_discovery import (
        CapabilityDecisionStatus,
        OpenPulseControlReadiness,
        ProviderCapabilityDecision,
        ProviderCapabilitySnapshot,
        ProviderMetadataProbe,
        assess_provider_capability_snapshot,
        build_openpulse_control_readiness,
        probe_aggregator_provider_capability,
        snapshot_from_azure_target,
        snapshot_from_braket_device,
        snapshot_from_dwave_solver,
        snapshot_from_ionq_backend,
        snapshot_from_iqm_backend,
        snapshot_from_oqc_target,
        snapshot_from_pasqal_target,
        snapshot_from_qbraid_device,
        snapshot_from_qiskit_runtime_backend,
        snapshot_from_quandela_processor,
        snapshot_from_quantinuum_backend,
        snapshot_from_quera_bloqade,
        snapshot_from_rigetti_qcs,
        snapshot_from_strangeworks_backend,
    )
    from .provider_certification import (
        CERTIFICATION_CRITERIA,
        CertificationCriterion,
        ProviderCertificationRecord,
        ProviderCertificationReport,
        certify_provider_matrix,
        documented_backend_ids,
        focused_adapter_test_path,
        resolve_source_root,
    )
    from .provider_smoke import (
        AggregatorProviderOptionalDependencyRow,
        ProviderOptionalDependencyRow,
        aggregator_provider_optional_dependency_matrix,
        provider_optional_dependency_matrix,
    )
    from .qubit_mapper import (
        ExecutionRegion,
        QubitMappingResult,
        build_calibration_graph,
        detect_execution_regions,
        dynq_initial_layout,
        select_best_region,
    )
    from .runner import HardwareRunner, JobResult
    from .s1_feedback_ibm import (
        S1_CONTROL_ARM,
        S1_FEEDBACK_ARM,
        S1FeedbackArmCircuit,
        binary_phase_synchrony_from_counts,
        build_s1_arm_command,
        build_s1_feedback_arm_circuits,
        build_s1_xy_observable_arm_circuits,
        pauli_expectation_from_counts,
        raw_count_package_from_feedback_results,
        raw_count_package_from_xy_observable_results,
        run_ibm_sampler_arm,
    )
    from .trapped_ion import transpile_for_trapped_ion, trapped_ion_noise_model

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "AggregatorProviderRoute": (
        "scpn_quantum_control.hardware.aggregators",
        "AggregatorProviderRoute",
    ),
    "ResolvedAggregatorProviderRoute": (
        "scpn_quantum_control.hardware.aggregators",
        "ResolvedAggregatorProviderRoute",
    ),
    "aggregator_provider_routes_for": (
        "scpn_quantum_control.hardware.aggregators",
        "aggregator_provider_routes_for",
    ),
    "built_in_aggregator_provider_routes": (
        "scpn_quantum_control.hardware.aggregators",
        "built_in_aggregator_provider_routes",
    ),
    "resolve_aggregator_provider_route": (
        "scpn_quantum_control.hardware.aggregators",
        "resolve_aggregator_provider_route",
    ),
    "AnalogBackendCapabilities": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "AnalogBackendCapabilities",
    ),
    "AnalogCouplingTerm": ("scpn_quantum_control.hardware.analog_kuramoto", "AnalogCouplingTerm"),
    "AnalogDriveTerm": ("scpn_quantum_control.hardware.analog_kuramoto", "AnalogDriveTerm"),
    "AnalogFeedbackTerm": ("scpn_quantum_control.hardware.analog_kuramoto", "AnalogFeedbackTerm"),
    "AnalogKuramotoBackend": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "AnalogKuramotoBackend",
    ),
    "AnalogKuramotoBackendProtocol": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "AnalogKuramotoBackendProtocol",
    ),
    "AnalogKuramotoPlatform": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "AnalogKuramotoPlatform",
    ),
    "AnalogKuramotoProgram": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "AnalogKuramotoProgram",
    ),
    "AnalogProviderTarget": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "AnalogProviderTarget",
    ),
    "ProviderAnalogExecutionPlan": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "ProviderAnalogExecutionPlan",
    ),
    "ProviderAnalogPayload": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "ProviderAnalogPayload",
    ),
    "analog_kuramoto_factory": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "analog_kuramoto_factory",
    ),
    "compile_analog_kuramoto": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "compile_analog_kuramoto",
    ),
    "export_provider_payload": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "export_provider_payload",
    ),
    "prepare_provider_execution_plan": (
        "scpn_quantum_control.hardware.analog_kuramoto",
        "prepare_provider_execution_plan",
    ),
    "ANALOG_NATIVE_SCHEMA": (
        "scpn_quantum_control.hardware.analog_native_readiness",
        "ANALOG_NATIVE_SCHEMA",
    ),
    "AnalogNativePrimitiveComparison": (
        "scpn_quantum_control.hardware.analog_native_readiness",
        "AnalogNativePrimitiveComparison",
    ),
    "AnalogNativeReadinessConfig": (
        "scpn_quantum_control.hardware.analog_native_readiness",
        "AnalogNativeReadinessConfig",
    ),
    "AnalogProviderReadinessRow": (
        "scpn_quantum_control.hardware.analog_native_readiness",
        "AnalogProviderReadinessRow",
    ),
    "analog_native_markdown": (
        "scpn_quantum_control.hardware.analog_native_readiness",
        "analog_native_markdown",
    ),
    "analog_native_payload": (
        "scpn_quantum_control.hardware.analog_native_readiness",
        "analog_native_payload",
    ),
    "compare_native_to_digital_primitives": (
        "scpn_quantum_control.hardware.analog_native_readiness",
        "compare_native_to_digital_primitives",
    ),
    "provider_readiness_rows": (
        "scpn_quantum_control.hardware.analog_native_readiness",
        "provider_readiness_rows",
    ),
    "AsyncHardwareRunner": ("scpn_quantum_control.hardware.async_runner", "AsyncHardwareRunner"),
    "AsyncJobHandle": ("scpn_quantum_control.hardware.async_runner", "AsyncJobHandle"),
    "BackendSubstitutionError": (
        "scpn_quantum_control.hardware.async_runner",
        "BackendSubstitutionError",
    ),
    "BackendProtocol": ("scpn_quantum_control.hardware.backends", "BackendProtocol"),
    "BackendRegistrationError": (
        "scpn_quantum_control.hardware.backends",
        "BackendRegistrationError",
    ),
    "BackendRegistry": ("scpn_quantum_control.hardware.backends", "BackendRegistry"),
    "QuantumBackendDescriptor": (
        "scpn_quantum_control.hardware.backends",
        "QuantumBackendDescriptor",
    ),
    "describe_backend": ("scpn_quantum_control.hardware.backends", "describe_backend"),
    "describe_hal_backend_profile": (
        "scpn_quantum_control.hardware.backends",
        "describe_hal_backend_profile",
    ),
    "discover_backends": ("scpn_quantum_control.hardware.backends", "discover_backends"),
    "get_backend": ("scpn_quantum_control.hardware.backends", "get_backend"),
    "get_registry": ("scpn_quantum_control.hardware.backends", "get_registry"),
    "list_backends": ("scpn_quantum_control.hardware.backends", "list_backends"),
    "list_hal_backend_descriptors": (
        "scpn_quantum_control.hardware.backends",
        "list_hal_backend_descriptors",
    ),
    "list_quantum_backends": ("scpn_quantum_control.hardware.backends", "list_quantum_backends"),
    "register_backend": ("scpn_quantum_control.hardware.backends", "register_backend"),
    "unregister_backend": ("scpn_quantum_control.hardware.backends", "unregister_backend"),
    "INTEGRATION_GRID_RELATIVE_TOLERANCE": (
        "scpn_quantum_control.hardware.classical",
        "INTEGRATION_GRID_RELATIVE_TOLERANCE",
    ),
    "bloch_vectors_from_json": (
        "scpn_quantum_control.hardware.classical",
        "bloch_vectors_from_json",
    ),
    "classical_brute_mpc": ("scpn_quantum_control.hardware.classical", "classical_brute_mpc"),
    "classical_exact_diag": ("scpn_quantum_control.hardware.classical", "classical_exact_diag"),
    "classical_exact_evolution": (
        "scpn_quantum_control.hardware.classical",
        "classical_exact_evolution",
    ),
    "classical_kuramoto_reference": (
        "scpn_quantum_control.hardware.classical",
        "classical_kuramoto_reference",
    ),
    "integration_step_count": (
        "scpn_quantum_control.hardware.classical",
        "integration_step_count",
    ),
    "integration_times": ("scpn_quantum_control.hardware.classical", "integration_times"),
    "DynQLayoutPass": ("scpn_quantum_control.hardware.dynq_layout_pass", "DynQLayoutPass"),
    "calibration_from_target": (
        "scpn_quantum_control.hardware.dynq_layout_pass",
        "calibration_from_target",
    ),
    "ALL_EXPERIMENTS": ("scpn_quantum_control.hardware.experiments", "ALL_EXPERIMENTS"),
    "ansatz_comparison_hw_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "ansatz_comparison_hw_experiment",
    ),
    "bell_test_4q_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "bell_test_4q_experiment",
    ),
    "correlator_4q_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "correlator_4q_experiment",
    ),
    "decoherence_scaling_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "decoherence_scaling_experiment",
    ),
    "kuramoto_4osc_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "kuramoto_4osc_experiment",
    ),
    "kuramoto_4osc_trotter2_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "kuramoto_4osc_trotter2_experiment",
    ),
    "kuramoto_4osc_zne_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "kuramoto_4osc_zne_experiment",
    ),
    "kuramoto_8osc_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "kuramoto_8osc_experiment",
    ),
    "kuramoto_8osc_zne_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "kuramoto_8osc_zne_experiment",
    ),
    "noise_baseline_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "noise_baseline_experiment",
    ),
    "qaoa_mpc_4_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "qaoa_mpc_4_experiment",
    ),
    "qkd_qber_4q_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "qkd_qber_4q_experiment",
    ),
    "sync_threshold_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "sync_threshold_experiment",
    ),
    "upde_16_dd_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "upde_16_dd_experiment",
    ),
    "upde_16_snapshot_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "upde_16_snapshot_experiment",
    ),
    "vqe_4q_experiment": ("scpn_quantum_control.hardware.experiments", "vqe_4q_experiment"),
    "vqe_8q_experiment": ("scpn_quantum_control.hardware.experiments", "vqe_8q_experiment"),
    "vqe_8q_hardware_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "vqe_8q_hardware_experiment",
    ),
    "vqe_landscape_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "vqe_landscape_experiment",
    ),
    "zne_higher_order_experiment": (
        "scpn_quantum_control.hardware.experiments",
        "zne_higher_order_experiment",
    ),
    "BackendCapabilitySnapshot": (
        "scpn_quantum_control.hardware.feedback_capability_probe",
        "BackendCapabilitySnapshot",
    ),
    "FeedbackCapabilityDecision": (
        "scpn_quantum_control.hardware.feedback_capability_probe",
        "FeedbackCapabilityDecision",
    ),
    "assess_feedback_backend_capability": (
        "scpn_quantum_control.hardware.feedback_capability_probe",
        "assess_feedback_backend_capability",
    ),
    "assess_feedback_backend_fleet": (
        "scpn_quantum_control.hardware.feedback_capability_probe",
        "assess_feedback_backend_fleet",
    ),
    "required_s1_dynamic_features": (
        "scpn_quantum_control.hardware.feedback_capability_probe",
        "required_s1_dynamic_features",
    ),
    "FeedbackDryRunPayload": (
        "scpn_quantum_control.hardware.feedback_dryrun",
        "FeedbackDryRunPayload",
    ),
    "build_analog_native_review_payload": (
        "scpn_quantum_control.hardware.feedback_dryrun",
        "build_analog_native_review_payload",
    ),
    "build_ibm_runtime_dry_run": (
        "scpn_quantum_control.hardware.feedback_dryrun",
        "build_ibm_runtime_dry_run",
    ),
    "build_openqasm3_gate_dry_run": (
        "scpn_quantum_control.hardware.feedback_dryrun",
        "build_openqasm3_gate_dry_run",
    ),
    "build_s1_feedback_dry_run_bundle": (
        "scpn_quantum_control.hardware.feedback_dryrun",
        "build_s1_feedback_dry_run_bundle",
    ),
    "ApprovalGatedFeedbackHardwareScheduler": (
        "scpn_quantum_control.hardware.feedback_hardware_scheduler",
        "ApprovalGatedFeedbackHardwareScheduler",
    ),
    "HardwareApprovalRecord": (
        "scpn_quantum_control.hardware.feedback_hardware_scheduler",
        "HardwareApprovalRecord",
    ),
    "HardwareSubmissionRecord": (
        "scpn_quantum_control.hardware.feedback_hardware_scheduler",
        "HardwareSubmissionRecord",
    ),
    "hash_package_manifest": (
        "scpn_quantum_control.hardware.feedback_hardware_scheduler",
        "hash_package_manifest",
    ),
    "FeedbackCommand": ("scpn_quantum_control.hardware.feedback_loop", "FeedbackCommand"),
    "FeedbackLoopConfig": ("scpn_quantum_control.hardware.feedback_loop", "FeedbackLoopConfig"),
    "FeedbackLoopLatencySLA": (
        "scpn_quantum_control.hardware.feedback_loop",
        "FeedbackLoopLatencySLA",
    ),
    "FeedbackObserver": ("scpn_quantum_control.hardware.feedback_loop", "FeedbackObserver"),
    "FeedbackResult": ("scpn_quantum_control.hardware.feedback_loop", "FeedbackResult"),
    "FeedbackRunner": ("scpn_quantum_control.hardware.feedback_loop", "FeedbackRunner"),
    "FeedbackScheduler": ("scpn_quantum_control.hardware.feedback_loop", "FeedbackScheduler"),
    "FeedbackStepRecord": ("scpn_quantum_control.hardware.feedback_loop", "FeedbackStepRecord"),
    "ProportionalMetricObserver": (
        "scpn_quantum_control.hardware.feedback_loop",
        "ProportionalMetricObserver",
    ),
    "RealtimeControllerScheduler": (
        "scpn_quantum_control.hardware.feedback_loop",
        "RealtimeControllerScheduler",
    ),
    "snapshot_from_generic_metadata": (
        "scpn_quantum_control.hardware.feedback_provider_metadata",
        "snapshot_from_generic_metadata",
    ),
    "snapshot_from_qiskit_backend": (
        "scpn_quantum_control.hardware.feedback_provider_metadata",
        "snapshot_from_qiskit_backend",
    ),
    "FeedbackBudgetEstimate": (
        "scpn_quantum_control.hardware.feedback_submission",
        "FeedbackBudgetEstimate",
    ),
    "FeedbackCircuitSummary": (
        "scpn_quantum_control.hardware.feedback_submission",
        "FeedbackCircuitSummary",
    ),
    "FeedbackPlatformCapability": (
        "scpn_quantum_control.hardware.feedback_submission",
        "FeedbackPlatformCapability",
    ),
    "FeedbackSubmissionPackage": (
        "scpn_quantum_control.hardware.feedback_submission",
        "FeedbackSubmissionPackage",
    ),
    "PlatformReadiness": (
        "scpn_quantum_control.hardware.feedback_submission",
        "PlatformReadiness",
    ),
    "assess_platform_readiness": (
        "scpn_quantum_control.hardware.feedback_submission",
        "assess_platform_readiness",
    ),
    "build_s1_feedback_submission_package": (
        "scpn_quantum_control.hardware.feedback_submission",
        "build_s1_feedback_submission_package",
    ),
    "default_s1_platforms": (
        "scpn_quantum_control.hardware.feedback_submission",
        "default_s1_platforms",
    ),
    "summarise_feedback_circuit": (
        "scpn_quantum_control.hardware.feedback_submission",
        "summarise_feedback_circuit",
    ),
    "BackendCapabilities": ("scpn_quantum_control.hardware.hal", "BackendCapabilities"),
    "BackendProfile": ("scpn_quantum_control.hardware.hal", "BackendProfile"),
    "HardwareAbstractionLayer": ("scpn_quantum_control.hardware.hal", "HardwareAbstractionLayer"),
    "LocalDeterministicSimulator": (
        "scpn_quantum_control.hardware.hal",
        "LocalDeterministicSimulator",
    ),
    "QuantumBackend": ("scpn_quantum_control.hardware.hal", "QuantumBackend"),
    "QuantumJobRef": ("scpn_quantum_control.hardware.hal", "QuantumJobRef"),
    "QuantumJobResult": ("scpn_quantum_control.hardware.hal", "QuantumJobResult"),
    "QuantumWorkload": ("scpn_quantum_control.hardware.hal", "QuantumWorkload"),
    "built_in_backend_profiles": (
        "scpn_quantum_control.hardware.hal",
        "built_in_backend_profiles",
    ),
    "AzureQuantumHALAdapter": (
        "scpn_quantum_control.hardware.hal_azure",
        "AzureQuantumHALAdapter",
    ),
    "azure_openqasm3_to_workload": (
        "scpn_quantum_control.hardware.hal_azure",
        "azure_openqasm3_to_workload",
    ),
    "BraketAwsHALAdapter": ("scpn_quantum_control.hardware.hal_braket", "BraketAwsHALAdapter"),
    "BraketLocalHALAdapter": ("scpn_quantum_control.hardware.hal_braket", "BraketLocalHALAdapter"),
    "braket_circuit_to_workload": (
        "scpn_quantum_control.hardware.hal_braket",
        "braket_circuit_to_workload",
    ),
    "CirqLocalHALAdapter": ("scpn_quantum_control.hardware.hal_cirq", "CirqLocalHALAdapter"),
    "cirq_circuit_workload": ("scpn_quantum_control.hardware.hal_cirq", "cirq_circuit_workload"),
    "DWaveLeapHALAdapter": ("scpn_quantum_control.hardware.hal_dwave", "DWaveLeapHALAdapter"),
    "dwave_bqm_workload": ("scpn_quantum_control.hardware.hal_dwave", "dwave_bqm_workload"),
    "IonQCloudHALAdapter": ("scpn_quantum_control.hardware.hal_ionq", "IonQCloudHALAdapter"),
    "ionq_qis_workload": ("scpn_quantum_control.hardware.hal_ionq", "ionq_qis_workload"),
    "IQMHALAdapter": ("scpn_quantum_control.hardware.hal_iqm", "IQMHALAdapter"),
    "iqm_qiskit_workload": ("scpn_quantum_control.hardware.hal_iqm", "iqm_qiskit_workload"),
    "OQCHALAdapter": ("scpn_quantum_control.hardware.hal_oqc", "OQCHALAdapter"),
    "oqc_openqasm3_workload": ("scpn_quantum_control.hardware.hal_oqc", "oqc_openqasm3_workload"),
    "PasqalPulserHALAdapter": (
        "scpn_quantum_control.hardware.hal_pasqal",
        "PasqalPulserHALAdapter",
    ),
    "pulser_sequence_workload": (
        "scpn_quantum_control.hardware.hal_pasqal",
        "pulser_sequence_workload",
    ),
    "PennyLaneDeviceHALAdapter": (
        "scpn_quantum_control.hardware.hal_pennylane",
        "PennyLaneDeviceHALAdapter",
    ),
    "pennylane_gate_workload": (
        "scpn_quantum_control.hardware.hal_pennylane",
        "pennylane_gate_workload",
    ),
    "QbraidRuntimeHALAdapter": (
        "scpn_quantum_control.hardware.hal_qbraid",
        "QbraidRuntimeHALAdapter",
    ),
    "qbraid_program_to_workload": (
        "scpn_quantum_control.hardware.hal_qbraid",
        "qbraid_program_to_workload",
    ),
    "QiskitAerHALAdapter": ("scpn_quantum_control.hardware.hal_qiskit", "QiskitAerHALAdapter"),
    "QiskitRuntimeHALAdapter": (
        "scpn_quantum_control.hardware.hal_qiskit",
        "QiskitRuntimeHALAdapter",
    ),
    "qiskit_circuit_to_qasm3_workload": (
        "scpn_quantum_control.hardware.hal_qiskit",
        "qiskit_circuit_to_qasm3_workload",
    ),
    "qiskit_circuit_to_workload": (
        "scpn_quantum_control.hardware.hal_qiskit",
        "qiskit_circuit_to_workload",
    ),
    "QuandelaPercevalHALAdapter": (
        "scpn_quantum_control.hardware.hal_quandela",
        "QuandelaPercevalHALAdapter",
    ),
    "quandela_perceval_workload": (
        "scpn_quantum_control.hardware.hal_quandela",
        "quandela_perceval_workload",
    ),
    "QuantinuumCloudHALAdapter": (
        "scpn_quantum_control.hardware.hal_quantinuum",
        "QuantinuumCloudHALAdapter",
    ),
    "quantinuum_tket_workload": (
        "scpn_quantum_control.hardware.hal_quantinuum",
        "quantinuum_tket_workload",
    ),
    "QuEraBloqadeHALAdapter": (
        "scpn_quantum_control.hardware.hal_quera_bloqade",
        "QuEraBloqadeHALAdapter",
    ),
    "bloqade_ahs_workload": (
        "scpn_quantum_control.hardware.hal_quera_bloqade",
        "bloqade_ahs_workload",
    ),
    "RigettiQCSHALAdapter": ("scpn_quantum_control.hardware.hal_rigetti", "RigettiQCSHALAdapter"),
    "rigetti_quil_workload": (
        "scpn_quantum_control.hardware.hal_rigetti",
        "rigetti_quil_workload",
    ),
    "StrangeworksComputeHALAdapter": (
        "scpn_quantum_control.hardware.hal_strangeworks",
        "StrangeworksComputeHALAdapter",
    ),
    "strangeworks_program_to_workload": (
        "scpn_quantum_control.hardware.hal_strangeworks",
        "strangeworks_program_to_workload",
    ),
    "HybridCouplingAssignment": (
        "scpn_quantum_control.hardware.hybrid_digital_analog",
        "HybridCouplingAssignment",
    ),
    "HybridCouplingPartition": (
        "scpn_quantum_control.hardware.hybrid_digital_analog",
        "HybridCouplingPartition",
    ),
    "HybridDigitalAnalogBackend": (
        "scpn_quantum_control.hardware.hybrid_digital_analog",
        "HybridDigitalAnalogBackend",
    ),
    "HybridDigitalAnalogBackendProtocol": (
        "scpn_quantum_control.hardware.hybrid_digital_analog",
        "HybridDigitalAnalogBackendProtocol",
    ),
    "HybridDigitalAnalogProgram": (
        "scpn_quantum_control.hardware.hybrid_digital_analog",
        "HybridDigitalAnalogProgram",
    ),
    "HybridRoute": ("scpn_quantum_control.hardware.hybrid_digital_analog", "HybridRoute"),
    "compile_hybrid_digital_analog": (
        "scpn_quantum_control.hardware.hybrid_digital_analog",
        "compile_hybrid_digital_analog",
    ),
    "hybrid_digital_analog_factory": (
        "scpn_quantum_control.hardware.hybrid_digital_analog",
        "hybrid_digital_analog_factory",
    ),
    "partition_kuramoto_couplings": (
        "scpn_quantum_control.hardware.hybrid_digital_analog",
        "partition_kuramoto_couplings",
    ),
    "IQMBackendConfig": ("scpn_quantum_control.hardware.iqm_backend", "IQMBackendConfig"),
    "IQMQuantumBackend": ("scpn_quantum_control.hardware.iqm_backend", "IQMQuantumBackend"),
    "IQMRunResult": ("scpn_quantum_control.hardware.iqm_backend", "IQMRunResult"),
    "IQMTargetCompilationError": (
        "scpn_quantum_control.hardware.iqm_backend",
        "IQMTargetCompilationError",
    ),
    "iqm_factory": ("scpn_quantum_control.hardware.iqm_backend", "iqm_factory"),
    "is_iqm_available": ("scpn_quantum_control.hardware.iqm_backend", "is_iqm_available"),
    "HardwareJobDossier": ("scpn_quantum_control.hardware.job_dossier", "HardwareJobDossier"),
    "build_s1_feedback_job_dossier": (
        "scpn_quantum_control.hardware.job_dossier",
        "build_s1_feedback_job_dossier",
    ),
    "CostWeights": ("scpn_quantum_control.hardware.kuramoto_layout_cost", "CostWeights"),
    "LayoutCost": ("scpn_quantum_control.hardware.kuramoto_layout_cost", "LayoutCost"),
    "dynq_mean_gate_fidelity": (
        "scpn_quantum_control.hardware.kuramoto_layout_cost",
        "dynq_mean_gate_fidelity",
    ),
    "kuramoto_layout_cost": (
        "scpn_quantum_control.hardware.kuramoto_layout_cost",
        "kuramoto_layout_cost",
    ),
    "routed_layout_depth": (
        "scpn_quantum_control.hardware.kuramoto_layout_cost",
        "routed_layout_depth",
    ),
    "LayoutSearchConfig": (
        "scpn_quantum_control.hardware.kuramoto_layout_optimiser",
        "LayoutSearchConfig",
    ),
    "LayoutSearchResult": (
        "scpn_quantum_control.hardware.kuramoto_layout_optimiser",
        "LayoutSearchResult",
    ),
    "optimise_kuramoto_layout": (
        "scpn_quantum_control.hardware.kuramoto_layout_optimiser",
        "optimise_kuramoto_layout",
    ),
    "RelaxationSearchResult": (
        "scpn_quantum_control.hardware.kuramoto_layout_relaxation",
        "RelaxationSearchResult",
    ),
    "SinkhornRelaxationConfig": (
        "scpn_quantum_control.hardware.kuramoto_layout_relaxation",
        "SinkhornRelaxationConfig",
    ),
    "coupling_graph_distances": (
        "scpn_quantum_control.hardware.kuramoto_layout_relaxation",
        "coupling_graph_distances",
    ),
    "relax_kuramoto_layout": (
        "scpn_quantum_control.hardware.kuramoto_layout_relaxation",
        "relax_kuramoto_layout",
    ),
    "sinkhorn_normalise": (
        "scpn_quantum_control.hardware.kuramoto_layout_relaxation",
        "sinkhorn_normalise",
    ),
    "swap_distance_surrogate": (
        "scpn_quantum_control.hardware.kuramoto_layout_relaxation",
        "swap_distance_surrogate",
    ),
    "heron_r2_noise_model": ("scpn_quantum_control.hardware.noise_model", "heron_r2_noise_model"),
    "OpenPulseCalibrationWorkflow": (
        "scpn_quantum_control.hardware.openpulse_control",
        "OpenPulseCalibrationWorkflow",
    ),
    "OpenPulseInstruction": (
        "scpn_quantum_control.hardware.openpulse_control",
        "OpenPulseInstruction",
    ),
    "OpenPulseSchedule": ("scpn_quantum_control.hardware.openpulse_control", "OpenPulseSchedule"),
    "OpenPulseWaveform": ("scpn_quantum_control.hardware.openpulse_control", "OpenPulseWaveform"),
    "RabiCalibrationPoint": (
        "scpn_quantum_control.hardware.openpulse_control",
        "RabiCalibrationPoint",
    ),
    "RabiPiCalibrationEstimate": (
        "scpn_quantum_control.hardware.openpulse_control",
        "RabiPiCalibrationEstimate",
    ),
    "build_rabi_amplitude_calibration_workflow": (
        "scpn_quantum_control.hardware.openpulse_control",
        "build_rabi_amplitude_calibration_workflow",
    ),
    "compile_hypergeometric_openpulse_schedule": (
        "scpn_quantum_control.hardware.openpulse_control",
        "compile_hypergeometric_openpulse_schedule",
    ),
    "estimate_rabi_pi_amplitude": (
        "scpn_quantum_control.hardware.openpulse_control",
        "estimate_rabi_pi_amplitude",
    ),
    "schedule_to_qiskit_pulse": (
        "scpn_quantum_control.hardware.openpulse_control",
        "schedule_to_qiskit_pulse",
    ),
    "CapabilityDecisionStatus": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "CapabilityDecisionStatus",
    ),
    "OpenPulseControlReadiness": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "OpenPulseControlReadiness",
    ),
    "ProviderCapabilityDecision": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "ProviderCapabilityDecision",
    ),
    "ProviderCapabilitySnapshot": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "ProviderCapabilitySnapshot",
    ),
    "ProviderMetadataProbe": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "ProviderMetadataProbe",
    ),
    "assess_provider_capability_snapshot": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "assess_provider_capability_snapshot",
    ),
    "build_openpulse_control_readiness": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "build_openpulse_control_readiness",
    ),
    "probe_aggregator_provider_capability": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "probe_aggregator_provider_capability",
    ),
    "snapshot_from_azure_target": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_azure_target",
    ),
    "snapshot_from_braket_device": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_braket_device",
    ),
    "snapshot_from_dwave_solver": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_dwave_solver",
    ),
    "snapshot_from_ionq_backend": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_ionq_backend",
    ),
    "snapshot_from_iqm_backend": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_iqm_backend",
    ),
    "snapshot_from_oqc_target": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_oqc_target",
    ),
    "snapshot_from_pasqal_target": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_pasqal_target",
    ),
    "snapshot_from_qbraid_device": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_qbraid_device",
    ),
    "snapshot_from_qiskit_runtime_backend": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_qiskit_runtime_backend",
    ),
    "snapshot_from_quandela_processor": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_quandela_processor",
    ),
    "snapshot_from_quantinuum_backend": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_quantinuum_backend",
    ),
    "snapshot_from_quera_bloqade": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_quera_bloqade",
    ),
    "snapshot_from_rigetti_qcs": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_rigetti_qcs",
    ),
    "snapshot_from_strangeworks_backend": (
        "scpn_quantum_control.hardware.provider_capability_discovery",
        "snapshot_from_strangeworks_backend",
    ),
    "CERTIFICATION_CRITERIA": (
        "scpn_quantum_control.hardware.provider_certification",
        "CERTIFICATION_CRITERIA",
    ),
    "CertificationCriterion": (
        "scpn_quantum_control.hardware.provider_certification",
        "CertificationCriterion",
    ),
    "ProviderCertificationRecord": (
        "scpn_quantum_control.hardware.provider_certification",
        "ProviderCertificationRecord",
    ),
    "ProviderCertificationReport": (
        "scpn_quantum_control.hardware.provider_certification",
        "ProviderCertificationReport",
    ),
    "certify_provider_matrix": (
        "scpn_quantum_control.hardware.provider_certification",
        "certify_provider_matrix",
    ),
    "documented_backend_ids": (
        "scpn_quantum_control.hardware.provider_certification",
        "documented_backend_ids",
    ),
    "focused_adapter_test_path": (
        "scpn_quantum_control.hardware.provider_certification",
        "focused_adapter_test_path",
    ),
    "resolve_source_root": (
        "scpn_quantum_control.hardware.provider_certification",
        "resolve_source_root",
    ),
    "AggregatorProviderOptionalDependencyRow": (
        "scpn_quantum_control.hardware.provider_smoke",
        "AggregatorProviderOptionalDependencyRow",
    ),
    "ProviderOptionalDependencyRow": (
        "scpn_quantum_control.hardware.provider_smoke",
        "ProviderOptionalDependencyRow",
    ),
    "aggregator_provider_optional_dependency_matrix": (
        "scpn_quantum_control.hardware.provider_smoke",
        "aggregator_provider_optional_dependency_matrix",
    ),
    "provider_optional_dependency_matrix": (
        "scpn_quantum_control.hardware.provider_smoke",
        "provider_optional_dependency_matrix",
    ),
    "ExecutionRegion": ("scpn_quantum_control.hardware.qubit_mapper", "ExecutionRegion"),
    "QubitMappingResult": ("scpn_quantum_control.hardware.qubit_mapper", "QubitMappingResult"),
    "build_calibration_graph": (
        "scpn_quantum_control.hardware.qubit_mapper",
        "build_calibration_graph",
    ),
    "detect_execution_regions": (
        "scpn_quantum_control.hardware.qubit_mapper",
        "detect_execution_regions",
    ),
    "dynq_initial_layout": ("scpn_quantum_control.hardware.qubit_mapper", "dynq_initial_layout"),
    "select_best_region": ("scpn_quantum_control.hardware.qubit_mapper", "select_best_region"),
    "HardwareRunner": ("scpn_quantum_control.hardware.runner", "HardwareRunner"),
    "JobResult": ("scpn_quantum_control.hardware.runner", "JobResult"),
    "S1_CONTROL_ARM": ("scpn_quantum_control.hardware.s1_feedback_ibm", "S1_CONTROL_ARM"),
    "S1_FEEDBACK_ARM": ("scpn_quantum_control.hardware.s1_feedback_ibm", "S1_FEEDBACK_ARM"),
    "S1FeedbackArmCircuit": (
        "scpn_quantum_control.hardware.s1_feedback_ibm",
        "S1FeedbackArmCircuit",
    ),
    "binary_phase_synchrony_from_counts": (
        "scpn_quantum_control.hardware.s1_feedback_ibm",
        "binary_phase_synchrony_from_counts",
    ),
    "build_s1_arm_command": (
        "scpn_quantum_control.hardware.s1_feedback_ibm",
        "build_s1_arm_command",
    ),
    "build_s1_feedback_arm_circuits": (
        "scpn_quantum_control.hardware.s1_feedback_ibm",
        "build_s1_feedback_arm_circuits",
    ),
    "build_s1_xy_observable_arm_circuits": (
        "scpn_quantum_control.hardware.s1_feedback_ibm",
        "build_s1_xy_observable_arm_circuits",
    ),
    "pauli_expectation_from_counts": (
        "scpn_quantum_control.hardware.s1_feedback_ibm",
        "pauli_expectation_from_counts",
    ),
    "raw_count_package_from_feedback_results": (
        "scpn_quantum_control.hardware.s1_feedback_ibm",
        "raw_count_package_from_feedback_results",
    ),
    "raw_count_package_from_xy_observable_results": (
        "scpn_quantum_control.hardware.s1_feedback_ibm",
        "raw_count_package_from_xy_observable_results",
    ),
    "run_ibm_sampler_arm": (
        "scpn_quantum_control.hardware.s1_feedback_ibm",
        "run_ibm_sampler_arm",
    ),
    "transpile_for_trapped_ion": (
        "scpn_quantum_control.hardware.trapped_ion",
        "transpile_for_trapped_ion",
    ),
    "trapped_ion_noise_model": (
        "scpn_quantum_control.hardware.trapped_ion",
        "trapped_ion_noise_model",
    ),
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
    "AsyncHardwareRunner",
    "BackendSubstitutionError",
    "AsyncJobHandle",
    "AnalogBackendCapabilities",
    "AnalogCouplingTerm",
    "AnalogDriveTerm",
    "AnalogFeedbackTerm",
    "AnalogKuramotoBackend",
    "AnalogKuramotoBackendProtocol",
    "AnalogKuramotoPlatform",
    "AnalogKuramotoProgram",
    "AnalogProviderTarget",
    "ProviderAnalogExecutionPlan",
    "ProviderAnalogPayload",
    "analog_kuramoto_factory",
    "compile_analog_kuramoto",
    "export_provider_payload",
    "prepare_provider_execution_plan",
    "ANALOG_NATIVE_SCHEMA",
    "AnalogNativePrimitiveComparison",
    "AnalogNativeReadinessConfig",
    "AnalogProviderReadinessRow",
    "analog_native_markdown",
    "analog_native_payload",
    "compare_native_to_digital_primitives",
    "provider_readiness_rows",
    "AggregatorProviderRoute",
    "ResolvedAggregatorProviderRoute",
    "aggregator_provider_routes_for",
    "built_in_aggregator_provider_routes",
    "resolve_aggregator_provider_route",
    "HybridCouplingAssignment",
    "HybridCouplingPartition",
    "HybridDigitalAnalogBackend",
    "HybridDigitalAnalogBackendProtocol",
    "HybridDigitalAnalogProgram",
    "HybridRoute",
    "FeedbackCommand",
    "FeedbackLoopConfig",
    "FeedbackLoopLatencySLA",
    "FeedbackObserver",
    "FeedbackResult",
    "FeedbackRunner",
    "FeedbackScheduler",
    "FeedbackStepRecord",
    "ProportionalMetricObserver",
    "RealtimeControllerScheduler",
    "FeedbackBudgetEstimate",
    "FeedbackCircuitSummary",
    "FeedbackPlatformCapability",
    "FeedbackSubmissionPackage",
    "PlatformReadiness",
    "CapabilityDecisionStatus",
    "OpenPulseControlReadiness",
    "ProviderCapabilityDecision",
    "ProviderCapabilitySnapshot",
    "ProviderMetadataProbe",
    "AggregatorProviderOptionalDependencyRow",
    "CERTIFICATION_CRITERIA",
    "CertificationCriterion",
    "ProviderCertificationRecord",
    "ProviderCertificationReport",
    "ProviderOptionalDependencyRow",
    "assess_provider_capability_snapshot",
    "certify_provider_matrix",
    "documented_backend_ids",
    "focused_adapter_test_path",
    "resolve_source_root",
    "build_openpulse_control_readiness",
    "assess_platform_readiness",
    "aggregator_provider_optional_dependency_matrix",
    "build_s1_feedback_submission_package",
    "default_s1_platforms",
    "probe_aggregator_provider_capability",
    "provider_optional_dependency_matrix",
    "snapshot_from_azure_target",
    "snapshot_from_braket_device",
    "snapshot_from_dwave_solver",
    "snapshot_from_iqm_backend",
    "snapshot_from_ionq_backend",
    "snapshot_from_oqc_target",
    "snapshot_from_pasqal_target",
    "snapshot_from_quandela_processor",
    "snapshot_from_qiskit_runtime_backend",
    "snapshot_from_qbraid_device",
    "snapshot_from_quantinuum_backend",
    "snapshot_from_quera_bloqade",
    "snapshot_from_rigetti_qcs",
    "snapshot_from_strangeworks_backend",
    "summarise_feedback_circuit",
    "FeedbackDryRunPayload",
    "build_analog_native_review_payload",
    "build_ibm_runtime_dry_run",
    "build_openqasm3_gate_dry_run",
    "build_s1_feedback_dry_run_bundle",
    "AzureQuantumHALAdapter",
    "azure_openqasm3_to_workload",
    "BraketAwsHALAdapter",
    "BraketLocalHALAdapter",
    "braket_circuit_to_workload",
    "CirqLocalHALAdapter",
    "cirq_circuit_workload",
    "DWaveLeapHALAdapter",
    "dwave_bqm_workload",
    "IonQCloudHALAdapter",
    "ionq_qis_workload",
    "IQMHALAdapter",
    "iqm_qiskit_workload",
    "OQCHALAdapter",
    "oqc_openqasm3_workload",
    "PasqalPulserHALAdapter",
    "pulser_sequence_workload",
    "PennyLaneDeviceHALAdapter",
    "pennylane_gate_workload",
    "QbraidRuntimeHALAdapter",
    "StrangeworksComputeHALAdapter",
    "QuandelaPercevalHALAdapter",
    "quandela_perceval_workload",
    "QuEraBloqadeHALAdapter",
    "QuantinuumCloudHALAdapter",
    "RigettiQCSHALAdapter",
    "OpenPulseWaveform",
    "OpenPulseInstruction",
    "OpenPulseSchedule",
    "OpenPulseCalibrationWorkflow",
    "RabiCalibrationPoint",
    "RabiPiCalibrationEstimate",
    "compile_hypergeometric_openpulse_schedule",
    "build_rabi_amplitude_calibration_workflow",
    "estimate_rabi_pi_amplitude",
    "schedule_to_qiskit_pulse",
    "bloqade_ahs_workload",
    "qbraid_program_to_workload",
    "quantinuum_tket_workload",
    "rigetti_quil_workload",
    "strangeworks_program_to_workload",
    "BackendCapabilities",
    "BackendProfile",
    "HardwareAbstractionLayer",
    "LocalDeterministicSimulator",
    "QuantumBackend",
    "QuantumJobRef",
    "QuantumJobResult",
    "QuantumWorkload",
    "built_in_backend_profiles",
    "QiskitAerHALAdapter",
    "QiskitRuntimeHALAdapter",
    "qiskit_circuit_to_qasm3_workload",
    "qiskit_circuit_to_workload",
    "ApprovalGatedFeedbackHardwareScheduler",
    "HardwareApprovalRecord",
    "HardwareSubmissionRecord",
    "hash_package_manifest",
    "BackendCapabilitySnapshot",
    "FeedbackCapabilityDecision",
    "assess_feedback_backend_capability",
    "assess_feedback_backend_fleet",
    "required_s1_dynamic_features",
    "snapshot_from_generic_metadata",
    "snapshot_from_qiskit_backend",
    "HardwareJobDossier",
    "build_s1_feedback_job_dossier",
    "CostWeights",
    "LayoutCost",
    "dynq_mean_gate_fidelity",
    "kuramoto_layout_cost",
    "routed_layout_depth",
    "LayoutSearchConfig",
    "LayoutSearchResult",
    "optimise_kuramoto_layout",
    "RelaxationSearchResult",
    "SinkhornRelaxationConfig",
    "coupling_graph_distances",
    "relax_kuramoto_layout",
    "sinkhorn_normalise",
    "swap_distance_surrogate",
    "S1_CONTROL_ARM",
    "S1_FEEDBACK_ARM",
    "S1FeedbackArmCircuit",
    "binary_phase_synchrony_from_counts",
    "build_s1_arm_command",
    "build_s1_feedback_arm_circuits",
    "build_s1_xy_observable_arm_circuits",
    "pauli_expectation_from_counts",
    "raw_count_package_from_feedback_results",
    "raw_count_package_from_xy_observable_results",
    "run_ibm_sampler_arm",
    "compile_hybrid_digital_analog",
    "hybrid_digital_analog_factory",
    "partition_kuramoto_couplings",
    "IQMBackendConfig",
    "IQMQuantumBackend",
    "IQMRunResult",
    "IQMTargetCompilationError",
    "iqm_factory",
    "is_iqm_available",
    "BackendProtocol",
    "BackendRegistrationError",
    "BackendRegistry",
    "QuantumBackendDescriptor",
    "HardwareRunner",
    "describe_hal_backend_profile",
    "describe_backend",
    "discover_backends",
    "get_backend",
    "get_registry",
    "list_hal_backend_descriptors",
    "list_backends",
    "list_quantum_backends",
    "register_backend",
    "unregister_backend",
    "heron_r2_noise_model",
    "ALL_EXPERIMENTS",
    "ansatz_comparison_hw_experiment",
    "decoherence_scaling_experiment",
    "kuramoto_4osc_experiment",
    "kuramoto_4osc_trotter2_experiment",
    "kuramoto_4osc_zne_experiment",
    "kuramoto_8osc_experiment",
    "kuramoto_8osc_zne_experiment",
    "noise_baseline_experiment",
    "qaoa_mpc_4_experiment",
    "sync_threshold_experiment",
    "upde_16_dd_experiment",
    "upde_16_snapshot_experiment",
    "vqe_4q_experiment",
    "vqe_8q_experiment",
    "vqe_8q_hardware_experiment",
    "vqe_landscape_experiment",
    "zne_higher_order_experiment",
    "bell_test_4q_experiment",
    "correlator_4q_experiment",
    "qkd_qber_4q_experiment",
    "classical_kuramoto_reference",
    "classical_exact_diag",
    "classical_brute_mpc",
    "bloch_vectors_from_json",
    "classical_exact_evolution",
    "INTEGRATION_GRID_RELATIVE_TOLERANCE",
    "integration_step_count",
    "integration_times",
    "JobResult",
    "trapped_ion_noise_model",
    "transpile_for_trapped_ion",
    "ExecutionRegion",
    "QubitMappingResult",
    "build_calibration_graph",
    "detect_execution_regions",
    "dynq_initial_layout",
    "select_best_region",
    "DynQLayoutPass",
    "calibration_from_target",
]
