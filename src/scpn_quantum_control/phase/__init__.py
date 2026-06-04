# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Dynamics Solvers
"""Phase dynamics solvers: Kuramoto-XY Trotterisation, VQE ground-state search,
UPDE Trotter integration, Trotter error analysis, and ansatz benchmarking.
"""

from .adapt_vqe import ADAPTResult, adapt_vqe
from .adiabatic_preparation import AdiabaticResult, adiabatic_ramp
from .ansatz_bench import benchmark_ansatz, run_ansatz_benchmark
from .ansatz_methodology import AnsatzBenchmarkResult
from .avqds import AVQDSResult, avqds_simulate
from .cross_domain_transfer import TransferResult, build_systems, transfer_experiment
from .floquet_kuramoto import FloquetResult, floquet_evolve, scan_drive_amplitude
from .gradient_backend import (
    QuantumGradientBackendCapability,
    QuantumGradientPlan,
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
from .gradient_tape import QuantumGradientTape, TapeGradientRecord, gradient_tape
from .jax_bridge import (
    PhaseJAXGradientAgreementResult,
    PhaseJAXParameterShiftResult,
    check_jax_parameter_shift_agreement,
    is_phase_jax_available,
    jax_parameter_shift_value_and_grad,
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
from .param_shift import (
    GradientVerificationResult,
    HessianVerificationResult,
    ParamShiftConvergenceDiagnostics,
    ParamShiftVQEResult,
    multi_frequency_parameter_shift_rule,
    parameter_shift_gradient,
    parameter_shift_gradient_with_uncertainty,
    parameter_shift_hessian,
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
    PennyLaneRoundTripResult,
    check_pennylane_parameter_shift_agreement,
    check_pennylane_qnode_round_trip,
    is_phase_pennylane_available,
)
from .phase_vqe import PhaseVQE
from .provider_gradient import (
    ProviderExpectationSample,
    ProviderGradientExecutionResult,
    ProviderParameterShiftRecord,
    execute_provider_parameter_shift_gradient,
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
from .qiskit_bridge import (
    QiskitParameterShiftGradientResult,
    QiskitParameterShiftRecord,
    execute_qiskit_finite_shot_parameter_shift,
    execute_qiskit_statevector_parameter_shift,
    generate_qiskit_parameter_shift_circuits,
)
from .qsvt_evolution import QSVTResourceEstimate
from .results import TrajectoryResult
from .structured_ansatz import build_structured_ansatz
from .tensorflow_bridge import (
    PhaseTensorFlowParameterShiftResult,
    is_phase_tensorflow_available,
    tensorflow_parameter_shift_value_and_grad,
)
from .torch_bridge import (
    PhaseTorchParameterShiftResult,
    is_phase_torch_available,
    torch_parameter_shift_value_and_grad,
)
from .trotter_error import trotter_error_norm, trotter_error_sweep
from .trotter_upde import QuantumUPDESolver
from .varqite import VarQITEResult, varqite_ground_state
from .xy_kuramoto import QuantumKuramotoSolver, TrotterEvolutionConfig

__all__ = [
    "QuantumKuramotoSolver",
    "TrotterEvolutionConfig",
    "TrajectoryResult",
    "QuantumUPDESolver",
    "PhaseVQE",
    "trotter_error_norm",
    "trotter_error_sweep",
    "benchmark_ansatz",
    "run_ansatz_benchmark",
    "adapt_vqe",
    "ADAPTResult",
    "adiabatic_ramp",
    "AdiabaticResult",
    "avqds_simulate",
    "AVQDSResult",
    "varqite_ground_state",
    "VarQITEResult",
    "floquet_evolve",
    "scan_drive_amplitude",
    "FloquetResult",
    "quantum_gradient_backend_capability",
    "plan_quantum_gradient_backend",
    "QuantumGradientBackendCapability",
    "QuantumGradientPlan",
    "parameter_shift_gradient_descent",
    "validate_parameter_shift_training",
    "ParameterShiftTrainingCertificate",
    "ParameterShiftTrainingResult",
    "ParameterShiftTrainingStep",
    "gradient_tape",
    "QuantumGradientTape",
    "TapeGradientRecord",
    "is_phase_jax_available",
    "check_jax_parameter_shift_agreement",
    "jax_parameter_shift_value_and_grad",
    "PhaseJAXGradientAgreementResult",
    "PhaseJAXParameterShiftResult",
    "is_phase_pennylane_available",
    "check_pennylane_parameter_shift_agreement",
    "check_pennylane_qnode_round_trip",
    "PennyLaneGradientAgreementResult",
    "PennyLaneRoundTripResult",
    "ProviderExpectationSample",
    "ProviderGradientExecutionResult",
    "ProviderParameterShiftRecord",
    "execute_provider_parameter_shift_gradient",
    "QiskitParameterShiftGradientResult",
    "QiskitParameterShiftRecord",
    "execute_qiskit_finite_shot_parameter_shift",
    "execute_qiskit_statevector_parameter_shift",
    "generate_qiskit_parameter_shift_circuits",
    "is_phase_torch_available",
    "torch_parameter_shift_value_and_grad",
    "PhaseTorchParameterShiftResult",
    "is_phase_tensorflow_available",
    "tensorflow_parameter_shift_value_and_grad",
    "PhaseTensorFlowParameterShiftResult",
    "build_structured_ansatz",
    "LindbladSyncEngine",
    "multi_frequency_parameter_shift_rule",
    "parameter_shift_gradient",
    "parameter_shift_hessian",
    "parameter_shift_gradient_with_uncertainty",
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
]
