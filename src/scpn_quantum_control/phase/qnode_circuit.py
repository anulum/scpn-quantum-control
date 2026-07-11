# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Circuit Registry
"""Executable facade for registered Phase-QNode circuit routes.

Shared declarations and constructor validation live in :mod:`.qnode_circuit_contracts`;
registered vocabulary, observables, decompositions, and templates live in
:mod:`.qnode_circuit_builders`; registration, support analysis, depth profiling, and
parameter-shift planning live in :mod:`.qnode_circuit_support`; statevector and density
execution plus observable kernels live in :mod:`.qnode_circuit_execution`; gradient,
Fisher, metric, and derivative orchestration live in
:mod:`.qnode_circuit_differentiation`. This module is the shallow compatibility facade
that re-exports those exact public and private objects.
"""

from __future__ import annotations

from .qnode_circuit_builders import (
    _as_broadcast_coefficients as _as_broadcast_coefficients,
)
from .qnode_circuit_builders import (
    _as_ising_chain_width as _as_ising_chain_width,
)
from .qnode_circuit_builders import (
    _as_template_entangler as _as_template_entangler,
)
from .qnode_circuit_builders import (
    _as_template_layers as _as_template_layers,
)
from .qnode_circuit_builders import (
    _as_template_width as _as_template_width,
)
from .qnode_circuit_builders import (
    _entangler_operations as _entangler_operations,
)
from .qnode_circuit_builders import (
    _ghz_chain_operations as _ghz_chain_operations,
)
from .qnode_circuit_builders import (
    _hardware_efficient_operations as _hardware_efficient_operations,
)
from .qnode_circuit_builders import (
    _normalise_template_observable as _normalise_template_observable,
)
from .qnode_circuit_builders import (
    _require_qubit_width as _require_qubit_width,
)
from .qnode_circuit_builders import (
    _z_magnetization_observable as _z_magnetization_observable,
)
from .qnode_circuit_builders import (
    build_phase_qnode_template as build_phase_qnode_template,
)
from .qnode_circuit_builders import (
    build_sparse_ising_chain_hamiltonian as build_sparse_ising_chain_hamiltonian,
)
from .qnode_circuit_builders import (
    decompose_phase_qnode_controlled_gate as decompose_phase_qnode_controlled_gate,
)
from .qnode_circuit_builders import (
    registered_phase_qnode_decompositions as registered_phase_qnode_decompositions,
)
from .qnode_circuit_builders import (
    registered_phase_qnode_gates as registered_phase_qnode_gates,
)
from .qnode_circuit_builders import (
    registered_phase_qnode_noise_channels as registered_phase_qnode_noise_channels,
)
from .qnode_circuit_builders import (
    registered_phase_qnode_observables as registered_phase_qnode_observables,
)
from .qnode_circuit_builders import (
    registered_phase_qnode_templates as registered_phase_qnode_templates,
)
from .qnode_circuit_contracts import (
    _GATE_ARITY as _GATE_ARITY,
)
from .qnode_circuit_contracts import (
    _H as _H,
)
from .qnode_circuit_contracts import (
    _I as _I,
)
from .qnode_circuit_contracts import (
    _NON_PARAMETRIC_GATES as _NON_PARAMETRIC_GATES,
)
from .qnode_circuit_contracts import (
    _PARAMETRIC_GATES as _PARAMETRIC_GATES,
)
from .qnode_circuit_contracts import (
    _PAULI as _PAULI,
)
from .qnode_circuit_contracts import (
    _PAULI_MATRICES as _PAULI_MATRICES,
)
from .qnode_circuit_contracts import (
    _REGISTERED_DECOMPOSITIONS as _REGISTERED_DECOMPOSITIONS,
)
from .qnode_circuit_contracts import (
    _REGISTERED_GATES as _REGISTERED_GATES,
)
from .qnode_circuit_contracts import (
    _REGISTERED_NOISE_CHANNELS as _REGISTERED_NOISE_CHANNELS,
)
from .qnode_circuit_contracts import (
    _REGISTERED_OBSERVABLES as _REGISTERED_OBSERVABLES,
)
from .qnode_circuit_contracts import (
    _REGISTERED_TEMPLATES as _REGISTERED_TEMPLATES,
)
from .qnode_circuit_contracts import (
    _S as _S,
)
from .qnode_circuit_contracts import (
    _SX as _SX,
)
from .qnode_circuit_contracts import (
    _T as _T,
)
from .qnode_circuit_contracts import (
    _X as _X,
)
from .qnode_circuit_contracts import (
    _Y as _Y,
)
from .qnode_circuit_contracts import (
    _Z as _Z,
)
from .qnode_circuit_contracts import (
    ComplexArray as ComplexArray,
)
from .qnode_circuit_contracts import (
    DenseHermitianObservable as DenseHermitianObservable,
)
from .qnode_circuit_contracts import (
    DensityOperation as DensityOperation,
)
from .qnode_circuit_contracts import (
    DensityOperationSpec as DensityOperationSpec,
)
from .qnode_circuit_contracts import (
    FloatArray as FloatArray,
)
from .qnode_circuit_contracts import (
    OperationSpec as OperationSpec,
)
from .qnode_circuit_contracts import (
    PauliCovarianceObservable as PauliCovarianceObservable,
)
from .qnode_circuit_contracts import (
    PauliTerm as PauliTerm,
)
from .qnode_circuit_contracts import (
    PhaseQNodeCircuit as PhaseQNodeCircuit,
)
from .qnode_circuit_contracts import (
    PhaseQNodeClassicalFisherResult as PhaseQNodeClassicalFisherResult,
)
from .qnode_circuit_contracts import (
    PhaseQNodeDensityCircuit as PhaseQNodeDensityCircuit,
)
from .qnode_circuit_contracts import (
    PhaseQNodeDensityExecutionResult as PhaseQNodeDensityExecutionResult,
)
from .qnode_circuit_contracts import (
    PhaseQNodeDepthProfile as PhaseQNodeDepthProfile,
)
from .qnode_circuit_contracts import (
    PhaseQNodeExecutionResult as PhaseQNodeExecutionResult,
)
from .qnode_circuit_contracts import (
    PhaseQNodeGradientEvaluationGroup as PhaseQNodeGradientEvaluationGroup,
)
from .qnode_circuit_contracts import (
    PhaseQNodeGradientEvaluationPlan as PhaseQNodeGradientEvaluationPlan,
)
from .qnode_circuit_contracts import (
    PhaseQNodeGradientResult as PhaseQNodeGradientResult,
)
from .qnode_circuit_contracts import (
    PhaseQNodeMetricTensorResult as PhaseQNodeMetricTensorResult,
)
from .qnode_circuit_contracts import (
    PhaseQNodeNoiseChannel as PhaseQNodeNoiseChannel,
)
from .qnode_circuit_contracts import (
    PhaseQNodeOperation as PhaseQNodeOperation,
)
from .qnode_circuit_contracts import (
    PhaseQNodeRegisteredCircuitSpec as PhaseQNodeRegisteredCircuitSpec,
)
from .qnode_circuit_contracts import (
    PhaseQNodeSupportError as PhaseQNodeSupportError,
)
from .qnode_circuit_contracts import (
    PhaseQNodeSupportReport as PhaseQNodeSupportReport,
)
from .qnode_circuit_contracts import (
    PhaseQNodeTemplateSpec as PhaseQNodeTemplateSpec,
)
from .qnode_circuit_contracts import (
    SparsePauliHamiltonian as SparsePauliHamiltonian,
)
from .qnode_circuit_contracts import (
    _as_finite_scalar as _as_finite_scalar,
)
from .qnode_circuit_contracts import (
    _as_probability as _as_probability,
)
from .qnode_circuit_contracts import (
    _FiniteShotFisherEvidence as _FiniteShotFisherEvidence,
)
from .qnode_circuit_contracts import (
    _normalise_observable as _normalise_observable,
)
from .qnode_circuit_contracts import (
    _optional_float_array_to_list as _optional_float_array_to_list,
)
from .qnode_circuit_contracts import (
    _parse_density_operation as _parse_density_operation,
)
from .qnode_circuit_contracts import (
    _parse_operation as _parse_operation,
)
from .qnode_circuit_differentiation import (
    _as_confidence_level as _as_confidence_level,
)
from .qnode_circuit_differentiation import (
    _as_confidence_z as _as_confidence_z,
)
from .qnode_circuit_differentiation import (
    _as_min_probability as _as_min_probability,
)
from .qnode_circuit_differentiation import (
    _as_observed_count_record as _as_observed_count_record,
)
from .qnode_circuit_differentiation import (
    _as_shot_count as _as_shot_count,
)
from .qnode_circuit_differentiation import (
    _classical_fisher_delta_method_standard_error as _classical_fisher_delta_method_standard_error,
)
from .qnode_circuit_differentiation import (
    _classical_fisher_from_probabilities as _classical_fisher_from_probabilities,
)
from .qnode_circuit_differentiation import (
    _covariance_product_rule_gradient as _covariance_product_rule_gradient,
)
from .qnode_circuit_differentiation import (
    _execute_state_and_parameter_derivatives as _execute_state_and_parameter_derivatives,
)
from .qnode_circuit_differentiation import (
    _finite_shot_fisher_evidence as _finite_shot_fisher_evidence,
)
from .qnode_circuit_differentiation import (
    _gate_derivative_matrix as _gate_derivative_matrix,
)
from .qnode_circuit_differentiation import (
    _operation_derivative_matrix as _operation_derivative_matrix,
)
from .qnode_circuit_differentiation import (
    _parametric_operations_by_parameter as _parametric_operations_by_parameter,
)
from .qnode_circuit_differentiation import (
    parameter_shift_phase_qnode_gradient as parameter_shift_phase_qnode_gradient,
)
from .qnode_circuit_differentiation import (
    phase_qnode_computational_basis_fisher_information as phase_qnode_computational_basis_fisher_information,
)
from .qnode_circuit_differentiation import (
    phase_qnode_computational_basis_fisher_support_report as phase_qnode_computational_basis_fisher_support_report,
)
from .qnode_circuit_differentiation import (
    phase_qnode_natural_gradient_metric as phase_qnode_natural_gradient_metric,
)
from .qnode_circuit_differentiation import (
    phase_qnode_quantum_fisher_information as phase_qnode_quantum_fisher_information,
)
from .qnode_circuit_execution import (
    _apply_gate_matrix as _apply_gate_matrix,
)
from .qnode_circuit_execution import (
    _apply_noise_channel_density_matrix as _apply_noise_channel_density_matrix,
)
from .qnode_circuit_execution import (
    _apply_operation as _apply_operation,
)
from .qnode_circuit_execution import (
    _apply_term_operator as _apply_term_operator,
)
from .qnode_circuit_execution import (
    _apply_unitary_density_matrix as _apply_unitary_density_matrix,
)
from .qnode_circuit_execution import (
    _as_density_circuit as _as_density_circuit,
)
from .qnode_circuit_execution import (
    _ccnot_matrix as _ccnot_matrix,
)
from .qnode_circuit_execution import (
    _controlled as _controlled,
)
from .qnode_circuit_execution import (
    _covariance_expectation as _covariance_expectation,
)
from .qnode_circuit_execution import (
    _cswap_matrix as _cswap_matrix,
)
from .qnode_circuit_execution import (
    _density_covariance_expectation as _density_covariance_expectation,
)
from .qnode_circuit_execution import (
    _density_expectation_value as _density_expectation_value,
)
from .qnode_circuit_execution import (
    _density_term_expectation as _density_term_expectation,
)
from .qnode_circuit_execution import (
    _execute_state as _execute_state,
)
from .qnode_circuit_execution import (
    _expanded_operator as _expanded_operator,
)
from .qnode_circuit_execution import (
    _expectation_value as _expectation_value,
)
from .qnode_circuit_execution import (
    _gate_matrix as _gate_matrix,
)
from .qnode_circuit_execution import (
    _noise_channel_kraus as _noise_channel_kraus,
)
from .qnode_circuit_execution import (
    _operation_matrix as _operation_matrix,
)
from .qnode_circuit_execution import (
    _symmetrized_product_expectation as _symmetrized_product_expectation,
)
from .qnode_circuit_execution import (
    _term_expectation as _term_expectation,
)
from .qnode_circuit_execution import (
    _term_operator as _term_operator,
)
from .qnode_circuit_execution import (
    _term_product_expectation as _term_product_expectation,
)
from .qnode_circuit_execution import (
    execute_phase_qnode_circuit as execute_phase_qnode_circuit,
)
from .qnode_circuit_execution import (
    execute_phase_qnode_density_matrix as execute_phase_qnode_density_matrix,
)
from .qnode_circuit_support import (
    _as_optional_positive_int as _as_optional_positive_int,
)
from .qnode_circuit_support import (
    _as_parameter_vector as _as_parameter_vector,
)
from .qnode_circuit_support import (
    _base_generator_frequencies as _base_generator_frequencies,
)
from .qnode_circuit_support import (
    _blocked_density_route_support_report as _blocked_density_route_support_report,
)
from .qnode_circuit_support import (
    _collapsible_shared_parameter_group as _collapsible_shared_parameter_group,
)
from .qnode_circuit_support import (
    _generator_pauli_map as _generator_pauli_map,
)
from .qnode_circuit_support import (
    _group_generator_frequencies as _group_generator_frequencies,
)
from .qnode_circuit_support import (
    _is_controlled_parametric_gate as _is_controlled_parametric_gate,
)
from .qnode_circuit_support import (
    _observable_kind as _observable_kind,
)
from .qnode_circuit_support import (
    _operations_commute_for_shared_parameter as _operations_commute_for_shared_parameter,
)
from .qnode_circuit_support import (
    _parameter_generator_key as _parameter_generator_key,
)
from .qnode_circuit_support import (
    _parameter_generators_commute as _parameter_generators_commute,
)
from .qnode_circuit_support import (
    _parameter_shift_terms_for_group as _parameter_shift_terms_for_group,
)
from .qnode_circuit_support import (
    _parsed_density_operations as _parsed_density_operations,
)
from .qnode_circuit_support import (
    _parsed_operations as _parsed_operations,
)
from .qnode_circuit_support import (
    build_registered_phase_qnode_circuit as build_registered_phase_qnode_circuit,
)
from .qnode_circuit_support import (
    phase_qnode_density_support_report as phase_qnode_density_support_report,
)
from .qnode_circuit_support import (
    phase_qnode_depth_profile as phase_qnode_depth_profile,
)
from .qnode_circuit_support import (
    phase_qnode_gradient_support_report as phase_qnode_gradient_support_report,
)
from .qnode_circuit_support import (
    phase_qnode_metric_support_report as phase_qnode_metric_support_report,
)
from .qnode_circuit_support import (
    phase_qnode_support_report as phase_qnode_support_report,
)
from .qnode_circuit_support import (
    plan_phase_qnode_parameter_shift_evaluations as plan_phase_qnode_parameter_shift_evaluations,
)

__all__ = [
    "DenseHermitianObservable",
    "PauliCovarianceObservable",
    "PauliTerm",
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
    "build_sparse_ising_chain_hamiltonian",
    "build_registered_phase_qnode_circuit",
    "build_phase_qnode_template",
    "decompose_phase_qnode_controlled_gate",
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
]
