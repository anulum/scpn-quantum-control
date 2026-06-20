# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust parity tests for Phase-QNode differentials
"""Parity tests for Phase-QNode differentiable primitives exposed by PyO3."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    Parameter,
    parameter_shift_gradient_with_uncertainty,
)
from scpn_quantum_control.phase import (
    PhaseQNodeCircuit,
    execute_phase_qnode_hessian_vector_product,
    execute_phase_qnode_vector_hessian,
    execute_phase_qnode_vector_jvp,
    execute_phase_qnode_vector_vjp,
    phase_qnode_complex_derivative_contract,
    phase_qnode_computational_basis_fisher_information,
    phase_qnode_quantum_fisher_information,
)

engine = pytest.importorskip("scpn_quantum_engine")

_EXECUTABLE_SCALAR_PROGRAM_AD_IR = """{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [], "dtype": "float64", "effect": 6}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["y"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "pure", "target": "%2", "inputs": ["%0", "%0"], "version": 0, "ordering": 2, "operation": "mul"},
    {"index": 3, "kind": "pure", "target": "%3", "inputs": ["%1", "2.0"], "version": 0, "ordering": 3, "operation": "mul"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%2", "%3"], "version": 0, "ordering": 4, "operation": "add"},
    {"index": 5, "kind": "primitive", "target": "%5", "inputs": ["%0"], "version": 0, "ordering": 5, "operation": "sin"},
    {"index": 6, "kind": "pure", "target": "%6", "inputs": ["%4", "%5"], "version": 0, "ordering": 6, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"""


def _require_export(name: str) -> Any:
    if not hasattr(engine, name):
        pytest.skip(f"scpn_quantum_engine is installed without {name}")
    return getattr(engine, name)


def _ry_state_and_derivative(
    theta: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    state_re = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)], dtype=np.float64)
    state_im = np.zeros(2, dtype=np.float64)
    derivatives_re = np.array(
        [[-0.5 * np.sin(theta / 2.0), 0.5 * np.cos(theta / 2.0)]],
        dtype=np.float64,
    )
    derivatives_im = np.zeros((1, 2), dtype=np.float64)
    return state_re, state_im, derivatives_re, derivatives_im


def _vector_objective(params: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array(
        [
            np.cos(params[0]) + 0.1 * np.sin(params[1]),
            np.sin(params[0]) - 0.25 * np.cos(params[1]),
        ],
        dtype=float,
    )


def _scalar_objective(params: NDArray[np.float64]) -> float:
    return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))


def test_rust_phase_qnode_fubini_study_metric_matches_python_qfi_surface() -> None:
    rust_metric = _require_export("phase_qnode_fubini_study_metric_rust")
    theta = 0.31
    state_re, state_im, derivatives_re, derivatives_im = _ry_state_and_derivative(theta)
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    python_result = phase_qnode_quantum_fisher_information(
        circuit,
        np.array([theta], dtype=np.float64),
    )
    metric, qfi, derivative_norms = rust_metric(
        state_re,
        state_im,
        derivatives_re,
        derivatives_im,
    )

    np.testing.assert_allclose(metric, python_result.fubini_study_metric, atol=1e-12)
    np.testing.assert_allclose(qfi, python_result.quantum_fisher_information, atol=1e-12)
    np.testing.assert_allclose(derivative_norms, python_result.derivative_norms, atol=1e-12)


def test_rust_program_ad_value_and_gradient_replay_matches_scalar_reference() -> None:
    """Verify PyO3 Rust Program AD value+gradient replay for a scalar IR subset."""
    rust_value_and_gradient = _require_export("program_ad_effect_ir_interpret_value_and_gradient")

    payload = json.loads(rust_value_and_gradient(_EXECUTABLE_SCALAR_PROGRAM_AD_IR, [0.4, -0.2]))

    assert payload["supported"] is True
    assert payload["blocked_reasons"] == []
    assert payload["parameter_targets"] == ["%0", "%1"]
    assert payload["value"] == pytest.approx(0.4**2 + 2.0 * -0.2 + np.sin(0.4))
    assert payload["gradient"] == pytest.approx([2.0 * 0.4 + np.cos(0.4), 2.0])
    assert "scalar_primitives" in payload["claim_boundary"]
    assert "executed_branch_no_alias" in payload["claim_boundary"]


def test_rust_phase_qnode_computational_basis_fisher_matches_python_surface() -> None:
    rust_fisher = _require_export("phase_qnode_computational_basis_fisher_rust")
    theta = 0.31
    state_re, state_im, derivatives_re, derivatives_im = _ry_state_and_derivative(theta)
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    python_result = phase_qnode_computational_basis_fisher_information(
        circuit,
        np.array([theta], dtype=np.float64),
    )
    fisher, probabilities, probability_derivatives = rust_fisher(
        state_re,
        state_im,
        derivatives_re,
        derivatives_im,
        1e-15,
    )

    np.testing.assert_allclose(
        fisher,
        python_result.classical_fisher_information,
        atol=1e-12,
    )
    np.testing.assert_allclose(probabilities, python_result.probabilities, atol=1e-12)
    np.testing.assert_allclose(
        probability_derivatives,
        python_result.probability_derivatives,
        atol=1e-12,
    )


def test_rust_phase_qnode_directional_transforms_match_python_surfaces() -> None:
    rust_jvp = _require_export("phase_qnode_vector_jvp_rust")
    rust_vjp = _require_export("phase_qnode_vector_vjp_rust")
    rust_hvp = _require_export("phase_qnode_hessian_vector_product_rust")
    params = np.array([0.31, -0.17], dtype=np.float64)
    tangent = np.array([0.5, -1.25], dtype=np.float64)
    cotangent = np.array([2.0, -0.75], dtype=np.float64)
    vector = np.array([0.75, -2.0], dtype=np.float64)

    python_jvp = execute_phase_qnode_vector_jvp(_vector_objective, params, tangent)
    python_vjp = execute_phase_qnode_vector_vjp(_vector_objective, params, cotangent)
    python_hvp = execute_phase_qnode_hessian_vector_product(_scalar_objective, params, vector)
    assert python_jvp.jacobian is not None
    assert python_vjp.jacobian is not None
    assert python_hvp.hessian is not None
    assert python_jvp.jvp is not None
    assert python_vjp.vjp is not None
    assert python_hvp.hessian_vector_product is not None

    np.testing.assert_allclose(rust_jvp(python_jvp.jacobian, tangent), python_jvp.jvp, atol=1e-12)
    np.testing.assert_allclose(
        rust_vjp(python_vjp.jacobian, cotangent), python_vjp.vjp, atol=1e-12
    )
    np.testing.assert_allclose(
        rust_hvp(python_hvp.hessian, vector),
        python_hvp.hessian_vector_product,
        atol=1e-12,
    )


def test_rust_phase_qnode_vector_hessian_tensor_matches_python_surface() -> None:
    rust_hessian_tensor = _require_export("phase_qnode_vector_hessian_tensor_rust")
    params = np.array([0.31, -0.17], dtype=np.float64)

    python_result = execute_phase_qnode_vector_hessian(_vector_objective, params)
    assert python_result.hessian_tensor is not None

    np.testing.assert_allclose(
        rust_hessian_tensor(python_result.hessian_tensor),
        python_result.hessian_tensor,
        atol=1e-12,
    )


def test_rust_phase_qnode_complex_contract_matches_python_fail_closed_boundary() -> None:
    rust_contract = _require_export("phase_qnode_complex_derivative_contract_rust")()
    python_contract = phase_qnode_complex_derivative_contract()

    assert rust_contract["parameter_domain"] == python_contract.parameter_domain
    assert rust_contract["accepts_complex_parameters"] is False
    assert rust_contract["accepts_complex_tangents"] is False
    assert rust_contract["holomorphic_derivatives"] is False
    assert rust_contract["wirtinger_partials"] is False


def test_rust_parameter_shift_gradient_uncertainty_matches_python_surface() -> None:
    rust_uncertainty = engine.parameter_shift_gradient_uncertainty_rust
    plus_values = np.array([0.8, 0.1], dtype=np.float64)
    minus_values = np.array([0.2, -0.3], dtype=np.float64)
    plus_variances = np.array([0.36, 0.25], dtype=np.float64)
    minus_variances = np.array([0.16, 0.09], dtype=np.float64)
    plus_shots = np.array([900.0, 400.0], dtype=np.float64)
    minus_shots = np.array([400.0, 100.0], dtype=np.float64)
    trainable = np.array([True, False], dtype=np.bool_)
    coefficients = np.array([0.5], dtype=np.float64)

    python_result = parameter_shift_gradient_with_uncertainty(
        plus_values=plus_values,
        minus_values=minus_values,
        plus_variances=plus_variances,
        minus_variances=minus_variances,
        plus_shots=plus_shots,
        minus_shots=minus_shots,
        parameters=(Parameter("theta"), Parameter("frozen", trainable=False)),
    )
    gradient, standard_error, covariance, confidence_radius = rust_uncertainty(
        plus_values.reshape(1, -1),
        minus_values.reshape(1, -1),
        plus_variances.reshape(1, -1),
        minus_variances.reshape(1, -1),
        plus_shots.reshape(1, -1),
        minus_shots.reshape(1, -1),
        coefficients,
        trainable,
    )

    np.testing.assert_allclose(gradient, python_result.gradient, atol=1e-12)
    np.testing.assert_allclose(standard_error, python_result.standard_error, atol=1e-12)
    np.testing.assert_allclose(covariance, python_result.covariance, atol=1e-12)
    np.testing.assert_allclose(confidence_radius, python_result.confidence_radius, atol=1e-12)
