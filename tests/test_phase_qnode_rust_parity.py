# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust parity tests for Phase-QNode differentials
"""Parity tests for Phase-QNode differentiable primitives exposed by PyO3."""

from __future__ import annotations

import numpy as np
import pytest

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


def _require_export(name: str):
    if not hasattr(engine, name):
        pytest.skip(f"scpn_quantum_engine is installed without {name}")
    return getattr(engine, name)


def _ry_state_and_derivative(
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_re = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)], dtype=np.float64)
    state_im = np.zeros(2, dtype=np.float64)
    derivatives_re = np.array(
        [[-0.5 * np.sin(theta / 2.0), 0.5 * np.cos(theta / 2.0)]],
        dtype=np.float64,
    )
    derivatives_im = np.zeros((1, 2), dtype=np.float64)
    return state_re, state_im, derivatives_re, derivatives_im


def _vector_objective(params: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.cos(params[0]) + 0.1 * np.sin(params[1]),
            np.sin(params[0]) - 0.25 * np.cos(params[1]),
        ],
        dtype=float,
    )


def _scalar_objective(params: np.ndarray) -> float:
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
