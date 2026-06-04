# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Parameter Shift
"""Tests for phase/param_shift.py parameter-shift VQE gradients."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import vqe_with_param_shift
from scpn_quantum_control.phase.param_shift import (
    ParamShiftVQEResult,
    parameter_shift_gradient,
    value_and_vqe_grad,
)
from scpn_quantum_control.phase.phase_vqe import PhaseVQE


def _finite_difference_gradient(
    objective,
    params: np.ndarray,
    *,
    step: float = 1e-6,
) -> np.ndarray:
    gradient = np.zeros_like(params, dtype=float)
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    return gradient


def test_phase_param_shift_module_exports_core_gradient() -> None:
    params = np.array([0.2, -0.4], dtype=float)

    def objective(values: np.ndarray) -> float:
        return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))

    grad = parameter_shift_gradient(objective, params, shift=np.pi / 2.0)
    expected = np.array([-np.sin(params[0]), 0.25 * np.cos(params[1])], dtype=float)
    np.testing.assert_allclose(grad, expected, atol=1e-12)


def test_phase_vqe_parameter_shift_matches_finite_difference() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    params = np.linspace(-0.3, 0.4, vqe.n_params, dtype=float)

    analytic = vqe.parameter_shift_gradient(params)
    finite_difference = _finite_difference_gradient(vqe._cost, params)

    np.testing.assert_allclose(analytic, finite_difference, atol=1e-5, rtol=1e-5)


def test_phase_vqe_structured_gradient_metadata() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    params = np.linspace(0.1, 0.4, vqe.n_params, dtype=float)

    result = value_and_vqe_grad(vqe, params)

    assert result.method == "parameter_shift"
    assert result.gradient.shape == (vqe.n_params,)
    assert result.evaluations == 1 + 2 * vqe.n_params
    assert np.isfinite(result.value)
    assert np.all(np.isfinite(result.gradient))


def test_phase_vqe_solve_uses_gradient_aware_optimizer_for_parameter_shift() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)

    result = vqe.solve(maxiter=12, seed=0, gradient_method="parameter_shift")

    assert result["gradient_method"] == "parameter_shift"
    assert result["optimizer"] == "L-BFGS-B"
    assert result["n_grad_evals"] > 0
    assert np.isfinite(result["ground_energy"])
    assert np.isfinite(result["gradient_norm"])


def test_phase_vqe_rejects_unknown_gradient_method() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)

    with pytest.raises(ValueError, match="gradient_method"):
        vqe.solve(maxiter=5, gradient_method="finite_difference")


def test_vqe_with_param_shift_tracks_non_increasing_best_energy() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]

    result = vqe_with_param_shift(
        K,
        omega,
        ansatz_reps=1,
        initial_params=np.linspace(-0.2, 0.2, 4, dtype=float),
        learning_rate=0.05,
        steps=8,
    )

    assert isinstance(result, ParamShiftVQEResult)
    assert result.best_energy <= result.initial_energy
    assert result.final_params.shape == (4,)
    assert result.best_params.shape == (4,)
    assert result.steps <= 8
    assert result.energy_gap is not None


def test_vqe_with_param_shift_supports_generic_callable_route() -> None:
    def objective(params: np.ndarray) -> float:
        return float(np.cos(params[0]) + np.sin(params[1]))

    result = vqe_with_param_shift(
        objective,
        n_params=2,
        initial_params=np.array([0.25, -0.5], dtype=float),
        learning_rate=0.1,
        steps=4,
    )

    as_dict = result.to_dict()
    assert result.best_energy <= result.initial_energy
    assert as_dict["energy"] == result.best_energy
    assert as_dict["optimal_params"].shape == (2,)
