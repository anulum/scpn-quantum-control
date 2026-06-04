# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Gradient Backend Planner
"""Tests for phase/gradient_backend.py quantum-gradient planning."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    plan_parameter_shift_shots,
    plan_quantum_gradient_backend,
)
from scpn_quantum_control.phase.gradient_backend import quantum_gradient_backend_capability
from scpn_quantum_control.phase.param_shift import parameter_shift_gradient_with_uncertainty


def test_statevector_backend_auto_selects_deterministic_parameter_shift() -> None:
    plan = plan_quantum_gradient_backend("statevector", n_params=3)

    assert plan.supported
    assert not plan.fail_closed
    assert plan.method == "parameter_shift"
    assert plan.evaluations == 6
    assert plan.shots is None
    assert not plan.finite_shot


def test_finite_shot_backend_auto_selects_stochastic_parameter_shift() -> None:
    plan = plan_quantum_gradient_backend(
        "qasm_simulator",
        n_params=4,
        shots=2048,
        seed=0,
    )

    assert plan.supported
    assert plan.method == "stochastic_parameter_shift"
    assert plan.evaluations == 8
    assert plan.shots == 2048
    assert plan.seed == 0
    assert plan.confidence_level == 0.95


def test_hardware_backend_fails_closed_without_policy_approval() -> None:
    plan = plan_quantum_gradient_backend("ibm_quantum", n_params=2, shots=1024)

    assert plan.fail_closed
    assert plan.method == "unsupported"
    assert plan.requires_hardware_approval
    assert "hardware gradient execution requires explicit hardware policy approval" in plan.reasons


def test_unknown_backend_is_unsupported_with_safe_alternatives() -> None:
    capability = quantum_gradient_backend_capability("new_vendor_backend")
    plan = plan_quantum_gradient_backend("new_vendor_backend", n_params=2)

    assert capability.family == "unknown"
    assert plan.fail_closed
    assert "statevector_simulator" in plan.alternatives
    assert "finite_shot_simulator" in plan.alternatives


def test_planner_rejects_invalid_method_and_shape_controls() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        plan_quantum_gradient_backend("statevector", n_params=2, method="adjoint")
    with pytest.raises(ValueError, match="n_params"):
        plan_quantum_gradient_backend("statevector", n_params=0)
    with pytest.raises(ValueError, match="shots"):
        plan_quantum_gradient_backend("qasm_simulator", n_params=2, shots=0)


def test_parameter_shift_uncertainty_propagates_finite_shot_noise() -> None:
    result = parameter_shift_gradient_with_uncertainty(
        plus_values=np.array([1.2, -0.3], dtype=float),
        minus_values=np.array([0.8, -0.7], dtype=float),
        plus_variances=np.array([0.04, 0.09], dtype=float),
        minus_variances=np.array([0.04, 0.09], dtype=float),
        shots=400,
        value=0.5,
    )

    np.testing.assert_allclose(result.gradient, np.array([0.2, 0.2], dtype=float))
    assert result.method == "parameter_shift_shot_noise"
    assert result.shots.shape == (2, 2)
    assert np.all(result.standard_error > 0.0)
    assert np.all(result.confidence_radius >= result.standard_error)


def test_parameter_shift_uncertainty_fails_closed_for_hardware_backend() -> None:
    with pytest.raises(ValueError, match="hardware gradient execution requires"):
        parameter_shift_gradient_with_uncertainty(
            plus_values=np.array([1.0], dtype=float),
            minus_values=np.array([0.5], dtype=float),
            plus_variances=np.array([0.1], dtype=float),
            minus_variances=np.array([0.1], dtype=float),
            shots=128,
            backend="hardware",
        )


def test_parameter_shift_shot_allocation_bounds_variance_target() -> None:
    allocation = plan_parameter_shift_shots(
        plus_variances=np.array([0.04, 0.09], dtype=float),
        minus_variances=np.array([0.04, 0.09], dtype=float),
        target_standard_error=0.02,
        min_shots=10,
        max_shots_per_evaluation=1000,
    )

    assert allocation.shots.shape == (2, 2)
    assert np.all(allocation.shots >= 10)
    assert np.all(allocation.shots <= 1000)
    assert np.all(allocation.predicted_standard_error <= 0.02)
