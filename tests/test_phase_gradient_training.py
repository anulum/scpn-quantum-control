# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Gradient Training
"""Convergence evidence tests for parameter-shift phase training."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import (
    ParamShiftConvergenceDiagnostics,
    validate_param_shift_convergence,
    vqe_with_param_shift,
)


def test_parameter_shift_training_converges_on_known_trig_objective() -> None:
    def objective(params: np.ndarray) -> float:
        return float(np.cos(params[0]) + np.sin(params[1]))

    result = vqe_with_param_shift(
        objective,
        n_params=2,
        initial_params=np.array([2.7, -0.4], dtype=float),
        learning_rate=0.35,
        steps=28,
        tolerance=1e-9,
    )
    diagnostics = validate_param_shift_convergence(
        result,
        gradient_tolerance=0.08,
    )

    assert isinstance(diagnostics, ParamShiftConvergenceDiagnostics)
    assert diagnostics.monotone_energy
    assert diagnostics.best_improved
    assert diagnostics.accepted_steps > 0
    assert diagnostics.rejected_steps == 0
    assert diagnostics.parameter_shift_evaluations == result.steps * 5
    assert diagnostics.within_gradient_tolerance
    assert result.best_energy < -1.99


def test_parameter_shift_vqe_convergence_certificate_tracks_exact_gap() -> None:
    k_matrix = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]

    result = vqe_with_param_shift(
        k_matrix,
        omega,
        ansatz_reps=1,
        initial_params=np.linspace(-0.2, 0.2, 4, dtype=float),
        learning_rate=0.05,
        steps=8,
    )
    diagnostics = validate_param_shift_convergence(result)
    as_dict = result.to_dict()

    assert diagnostics.monotone_energy
    assert diagnostics.best_improved
    assert diagnostics.exact_energy is not None
    assert diagnostics.exact_gap == result.energy_gap
    assert diagnostics.parameter_shift_evaluations == result.steps * 9
    assert as_dict["parameter_shift_evaluations"] == diagnostics.parameter_shift_evaluations
    assert isinstance(as_dict["convergence_diagnostics"], dict)


def test_parameter_shift_convergence_validator_rejects_bad_thresholds() -> None:
    def objective(params: np.ndarray) -> float:
        return float(np.cos(params[0]))

    result = vqe_with_param_shift(
        objective,
        n_params=1,
        initial_params=np.array([0.2], dtype=float),
        steps=2,
    )

    with pytest.raises(ValueError, match="target_gap"):
        validate_param_shift_convergence(result, target_gap=-1.0)
