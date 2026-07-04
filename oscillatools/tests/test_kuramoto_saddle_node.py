# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the Kuramoto saddle-node (fold) locator
"""Tests for :mod:`kuramoto_saddle_node`.

The locator is anchored three independent ways: for two oscillators it must
recover the closed-form fold ``K = |Δω|/2`` to Newton precision; for a larger
system the located point must be a genuine saddle-node (a valid locked state
whose transverse stability eigenvalue vanishes, cross-checked against the
independent :mod:`kuramoto_stability_spectrum`) that sits exactly on the boundary
of locked-state existence; and the analytic fold Jacobian must match a central
finite-difference of the defining residual — the exact derivative the direct
locator depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel import kuramoto_saddle_node as sn
from oscillatools.accel.kuramoto_stability_spectrum import stability_spectrum

_PAIR = np.array([[0.0, 1.0], [1.0, 0.0]])
_TRIANGLE = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


# --------------------------------------------------------------------------- #
# Two-oscillator analytic anchor: K_fold = |omega_0 - omega_1| / 2
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("omega0", "omega1"),
    [(0.7, -0.3), (1.0, 0.0), (0.4, 0.1), (-0.2, 0.6)],
)
def test_two_oscillator_fold_matches_closed_form(omega0: float, omega1: float) -> None:
    omega = np.array([omega0, omega1])
    fold = sn.locate_saddle_node(omega, _PAIR, initial_coupling=2.0)
    expected = abs(omega0 - omega1) / 2.0
    assert fold.converged
    assert fold.critical_coupling == pytest.approx(expected, abs=1e-9)
    # gauge and collective frequency
    assert fold.phases[0] == 0.0
    assert fold.collective_frequency == pytest.approx((omega0 + omega1) / 2.0, abs=1e-9)
    assert np.linalg.norm(fold.null_vector) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Larger system: a genuine saddle-node on the existence boundary
# --------------------------------------------------------------------------- #


def test_three_oscillator_fold_is_a_genuine_saddle_node() -> None:
    omega = np.array([1.0, 0.1, -1.1])
    fold = sn.locate_saddle_node(omega, _TRIANGLE, initial_coupling=3.0)
    assert fold.converged
    assert fold.critical_coupling > 0.0
    coupling = fold.critical_coupling * _TRIANGLE
    # (a) it is a valid phase-locked state
    residual = (
        omega + sn.networked_kuramoto_force(fold.phases, coupling) - fold.collective_frequency
    )
    assert np.max(np.abs(residual)) < 1e-9
    # (b) the transverse stability eigenvalue vanishes — the marginal fold mode
    spectrum = stability_spectrum(fold.phases, coupling)
    assert abs(spectrum.leading_nontrivial_eigenvalue.real) < 1e-7


def test_fold_sits_on_the_existence_boundary() -> None:
    omega = np.array([1.0, 0.1, -1.1])
    fold = sn.locate_saddle_node(omega, _TRIANGLE, initial_coupling=3.0)
    # A locked state exists above the fold (seeded from the aligned state, as the
    # locator's own continuation does) ...
    aligned = np.concatenate([np.zeros(omega.shape[0] - 1), [float(np.mean(omega))]])
    above = sn._solve_locked_state(
        omega,
        _TRIANGLE,
        fold.critical_coupling + 0.1,
        aligned,
        tolerance=1e-11,
        max_iterations=100,
    )
    # ... but not below it (seeded from the folding state, the nearest locked state).
    fold_seed = np.concatenate([fold.phases[1:], [fold.collective_frequency]])
    below = sn._solve_locked_state(
        omega,
        _TRIANGLE,
        fold.critical_coupling - 0.01,
        fold_seed,
        tolerance=1e-11,
        max_iterations=100,
    )
    assert above is not None
    assert below is None


def test_initial_phases_seed_is_accepted() -> None:
    omega = np.array([0.7, -0.3])
    fold = sn.locate_saddle_node(
        omega, _PAIR, initial_coupling=2.0, initial_phases=np.array([0.0, -0.4])
    )
    assert fold.converged
    assert fold.critical_coupling == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# Building blocks: residual + exact Jacobian
# --------------------------------------------------------------------------- #


def _solution_vector(omega: np.ndarray, structure: np.ndarray, k0: float) -> np.ndarray:
    fold = sn.locate_saddle_node(omega, structure, initial_coupling=k0)
    return np.concatenate(
        [fold.phases[1:], [fold.collective_frequency], fold.null_vector, [fold.critical_coupling]]
    ), fold.null_vector / np.linalg.norm(fold.null_vector)


def test_defining_residual_vanishes_at_the_fold() -> None:
    omega = np.array([1.0, 0.1, -1.1])
    n = omega.shape[0]
    x, cvec = _solution_vector(omega, _TRIANGLE, 3.0)
    residual = sn.fold_defining_residual(x, omega, _TRIANGLE, cvec)
    # The physical fold conditions — locked state (F - Omega = 0) and singular
    # reduced Jacobian (J_R v = 0) — vanish; the trailing entry is the
    # null-vector scaling convention, not a physical condition.
    assert np.max(np.abs(residual[: 2 * n])) < 1e-8


def test_defining_jacobian_matches_finite_difference() -> None:
    omega = np.array([1.0, 0.1, -1.1])
    # evaluate away from the solution (a generic point) so every block is exercised
    x, cvec = _solution_vector(omega, _TRIANGLE, 3.0)
    x = x + 0.05
    analytic = sn.fold_defining_jacobian(x, omega, _TRIANGLE, cvec)
    fd = np.zeros_like(analytic)
    eps = 1e-6
    for k in range(x.size):
        xp = x.copy()
        xp[k] += eps
        xm = x.copy()
        xm[k] -= eps
        fd[:, k] = (
            sn.fold_defining_residual(xp, omega, _TRIANGLE, cvec)
            - sn.fold_defining_residual(xm, omega, _TRIANGLE, cvec)
        ) / (2 * eps)
    assert np.max(np.abs(analytic - fd)) < 1e-8


# --------------------------------------------------------------------------- #
# Input validation + failure modes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"omega": np.zeros((2, 2)), "structure": _PAIR}, "one-dimensional"),
        ({"omega": np.array([1.0]), "structure": np.array([[0.0]])}, "at least two"),
        ({"omega": np.array([1.0, 0.0]), "structure": _TRIANGLE}, "structure must be"),
    ],
)
def test_shape_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sn.locate_saddle_node(initial_coupling=1.0, **kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"initial_coupling": 0.0}, "initial_coupling must be positive"),
        (
            {"initial_coupling": 1.0, "continuation_step": 0.0},
            "continuation_step must be positive",
        ),
        (
            {"initial_coupling": 1.0, "initial_phases": np.zeros(3)},
            "initial_phases must be",
        ),
    ],
)
def test_parameter_validation(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sn.locate_saddle_node(np.array([0.7, -0.3]), _PAIR, **overrides)


def test_no_locked_state_at_initial_coupling_raises() -> None:
    # start below the two-oscillator fold (0.5) => no locked state to continue from
    with pytest.raises(ValueError, match="no phase-locked state"):
        sn.locate_saddle_node(np.array([0.7, -0.3]), _PAIR, initial_coupling=0.3)


def test_persistent_lock_without_fold_raises() -> None:
    # identical frequencies lock at every positive coupling => no fold down to zero
    omega = np.array([0.4, 0.4])
    with pytest.raises(RuntimeError, match="without bracketing a fold"):
        sn.locate_saddle_node(omega, _PAIR, initial_coupling=1.0, continuation_step=0.2)


# --------------------------------------------------------------------------- #
# Helper units (fold Newton non-convergence; locked-state loss)
# --------------------------------------------------------------------------- #


def test_newton_fold_reports_non_convergence() -> None:
    omega = np.array([1.0, 0.1, -1.1])
    x, cvec = _solution_vector(omega, _TRIANGLE, 3.0)
    seed = x + 0.4  # far seed, one step is not enough
    _, iterations, _, converged = sn._newton_fold(
        seed, omega, _TRIANGLE, cvec, tolerance=1e-12, max_iterations=1
    )
    assert converged is False
    assert iterations == 1


def test_solve_locked_state_returns_none_on_singular_step(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: object, **_k: object) -> np.ndarray:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(sn.np.linalg, "solve", _raise)
    lost = sn._solve_locked_state(
        np.array([0.7, -0.3]), _PAIR, 1.0, np.array([0.1, 0.2]), tolerance=1e-11, max_iterations=10
    )
    assert lost is None
