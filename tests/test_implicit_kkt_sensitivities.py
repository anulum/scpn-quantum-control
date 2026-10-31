# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — qualification of implicit root and KKT sensitivities
"""Implicit derivatives are issued only at a verified solution with a fixed active set.

Oracles do not share the implicit derivation: closed-form roots, an
independent constrained solver (SciPy SLSQP) re-solved at perturbed
parameters, one-sided differences across an active-set switch, and the
forward-mode derivative of unrolled Newton iterations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import minimize

from scpn_quantum_control.differentiable import (
    IMPLICIT_DERIVATIVE_KIND,
    ImplicitDerivativeRefusal,
    KktSensitivity,
    implicit_kkt_sensitivity,
    implicit_root_sensitivity,
)

Vector = NDArray[np.float64]


def _square_root_residual(x: Vector, p: Vector) -> Vector:
    """``F(x, p) = x^2 - p``."""
    return np.asarray(x**2 - p, dtype=np.float64)


def _square_root_state(x: Vector, p: Vector) -> Vector:
    """``dF/dx = 2x``."""
    return np.asarray([[2.0 * x[0]]], dtype=np.float64)


def _square_root_parameter(x: Vector, p: Vector) -> Vector:
    """``dF/dp = -1``."""
    return np.asarray([[-1.0]], dtype=np.float64)


def _root(x: float, p: float, **settings: float) -> float:
    """Return the implicit ``dx/dp`` of ``x^2 = p`` at ``x``."""
    result = implicit_root_sensitivity(
        _square_root_residual,
        _square_root_state,
        _square_root_parameter,
        [x],
        [p],
        **settings,
    )
    return float(result.sensitivity[0, 0])


def test_implicit_kkt_sensitivities_01_root_on_the_positive_branch() -> None:
    """``x^2 - p = 0`` gives ``dx/dp = 1/(2 sqrt p)`` at the verified positive root."""
    for p in (0.25, 2.25, 9.0):
        result = implicit_root_sensitivity(
            _square_root_residual,
            _square_root_state,
            _square_root_parameter,
            [np.sqrt(p)],
            [p],
        )
        assert result.sensitivity[0, 0] == pytest.approx(1.0 / (2.0 * np.sqrt(p)), rel=1e-14)
        assert result.residual_norm <= 1e-10 and result.kind == IMPLICIT_DERIVATIVE_KIND
    # An unverified point is refused, whatever its Jacobian says.
    with pytest.raises(ImplicitDerivativeRefusal, match="not a verified root") as refused:
        _root(1.5 + 1e-6, 2.25)
    assert refused.value.reason == "incomplete_solve"


def test_implicit_kkt_sensitivities_01_unrolled_newton_is_a_different_object() -> None:
    """Unrolled Newton derivatives depend on the iteration count; the implicit one does not.

    ``x_{k+1} = (x_k + p / x_k) / 2`` differentiated forward in ``p`` gives
    the unrolled derivative. After two iterations it is wrong and the iterate
    is refused; once converged the two agree.
    """
    p = 2.25
    x, dx = 1.0, 0.0
    unrolled: list[tuple[float, float]] = []
    for _ in range(8):
        x, dx = (x + p / x) / 2.0, (dx * (1.0 - p / x**2) + 1.0 / x) / 2.0
        unrolled.append((x, dx))
    early_x, early_dx = unrolled[1]
    assert abs(early_dx - 1.0 / 3.0) > 1e-3
    with pytest.raises(ImplicitDerivativeRefusal) as refused:
        _root(early_x, p)
    assert refused.value.reason == "incomplete_solve"
    final_x, final_dx = unrolled[-1]
    assert _root(final_x, p) == pytest.approx(final_dx, rel=1e-12)
    assert _root(final_x, p) == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_implicit_kkt_sensitivities_02_singular_root_refuses() -> None:
    """At ``p = 0`` the root is a double root; ``dF/dx = 0`` and nothing is issued."""
    with pytest.raises(ImplicitDerivativeRefusal, match="condition number") as refused:
        _root(0.0, 0.0)
    assert refused.value.reason == "singular"
    # A nearly dependent two-equation system is refused by its condition number.
    with pytest.raises(ImplicitDerivativeRefusal) as near:
        implicit_root_sensitivity(
            lambda x, p: np.asarray([x[0] + x[1] - p[0], x[0] + x[1] - p[0]]),
            lambda x, p: np.asarray([[1.0, 1.0], [1.0, 1.0 + 1e-14]]),
            lambda x, p: np.asarray([[-1.0], [-1.0]]),
            [0.5, 0.5],
            [1.0],
        )
    assert near.value.reason == "singular"


def _projection(p: Vector) -> tuple[Vector, Vector]:
    """Solve ``min ½|x - p|^2`` s.t. ``x1 + x2 <= 1`` and ``x1 >= 0`` with SLSQP.

    Returns
    -------
    tuple
        ``x*`` and the inequality multipliers recovered from stationarity.

    """
    solved = minimize(
        lambda x: 0.5 * float(np.sum((x - p) ** 2)),
        x0=np.zeros(2),
        jac=lambda x: x - p,
        method="SLSQP",
        constraints=[
            {"type": "ineq", "fun": lambda x: 1.0 - x[0] - x[1], "jac": lambda x: -np.ones(2)},
            {"type": "ineq", "fun": lambda x: x[0], "jac": lambda x: np.array([1.0, 0.0])},
        ],
        options={"ftol": 1e-15, "maxiter": 500},
    )
    assert solved.success
    x = np.asarray(solved.x, dtype=np.float64)
    gradients = np.array([[1.0, 1.0], [-1.0, 0.0]])
    values = gradients @ x - np.array([1.0, 0.0])
    active = values > -1e-8
    lam = np.zeros(2)
    if np.any(active):
        lam[active] = np.linalg.lstsq(gradients[active].T, -(x - p), rcond=None)[0]
    return x, lam


Hessian = NDArray[np.float64] | Callable[[Vector], Vector]


def _kkt_at(p: Vector, x: Vector, lam: Vector, hessian: Hessian | None = None) -> KktSensitivity:
    """Implicit KKT sensitivity of the projection problem at ``(x, lambda)``."""
    gradients = np.array([[1.0, 1.0], [-1.0, 0.0]])
    return implicit_kkt_sensitivity(
        x,
        x - p,
        np.eye(2) if hessian is None else hessian,
        -np.eye(2),
        constraint_values=gradients @ x - np.array([1.0, 0.0]),
        constraint_jacobian=gradients,
        constraint_parameter_jacobian=np.zeros((2, 2)),
        multipliers=lam,
    )


def test_implicit_kkt_sensitivities_03_perturb_and_resolve_inside_the_active_set() -> None:
    """The implicit derivative equals re-solving with SLSQP at ``p ± h`` while the active set holds."""
    p = np.array([1.0, 0.8])
    x, lam = _projection(p)
    np.testing.assert_allclose(x, [0.6, 0.4], atol=1e-9)
    result = _kkt_at(p, x, lam)
    assert result.active == (True, False)
    step = 1e-4
    for column in range(2):
        shift = np.zeros(2)
        shift[column] = step
        (x_plus, lam_plus), (x_minus, lam_minus) = _projection(p + shift), _projection(p - shift)
        np.testing.assert_allclose(
            result.sensitivity[:, column], (x_plus - x_minus) / (2 * step), atol=1e-6
        )
        np.testing.assert_allclose(
            result.multiplier_sensitivity[:, column],
            (lam_plus - lam_minus) / (2 * step),
            atol=1e-6,
        )
    np.testing.assert_allclose(result.sensitivity, np.eye(2) - 0.5, atol=1e-12)
    # The same system given as a Hessian-vector product gives the same derivative.
    product = _kkt_at(p, x, lam, hessian=lambda v: np.asarray(v, dtype=np.float64))
    np.testing.assert_allclose(product.sensitivity, result.sensitivity, atol=0.0)


def test_implicit_kkt_sensitivities_03_equality_constraint() -> None:
    """``min ½|x - p|^2`` s.t. ``x1 = x2`` gives ``dx/dp = ½ [[1, 1], [1, 1]]``."""
    p = np.array([0.3, 1.1])
    x = np.full(2, p.mean())
    mu = np.array([x[0] - p[0]])
    result = implicit_kkt_sensitivity(
        x,
        x - p,
        np.eye(2),
        -np.eye(2),
        equality_values=[x[0] - x[1]],
        equality_jacobian=[[1.0, -1.0]],
        equality_parameter_jacobian=np.zeros((1, 2)),
        equality_multipliers=-mu,
    )
    np.testing.assert_allclose(result.sensitivity, np.full((2, 2), 0.5), atol=1e-12)
    np.testing.assert_allclose(result.equality_multiplier_sensitivity, [[0.5, -0.5]], atol=1e-12)
    assert result.reduced_hessian_min_eigenvalue == pytest.approx(1.0)


def test_implicit_kkt_sensitivities_04_active_set_switch_invalidates_the_derivative() -> None:
    """On the switching surface the one-sided derivatives differ, so nothing is issued."""
    p = np.array([0.6, 0.4])
    x, lam = _projection(p)
    with pytest.raises(ImplicitDerivativeRefusal, match="weakly active") as refused:
        _kkt_at(p, x, np.zeros(2))
    assert refused.value.reason == "active_set_switch"
    step = 1e-4
    right = (_projection(p + [step, 0.0])[0] - x) / step
    left = (x - _projection(p - [step, 0.0])[0]) / step
    assert right[0] == pytest.approx(0.5, abs=1e-5) and left[0] == pytest.approx(1.0, abs=1e-5)


def test_implicit_kkt_sensitivities_04_step_limit_marks_where_the_derivative_stops() -> None:
    """Inside ``step_limit`` the linear prediction is exact for this QP; beyond it the set switches."""
    p = np.array([1.0, 0.8])
    x, lam = _projection(p)
    result = _kkt_at(p, x, lam)
    direction = np.array([-1.0, 0.0])
    limit = result.step_limit(direction)
    assert limit == pytest.approx(0.8, rel=1e-9)
    for fraction, holds in ((0.9, True), (1.2, False)):
        moved = _projection(p + fraction * limit * direction)[0]
        predicted = x + result.sensitivity @ (fraction * limit * direction)
        assert bool(np.allclose(moved, predicted, atol=1e-7)) is holds
    # Along +p2 the multiplier grows; the bound x1 >= 0 is reached at 0.6 / 0.5.
    assert result.step_limit([0.0, 1.0]) == pytest.approx(1.2, rel=1e-9)
    assert _projection(p + [0.0, 1.2])[0][0] == pytest.approx(0.0, abs=1e-8)
    assert result.step_limit([1.0, 1.0]) == float("inf")
    for bad in ([0.0, 0.0], [1.0], [np.nan, 1.0]):
        with pytest.raises(ValueError, match="direction"):
            result.step_limit(bad)


def test_kkt_refusals_name_their_reason() -> None:
    """Incomplete, dependent, non-unique, saddle and mislabelled solutions are each refused."""
    p = np.array([1.0, 0.8])
    x, lam = _projection(p)
    cases: list[tuple[str, Callable[[], KktSensitivity]]] = [
        ("incomplete_solve", lambda: _kkt_at(p, x + 1e-5, lam)),
        ("incomplete_solve", lambda: _kkt_at(p, x, -lam)),
        (
            "active_set_switch",
            lambda: implicit_kkt_sensitivity(
                [0.0],
                [-1e-4],
                [[1.0]],
                [[1.0]],
                constraint_values=[-1e-6],
                constraint_jacobian=[[1.0]],
                constraint_parameter_jacobian=[[0.0]],
                multipliers=[1e-4],
            ),
        ),
        (
            "singular",
            lambda: implicit_kkt_sensitivity(
                x,
                x - p,
                np.eye(2),
                -np.eye(2),
                constraint_values=[0.0, 0.0],
                constraint_jacobian=[[1.0, 1.0], [2.0, 2.0]],
                constraint_parameter_jacobian=np.zeros((2, 2)),
                multipliers=[0.2, 0.1],
            ),
        ),
        (
            "not_a_strict_minimum",
            lambda: implicit_kkt_sensitivity(
                [0.0, 0.0], [0.0, 0.0], np.diag([1.0, -1.0]), np.eye(2)
            ),
        ),
        (
            "not_a_strict_minimum",
            lambda: implicit_kkt_sensitivity(
                [0.0, 0.0], [0.0, 0.0], np.diag([1.0, 0.0]), np.eye(2)
            ),
        ),
    ]
    for reason, call in cases:
        with pytest.raises(ImplicitDerivativeRefusal) as refused:
            call()
        assert refused.value.reason == reason
    # Zero curvature along a direction that an active constraint removes is admitted.
    pinned = implicit_kkt_sensitivity(
        [0.0, 1.0],
        [0.0, -1.0],
        np.diag([1.0, 0.0]),
        np.eye(2),
        constraint_values=[0.0],
        constraint_jacobian=[[0.0, 1.0]],
        constraint_parameter_jacobian=[[0.0, 0.0]],
        multipliers=[1.0],
    )
    np.testing.assert_allclose(pinned.sensitivity, [[-1.0, 0.0], [0.0, 0.0]], atol=1e-12)


def test_kkt_pinned_optimum_and_badly_scaled_constraint() -> None:
    """Two active constraints pin ``x`` in 2-D; a tiny active gradient makes the KKT matrix singular."""
    # min ½|x - p|^2 with x1 <= p3 and x2 <= p3 both active: x* = (p3, p3).
    pinned = implicit_kkt_sensitivity(
        [0.5, 0.5],
        [-0.5, -0.5],
        np.eye(2),
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        constraint_values=[0.0, 0.0],
        constraint_jacobian=np.eye(2),
        constraint_parameter_jacobian=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        multipliers=[0.5, 0.5],
    )
    np.testing.assert_allclose(pinned.sensitivity, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(pinned.multiplier_sensitivity, [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    assert pinned.reduced_hessian_min_eigenvalue == float("inf")
    with pytest.raises(ImplicitDerivativeRefusal, match="KKT matrix") as refused:
        implicit_kkt_sensitivity(
            [0.0, 0.0],
            [-1e-13, 0.0],
            np.eye(2),
            np.eye(2),
            constraint_values=[0.0],
            constraint_jacobian=[[1e-13, 0.0]],
            constraint_parameter_jacobian=[[0.0, 0.0]],
            multipliers=[1.0],
        )
    assert refused.value.reason == "singular"


def test_kkt_rejects_malformed_inputs() -> None:
    """Shapes, partial constraint blocks, asymmetric Hessians and bad tolerances are refused."""
    x, grad = np.zeros(2), np.zeros(2)
    with pytest.raises(ValueError, match="given together"):
        implicit_kkt_sensitivity(x, grad, np.eye(2), np.eye(2), constraint_values=[0.0])
    with pytest.raises(ValueError, match="symmetric"):
        implicit_kkt_sensitivity(x, grad, np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(2))
    with pytest.raises(ValueError, match="vectors of length 2"):
        implicit_kkt_sensitivity(x, grad, lambda v: np.zeros(3), np.eye(2))
    with pytest.raises(ValueError, match="objective_gradient"):
        implicit_kkt_sensitivity(x, np.zeros(3), np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="cross_derivative"):
        implicit_kkt_sensitivity(x, grad, np.eye(2), np.zeros((3, 1)))
    with pytest.raises(ValueError, match="non-empty"):
        implicit_kkt_sensitivity([], [], np.zeros((0, 0)), np.zeros((0, 1)))
    with pytest.raises(ValueError, match="multipliers must have length"):
        implicit_kkt_sensitivity(
            x,
            grad,
            np.eye(2),
            np.eye(2),
            constraint_values=[-1.0],
            constraint_jacobian=[[1.0, 0.0]],
            constraint_parameter_jacobian=[[0.0, 0.0]],
            multipliers=[0.0, 0.0],
        )
    with pytest.raises(ValueError, match="kkt_tolerance"):
        implicit_kkt_sensitivity(x, grad, np.eye(2), np.eye(2), kkt_tolerance=0.0)
    with pytest.raises(ValueError, match="active_tolerance"):
        implicit_kkt_sensitivity(x, grad, np.eye(2), np.eye(2), active_tolerance=0.0)
    with pytest.raises(ValueError, match="rcond"):
        implicit_kkt_sensitivity(x, grad, np.eye(2), np.eye(2), rcond=0.0)
    one = implicit_kkt_sensitivity(x, grad, np.eye(2), [1.0, 2.0])
    np.testing.assert_allclose(one.sensitivity, [[-1.0], [-2.0]])
    assert one.reduced_hessian_min_eigenvalue == pytest.approx(1.0) and one.active == ()


def test_root_rejects_malformed_inputs() -> None:
    """Empty, wrongly shaped and non-finite inputs are refused before any solve."""
    with pytest.raises(ValueError, match="non-empty"):
        implicit_root_sensitivity(
            _square_root_residual, _square_root_state, _square_root_parameter, [], [1.0]
        )
    with pytest.raises(ValueError, match="residual must return"):
        implicit_root_sensitivity(
            lambda x, p: np.zeros(2), _square_root_state, _square_root_parameter, [1.0], [1.0]
        )
    with pytest.raises(ValueError, match="state_jacobian"):
        implicit_root_sensitivity(
            _square_root_residual,
            lambda x, p: np.zeros((2, 2)),
            _square_root_parameter,
            [1.0],
            [1.0],
        )
    with pytest.raises(ValueError, match="finite"):
        implicit_root_sensitivity(
            _square_root_residual, _square_root_state, _square_root_parameter, [np.nan], [1.0]
        )
    with pytest.raises(ImplicitDerivativeRefusal):
        implicit_root_sensitivity(
            lambda x, p: np.full(1, np.nan),
            _square_root_state,
            _square_root_parameter,
            [1.0],
            [1.0],
        )
    with pytest.raises(ValueError, match="residual_tolerance"):
        _root(1.0, 1.0, residual_tolerance=-1.0)
    with pytest.raises(ValueError, match="rcond"):
        _root(1.0, 1.0, rcond=-1.0)
