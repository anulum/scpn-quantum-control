# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — conformance of verified constrained co-design proposals
"""Proposals are adopted only after independent verification; failures keep the last plan.

Oracles: an exhaustive grid over the tested domain for the bounded quadratic,
constraint values recomputed in the test, and optimisers that deliberately
return infeasible, out-of-domain, unconverged or over-budget plans.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import minimize

from scpn_quantum_control.control.closed_loop_analysis import ClosedLoopExecutionPolicy
from scpn_quantum_control.control_stack_runtime_adapters import (
    CODESIGN_PROPOSAL_STATUSES,
    ConstrainedCodesignSession,
    ConstrainedOptimiserPort,
    OptimiserOutcome,
    PolicyGatedAdapterError,
)

Vector = NDArray[np.float64]
_TARGET = np.array([0.9, 0.8])
_WEIGHTS = np.array([[2.0, 0.4], [0.4, 1.0]])


def _objective(plan: Vector) -> float:
    """Return the bounded quadratic ``(x - t)^T W (x - t)``."""
    offset = plan - _TARGET
    return float(offset @ _WEIGHTS @ offset)


def _constraints(plan: Vector, disturbance: Vector | None) -> Vector:
    """Return ``x0 + x1 - 1`` and ``x0 - x1 - 0.5`` (feasible when ``<= 0``), plus a disturbance."""
    shift = np.zeros(2) if disturbance is None else disturbance
    return np.array([plan[0] + plan[1] - 1.0, plan[0] - plan[1] - 0.5]) + shift


def _session(
    held_out: Sequence[Vector] = (),
    initial_plan: Vector | None = None,
    policy: ClosedLoopExecutionPolicy | None = None,
    budget: int = 10_000,
) -> ConstrainedCodesignSession:
    """Build the bounded quadratic problem on the tested domain ``[0, 1]^2``."""
    return ConstrainedCodesignSession(
        _objective,
        _constraints,
        [0.0, 0.0],
        [1.0, 1.0],
        policy=ClosedLoopExecutionPolicy() if policy is None else policy,
        held_out_disturbances=held_out,
        initial_plan=initial_plan,
        evaluation_budget=budget,
    )


class _Slsqp:
    """SciPy SLSQP on the nominal problem; it never sees held-out disturbances."""

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.calls = 0

    def optimize(self, start: Vector) -> OptimiserOutcome:
        """Return SLSQP's plan from ``start``."""
        self.calls += 1
        solved = minimize(
            _objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            constraints=[{"type": "ineq", "fun": lambda x: -_constraints(x, None)}],
            options={"ftol": 1e-14, "maxiter": 500},
        )
        return OptimiserOutcome(np.asarray(solved.x), bool(solved.success), int(solved.nfev))


class _Fixed:
    """Return a fixed plan with a fixed convergence flag and cost, or raise."""

    def __init__(self, plan: Vector, converged: bool = True, evaluations: int = 1) -> None:
        """Store the outcome to return."""
        self.outcome = OptimiserOutcome(plan, converged, evaluations)
        self.calls = 0

    def optimize(self, start: Vector) -> OptimiserOutcome:
        """Return the stored outcome."""
        self.calls += 1
        return self.outcome


class _Failing:
    """Raise as a failed optimisation would."""

    def optimize(self, start: Vector) -> OptimiserOutcome:
        """Raise an arithmetic failure."""
        raise FloatingPointError("line search produced NaN")


def test_constrained_codesign_conformance_01_bounded_quadratic_matches_an_exhaustive_grid() -> (
    None
):
    """The accepted plan's objective equals the best feasible point of a 801 x 801 grid."""
    session = _session()
    proposal = session.propose(_Slsqp())
    assert proposal.status == "accepted" and proposal.adopted_plan == proposal.plan
    axis = np.linspace(0.0, 1.0, 801)
    x0, x1 = np.meshgrid(axis, axis, indexing="ij")
    feasible = (x0 + x1 - 1.0 <= 0.0) & (x0 - x1 - 0.5 <= 0.0)
    d0, d1 = x0 - _TARGET[0], x1 - _TARGET[1]
    grid = _WEIGHTS[0, 0] * d0**2 + 2 * _WEIGHTS[0, 1] * d0 * d1 + _WEIGHTS[1, 1] * d1**2
    best = float(np.min(grid[feasible]))
    assert proposal.objective is not None
    assert proposal.objective <= best + 1e-12
    assert best - proposal.objective < 5e-3  # the grid spacing bounds the grid's own error
    assert proposal.objective == pytest.approx(_objective(np.array(proposal.plan)), abs=0.0)


def test_constrained_codesign_conformance_02_every_accepted_proposal_satisfies_the_inequalities() -> (
    None
):
    """Over random plans, exactly the feasible in-domain ones are adopted."""
    rng = np.random.default_rng(4)
    session = _session(held_out=[np.array([0.05, 0.0])])
    accepted = 0
    for _ in range(200):
        plan = rng.uniform(-0.1, 1.1, size=2)
        proposal = session.propose(_Fixed(plan))
        recomputed = [_constraints(plan, None), _constraints(plan, np.array([0.05, 0.0]))]
        inside = bool(np.all((plan >= 0.0) & (plan <= 1.0)))
        feasible = inside and all(np.all(row <= 1e-9) for row in recomputed)
        assert (proposal.status == "accepted") is feasible
        if feasible:
            accepted += 1
            np.testing.assert_allclose(proposal.constraint_values, recomputed, atol=0.0)
            assert proposal.max_violation is not None and proposal.max_violation <= 1e-9
        assert proposal.hardware_actuation is False
    assert accepted > 20
    # A plan feasible nominally but not under the held-out disturbance is refused and said so.
    edge = session.propose(_Fixed(np.array([0.5, 0.48])))
    assert edge.status == "infeasible" and "held-out disturbance" in edge.reason


def test_constrained_codesign_conformance_03_infeasible_constraints_produce_no_applied_action() -> (
    None
):
    """With contradictory constraints nothing is adopted and the optimiser's plan stays a proposal."""
    contradictory = ConstrainedCodesignSession(
        _objective,
        lambda plan, disturbance: np.array([plan[0] - 0.2, 0.6 - plan[0]]),
        [0.0, 0.0],
        [1.0, 1.0],
        policy=ClosedLoopExecutionPolicy(),
    )
    for plan in (np.array([0.2, 0.5]), np.array([0.6, 0.5]), np.array([0.4, 0.5])):
        proposal = contradictory.propose(_Fixed(plan))
        assert proposal.status == "infeasible"
        assert proposal.adopted_plan is None and contradictory.current_plan is None
        assert proposal.hardware_actuation is False
    with pytest.raises(ValueError, match="never actuates hardware"):
        replace(proposal, hardware_actuation=True)
    with pytest.raises(PolicyGatedAdapterError):
        _session(policy=ClosedLoopExecutionPolicy(allow_hardware=True, live_ticket="x")).propose(
            _Slsqp()
        )


def test_constrained_codesign_conformance_04_failures_restore_the_previous_accepted_plan() -> None:
    """A failed, unconverged, out-of-domain or over-budget proposal keeps the last accepted plan."""
    session = _session(initial_plan=np.array([0.2, 0.3]), budget=50)
    assert session.current_plan == (0.2, 0.3)
    accepted = session.propose(_Fixed(np.array([0.4, 0.4])))
    assert accepted.status == "accepted" and session.current_plan == (0.4, 0.4)
    failures: list[tuple[ConstrainedOptimiserPort, str, tuple[float, float] | None]] = [
        (_Failing(), "nonconvergent", None),
        (_Fixed(np.array([0.1, 0.1]), converged=False), "nonconvergent", (0.1, 0.1)),
        (_Fixed(np.array([1.2, 0.0])), "out_of_domain", (1.2, 0.0)),
        (_Fixed(np.array([np.nan, 0.0])), "out_of_domain", None),
        (_Fixed(np.array([0.1, 0.1]), evaluations=51), "over_budget", (0.1, 0.1)),
        (_Fixed(np.array([0.9, 0.05])), "infeasible", (0.9, 0.05)),
    ]
    for optimiser, status, plan in failures:
        proposal = session.propose(optimiser)
        assert proposal.status == status
        assert proposal.adopted_plan == (0.4, 0.4) == session.current_plan
        if plan is not None and status != "out_of_domain":
            assert proposal.plan == plan
    assert set(CODESIGN_PROPOSAL_STATUSES) == {
        "accepted",
        "infeasible",
        "nonconvergent",
        "out_of_domain",
        "over_budget",
    }


def test_session_rejects_malformed_problems() -> None:
    """Bounds, tolerance, budget, an inadmissible initial plan and bad callbacks are refused."""
    with pytest.raises(ValueError, match="lower < upper"):
        ConstrainedCodesignSession(
            _objective, _constraints, [1.0, 0.0], [0.0, 1.0], policy=ClosedLoopExecutionPolicy()
        )
    with pytest.raises(ValueError, match="tolerance"):
        ConstrainedCodesignSession(
            _objective,
            _constraints,
            [0.0, 0.0],
            [1.0, 1.0],
            policy=ClosedLoopExecutionPolicy(),
            tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="evaluation_budget"):
        _session(budget=0)
    with pytest.raises(ValueError, match="initial plan is not admissible"):
        _session(initial_plan=np.array([0.9, 0.9]))
    broken = ConstrainedCodesignSession(
        lambda plan: float("nan"),
        _constraints,
        [0.0, 0.0],
        [1.0, 1.0],
        policy=ClosedLoopExecutionPolicy(),
    )
    with pytest.raises(ValueError, match="objective must return a finite value"):
        broken.propose(_Fixed(np.array([0.2, 0.2])))
    shapeless = ConstrainedCodesignSession(
        _objective,
        lambda plan, disturbance: np.zeros((2, 2)),
        [0.0, 0.0],
        [1.0, 1.0],
        policy=ClosedLoopExecutionPolicy(),
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        shapeless.propose(_Fixed(np.array([0.2, 0.2])))
    assert _session().domain == ((0.0, 1.0), (0.0, 1.0))
