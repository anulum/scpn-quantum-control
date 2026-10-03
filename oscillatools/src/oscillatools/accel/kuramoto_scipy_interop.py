# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SciPy solve_ivp interop for the Kuramoto system object
r"""Run a :class:`KuramotoSystem` through the SciPy ``solve_ivp`` solver stack.

The adoption discipline for a scientific dynamical-systems package is *compose,
don't reimplement*: rather than ship yet another adaptive integrator, expose the
system so the established SciPy solvers drive it. A :class:`KuramotoSystem`
already carries the rule ``f(u, p, t)`` and — for the shipped topologies — its
analytic state Jacobian ``∂f/∂θ``, which are exactly the two callables
:func:`scipy.integrate.solve_ivp` consumes (``fun`` and the optional ``jac`` that
the implicit BDF/Radau/LSODA methods use for stiff regimes). This module adapts
between the two:

* :func:`kuramoto_ode_rhs` turns a system into the ``fun(t, y)`` right-hand side;
* :func:`kuramoto_ode_jacobian` turns it into the ``jac(t, y)`` Jacobian (or
  ``None`` when the system has no analytic Jacobian);
* :func:`solve_kuramoto_ivp` runs ``solve_ivp`` from the system's current state
  and returns a :class:`KuramotoIvpSolution`.

The reverse direction — accepting an arbitrary externally-defined rule — needs no
adapter: the general :class:`KuramotoSystem` constructor already takes any
``f(u, p, t)`` (and optional Jacobian), so a foreign ODE right-hand side becomes a
first-class system object directly.

``solve_kuramoto_ivp`` is a pure query: it reads the system's current state as the
initial condition and does not advance or mutate the system (unlike
:meth:`KuramotoSystem.step` / :meth:`KuramotoSystem.trajectory`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .kuramoto_system import KuramotoSystem

#: A SciPy ``solve_ivp`` right-hand side ``fun(t, y)`` returning ``dy/dt``.
OdeRightHandSide = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]

#: A SciPy ``solve_ivp`` Jacobian ``jac(t, y)`` returning the ``(N, N)`` matrix.
OdeJacobian = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class KuramotoIvpSolution:
    """The outcome of integrating a :class:`KuramotoSystem` with ``solve_ivp``.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(T,)`` solution times returned by the solver.
    phases : numpy.ndarray
        The ``(T, N)`` phase trajectory (transposed from the SciPy ``(N, T)``
        layout so it matches :meth:`KuramotoSystem.trajectory`).
    success : bool
        SciPy success: either the interval completed or a terminal event occurred.
    status : int
        SciPy status: ``0`` completed, ``1`` terminal event, ``-1`` failed.
    message : str
        The human-readable SciPy termination message.
    function_evaluations : int
        The number of right-hand-side evaluations (``nfev``).
    jacobian_evaluations : int
        The number of Jacobian evaluations (``njev``; ``0`` for explicit methods).
    event_times : tuple of numpy.ndarray
        Root times ``(E,)`` for each event, including roots between sample times.
    event_phases : tuple of numpy.ndarray
        Unwrapped phases ``(E, N)`` at each root, from SciPy interpolation.

    """

    times: NDArray[np.float64]
    phases: NDArray[np.float64]
    success: bool
    status: int
    message: str
    function_evaluations: int
    jacobian_evaluations: int
    event_times: tuple[NDArray[np.float64], ...] = ()
    event_phases: tuple[NDArray[np.float64], ...] = ()

    @property
    def termination(self) -> Literal["completed", "event", "failed"]:
        """Distinguish reaching the endpoint, a terminal event and solver failure."""
        if not self.success:
            return "failed"
        return "event" if self.status == 1 else "completed"

    @property
    def terminal_phases(self) -> NDArray[np.float64]:
        """Copy the final sampled state; raise ValueError if no samples were stored.

        A terminal event may occur before the first requested sample; its state
        remains available in :attr:`event_phases` even when this accessor refuses.
        """
        if self.phases.shape[0] == 0:
            raise ValueError("solution has no stored samples; inspect event_phases")
        return np.asarray(self.phases[-1], dtype=np.float64).copy()


def kuramoto_ode_rhs(system: KuramotoSystem) -> OdeRightHandSide:
    """Return the ``fun(t, y)`` right-hand side of ``system`` for ``solve_ivp``.

    The returned callable evaluates the system rule at an arbitrary phase state,
    so it is independent of the system's own internal state.
    """
    parameters = system.current_parameters
    rule = system.rule

    def right_hand_side(time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        velocity = np.asarray(
            rule(np.asarray(state, dtype=np.float64), parameters, time), dtype=np.float64
        )
        if velocity.shape != (system.dimension,) or not np.all(np.isfinite(velocity)):
            raise ValueError("rule must return a finite vector matching the state shape")
        return velocity

    return right_hand_side


def kuramoto_ode_jacobian(system: KuramotoSystem) -> OdeJacobian | None:
    """Return the ``jac(t, y)`` Jacobian of ``system``, or ``None`` if it has none."""
    parameters = system.current_parameters
    jacobian = system.jacobian
    if jacobian is None:
        return None

    def ode_jacobian(time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(
            jacobian(np.asarray(state, dtype=np.float64), parameters, time), dtype=np.float64
        )

    return ode_jacobian


def solve_kuramoto_ivp(
    system: KuramotoSystem,
    t_span: Sequence[float],
    *,
    t_eval: Sequence[float] | NDArray[np.float64] | None = None,
    method: str = "RK45",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-9,
    use_jacobian: bool = False,
    events: Callable[[float, NDArray[np.float64]], float]
    | Sequence[Callable[[float, NDArray[np.float64]], float]]
    | None = None,
) -> KuramotoIvpSolution:
    """Integrate ``system`` over ``t_span`` with :func:`scipy.integrate.solve_ivp`.

    The system's :attr:`~KuramotoSystem.current_state` is the initial condition;
    the system is not mutated. For stiff regimes pass ``use_jacobian=True`` with an
    implicit ``method`` (``"BDF"``, ``"Radau"`` or ``"LSODA"``) to feed the analytic
    Jacobian.

    Parameters
    ----------
    system : KuramotoSystem
        The system to integrate.
    t_span : sequence of float
        The ``(t0, tf)`` integration interval.
    t_eval : sequence of float, optional
        Times at which to store the solution; defaults to solver-chosen steps.
    method : str, optional
        Any :func:`scipy.integrate.solve_ivp` method name (default ``"RK45"``).
    rtol, atol : float, optional
        Relative and absolute solver tolerances.
    use_jacobian : bool, optional
        Supply the analytic Jacobian to the solver (requires the system to have
        one; ignored by explicit methods but accepted for uniformity).
    events : callable or sequence of callable, optional
        SciPy ``event(t, phases)`` functions. Their ``terminal`` and ``direction``
        attributes are passed through; root times/states use SciPy interpolation.

    Returns
    -------
    KuramotoIvpSolution
        The solution times, phase trajectory and solver diagnostics.

    Raises
    ------
    ValueError
        If the interval, sample times or tolerances are nonfinite or malformed,
        or ``use_jacobian`` is set on a system without an analytic Jacobian.

    """
    span = tuple(float(value) for value in t_span)
    if len(span) != 2:
        raise ValueError("t_span must be a (t0, tf) pair")
    if not np.all(np.isfinite(span)):
        raise ValueError("t_span must be finite")
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be finite and non-negative")
    keyword_options: dict[str, OdeJacobian] = {}
    if use_jacobian:
        jacobian = kuramoto_ode_jacobian(system)
        if jacobian is None:
            raise ValueError("use_jacobian=True requires a system with an analytic Jacobian")
        keyword_options["jac"] = jacobian
    evaluation_times = None if t_eval is None else np.asarray(t_eval, dtype=np.float64)
    if evaluation_times is not None and not np.all(np.isfinite(evaluation_times)):
        raise ValueError("t_eval must be finite")
    solution = solve_ivp(
        kuramoto_ode_rhs(system),
        span,
        system.current_state,
        method=method,
        t_eval=evaluation_times,
        rtol=rtol,
        atol=atol,
        events=events,
        **keyword_options,
    )
    return KuramotoIvpSolution(
        times=np.asarray(solution.t, dtype=np.float64),
        phases=np.asarray(solution.y, dtype=np.float64).reshape(system.dimension, -1).T,
        success=bool(solution.success),
        status=int(solution.status),
        message=str(solution.message),
        function_evaluations=int(solution.nfev),
        jacobian_evaluations=int(solution.njev),
        event_times=tuple(
            np.asarray(values, dtype=np.float64) for values in (solution.t_events or ())
        ),
        event_phases=tuple(
            np.asarray(values, dtype=np.float64).reshape(-1, system.dimension)
            for values in (solution.y_events or ())
        ),
    )


__all__ = [
    "KuramotoIvpSolution",
    "kuramoto_ode_jacobian",
    "kuramoto_ode_rhs",
    "solve_kuramoto_ivp",
]
