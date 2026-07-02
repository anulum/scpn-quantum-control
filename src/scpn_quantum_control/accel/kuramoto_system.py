# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Unified first-order Kuramoto system/problem object (state/parameter/rule contract)
r"""A unified system/problem object over the first-order Kuramoto phase dynamics.

The adoption-winning dynamical-systems packages expose a single *system object*
that separates the evolving **state** ``u`` from a tunable **parameter
container** ``p`` and a fixed **dynamic rule** ``f(u, p, t)``, and drive it
through a uniform ``current_state`` / ``set_state`` / ``current_parameters`` /
``set_parameter`` / ``reinit`` / ``step`` contract (the DynamicalSystems.jl
``CoupledODEs`` convention). This module brings that contract to the first-order
Kuramoto flow

.. math::

    \dot{\theta} = f(\theta, p, t) = \omega + F(\theta),

where :math:`\omega` are the natural frequencies and :math:`F` is any of the
phase-coupling forces already shipped in :mod:`scpn_quantum_control.accel`
(mean-field, networked/graph, and their Sakaguchi phase-frustrated variants).
The object is a facade: it holds no compute kernel of its own but composes the
existing polyglot-dispatched forces under a fixed-step Euler or classical RK4
stepper, exactly as the model-variant integrators in this package do. The
force evaluation therefore keeps its Rust/Julia/Python dispatch; only the outer
step combination is Python.

Scope is deliberately the first-order deterministic phase flow — the model that
maps directly onto the ``CoupledODEs`` first-order-ODE convention. Systems whose
state is not the bare phase vector are distinct system types with their own
integrators and are not folded in here (keeping this object single-purpose):
the second-order inertial model (:func:`~scpn_quantum_control.accel.kuramoto_inertial.integrate_inertial`,
state :math:`(\theta, v)`), the stochastic model
(:func:`~scpn_quantum_control.accel.kuramoto_noisy.integrate_noisy_kuramoto`),
the delay-differential model
(:func:`~scpn_quantum_control.accel.kuramoto_delayed.integrate_delayed_kuramoto`,
state is a history buffer), and the adaptive model
(:func:`~scpn_quantum_control.accel.kuramoto_adaptive.integrate_adaptive_kuramoto`,
augmented ``(theta, K)`` state).

The rule is autonomous — it does not depend on ``t`` — but the ``f(u, p, t)``
signature is kept so a non-autonomous (e.g. externally driven) rule can be
supplied to the general constructor without changing the contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kuramoto_mean_field import mean_field_force, mean_field_jacobian
from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian
from .sakaguchi_kuramoto import sakaguchi_force, sakaguchi_jacobian
from .sakaguchi_mean_field import sakaguchi_mean_field_force, sakaguchi_mean_field_jacobian

#: The names a :class:`KuramotoParameters` exposes to :meth:`KuramotoSystem.set_parameter`.
_TUNABLE_PARAMETERS = ("natural_frequencies", "coupling", "frustration")

#: The fixed-step integration schemes :class:`KuramotoSystem` can advance under.
_INTEGRATION_SCHEMES = ("euler", "rk4")


@dataclass(frozen=True)
class KuramotoParameters:
    r"""The tunable parameter container ``p`` of a first-order Kuramoto system.

    Separating the parameters from the evolving phase state is what lets a single
    system object be re-run across a parameter sweep without rebuilding the rule.
    The coupling doubles as the topology selector: a scalar drives the all-to-all
    mean-field force, an ``(N, N)`` matrix drives the networked/graph force.

    Attributes
    ----------
    natural_frequencies : numpy.ndarray
        The ``(N,)`` intrinsic frequencies :math:`\omega`.
    coupling : float or numpy.ndarray
        The coupling strength. A scalar ``K`` selects the mean-field topology; an
        ``(N, N)`` matrix ``K_{jk}`` selects the networked topology.
    frustration : float, optional
        The Sakaguchi phase-lag :math:`\alpha` (radians). ``0`` (the default)
        recovers the plain Kuramoto force; a non-zero value selects the
        phase-frustrated Sakaguchi force of the same topology.
    """

    natural_frequencies: NDArray[np.float64]
    coupling: float | NDArray[np.float64]
    frustration: float = 0.0

    def __post_init__(self) -> None:
        omega = np.asarray(self.natural_frequencies, dtype=np.float64)
        if omega.ndim != 1 or omega.size == 0:
            raise ValueError("natural_frequencies must be a non-empty one-dimensional array")
        object.__setattr__(self, "natural_frequencies", omega)
        if np.ndim(self.coupling) == 0:
            object.__setattr__(self, "coupling", float(self.coupling))
        else:
            matrix = np.asarray(self.coupling, dtype=np.float64)
            if matrix.shape != (omega.size, omega.size):
                raise ValueError(
                    "a networked coupling must be an (N, N) matrix matching natural_frequencies"
                )
            object.__setattr__(self, "coupling", matrix)
        object.__setattr__(self, "frustration", float(self.frustration))

    @property
    def size(self) -> int:
        """The oscillator count ``N``."""

        return int(self.natural_frequencies.size)

    @property
    def is_networked(self) -> bool:
        """``True`` when the coupling is a matrix (networked topology)."""

        return isinstance(self.coupling, np.ndarray)

    def with_parameter(self, name: str, value: float | NDArray[np.float64]) -> KuramotoParameters:
        """Return a copy with one parameter replaced (re-validated).

        The replacement is validated by re-running construction, so an invalid
        shape or topology mismatch is reported exactly as at build time.

        Parameters
        ----------
        name : str
            One of ``"natural_frequencies"``, ``"coupling"``, ``"frustration"``.
        value : float or numpy.ndarray
            The replacement value.

        Raises
        ------
        ValueError
            If ``name`` is not a tunable parameter.
        """

        if name == "natural_frequencies":
            return KuramotoParameters(
                np.asarray(value, dtype=np.float64), self.coupling, self.frustration
            )
        if name == "coupling":
            return KuramotoParameters(self.natural_frequencies, value, self.frustration)
        if name == "frustration":
            return KuramotoParameters(self.natural_frequencies, self.coupling, float(value))
        raise ValueError(
            f"unknown Kuramoto parameter {name!r}; expected one of {_TUNABLE_PARAMETERS}"
        )


#: A dynamic rule ``f(u, p, t)`` returning ``dθ/dt`` for a phase state ``u``,
#: parameter container ``p`` and time ``t``. :func:`mean_field_phase_rule` and
#: :func:`networked_phase_rule` are the shipped Kuramoto rules; a custom rule of
#: this shape can be passed to :class:`KuramotoSystem` directly.
PhaseRule = Callable[[NDArray[np.float64], KuramotoParameters, float], NDArray[np.float64]]


def mean_field_phase_rule(
    state: NDArray[np.float64], parameters: KuramotoParameters, time: float
) -> NDArray[np.float64]:
    r"""The all-to-all Kuramoto rule :math:`\omega + F_{\mathrm{mf}}(\theta)`.

    Uses the plain mean-field force when ``frustration`` is zero and the
    Sakaguchi mean-field force otherwise, so a phase-lag set through
    :meth:`KuramotoSystem.set_parameter` takes effect without swapping the rule.

    Parameters
    ----------
    state : numpy.ndarray
        The ``(N,)`` phase vector :math:`\theta`.
    parameters : KuramotoParameters
        Must carry a scalar coupling (mean-field topology).
    time : float
        Ignored — the rule is autonomous; present for the ``f(u, p, t)`` contract.

    Raises
    ------
    ValueError
        If the parameter coupling is a matrix rather than a scalar.
    """

    del time  # autonomous rule; the parameter completes the f(u, p, t) contract
    coupling = parameters.coupling
    if isinstance(coupling, np.ndarray):
        raise ValueError(
            "mean_field_phase_rule needs a scalar coupling; use networked_phase_rule for a matrix"
        )
    if parameters.frustration == 0.0:
        force = mean_field_force(state, coupling)
    else:
        force = sakaguchi_mean_field_force(state, coupling, parameters.frustration)
    return parameters.natural_frequencies + force


def networked_phase_rule(
    state: NDArray[np.float64], parameters: KuramotoParameters, time: float
) -> NDArray[np.float64]:
    r"""The graph Kuramoto rule :math:`\omega + F_{\mathrm{net}}(\theta)`.

    Uses the plain networked force when ``frustration`` is zero and the Sakaguchi
    (phase-frustrated) networked force otherwise.

    Parameters
    ----------
    state : numpy.ndarray
        The ``(N,)`` phase vector :math:`\theta`.
    parameters : KuramotoParameters
        Must carry an ``(N, N)`` coupling matrix (networked topology).
    time : float
        Ignored — the rule is autonomous; present for the ``f(u, p, t)`` contract.

    Raises
    ------
    ValueError
        If the parameter coupling is a scalar rather than a matrix.
    """

    del time  # autonomous rule; the parameter completes the f(u, p, t) contract
    coupling = parameters.coupling
    if not isinstance(coupling, np.ndarray):
        raise ValueError(
            "networked_phase_rule needs an (N, N) coupling matrix; use mean_field_phase_rule for a scalar"
        )
    if parameters.frustration == 0.0:
        force = networked_kuramoto_force(state, coupling)
    else:
        force = sakaguchi_force(state, coupling, parameters.frustration)
    return parameters.natural_frequencies + force


#: The state Jacobian ``∂f/∂θ`` of a :data:`PhaseRule`, returning an ``(N, N)``
#: matrix. Because ``f = ω + F(θ)`` and ``ω`` is constant, the rule Jacobian is
#: the coupling-force Jacobian ``J_F``; it feeds implicit ODE solvers.
PhaseRuleJacobian = Callable[[NDArray[np.float64], KuramotoParameters, float], NDArray[np.float64]]


def mean_field_phase_rule_jacobian(
    state: NDArray[np.float64], parameters: KuramotoParameters, time: float
) -> NDArray[np.float64]:
    r"""The state Jacobian of :func:`mean_field_phase_rule` (an ``(N, N)`` matrix).

    Parameters
    ----------
    state : numpy.ndarray
        The ``(N,)`` phase vector :math:`\theta`.
    parameters : KuramotoParameters
        Must carry a scalar coupling (mean-field topology).
    time : float
        Ignored — the rule is autonomous; present for the ``f(u, p, t)`` contract.

    Raises
    ------
    ValueError
        If the parameter coupling is a matrix rather than a scalar.
    """

    del time  # autonomous rule; the parameter completes the f(u, p, t) contract
    coupling = parameters.coupling
    if isinstance(coupling, np.ndarray):
        raise ValueError(
            "mean_field_phase_rule_jacobian needs a scalar coupling; "
            "use networked_phase_rule_jacobian for a matrix"
        )
    if parameters.frustration == 0.0:
        return mean_field_jacobian(state, coupling)
    return sakaguchi_mean_field_jacobian(state, coupling, parameters.frustration)


def networked_phase_rule_jacobian(
    state: NDArray[np.float64], parameters: KuramotoParameters, time: float
) -> NDArray[np.float64]:
    r"""The state Jacobian of :func:`networked_phase_rule` (an ``(N, N)`` matrix).

    Parameters
    ----------
    state : numpy.ndarray
        The ``(N,)`` phase vector :math:`\theta`.
    parameters : KuramotoParameters
        Must carry an ``(N, N)`` coupling matrix (networked topology).
    time : float
        Ignored — the rule is autonomous; present for the ``f(u, p, t)`` contract.

    Raises
    ------
    ValueError
        If the parameter coupling is a scalar rather than a matrix.
    """

    del time  # autonomous rule; the parameter completes the f(u, p, t) contract
    coupling = parameters.coupling
    if not isinstance(coupling, np.ndarray):
        raise ValueError(
            "networked_phase_rule_jacobian needs an (N, N) coupling matrix; "
            "use mean_field_phase_rule_jacobian for a scalar"
        )
    if parameters.frustration == 0.0:
        return networked_kuramoto_jacobian(state, coupling)
    return sakaguchi_jacobian(state, coupling, parameters.frustration)


class KuramotoSystem:
    r"""A first-order Kuramoto system with a uniform state/parameter/step contract.

    The object separates the evolving phase **state** from the **parameters** and
    the **rule**, and advances the flow :math:`\dot\theta = f(\theta, p, t)` with a
    fixed-step Euler or classical RK4 scheme. Construct it from a shipped topology
    with :meth:`mean_field` / :meth:`networked`, or pass any :data:`PhaseRule` to
    the general constructor.

    Parameters
    ----------
    rule : PhaseRule
        The dynamic rule ``f(u, p, t)`` returning ``dθ/dt``.
    initial_state : numpy.ndarray
        The ``(N,)`` initial phase vector; also the target of an argument-free
        :meth:`reinit`.
    parameters : KuramotoParameters
        The parameter container; its ``size`` must match ``initial_state``.
    dt : float
        The default integration step; must be positive.
    scheme : str, optional
        ``"rk4"`` (default, fourth order) or ``"euler"`` (first order).
    jacobian : PhaseRuleJacobian, optional
        The analytic state Jacobian ``∂f/∂θ`` of ``rule``. The :meth:`mean_field`
        and :meth:`networked` factories attach the matching one; pass one here for
        a custom rule to feed implicit ODE solvers, or leave it ``None``.

    Raises
    ------
    ValueError
        On a non-vector or mismatched state, a non-positive ``dt``, or an unknown
        ``scheme``.
    """

    def __init__(
        self,
        rule: PhaseRule,
        initial_state: NDArray[np.float64],
        parameters: KuramotoParameters,
        *,
        dt: float,
        scheme: str = "rk4",
        jacobian: PhaseRuleJacobian | None = None,
    ) -> None:
        state = np.asarray(initial_state, dtype=np.float64)
        if state.ndim != 1 or state.size == 0:
            raise ValueError("initial_state must be a non-empty one-dimensional phase vector")
        if state.size != parameters.size:
            raise ValueError("initial_state and natural_frequencies must have the same length")
        if scheme not in _INTEGRATION_SCHEMES:
            raise ValueError(f"scheme must be one of {_INTEGRATION_SCHEMES}, got {scheme!r}")
        if not dt > 0.0:
            raise ValueError("dt must be positive")
        self._rule = rule
        self._jacobian = jacobian
        self._parameters = parameters
        self._initial_state = state.copy()
        self._state = state.copy()
        self._initial_time = 0.0
        self._time = 0.0
        self._dt = float(dt)
        self._scheme = scheme

    @classmethod
    def mean_field(
        cls,
        initial_phases: NDArray[np.float64],
        natural_frequencies: NDArray[np.float64],
        coupling: float,
        *,
        frustration: float = 0.0,
        dt: float,
        scheme: str = "rk4",
    ) -> KuramotoSystem:
        """Build an all-to-all mean-field system from raw arrays.

        Parameters
        ----------
        initial_phases, natural_frequencies : numpy.ndarray
            The ``(N,)`` initial phases and intrinsic frequencies.
        coupling : float
            The scalar coupling strength ``K``.
        frustration : float, optional
            The Sakaguchi phase-lag :math:`\\alpha`; ``0`` gives plain Kuramoto.
        dt : float
            The default integration step.
        scheme : str, optional
            ``"rk4"`` or ``"euler"``.
        """

        parameters = KuramotoParameters(
            np.asarray(natural_frequencies, dtype=np.float64), float(coupling), frustration
        )
        return cls(
            mean_field_phase_rule,
            initial_phases,
            parameters,
            dt=dt,
            scheme=scheme,
            jacobian=mean_field_phase_rule_jacobian,
        )

    @classmethod
    def networked(
        cls,
        initial_phases: NDArray[np.float64],
        natural_frequencies: NDArray[np.float64],
        coupling: NDArray[np.float64],
        *,
        frustration: float = 0.0,
        dt: float,
        scheme: str = "rk4",
    ) -> KuramotoSystem:
        """Build a networked/graph system from raw arrays.

        Parameters
        ----------
        initial_phases, natural_frequencies : numpy.ndarray
            The ``(N,)`` initial phases and intrinsic frequencies.
        coupling : numpy.ndarray
            The ``(N, N)`` coupling matrix ``K_{jk}``.
        frustration : float, optional
            The Sakaguchi phase-lag :math:`\\alpha`; ``0`` gives plain Kuramoto.
        dt : float
            The default integration step.
        scheme : str, optional
            ``"rk4"`` or ``"euler"``.
        """

        parameters = KuramotoParameters(
            np.asarray(natural_frequencies, dtype=np.float64),
            np.asarray(coupling, dtype=np.float64),
            frustration,
        )
        return cls(
            networked_phase_rule,
            initial_phases,
            parameters,
            dt=dt,
            scheme=scheme,
            jacobian=networked_phase_rule_jacobian,
        )

    @property
    def current_state(self) -> NDArray[np.float64]:
        """A copy of the current phase state ``u``."""

        return self._state.copy()

    @property
    def initial_state(self) -> NDArray[np.float64]:
        """A copy of the state an argument-free :meth:`reinit` returns to."""

        return self._initial_state.copy()

    @property
    def current_parameters(self) -> KuramotoParameters:
        """The current (immutable) parameter container ``p``."""

        return self._parameters

    @property
    def rule(self) -> PhaseRule:
        """The dynamic rule ``f(u, p, t)``; callable at an arbitrary state.

        Exposed so external solvers can evaluate the flow at any state rather than
        only the internal one, e.g. ``system.rule(y, system.current_parameters, t)``.
        """

        return self._rule

    @property
    def jacobian(self) -> PhaseRuleJacobian | None:
        """The analytic state Jacobian ``∂f/∂θ`` if one was supplied, else ``None``."""

        return self._jacobian

    @property
    def current_time(self) -> float:
        """The current integration time ``t``."""

        return self._time

    @property
    def dimension(self) -> int:
        """The oscillator count ``N``."""

        return int(self._initial_state.size)

    @property
    def scheme(self) -> str:
        """The fixed-step integration scheme in use."""

        return self._scheme

    @property
    def dt(self) -> float:
        """The default integration step."""

        return self._dt

    def set_state(self, state: NDArray[np.float64]) -> None:
        """Overwrite the current phase state (time is unchanged).

        Raises
        ------
        ValueError
            If ``state`` is not an ``(N,)`` vector matching the system.
        """

        new_state = np.asarray(state, dtype=np.float64)
        if new_state.shape != self._state.shape:
            raise ValueError(f"state must have shape {self._state.shape}, got {new_state.shape}")
        self._state = new_state.copy()

    def set_parameter(self, name: str, value: float | NDArray[np.float64]) -> None:
        """Replace one parameter in place, keeping the oscillator count fixed.

        Raises
        ------
        ValueError
            If ``name`` is unknown, the value is invalid, or it would change ``N``.
        """

        updated = self._parameters.with_parameter(name, value)
        if updated.size != self._initial_state.size:
            raise ValueError("set_parameter cannot change the number of oscillators")
        self._parameters = updated

    def reinit(
        self, state: NDArray[np.float64] | None = None, *, time: float | None = None
    ) -> None:
        """Reset the state and time (the DynamicalSystems ``reinit!`` contract).

        Parameters
        ----------
        state : numpy.ndarray, optional
            The state to reset to; defaults to the construction :attr:`initial_state`.
            The stored initial state is not changed.
        time : float, optional
            The time to reset to; defaults to the initial time ``0``.

        Raises
        ------
        ValueError
            If ``state`` is given and does not match the system shape.
        """

        if state is None:
            target = self._initial_state.copy()
        else:
            target = np.asarray(state, dtype=np.float64)
            if target.shape != self._initial_state.shape:
                raise ValueError(
                    f"state must have shape {self._initial_state.shape}, got {target.shape}"
                )
            target = target.copy()
        self._state = target
        self._time = self._initial_time if time is None else float(time)

    def rule_value(self, *, time: float | None = None) -> NDArray[np.float64]:
        """Evaluate ``f(u, p, t)`` — the phase velocity ``dθ/dt`` at the current state.

        Parameters
        ----------
        time : float, optional
            The time to evaluate at; defaults to the current time.
        """

        evaluation_time = self._time if time is None else float(time)
        return np.asarray(
            self._rule(self._state, self._parameters, evaluation_time), dtype=np.float64
        )

    def rule_jacobian(self, *, time: float | None = None) -> NDArray[np.float64]:
        """Evaluate the analytic state Jacobian ``∂f/∂θ`` at the current state.

        Parameters
        ----------
        time : float, optional
            The time to evaluate at; defaults to the current time.

        Raises
        ------
        ValueError
            If the system was built without an analytic Jacobian.
        """

        if self._jacobian is None:
            raise ValueError("this system has no analytic Jacobian; supply one at construction")
        evaluation_time = self._time if time is None else float(time)
        return np.asarray(
            self._jacobian(self._state, self._parameters, evaluation_time), dtype=np.float64
        )

    def _advance(self, state: NDArray[np.float64], time: float, dt: float) -> NDArray[np.float64]:
        """Return the state one fixed step later under the active scheme."""

        if self._scheme == "euler":
            return state + dt * self._rule(state, self._parameters, time)
        k1 = self._rule(state, self._parameters, time)
        k2 = self._rule(state + 0.5 * dt * k1, self._parameters, time + 0.5 * dt)
        k3 = self._rule(state + 0.5 * dt * k2, self._parameters, time + 0.5 * dt)
        k4 = self._rule(state + dt * k3, self._parameters, time + dt)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, *, n: int = 1, dt: float | None = None) -> NDArray[np.float64]:
        """Advance the system ``n`` steps and return the new state.

        Parameters
        ----------
        n : int, optional
            The number of steps to take; must be positive.
        dt : float, optional
            The step size; defaults to the system :attr:`dt`.

        Raises
        ------
        ValueError
            If ``n`` is not positive or ``dt`` is not positive.
        """

        if n < 1:
            raise ValueError("n must be a positive integer")
        step_size = self._dt if dt is None else float(dt)
        if not step_size > 0.0:
            raise ValueError("dt must be positive")
        for _ in range(n):
            self._state = self._advance(self._state, self._time, step_size)
            self._time += step_size
        return self._state.copy()

    def trajectory(self, n_steps: int, *, dt: float | None = None) -> NDArray[np.float64]:
        """Advance ``n_steps`` steps, recording every state, and return the path.

        The system is left at the final state and time (the recorded path includes
        the starting state as row ``0``).

        Parameters
        ----------
        n_steps : int
            The number of steps to record; must be positive.
        dt : float, optional
            The step size; defaults to the system :attr:`dt`.

        Returns
        -------
        numpy.ndarray
            The ``(n_steps + 1, N)`` phase trajectory.

        Raises
        ------
        ValueError
            If ``n_steps`` is not positive or ``dt`` is not positive.
        """

        if n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        step_size = self._dt if dt is None else float(dt)
        if not step_size > 0.0:
            raise ValueError("dt must be positive")
        path = np.empty((n_steps + 1, self._state.size), dtype=np.float64)
        path[0] = self._state
        for index in range(n_steps):
            self._state = self._advance(self._state, self._time, step_size)
            self._time += step_size
            path[index + 1] = self._state
        return path

    def __repr__(self) -> str:
        topology = "networked" if self._parameters.is_networked else "mean_field"
        return (
            f"KuramotoSystem(N={self.dimension}, topology={topology}, "
            f"scheme={self._scheme!r}, dt={self._dt}, t={self._time})"
        )


__all__ = [
    "KuramotoParameters",
    "KuramotoSystem",
    "mean_field_phase_rule",
    "mean_field_phase_rule_jacobian",
    "networked_phase_rule",
    "networked_phase_rule_jacobian",
]
