# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Parameter-grid sweep and observable-measurement API over the Kuramoto system
r"""Sweep a :class:`KuramotoSystem` across a parameter grid and measure observables.

The parameter sweep is the workflow that turns a single dynamical-system object
into a *phase diagram*: fix an initial condition, vary one or more parameters over
a grid, integrate at each grid point, and reduce every trajectory to a handful of
scalar **observables** (order parameter, metastability, frequency spread, …). It
is the ``BoxSearch`` / ``ParameterSpace`` convention that the ``neurolib`` brain
network simulator popularised (Cakan *et al.*, *Cognitive Computation* 13, 2021,
DOI ``10.1007/s12559-021-09931-9``), brought to the shipped Kuramoto system.

This module is a facade: it adds no dynamics and no measurement kernels of its own.
It composes the existing pieces —

* :class:`~oscillatools.accel.kuramoto_system.KuramotoSystem` supplies the
  rule, the initial state and the fixed-step integrator via
  :meth:`~oscillatools.accel.kuramoto_system.KuramotoSystem.trajectory`;
* :class:`~oscillatools.accel.kuramoto_system.KuramotoParameters` supplies
  the immutable ``with_parameter`` machinery that re-points the system at each grid
  cell without rebuilding the rule;
* the shipped observables (:func:`~oscillatools.accel.order_parameter_observables.order_parameter`,
  :func:`~oscillatools.accel.kuramoto_chimera.metastability_index`,
  :func:`~oscillatools.accel.kuramoto_frequency_order.frequency_synchronisation_index`)
  reduce each trajectory to a scalar, keeping their own Rust/Julia/Python dispatch.

Three objects make up the API:

* :class:`KuramotoParameterGrid` — the grid specification (a mapping of tunable
  parameter names to the values to sweep), yielding the Cartesian product of cells;
* :class:`Observable` — a named scalar reduction of a ``(T, N)`` trajectory, with a
  small library of constructors for the shipped measurements;
* :func:`sweep_parameter_grid` — the runner that evaluates every observable at every
  grid cell and returns a :class:`ParameterSweepResult`.

The runner is a pure query: it never mutates the input system. Each grid cell is
integrated on a freshly built copy that shares the input system's rule, initial
state, step and scheme, so the caller's system is left exactly as it was.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .kuramoto_chimera import metastability_index
from .kuramoto_frequency_order import frequency_synchronisation_index
from .kuramoto_system import _TUNABLE_PARAMETERS, KuramotoParameters, KuramotoSystem
from .order_parameter_observables import order_parameter

#: A single swept value — a scalar parameter (coupling ``K`` or frustration
#: :math:`\alpha`) or a vector/matrix parameter (natural frequencies, coupling
#: matrix). It is whatever
#: :meth:`~oscillatools.accel.kuramoto_system.KuramotoParameters.with_parameter`
#: accepts for the corresponding name.
SweptValue: TypeAlias = float | NDArray[np.float64]

#: A scalar reduction of a ``(T, N)`` phase trajectory and its sampling step ``dt``.
#: Every observable shares this signature so that measurements needing the step
#: (e.g. effective frequencies) and those that do not compose uniformly.
ObservableFunction = Callable[[NDArray[np.float64], float], float]


class KuramotoParameterGrid:
    r"""A Cartesian grid over the tunable parameters of a Kuramoto system.

    The grid is the sweep's *design of experiment*: a mapping from tunable
    parameter names to the sequence of values each should take. Iterating the grid
    yields one assignment per cell of the Cartesian product, in row-major order
    over the axes in the order they were given.

    Parameters
    ----------
    axes : Mapping[str, Sequence]
        A mapping from a tunable parameter name — one of ``"natural_frequencies"``,
        ``"coupling"``, ``"frustration"`` — to the non-empty sequence of values to
        sweep for it. The coupling values must match the system topology: scalars
        for a mean-field system, ``(N, N)`` matrices for a networked one.

    Raises
    ------
    ValueError
        If ``axes`` is empty, names a parameter that is not tunable, or gives an
        empty sequence of values for any axis.
    """

    def __init__(self, axes: Mapping[str, Sequence[SweptValue]]) -> None:
        if not axes:
            raise ValueError("a parameter grid needs at least one axis")
        names: list[str] = []
        values: list[tuple[SweptValue, ...]] = []
        for name, sequence in axes.items():
            if name not in _TUNABLE_PARAMETERS:
                raise ValueError(
                    f"unknown Kuramoto parameter {name!r}; expected one of {_TUNABLE_PARAMETERS}"
                )
            cells = tuple(sequence)
            if not cells:
                raise ValueError(f"axis {name!r} needs at least one value")
            names.append(name)
            values.append(cells)
        self._axis_names: tuple[str, ...] = tuple(names)
        self._axis_values: tuple[tuple[SweptValue, ...], ...] = tuple(values)

    @property
    def axis_names(self) -> tuple[str, ...]:
        """The swept parameter names, in grid (axis) order."""

        return self._axis_names

    @property
    def axis_values(self) -> tuple[tuple[SweptValue, ...], ...]:
        """The swept values per axis, aligned with :attr:`axis_names`."""

        return self._axis_values

    @property
    def shape(self) -> tuple[int, ...]:
        """The number of values on each axis, in axis order."""

        return tuple(len(values) for values in self._axis_values)

    @property
    def size(self) -> int:
        """The total number of grid cells (the product of the axis lengths)."""

        return int(np.prod(self.shape, dtype=np.int64))

    def points(self) -> Iterator[dict[str, SweptValue]]:
        """Yield one ``{name: value}`` assignment per cell, in row-major order.

        The last axis varies fastest, matching :func:`itertools.product` and the
        row-major reshape used by :meth:`ParameterSweepResult.grid_values`.
        """

        for combination in product(*self._axis_values):
            yield dict(zip(self._axis_names, combination, strict=True))

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[dict[str, SweptValue]]:
        return self.points()

    def __repr__(self) -> str:
        axes = ", ".join(
            f"{name}[{length}]" for name, length in zip(self._axis_names, self.shape, strict=True)
        )
        return f"KuramotoParameterGrid({axes}, size={self.size})"


@dataclass(frozen=True)
class Observable:
    """A named scalar reduction of a ``(T, N)`` phase trajectory.

    An observable pairs a human-readable name (the column label on a
    :class:`ParameterSweepResult`) with a reduction that maps a post-transient
    trajectory and its sampling step to a single number.

    Attributes
    ----------
    name : str
        The label the measurement appears under in the sweep result.
    measure : ObservableFunction
        The reduction ``measure(trajectory, dt) -> float`` applied at each grid
        cell, where ``trajectory`` is the ``(T, N)`` post-transient phase path and
        ``dt`` is the integration step.
    """

    name: str
    measure: ObservableFunction

    def __call__(self, trajectory: NDArray[np.float64], dt: float) -> float:
        """Evaluate the observable on ``trajectory`` sampled at ``dt``."""

        return float(self.measure(trajectory, dt))


def _time_averaged_order_parameter(trajectory: NDArray[np.float64], dt: float) -> float:
    del dt  # the order parameter is instantaneous; the step is not needed
    return float(np.mean([order_parameter(row) for row in trajectory]))


def _terminal_order_parameter(trajectory: NDArray[np.float64], dt: float) -> float:
    del dt  # instantaneous measurement of the final state
    return order_parameter(np.asarray(trajectory[-1], dtype=np.float64))


def _metastability(trajectory: NDArray[np.float64], dt: float) -> float:
    del dt  # metastability is the temporal variance of an instantaneous quantity
    return metastability_index(trajectory)


def mean_order_parameter() -> Observable:
    r"""The time-averaged Kuramoto order parameter :math:`\langle R(t)\rangle_t`.

    Averages the instantaneous global coherence
    :func:`~oscillatools.accel.order_parameter_observables.order_parameter`
    over the post-transient trajectory — the canonical scalar for a
    synchronisation phase diagram.
    """

    return Observable("mean_order_parameter", _time_averaged_order_parameter)


def terminal_order_parameter() -> Observable:
    r"""The Kuramoto order parameter :math:`R(T)` of the final trajectory row.

    Reports the coherence reached at the end of the integration window, the
    natural read-out when the transient has been discarded and the state has
    settled onto its attractor.
    """

    return Observable("terminal_order_parameter", _terminal_order_parameter)


def metastability() -> Observable:
    r"""The metastability index :math:`M = \operatorname{Var}_t R(t)`.

    The temporal variance of the global order parameter
    (:func:`~oscillatools.accel.kuramoto_chimera.metastability_index`):
    zero for a stationary collective state and growing as the coherence wanders,
    which marks the chimera / metastable region of a phase diagram.
    """

    return Observable("metastability", _metastability)


def frequency_spread() -> Observable:
    r"""The effective-frequency spread (frequency-synchronisation index).

    The population standard deviation of the effective rotation rates
    (:func:`~oscillatools.accel.kuramoto_frequency_order.frequency_synchronisation_index`),
    evaluated at the sweep's integration step: zero for a frequency-locked state
    and growing with the spread of observed frequencies.
    """

    return Observable(
        "frequency_spread",
        lambda trajectory, dt: frequency_synchronisation_index(trajectory, dt=dt),
    )


@dataclass(frozen=True)
class ParameterSweepResult:
    r"""The measured observables over every cell of a :class:`KuramotoParameterGrid`.

    The measurements are stored as a ``(cells, observables)`` matrix in the grid's
    row-major cell order, so a column reshaped to :attr:`grid_shape` recovers the
    phase-diagram array for one observable.

    Attributes
    ----------
    grid : KuramotoParameterGrid
        The grid that was swept.
    observable_names : tuple of str
        The observable labels, in measurement-column order.
    measurements : numpy.ndarray
        The ``(grid.size, len(observable_names))`` measured values, row ``k`` being
        the observables at the ``k``-th grid cell (row-major over the axes).
    """

    grid: KuramotoParameterGrid
    observable_names: tuple[str, ...]
    measurements: NDArray[np.float64]

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """The grid axis lengths — the shape of each :meth:`grid_values` array."""

        return self.grid.shape

    def _observable_index(self, name: str) -> int:
        try:
            return self.observable_names.index(name)
        except ValueError:
            available = ", ".join(self.observable_names)
            raise ValueError(f"unknown observable {name!r}; measured: {available}") from None

    def grid_values(self, name: str) -> NDArray[np.float64]:
        """Return one observable's values reshaped to the grid.

        Parameters
        ----------
        name : str
            The observable label; must be one of :attr:`observable_names`.

        Returns
        -------
        numpy.ndarray
            The observable over the grid, shaped like :attr:`grid_shape` (axis
            order matching :attr:`KuramotoParameterGrid.axis_names`).

        Raises
        ------
        ValueError
            If ``name`` was not measured.
        """

        column = self.measurements[:, self._observable_index(name)]
        return column.reshape(self.grid_shape)

    def records(self) -> list[dict[str, SweptValue]]:
        """Return one flat ``{**parameters, **observables}`` record per grid cell.

        Each record merges the cell's swept parameter assignment with its measured
        observables — the tidy, row-per-cell form for tabulation or a DataFrame.
        """

        rows: list[dict[str, SweptValue]] = []
        for cell, point in enumerate(self.grid.points()):
            record: dict[str, SweptValue] = dict(point)
            for column, name in enumerate(self.observable_names):
                record[name] = float(self.measurements[cell, column])
            rows.append(record)
        return rows

    def best(self, name: str, *, maximise: bool = True) -> tuple[dict[str, SweptValue], float]:
        """Return the grid cell optimising one observable and its value.

        Parameters
        ----------
        name : str
            The observable to optimise; must be one of :attr:`observable_names`.
        maximise : bool, optional
            Return the maximising cell (default) or, when ``False``, the minimising
            cell.

        Returns
        -------
        tuple
            The ``({name: value}, observable_value)`` pair for the optimal cell.

        Raises
        ------
        ValueError
            If ``name`` was not measured.
        """

        column = self.measurements[:, self._observable_index(name)]
        cell = int(np.argmax(column) if maximise else np.argmin(column))
        point = next(
            cell_point for index, cell_point in enumerate(self.grid.points()) if index == cell
        )
        return point, float(column[cell])


def sweep_parameter_grid(
    system: KuramotoSystem,
    grid: KuramotoParameterGrid,
    observables: Sequence[Observable],
    *,
    n_steps: int,
    dt: float | None = None,
    transient: int = 0,
) -> ParameterSweepResult:
    r"""Measure ``observables`` across ``grid`` on copies of ``system``.

    For each grid cell the base parameters of ``system`` are re-pointed by the
    cell's assignment (via
    :meth:`~oscillatools.accel.kuramoto_system.KuramotoParameters.with_parameter`),
    a fresh system is integrated for ``n_steps`` from the input system's initial
    state, the leading ``transient`` rows are discarded, and every observable is
    reduced over the remainder. The input ``system`` is never mutated.

    Parameters
    ----------
    system : KuramotoSystem
        The base system; its rule, initial state, step and scheme are shared by the
        per-cell copies. Its topology fixes the admissible swept coupling (scalar
        for mean-field, ``(N, N)`` matrix for networked).
    grid : KuramotoParameterGrid
        The parameter grid to sweep.
    observables : sequence of Observable
        The measurements to record at each cell; must be non-empty and carry
        distinct names.
    n_steps : int
        The number of integration steps per cell (the trajectory has
        ``n_steps + 1`` rows including the initial state); must be positive.
    dt : float, optional
        The integration step; defaults to the system's own
        :attr:`~oscillatools.accel.kuramoto_system.KuramotoSystem.dt`.
    transient : int, optional
        The number of leading trajectory rows to discard before measuring; must be
        non-negative and leave at least one row.

    Returns
    -------
    ParameterSweepResult
        The observables measured over every grid cell.

    Raises
    ------
    ValueError
        If ``observables`` is empty or has duplicate names, ``n_steps`` is not
        positive, or ``transient`` is negative or discards the whole trajectory.
    """

    if not observables:
        raise ValueError("at least one observable is required")
    names = tuple(observable.name for observable in observables)
    if len(set(names)) != len(names):
        raise ValueError("observable names must be distinct")
    if n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if transient < 0:
        raise ValueError("transient must be non-negative")
    if transient > n_steps:
        raise ValueError("transient must leave at least one trajectory row")

    step_size = system.dt if dt is None else float(dt)
    base_parameters = system.current_parameters
    initial_state = system.initial_state
    rule = system.rule
    jacobian = system.jacobian
    scheme = system.scheme

    measurements = np.empty((grid.size, len(observables)), dtype=np.float64)
    for cell, point in enumerate(grid.points()):
        parameters = _apply_point(base_parameters, point)
        cell_system = KuramotoSystem(
            rule,
            initial_state,
            parameters,
            dt=step_size,
            scheme=scheme,
            jacobian=jacobian,
        )
        trajectory = cell_system.trajectory(n_steps, dt=step_size)
        measured = trajectory[transient:]
        for column, observable in enumerate(observables):
            measurements[cell, column] = observable(measured, step_size)

    return ParameterSweepResult(grid, names, measurements)


def _apply_point(base: KuramotoParameters, point: Mapping[str, SweptValue]) -> KuramotoParameters:
    """Return ``base`` with each parameter in ``point`` replaced (re-validated)."""

    parameters = base
    for name, value in point.items():
        parameters = parameters.with_parameter(name, value)
    return parameters


__all__ = [
    "KuramotoParameterGrid",
    "Observable",
    "ParameterSweepResult",
    "frequency_spread",
    "mean_order_parameter",
    "metastability",
    "sweep_parameter_grid",
    "terminal_order_parameter",
]
