# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Matplotlib rendering layer for Kuramoto oscillator and network dynamics
r"""Matplotlib rendering layer for Kuramoto oscillator and network dynamics.

Standard diagnostics for coupled-phase-oscillator trajectories, rendered from the shipped trajectory
and observable surfaces:

* :func:`phase_raster` — the ``(T, N)`` phase trajectory as an oscillator-by-time raster with a cyclic
  colour map, the canonical way to read travelling waves, locking and incoherence at a glance;
* :func:`order_parameter_timeseries` — the Kuramoto order parameter ``r(t)`` over the trajectory,
  computed from the same complex mean field as
  :func:`~scpn_quantum_control.accel.order_parameter_observables.order_parameter`;
* :func:`chimera_snapshot` — the per-community order parameters
  :func:`~scpn_quantum_control.accel.kuramoto_chimera.community_order_parameters` over time, annotated
  with the :func:`~scpn_quantum_control.accel.kuramoto_chimera.chimera_index`, to read a chimera state
  (one community locked while another drifts);
* :func:`network_phase_embedding` — a single-time phase snapshot on a circular node layout with the
  coupling drawn as edges, to read the spatial organisation of a locked or clustered state.

Matplotlib is an optional dependency (the ``viz`` extra); it is imported lazily, so importing this
module — and therefore the :mod:`scpn_quantum_control.accel` and :mod:`scpn_quantum_control.kuramoto`
facades — never requires matplotlib. Calling a renderer without it raises a clear :class:`ImportError`
naming the install command. Every renderer draws into a caller-supplied axis when given (so panels
compose into a figure) or creates its own otherwise, returns the axis, and never calls ``show`` — the
caller owns display and saving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .kuramoto_chimera import CommunityList, chimera_index, community_order_parameters

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Cyclic colour map used for phase-valued data (wraps cleanly at ``±π``).
_PHASE_COLOUR_MAP = "twilight"


def _require_pyplot() -> Any:
    r"""Return :mod:`matplotlib.pyplot`, raising a helpful error when matplotlib is not installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "matplotlib is required for the Kuramoto visualisation layer; install it with "
            "\"pip install 'scpn-quantum-control[viz]'\""
        ) from error
    return plt


def _validate_trajectory(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Return a validated contiguous ``(T, N)`` float64 phase trajectory."""
    trajectory = np.ascontiguousarray(phases, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError(
            f"phases must be a two-dimensional (T, N) trajectory, got shape {trajectory.shape}"
        )
    if trajectory.shape[0] < 1 or trajectory.shape[1] < 1:
        raise ValueError(
            f"phases must have at least one step and one oscillator, got shape {trajectory.shape}"
        )
    return trajectory


def _validate_times(times: NDArray[np.float64] | None, steps: int) -> NDArray[np.float64]:
    r"""Return the sample times, defaulting to the integer step index when ``times`` is ``None``."""
    if times is None:
        return np.arange(steps, dtype=np.float64)
    resolved = np.ascontiguousarray(times, dtype=np.float64)
    if resolved.shape != (steps,):
        raise ValueError(f"times must have shape ({steps},), got {resolved.shape}")
    return resolved


def _wrap_to_pi(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Wrap phases into ``[-π, π]`` via the complex representation (branch-cut free)."""
    return np.asarray(np.angle(np.exp(1j * phases)), dtype=np.float64)


def _resolve_axis(ax: Axes | None) -> tuple[Any, Axes]:
    r"""Return ``(figure, axis)``, creating a fresh single-axis figure when ``ax`` is ``None``."""
    if ax is None:
        plt = _require_pyplot()
        figure, axis = plt.subplots()
        return figure, cast("Axes", axis)
    return ax.figure, ax


def phase_raster(
    phases: NDArray[np.float64],
    *,
    times: NDArray[np.float64] | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Render a ``(T, N)`` phase trajectory as an oscillator-by-time raster.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    times : numpy.ndarray, optional
        The ``T`` sample times for the horizontal extent; the integer step index when omitted.
    ax : matplotlib.axes.Axes, optional
        The axis to draw into; a fresh figure and axis are created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axis carrying the raster image and its phase colour bar.
    """
    trajectory = _validate_trajectory(phases)
    steps, count = trajectory.shape
    sample_times = _validate_times(times, steps)
    figure, axis = _resolve_axis(ax)

    span = float(sample_times[-1] - sample_times[0])
    right = float(sample_times[-1]) if span > 0.0 else float(sample_times[0]) + 1.0
    image = axis.imshow(
        _wrap_to_pi(trajectory).T,
        aspect="auto",
        origin="lower",
        extent=(float(sample_times[0]), right, 0.0, float(count)),
        cmap=_PHASE_COLOUR_MAP,
        vmin=-np.pi,
        vmax=np.pi,
    )
    axis.set_xlabel("time")
    axis.set_ylabel("oscillator index")
    axis.set_title("phase raster")
    figure.colorbar(image, ax=axis, label="phase (rad)")
    return axis


def order_parameter_timeseries(
    phases: NDArray[np.float64],
    *,
    times: NDArray[np.float64] | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot the Kuramoto order parameter ``r(t)`` over a ``(T, N)`` trajectory.

    ``r(t) = |⟨e^{iθ(t)}⟩|`` is read from the same complex mean field as
    :func:`~scpn_quantum_control.accel.order_parameter_observables.order_parameter`.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    times : numpy.ndarray, optional
        The ``T`` sample times for the horizontal axis; the integer step index when omitted.
    ax : matplotlib.axes.Axes, optional
        The axis to draw into; a fresh figure and axis are created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axis carrying the order-parameter curve.
    """
    trajectory = _validate_trajectory(phases)
    steps = trajectory.shape[0]
    sample_times = _validate_times(times, steps)
    _, axis = _resolve_axis(ax)

    mean_field = np.exp(1j * trajectory).mean(axis=1)
    coherence = np.abs(mean_field)
    axis.plot(sample_times, coherence, color="C0", label="order parameter $r(t)$")
    axis.set_xlabel("time")
    axis.set_ylabel("order parameter $r$")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("order parameter")
    axis.legend(loc="best")
    return axis


def chimera_snapshot(
    phases: NDArray[np.float64],
    communities: CommunityList,
    *,
    times: NDArray[np.float64] | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot the per-community order parameters over time, annotated with the chimera index.

    Reuses :func:`~scpn_quantum_control.accel.kuramoto_chimera.community_order_parameters` and
    :func:`~scpn_quantum_control.accel.kuramoto_chimera.chimera_index`, so the rendered curves and the
    annotation are exactly the shipped diagnostics.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    communities : sequence of numpy.ndarray or sequence of int
        The community partition — the oscillator indices of each community.
    times : numpy.ndarray, optional
        The ``T`` sample times for the horizontal axis; the integer step index when omitted.
    ax : matplotlib.axes.Axes, optional
        The axis to draw into; a fresh figure and axis are created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axis carrying one order-parameter curve per community.
    """
    trajectory = _validate_trajectory(phases)
    steps = trajectory.shape[0]
    sample_times = _validate_times(times, steps)
    _, axis = _resolve_axis(ax)

    community_curves = community_order_parameters(trajectory, communities)
    index = chimera_index(trajectory, communities)
    for community, curve in enumerate(community_curves.T):
        axis.plot(sample_times, curve, label=f"community {community}")
    axis.set_xlabel("time")
    axis.set_ylabel("community order parameter")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(f"chimera index = {index:.3f}")
    axis.legend(loc="best")
    return axis


def network_phase_embedding(
    phases: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> Axes:
    r"""Render a single-time phase snapshot on a circular node layout with coupling edges.

    Nodes are placed on the unit circle in index order and coloured by phase with a cyclic colour map;
    each non-zero off-diagonal coupling ``K_ij`` is drawn as an edge whose width scales with its
    magnitude, so the spatial organisation of a locked or clustered state is legible.

    Parameters
    ----------
    phases : numpy.ndarray
        One-dimensional ``(N,)`` array of oscillator phases in radians at a single time.
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix; the diagonal is ignored.
    ax : matplotlib.axes.Axes, optional
        The axis to draw into; a fresh figure and axis are created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axis carrying the coupling edges, the phase-coloured nodes and the phase colour bar.
    """
    snapshot = np.ascontiguousarray(phases, dtype=np.float64)
    if snapshot.ndim != 1 or snapshot.size < 1:
        raise ValueError(
            f"phases must be a one-dimensional (N,) snapshot, got shape {snapshot.shape}"
        )
    count = int(snapshot.size)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {matrix.shape}")

    figure, axis = _resolve_axis(ax)
    node_angles = 2.0 * np.pi * np.arange(count) / count
    positions = np.column_stack((np.cos(node_angles), np.sin(node_angles)))

    magnitudes = np.abs(matrix)
    np.fill_diagonal(magnitudes, 0.0)
    upper = np.triu(magnitudes, k=1)
    peak = float(upper.max())
    if peak > 0.0:
        from matplotlib.collections import LineCollection

        rows, columns = np.nonzero(upper)
        segments = [(positions[i], positions[j]) for i, j in zip(rows, columns, strict=True)]
        widths = [
            0.2 + 2.5 * float(upper[i, j]) / peak for i, j in zip(rows, columns, strict=True)
        ]
        axis.add_collection(
            LineCollection(segments, colors="0.7", linewidths=widths, alpha=0.6, zorder=1)
        )

    nodes = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=_wrap_to_pi(snapshot),
        cmap=_PHASE_COLOUR_MAP,
        vmin=-np.pi,
        vmax=np.pi,
        s=120,
        edgecolors="black",
        linewidths=0.5,
        zorder=2,
    )
    axis.set_aspect("equal")
    axis.set_xlim(-1.2, 1.2)
    axis.set_ylim(-1.2, 1.2)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("network phase embedding")
    figure.colorbar(nodes, ax=axis, label="phase (rad)")
    return axis


__all__ = [
    "chimera_snapshot",
    "network_phase_embedding",
    "order_parameter_timeseries",
    "phase_raster",
]
