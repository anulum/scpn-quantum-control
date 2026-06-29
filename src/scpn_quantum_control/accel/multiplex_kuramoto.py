# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multiplex (multilayer) Kuramoto networks
r"""Multiplex (multilayer) Kuramoto networks — synchronisation across coupled layers.

A multiplex network stacks the *same* ``N`` nodes into ``L`` layers, each layer its own coupling
graph, with every node additionally tied to its own replica in the other layers. Phase oscillators on
such a network obey

.. math::

    \dot\theta_i^\alpha = \omega_i^\alpha
        + \sum_j A_{ij}^\alpha \sin(\theta_j^\alpha - \theta_i^\alpha)
        + \sum_\beta B_{\alpha\beta}\,\sin(\theta_i^\beta - \theta_i^\alpha),

with the *intra-layer* coupling ``A^\alpha`` (the network within layer ``\alpha``) and the
*inter-layer* coupling ``B`` (how layer ``\alpha`` couples to layer ``\beta`` through the shared
node). Multilayer networks are the dominant modern network architecture, and the interplay of the two
couplings produces genuinely multilayer synchronisation — layer-by-layer coherence and inter-layer
locking — that a single-layer model cannot express.

The field reuses the toolkit's networked-Kuramoto force *twice*: once per layer for the intra-layer
term and once per node across layers for the inter-layer term (the inter-layer coupling is itself a
small Kuramoto network over the ``L`` replicas). With the inter-layer coupling switched off the layers
decouple into independent single-layer Kuramoto systems exactly; with strong inter-layer coupling the
replicas of each node phase-lock across layers. The companion Jacobian is the ``LN \times LN``
block matrix (intra blocks on the layer diagonal, inter blocks scattered over the shared nodes) for
linear-stability analysis. It adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian


@dataclass(frozen=True)
class MultiplexTrajectory:
    """A multiplex Kuramoto phase trajectory.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    phases : numpy.ndarray
        The ``(n_steps + 1, L, N)`` phase trajectory (``L`` layers of ``N`` nodes).
    """

    times: NDArray[np.float64]
    phases: NDArray[np.float64]

    @property
    def terminal_phases(self) -> NDArray[np.float64]:
        """The final phases ``θ(T)`` (shape ``(L, N)``)."""
        return np.ascontiguousarray(self.phases[-1], dtype=np.float64)


def _validate(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    intra_coupling: NDArray[np.float64],
    inter_coupling: NDArray[np.float64],
) -> tuple[int, int]:
    if phases.ndim != 2 or phases.shape[0] < 1 or phases.shape[1] < 2:
        raise ValueError("phases must be an (L >= 1, N >= 2) array")
    layers, nodes = phases.shape
    if omega.shape != (layers, nodes):
        raise ValueError(f"omega must have shape ({layers}, {nodes}), got {omega.shape}")
    if intra_coupling.shape != (layers, nodes, nodes):
        raise ValueError(
            f"intra_coupling must have shape ({layers}, {nodes}, {nodes}), "
            f"got {intra_coupling.shape}"
        )
    if inter_coupling.shape != (layers, layers):
        raise ValueError(
            f"inter_coupling must have shape ({layers}, {layers}), got {inter_coupling.shape}"
        )
    if not (
        np.all(np.isfinite(phases))
        and np.all(np.isfinite(omega))
        and np.all(np.isfinite(intra_coupling))
        and np.all(np.isfinite(inter_coupling))
    ):
        raise ValueError("phases, omega and coupling matrices must be finite")
    return layers, nodes


def _field(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    intra_coupling: NDArray[np.float64],
    inter_coupling: NDArray[np.float64],
    layers: int,
) -> NDArray[np.float64]:
    """The multiplex field ``(L, N)`` reusing the networked force per layer and per node."""
    intra = np.stack(
        [networked_kuramoto_force(phases[layer], intra_coupling[layer]) for layer in range(layers)]
    )
    difference = phases[None, :, :] - phases[:, None, :]
    inter = np.sum(inter_coupling[:, :, None] * np.sin(difference), axis=1)
    return np.asarray(omega + intra + inter, dtype=np.float64)


def multiplex_field(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    intra_coupling: NDArray[np.float64],
    inter_coupling: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""The multiplex Kuramoto vector field ``θ̇`` (shape ``(L, N)``).

    Parameters
    ----------
    phases : numpy.ndarray
        The phases ``θ`` (shape ``(L, N)``, ``L ≥ 1`` layers of ``N ≥ 2`` nodes).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (shape ``(L, N)``).
    intra_coupling : numpy.ndarray
        The per-layer intra-layer coupling ``A`` (shape ``(L, N, N)``).
    inter_coupling : numpy.ndarray
        The inter-layer coupling ``B`` (shape ``(L, L)``).

    Returns
    -------
    numpy.ndarray
        The phase velocities ``θ̇`` (shape ``(L, N)``).

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    intra = np.ascontiguousarray(intra_coupling, dtype=np.float64)
    inter = np.ascontiguousarray(inter_coupling, dtype=np.float64)
    layers, _ = _validate(angle, frequencies, intra, inter)
    return _field(angle, frequencies, intra, inter, layers)


def multiplex_jacobian(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    intra_coupling: NDArray[np.float64],
    inter_coupling: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""The ``(LN, LN)`` Jacobian of the multiplex field (row-major ``(layer, node)`` order).

    Parameters
    ----------
    phases, omega, intra_coupling, inter_coupling
        As for :func:`multiplex_field` (the Jacobian is independent of ``ω``; it is validated for
        contract consistency).

    Returns
    -------
    numpy.ndarray
        The block Jacobian: intra-layer Jacobians on the layer diagonal, inter-layer Jacobians
        scattered over the shared nodes.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    intra = np.ascontiguousarray(intra_coupling, dtype=np.float64)
    inter = np.ascontiguousarray(inter_coupling, dtype=np.float64)
    layers, nodes = _validate(angle, frequencies, intra, inter)

    jacobian = np.zeros((layers * nodes, layers * nodes), dtype=np.float64)
    for layer in range(layers):
        block = networked_kuramoto_jacobian(angle[layer], intra[layer])
        jacobian[layer * nodes : (layer + 1) * nodes, layer * nodes : (layer + 1) * nodes] += block
    for node in range(nodes):
        replica_block = networked_kuramoto_jacobian(angle[:, node], inter)
        for row in range(layers):
            for column in range(layers):
                jacobian[row * nodes + node, column * nodes + node] += replica_block[row, column]
    return jacobian


def integrate_multiplex(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    intra_coupling: NDArray[np.float64],
    inter_coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> MultiplexTrajectory:
    r"""Integrate the multiplex Kuramoto network with classic RK4.

    Parameters
    ----------
    phases, omega, intra_coupling, inter_coupling
        As for :func:`multiplex_field`.
    dt : float
        The RK4 step (finite, ``> 0``).
    n_steps : int
        The number of steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    MultiplexTrajectory
        The sampled ``(n_steps + 1, L, N)`` phase trajectory.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    intra = np.ascontiguousarray(intra_coupling, dtype=np.float64)
    inter = np.ascontiguousarray(inter_coupling, dtype=np.float64)
    layers, nodes = _validate(angle, frequencies, intra, inter)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    trajectory = np.empty((n_steps + 1, layers, nodes), dtype=np.float64)
    trajectory[0] = angle
    current = angle
    for step in range(n_steps):
        k1 = _field(current, frequencies, intra, inter, layers)
        k2 = _field(current + 0.5 * dt * k1, frequencies, intra, inter, layers)
        k3 = _field(current + 0.5 * dt * k2, frequencies, intra, inter, layers)
        k4 = _field(current + dt * k3, frequencies, intra, inter, layers)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return MultiplexTrajectory(times=times, phases=trajectory)


def layer_order_parameters(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""The per-layer Kuramoto order-parameter magnitudes ``r_α = |⟨e^{iθ^α}⟩|`` (shape ``(L,)``).

    Parameters
    ----------
    phases : numpy.ndarray
        The phases ``θ`` (shape ``(L, N)``).

    Returns
    -------
    numpy.ndarray
        The ``(L,)`` per-layer order-parameter magnitudes.

    Raises
    ------
    ValueError
        If ``phases`` is not an ``(L, N)`` array.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    if angle.ndim != 2 or angle.shape[0] < 1 or angle.shape[1] < 1:
        raise ValueError("phases must be an (L >= 1, N >= 1) array")
    return np.ascontiguousarray(np.abs(np.mean(np.exp(1j * angle), axis=1)), dtype=np.float64)


def interlayer_synchronisation(phases: NDArray[np.float64]) -> float:
    r"""The mean per-node inter-layer coherence ``⟨|⟨e^{iθ^α}⟩_α|⟩_node``.

    This is ``1`` when every node shares a common phase across all layers (the replicas are locked).

    Parameters
    ----------
    phases : numpy.ndarray
        The phases ``θ`` (shape ``(L, N)``).

    Returns
    -------
    float
        The inter-layer synchronisation in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``phases`` is not an ``(L, N)`` array.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    if angle.ndim != 2 or angle.shape[0] < 1 or angle.shape[1] < 1:
        raise ValueError("phases must be an (L >= 1, N >= 1) array")
    return float(np.mean(np.abs(np.mean(np.exp(1j * angle), axis=0))))


__all__ = [
    "MultiplexTrajectory",
    "integrate_multiplex",
    "interlayer_synchronisation",
    "layer_order_parameters",
    "multiplex_field",
    "multiplex_jacobian",
]
