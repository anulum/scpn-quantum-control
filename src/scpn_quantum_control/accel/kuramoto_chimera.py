# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera and metastability diagnostics for a Kuramoto trajectory
r"""Chimera and metastability diagnostics for a community-structured Kuramoto trajectory.

A chimera state is the coexistence of a synchronised and a desynchronised population in an
otherwise symmetric oscillator ensemble; metastability is the restless wandering of the
collective coherence in time. Following Shanahan (2010) both are read off the per-community
order parameters of a phase trajectory. For a partition of the oscillators into communities
``c`` the community order parameter is

.. math::

    \Phi_c(t) = \Big| \frac{1}{N_c} \sum_{k \in c} e^{i\theta_k(t)} \Big| \in [0, 1] ,

and the diagnostics are

.. math::

    \chi = \big\langle \operatorname{Var}_c \Phi_c(t) \big\rangle_t , \qquad
    M = \operatorname{Var}_t R(t) , \qquad
    \lambda = \big\langle \operatorname{Var}_t \Phi_c(t) \big\rangle_c ,

where :math:`R(t) = |\langle e^{i\theta}\rangle|` is the global order parameter. The chimera
index :math:`\chi` is the time-averaged variance of coherence *across* communities — it is
zero when every community shares the same coherence (full synchrony or uniform incoherence)
and positive when some communities lock while others drift. The metastability :math:`M` is
the temporal variance of the global coherence — zero for a stationary state and positive
when :math:`R` wanders. The community metastability :math:`\lambda` is Shanahan's companion,
the community-averaged temporal variance, which sees per-community wandering that the global
:math:`R` can average away.

The chimera and metastability indices are differentiable away from the incoherent points
where a community (or the whole ensemble) has zero coherence and its mean phase — hence the
gradient — is undefined; there the zero subgradient is returned, mirroring the convention of
:func:`~scpn_quantum_control.accel.order_parameter_observables.order_parameter_gradient`.

This is a pure-Python analysis layer over the order-parameter observable; it adds no compute
kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Coherence magnitude below which a community (or the ensemble) is treated as incoherent: its
# mean phase, and therefore the per-phase gradient, is undefined and the zero subgradient is used.
_COHERENCE_FLOOR = 1e-12

CommunityList = Sequence[NDArray[np.int_] | Sequence[int]]


def _validate_trajectory(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a contiguous ``(T, N)`` trajectory after validating its shape.

    Raises
    ------
    ValueError
        If ``phases`` is not a two-dimensional array with at least one time sample and one
        oscillator.
    """
    trajectory = np.ascontiguousarray(phases, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError(
            f"phases must be a two-dimensional (T, N) trajectory, got ndim {trajectory.ndim}"
        )
    if trajectory.shape[0] < 1:
        raise ValueError("phases must hold at least one time sample")
    if trajectory.shape[1] < 1:
        raise ValueError("phases must hold at least one oscillator")
    return trajectory


def _resolve_communities(communities: CommunityList, count: int) -> list[NDArray[np.int_]]:
    """Validate and normalise a community partition into a list of index arrays.

    Raises
    ------
    ValueError
        If no community is supplied, a community is not a non-empty one-dimensional integer
        index set, an index is out of range, or two communities share an oscillator.
    """
    if len(communities) == 0:
        raise ValueError("at least one community must be supplied")
    resolved: list[NDArray[np.int_]] = []
    seen: set[int] = set()
    for position, members in enumerate(communities):
        group = np.ascontiguousarray(members, dtype=np.int_)
        if group.ndim != 1 or group.size == 0:
            raise ValueError(f"community {position} must be a non-empty one-dimensional index set")
        if group.min() < 0 or group.max() >= count:
            raise ValueError(
                f"community {position} has indices outside [0, {count}) for {count} oscillators"
            )
        members_set = {int(index) for index in group}
        if len(members_set) != group.size:
            raise ValueError(f"community {position} repeats an oscillator index")
        overlap = members_set & seen
        if overlap:
            raise ValueError(f"community {position} shares oscillator(s) {sorted(overlap)}")
        seen |= members_set
        resolved.append(group)
    return resolved


def _coherence(
    phases: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(cos_mean, sin_mean, magnitude)`` of a ``(T, members)`` phase block over members."""
    cos_mean = np.cos(phases).mean(axis=1)
    sin_mean = np.sin(phases).mean(axis=1)
    return cos_mean, sin_mean, np.hypot(cos_mean, sin_mean)


def community_order_parameters(
    phases: NDArray[np.float64], communities: CommunityList
) -> NDArray[np.float64]:
    r"""Per-community order parameter :math:`\Phi_c(t)` along a trajectory.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    communities : sequence of array-like
        A partition of the oscillator indices into disjoint, non-empty communities.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(T, M)`` float64 array of the ``M`` community order parameters in
        ``[0, 1]`` at each of the ``T`` time samples.

    Raises
    ------
    ValueError
        If the trajectory or the community partition is malformed.
    """
    trajectory = _validate_trajectory(phases)
    groups = _resolve_communities(communities, trajectory.shape[1])
    columns = [_coherence(trajectory[:, group])[2] for group in groups]
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float64)


def chimera_index(phases: NDArray[np.float64], communities: CommunityList) -> float:
    r"""Chimera index :math:`\chi = \langle \operatorname{Var}_c \Phi_c(t)\rangle_t`.

    The time average of the population variance of the community order parameters across
    communities. It is zero when all communities share the same coherence (full synchrony or
    uniform incoherence) and positive for a chimera, where some communities lock while others
    drift.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    communities : sequence of array-like
        A partition of the oscillator indices into disjoint, non-empty communities.

    Returns
    -------
    float
        The chimera index; ``0.0`` for a single community.

    Raises
    ------
    ValueError
        If the trajectory or the community partition is malformed.
    """
    phi = community_order_parameters(phases, communities)
    return float(np.var(phi, axis=1).mean())


def community_metastability(phases: NDArray[np.float64], communities: CommunityList) -> float:
    r"""Community metastability :math:`\lambda = \langle \operatorname{Var}_t \Phi_c(t)\rangle_c`.

    Shanahan's companion to the chimera index: the community-averaged temporal variance of the
    community order parameters. It captures per-community wandering that the global order
    parameter can average away, and is zero for a stationary state.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    communities : sequence of array-like
        A partition of the oscillator indices into disjoint, non-empty communities.

    Returns
    -------
    float
        The community metastability index.

    Raises
    ------
    ValueError
        If the trajectory or the community partition is malformed.
    """
    phi = community_order_parameters(phases, communities)
    return float(np.var(phi, axis=0).mean())


def metastability_index(phases: NDArray[np.float64]) -> float:
    r"""Metastability index :math:`M = \operatorname{Var}_t R(t)`.

    The temporal (population) variance of the global Kuramoto order parameter. It is zero for
    a stationary collective state — full synchrony or steady partial synchrony — and grows as
    the global coherence wanders.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.

    Returns
    -------
    float
        The metastability index; ``0.0`` for a single time sample.

    Raises
    ------
    ValueError
        If ``phases`` is not a ``(T, N)`` array with ``T ≥ 1`` and ``N ≥ 1``.
    """
    trajectory = _validate_trajectory(phases)
    magnitude = _coherence(trajectory)[2]
    return float(np.var(magnitude))


def _coherence_gradient(
    block: NDArray[np.float64],
    cos_mean: NDArray[np.float64],
    sin_mean: NDArray[np.float64],
    magnitude: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Return :math:`\partial\Phi/\partial\theta` for a ``(T, members)`` block.

    Uses :math:`\partial\Phi/\partial\theta_j = (S\cos\theta_j - C\sin\theta_j)/(N_c\,\Phi)`
    with the zero subgradient where :math:`\Phi` is below the coherence floor.
    """
    size = block.shape[1]
    numerator = sin_mean[:, None] * np.cos(block) - cos_mean[:, None] * np.sin(block)
    safe = magnitude > _COHERENCE_FLOOR
    scale = np.zeros_like(magnitude)
    scale[safe] = 1.0 / (size * magnitude[safe])
    return np.ascontiguousarray(numerator * scale[:, None], dtype=np.float64)


def chimera_index_gradient(
    phases: NDArray[np.float64], communities: CommunityList
) -> NDArray[np.float64]:
    r"""Gradient of the chimera index with respect to the trajectory.

    Returns :math:`\partial\chi/\partial\theta_{t,j}`, shaped like the ``(T, N)`` trajectory.
    Each phase :math:`\theta_{t,j}` influences only its own community's order parameter at its
    own time, giving

    .. math::

        \frac{\partial\chi}{\partial\theta_{t,j}} = \frac{2}{T M}\,
            \big(\Phi_{c(j)}(t) - \bar\Phi(t)\big)\,
            \frac{\partial\Phi_{c(j)}(t)}{\partial\theta_{t,j}} ,

    with :math:`\bar\Phi(t)` the across-community mean. Oscillators outside every community,
    and time slices where the relevant community is incoherent, contribute zero (subgradient).

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    communities : sequence of array-like
        A partition of the oscillator indices into disjoint, non-empty communities.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(T, N)`` float64 gradient.

    Raises
    ------
    ValueError
        If the trajectory or the community partition is malformed.
    """
    trajectory = _validate_trajectory(phases)
    samples, count = trajectory.shape
    groups = _resolve_communities(communities, count)
    group_count = len(groups)
    phi = np.stack([_coherence(trajectory[:, group])[2] for group in groups], axis=1)
    phi_mean = phi.mean(axis=1)
    gradient = np.zeros_like(trajectory)
    for position, group in enumerate(groups):
        block = trajectory[:, group]
        cos_mean, sin_mean, magnitude = _coherence(block)
        local = _coherence_gradient(block, cos_mean, sin_mean, magnitude)
        coefficient = (2.0 / (samples * group_count)) * (phi[:, position] - phi_mean)
        gradient[:, group] += coefficient[:, None] * local
    return np.ascontiguousarray(gradient, dtype=np.float64)


def metastability_index_gradient(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Gradient of the metastability index with respect to the trajectory.

    Returns :math:`\partial M/\partial\theta_{t,j} = (2/T)\,(R(t) - \langle R\rangle)\,
    \partial R(t)/\partial\theta_{t,j}`, shaped like the ``(T, N)`` trajectory, with the
    global order-parameter gradient :math:`\partial R/\partial\theta_j = (1/N)\sin(\psi -
    \theta_j)`. Time slices where the ensemble is incoherent contribute the zero subgradient.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(T, N)`` float64 gradient.

    Raises
    ------
    ValueError
        If ``phases`` is not a ``(T, N)`` array with ``T ≥ 1`` and ``N ≥ 1``.
    """
    trajectory = _validate_trajectory(phases)
    samples = trajectory.shape[0]
    cos_mean, sin_mean, magnitude = _coherence(trajectory)
    local = _coherence_gradient(trajectory, cos_mean, sin_mean, magnitude)
    coefficient = (2.0 / samples) * (magnitude - float(magnitude.mean()))
    return np.ascontiguousarray(coefficient[:, None] * local, dtype=np.float64)


@dataclass(frozen=True)
class ChimeraDiagnostics:
    """Bundled chimera and metastability diagnostics for a trajectory.

    Attributes
    ----------
    community_order_parameters : numpy.ndarray
        The ``(T, M)`` per-community order parameters.
    chimera_index : float
        The time-averaged across-community coherence variance.
    metastability_index : float
        The temporal variance of the global order parameter.
    community_metastability : float
        The community-averaged temporal variance of the community order parameters.
    """

    community_order_parameters: NDArray[np.float64]
    chimera_index: float
    metastability_index: float
    community_metastability: float


def chimera_diagnostics(
    phases: NDArray[np.float64], communities: CommunityList
) -> ChimeraDiagnostics:
    """Compute the chimera and metastability diagnostics in a single pass.

    Evaluates the community order parameters once and derives the chimera index and community
    metastability from them, plus the global metastability from the whole-ensemble coherence.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    communities : sequence of array-like
        A partition of the oscillator indices into disjoint, non-empty communities.

    Returns
    -------
    ChimeraDiagnostics
        The bundled community order parameters, chimera index, metastability index and
        community metastability.

    Raises
    ------
    ValueError
        If the trajectory or the community partition is malformed.
    """
    trajectory = _validate_trajectory(phases)
    phi = community_order_parameters(trajectory, communities)
    magnitude = _coherence(trajectory)[2]
    return ChimeraDiagnostics(
        community_order_parameters=phi,
        chimera_index=float(np.var(phi, axis=1).mean()),
        metastability_index=float(np.var(magnitude)),
        community_metastability=float(np.var(phi, axis=0).mean()),
    )


__all__ = [
    "ChimeraDiagnostics",
    "chimera_diagnostics",
    "chimera_index",
    "chimera_index_gradient",
    "community_metastability",
    "community_order_parameters",
    "metastability_index",
    "metastability_index_gradient",
]
