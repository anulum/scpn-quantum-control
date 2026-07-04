# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Heterogeneous Kuramoto coupling, additive mix of pairwise and higher-order terms
r"""Heterogeneous Kuramoto coupling — pairwise and higher-order interactions on one network.

A real oscillator network rarely couples through a single mechanism: pairwise links, triangular
(triadic) interactions and higher simplices can all act on the same nodes at once. Because every
Kuramoto coupling enters the phase velocity additively, the heterogeneous force is simply the sum
of its parts,

.. math::

    \dot{θ}_i = ω_i + \sum_{\text{terms } t} F^{(t)}_i(θ), \qquad
    J = \sum_{\text{terms } t} J^{(t)},

and the Jacobian is the sum of the per-term Jacobians. This module makes that additive structure
first class: a :class:`CouplingTerm` bundles a force with its Jacobian, the builders
:func:`pairwise_term`, :func:`hyperedge_term` and :func:`simplex_mean_field_term` wrap the
networked, structural-hypergraph and mean-field simplex couplings, and
:func:`heterogeneous_force` / :func:`heterogeneous_jacobian` sum a list of terms.
:func:`heterogeneous_force_components` returns the per-term contributions so the additive
decomposition can be read directly.

Because every contributing coupling depends only on phase differences, each term — and therefore
the sum — has zero Jacobian row sums (the global-phase Goldstone mode). This is an analysis layer
composing the existing polyglot forces and Jacobians, so it adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kuramoto_hyperedge import hyperedge_force, hyperedge_jacobian
from .kuramoto_simplex_mean_field import simplex_mean_field_force, simplex_mean_field_jacobian
from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian

#: A phase force ``F(θ)`` already closed over its coupling, mapping the phase vector to a force
#: vector of the same length.
PhaseForce = Callable[[NDArray[np.float64]], NDArray[np.float64]]

#: The Jacobian ``∂F/∂θ`` of a :data:`PhaseForce`, returning an ``(N, N)`` matrix.
PhaseJacobian = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class CouplingTerm:
    """One additive coupling contribution: a force, its Jacobian and a label.

    Attributes
    ----------
    force : callable
        The phase force ``F(θ)`` of this term (see :data:`PhaseForce`).
    jacobian : callable
        The Jacobian ``∂F/∂θ`` of this term (see :data:`PhaseJacobian`).
    label : str
        A human-readable label, used to identify the term in error messages.
    """

    force: PhaseForce
    jacobian: PhaseJacobian
    label: str


def pairwise_term(
    coupling_matrix: NDArray[np.float64], *, label: str = "pairwise"
) -> CouplingTerm:
    """Return a :class:`CouplingTerm` for the networked pairwise force on ``coupling_matrix``.

    Wraps :func:`~oscillatools.accel.networked_kuramoto.networked_kuramoto_force` and its
    Jacobian, bound to the ``(N, N)`` coupling matrix.
    """
    weights = np.ascontiguousarray(coupling_matrix, dtype=np.float64)
    return CouplingTerm(
        force=lambda theta: networked_kuramoto_force(theta, weights),
        jacobian=lambda theta: networked_kuramoto_jacobian(theta, weights),
        label=label,
    )


def hyperedge_term(
    hyperedges: Sequence[Sequence[int]],
    coupling: float | Sequence[float],
    *,
    label: str = "hyperedge",
) -> CouplingTerm:
    """Return a :class:`CouplingTerm` for the structural hypergraph force on ``hyperedges``.

    Wraps :func:`~oscillatools.accel.kuramoto_hyperedge.hyperedge_force` and its Jacobian,
    bound to the hyperedge list and its weights.
    """
    edges = [tuple(edge) for edge in hyperedges]
    return CouplingTerm(
        force=lambda theta: hyperedge_force(theta, edges, coupling),
        jacobian=lambda theta: hyperedge_jacobian(theta, edges, coupling),
        label=label,
    )


def simplex_mean_field_term(
    coupling: float, order: int, *, label: str | None = None
) -> CouplingTerm:
    """Return a :class:`CouplingTerm` for the all-to-all order-``order`` simplex mean field.

    Wraps :func:`~oscillatools.accel.kuramoto_simplex_mean_field.simplex_mean_field_force`
    and its Jacobian, bound to the coupling strength and the simplex order.
    """
    resolved_label = label if label is not None else f"simplex-order-{order}"
    return CouplingTerm(
        force=lambda theta: simplex_mean_field_force(theta, coupling, order),
        jacobian=lambda theta: simplex_mean_field_jacobian(theta, coupling, order),
        label=resolved_label,
    )


def _validate_terms(phases: NDArray[np.float64], terms: Sequence[CouplingTerm]) -> None:
    """Validate that ``phases`` is a non-empty vector and ``terms`` is non-empty."""
    if phases.ndim != 1 or phases.size < 1:
        raise ValueError("theta must be a non-empty one-dimensional array")
    if len(terms) < 1:
        raise ValueError("terms must contain at least one coupling term")


def heterogeneous_force_components(
    theta: NDArray[np.float64], terms: Sequence[CouplingTerm]
) -> list[NDArray[np.float64]]:
    """Return the per-term force contributions ``[F^{(t)}(θ)]`` (the additive decomposition).

    Each contribution is the force of one :class:`CouplingTerm` at ``theta``; their sum is
    :func:`heterogeneous_force`. Reading them separately exposes how much each coupling mechanism
    contributes.

    Parameters
    ----------
    theta : numpy.ndarray
        The oscillator phases ``θ`` (one-dimensional, length ``N``).
    terms : sequence of CouplingTerm
        The coupling terms (at least one).

    Returns
    -------
    list of numpy.ndarray
        One length-``N`` force vector per term, in order.

    Raises
    ------
    ValueError
        If ``theta`` is malformed, ``terms`` is empty, or a term's force has the wrong length.
    """
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    _validate_terms(phases, terms)
    components: list[NDArray[np.float64]] = []
    for term in terms:
        contribution = np.ascontiguousarray(term.force(phases), dtype=np.float64)
        if contribution.shape != (phases.size,):
            raise ValueError(
                f"term '{term.label}' force returned shape {contribution.shape}, "
                f"expected {(phases.size,)}"
            )
        components.append(contribution)
    return components


def heterogeneous_force(
    theta: NDArray[np.float64], terms: Sequence[CouplingTerm]
) -> NDArray[np.float64]:
    r"""Return the heterogeneous force ``F_i = Σ_t F^{(t)}_i(θ)`` summed over coupling terms.

    Parameters
    ----------
    theta : numpy.ndarray
        The oscillator phases ``θ`` (one-dimensional, length ``N``).
    terms : sequence of CouplingTerm
        The coupling terms (at least one); see :func:`pairwise_term`, :func:`hyperedge_term`,
        :func:`simplex_mean_field_term`.

    Returns
    -------
    numpy.ndarray
        The combined force on each oscillator (length ``N``).

    Raises
    ------
    ValueError
        If ``theta`` is malformed, ``terms`` is empty, or a term's force has the wrong length.
    """
    components = heterogeneous_force_components(theta, terms)
    return np.asarray(np.sum(components, axis=0), dtype=np.float64)


def heterogeneous_jacobian(
    theta: NDArray[np.float64], terms: Sequence[CouplingTerm]
) -> NDArray[np.float64]:
    r"""Return the heterogeneous Jacobian ``J = Σ_t J^{(t)}`` summed over coupling terms.

    Parameters
    ----------
    theta : numpy.ndarray
        The oscillator phases ``θ`` (one-dimensional, length ``N``).
    terms : sequence of CouplingTerm
        The coupling terms (at least one).

    Returns
    -------
    numpy.ndarray
        The combined ``(N, N)`` Jacobian.

    Raises
    ------
    ValueError
        If ``theta`` is malformed, ``terms`` is empty, or a term's Jacobian has the wrong shape.
    """
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    _validate_terms(phases, terms)
    total = np.zeros((phases.size, phases.size), dtype=np.float64)
    for term in terms:
        contribution = np.ascontiguousarray(term.jacobian(phases), dtype=np.float64)
        if contribution.shape != (phases.size, phases.size):
            raise ValueError(
                f"term '{term.label}' Jacobian returned shape {contribution.shape}, "
                f"expected {(phases.size, phases.size)}"
            )
        total += contribution
    return total


__all__ = [
    "CouplingTerm",
    "PhaseForce",
    "PhaseJacobian",
    "heterogeneous_force",
    "heterogeneous_force_components",
    "heterogeneous_jacobian",
    "hyperedge_term",
    "pairwise_term",
    "simplex_mean_field_term",
]
