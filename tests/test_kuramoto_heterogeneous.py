# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the heterogeneous (mixed-order) Kuramoto coupling
r"""Tests for :mod:`scpn_quantum_control.accel.kuramoto_heterogeneous`.

The heterogeneous coupling is checked for the additive decomposition that is the roadmap
acceptance: the combined force equals the sum of its per-term contributions and each term matches
the standalone coupling it wraps; the combined Jacobian equals the sum of the term Jacobians and a
central finite difference of the combined force, with zero row sums. The term builders and the
malformed-input branches are covered explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.accel.kuramoto_heterogeneous import (
    CouplingTerm,
    heterogeneous_force,
    heterogeneous_force_components,
    heterogeneous_jacobian,
    hyperedge_term,
    pairwise_term,
    simplex_mean_field_term,
)
from scpn_quantum_control.accel.kuramoto_hyperedge import hyperedge_force, hyperedge_jacobian
from scpn_quantum_control.accel.kuramoto_simplex_mean_field import (
    simplex_mean_field_force,
    simplex_mean_field_jacobian,
)
from scpn_quantum_control.accel.networked_kuramoto import (
    networked_kuramoto_force,
    networked_kuramoto_jacobian,
)


def _symmetric_coupling(count, seed):
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.normal(0.0, 0.5, (count, count)), 1)
    return upper + upper.T


def _mixed_terms(coupling):
    return [
        pairwise_term(coupling),
        hyperedge_term([(0, 1, 2), (1, 3, 4, 5)], 0.7),
        simplex_mean_field_term(1.1, 3),
    ]


# --------------------------------------------------------------------------- term builders


def test_pairwise_term_wraps_networked_coupling() -> None:
    coupling = _symmetric_coupling(5, 0)
    theta = np.random.default_rng(1).uniform(0.0, 2.0 * np.pi, 5)
    term = pairwise_term(coupling)
    assert term.label == "pairwise"
    assert np.allclose(term.force(theta), networked_kuramoto_force(theta, coupling))
    assert np.allclose(term.jacobian(theta), networked_kuramoto_jacobian(theta, coupling))


def test_hyperedge_term_wraps_structural_coupling() -> None:
    theta = np.random.default_rng(2).uniform(0.0, 2.0 * np.pi, 6)
    edges = [(0, 1, 2), (2, 3, 4, 5)]
    term = hyperedge_term(edges, [0.8, -0.4], label="triangles")
    assert term.label == "triangles"
    assert np.allclose(term.force(theta), hyperedge_force(theta, edges, [0.8, -0.4]))
    assert np.allclose(term.jacobian(theta), hyperedge_jacobian(theta, edges, [0.8, -0.4]))


def test_simplex_term_wraps_mean_field_and_labels_by_order() -> None:
    theta = np.random.default_rng(3).uniform(0.0, 2.0 * np.pi, 7)
    term = simplex_mean_field_term(1.3, 3)
    assert term.label == "simplex-order-3"
    assert np.allclose(term.force(theta), simplex_mean_field_force(theta, 1.3, 3))
    assert np.allclose(term.jacobian(theta), simplex_mean_field_jacobian(theta, 1.3, 3))


def test_simplex_term_accepts_custom_label() -> None:
    assert simplex_mean_field_term(1.0, 2, label="triadic").label == "triadic"


def test_coupling_term_can_be_constructed_directly() -> None:
    term = CouplingTerm(
        force=lambda theta: np.zeros(theta.size),
        jacobian=lambda theta: np.zeros((theta.size, theta.size)),
        label="zero",
    )
    assert term.label == "zero"
    assert np.array_equal(term.force(np.zeros(3)), np.zeros(3))


# --------------------------------------------------------------------------- additive decomposition


def test_force_is_sum_of_components() -> None:
    coupling = _symmetric_coupling(6, 4)
    theta = np.random.default_rng(5).uniform(0.0, 2.0 * np.pi, 6)
    terms = _mixed_terms(coupling)
    components = heterogeneous_force_components(theta, terms)
    assert len(components) == 3
    assert np.allclose(heterogeneous_force(theta, terms), sum(components))


def test_force_equals_explicit_term_sum() -> None:
    coupling = _symmetric_coupling(6, 6)
    theta = np.random.default_rng(7).uniform(0.0, 2.0 * np.pi, 6)
    terms = _mixed_terms(coupling)
    explicit = (
        networked_kuramoto_force(theta, coupling)
        + hyperedge_force(theta, [(0, 1, 2), (1, 3, 4, 5)], 0.7)
        + simplex_mean_field_force(theta, 1.1, 3)
    )
    assert np.allclose(heterogeneous_force(theta, terms), explicit)


def test_components_match_individual_terms() -> None:
    coupling = _symmetric_coupling(6, 8)
    theta = np.random.default_rng(9).uniform(0.0, 2.0 * np.pi, 6)
    terms = _mixed_terms(coupling)
    components = heterogeneous_force_components(theta, terms)
    for term, contribution in zip(terms, components):
        assert np.allclose(contribution, term.force(theta))


def test_single_term_reduces_to_that_term() -> None:
    coupling = _symmetric_coupling(5, 10)
    theta = np.random.default_rng(11).uniform(0.0, 2.0 * np.pi, 5)
    assert np.allclose(
        heterogeneous_force(theta, [pairwise_term(coupling)]),
        networked_kuramoto_force(theta, coupling),
    )


# --------------------------------------------------------------------------- combined Jacobian


def test_jacobian_is_sum_of_term_jacobians() -> None:
    coupling = _symmetric_coupling(6, 12)
    theta = np.random.default_rng(13).uniform(0.0, 2.0 * np.pi, 6)
    terms = _mixed_terms(coupling)
    summed = sum(term.jacobian(theta) for term in terms)
    assert np.allclose(heterogeneous_jacobian(theta, terms), summed)


def test_jacobian_matches_central_difference() -> None:
    coupling = _symmetric_coupling(6, 14)
    theta = np.random.default_rng(15).uniform(0.0, 2.0 * np.pi, 6)
    terms = _mixed_terms(coupling)
    analytic = heterogeneous_jacobian(theta, terms)
    count = theta.size
    numeric = np.empty((count, count))
    step = 1e-6
    for column in range(count):
        forward = theta.copy()
        forward[column] += step
        backward = theta.copy()
        backward[column] -= step
        numeric[:, column] = (
            heterogeneous_force(forward, terms) - heterogeneous_force(backward, terms)
        ) / (2.0 * step)
    assert np.allclose(analytic, numeric, atol=1e-7)


def test_jacobian_rows_sum_to_zero() -> None:
    coupling = _symmetric_coupling(6, 16)
    theta = np.random.default_rng(17).uniform(0.0, 2.0 * np.pi, 6)
    jacobian = heterogeneous_jacobian(theta, _mixed_terms(coupling))
    assert np.allclose(jacobian.sum(axis=1), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- validation


def test_force_rejects_empty_terms() -> None:
    with pytest.raises(ValueError, match="at least one coupling term"):
        heterogeneous_force(np.zeros(3), [])


def test_force_rejects_non_vector_theta() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        heterogeneous_force(np.zeros((2, 2)), [pairwise_term(np.zeros((2, 2)))])


def test_jacobian_rejects_empty_terms() -> None:
    with pytest.raises(ValueError, match="at least one coupling term"):
        heterogeneous_jacobian(np.zeros(3), [])


def test_jacobian_rejects_empty_theta() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        heterogeneous_jacobian(np.empty(0), [pairwise_term(np.zeros((0, 0)))])


def test_force_rejects_term_with_wrong_output_length() -> None:
    bad = CouplingTerm(
        force=lambda theta: np.zeros(theta.size + 1),
        jacobian=lambda theta: np.zeros((theta.size, theta.size)),
        label="bad-force",
    )
    with pytest.raises(ValueError, match="term 'bad-force' force returned shape"):
        heterogeneous_force(np.zeros(3), [bad])


def test_jacobian_rejects_term_with_wrong_output_shape() -> None:
    bad = CouplingTerm(
        force=lambda theta: np.zeros(theta.size),
        jacobian=lambda theta: np.zeros((theta.size, theta.size + 1)),
        label="bad-jac",
    )
    with pytest.raises(ValueError, match="term 'bad-jac' Jacobian returned shape"):
        heterogeneous_jacobian(np.zeros(3), [bad])
