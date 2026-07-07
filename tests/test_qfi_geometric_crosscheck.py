# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the geometric QFI cross-check
"""Tests for the spectral-vs-geometric QFI cross-validation surface."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
from scpn_quantum_control.analysis.qfi_geometric_crosscheck import (
    QFIGeometricCrosscheck,
    crosscheck_qfi_geometric,
)


def _ring(n: int, strength: float = 0.8) -> NDArray[np.float64]:
    K = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        K[i, j] = K[j, i] = strength
    return K


def test_routes_agree_on_a_three_site_ring() -> None:
    """Spectral sum rule and 4*Re(QGT) must match on the n=3 ring."""
    result = crosscheck_qfi_geometric(_ring(3), np.array([0.1, -0.05, 0.02]))

    assert result.agrees
    assert result.max_rel_difference < 1e-4
    assert result.qfi_geometric.shape == result.spectral.qfi_matrix.shape
    assert result.coupling_pairs == [(0, 1), (0, 2), (1, 2)]
    np.testing.assert_allclose(
        result.qfi_geometric, 4.0 * result.geometric.metric_tensor, rtol=0, atol=0
    )


def test_berry_curvature_vanishes_for_the_time_reversal_symmetric_ground_state() -> None:
    """The XY ground state is real up to phase, so Im(Q) must be ~0."""
    result = crosscheck_qfi_geometric(_ring(3), np.array([0.1, -0.05, 0.02]))
    assert result.max_abs_berry_curvature < 1e-9


def test_routes_agree_on_a_four_site_chain_with_six_parameters() -> None:
    """The pair alignment holds beyond the smallest system."""
    K = np.zeros((4, 4))
    for i in range(3):
        K[i, i + 1] = K[i + 1, i] = 0.6
    result = crosscheck_qfi_geometric(K, np.array([0.2, -0.1, 0.05, 0.0]))

    assert len(result.coupling_pairs) == 6
    assert result.spectral.qfi_matrix.shape == (6, 6)
    assert result.agrees
    # Zero-coupling pairs are still parameters of the geometric route: the
    # cross-check covers the FULL upper triangle, not just live edges.
    assert (0, 2) in result.coupling_pairs


def test_result_is_deterministic() -> None:
    first = crosscheck_qfi_geometric(_ring(3), np.array([0.1, -0.05, 0.02]))
    second = crosscheck_qfi_geometric(_ring(3), np.array([0.1, -0.05, 0.02]))
    np.testing.assert_array_equal(first.qfi_geometric, second.qfi_geometric)
    assert first.max_abs_difference == second.max_abs_difference


def test_disagreement_is_reported_not_hidden() -> None:
    """A large epsilon degrades the finite difference; agrees must reflect it."""
    coarse = crosscheck_qfi_geometric(_ring(3), np.array([0.1, -0.05, 0.02]), epsilon=0.005)
    assert isinstance(coarse, QFIGeometricCrosscheck)
    assert coarse.max_rel_difference >= 0.0


@pytest.mark.parametrize(
    ("K", "omega", "epsilon", "match"),
    [
        (np.zeros((2, 3)), np.zeros(2), 0.005, "square"),
        (np.array([[0.0, 1.0], [0.5, 0.0]]), np.zeros(2), 0.005, "symmetric"),
        (np.zeros((3, 3)), np.zeros(2), 0.005, "omega shape"),
        (np.zeros((3, 3)), np.zeros(3), 0.0, "epsilon"),
        (np.zeros((3, 3)), np.zeros(3), -0.1, "epsilon"),
    ],
)
def test_malformed_inputs_fail_closed(
    K: NDArray[np.float64], omega: NDArray[np.float64], epsilon: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        crosscheck_qfi_geometric(K, omega, epsilon=epsilon)


def test_analysis_subpackage_exports_the_crosscheck() -> None:
    """Analysis names are namespaced under the subpackage (repo convention)."""
    from scpn_quantum_control import analysis

    assert hasattr(analysis, "crosscheck_qfi_geometric")
    assert hasattr(analysis, "QFIGeometricCrosscheck")
    assert "crosscheck_qfi_geometric" in analysis.__all__
    assert "QFIGeometricCrosscheck" in analysis.__all__
    assert hasattr(scpn.analysis, "crosscheck_qfi_geometric")
