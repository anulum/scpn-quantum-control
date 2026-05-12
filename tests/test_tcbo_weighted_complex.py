# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for TCBO coupling-weighted complex
"""Tests for the TCBO coupling-weighted simplicial-complex reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.tcbo_weighted_complex import (
    coupling_weighted_edge_matrix,
    tcbo_weighted_complex,
    tcbo_weighted_threshold_scan,
)


def test_edge_weights_follow_kij_abs_cos_phase_difference() -> None:
    K = np.array(
        [
            [0.0, 2.0, 3.0],
            [2.0, 0.0, 4.0],
            [3.0, 4.0, 0.0],
        ]
    )
    theta = np.array([0.0, 0.0, np.pi / 2.0])

    weights = coupling_weighted_edge_matrix(K, theta, normalise=False)

    assert weights[0, 1] == pytest.approx(2.0)
    assert weights[0, 2] == pytest.approx(0.0, abs=1e-12)
    assert weights[1, 2] == pytest.approx(0.0, abs=1e-12)


def test_square_cycle_has_one_unfilled_h1_cycle() -> None:
    K = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    theta = np.zeros(4)

    result = tcbo_weighted_complex(K, theta, threshold=0.5)

    assert result.n_edges == 4
    assert result.n_triangles == 0
    assert result.beta_1 == 1
    assert result.p_h1 == pytest.approx(1.0 / 3.0)


def test_filled_triangle_has_no_h1_cycle() -> None:
    K = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    theta = np.zeros(3)

    result = tcbo_weighted_complex(K, theta, threshold=0.5)

    assert result.n_edges == 3
    assert result.n_triangles == 1
    assert result.beta_1 == 0
    assert result.p_h1 == 0.0


def test_threshold_scan_reports_target_distance_without_promotion() -> None:
    K = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    theta = np.zeros(4)

    scan = tcbo_weighted_threshold_scan(K, theta, thresholds=np.array([0.25, 0.75]))

    assert scan.target_p_h1 == pytest.approx(0.72)
    assert scan.best_threshold in {0.25, 0.75}
    assert scan.promotes_target is False
    assert len(scan.results) == 2


def test_rejects_non_symmetric_coupling_matrix() -> None:
    K = np.array([[0.0, 1.0], [0.5, 0.0]])
    theta = np.zeros(2)

    with pytest.raises(ValueError, match="K must be symmetric"):
        tcbo_weighted_complex(K, theta)


def test_rejects_phase_shape_mismatch() -> None:
    K = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="theta must match"):
        tcbo_weighted_complex(K, np.zeros(3))
