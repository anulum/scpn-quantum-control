# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Persistent Homology
"""Tests for persistent homology of phase configurations."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.persistent_homology import (
    _RIPSER_AVAILABLE,
    PersistenceResult,
    compute_persistence,
    phase_distance_matrix,
)

pytestmark = pytest.mark.skipif(not _RIPSER_AVAILABLE, reason="ripser not installed")


class TestPhaseDistanceMatrix:
    def test_symmetric(self):
        theta = np.array([0.0, 0.5, 1.0, 1.5])
        D = phase_distance_matrix(theta)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_zero_diagonal(self):
        theta = np.array([0.0, 0.5, 1.0])
        D = phase_distance_matrix(theta)
        np.testing.assert_allclose(np.diag(D), 0.0)

    def test_bounded(self):
        theta = np.random.default_rng(42).uniform(0, 2 * np.pi, 8)
        D = phase_distance_matrix(theta)
        assert np.all(D >= 0)
        assert np.all(D <= 2.0)


class TestComputePersistence:
    def test_returns_result(self):
        theta = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        pr = compute_persistence(theta)
        assert isinstance(pr, PersistenceResult)

    def test_synchronized_zero_h1(self):
        theta = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
        pr = compute_persistence(theta)
        assert pr.n_h1 == 0

    def test_n_oscillators(self):
        theta = np.zeros(8)
        pr = compute_persistence(theta)
        assert pr.n_oscillators == 8

    def test_p_h1_bounded(self):
        theta = np.random.default_rng(42).uniform(0, 2 * np.pi, 10)
        pr = compute_persistence(theta, persistence_threshold=0.01)
        assert 0 <= pr.p_h1 <= 1.0


def test_persistence_all_pi():
    """All phases at pi — should have some structure."""
    theta = np.full(8, np.pi)
    pr = compute_persistence(theta)
    assert pr.n_oscillators == 8


def test_persistence_random_phases():
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, 12)
    pr = compute_persistence(theta, persistence_threshold=0.01)
    assert 0 <= pr.p_h1 <= 1.0
    assert pr.n_oscillators == 12


def test_persistence_2_oscillators():
    theta = np.array([0.0, np.pi])
    pr = compute_persistence(theta)
    assert pr.n_oscillators == 2


def test_persistence_threshold_effect():
    """Higher threshold → fewer persistent features."""
    theta = np.random.default_rng(42).uniform(0, 2 * np.pi, 8)
    pr_low = compute_persistence(theta, persistence_threshold=0.01)
    pr_high = compute_persistence(theta, persistence_threshold=0.5)
    assert pr_high.n_h1 <= pr_low.n_h1
