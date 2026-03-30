# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Ssgf W Adapter
"""Tests for SSGF W adaptation from quantum feedback."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.ssgf_w_adapter import (
    WAdaptResult,
    adapt_w_from_quantum,
)


def _test_system():
    W = np.array([[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]])
    theta = np.array([0.0, 0.3, 0.6])
    return W, theta


class TestAdaptWFromQuantum:
    def test_returns_result(self):
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        assert isinstance(result, WAdaptResult)

    def test_w_shape_preserved(self):
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        assert result.W_updated.shape == W.shape

    def test_w_symmetric(self):
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        np.testing.assert_allclose(result.W_updated, result.W_updated.T, atol=1e-12)

    def test_w_non_negative(self):
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta, min_coupling=0.0)
        assert np.all(result.W_updated >= 0)

    def test_w_zero_diagonal(self):
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        np.testing.assert_allclose(np.diag(result.W_updated), 0.0)

    def test_r_global_bounded(self):
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        assert 0 <= result.r_global <= 1.0

    def test_correlators_shape(self):
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        assert result.correlators.shape == (3, 3)

    def test_small_learning_rate_small_update(self):
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta, learning_rate=1e-6)
        assert result.max_update < 0.01

    def test_w_changes(self):
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta, learning_rate=0.1)
        assert not np.allclose(result.W_updated, W)
