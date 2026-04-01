# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Crypto Percolation
"""Tests for percolation: entanglement percolation on K_nm graph."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge import build_knm_paper27
from scpn_quantum_control.crypto.percolation import (
    active_channel_graph,
    key_rate_per_channel,
    percolation_threshold,
)


def test_percolation_threshold_positive():
    K = build_knm_paper27(L=8)
    threshold = percolation_threshold(K)
    assert threshold > 0
    assert threshold < K.max()


def test_active_channels_above_threshold():
    K = build_knm_paper27(L=8)
    threshold = percolation_threshold(K)
    channels = active_channel_graph(K, threshold)
    assert len(channels) > 0
    for i, j, w in channels:
        assert w >= threshold
        assert i < j


def test_all_channels_at_zero_threshold():
    K = build_knm_paper27(L=4)
    channels = active_channel_graph(K, threshold=0.0)
    # 4 nodes → 6 pairs
    assert len(channels) == 6


def test_no_channels_above_max():
    K = build_knm_paper27(L=4)
    channels = active_channel_graph(K, threshold=K.max() + 1)
    assert len(channels) == 0


def test_key_rate_symmetric():
    conc = np.array([[0, 0.5, 0.1], [0.5, 0, 0.3], [0.1, 0.3, 0]])
    rates = key_rate_per_channel(conc)
    assert np.allclose(rates, rates.T)


def test_key_rate_zero_for_zero_concurrence():
    conc = np.zeros((3, 3))
    rates = key_rate_per_channel(conc)
    assert np.allclose(rates, 0)


def test_key_rate_positive_for_high_concurrence():
    conc = np.array([[0, 0.9], [0.9, 0]])
    rates = key_rate_per_channel(conc)
    assert rates[0, 1] > 0


def test_active_channels_upper_triangle():
    """active_channel_graph returns unique pairs (i<j)."""
    K = build_knm_paper27(L=4)
    channels = active_channel_graph(K, threshold=0.1)
    for c in channels:
        assert c[0] < c[1]  # upper triangle only
        assert len(c) == 3  # (i, j, weight)


def test_key_rate_shape():
    conc = np.zeros((4, 4))
    rates = key_rate_per_channel(conc)
    assert rates.shape == (4, 4)


def test_key_rate_diagonal_zero():
    conc = np.array([[0, 0.8], [0.8, 0]])
    rates = key_rate_per_channel(conc)
    assert rates[0, 0] == 0 and rates[1, 1] == 0


def test_active_channels_threshold_sweep():
    K = build_knm_paper27(L=4)
    n_low = len(active_channel_graph(K, threshold=0.01))
    n_high = len(active_channel_graph(K, threshold=0.5))
    assert n_low >= n_high
