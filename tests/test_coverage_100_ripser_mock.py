# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ripser/Persistent Homology Mock Tests
"""Mock-based tests for persistent homology and graph topology scan."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_quantum_control.analysis import graph_topology_scan as gts_mod
from scpn_quantum_control.analysis import persistent_homology as ph_mod


def _fake_ripser(D, maxdim=1, distance_matrix=False):
    """Return plausible persistence diagrams for any distance matrix."""
    n = D.shape[0]
    rng = np.random.default_rng(42)
    # H0: n-1 finite bars + 1 infinite
    h0 = np.column_stack(
        [
            rng.uniform(0, 0.1, n),
            np.concatenate([rng.uniform(0.2, 0.5, n - 1), [np.inf]]),
        ]
    )
    # H1: a few 1-cycles
    n_h1 = max(1, n // 4)
    births = rng.uniform(0.1, 0.3, n_h1)
    deaths = births + rng.uniform(0.05, 0.4, n_h1)
    h1 = np.column_stack([births, deaths])
    return {"dgms": [h0, h1]}


@pytest.fixture()
def mock_ripser(monkeypatch):
    """Patch ripser as available and provide a fake ripser function."""
    monkeypatch.setattr(ph_mod, "_RIPSER_AVAILABLE", True)
    monkeypatch.setattr(gts_mod, "_RIPSER_AVAILABLE", True)

    fake_ripser_mod = MagicMock(ripser=_fake_ripser)
    monkeypatch.setattr(ph_mod, "ripser", _fake_ripser, raising=False)
    monkeypatch.setitem(sys.modules, "ripser", fake_ripser_mod)
    yield


def test_compute_persistence(mock_ripser):
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    result = ph_mod.compute_persistence(theta, persistence_threshold=0.05)
    assert isinstance(result, ph_mod.PersistenceResult)
    assert result.n_oscillators == 8
    assert result.n_h0 > 0
    assert isinstance(result.p_h1, float)
    assert 0 <= result.p_h1 <= 1


def test_compute_persistence_raises_without_ripser(monkeypatch):
    monkeypatch.setattr(ph_mod, "_RIPSER_AVAILABLE", False)
    with pytest.raises(ImportError, match="ripser"):
        ph_mod.compute_persistence(np.zeros(4))


def test_phase_distance_matrix():
    theta = np.array([0, np.pi / 2, np.pi])
    D = ph_mod.phase_distance_matrix(theta)
    assert D.shape == (3, 3)
    np.testing.assert_allclose(D[0, 0], 0.0)
    np.testing.assert_allclose(D, D.T)


def test_p_h1_vs_temperature(mock_ripser, monkeypatch):
    K = np.array([[0, 0.5, 0.2], [0.5, 0, 0.3], [0.2, 0.3, 0]])
    result = ph_mod.p_h1_vs_temperature(
        K,
        t_range=(0.1, 0.5),
        n_temps=3,
        n_thermalize=10,
        n_samples=2,
        persistence_threshold=0.05,
        seed=42,
    )
    assert "temperature" in result
    assert "p_h1_mean" in result
    assert len(result["temperature"]) == 3


def test_p_h1_vs_temperature_raises_without_ripser(monkeypatch):
    monkeypatch.setattr(ph_mod, "_RIPSER_AVAILABLE", False)
    with pytest.raises(ImportError, match="ripser"):
        ph_mod.p_h1_vs_temperature(np.eye(3))


def test_erdos_renyi_coupling():
    K = gts_mod._erdos_renyi_coupling(8, 0.5, strength=0.4, seed=42)
    assert K.shape == (8, 8)
    np.testing.assert_allclose(K, K.T)


def test_watts_strogatz_coupling():
    K = gts_mod._watts_strogatz_coupling(8, k=4, beta=0.3, seed=42)
    assert K.shape == (8, 8)
    np.testing.assert_allclose(K, K.T)


def test_ring_coupling():
    K = gts_mod._ring_coupling(8, k=2, strength=0.5)
    assert K.shape == (8, 8)
    np.testing.assert_allclose(K, K.T)
    assert np.sum(K > 0) == 8 * 2 * 2  # each node has 2k neighbours


def test_measure_p_h1_at_transition(mock_ripser):
    K = np.array([[0, 0.5], [0.5, 0]])
    mean, std = gts_mod._measure_p_h1_at_transition(
        K,
        n_thermalize=5,
        n_samples=2,
        seed=42,
    )
    assert isinstance(mean, float)
    assert isinstance(std, float)


def test_scan_graph_topologies_raises_without_ripser(monkeypatch):
    monkeypatch.setattr(gts_mod, "_RIPSER_AVAILABLE", False)
    with pytest.raises(ImportError, match="ripser"):
        gts_mod.scan_graph_topologies()


def test_scan_graph_topologies(mock_ripser, monkeypatch):
    # Mock _measure_p_h1_at_transition to avoid slow MC simulation
    monkeypatch.setattr(
        gts_mod,
        "_measure_p_h1_at_transition",
        lambda K, **kw: (0.5, 0.1),
    )
    # n must be >= k+2 (k=4 default) to avoid infinite rewiring loop in Watts-Strogatz
    results = gts_mod.scan_graph_topologies(n=8, n_samples=1, seed=42)
    assert len(results) > 0
    for r in results:
        assert isinstance(r, gts_mod.GraphP_H1_Result)
        assert r.n_nodes == 8
