# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Graph Topology Scan
"""Multi-angle tests for graph topology → p_h1 scan.

Angles:
  1. Graph generator structural invariants (symmetry, degree, bounds)
  2. Graph generator parametric sensitivity
  3. p_h1 measurement pipeline (mock ripser)
  4. scan_graph_topologies orchestration (mock ripser)
  5. Edge cases and reproducibility
  6. Pipeline wiring (graph → Hamiltonian)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from scpn_quantum_control.analysis.graph_topology_scan import (
    GraphP_H1_Result,
    _erdos_renyi_coupling,
    _measure_p_h1_at_transition,
    _ring_coupling,
    _watts_strogatz_coupling,
    scan_graph_topologies,
)


class TestGraphGenerators:
    def test_erdos_renyi_symmetric(self):
        K = _erdos_renyi_coupling(16, 0.5)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_erdos_renyi_density(self):
        K = _erdos_renyi_coupling(16, 1.0)
        assert np.all(K[np.eye(16) == 0] > 0)

    def test_erdos_renyi_empty(self):
        K = _erdos_renyi_coupling(16, 0.0)
        assert np.sum(K) == 0.0

    def test_ring_degree(self):
        K = _ring_coupling(16, k=2)
        degrees = np.sum(K > 0, axis=1)
        assert np.all(degrees == 4)

    def test_ring_shape(self):
        K = _ring_coupling(8, k=1)
        assert K.shape == (8, 8)

    def test_watts_strogatz_shape(self):
        K = _watts_strogatz_coupling(16, k=4, beta=0.3)
        assert K.shape == (16, 16)

    def test_watts_strogatz_symmetric(self):
        K = _watts_strogatz_coupling(16, k=4, beta=0.5)
        np.testing.assert_allclose(K, K.T, atol=1e-12)


class TestGraphP_H1_Result:
    def test_dataclass_fields(self):
        r = GraphP_H1_Result(
            graph_family="test",
            parameter=0.5,
            n_nodes=16,
            avg_degree=4.0,
            p_h1_mean=0.01,
            p_h1_std=0.005,
            n_samples=10,
        )
        assert r.graph_family == "test"
        assert 0 <= r.p_h1_mean <= 1.0


# ---------------------------------------------------------------------------
# Graph generator invariants
# ---------------------------------------------------------------------------


class TestGraphInvariants:
    def test_ring_zero_diagonal(self):
        K = _ring_coupling(8, k=1)
        np.testing.assert_allclose(np.diag(K), 0.0)

    def test_erdos_renyi_zero_diagonal(self):
        K = _erdos_renyi_coupling(8, 0.5)
        np.testing.assert_allclose(np.diag(K), 0.0)

    def test_watts_strogatz_zero_diagonal(self):
        K = _watts_strogatz_coupling(8, k=2, beta=0.3)
        np.testing.assert_allclose(np.diag(K), 0.0)

    def test_ring_non_negative(self):
        K = _ring_coupling(16, k=2)
        assert np.all(K >= 0)

    def test_erdos_renyi_non_negative(self):
        K = _erdos_renyi_coupling(16, 0.3)
        assert np.all(K >= 0)


# ---------------------------------------------------------------------------
# Pipeline: graph → coupling → Hamiltonian wiring
# ---------------------------------------------------------------------------


class TestGraphPipeline:
    def test_ring_to_hamiltonian(self):
        """Ring topology coupling feeds into knm_to_hamiltonian without error."""
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            knm_to_hamiltonian,
        )

        K = _ring_coupling(4, k=1)
        omega = OMEGA_N_16[:4]
        H = knm_to_hamiltonian(K, omega)
        assert H.num_qubits == 4

    def test_erdos_renyi_to_hamiltonian(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            knm_to_hamiltonian,
        )

        K = _erdos_renyi_coupling(4, 0.8)
        omega = OMEGA_N_16[:4]
        H = knm_to_hamiltonian(K, omega)
        assert H.num_qubits == 4


# ---------------------------------------------------------------------------
# Mock for ripser-dependent functions
# ---------------------------------------------------------------------------


def _mock_compute_persistence(theta, persistence_threshold=0.1):
    """Deterministic mock: p_h1 proportional to angle variance."""
    from scpn_quantum_control.analysis.persistent_homology import PersistenceResult

    var = float(np.var(theta))
    p_h1 = min(var / 5.0, 1.0)
    return PersistenceResult(
        n_h0=1,
        n_h1=int(p_h1 * 3),
        p_h1=p_h1,
        persistence_h1=[0.2] * int(p_h1 * 3),
        n_oscillators=len(theta),
    )


_MOCK_CP = patch(
    "scpn_quantum_control.analysis.graph_topology_scan.compute_persistence",
    side_effect=_mock_compute_persistence,
)
_MOCK_RIPSER = patch(
    "scpn_quantum_control.analysis.graph_topology_scan._RIPSER_AVAILABLE",
    True,
)


def _fast_measure_p_h1(K, n_thermalize=3000, n_samples=30, persistence_threshold=0.01, seed=42):
    """Fast mock for _measure_p_h1_at_transition — skips MC, returns deterministic p_h1."""
    avg_coupling = float(np.mean(K[K > 0])) if np.any(K > 0) else 0.0
    p_h1 = min(avg_coupling, 1.0)
    return p_h1, 0.05


_MOCK_MEASURE = patch(
    "scpn_quantum_control.analysis.graph_topology_scan._measure_p_h1_at_transition",
    side_effect=_fast_measure_p_h1,
)


# ---------------------------------------------------------------------------
# Angle 3: _measure_p_h1_at_transition with mocked ripser
# ---------------------------------------------------------------------------


class TestMeasurePH1AtTransition:
    """p_h1 measurement at approximate BKT transition."""

    @_MOCK_CP
    def test_returns_mean_and_std(self, _mock):
        K = _ring_coupling(8, k=2)
        mean, std = _measure_p_h1_at_transition(K, n_thermalize=10, n_samples=5, seed=42)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert 0 <= mean <= 1
        assert std >= 0

    @_MOCK_CP
    def test_reproducible(self, _mock):
        K = _ring_coupling(8, k=2)
        m1, s1 = _measure_p_h1_at_transition(K, n_thermalize=10, n_samples=5, seed=99)
        m2, s2 = _measure_p_h1_at_transition(K, n_thermalize=10, n_samples=5, seed=99)
        assert m1 == m2
        assert s1 == s2

    @_MOCK_CP
    def test_zero_coupling(self, _mock):
        """Zero coupling → MC always accepts, angles random, still returns valid."""
        K = np.zeros((6, 6))
        mean, std = _measure_p_h1_at_transition(K, n_thermalize=10, n_samples=5, seed=42)
        assert isinstance(mean, float)
        assert 0 <= mean <= 1

    @_MOCK_CP
    def test_erdos_renyi_input(self, _mock):
        K = _erdos_renyi_coupling(8, 0.5)
        mean, std = _measure_p_h1_at_transition(K, n_thermalize=10, n_samples=5, seed=42)
        assert 0 <= mean <= 1

    @_MOCK_CP
    def test_watts_strogatz_input(self, _mock):
        K = _watts_strogatz_coupling(8, k=4, beta=0.3)
        mean, std = _measure_p_h1_at_transition(K, n_thermalize=10, n_samples=5, seed=42)
        assert 0 <= mean <= 1

    @_MOCK_CP
    def test_measurement_samples_collected(self, _mock):
        """Measurement phase collects n_samples persistence measurements."""
        K = _ring_coupling(6, k=1)
        _measure_p_h1_at_transition(K, n_thermalize=5, n_samples=3, seed=42)
        # With Rust path: temp scan uses order parameter (no compute_persistence),
        # measurement phase calls compute_persistence n_samples=3 times.
        # Without Rust: temp scan (6) + measurement (3) = 9.
        assert _mock.call_count >= 3


# ---------------------------------------------------------------------------
# Angle 4: scan_graph_topologies orchestration
# ---------------------------------------------------------------------------


def _run_scan(**kwargs):
    """Run scan_graph_topologies with both mocks active."""
    import scpn_quantum_control.analysis.graph_topology_scan as _mod

    orig = _mod._measure_p_h1_at_transition
    _mod._measure_p_h1_at_transition = _fast_measure_p_h1
    orig_ripser = _mod._RIPSER_AVAILABLE
    _mod._RIPSER_AVAILABLE = True
    try:
        return scan_graph_topologies(**kwargs)
    finally:
        _mod._measure_p_h1_at_transition = orig
        _mod._RIPSER_AVAILABLE = orig_ripser


class TestScanGraphTopologies:
    """Full scan pipeline with mocked _measure_p_h1_at_transition for speed."""

    def test_returns_list_of_results(self):
        results = _run_scan(n=6, n_samples=3, seed=42)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, GraphP_H1_Result) for r in results)

    def test_all_families_present(self):
        results = _run_scan(n=6, n_samples=3, seed=42)
        families = {r.graph_family for r in results}
        assert families == {"erdos_renyi", "ring_lattice", "watts_strogatz", "knm_complete"}

    def test_expected_count(self):
        """6 ER + 4 ring + 5 WS + 1 KNM = 16 results."""
        results = _run_scan(n=6, n_samples=3, seed=42)
        assert len(results) == 16

    def test_result_fields_valid(self):
        results = _run_scan(n=6, n_samples=3, seed=42)
        for r in results:
            assert r.n_nodes == 6
            assert r.n_samples == 3
            assert 0 <= r.p_h1_mean <= 1
            assert r.p_h1_std >= 0
            assert r.avg_degree >= 0

    def test_erdos_renyi_parameters(self):
        """ER results have expected edge probability values."""
        results = _run_scan(n=6, n_samples=3, seed=42)
        er = [r for r in results if r.graph_family == "erdos_renyi"]
        params = sorted(r.parameter for r in er)
        assert params == pytest.approx([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])

    def test_knm_complete_uses_paper27(self):
        """KNM complete graph uses build_knm_paper27."""
        results = _run_scan(n=6, n_samples=3, seed=42)
        knm = [r for r in results if r.graph_family == "knm_complete"]
        assert len(knm) == 1
        assert knm[0].avg_degree > 0

    def test_deterministic(self):
        r1 = _run_scan(n=6, n_samples=3, seed=42)
        r2 = _run_scan(n=6, n_samples=3, seed=42)
        for a, b in zip(r1, r2):
            assert a.graph_family == b.graph_family
            assert a.p_h1_mean == b.p_h1_mean

    def test_raises_without_ripser(self):
        with (
            patch(
                "scpn_quantum_control.analysis.graph_topology_scan._RIPSER_AVAILABLE",
                False,
            ),
            pytest.raises(ImportError, match="ripser"),
        ):
            scan_graph_topologies(n=6, n_samples=3)


# ---------------------------------------------------------------------------
# Angle 5: Parametric sensitivity across sizes
# ---------------------------------------------------------------------------


class TestParametricSensitivity:
    """Graph generators scale correctly with parameters."""

    def test_erdos_renyi_density_monotone(self):
        """Higher p → more edges."""
        K_lo = _erdos_renyi_coupling(16, 0.1, seed=0)
        K_hi = _erdos_renyi_coupling(16, 0.9, seed=0)
        assert np.sum(K_lo > 0) < np.sum(K_hi > 0)

    def test_erdos_renyi_strength(self):
        K = _erdos_renyi_coupling(8, 1.0, strength=3.5)
        nonzero = K[K > 0]
        np.testing.assert_allclose(nonzero, 3.5)

    def test_ring_strength(self):
        K = _ring_coupling(6, k=1, strength=0.7)
        nonzero = K[K > 0]
        np.testing.assert_allclose(nonzero, 0.7)

    def test_watts_strogatz_strength(self):
        K = _watts_strogatz_coupling(8, k=4, beta=0.0, strength=2.0)
        nonzero = K[K > 0]
        np.testing.assert_allclose(nonzero, 2.0)

    @pytest.mark.parametrize("n", [8, 12, 16])
    def test_ring_degree_invariant(self, n):
        K = _ring_coupling(n, k=2)
        degrees = np.sum(K > 0, axis=1)
        np.testing.assert_array_equal(degrees, 4)


# ---------------------------------------------------------------------------
# Angle 6: Coverage gap — dense graph rewiring, Python MC fallback
# ---------------------------------------------------------------------------


class TestWattsStrogatzDenseGraph:
    """Cover line 75: 'continue' when graph too dense to rewire."""

    def test_complete_graph_no_candidates(self):
        """n=4, k=4 → ring lattice is complete K4 → no candidates for rewiring."""
        K = _watts_strogatz_coupling(4, k=4, beta=1.0, seed=42)
        assert K.shape == (4, 4)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        # Every pair connected in K4, so rewiring has no effect
        assert np.all(K[np.eye(4) == 0] > 0)

    def test_nearly_complete(self):
        """n=5, k=4 → nearly complete, most rewiring skipped."""
        K = _watts_strogatz_coupling(5, k=4, beta=1.0, seed=42)
        assert K.shape == (5, 5)
        np.testing.assert_allclose(K, K.T, atol=1e-12)


class TestMeasurePH1PythonFallback:
    """Cover lines 122-123, 134-143: Python MC fallback when Rust unavailable."""

    @_MOCK_CP
    def test_python_fallback_no_rust(self, _mock):
        """Mock Rust import to fail → Python MC fallback executes."""
        K = _ring_coupling(6, k=1)
        with patch.dict("sys.modules", {"scpn_quantum_engine": None}):
            mean, std = _measure_p_h1_at_transition(K, n_thermalize=5, n_samples=3, seed=42)
        assert isinstance(mean, float)
        assert 0 <= mean <= 1
        assert isinstance(std, float)
        assert std >= 0

    @_MOCK_CP
    def test_python_fallback_finds_best_temperature(self, _mock):
        """Python fallback iterates temperatures and selects best."""
        K = _erdos_renyi_coupling(6, 0.5, seed=42)
        with patch.dict("sys.modules", {"scpn_quantum_engine": None}):
            mean, std = _measure_p_h1_at_transition(K, n_thermalize=5, n_samples=3, seed=99)
        assert 0 <= mean <= 1
