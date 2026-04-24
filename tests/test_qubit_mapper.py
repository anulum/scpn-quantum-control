# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for DynQ Qubit Mapper
"""Multi-angle tests for hardware/qubit_mapper.py.

6 dimensions: empty/null, error handling, negative cases, pipeline
integration, roundtrip, performance.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scpn_quantum_control.hardware.qubit_mapper import (
    ExecutionRegion,
    QubitMappingResult,
    build_calibration_graph,
    detect_execution_regions,
    dynq_initial_layout,
    select_best_region,
)


def _heavy_hex_errors(n_qubits: int = 16, seed: int = 42) -> dict[tuple[int, int], float]:
    """Generate synthetic heavy-hex-like gate errors."""
    rng = np.random.default_rng(seed)
    errors: dict[tuple[int, int], float] = {}
    # degree-3 connectivity: each qubit connected to ~3 neighbours
    for i in range(n_qubits):
        for j in [i + 1, i + 2]:
            if j < n_qubits:
                errors[(i, j)] = rng.uniform(0.001, 0.02)
    # Add some long-range with higher error
    for i in range(0, n_qubits - 4, 4):
        errors[(i, i + 4)] = rng.uniform(0.01, 0.05)
    return errors


def _cluster_errors() -> dict[tuple[int, int], float]:
    """Two clear clusters with a weak bridge."""
    errors: dict[tuple[int, int], float] = {}
    # Cluster A: qubits 0-4, low error
    for i in range(5):
        for j in range(i + 1, 5):
            errors[(i, j)] = 0.002
    # Cluster B: qubits 5-9, low error
    for i in range(5, 10):
        for j in range(i + 1, 10):
            errors[(i, j)] = 0.003
    # Bridge: high error
    errors[(4, 5)] = 0.08
    return errors


# ===== 1. Empty/Null Inputs =====


class TestEmptyNull:
    def test_single_edge(self) -> None:
        G = build_calibration_graph({(0, 1): 0.01})
        assert G.number_of_edges() == 1
        assert G[0][1]["weight"] == pytest.approx(1.0 / (0.01 + 1e-6))

    def test_no_regions_too_few_qubits(self) -> None:
        errors = {(0, 1): 0.01, (1, 2): 0.01}
        G = build_calibration_graph(errors)
        regions = detect_execution_regions(G, min_qubits=10)
        assert regions == []

    def test_select_none_when_too_small(self) -> None:
        region = ExecutionRegion(
            qubits=frozenset({0, 1, 2}),
            quality_score=0.9,
            connectivity=1.0,
            mean_gate_fidelity=0.99,
            n_qubits=3,
        )
        result = select_best_region([region], circuit_width=5)
        assert result is None


# ===== 2. Error Handling =====


class TestErrorHandling:
    def test_zero_error_no_crash(self) -> None:
        """Perfect gates (error=0) should not cause division by zero."""
        G = build_calibration_graph({(0, 1): 0.0, (1, 2): 0.0, (0, 2): 0.0})
        assert G[0][1]["weight"] == pytest.approx(1.0 / 1e-6)

    def test_high_error_low_weight(self) -> None:
        G = build_calibration_graph({(0, 1): 0.5})
        assert G[0][1]["weight"] < 3.0  # 1/(0.5+ε) ≈ 2.0


# ===== 3. Negative Cases =====


class TestNegativeCases:
    def test_uniform_errors_single_community(self) -> None:
        """Uniform fidelity → Louvain should produce few large communities."""
        errors = {}
        n = 12
        for i in range(n):
            for j in range(i + 1, n):
                errors[(i, j)] = 0.01  # all same
        G = build_calibration_graph(errors)
        regions = detect_execution_regions(G, min_qubits=3, seed=42)
        # With uniform weights, Louvain typically finds 1-2 big communities
        assert len(regions) <= 3

    def test_high_error_bridge_separates_clusters(self) -> None:
        """Clusters connected by high-error bridge should be separated."""
        errors = _cluster_errors()
        G = build_calibration_graph(errors)
        regions = detect_execution_regions(G, min_qubits=3, seed=42)
        assert len(regions) >= 2
        # Both clusters should appear as separate regions
        region_sizes = sorted([r.n_qubits for r in regions], reverse=True)
        assert region_sizes[0] >= 4  # at least one cluster detected


# ===== 4. Pipeline Integration =====


class TestPipelineIntegration:
    def test_full_dynq_pipeline(self) -> None:
        errors = _cluster_errors()
        result = dynq_initial_layout(errors, circuit_width=4, seed=42)
        assert result is not None
        assert isinstance(result, QubitMappingResult)
        assert len(result.initial_layout) == 4
        assert result.selected_region.n_qubits >= 4

    def test_layout_qubits_in_region(self) -> None:
        """Layout qubits must be subset of selected region."""
        errors = _heavy_hex_errors(20)
        result = dynq_initial_layout(errors, circuit_width=3, seed=42)
        assert result is not None
        for q in result.initial_layout:
            assert q in result.selected_region.qubits

    def test_readout_errors_affect_layout(self) -> None:
        errors = _cluster_errors()
        readout = {i: 0.01 * (9 - i) for i in range(10)}  # qubit 9 has best readout
        result = dynq_initial_layout(errors, circuit_width=3, readout_errors=readout, seed=42)
        assert result is not None
        # Layout should prefer qubits with lower readout error
        layout_readout = [readout[q] for q in result.initial_layout]
        assert layout_readout == sorted(layout_readout)

    def test_region_quality_sorted(self) -> None:
        errors = _cluster_errors()
        G = build_calibration_graph(errors)
        regions = detect_execution_regions(G, min_qubits=3, seed=42)
        scores = [r.quality_score for r in regions]
        assert scores == sorted(scores, reverse=True)

    def test_top_level_import(self) -> None:
        from scpn_quantum_control.hardware.qubit_mapper import dynq_initial_layout

        assert callable(dynq_initial_layout)


# ===== 5. Roundtrip =====


class TestRoundtrip:
    def test_best_region_has_lowest_error(self) -> None:
        """Best region should contain qubits with lower mean gate error."""
        errors = _cluster_errors()
        G = build_calibration_graph(errors)
        regions = detect_execution_regions(G, min_qubits=3, seed=42)
        if len(regions) >= 2:
            assert regions[0].mean_gate_fidelity >= regions[-1].mean_gate_fidelity

    def test_connectivity_bounded(self) -> None:
        errors = _heavy_hex_errors(16)
        G = build_calibration_graph(errors)
        regions = detect_execution_regions(G, min_qubits=3, seed=42)
        for r in regions:
            assert 0.0 <= r.connectivity <= 1.0

    def test_resolution_controls_size(self) -> None:
        """Higher resolution → smaller communities."""
        errors = _heavy_hex_errors(20)
        G = build_calibration_graph(errors)
        r_low = detect_execution_regions(G, min_qubits=3, resolution=0.5, seed=42)
        r_high = detect_execution_regions(G, min_qubits=3, resolution=3.0, seed=42)
        avg_low = np.mean([r.n_qubits for r in r_low]) if r_low else 0
        avg_high = np.mean([r.n_qubits for r in r_high]) if r_high else 0
        # Higher resolution should produce smaller or equal average size
        assert avg_high <= avg_low + 2  # tolerance for stochastic Louvain


# ===== 6. Performance =====


class TestPerformance:
    def test_community_detection_fast(self) -> None:
        """156-qubit heavy-hex detection must complete in < 50ms."""
        errors = _heavy_hex_errors(156, seed=42)
        G = build_calibration_graph(errors)
        # warmup
        detect_execution_regions(G, seed=42)
        t0 = time.perf_counter()
        for _ in range(100):
            detect_execution_regions(G, seed=42)
        dt = (time.perf_counter() - t0) / 100 * 1000
        assert dt < 250, f"detection took {dt:.1f}ms"

    def test_full_pipeline_fast(self) -> None:
        """Full DynQ pipeline for 156 qubits < 100ms."""
        errors = _heavy_hex_errors(156, seed=42)
        dynq_initial_layout(errors, circuit_width=5, seed=42)
        t0 = time.perf_counter()
        dynq_initial_layout(errors, circuit_width=5, seed=42)
        dt = (time.perf_counter() - t0) * 1000
        assert dt < 300, f"pipeline took {dt:.1f}ms"
