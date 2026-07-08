# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- deterministic sensing coverage tests
"""Coverage-focused tests for the S11 sensing readiness module."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.analysis import sensing


@dataclass(frozen=True)
class _CrosscheckStub:
    agrees: bool
    max_rel_difference: float


def _inputs() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.array([-0.2, 0.0, 0.25], dtype=np.float64),
        np.ones((3, 3), dtype=np.float64) - np.eye(3, dtype=np.float64),
        np.array([0.4, 0.8, 1.2], dtype=np.float64),
    )


@pytest.fixture
def stubbed_qfi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace dense QFI/QGT calls with deterministic arithmetic fixtures."""

    def fake_qfi_vs_coupling(
        omega: NDArray[np.float64],
        K_topology: NDArray[np.float64],
        k_range: NDArray[np.float64] | None = None,
        *,
        max_dense_gib: float | None = None,
    ) -> SimpleNamespace:
        assert omega.shape == (3,)
        assert K_topology.shape == (3, 3)
        assert max_dense_gib == pytest.approx(0.01)
        grid = np.array([0.4, 0.8, 1.2], dtype=np.float64) if k_range is None else k_range
        return SimpleNamespace(
            k_values=grid,
            max_qfi=np.array([1.0, 4.0, 2.0], dtype=np.float64),
            spectral_gap=np.array([0.3, 0.1, 0.2], dtype=np.float64),
            total_qfi=np.array([1.2, 4.5, 2.4], dtype=np.float64),
            peak_k=float(grid[1]),
            peak_qfi=4.0,
        )

    def fake_compute_qfi(
        K: NDArray[np.float64],
        omega: NDArray[np.float64],
        pairs: list[tuple[int, int]] | None = None,
        *,
        max_dense_gib: float | None = None,
    ) -> SimpleNamespace:
        del pairs
        assert K.shape == (3, 3)
        assert omega.shape == (3,)
        assert max_dense_gib == pytest.approx(0.01)
        return SimpleNamespace(
            qfi_matrix=np.array([[1.0, 0.0], [0.0, 8.0]], dtype=np.float64),
            coupling_pairs=[(0, 1), (1, 2)],
        )

    def fake_crosscheck(
        K: NDArray[np.float64],
        omega: NDArray[np.float64],
        *,
        epsilon: float = 0.005,
        max_dense_gib: float | None = None,
    ) -> _CrosscheckStub:
        assert K.shape == (3, 3)
        assert omega.shape == (3,)
        assert epsilon == pytest.approx(0.005)
        assert max_dense_gib == pytest.approx(0.01)
        return _CrosscheckStub(agrees=True, max_rel_difference=0.004)

    monkeypatch.setattr(sensing, "qfi_vs_coupling", fake_qfi_vs_coupling)
    monkeypatch.setattr(sensing, "compute_qfi", fake_compute_qfi)
    monkeypatch.setattr(sensing, "crosscheck_qfi_geometric", fake_crosscheck)


def test_sensing_payload_and_markdown_cover_json_surfaces(stubbed_qfi: None) -> None:
    """The full S11 payload and generated Markdown are JSON-ready."""
    payload = sensing.quantum_sensing_payload()
    markdown = sensing.quantum_sensing_markdown()

    assert payload["config"] == sensing.QuantumSensingReadinessConfig().to_dict()
    assert payload["gain_scan"]["schema"] == sensing.GAIN_SCAN_SCHEMA
    assert payload["criticality_tail"]["schema"] == sensing.CRITICALITY_TAIL_SCHEMA
    assert payload["criticality_tail"]["selected_pair"] == [1, 2]
    assert payload["criticality_tail"]["cramer_rao_variance_bound"] == pytest.approx(
        1.0 / (10000.0 * 8.0)
    )
    assert "## QFI-Criticality Tail" in markdown
    assert "scpn-bench s11-quantum-sensing-readiness" in markdown


def test_gain_scan_and_tail_to_dicts_are_complete(stubbed_qfi: None) -> None:
    """Explicit scan and tail calls expose all readiness fields."""
    omega, topology, k_grid = _inputs()
    config = sensing.QuantumSensingReadinessConfig(max_dense_gib=0.01)

    scan = sensing.metrological_gain_vs_k(omega, topology, k_grid, config=config)
    scan_dict = scan.to_dict()
    row_dict = scan.rows[0].to_dict()
    optimal = sensing.optimal_sensing_k(omega, topology, k_grid, config=config)
    tail = sensing.qfi_criticality_sensing_tail(
        omega,
        topology,
        k_grid,
        measurements=20,
        config=config,
    )
    tail_dict = tail.to_dict()

    assert scan.schema == sensing.GAIN_SCAN_SCHEMA
    assert scan.best_gain_ratio == max(row.gain_ratio for row in scan.rows)
    assert scan_dict["optimal_row"] == optimal.to_dict()
    assert row_dict["claim_boundary"] == sensing.ROW_BOUNDARY
    assert tail.selected_pair == (1, 2)
    assert tail.qfi_trace == pytest.approx(9.0)
    assert tail.gap_min_k == pytest.approx(0.8)
    assert tail.peak_gap_delta == pytest.approx(0.0)
    assert tail.geometric_crosscheck_agrees is True
    assert tail_dict["geometric_crosscheck_max_rel_difference"] == pytest.approx(0.004)


def test_tail_records_when_geometric_crosscheck_is_skipped(stubbed_qfi: None) -> None:
    """Skipping the QGT route is represented as non-agreement, not success."""
    omega, topology, k_grid = _inputs()

    tail = sensing.qfi_criticality_sensing_tail(
        omega,
        topology,
        k_grid,
        run_geometric_crosscheck=False,
        config=sensing.QuantumSensingReadinessConfig(max_dense_gib=0.01),
    )

    assert tail.geometric_crosscheck_agrees is False
    assert np.isinf(tail.geometric_crosscheck_max_rel_difference)


def test_input_validation_and_precision_bound_branches(stubbed_qfi: None) -> None:
    """Validation failures and zero-QFI precision bounds fail closed."""
    omega, topology, k_grid = _inputs()

    with pytest.raises(ValueError, match="readout_variance_floor"):
        sensing.QuantumSensingReadinessConfig(readout_variance_floor=0.0)
    with pytest.raises(ValueError, match="topology must be a square matrix"):
        sensing.metrological_gain_vs_k(omega, topology[:2], k_grid)
    with pytest.raises(ValueError, match="omega length"):
        sensing.metrological_gain_vs_k(omega[:2], topology, k_grid)
    with pytest.raises(ValueError, match="at least two"):
        sensing.metrological_gain_vs_k(omega, topology, np.array([0.4], dtype=np.float64))
    with pytest.raises(ValueError, match="finite values"):
        sensing.metrological_gain_vs_k(
            np.array([np.inf, 0.0, 0.25], dtype=np.float64),
            topology,
            k_grid,
        )
    with pytest.raises(ValueError, match="finite positive"):
        sensing.metrological_gain_vs_k(
            omega,
            topology,
            np.array([0.0, 0.8, 1.2], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        sensing.metrological_gain_vs_k(
            omega,
            topology,
            np.array([0.4, 0.4, 1.2], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="symmetric"):
        sensing.metrological_gain_vs_k(
            omega,
            np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            k_grid,
        )
    with pytest.raises(ValueError, match="measurements"):
        sensing.qfi_criticality_sensing_tail(omega, topology, k_grid, measurements=0)
    with pytest.raises(ValueError, match="geometric_epsilon"):
        sensing.qfi_criticality_sensing_tail(omega, topology, k_grid, geometric_epsilon=0.0)
    with pytest.raises(ValueError, match="nonzero coupling edge"):
        sensing.qfi_criticality_sensing_tail(omega, np.zeros_like(topology), k_grid)

    assert np.isinf(sensing._cramer_rao_variance_bound(0.0, 10))


def test_markdown_uses_supplied_payload_without_recomputing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supplied payload is rendered directly."""

    def fail_if_recomputed() -> dict[str, Any]:
        raise AssertionError("payload should not be recomputed")

    monkeypatch.setattr(sensing, "quantum_sensing_payload", fail_if_recomputed)
    row = sensing.SensingGainRow(
        k_value=1.0,
        qfi_value=2.0,
        spectral_gap=0.5,
        sync_order_parameter=0.8,
        classical_fisher_proxy=4.0,
        gain_ratio=0.5,
    )
    scan = sensing.SensingGainScan(
        schema=sensing.GAIN_SCAN_SCHEMA,
        k_values=(1.0,),
        rows=(row,),
        peak_k=1.0,
        peak_qfi=2.0,
        best_gain_ratio=0.5,
        optimal_row=row,
        falsifier=sensing.FALSIFIER,
    )
    tail = sensing.CriticalitySensingTail(
        schema=sensing.CRITICALITY_TAIL_SCHEMA,
        operating_k=1.0,
        selected_pair=(0, 1),
        qfi_value=2.0,
        qfi_trace=3.0,
        spectral_gap=0.5,
        measurements=100,
        cramer_rao_variance_bound=0.005,
        cramer_rao_std_bound=0.070710678,
        gap_min_k=1.0,
        peak_gap_delta=0.0,
        geometric_crosscheck_agrees=True,
        geometric_crosscheck_max_rel_difference=0.001,
        claim_boundary=sensing.TAIL_BOUNDARY,
        falsifier=sensing.TAIL_FALSIFIER,
    )
    payload = {
        "claim_boundary": sensing.CLAIM_BOUNDARY,
        "gain_scan": scan.to_dict(),
        "criticality_tail": tail.to_dict(),
        "prerequisites": ("fixed benchmark",),
        "falsifier": sensing.FALSIFIER,
    }

    markdown = sensing.quantum_sensing_markdown(payload)

    assert "fixed benchmark" in markdown
    assert "Selected coupling pair: `(0, 1)`" in markdown
