# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control Core Tests
"""Tests for constrained persistent-H1 coupling graph control."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.topology_control import (
    CouplingGraphBounds,
    CouplingTopologyObjective,
    DegeneracyMode,
    NetworkCycleBackend,
    ProjectedScipyOptimizer,
    ProjectedSPSAOptimizer,
    TopologyConstraintLedger,
    build_coupling_distance_matrix,
)


def test_coupling_distance_matrix_is_symmetric_and_zero_diagonal() -> None:
    K = np.array(
        [
            [0.0, 0.2, 0.9],
            [0.2, 0.0, 0.4],
            [0.9, 0.4, 0.0],
        ]
    )

    distance = build_coupling_distance_matrix(K)

    np.testing.assert_allclose(distance, distance.T)
    np.testing.assert_allclose(np.diag(distance), 0.0)
    assert distance[0, 2] < distance[0, 1]


def test_network_cycle_backend_detects_one_square_hole() -> None:
    distance = np.array(
        [
            [0.0, 0.1, 1.0, 0.1],
            [0.1, 0.0, 0.1, 1.0],
            [1.0, 0.1, 0.0, 0.1],
            [0.1, 1.0, 0.1, 0.0],
        ]
    )

    summary = NetworkCycleBackend(threshold=0.2).compute(distance)

    assert summary.n_h1_persistent == 1
    assert summary.p_h1 == pytest.approx(1.0 / 3.0)
    assert summary.lifetimes == [1.0]


def test_constraint_projection_preserves_frozen_edges_and_budget() -> None:
    K0 = np.array(
        [
            [0.0, 0.4, 0.2],
            [0.4, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ]
    )
    candidate = np.array(
        [
            [9.0, -2.0, 0.8],
            [0.7, 3.0, 0.9],
            [0.1, 0.5, 4.0],
        ]
    )
    ledger = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(lower=0.0, upper=1.0),
        total_weight=(float(np.sum(K0)) * 0.95, float(np.sum(K0)) * 1.05),
        frozen_edges={(0, 1): 0.4},
        hardware_edges={(0, 1), (1, 2)},
    )

    projected = ledger.project(candidate)

    np.testing.assert_allclose(projected, projected.T)
    np.testing.assert_allclose(np.diag(projected), 0.0)
    assert projected[0, 1] == pytest.approx(0.4)
    assert projected[0, 2] == pytest.approx(0.0)
    assert ledger.total_weight[0] <= float(np.sum(projected)) <= ledger.total_weight[1]


def test_objective_penalises_degenerate_zero_graph() -> None:
    K0 = np.array(
        [
            [0.0, 0.3, 0.0, 0.3],
            [0.3, 0.0, 0.3, 0.0],
            [0.0, 0.3, 0.0, 0.3],
            [0.3, 0.0, 0.3, 0.0],
        ]
    )
    objective = CouplingTopologyObjective(
        ph_backend=NetworkCycleBackend(threshold=0.2),
        ledger=TopologyConstraintLedger(
            total_weight=(float(np.sum(K0)) * 0.9, float(np.sum(K0)) * 1.1),
            algebraic_connectivity_min=0.01,
        ),
        source_matrix=K0,
        source_distance_weight=0.25,
        allow_approximate_ph_backend=True,
    )

    zero_breakdown = objective.evaluate(np.zeros_like(K0))
    source_breakdown = objective.evaluate(K0)

    assert zero_breakdown.degeneracy_mode is DegeneracyMode.ZERO_GRAPH
    assert zero_breakdown.total > source_breakdown.total
    assert zero_breakdown.terms["degeneracy_penalty"] > 0.0


def test_projected_spsa_is_deterministic_and_respects_constraints() -> None:
    K0 = np.array(
        [
            [0.0, 0.3, 0.0, 0.3],
            [0.3, 0.0, 0.3, 0.0],
            [0.0, 0.3, 0.0, 0.3],
            [0.3, 0.0, 0.3, 0.0],
        ]
    )
    objective = CouplingTopologyObjective(
        ph_backend=NetworkCycleBackend(threshold=0.2),
        ledger=TopologyConstraintLedger(
            bounds=CouplingGraphBounds(lower=0.0, upper=0.6),
            total_weight=(float(np.sum(K0)) * 0.95, float(np.sum(K0)) * 1.05),
            algebraic_connectivity_min=0.01,
            hardware_edges={(0, 1), (1, 2), (2, 3), (0, 3)},
        ),
        source_matrix=K0,
        source_distance_weight=0.1,
        allow_approximate_ph_backend=True,
    )

    first = ProjectedSPSAOptimizer(seed=123, max_steps=5).optimise(K0, objective)
    second = ProjectedSPSAOptimizer(seed=123, max_steps=5).optimise(K0, objective)

    np.testing.assert_allclose(first.final_matrix, second.final_matrix)
    assert len(first.steps) == 5
    np.testing.assert_allclose(first.final_matrix, first.final_matrix.T)
    assert np.all(first.final_matrix >= 0.0)
    assert first.steps[-1].objective.total == pytest.approx(
        objective.evaluate(first.final_matrix).total
    )


def test_objective_rejects_approximate_backend_without_explicit_opt_in() -> None:
    K0 = np.array(
        [
            [0.0, 0.3, 0.0, 0.3],
            [0.3, 0.0, 0.3, 0.0],
            [0.0, 0.3, 0.0, 0.3],
            [0.3, 0.0, 0.3, 0.0],
        ]
    )
    objective = CouplingTopologyObjective(
        ph_backend=NetworkCycleBackend(threshold=0.2),
        ledger=TopologyConstraintLedger(),
        source_matrix=K0,
    )
    with pytest.raises(ValueError, match="allow_approximate_ph_backend"):
        objective.evaluate(K0)


def test_projected_scipy_optimizer_records_callback_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    K0 = np.array(
        [
            [0.0, 0.3, 0.0, 0.3],
            [0.3, 0.0, 0.3, 0.0],
            [0.0, 0.3, 0.0, 0.3],
            [0.3, 0.0, 0.3, 0.0],
        ]
    )
    objective = CouplingTopologyObjective(
        ph_backend=NetworkCycleBackend(threshold=0.2),
        ledger=TopologyConstraintLedger(),
        source_matrix=K0,
        allow_approximate_ph_backend=True,
    )

    class _FakeResult:
        def __init__(self, x: np.ndarray) -> None:
            self.x = x

    def _fake_minimize(fun, x0, method, callback, options):  # type: ignore[no-untyped-def]
        assert method == "COBYLA"
        assert options == {"maxiter": 5}
        x1 = np.asarray(x0, dtype=np.float64) * 0.95
        x2 = np.asarray(x0, dtype=np.float64) * 0.90
        callback(x1)
        callback(x2)
        _ = fun(x2)
        return _FakeResult(x2)

    monkeypatch.setattr("scipy.optimize.minimize", _fake_minimize)

    trace = ProjectedScipyOptimizer(maxiter=5).optimise(K0, objective)

    assert len(trace.steps) == 2
    np.testing.assert_allclose(trace.final_matrix, trace.steps[-1].matrix)
