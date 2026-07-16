# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the Sinkhorn layout relaxation (KT-4)
"""Multi-angle tests for hardware/kuramoto_layout_relaxation.py (RESEARCH).

Dimensions: configuration invariants and the annealing schedule, the Sinkhorn
operator (doubly-stochastic convergence, shape guards), coupling-graph
distances (BFS correctness, disconnection fail-closed), the surrogate and its
closed-form gradient (finite-difference check), Hungarian rounding, and the
full relaxed-then-rounded search with an injected cheap true cost —
warm-start bias, budget enforcement, determinism, and the research label.
All numpy/scipy — no qiskit simulation, tracer-safe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.hardware.kuramoto_layout_relaxation import (
    RESEARCH_LABEL,
    RelaxationSearchResult,
    SinkhornRelaxationConfig,
    _surrogate_gradient,
    coupling_graph_distances,
    relax_kuramoto_layout,
    sinkhorn_normalise,
    swap_distance_surrogate,
)

_N = 3
_K = np.ones((_N, _N)) - np.eye(_N)
_OMEGA = np.array([0.1, 0.2, 0.3])
#: Line 0-1-2-3-4: distances between candidates are plain index gaps.
_LINE_EDGES = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]
_FIDELITY = 0.99


def _pair_distance_depth(
    layout: tuple[int, ...],
    K: Any,
    omega: Any,
    coupling_map: Any,
    *,
    t: float,
    reps: int,
) -> int:
    """Cheap true-cost depth: total pairwise index distance of the layout."""
    return int(sum(abs(a - b) for a in layout for b in layout))


def _relax(**overrides: Any) -> RelaxationSearchResult:
    arguments: dict[str, Any] = {
        "mean_gate_fidelity": _FIDELITY,
        "config": SinkhornRelaxationConfig(seed=3, n_anneal_steps=4, n_gradient_steps=8),
        "depth_provider": _pair_distance_depth,
    }
    arguments.update(overrides)
    return relax_kuramoto_layout(_K, _OMEGA, _LINE_EDGES, (0, 1, 2, 3, 4), **arguments)


class TestSinkhornRelaxationConfig:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"tau_initial": 0.0}, "tau_initial"),
            ({"tau_final": 0.0}, "tau_final"),
            ({"tau_initial": 0.05, "tau_final": 0.5}, "must not exceed"),
            ({"n_anneal_steps": 0}, "n_anneal_steps"),
            ({"n_gradient_steps": 0}, "n_gradient_steps"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"n_sinkhorn_iterations": 0}, "n_sinkhorn_iterations"),
            ({"max_true_cost_evaluations": 0}, "max_true_cost_evaluations"),
            ({"t": 0.0}, "t must be"),
            ({"reps": 0}, "reps"),
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SinkhornRelaxationConfig(**kwargs)

    def test_temperature_schedule_geometric_and_ordered(self) -> None:
        schedule = SinkhornRelaxationConfig(
            tau_initial=1.0, tau_final=0.1, n_anneal_steps=5
        ).temperatures()
        assert schedule.shape == (5,)
        assert schedule[0] == pytest.approx(1.0)
        assert schedule[-1] == pytest.approx(0.1)
        assert np.all(np.diff(schedule) < 0)

    def test_to_dict_serialises_weights(self) -> None:
        payload = SinkhornRelaxationConfig().to_dict()
        assert payload["weights"] is None
        assert payload["max_true_cost_evaluations"] is None
        assert payload["n_anneal_steps"] == 8


class TestSinkhornOperator:
    def test_output_is_doubly_stochastic(self) -> None:
        logits = np.random.default_rng(0).standard_normal((5, 5))
        P = sinkhorn_normalise(logits, 60)
        # np.sum (not ndarray.sum) dodges the numpy-reload _NoValueType
        # sentinel bug under focused pytest-cov runs.
        assert np.allclose(np.sum(P, axis=0), 1.0, atol=1e-6)
        assert np.allclose(np.sum(P, axis=1), 1.0, atol=1e-6)
        assert np.all(P >= 0.0)

    def test_low_temperature_sharpens_to_permutation(self) -> None:
        logits = np.array([[5.0, 0.0], [0.0, 5.0]])
        P = sinkhorn_normalise(logits / 0.05, 60)
        assert P[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert P[1, 1] == pytest.approx(1.0, abs=1e-6)

    def test_non_square_rejected(self) -> None:
        with pytest.raises(ValueError, match="square"):
            sinkhorn_normalise(np.zeros((2, 3)), 5)


class TestCouplingGraphDistances:
    def test_line_graph_distances_are_index_gaps(self) -> None:
        distances = coupling_graph_distances(_LINE_EDGES, (0, 2, 4))
        expected = np.array([[0.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 0.0]])
        assert np.array_equal(distances, expected)

    def test_paths_may_leave_the_candidate_set(self) -> None:
        # Candidates 0 and 4 connect only through non-candidate qubits.
        distances = coupling_graph_distances(_LINE_EDGES, (0, 4))
        assert distances[0, 1] == 4.0

    def test_qiskit_coupling_map_accepted(self) -> None:
        from qiskit.transpiler import CouplingMap

        distances = coupling_graph_distances(CouplingMap(_LINE_EDGES), (0, 1, 2))
        assert distances[0, 2] == 2.0

    def test_disconnected_candidates_fail_closed(self) -> None:
        with pytest.raises(ValueError, match="disconnected"):
            coupling_graph_distances([(0, 1), (1, 0), (3, 4), (4, 3)], (0, 3))


class TestSurrogate:
    def test_surrogate_prefers_adjacent_strong_pairs(self) -> None:
        distances = coupling_graph_distances(_LINE_EDGES, (0, 1, 2, 3, 4))
        K2 = np.zeros((5, 5))
        K2[0, 1] = K2[1, 0] = 1.0
        adjacent = np.eye(5)  # logical 0->0, 1->1: distance 1
        spread = np.eye(5)[[0, 4, 1, 2, 3]]  # logical 0->0, 1->4: distance 4
        assert swap_distance_surrogate(adjacent, K2, distances) < swap_distance_surrogate(
            spread, K2, distances
        )

    def test_gradient_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(1)
        distances = coupling_graph_distances(_LINE_EDGES, (0, 1, 2, 3))
        K2 = np.zeros((4, 4))
        K2[:3, :3] = _K
        P = sinkhorn_normalise(rng.standard_normal((4, 4)), 40)
        analytic = _surrogate_gradient(P, K2, distances)
        step = 1e-6
        for index in [(0, 0), (1, 2), (3, 3)]:
            bumped = P.copy()
            bumped[index] += step
            numeric = (
                swap_distance_surrogate(bumped, K2, distances)
                - swap_distance_surrogate(P, K2, distances)
            ) / step
            assert analytic[index] == pytest.approx(numeric, rel=1e-3, abs=1e-6)


class TestRelaxedSearch:
    def test_finds_compact_layout_on_line(self) -> None:
        result = _relax()
        assert result.best_cost.total > 0.0
        # Compact contiguous placements minimise the pairwise-distance depth.
        spread = max(result.best_layout) - min(result.best_layout)
        assert spread <= 3
        assert result.n_true_evaluations >= 1
        assert len(result.surrogate_trajectory) == 4

    def test_warm_start_layout_biases_first_round(self) -> None:
        result = _relax(
            initial_layout=(0, 1, 2),
            config=SinkhornRelaxationConfig(seed=3, n_anneal_steps=1, n_gradient_steps=1),
        )
        assert sorted(result.best_layout) == [0, 1, 2]

    def test_budget_is_enforced(self) -> None:
        result = _relax(
            config=SinkhornRelaxationConfig(
                seed=5, n_anneal_steps=6, n_gradient_steps=5, max_true_cost_evaluations=2
            )
        )
        assert result.n_true_evaluations <= 2

    def test_deterministic_for_fixed_seed(self) -> None:
        first, second = _relax(), _relax()
        assert first.best_layout == second.best_layout
        assert first.surrogate_trajectory == second.surrogate_trajectory

    def test_research_label_carried(self) -> None:
        result = _relax()
        assert result.research_label == RESEARCH_LABEL
        payload = result.to_dict()
        assert "RESEARCH" in payload["research_label"]
        assert payload["best_layout"] == list(result.best_layout)

    def test_search_space_validation_shared_with_discrete_optimiser(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            relax_kuramoto_layout(
                _K,
                _OMEGA,
                _LINE_EDGES,
                (0, 1, 1),
                mean_gate_fidelity=_FIDELITY,
                depth_provider=_pair_distance_depth,
            )

    def test_cost_validation_propagates(self) -> None:
        with pytest.raises(ValueError, match="mean_gate_fidelity"):
            relax_kuramoto_layout(
                _K,
                _OMEGA,
                _LINE_EDGES,
                (0, 1, 2, 3),
                mean_gate_fidelity=1.5,
                config=SinkhornRelaxationConfig(n_anneal_steps=1, n_gradient_steps=1),
                depth_provider=_pair_distance_depth,
            )


class TestPackageExport:
    def test_hardware_package_exports_relaxation(self) -> None:
        from scpn_quantum_control import hardware

        assert hardware.relax_kuramoto_layout is relax_kuramoto_layout
        assert hardware.SinkhornRelaxationConfig is SinkhornRelaxationConfig
        assert hardware.RelaxationSearchResult is RelaxationSearchResult
