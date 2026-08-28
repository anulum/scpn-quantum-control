# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the discrete Kuramoto layout optimiser
"""Multi-angle tests for hardware/kuramoto_layout_optimiser.py.

Dimensions: configuration invariants and serialisation, search-space
validation, neighbourhood structure, hill-climbing behaviour on controlled
cost landscapes (injected depth providers keep the search pure and
coverage-safe), determinism, memoisation, convergence flags, and one
integration test with the real routed-depth adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qiskit.transpiler import CouplingMap

from scpn_quantum_control.hardware.kuramoto_layout_cost import CostWeights, kuramoto_layout_cost
from scpn_quantum_control.hardware.kuramoto_layout_optimiser import (
    LayoutSearchConfig,
    LayoutSearchResult,
    _neighbours,
    optimise_kuramoto_layout,
)

_K = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
_OMEGA = np.array([0.1, 0.2, 0.3])
_FIDELITY = 0.99


def _sum_depth(
    layout: tuple[int, ...],
    K: Any,
    omega: Any,
    coupling_map: Any,
    *,
    t: float,
    reps: int,
) -> int:
    """Depth model whose optimum is the layout with the smallest indices."""
    return int(sum(layout))


def _weighted_depth(
    layout: tuple[int, ...],
    K: Any,
    omega: Any,
    coupling_map: Any,
    *,
    t: float,
    reps: int,
) -> int:
    """Depth model sensitive to the order of the placed qubits."""
    return int(sum(position * physical for position, physical in enumerate(layout, start=1)))


class TestLayoutSearchConfig:
    """Verify discrete layout-search configuration contracts."""

    def test_defaults_valid_and_serialisable(self) -> None:
        """Expose valid deterministic defaults as a serializable payload."""
        config = LayoutSearchConfig()
        assert config.to_dict() == {
            "n_restarts": 4,
            "max_sweeps": 20,
            "seed": 0,
            "weights": None,
            "t": 0.1,
            "reps": 5,
            "order": 1,
        }

    def test_weights_serialised_when_set(self) -> None:
        """Serialize explicit cost weights into the configuration payload."""
        config = LayoutSearchConfig(weights=CostWeights(depth=2.0))
        assert config.to_dict()["weights"] == {
            "depth": 2.0,
            "trotter_error": 1.0,
            "infidelity": 1.0,
        }

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_restarts": 0}, "n_restarts"),
            ({"max_sweeps": 0}, "max_sweeps"),
            ({"t": 0.0}, "t must be"),
            ({"t": float("inf")}, "t must be"),
            ({"reps": 0}, "reps"),
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs: dict[str, Any], match: str) -> None:
        """Reject non-positive or non-finite search controls."""
        with pytest.raises(ValueError, match=match):
            LayoutSearchConfig(**kwargs)


class TestLayoutSearchResult:
    """Verify immutable layout-search result serialization."""

    def test_to_dict_shape(self) -> None:
        """Serialize the complete result and nested cost breakdown."""
        result = optimise_kuramoto_layout(
            _K,
            _OMEGA,
            coupling_map=None,
            physical_qubits=(0, 1, 2),
            mean_gate_fidelity=_FIDELITY,
            config=LayoutSearchConfig(n_restarts=1),
            depth_provider=_sum_depth,
        )
        assert isinstance(result, LayoutSearchResult)
        payload = result.to_dict()
        assert sorted(payload) == [
            "best_cost",
            "best_layout",
            "converged",
            "n_evaluations",
            "n_restarts",
        ]
        assert payload["best_layout"] == list(result.best_layout)
        assert payload["best_cost"]["routed_depth"] == result.best_cost.routed_depth


class TestSearchSpaceValidation:
    """Verify fail-closed physical search-space admission."""

    @pytest.mark.parametrize(
        ("physical_qubits", "initial_layout", "match"),
        [
            ((0, 1, 1), None, "duplicates"),
            ((0, 1, -2), None, "non-negative"),
            ((0, 1), None, "at least 3 candidate"),
            ((0, 1, 2, 3), (0, 1), "length 3"),
            ((0, 1, 2, 3), (0, 1, 1), "must not contain duplicates"),
            ((0, 1, 2, 3), (0, 1, 9), "drawn from physical_qubits"),
        ],
    )
    def test_malformed_search_space_rejected(
        self,
        physical_qubits: tuple[int, ...],
        initial_layout: tuple[int, ...] | None,
        match: str,
    ) -> None:
        """Reject malformed candidate sets and initial layouts."""
        with pytest.raises(ValueError, match=match):
            optimise_kuramoto_layout(
                _K,
                _OMEGA,
                coupling_map=None,
                physical_qubits=physical_qubits,
                mean_gate_fidelity=_FIDELITY,
                initial_layout=initial_layout,
                depth_provider=_sum_depth,
            )

    def test_cost_validation_propagates(self) -> None:
        """Propagate invalid cost inputs through the optimizer boundary."""
        with pytest.raises(ValueError, match="mean_gate_fidelity"):
            optimise_kuramoto_layout(
                _K,
                _OMEGA,
                coupling_map=None,
                physical_qubits=(0, 1, 2),
                mean_gate_fidelity=1.5,
                depth_provider=_sum_depth,
            )


class TestNeighbourhood:
    """Verify swap and relocate neighborhood construction."""

    def test_swap_only_when_candidates_exhausted(self) -> None:
        """Generate only pairwise swaps when every candidate is occupied."""
        moves = _neighbours((0, 1, 2), (0, 1, 2))
        assert len(moves) == 3  # C(3, 2) swaps, no relocations
        assert all(sorted(move) == [0, 1, 2] for move in moves)

    def test_relocations_added_for_unused_candidates(self) -> None:
        """Generate injective relocations for every unused candidate."""
        moves = _neighbours((0, 1), (0, 1, 5, 7))
        swaps = [move for move in moves if sorted(move) == [0, 1]]
        relocations = [move for move in moves if sorted(move) != [0, 1]]
        assert len(swaps) == 1
        assert len(relocations) == 4  # 2 positions × 2 unused candidates
        assert all(len(set(move)) == 2 for move in moves)


class TestHillClimbing:
    """Verify deterministic best-improvement hill-climbing behavior."""

    def test_finds_known_optimum_via_relocations(self) -> None:
        """Reach the controlled optimum through relocate moves."""
        result = optimise_kuramoto_layout(
            _K,
            _OMEGA,
            coupling_map=None,
            physical_qubits=(0, 1, 2, 50, 60, 70),
            mean_gate_fidelity=_FIDELITY,
            config=LayoutSearchConfig(n_restarts=1),
            initial_layout=(50, 60, 70),
            depth_provider=_sum_depth,
        )
        assert sorted(result.best_layout) == [0, 1, 2]
        assert result.best_cost.routed_depth == 3
        assert result.converged is True

    def test_never_worse_than_seed_layout(self) -> None:
        """Never regress relative to the admitted seeded layout."""
        seed_layout = (2, 1, 0)
        seed_cost = kuramoto_layout_cost(
            seed_layout,
            _K,
            _OMEGA,
            None,
            mean_gate_fidelity=_FIDELITY,
            depth_provider=_weighted_depth,
        )
        result = optimise_kuramoto_layout(
            _K,
            _OMEGA,
            coupling_map=None,
            physical_qubits=(0, 1, 2),
            mean_gate_fidelity=_FIDELITY,
            config=LayoutSearchConfig(n_restarts=1),
            initial_layout=seed_layout,
            depth_provider=_weighted_depth,
        )
        assert result.best_cost.total <= seed_cost.total
        # weighted depth is minimised by descending physical indices
        assert result.best_layout == (2, 1, 0)

    def test_deterministic_for_fixed_seed(self) -> None:
        """Return identical outcomes for repeated fixed-seed searches."""
        outcomes = [
            optimise_kuramoto_layout(
                _K,
                _OMEGA,
                coupling_map=None,
                physical_qubits=(0, 1, 2, 3, 4),
                mean_gate_fidelity=_FIDELITY,
                config=LayoutSearchConfig(n_restarts=3, seed=11),
                depth_provider=_weighted_depth,
            )
            for _ in range(2)
        ]
        assert outcomes[0].best_layout == outcomes[1].best_layout
        assert outcomes[0].n_evaluations == outcomes[1].n_evaluations

    def test_sweep_cap_reports_non_convergence(self) -> None:
        """Report non-convergence when the sweep cap stops improvement."""
        result = optimise_kuramoto_layout(
            _K,
            _OMEGA,
            coupling_map=None,
            physical_qubits=(0, 1, 2, 50, 60, 70),
            mean_gate_fidelity=_FIDELITY,
            config=LayoutSearchConfig(n_restarts=1, max_sweeps=1),
            initial_layout=(50, 60, 70),
            depth_provider=_sum_depth,
        )
        assert result.converged is False
        assert result.best_cost.routed_depth < 50 + 60 + 70

    def test_memoisation_counts_distinct_layouts_once(self) -> None:
        """Evaluate each distinct layout no more than once."""
        calls: list[tuple[int, ...]] = []

        def counting_depth(
            layout: tuple[int, ...],
            K: Any,
            omega: Any,
            coupling_map: Any,
            *,
            t: float,
            reps: int,
        ) -> int:
            calls.append(layout)
            return int(sum(layout))

        result = optimise_kuramoto_layout(
            _K,
            _OMEGA,
            coupling_map=None,
            physical_qubits=(0, 1, 2, 3),
            mean_gate_fidelity=_FIDELITY,
            config=LayoutSearchConfig(n_restarts=2, seed=3),
            depth_provider=counting_depth,
        )
        assert len(calls) == result.n_evaluations
        assert len(set(calls)) == len(calls)

    def test_later_restart_can_improve_on_seeded_first(self) -> None:
        """Allow later deterministic restarts to improve the seeded run."""
        result = optimise_kuramoto_layout(
            _K,
            _OMEGA,
            coupling_map=None,
            physical_qubits=(0, 1, 2, 50, 60, 70),
            mean_gate_fidelity=_FIDELITY,
            config=LayoutSearchConfig(n_restarts=4, max_sweeps=1, seed=0),
            initial_layout=(50, 60, 70),
            depth_provider=_sum_depth,
        )
        first_restart_best = 60 + 70  # one sweep relocates a single qubit to 0
        assert result.best_cost.routed_depth < first_restart_best
        assert result.n_restarts == 4


class TestRoutedDepthIntegration:
    """Verify integration with the real routed-depth adapter."""

    def test_real_depth_provider_on_line_coupling(self) -> None:
        """Route a bounded three-qubit problem on a line coupling map."""
        coupling = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1)])
        result = optimise_kuramoto_layout(
            _K,
            _OMEGA,
            coupling,
            physical_qubits=(0, 1, 2),
            mean_gate_fidelity=_FIDELITY,
            config=LayoutSearchConfig(n_restarts=1, max_sweeps=1, reps=1),
        )
        assert result.best_cost.routed_depth > 0
        assert sorted(result.best_layout) == [0, 1, 2]


class TestPackageExport:
    """Verify stable hardware-package exports."""

    def test_hardware_package_exports_optimiser(self) -> None:
        """Re-export the optimizer function and immutable record types."""
        from scpn_quantum_control import hardware

        assert hardware.optimise_kuramoto_layout is optimise_kuramoto_layout
        assert hardware.LayoutSearchConfig is LayoutSearchConfig
        assert hardware.LayoutSearchResult is LayoutSearchResult
