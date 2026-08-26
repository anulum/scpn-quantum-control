# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the Kuramoto-XY-aware layout cost model
"""Multi-angle tests for hardware/kuramoto_layout_cost.py.

Dimensions: weight invariants and serialisation, cost-input validation, the
pure cost combinator with an injected depth provider, the DynQ-fidelity helper,
and one integration test of the real routed-depth adapter on a small coupling
map.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qiskit.transpiler import CouplingMap

from scpn_quantum_control.hardware.kuramoto_layout_cost import (
    CostWeights,
    LayoutCost,
    dynq_mean_gate_fidelity,
    kuramoto_layout_cost,
    routed_layout_depth,
)
from scpn_quantum_control.hardware.qubit_mapper import ExecutionRegion, QubitMappingResult

_K = np.array(
    [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
)
_OMEGA = np.array([0.1, 0.2, 0.3, 0.4])


def _fixed_depth(value: int) -> Any:
    def _provider(
        layout: tuple[int, ...],
        K: Any,
        omega: Any,
        coupling_map: Any,
        *,
        t: float,
        reps: int,
    ) -> int:
        return value

    return _provider


class TestCostWeights:
    """Exercise weighted-objective configuration invariants."""

    def test_defaults_valid(self) -> None:
        """Use unit weights for every objective term by default."""
        assert CostWeights().to_dict() == {"depth": 1.0, "trotter_error": 1.0, "infidelity": 1.0}

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"depth": -1.0}, "depth"),
            ({"trotter_error": float("inf")}, "trotter_error"),
            ({"infidelity": float("nan")}, "infidelity"),
            ({"depth": 0.0, "trotter_error": 0.0, "infidelity": 0.0}, "at least one"),
        ],
    )
    def test_invalid_weights_raise(self, kwargs: dict[str, Any], match: str) -> None:
        """Reject negative, non-finite, or uniformly zero weights."""
        with pytest.raises(ValueError, match=match):
            CostWeights(**kwargs)


class TestValidation:
    """Exercise fail-closed layout and problem validation."""

    @pytest.mark.parametrize(
        ("layout", "fidelity", "t", "reps", "match"),
        [
            ((0, 1, 2), 0.99, 0.1, 5, "layout must have length"),
            ((0, 1, 2, 2), 0.99, 0.1, 5, "distinct"),
            ((0, 1, 2, -1), 0.99, 0.1, 5, "non-negative"),
            ((0, 1, 2, 3), 1.5, 0.1, 5, "mean_gate_fidelity"),
            ((0, 1, 2, 3), 0.99, 0.0, 5, "t must be finite"),
            ((0, 1, 2, 3), 0.99, 0.1, 0, "reps"),
        ],
    )
    def test_invalid_inputs_raise(
        self, layout: tuple[int, ...], fidelity: float, t: float, reps: int, match: str
    ) -> None:
        """Reject invalid placements, fidelity, time, or repetitions."""
        with pytest.raises(ValueError, match=match):
            kuramoto_layout_cost(
                layout,
                _K,
                _OMEGA,
                CouplingMap([[0, 1], [1, 2], [2, 3]]),
                mean_gate_fidelity=fidelity,
                t=t,
                reps=reps,
                depth_provider=_fixed_depth(10),
            )

    def test_non_square_K_raises(self) -> None:
        """Reject a non-square coupling matrix."""
        with pytest.raises(ValueError, match="square"):
            kuramoto_layout_cost(
                (0, 1),
                np.ones((2, 3)),
                _OMEGA,
                None,
                mean_gate_fidelity=0.9,
                depth_provider=_fixed_depth(1),
            )

    def test_omega_shape_mismatch_raises(self) -> None:
        """Reject a frequency vector with the wrong oscillator count."""
        with pytest.raises(ValueError, match="omega must have shape"):
            kuramoto_layout_cost(
                (0, 1, 2, 3),
                _K,
                np.array([0.1, 0.2]),
                None,
                mean_gate_fidelity=0.9,
                depth_provider=_fixed_depth(1),
            )


class TestCostCombinator:
    """Exercise objective-term assembly and serialisation."""

    def test_weighted_sum_with_injected_depth(self) -> None:
        """Combine injected depth, error, and infidelity by their weights."""
        cost = kuramoto_layout_cost(
            (0, 1, 2, 3),
            _K,
            _OMEGA,
            None,
            mean_gate_fidelity=0.9,
            weights=CostWeights(depth=2.0, trotter_error=0.0, infidelity=10.0),
            depth_provider=_fixed_depth(7),
        )
        assert isinstance(cost, LayoutCost)
        assert cost.depth_term == 14.0
        assert cost.trotter_error_term == 0.0
        assert cost.infidelity_term == pytest.approx(1.0)
        assert cost.total == pytest.approx(15.0)
        assert cost.routed_depth == 7

    def test_default_weights_and_serialisation(self) -> None:
        """Serialise every component under default unit weights."""
        cost = kuramoto_layout_cost(
            (0, 1, 2, 3),
            _K,
            _OMEGA,
            None,
            mean_gate_fidelity=1.0,
            depth_provider=_fixed_depth(5),
        )
        # Unit weights, perfect fidelity → infidelity term is zero.
        assert cost.infidelity_term == 0.0
        assert cost.trotter_error > 0.0
        payload = cost.to_dict()
        assert payload["routed_depth"] == 5
        assert set(payload) == {
            "total",
            "routed_depth",
            "trotter_error",
            "mean_gate_fidelity",
            "depth_term",
            "trotter_error_term",
            "infidelity_term",
        }

    def test_second_order_trotter(self) -> None:
        """Price a second-order product-formula error bound."""
        cost = kuramoto_layout_cost(
            (0, 1, 2, 3),
            _K,
            _OMEGA,
            None,
            mean_gate_fidelity=0.95,
            order=2,
            depth_provider=_fixed_depth(3),
        )
        assert cost.trotter_error > 0.0


class TestDynqFidelity:
    """Exercise fidelity extraction from a selected execution region."""

    def test_extracts_region_fidelity(self) -> None:
        """Return the selected region's mean gate fidelity."""
        region = ExecutionRegion(
            qubits=frozenset({0, 1, 2}),
            quality_score=0.8,
            connectivity=0.9,
            mean_gate_fidelity=0.97,
            n_qubits=3,
        )
        result = QubitMappingResult(
            selected_region=region,
            all_regions=[region],
            initial_layout=[0, 1, 2],
            resolution=1.0,
        )
        assert dynq_mean_gate_fidelity(result) == pytest.approx(0.97)


class TestRoutedDepthAdapter:
    """Exercise the real Qiskit routing boundary."""

    def test_real_routing_returns_positive_depth(self) -> None:
        """Return a positive integer depth for a connected target."""
        coupling_map = CouplingMap([[0, 1], [1, 2], [2, 3]])
        depth = routed_layout_depth((0, 1, 2, 3), _K, _OMEGA, coupling_map, t=0.1, reps=1)
        assert isinstance(depth, int)
        assert depth > 0

    def test_end_to_end_default_provider(self) -> None:
        """Use real routing through the public cost function default."""
        coupling_map = CouplingMap([[0, 1], [1, 2], [2, 3]])
        cost = kuramoto_layout_cost(
            (0, 1, 2, 3), _K, _OMEGA, coupling_map, mean_gate_fidelity=0.99, t=0.1, reps=1
        )
        assert cost.routed_depth > 0
        assert cost.total > 0.0

    def test_seeded_routing_is_reproducible(self) -> None:
        """Return the same routed depth for a fixed transpiler seed."""
        coupling_map = CouplingMap([[0, 1], [1, 2], [2, 3]])
        depths = [
            routed_layout_depth(
                (3, 1, 0, 2), _K, _OMEGA, coupling_map, t=0.1, reps=1, seed_transpiler=11
            )
            for _ in range(2)
        ]
        assert depths[0] == depths[1]
        assert depths[0] > 0
