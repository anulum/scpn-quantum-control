# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for error-aware chain selection
"""Tests for hardware/error_aware_chain.py.

Chain growth is exercised on hand-built line, ring, star, and disconnected
graphs where the optimal path is known by inspection, plus a heavy-hex-like
fragment; every dataclass invariant and fail-closed branch is pinned.
"""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.error_aware_chain import (
    ChainSelection,
    longest_error_aware_chain,
    select_error_aware_chain,
)


def line_graph(
    n: int, gate: float = 0.01, readout: float = 0.02
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    gates = {(i, i + 1): gate for i in range(n - 1)}
    readouts = {i: readout for i in range(n)}
    return gates, readouts


class TestChainSelectionDataclass:
    def test_valid_selection_and_derived_properties(self) -> None:
        selection = ChainSelection(
            qubits=(3, 1, 2),
            edge_errors=(0.01, 0.03),
            readout_errors=(0.02, 0.04, 0.06),
        )
        assert selection.length == 3
        # edge 0: 0.01 + (0.02+0.04)/2 = 0.04; edge 1: 0.03 + (0.04+0.06)/2 = 0.08
        assert selection.total_score == pytest.approx(0.12)
        assert selection.median_edge_error == pytest.approx(0.02)

    def test_median_of_odd_edge_count(self) -> None:
        selection = ChainSelection(
            qubits=(0, 1, 2, 3),
            edge_errors=(0.05, 0.01, 0.03),
            readout_errors=(0.0, 0.0, 0.0, 0.0),
        )
        assert selection.median_edge_error == pytest.approx(0.03)

    def test_too_short_chain_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="at least two qubits"):
            ChainSelection(qubits=(0,), edge_errors=(), readout_errors=(0.1,))

    def test_revisited_qubit_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="must not revisit"):
            ChainSelection(
                qubits=(0, 1, 0),
                edge_errors=(0.1, 0.1),
                readout_errors=(0.1, 0.1, 0.1),
            )

    def test_wrong_edge_count_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="one entry per chain edge"):
            ChainSelection(qubits=(0, 1), edge_errors=(0.1, 0.2), readout_errors=(0.1, 0.1))

    def test_wrong_readout_count_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="one entry per chain qubit"):
            ChainSelection(qubits=(0, 1), edge_errors=(0.1,), readout_errors=(0.1,))


class TestSelectErrorAwareChain:
    def test_full_line_is_recovered_in_order(self) -> None:
        gates, readouts = line_graph(6)
        selection = select_error_aware_chain(gates, readouts, 6)
        assert selection is not None
        assert selection.qubits in ((0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0))
        assert selection.edge_errors == (0.01,) * 5

    def test_prefers_the_low_error_branch(self) -> None:
        # Star at 1: branches 1-0 (cheap) and 1-2-3 (cheap), 1-9 expensive.
        gates = {
            (0, 1): 0.001,
            (1, 2): 0.001,
            (2, 3): 0.001,
            (1, 9): 0.5,
        }
        readouts = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 9: 0.01}
        selection = select_error_aware_chain(gates, readouts, 4)
        assert selection is not None
        assert set(selection.qubits) == {0, 1, 2, 3}
        assert 9 not in selection.qubits

    def test_unreachable_length_returns_none(self) -> None:
        gates, readouts = line_graph(4)
        assert select_error_aware_chain(gates, readouts, 5) is None

    def test_disconnected_components_do_not_bridge(self) -> None:
        gates = {(0, 1): 0.001, (2, 3): 0.001}
        readouts = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01}
        assert select_error_aware_chain(gates, readouts, 3) is None

    def test_edges_without_readout_calibration_are_excluded(self) -> None:
        gates = {(0, 1): 0.001, (1, 2): 0.001}
        readouts = {0: 0.01, 1: 0.01}
        assert select_error_aware_chain(gates, readouts, 3) is None
        selection = select_error_aware_chain(gates, readouts, 2)
        assert selection is not None
        assert set(selection.qubits) == {0, 1}

    def test_reversed_duplicate_edges_keep_the_smaller_error(self) -> None:
        gates = {(0, 1): 0.02, (1, 0): 0.005, (1, 2): 0.01}
        readouts = {0: 0.0, 1: 0.0, 2: 0.0}
        selection = select_error_aware_chain(gates, readouts, 3)
        assert selection is not None
        assert selection.edge_errors in ((0.005, 0.01), (0.01, 0.005))

    def test_duplicate_edge_with_larger_error_is_ignored(self) -> None:
        # Smaller error seen first: the later, larger duplicate must not win.
        gates = {(1, 0): 0.005, (0, 1): 0.02, (1, 2): 0.01}
        readouts = {0: 0.0, 1: 0.0, 2: 0.0}
        selection = select_error_aware_chain(gates, readouts, 3)
        assert selection is not None
        assert selection.edge_errors in ((0.005, 0.01), (0.01, 0.005))

    def test_self_loop_edges_are_ignored(self) -> None:
        gates = {(1, 1): 0.0001, (0, 1): 0.01, (1, 2): 0.01}
        readouts = {0: 0.01, 1: 0.01, 2: 0.01}
        selection = select_error_aware_chain(gates, readouts, 3)
        assert selection is not None
        assert set(selection.qubits) == {0, 1, 2}

    def test_heavy_hex_fragment_ring_walks_around(self) -> None:
        # A 12-node ring (heavy-hex loops are rings of 12): a chain of 12
        # exists; a chain of 13 does not.
        gates = {(i, (i + 1) % 12): 0.01 for i in range(12)}
        readouts = {i: 0.02 for i in range(12)}
        assert select_error_aware_chain(gates, readouts, 12) is not None
        assert select_error_aware_chain(gates, readouts, 13) is None

    def test_multiple_seeds_find_the_globally_better_chain(self) -> None:
        # Cheapest edge (10, 11) is isolated; only the line reaches width 4.
        gates = {(10, 11): 0.0001, (0, 1): 0.01, (1, 2): 0.01, (2, 3): 0.01}
        readouts = {10: 0.0, 11: 0.0, 0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02}
        selection = select_error_aware_chain(gates, readouts, 4, seed_count=4)
        assert selection is not None
        assert set(selection.qubits) == {0, 1, 2, 3}

    def test_invalid_length_fails_closed(self) -> None:
        gates, readouts = line_graph(3)
        with pytest.raises(ValueError, match="at least 2"):
            select_error_aware_chain(gates, readouts, 1)

    def test_invalid_seed_count_fails_closed(self) -> None:
        gates, readouts = line_graph(3)
        with pytest.raises(ValueError, match="seed_count"):
            select_error_aware_chain(gates, readouts, 2, seed_count=0)

    def test_empty_graph_returns_none(self) -> None:
        assert select_error_aware_chain({}, {}, 2) is None


def t_junction_graph() -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    """A greedy trap: a cheap dead-end branch off a 7-qubit line.

    The cheapest edge (3, 7) seeds the greedy walk into the two-qubit spur
    7-8, wasting qubit 3's line position; the longest simple path is the
    plain line 0..6 (7 qubits), which only backtracking recovers.
    """
    gates = {
        (0, 1): 0.01,
        (1, 2): 0.01,
        (2, 3): 0.01,
        (3, 4): 0.01,
        (4, 5): 0.01,
        (5, 6): 0.01,
        (3, 7): 0.0001,
        (7, 8): 0.0001,
    }
    readouts = {index: 0.01 for index in range(9)}
    return gates, readouts


class TestBacktrackingSearch:
    def test_greedy_alone_dead_ends_in_the_spur(self) -> None:
        # Every greedy walk reaching qubit 3 takes the cheap spur and ends
        # trapped at 8, so no seed recovers the full 7-qubit line.
        gates, readouts = t_junction_graph()
        greedy = longest_error_aware_chain(gates, readouts, seed_count=4)
        assert greedy is not None
        assert greedy.length < 7
        assert 8 in greedy.qubits

    def test_backtracking_recovers_the_full_line(self) -> None:
        gates, readouts = t_junction_graph()
        selection = longest_error_aware_chain(gates, readouts, seed_count=4, backtrack_steps=1000)
        assert selection is not None
        assert selection.length == 7
        assert set(selection.qubits) == set(range(7))

    def test_select_by_length_uses_the_backtracking_budget(self) -> None:
        gates, readouts = t_junction_graph()
        assert select_error_aware_chain(gates, readouts, 7, seed_count=4) is None
        selection = select_error_aware_chain(
            gates, readouts, 7, seed_count=4, backtrack_steps=1000
        )
        assert selection is not None
        assert selection.length == 7

    def test_exhausted_budget_still_returns_the_best_partial(self) -> None:
        gates, readouts = t_junction_graph()
        selection = longest_error_aware_chain(gates, readouts, seed_count=1, backtrack_steps=1)
        assert selection is not None
        assert selection.length >= 2

    def test_negative_budget_fails_closed(self) -> None:
        gates, readouts = t_junction_graph()
        with pytest.raises(ValueError, match="backtrack_steps"):
            longest_error_aware_chain(gates, readouts, backtrack_steps=-1)
        with pytest.raises(ValueError, match="backtrack_steps"):
            select_error_aware_chain(gates, readouts, 3, backtrack_steps=-1)

    def test_deterministic_across_repeat_calls(self) -> None:
        gates, readouts = t_junction_graph()
        first = longest_error_aware_chain(gates, readouts, backtrack_steps=500)
        second = longest_error_aware_chain(gates, readouts, backtrack_steps=500)
        assert first is not None and second is not None
        assert first.qubits == second.qubits

    def test_heavy_hex_like_ring_with_spurs_reaches_full_coverage(self) -> None:
        # A 12-ring with a pendant qubit on every even node: the longest
        # simple path is 12 + 2 = 14 (walk the ring, ending on two pendants).
        gates: dict[tuple[int, int], float] = {(i, (i + 1) % 12): 0.01 for i in range(12)}
        for even in range(0, 12, 2):
            gates[(even, 20 + even)] = 0.001
        readouts = {q: 0.01 for q in list(range(12)) + [20 + e for e in range(0, 12, 2)]}
        selection = longest_error_aware_chain(
            gates, readouts, seed_count=4, backtrack_steps=50_000
        )
        assert selection is not None
        assert selection.length >= 13


class TestLongestErrorAwareChain:
    def test_line_yields_its_full_length(self) -> None:
        gates, readouts = line_graph(7)
        selection = longest_error_aware_chain(gates, readouts)
        assert selection is not None
        assert selection.length == 7

    def test_longer_beats_cheaper(self) -> None:
        # A cheap isolated pair vs a longer, costlier line: longest wins.
        gates = {(10, 11): 0.0001, (0, 1): 0.05, (1, 2): 0.05}
        readouts = {10: 0.0, 11: 0.0, 0: 0.05, 1: 0.05, 2: 0.05}
        selection = longest_error_aware_chain(gates, readouts, seed_count=4)
        assert selection is not None
        assert selection.length == 3
        assert set(selection.qubits) == {0, 1, 2}

    def test_equal_length_ties_resolve_to_cheaper_chain(self) -> None:
        gates = {(0, 1): 0.05, (10, 11): 0.001}
        readouts = {0: 0.05, 1: 0.05, 10: 0.001, 11: 0.001}
        selection = longest_error_aware_chain(gates, readouts, seed_count=4)
        assert selection is not None
        assert selection.qubits in ((10, 11), (11, 10))

    def test_empty_graph_returns_none(self) -> None:
        assert longest_error_aware_chain({}, {}) is None

    def test_invalid_seed_count_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="seed_count"):
            longest_error_aware_chain({}, {}, seed_count=0)
