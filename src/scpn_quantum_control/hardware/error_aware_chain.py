# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — error-aware 1-D chain selection on device graphs
"""Error-aware selection of 1-D nearest-neighbour qubit chains.

The maximum-width Kuramoto-XY campaign
(``docs/campaigns/max_width_kuramoto_xy_prereg_2026-07-16.md``) executes
shallow chain circuits at widths up to the usable size of the device, so it
needs an ordered simple path through the coupling graph rather than the
compact regions :mod:`.qubit_mapper` selects. This module walks the
calibrated coupling graph greedily: chains grow from low-error seed edges by
repeatedly attaching the cheapest available edge at either endpoint, where an
edge costs its two-qubit gate error plus the mean readout error of its
qubits — the same scoring convention as the DynQ calibration graph.

Greedy bidirectional growth is a heuristic (maximum simple path is NP-hard);
restarting from several seed edges and keeping the best chain makes it
robust on heavy-hex topologies while staying deterministic for a given
calibration snapshot. Greedy alone dead-ends well short of the device on
heavy-hex (51 of 156 qubits on a real Heron r2 snapshot), so both entry
points also take a ``backtrack_steps`` budget: a deterministic
depth-first search with score-ordered children that backtracks out of
dead ends until the budget is spent, which reaches near-device-spanning
chains at a bounded, deterministic cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "ChainSelection",
    "longest_error_aware_chain",
    "select_error_aware_chain",
]

DEFAULT_SEED_COUNT = 8


@dataclass(frozen=True)
class ChainSelection:
    """An ordered 1-D nearest-neighbour chain of physical qubits.

    ``qubits`` is the path order (each consecutive pair is a calibrated
    coupling-graph edge); ``edge_errors`` and ``readout_errors`` follow that
    order, so ``edge_errors[i]`` joins ``qubits[i]`` and ``qubits[i + 1]``.
    """

    qubits: tuple[int, ...]
    edge_errors: tuple[float, ...]
    readout_errors: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the path-shape invariants of the selection."""
        if len(self.qubits) < 2:
            raise ValueError("a chain needs at least two qubits")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("a chain must not revisit a qubit")
        if len(self.edge_errors) != len(self.qubits) - 1:
            raise ValueError("edge_errors must have one entry per chain edge")
        if len(self.readout_errors) != len(self.qubits):
            raise ValueError("readout_errors must have one entry per chain qubit")

    @property
    def length(self) -> int:
        """Number of qubits in the chain."""
        return len(self.qubits)

    @property
    def total_score(self) -> float:
        """Sum of edge scores (gate error + mean endpoint readout error)."""
        total = 0.0
        for index, edge_error in enumerate(self.edge_errors):
            mean_readout = (self.readout_errors[index] + self.readout_errors[index + 1]) / 2.0
            total += edge_error + mean_readout
        return total

    @property
    def median_edge_error(self) -> float:
        """Median two-qubit gate error along the chain."""
        ordered = sorted(self.edge_errors)
        middle = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0


def _edge_score(
    edge: tuple[int, int],
    gate_errors: Mapping[tuple[int, int], float],
    readout_errors: Mapping[int, float],
) -> float:
    gate = gate_errors[edge]
    return gate + (readout_errors[edge[0]] + readout_errors[edge[1]]) / 2.0


def _calibrated_adjacency(
    gate_errors: Mapping[tuple[int, int], float],
    readout_errors: Mapping[int, float],
) -> dict[int, dict[int, float]]:
    """Adjacency of fully calibrated edges (gate + both readout errors known)."""
    adjacency: dict[int, dict[int, float]] = {}
    for (first, second), gate_error in gate_errors.items():
        if first == second:
            continue
        if first not in readout_errors or second not in readout_errors:
            continue
        score = gate_error + (readout_errors[first] + readout_errors[second]) / 2.0
        adjacency.setdefault(first, {})[second] = score
        adjacency.setdefault(second, {})[first] = score
    return adjacency


def _grow_chain(
    seed: tuple[int, int],
    adjacency: Mapping[int, Mapping[int, float]],
    target_length: int | None,
) -> list[int]:
    """Grow a chain from a seed edge by cheapest-endpoint extension."""
    chain = [seed[0], seed[1]]
    visited = set(chain)
    while target_length is None or len(chain) < target_length:
        candidates: list[tuple[float, str, int]] = []
        for side, endpoint in (("head", chain[0]), ("tail", chain[-1])):
            for neighbour, score in adjacency[endpoint].items():
                if neighbour not in visited:
                    candidates.append((score, side, neighbour))
        if not candidates:
            break
        candidates.sort()
        _, side, neighbour = candidates[0]
        if side == "head":
            chain.insert(0, neighbour)
        else:
            chain.append(neighbour)
        visited.add(neighbour)
    return chain


def _dfs_longest(
    seed: tuple[int, int],
    adjacency: Mapping[int, Mapping[int, float]],
    step_budget: int,
    target_length: int | None,
) -> list[int]:
    """Deterministic DFS chain growth from ``seed`` with backtracking.

    Children are explored in ascending edge-score order, so the first
    completed chain follows the greedy path and backtracking then unwinds
    dead ends. The search charges one step per extension attempt and stops
    when the budget is spent, the target length is reached, or the space is
    exhausted; the longest chain seen is returned.
    """
    chain = [seed[0], seed[1]]
    visited = {seed[0], seed[1]}
    best = list(chain)
    steps_left = step_budget

    def extend() -> bool:
        nonlocal steps_left, best
        if len(chain) > len(best):
            best = list(chain)
        if target_length is not None and len(chain) >= target_length:
            return True
        tail = chain[-1]
        for _score, neighbour in sorted(
            (score, neighbour)
            for neighbour, score in adjacency[tail].items()
            if neighbour not in visited
        ):
            if steps_left <= 0:
                return False
            steps_left -= 1
            chain.append(neighbour)
            visited.add(neighbour)
            if extend():
                return True
            chain.pop()
            visited.remove(neighbour)
        return False

    extend()
    return best


def _chain_selection(
    chain: list[int],
    gate_errors: Mapping[tuple[int, int], float],
    readout_errors: Mapping[int, float],
) -> ChainSelection:
    edges = []
    for first, second in zip(chain, chain[1:], strict=False):
        key = (min(first, second), max(first, second))
        edges.append(float(gate_errors[key]))
    return ChainSelection(
        qubits=tuple(chain),
        edge_errors=tuple(edges),
        readout_errors=tuple(float(readout_errors[qubit]) for qubit in chain),
    )


def _canonical_gate_errors(
    gate_errors: Mapping[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    canonical: dict[tuple[int, int], float] = {}
    for (first, second), error in gate_errors.items():
        key = (min(first, second), max(first, second))
        known = canonical.get(key)
        if known is None or float(error) < known:
            canonical[key] = float(error)
    return canonical


def _seed_edges(
    adjacency: Mapping[int, Mapping[int, float]],
    gate_errors: Mapping[tuple[int, int], float],
    readout_errors: Mapping[int, float],
    seed_count: int,
) -> list[tuple[int, int]]:
    edges = {
        (min(first, second), max(first, second))
        for first, neighbours in adjacency.items()
        for second in neighbours
    }
    ranked = sorted(edges, key=lambda edge: _edge_score(edge, gate_errors, readout_errors))
    return ranked[:seed_count]


def _candidate_chains(
    seed: tuple[int, int],
    adjacency: Mapping[int, Mapping[int, float]],
    target_length: int | None,
    backtrack_steps: int,
) -> list[list[int]]:
    """Chains grown from one seed: greedy, plus both DFS orientations."""
    chains = [_grow_chain(seed, adjacency, target_length)]
    if backtrack_steps > 0:
        chains.append(_dfs_longest(seed, adjacency, backtrack_steps, target_length))
        chains.append(_dfs_longest((seed[1], seed[0]), adjacency, backtrack_steps, target_length))
    return chains


def select_error_aware_chain(
    gate_errors: Mapping[tuple[int, int], float],
    readout_errors: Mapping[int, float],
    length: int,
    *,
    seed_count: int = DEFAULT_SEED_COUNT,
    backtrack_steps: int = 0,
) -> ChainSelection | None:
    """Select an ordered chain of ``length`` qubits minimising the error score.

    Grows chains from the ``seed_count`` lowest-error edges and returns the
    reachable chain of exactly ``length`` qubits with the smallest total
    score, or ``None`` when no seed reaches the requested length. The result
    is deterministic for a given calibration snapshot.

    Parameters
    ----------
    gate_errors:
        Two-qubit gate error per coupling-graph edge (any qubit order).
    readout_errors:
        Readout assignment error per qubit; edges touching a qubit with no
        readout calibration are excluded.
    length:
        Requested chain width; must be at least two.
    seed_count:
        Number of lowest-error seed edges to restart from.
    backtrack_steps:
        Extra depth-first backtracking budget per seed and orientation;
        zero keeps the pure greedy behaviour.
    """
    if length < 2:
        raise ValueError("chain length must be at least 2")
    if seed_count < 1:
        raise ValueError("seed_count must be at least 1")
    if backtrack_steps < 0:
        raise ValueError("backtrack_steps must be non-negative")
    canonical = _canonical_gate_errors(gate_errors)
    adjacency = _calibrated_adjacency(canonical, readout_errors)
    best: ChainSelection | None = None
    for seed in _seed_edges(adjacency, canonical, readout_errors, seed_count):
        for chain in _candidate_chains(seed, adjacency, length, backtrack_steps):
            if len(chain) != length:
                continue
            selection = _chain_selection(chain, canonical, readout_errors)
            if best is None or selection.total_score < best.total_score:
                best = selection
    return best


def longest_error_aware_chain(
    gate_errors: Mapping[tuple[int, int], float],
    readout_errors: Mapping[int, float],
    *,
    seed_count: int = DEFAULT_SEED_COUNT,
    backtrack_steps: int = 0,
) -> ChainSelection | None:
    """Grow chains to exhaustion and return the longest one found.

    Ties in length resolve to the smaller total score. Returns ``None`` when
    the calibrated coupling graph has no usable edge at all. A positive
    ``backtrack_steps`` budget lets the search unwind greedy dead ends,
    which materially lengthens the result on heavy-hex devices.
    """
    if seed_count < 1:
        raise ValueError("seed_count must be at least 1")
    if backtrack_steps < 0:
        raise ValueError("backtrack_steps must be non-negative")
    canonical = _canonical_gate_errors(gate_errors)
    adjacency = _calibrated_adjacency(canonical, readout_errors)
    best: ChainSelection | None = None
    for seed in _seed_edges(adjacency, canonical, readout_errors, seed_count):
        for chain in _candidate_chains(seed, adjacency, None, backtrack_steps):
            selection = _chain_selection(chain, canonical, readout_errors)
            if (
                best is None
                or selection.length > best.length
                or (selection.length == best.length and selection.total_score < best.total_score)
            ):
                best = selection
    return best
