# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Discrete layout optimiser over the Kuramoto-XY cost
"""Discrete layout optimiser over the Kuramoto-XY-aware cost model.

Searches the space of injective placements of the ``n`` logical qubits of the
XY-Trotter circuit onto a set of candidate physical qubits, minimising
:func:`~scpn_quantum_control.hardware.kuramoto_layout_cost.kuramoto_layout_cost`.
The search is multi-restart best-improvement hill climbing over two move
types:

* **swap** — exchange the physical qubits of two logical qubits;
* **relocate** — move one logical qubit onto an unused candidate physical
  qubit (only available when there are more candidates than logical qubits).

The first restart is seeded by ``initial_layout`` when given (typically the
DynQ layout from
:func:`~scpn_quantum_control.hardware.qubit_mapper.dynq_initial_layout`), so
the optimiser never returns a layout worse than its seed on the same cost.
Later restarts start from seeded random permutations, making the whole search
deterministic for a fixed :class:`LayoutSearchConfig`.

Costs are memoised per layout tuple, so the reported evaluation count is the
number of *distinct* layouts scored. The search inherits the purity of the
cost function: with an injected ``depth_provider`` it performs no I/O and no
transpilation, which keeps optimiser unit tests fast and coverage-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

import numpy as np

from .kuramoto_layout_cost import (
    CostWeights,
    DepthProvider,
    FloatArray,
    LayoutCost,
    kuramoto_layout_cost,
    routed_layout_depth,
)


@dataclass(frozen=True)
class LayoutSearchConfig:
    """Configuration of the multi-restart hill-climbing layout search.

    Parameters
    ----------
    n_restarts
        Number of independent hill-climbing restarts (the first is seeded by
        ``initial_layout`` when one is supplied).
    max_sweeps
        Maximum number of best-improvement sweeps per restart; each sweep
        scores every swap and relocate neighbour of the current layout.
    seed
        Seed for the restart permutations; the search is deterministic for a
        fixed configuration.
    weights
        Cost-term weights forwarded to
        :func:`~scpn_quantum_control.hardware.kuramoto_layout_cost.kuramoto_layout_cost`;
        ``None`` selects the unit defaults.
    t, reps, order
        Evolution time, Trotter repetitions, and product-formula order
        forwarded to the cost function.
    """

    n_restarts: int = 4
    max_sweeps: int = 20
    seed: int = 0
    weights: CostWeights | None = None
    t: float = 0.1
    reps: int = 5
    order: int = 1

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises
        ------
        ValueError
            If ``n_restarts`` or ``max_sweeps`` is not positive, or ``t`` is
            not finite and positive, or ``reps`` is not positive.
        """
        if self.n_restarts < 1:
            raise ValueError("n_restarts must be a positive integer")
        if self.max_sweeps < 1:
            raise ValueError("max_sweeps must be a positive integer")
        if not isfinite(self.t) or self.t <= 0.0:
            raise ValueError("t must be finite and positive")
        if self.reps < 1:
            raise ValueError("reps must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the configuration."""
        return {
            "n_restarts": self.n_restarts,
            "max_sweeps": self.max_sweeps,
            "seed": self.seed,
            "weights": None if self.weights is None else self.weights.to_dict(),
            "t": self.t,
            "reps": self.reps,
            "order": self.order,
        }


@dataclass(frozen=True)
class LayoutSearchResult:
    """Outcome of the discrete layout search.

    ``converged`` is ``True`` when every restart stopped because it had no
    improving neighbour (a local optimum), rather than because it exhausted
    ``max_sweeps``.
    """

    best_layout: tuple[int, ...]
    best_cost: LayoutCost
    n_evaluations: int
    n_restarts: int
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the search outcome."""
        return {
            "best_layout": list(self.best_layout),
            "best_cost": self.best_cost.to_dict(),
            "n_evaluations": self.n_evaluations,
            "n_restarts": self.n_restarts,
            "converged": self.converged,
        }


def _validate_search_inputs(
    K: FloatArray,
    physical_qubits: tuple[int, ...],
    initial_layout: tuple[int, ...] | None,
) -> int:
    """Validate the search space and return the logical qubit count.

    Raises
    ------
    ValueError
        If the candidate set is too small, contains duplicates or negative
        indices, or the seed layout is not an injective placement of the right
        length drawn from the candidate set.
    """
    n = K.shape[0]
    if len(set(physical_qubits)) != len(physical_qubits):
        raise ValueError("physical_qubits must not contain duplicates")
    if any(index < 0 for index in physical_qubits):
        raise ValueError("physical_qubits indices must be non-negative")
    if len(physical_qubits) < n:
        raise ValueError(
            f"need at least {n} candidate physical qubits, got {len(physical_qubits)}"
        )
    if initial_layout is not None:
        if len(initial_layout) != n:
            raise ValueError(f"initial_layout must have length {n}, got {len(initial_layout)}")
        if len(set(initial_layout)) != n:
            raise ValueError("initial_layout must not contain duplicates")
        if not set(initial_layout) <= set(physical_qubits):
            raise ValueError("initial_layout must be drawn from physical_qubits")
    return n


def _neighbours(
    layout: tuple[int, ...],
    physical_qubits: tuple[int, ...],
) -> list[tuple[int, ...]]:
    """Return all swap and relocate neighbours of ``layout``."""
    n = len(layout)
    moves: list[tuple[int, ...]] = []
    for i in range(n):
        for j in range(i + 1, n):
            swapped = list(layout)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            moves.append(tuple(swapped))
    unused = [q for q in physical_qubits if q not in set(layout)]
    for i in range(n):
        for q in unused:
            relocated = list(layout)
            relocated[i] = q
            moves.append(tuple(relocated))
    return moves


def optimise_kuramoto_layout(
    K: FloatArray,
    omega: FloatArray,
    coupling_map: Any,
    physical_qubits: tuple[int, ...],
    *,
    mean_gate_fidelity: float,
    config: LayoutSearchConfig | None = None,
    initial_layout: tuple[int, ...] | None = None,
    depth_provider: DepthProvider = routed_layout_depth,
) -> LayoutSearchResult:
    """Minimise the Kuramoto-XY layout cost over injective placements.

    Parameters
    ----------
    K, omega
        Coupling matrix and frequency vector for the XY problem; ``K`` fixes
        the logical qubit count ``n``.
    coupling_map
        Hardware connectivity forwarded to ``depth_provider``.
    physical_qubits
        Candidate physical qubits (distinct, non-negative, at least ``n``);
        typically the DynQ selected-region qubits.
    mean_gate_fidelity
        DynQ selected-region mean gate fidelity in ``[0, 1]``.
    config
        Search configuration; ``None`` selects :class:`LayoutSearchConfig`
        defaults.
    initial_layout
        Optional seed layout for the first restart (e.g. the DynQ layout);
        the result is never worse than this seed on the same cost.
    depth_provider
        Post-routing depth callable forwarded to the cost function; defaults
        to
        :func:`~scpn_quantum_control.hardware.kuramoto_layout_cost.routed_layout_depth`.

    Returns
    -------
    LayoutSearchResult
        The best layout found, its cost breakdown, the number of distinct
        layouts scored, and whether every restart reached a local optimum.

    Raises
    ------
    ValueError
        If the search space or the seed layout is malformed (see
        :func:`_validate_search_inputs`), or the cost inputs are invalid.
    """
    config = config or LayoutSearchConfig()
    n = _validate_search_inputs(K, physical_qubits, initial_layout)

    cache: dict[tuple[int, ...], LayoutCost] = {}

    def score(layout: tuple[int, ...]) -> LayoutCost:
        cached = cache.get(layout)
        if cached is None:
            cached = kuramoto_layout_cost(
                layout,
                K,
                omega,
                coupling_map,
                mean_gate_fidelity=mean_gate_fidelity,
                weights=config.weights,
                t=config.t,
                reps=config.reps,
                order=config.order,
                depth_provider=depth_provider,
            )
            cache[layout] = cached
        return cached

    rng = np.random.default_rng(config.seed)
    candidates = tuple(physical_qubits)

    best_layout: tuple[int, ...] | None = None
    best_cost: LayoutCost | None = None
    all_converged = True

    for restart in range(config.n_restarts):
        if restart == 0 and initial_layout is not None:
            current = tuple(initial_layout)
        else:
            permutation = rng.permutation(len(candidates))[:n]
            current = tuple(candidates[index] for index in permutation)
        current_cost = score(current)

        reached_local_optimum = False
        for _ in range(config.max_sweeps):
            scored = [
                (score(neighbour), neighbour) for neighbour in _neighbours(current, candidates)
            ]
            best_neighbour_cost, best_neighbour = min(scored, key=lambda item: item[0].total)
            if best_neighbour_cost.total < current_cost.total:
                current, current_cost = best_neighbour, best_neighbour_cost
            else:
                reached_local_optimum = True
                break
        all_converged = all_converged and reached_local_optimum

        if best_cost is None or current_cost.total < best_cost.total:
            best_layout, best_cost = current, current_cost

    # n_restarts >= 1 guarantees at least one restart assigned the best pair.
    return LayoutSearchResult(
        best_layout=cast("tuple[int, ...]", best_layout),
        best_cost=cast(LayoutCost, best_cost),
        n_evaluations=len(cache),
        n_restarts=config.n_restarts,
        converged=all_converged,
    )
