# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Sinkhorn continuous relaxation of layout search (KT-4)
"""RESEARCH: Sinkhorn continuous relaxation of the Kuramoto layout search.

**Research label.** This module implements the KT-4 open question
preregistered in
``docs/internal/research_synthesis/2026-07-16T1345_kt4_sinkhorn_layout_relaxation_design.md``:
does an annealed Sinkhorn relaxation over placement logits beat the KT-3
discrete optimiser
(:func:`~scpn_quantum_control.hardware.kuramoto_layout_optimiser.optimise_kuramoto_layout`)
on the *true* seeded KT-2 cost at a matched evaluation budget? The honest
outcome may be "modest or no gain"; nothing here is promoted beyond a
research result without the owner-gated KT-5 isolated benchmark.

Method (Mena et al., arXiv:1802.08665; Jang et al., arXiv:1611.01144;
Maddison et al., arXiv:1611.00712 — verified in the design doc):

1. relax the injective placement of ``n`` logical onto ``m`` candidate
   physical qubits into a doubly-stochastic matrix
   ``P = Sinkhorn(logits / τ)`` (dummy rows pad ``n < m``);
2. descend a **differentiable SWAP-distance surrogate**
   ``S(P) = Σ_{i<j} K_ij · (P D Pᵀ)_{ij}`` (``D`` = coupling-graph distances
   between candidates) with the closed-form gradient ``∇_P S = K P D``,
   applied straight-through to the logits (the Jang et al. estimator);
3. anneal τ downward; after each temperature, round with the Hungarian
   assignment and score the rounded layout with the **true seeded KT-2
   cost** — the surrogate never enters the comparison;
4. stop at the preregistered true-cost evaluation budget.

The comparison protocol, seeds, and the baseline reference numbers live in
the design doc and in ``dynq_qubit_mapping.md`` §7.6/§8.5.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kuramoto_layout_cost import (
    CostWeights,
    DepthProvider,
    FloatArray,
    LayoutCost,
    kuramoto_layout_cost,
    routed_layout_depth,
)
from .kuramoto_layout_optimiser import _validate_search_inputs

RESEARCH_LABEL = (
    "RESEARCH (KT-4): relaxed-then-rounded layout search under a preregistered "
    "protocol; results are research observations, not a promoted capability"
)


@dataclass(frozen=True)
class SinkhornRelaxationConfig:
    """Configuration of the annealed Sinkhorn relaxation search.

    Parameters
    ----------
    tau_initial, tau_final
        Initial and final Sinkhorn temperatures (annealed geometrically).
    n_anneal_steps
        Number of temperatures in the schedule; each proposes one rounded
        layout for true-cost scoring.
    n_gradient_steps
        Surrogate gradient-descent steps per temperature.
    learning_rate
        Step size on the placement logits.
    n_sinkhorn_iterations
        Row/column normalisation sweeps per Sinkhorn projection.
    max_true_cost_evaluations
        Budget of distinct rounded layouts scored with the true cost;
        ``None`` allows one per anneal step. This is the preregistered
        budget-match knob against the KT-3 baseline.
    seed
        Seed for the logit initialisation.
    weights
        True-cost term weights; ``None`` selects the unit defaults.
    t, reps, order
        Evolution time, Trotter repetitions, and product-formula order
        forwarded to the true cost.
    """

    tau_initial: float = 1.0
    tau_final: float = 0.1
    n_anneal_steps: int = 8
    n_gradient_steps: int = 30
    learning_rate: float = 0.5
    n_sinkhorn_iterations: int = 50
    max_true_cost_evaluations: int | None = None
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
            If the temperatures are not positive and ordered, or any count
            or the learning rate is not positive, or the budget is not
            positive when set.
        """
        if not isfinite(self.tau_initial) or self.tau_initial <= 0.0:
            raise ValueError("tau_initial must be finite and positive")
        if not isfinite(self.tau_final) or self.tau_final <= 0.0:
            raise ValueError("tau_final must be finite and positive")
        if self.tau_final > self.tau_initial:
            raise ValueError("tau_final must not exceed tau_initial")
        if self.n_anneal_steps < 1:
            raise ValueError("n_anneal_steps must be a positive integer")
        if self.n_gradient_steps < 1:
            raise ValueError("n_gradient_steps must be a positive integer")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if self.n_sinkhorn_iterations < 1:
            raise ValueError("n_sinkhorn_iterations must be a positive integer")
        if self.max_true_cost_evaluations is not None and self.max_true_cost_evaluations < 1:
            raise ValueError("max_true_cost_evaluations must be positive when set")
        if not isfinite(self.t) or self.t <= 0.0:
            raise ValueError("t must be finite and positive")
        if self.reps < 1:
            raise ValueError("reps must be a positive integer")

    def temperatures(self) -> FloatArray:
        """Return the geometric annealing schedule from initial to final τ."""
        return np.geomspace(self.tau_initial, self.tau_final, self.n_anneal_steps).astype(
            np.float64
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the configuration."""
        return {
            "tau_initial": self.tau_initial,
            "tau_final": self.tau_final,
            "n_anneal_steps": self.n_anneal_steps,
            "n_gradient_steps": self.n_gradient_steps,
            "learning_rate": self.learning_rate,
            "n_sinkhorn_iterations": self.n_sinkhorn_iterations,
            "max_true_cost_evaluations": self.max_true_cost_evaluations,
            "seed": self.seed,
            "weights": None if self.weights is None else self.weights.to_dict(),
            "t": self.t,
            "reps": self.reps,
            "order": self.order,
        }


@dataclass(frozen=True)
class RelaxationSearchResult:
    """Outcome of the relaxed-then-rounded layout search (research result)."""

    best_layout: tuple[int, ...]
    best_cost: LayoutCost
    n_true_evaluations: int
    surrogate_trajectory: tuple[float, ...]
    research_label: str = RESEARCH_LABEL

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the search outcome."""
        return {
            "best_layout": list(self.best_layout),
            "best_cost": self.best_cost.to_dict(),
            "n_true_evaluations": self.n_true_evaluations,
            "surrogate_trajectory": list(self.surrogate_trajectory),
            "research_label": self.research_label,
        }


def sinkhorn_normalise(logits: FloatArray, n_iterations: int) -> FloatArray:
    """Project logits to a doubly-stochastic matrix in log space.

    Alternating row and column log-sum-exp normalisation (the Sinkhorn
    operator of Mena et al., arXiv:1802.08665).

    Parameters
    ----------
    logits
        Square real matrix.
    n_iterations
        Number of alternating normalisation sweeps.

    Returns
    -------
    numpy.ndarray
        A (numerically) doubly-stochastic matrix of the same shape.
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must be a square matrix")
    log_p = logits.astype(np.float64, copy=True)
    for _ in range(n_iterations):
        log_p = log_p - _logsumexp_rows(log_p)[:, None]
        log_p = log_p - _logsumexp_rows(log_p.T)[None, :]
    return cast(FloatArray, np.exp(log_p))


def _logsumexp_rows(matrix: FloatArray) -> FloatArray:
    """Return the row-wise log-sum-exp of ``matrix`` (stable)."""
    row_max = np.max(matrix, axis=1)
    return cast(
        FloatArray, row_max + np.log(np.sum(np.exp(matrix - row_max[:, None]), axis=1))
    )


def coupling_graph_distances(coupling_map: Any, physical_qubits: tuple[int, ...]) -> FloatArray:
    """Return pairwise shortest-path distances between candidate qubits.

    Breadth-first search over the undirected coupling graph restricted to
    the full device (paths may leave the candidate set), evaluated between
    every pair of candidates.

    Parameters
    ----------
    coupling_map
        A Qiskit ``CouplingMap`` (anything exposing ``get_edges()``) or an
        iterable of directed edge pairs.
    physical_qubits
        Candidate physical qubits.

    Returns
    -------
    numpy.ndarray
        ``(m, m)`` matrix of hop distances between candidates.

    Raises
    ------
    ValueError
        If any pair of candidates is disconnected in the coupling graph —
        the surrogate is undefined there (fail-closed).
    """
    edges = coupling_map.get_edges() if hasattr(coupling_map, "get_edges") else coupling_map
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))

    m = len(physical_qubits)
    distances = np.full((m, m), np.inf)
    index_of = {qubit: index for index, qubit in enumerate(physical_qubits)}
    for source_index, source in enumerate(physical_qubits):
        seen = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency.get(node, ()):
                if neighbour not in seen:
                    seen[neighbour] = seen[node] + 1
                    queue.append(neighbour)
        distances[source_index, source_index] = 0.0
        for target, hops in seen.items():
            if target in index_of:
                distances[source_index, index_of[target]] = float(hops)
    if not np.all(np.isfinite(distances)):
        raise ValueError("candidate qubits are disconnected in the coupling graph")
    return distances


def swap_distance_surrogate(P: FloatArray, K: FloatArray, distances: FloatArray) -> float:
    """Return the expected SWAP-distance load of a relaxed placement.

    ``S(P) = Σ_{i<j} K_ij · (P D Pᵀ)_{ij}`` — continuous in ``P``,
    correlating with the SWAP overhead routing must pay for distant strongly
    coupled pairs. Only the first ``n`` rows of ``P`` (the real logical
    qubits) contribute, because ``K`` is zero on the dummy padding.

    Parameters
    ----------
    P
        Doubly-stochastic placement matrix (``m × m``; rows ``n..m`` are
        dummy padding when ``n < m``).
    K
        Coupling matrix zero-padded to ``m × m``.
    distances
        Candidate-pair distance matrix ``D``.

    Returns
    -------
    float
        The surrogate value.
    """
    expected = P @ distances @ P.T
    return float(0.5 * np.sum(K * expected))


def _surrogate_gradient(P: FloatArray, K: FloatArray, distances: FloatArray) -> FloatArray:
    """Return ``∇_P S = K P D`` (``K``, ``D`` symmetric)."""
    return K @ P @ distances


def _rounded_layout(P: FloatArray, n: int, physical_qubits: tuple[int, ...]) -> tuple[int, ...]:
    """Round the relaxed placement with the Hungarian assignment."""
    rows, columns = linear_sum_assignment(-P)
    assignment = dict(zip(rows.tolist(), columns.tolist(), strict=True))
    return tuple(physical_qubits[assignment[row]] for row in range(n))


def relax_kuramoto_layout(
    K: FloatArray,
    omega: FloatArray,
    coupling_map: Any,
    physical_qubits: tuple[int, ...],
    *,
    mean_gate_fidelity: float,
    config: SinkhornRelaxationConfig | None = None,
    initial_layout: tuple[int, ...] | None = None,
    depth_provider: DepthProvider = routed_layout_depth,
) -> RelaxationSearchResult:
    """Run the annealed Sinkhorn relaxation and score rounded layouts truly.

    Parameters
    ----------
    K, omega
        Coupling matrix and frequency vector; ``K`` fixes the logical count.
    coupling_map
        Hardware connectivity (distances for the surrogate; forwarded to the
        true cost's ``depth_provider``).
    physical_qubits
        Candidate physical qubits (distinct, non-negative, at least ``n``).
    mean_gate_fidelity
        DynQ selected-region mean gate fidelity in ``[0, 1]``.
    config
        Relaxation configuration; ``None`` selects the defaults.
    initial_layout
        Optional warm-start layout: its placements receive a positive logit
        bias so the relaxation starts near the (e.g. DynQ) seed.
    depth_provider
        Depth callable for the **true** cost; defaults to
        :func:`~scpn_quantum_control.hardware.kuramoto_layout_cost.routed_layout_depth`.

    Returns
    -------
    RelaxationSearchResult
        The best rounded layout by **true** cost, the number of distinct
        true-cost evaluations spent, and the surrogate trajectory (reported
        for diagnostics only — never used for the comparison).

    Raises
    ------
    ValueError
        If the search space, seed layout, or cost inputs are malformed, or
        the candidates are disconnected in the coupling graph.
    """
    config = config or SinkhornRelaxationConfig()
    n = _validate_search_inputs(K, physical_qubits, initial_layout)
    m = len(physical_qubits)

    distances = coupling_graph_distances(coupling_map, physical_qubits)
    K_padded = np.zeros((m, m))
    K_padded[:n, :n] = K

    rng = np.random.default_rng(config.seed)
    logits = 0.01 * rng.standard_normal((m, m))
    if initial_layout is not None:
        index_of = {qubit: index for index, qubit in enumerate(physical_qubits)}
        for row, qubit in enumerate(initial_layout):
            logits[row, index_of[qubit]] += 1.0

    cache: dict[tuple[int, ...], LayoutCost] = {}

    def true_cost(layout: tuple[int, ...]) -> LayoutCost:
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

    budget = config.max_true_cost_evaluations or config.n_anneal_steps
    best_layout: tuple[int, ...] | None = None
    best_cost: LayoutCost | None = None
    surrogate_trajectory: list[float] = []

    for tau in config.temperatures():
        P = sinkhorn_normalise(logits / tau, config.n_sinkhorn_iterations)
        for _ in range(config.n_gradient_steps):
            # Straight-through estimator (Jang et al., arXiv:1611.01144):
            # the surrogate gradient at P is applied directly to the logits.
            logits = logits - config.learning_rate * _surrogate_gradient(P, K_padded, distances)
            P = sinkhorn_normalise(logits / tau, config.n_sinkhorn_iterations)
        surrogate_trajectory.append(swap_distance_surrogate(P, K_padded, distances))

        candidate = _rounded_layout(P, n, physical_qubits)
        if candidate not in cache and len(cache) >= budget:
            continue
        cost = true_cost(candidate)
        if best_cost is None or cost.total < best_cost.total:
            best_layout, best_cost = candidate, cost

    # The first anneal step always fits the (>= 1) budget, so best is set.
    return RelaxationSearchResult(
        best_layout=cast("tuple[int, ...]", best_layout),
        best_cost=cast(LayoutCost, best_cost),
        n_true_evaluations=len(cache),
        surrogate_trajectory=tuple(surrogate_trajectory),
    )
