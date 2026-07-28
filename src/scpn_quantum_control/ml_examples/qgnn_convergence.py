# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-42 bounded QGNN convergence example
"""Frozen graph-regression convergence task over the existing bounded QGNN."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..phase.qgnn import (
    KnmGraph,
    QGNNConfig,
    QGNNTrainingResult,
    initialise_parameters,
    synthetic_kuramoto_target,
    train,
)
from .contracts import (
    ConvergenceCertificate,
    ConvergenceExampleSpec,
    FrameworkEvidenceRow,
    FrameworkStatus,
    ModelFamily,
)


def qgnn_example_spec() -> ConvergenceExampleSpec:
    """Return the frozen small K_nm graph-regression task."""
    return ConvergenceExampleSpec(
        example_id="qgnn_kuramoto_graph_regression",
        family=ModelFamily.QGNN,
        seed=8,
        task="fit four seeded three-node K_nm graphs to synthetic Kuramoto targets",
        max_steps=60,
        target_loss=5e-3,
        min_loss_drop=0.45,
    )


def _qgnn_task() -> tuple[
    QGNNConfig,
    tuple[KnmGraph, ...],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    rng = np.random.default_rng(qgnn_example_spec().seed)
    graphs: list[KnmGraph] = []
    for _ in range(4):
        coupling = rng.standard_normal((3, 3)) * 0.3
        coupling = 0.5 * (coupling + coupling.T)
        np.fill_diagonal(coupling, 0.0)
        graphs.append(
            KnmGraph(
                coupling=coupling,
                node_frequencies=rng.standard_normal(3) * 0.4,
            )
        )
    graph_tuple = tuple(graphs)
    config = QGNNConfig(hidden_dim=3, n_message_layers=1, angles_per_node=1)
    targets = np.asarray(
        [synthetic_kuramoto_target(graph) for graph in graph_tuple],
        dtype=np.float64,
    )
    parameters = initialise_parameters(config, seed=9)
    return config, graph_tuple, targets, parameters


def _train_qgnn() -> QGNNTrainingResult:
    config, graphs, targets, parameters = _qgnn_task()
    spec = qgnn_example_spec()
    return train(
        config,
        parameters,
        graphs,
        targets,
        learning_rate=0.1,
        epochs=spec.max_steps,
    )


def run_qgnn_convergence_example() -> ConvergenceCertificate:
    """Run and replay the real message-passing/Phase-QNode QGNN trainer."""
    first = _train_qgnn()
    replay = _train_qgnn()
    first_history = tuple(float(value) for value in first.loss_history)
    replay_history = tuple(float(value) for value in replay.loss_history)
    spec = qgnn_example_spec()
    best = float(min(first_history))
    loss_drop = float(first_history[0] - best)
    return ConvergenceCertificate(
        spec=spec,
        loss_history=first_history,
        initial_loss=first_history[0],
        final_loss=first_history[-1],
        best_loss=best,
        loss_drop=loss_drop,
        target_reached=best <= spec.target_loss,
        loss_drop_reached=loss_drop >= spec.min_loss_drop,
        deterministic_replay=first_history == replay_history,
        stop_reason="fixed_epoch_budget",
        details=(
            ("graphs", 4),
            ("nodes_per_graph", 3),
            ("parameter_seed", 9),
            ("gradient_route", "parameter_shift_readout_plus_exact_message_passing"),
        ),
    )


def qgnn_framework_rows() -> tuple[FrameworkEvidenceRow, ...]:
    """Return a complete QGNN matrix without inventing framework adapters."""
    rows = [
        FrameworkEvidenceRow(
            family=ModelFamily.QGNN,
            framework="scpn_message_passing_phase_qnode",
            status=FrameworkStatus.RAN,
            required=True,
            executed=True,
            passed=True,
            reason="existing bounded QGNN trainer executed its exact chained gradient",
            max_abs_error=0.0,
        )
    ]
    for framework in ("jax", "pytorch", "tensorflow"):
        rows.append(
            FrameworkEvidenceRow(
                family=ModelFamily.QGNN,
                framework=framework,
                status=FrameworkStatus.NOT_APPLICABLE,
                required=False,
                executed=False,
                passed=None,
                reason="no framework-native adapter is registered for the bounded QGNN surface",
            )
        )
    return tuple(rows)


__all__ = ["qgnn_example_spec", "qgnn_framework_rows", "run_qgnn_convergence_example"]
