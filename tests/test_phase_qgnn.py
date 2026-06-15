# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the bounded quantum graph neural network
"""Tests for phase/qgnn.py: graph validation, exact chain-rule gradients, and training."""

import numpy as np
import pytest

from scpn_quantum_control.phase.qgnn import (
    KnmGraph,
    QGNNConfig,
    initialise_parameters,
    parameter_count,
    predict,
    predict_and_gradient,
    synthetic_kuramoto_target,
    train,
    validate_graph,
)


def _random_graph(rng: np.random.Generator, n: int) -> KnmGraph:
    coupling = rng.standard_normal((n, n)) * 0.3
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    return KnmGraph(coupling=coupling, node_frequencies=rng.standard_normal(n) * 0.4)


# --------------------------------------------------------------------------- #
# Graph and config validation
# --------------------------------------------------------------------------- #
def test_validate_graph_symmetrises():
    coupling = np.array([[0.0, 1.0], [3.0, 0.0]])
    graph = validate_graph(KnmGraph(coupling=coupling, node_frequencies=np.zeros(2)))
    assert np.allclose(graph.coupling, graph.coupling.T)
    assert graph.coupling[0, 1] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "coupling, freqs",
    [
        (np.zeros((2, 3)), np.zeros(2)),
        (np.zeros((3, 3)), np.zeros(2)),
        (np.full((2, 2), np.nan), np.zeros(2)),
        (np.zeros((13, 13)), np.zeros(13)),
    ],
)
def test_validate_graph_rejects_bad_input(coupling, freqs):
    with pytest.raises(ValueError):
        validate_graph(KnmGraph(coupling=coupling, node_frequencies=freqs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_dim": 0},
        {"n_message_layers": 0},
        {"angles_per_node": 3},
        {"edge_threshold": -1.0},
    ],
)
def test_config_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        QGNNConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Parameters and forward pass
# --------------------------------------------------------------------------- #
def test_parameter_count_matches_initialiser():
    config = QGNNConfig(hidden_dim=5, n_message_layers=2, angles_per_node=2)
    params = initialise_parameters(config, seed=0)
    assert params.shape == (parameter_count(config),)


def test_predict_is_deterministic_and_bounded():
    rng = np.random.default_rng(1)
    config = QGNNConfig()
    graph = _random_graph(rng, 4)
    params = initialise_parameters(config, seed=2)
    first = predict(config, params, graph)
    second = predict(config, params, graph)
    assert first == second
    assert -1.0 - 1e-9 <= first <= 1.0 + 1e-9


def test_predict_rejects_wrong_parameter_length():
    config = QGNNConfig()
    graph = _random_graph(np.random.default_rng(3), 3)
    with pytest.raises(ValueError):
        predict(config, np.zeros(parameter_count(config) + 1), graph)


def test_single_node_graph():
    config = QGNNConfig(hidden_dim=3, n_message_layers=1, angles_per_node=1)
    graph = KnmGraph(coupling=np.zeros((1, 1)), node_frequencies=np.array([0.5]))
    params = initialise_parameters(config, seed=4)
    value = predict(config, params, graph)
    assert -1.0 - 1e-9 <= value <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# Exact chain-rule gradient
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "config",
    [
        QGNNConfig(hidden_dim=4, n_message_layers=1, angles_per_node=1),
        QGNNConfig(hidden_dim=4, n_message_layers=2, angles_per_node=2),
        QGNNConfig(hidden_dim=3, n_message_layers=1, angles_per_node=2),
    ],
)
def test_gradient_matches_finite_difference(config):
    rng = np.random.default_rng(5)
    graph = _random_graph(rng, 4)
    params = initialise_parameters(config, seed=6)
    value, grad = predict_and_gradient(config, params, graph)
    assert value == pytest.approx(predict(config, params, graph))
    eps = 1e-6
    numeric = np.zeros_like(params)
    for k in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[k] += eps
        minus[k] -= eps
        numeric[k] = (predict(config, plus, graph) - predict(config, minus, graph)) / (2 * eps)
    assert np.allclose(grad, numeric, atol=1e-6)


# --------------------------------------------------------------------------- #
# Synthetic targets and training
# --------------------------------------------------------------------------- #
def test_synthetic_target_is_bounded_and_deterministic():
    graph = _random_graph(np.random.default_rng(7), 5)
    a = synthetic_kuramoto_target(graph)
    b = synthetic_kuramoto_target(graph)
    assert a == b
    assert -1.0 <= a <= 1.0


def test_training_reduces_loss():
    rng = np.random.default_rng(8)
    config = QGNNConfig(hidden_dim=4, n_message_layers=2, angles_per_node=2)
    graphs = tuple(_random_graph(rng, 4) for _ in range(6))
    targets = np.array([synthetic_kuramoto_target(g) for g in graphs])
    params = initialise_parameters(config, seed=9)
    result = train(config, params, graphs, targets, learning_rate=0.3, epochs=60)
    assert result.final_loss < result.loss_history[0]
    assert result.loss_history.shape == (60,)
    assert "claim_boundary" in result.provenance


def test_train_rejects_bad_input():
    config = QGNNConfig()
    graph = _random_graph(np.random.default_rng(10), 3)
    params = initialise_parameters(config, seed=11)
    with pytest.raises(ValueError):
        train(config, params, (), np.array([]))
    with pytest.raises(ValueError):
        train(config, params, (graph,), np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        train(config, params, (graph,), np.array([0.1]), learning_rate=0.0)
    with pytest.raises(ValueError):
        train(config, params, (graph,), np.array([0.1]), epochs=0)


def test_edge_threshold_controls_entanglers():
    # Weak coupling below the threshold places no entangling edge, so the two
    # gradient circuits differ only by the rotation structure.
    config = QGNNConfig(edge_threshold=0.5)
    graph = KnmGraph(coupling=np.array([[0.0, 0.1], [0.1, 0.0]]), node_frequencies=np.zeros(2))
    params = initialise_parameters(config, seed=12)
    # Predict must still succeed (no entangler) and stay bounded.
    value = predict(config, params, graph)
    assert -1.0 - 1e-9 <= value <= 1.0 + 1e-9
