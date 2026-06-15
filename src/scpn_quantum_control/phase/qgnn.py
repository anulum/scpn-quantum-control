# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded quantum graph neural network over K_nm graphs
"""A local quantum graph neural network that maps K_nm graphs to circuit outputs.

A classical message-passing stack turns a K_nm coupling graph (nodes carry
natural frequencies, edges carry couplings) into the rotation angles of a
registered Phase-QNode circuit whose entangling structure follows the same
graph. The circuit expectation is the model output. Trainable weights live in
the classical message passing; their gradient is the exact chain of the analytic
parameter-shift gradient (output with respect to circuit angles) and the
analytic message-passing backward pass (angles with respect to weights).

This is a bounded local model: small statevector circuits, synthetic
Kuramoto-XY regression targets, and convergence reported as observed local loss
decrease only. It makes no claim of arbitrary-graph, arbitrary-depth, provider,
hardware, or production QGNN convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    SparsePauliHamiltonian,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

# Couplings below this magnitude do not place an entangling edge.
_EDGE_THRESHOLD = 1e-9
# Statevector ceiling for the per-graph circuit.
MAX_QGNN_NODES = 12


@dataclass(frozen=True)
class KnmGraph:
    """A K_nm coupling graph: symmetric couplings plus per-node frequencies."""

    coupling: np.ndarray
    node_frequencies: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.coupling.shape[0])


@dataclass(frozen=True)
class QGNNConfig:
    """Architecture of the message-passing stack and the readout circuit."""

    hidden_dim: int = 4
    n_message_layers: int = 1
    angles_per_node: int = 2
    edge_threshold: float = 1e-3

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if self.n_message_layers < 1:
            raise ValueError("n_message_layers must be a positive integer")
        if self.angles_per_node not in (1, 2):
            raise ValueError("angles_per_node must be 1 or 2")
        if self.edge_threshold < 0.0:
            raise ValueError("edge_threshold must be non-negative")


@dataclass(frozen=True)
class QGNNTrainingResult:
    """Trained weights and the observed local loss trajectory."""

    parameters: np.ndarray
    loss_history: np.ndarray
    final_loss: float
    provenance: dict[str, Any] = field(default_factory=dict)


def validate_graph(graph: KnmGraph) -> KnmGraph:
    """Validate and symmetrise a :class:`KnmGraph`, fail-closed on bad input."""
    coupling = np.asarray(graph.coupling, dtype=np.float64)
    frequencies = np.asarray(graph.node_frequencies, dtype=np.float64)
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("coupling must be a square 2-D matrix")
    n = coupling.shape[0]
    if not 1 <= n <= MAX_QGNN_NODES:
        raise ValueError(f"graph must have between 1 and {MAX_QGNN_NODES} nodes")
    if frequencies.shape != (n,):
        raise ValueError(f"node_frequencies must have length {n}")
    if not np.all(np.isfinite(coupling)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("graph must be finite (no NaN/Inf)")
    return KnmGraph(coupling=0.5 * (coupling + coupling.T), node_frequencies=frequencies)


def _node_features(graph: KnmGraph) -> np.ndarray:
    """Per-node input features: natural frequency and weighted degree."""
    degree = np.sum(np.abs(graph.coupling), axis=1)
    return np.stack([graph.node_frequencies, degree], axis=1)


def _normalised_adjacency(coupling: np.ndarray) -> np.ndarray:
    degree = np.sum(np.abs(coupling), axis=1, keepdims=True)
    degree[degree < _EDGE_THRESHOLD] = 1.0
    adjacency: np.ndarray = coupling / degree
    return adjacency


_INPUT_FEATURES = 2


def _layer_shapes(config: QGNNConfig) -> list[tuple[int, int]]:
    dims = [_INPUT_FEATURES] + [config.hidden_dim] * config.n_message_layers
    return [(dims[i], dims[i + 1]) for i in range(config.n_message_layers)]


def _parameter_layout(config: QGNNConfig) -> list[tuple[str, tuple[int, ...]]]:
    layout: list[tuple[str, tuple[int, ...]]] = []
    for layer, (d_in, d_out) in enumerate(_layer_shapes(config)):
        layout.append((f"w_self_{layer}", (d_in, d_out)))
        layout.append((f"w_msg_{layer}", (d_in, d_out)))
        layout.append((f"bias_{layer}", (d_out,)))
    layout.append(("w_out", (config.hidden_dim, config.angles_per_node)))
    layout.append(("c_out", (config.angles_per_node,)))
    return layout


def parameter_count(config: QGNNConfig) -> int:
    """Number of trainable scalars for ``config``."""
    return sum(int(np.prod(shape)) for _name, shape in _parameter_layout(config))


def initialise_parameters(config: QGNNConfig, seed: int | None = None) -> np.ndarray:
    """Draw a small-variance initial weight vector."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(parameter_count(config)) * 0.1


def _unflatten(config: QGNNConfig, parameters: np.ndarray) -> dict[str, np.ndarray]:
    parameters = np.asarray(parameters, dtype=np.float64)
    expected = parameter_count(config)
    if parameters.shape != (expected,):
        raise ValueError(f"parameters must be a length-{expected} vector")
    blocks: dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in _parameter_layout(config):
        size = int(np.prod(shape))
        blocks[name] = parameters[offset : offset + size].reshape(shape)
        offset += size
    return blocks


def _message_passing_forward(
    config: QGNNConfig,
    weights: dict[str, np.ndarray],
    features: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Run the message-passing stack, returning the angle vector and a tape."""
    activation = features
    tape: list[dict[str, np.ndarray]] = []
    for layer in range(config.n_message_layers):
        aggregate = adjacency @ activation
        pre = (
            activation @ weights[f"w_self_{layer}"]
            + aggregate @ weights[f"w_msg_{layer}"]
            + weights[f"bias_{layer}"]
        )
        post = np.tanh(pre)
        tape.append({"input": activation, "aggregate": aggregate, "pre": pre, "post": post})
        activation = post
    readout_pre = activation @ weights["w_out"] + weights["c_out"]
    angles_per_node = np.pi * np.tanh(readout_pre)
    tape.append({"readout_input": activation, "readout_pre": readout_pre})
    return angles_per_node.reshape(-1), tape


def _readout_circuit(config: QGNNConfig, graph: KnmGraph, angles: np.ndarray) -> PhaseQNodeCircuit:
    n = graph.n_nodes
    operations: list[PhaseQNodeOperation] = []
    index = 0
    for node in range(n):
        operations.append(PhaseQNodeOperation("ry", (node,), parameter_index=index))
        index += 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(graph.coupling[i, j]) >= config.edge_threshold:
                operations.append(PhaseQNodeOperation("cz", (i, j)))
    if config.angles_per_node == 2:
        for node in range(n):
            operations.append(PhaseQNodeOperation("ry", (node,), parameter_index=index))
            index += 1
    observable: PauliTerm | SparsePauliHamiltonian
    if n == 1:
        observable = PauliTerm(1.0, ((0, "z"),))
    else:
        observable = SparsePauliHamiltonian(
            tuple(PauliTerm(1.0 / n, ((node, "z"),)) for node in range(n))
        )
    return PhaseQNodeCircuit(n_qubits=n, operations=tuple(operations), observable=observable)


def _circuit_angles(config: QGNNConfig, graph: KnmGraph, flat_angles: np.ndarray) -> np.ndarray:
    """Reorder per-node angle blocks into circuit-operation order."""
    n = graph.n_nodes
    per_node = flat_angles.reshape(n, config.angles_per_node)
    if config.angles_per_node == 1:
        return per_node[:, 0]
    return np.concatenate([per_node[:, 0], per_node[:, 1]])


def predict(config: QGNNConfig, parameters: np.ndarray, graph: KnmGraph) -> float:
    """Forward pass: graph + weights -> circuit expectation."""
    graph = validate_graph(graph)
    weights = _unflatten(config, parameters)
    features = _node_features(graph)
    adjacency = _normalised_adjacency(graph.coupling)
    flat_angles, _tape = _message_passing_forward(config, weights, features, adjacency)
    circuit = _readout_circuit(config, graph, flat_angles)
    angles = _circuit_angles(config, graph, flat_angles)
    return float(execute_phase_qnode_circuit(circuit, angles).value)


def predict_and_gradient(
    config: QGNNConfig, parameters: np.ndarray, graph: KnmGraph
) -> tuple[float, np.ndarray]:
    """Return the prediction and its exact gradient with respect to the weights.

    The gradient chains the analytic parameter-shift gradient of the circuit
    output with respect to the rotation angles and the analytic backward pass of
    the message-passing stack with respect to the weights.
    """
    graph = validate_graph(graph)
    weights = _unflatten(config, parameters)
    features = _node_features(graph)
    adjacency = _normalised_adjacency(graph.coupling)
    flat_angles, tape = _message_passing_forward(config, weights, features, adjacency)
    circuit = _readout_circuit(config, graph, flat_angles)
    angles = _circuit_angles(config, graph, flat_angles)
    result = parameter_shift_phase_qnode_gradient(circuit, angles)
    value = float(result.value)

    # d output / d (circuit angle) -> d output / d (per-node angle block)
    n = graph.n_nodes
    grad_circuit = np.asarray(result.gradient, dtype=np.float64)
    grad_angles = np.empty((n, config.angles_per_node), dtype=np.float64)
    grad_angles[:, 0] = grad_circuit[:n]
    if config.angles_per_node == 2:
        grad_angles[:, 1] = grad_circuit[n : 2 * n]

    weight_grads = _message_passing_backward(config, weights, tape, grad_angles, adjacency)
    flat_grad = np.concatenate(
        [weight_grads[name].reshape(-1) for name, _shape in _parameter_layout(config)]
    )
    return value, flat_grad


def _message_passing_backward(
    config: QGNNConfig,
    weights: dict[str, np.ndarray],
    tape: list[dict[str, np.ndarray]],
    grad_angles: np.ndarray,
    adjacency: np.ndarray,
) -> dict[str, np.ndarray]:
    readout = tape[-1]
    grads: dict[str, np.ndarray] = {}
    # Readout: angles = pi * tanh(readout_pre); d angle / d readout_pre.
    grad_pre = grad_angles * np.pi * (1.0 - np.tanh(readout["readout_pre"]) ** 2)
    readout_input = readout["readout_input"]
    grads["w_out"] = readout_input.T @ grad_pre
    grads["c_out"] = np.sum(grad_pre, axis=0)
    grad_activation = grad_pre @ weights["w_out"].T

    for layer in reversed(range(config.n_message_layers)):
        record = tape[layer]
        grad_pre_layer = grad_activation * (1.0 - record["post"] ** 2)
        grads[f"w_self_{layer}"] = record["input"].T @ grad_pre_layer
        grads[f"w_msg_{layer}"] = record["aggregate"].T @ grad_pre_layer
        grads[f"bias_{layer}"] = np.sum(grad_pre_layer, axis=0)
        grad_aggregate = grad_pre_layer @ weights[f"w_msg_{layer}"].T
        # aggregate = A @ activation, so activation gets A^T @ grad_aggregate.
        grad_activation = (
            grad_pre_layer @ weights[f"w_self_{layer}"].T + adjacency.T @ grad_aggregate
        )
    return grads


def synthetic_kuramoto_target(graph: KnmGraph) -> float:
    """A deterministic Kuramoto-XY regression target in ``[-1, 1]``.

    Maps the phase-locked Kuramoto order parameter of the graph (evolved from a
    fixed synchronised start) onto ``[-1, 1]`` so a unit-norm circuit observable
    can represent it.
    """
    graph = validate_graph(graph)
    n = graph.n_nodes
    theta = np.zeros(n)
    dt, steps = 0.05, 40
    for _ in range(steps):
        diff = theta[None, :] - theta[:, None]
        theta = theta + dt * (
            graph.node_frequencies + np.sum(graph.coupling * np.sin(diff), axis=1)
        )
    order = float(np.abs(np.mean(np.exp(1j * theta))))
    return 2.0 * order - 1.0


def train(
    config: QGNNConfig,
    parameters: np.ndarray,
    graphs: tuple[KnmGraph, ...],
    targets: np.ndarray,
    *,
    learning_rate: float = 0.1,
    epochs: int = 50,
) -> QGNNTrainingResult:
    """Fit the message-passing weights to ``targets`` by gradient descent.

    Reports observed local loss decrease only; it does not claim convergence for
    arbitrary graphs, depths, or unseen distributions.
    """
    if len(graphs) == 0:
        raise ValueError("at least one training graph is required")
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (len(graphs),):
        raise ValueError("targets must have one entry per graph")
    if learning_rate <= 0.0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be a positive finite value")
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")

    params = np.array(parameters, dtype=np.float64)
    loss_history = np.empty(epochs, dtype=np.float64)
    for epoch in range(epochs):
        grad_total = np.zeros_like(params)
        loss = 0.0
        for graph, target in zip(graphs, targets, strict=True):
            value, grad = predict_and_gradient(config, params, graph)
            residual = value - float(target)
            loss += residual**2
            grad_total += 2.0 * residual * grad
        loss_history[epoch] = loss / len(graphs)
        params = params - learning_rate * grad_total / len(graphs)
    return QGNNTrainingResult(
        parameters=params,
        loss_history=loss_history,
        final_loss=float(loss_history[-1]),
        provenance={
            "hidden_dim": config.hidden_dim,
            "n_message_layers": config.n_message_layers,
            "angles_per_node": config.angles_per_node,
            "n_graphs": len(graphs),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "claim_boundary": (
                "bounded local QGNN; observed loss decrease on the training "
                "graphs only; no arbitrary-graph, depth, provider, hardware, or "
                "production convergence claim"
            ),
        },
    )
