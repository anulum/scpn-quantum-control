# Bounded Quantum Graph Neural Network

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.phase.qgnn` is a local quantum graph neural network that
maps a K_nm coupling graph to a registered Phase-QNode circuit output. A
classical message-passing stack turns the graph (nodes carry natural
frequencies and weighted degree, edges carry couplings) into the rotation angles
of a circuit whose entangling structure follows the same graph; the circuit
expectation is the model output.

This is a **bounded local model**: small statevector circuits (≤ 12 nodes),
synthetic Kuramoto-XY regression targets, and convergence reported only as the
observed training-loss decrease. It makes no claim of arbitrary-graph,
arbitrary-depth, provider, hardware, or production QGNN convergence.

## Architecture

1. **Node features** — natural frequency and weighted degree per node.
2. **Message passing** — `L` layers of
   `h' = tanh(h W_self + (Â h) W_msg + b)`, where `Â` is the degree-normalised
   coupling adjacency. The classical weights are the trainable parameters.
3. **Readout** — `angle = pi · tanh(h W_out + c_out)`, one or two angles per
   node.
4. **Circuit** — an `RY` layer per node, `CZ` entanglers on every coupled pair
   above `edge_threshold` (the physics-informed K_nm topology), an optional
   second `RY` layer, and a normalised `Z` observable.

`validate_graph` accepts one to twelve nodes, requires finite square coupling
and matching frequency arrays, and replaces the coupling with
`(K + K.T) / 2`. `QGNNConfig` requires positive hidden/layer dimensions, one or
two angles per node, and a non-negative edge threshold. `parameter_count`
sums both message matrices and the bias for each layer plus the readout matrix
and bias; `initialise_parameters(config, seed)` draws that many independent
normal weights with standard deviation `0.1`.

```python
from scpn_quantum_control.phase import qgnn

config = qgnn.QGNNConfig(hidden_dim=4, n_message_layers=2, angles_per_node=2)
graph = qgnn.KnmGraph(coupling=K, node_frequencies=omega)
params = qgnn.initialise_parameters(config, seed=0)
value = qgnn.predict(config, params, graph)
```

## Exact gradients

`predict_and_gradient` returns the model output and its exact gradient with
respect to the classical weights. The gradient chains the analytic
parameter-shift gradient of the circuit output with respect to the rotation
angles with the analytic backward pass of the message-passing stack with respect
to the weights. The chain is validated against finite differences to ~1e-9 in
`tests/test_phase_qgnn.py`.

## Training against synthetic Kuramoto-XY targets

`synthetic_kuramoto_target` maps the phase-locked Kuramoto order parameter of a
graph onto `[-1, 1]`, matching the range of the normalised circuit observable.
`train` fits the weights by gradient descent and returns the loss trajectory and
a claim-boundary record.

The synthetic label starts every node at phase zero, applies 40 explicit-Euler
steps of size `0.05`, and returns
`2 * abs(mean(exp(1j * theta))) - 1`. It is a deterministic bounded training
label, not a high-accuracy Kuramoto integrator or benchmark result. Training
records the mean-squared loss before each full-batch update; `final_loss` is the
last recorded value rather than an extra post-update evaluation.

```python
targets = np.array([qgnn.synthetic_kuramoto_target(g) for g in graphs])
result = qgnn.train(config, params, graphs, targets, learning_rate=0.3, epochs=60)
result.final_loss        # observed local loss on the training graphs only
```

## Claim boundary

The evidence is bounded local regression on the training graphs. The model does
not claim generalisation to unseen graph distributions, arbitrary node counts or
circuit depths, provider or hardware execution, or production convergence.
Larger graphs are refused fail-closed above the statevector ceiling.
