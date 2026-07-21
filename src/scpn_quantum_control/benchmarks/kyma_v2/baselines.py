# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2.1 stronger baselines (capacity + relational)
"""Stronger non-oscillator baselines for the v2 task (KYMA v2.1 #2).

* **Deep MLP (capacity control).** A two-hidden-layer softmax network. Combined
  with the over-parameterised single-layer MLP (via :mod:`.models` with a large
  hidden width), this tests whether the substrate's advantage is merely capacity.
* **Message-passing GNN (inductive-bias control).** A small graph network over the
  oscillator graph whose adjacency is **code-conditioned** — it is told which
  oscillators are in the active pairs, exactly as the substrate is — but it
  aggregates with *learned message passing*, not oscillator dynamics. If the GNN
  still loses to the substrate, the advantage is specific to the dynamics, not to
  a generic relational inductive bias; if it matches, the claim narrows to "any
  relational bias" (reported honestly).
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .coupling import base_coupling_matrix, partners_for
from .models import _BASIS, _adam_descent, _mlp_features, _mlp_in_dim
from .task import N_PAIRS, N_RELATIONS, READOUT_OSCILLATOR, ProbeConfigV2, TrialBatchV2

_Params = dict[str, jax.Array]


def _softmax_cross_entropy(logits: jax.Array, labels: jax.Array, n_bins: int) -> jax.Array:
    """Mean softmax cross-entropy to integer class labels."""
    logp = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
    onehot = jax.nn.one_hot(labels, n_bins)
    return -jnp.mean(jnp.sum(onehot * logp, axis=1))


# --------------------------------------------------------------------------- #
# Deep MLP (two hidden layers) — capacity / depth control                     #
# --------------------------------------------------------------------------- #
def deep_mlp_param_count(hidden: int, n_bins: int) -> int:
    """Parameter count of the two-hidden-layer MLP."""
    in_dim = _mlp_in_dim()
    return in_dim * hidden + hidden + hidden * hidden + hidden + hidden * n_bins + n_bins


def deep_mlp_init(seed: int, hidden: int, n_bins: int) -> _Params:
    """Initialise the two-hidden-layer MLP (Glorot-ish)."""
    rng = np.random.default_rng(seed + 104723)
    in_dim = _mlp_in_dim()
    return {
        "w1": jnp.asarray(rng.normal(0, np.sqrt(1.0 / in_dim), size=(in_dim, hidden))),
        "b1": jnp.zeros(hidden),
        "w2": jnp.asarray(rng.normal(0, np.sqrt(1.0 / hidden), size=(hidden, hidden))),
        "b2": jnp.zeros(hidden),
        "w3": jnp.asarray(rng.normal(0, np.sqrt(1.0 / hidden), size=(hidden, n_bins))),
        "b3": jnp.zeros(n_bins),
    }


def deep_mlp_logits(params: _Params, feats: jax.Array) -> jax.Array:
    """Class logits from the two-hidden-layer MLP."""
    h1 = jnp.tanh(feats @ params["w1"] + params["b1"])
    h2 = jnp.tanh(h1 @ params["w2"] + params["b2"])
    return h2 @ params["w3"] + params["b3"]


def train_deep_mlp(
    batch: TrialBatchV2,
    teacher_labels: NDArray[np.int64],
    hidden: int,
    config: ProbeConfigV2,
    seed: int,
    epochs: int = 2000,
    lr: float = 0.02,
) -> _Params:
    """Train the two-hidden-layer MLP on the training trials' teacher labels."""
    tr = ~batch.is_test
    feats = _mlp_features(batch.theta0[tr], batch.code[tr])
    labels = jnp.asarray(np.asarray(teacher_labels)[tr])
    n_bins = config.n_bins
    params = deep_mlp_init(seed, hidden, n_bins)

    @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
    def optimise() -> _Params:
        return _adam_descent(
            params,
            lambda p: _softmax_cross_entropy(deep_mlp_logits(p, feats), labels, n_bins),
            lr=lr,
            epochs=epochs,
        )

    return cast(_Params, optimise())


def deep_mlp_predict(params: _Params, batch_slice: TrialBatchV2) -> NDArray[np.int64]:
    """Predicted class per trial = argmax of the deep-MLP logits."""
    feats = _mlp_features(batch_slice.theta0, batch_slice.code)
    logits = np.asarray(deep_mlp_logits(params, feats))
    pred: NDArray[np.int64] = np.argmax(logits, axis=1).astype(np.int64)
    return pred


# --------------------------------------------------------------------------- #
# Message-passing GNN — relational inductive bias, no oscillator dynamics      #
# --------------------------------------------------------------------------- #
def gnn_param_count(hidden: int, n_bins: int) -> int:
    """Trainable-parameter count of the code-conditioned GNN (matches gnn_init)."""
    edge_gates = N_RELATIONS * N_PAIRS * int(_BASIS.shape[1])  # code-conditioned adjacency
    layer1 = 2 * hidden + 2 * hidden + hidden  # w_self1(2,H) + w_msg1(2,H) + b1(H)
    layer2 = hidden * hidden + hidden * hidden + hidden  # w_self2 + w_msg2 + b2
    head = hidden * n_bins + n_bins  # w_out + b_out
    return edge_gates + layer1 + layer2 + head


def gnn_init(seed: int, hidden: int, n_bins: int) -> _Params:
    """Initialise the GNN (edge gates + two message-passing layers + a readout head)."""
    rng = np.random.default_rng(seed + 271828)
    n_edge = _BASIS.shape[1]

    def glorot(shape: tuple[int, ...]) -> jax.Array:
        return jnp.asarray(rng.normal(0, np.sqrt(1.0 / shape[0]), size=shape))

    return {
        "edge_gates": jnp.asarray(rng.normal(0, 0.2, size=(N_RELATIONS, N_PAIRS, n_edge))),
        "w_self1": glorot((2, hidden)),
        "w_msg1": glorot((2, hidden)),
        "b1": jnp.zeros(hidden),
        "w_self2": glorot((hidden, hidden)),
        "w_msg2": glorot((hidden, hidden)),
        "b2": jnp.zeros(hidden),
        "w_out": glorot((hidden, n_bins)),
        "b_out": jnp.zeros(n_bins),
    }


def _gnn_adjacency(params: _Params, code: jax.Array, base: jax.Array) -> jax.Array:
    """Per-trial adjacency = fixed base coupling + code-gated learned edges."""
    gates = jnp.einsum("rpe,peij->rpij", params["edge_gates"], jnp.asarray(_BASIS))
    learned = jnp.einsum("nrp,rpij->nij", code, gates)  # (n, N, N)
    return base[None, :, :] + learned


def gnn_logits(params: _Params, theta0: jax.Array, code: jax.Array, base: jax.Array) -> jax.Array:
    """Two message-passing layers then read the readout node → class logits."""
    adj = _gnn_adjacency(params, code, base)  # (n, N, N)
    x = jnp.stack([jnp.sin(theta0), jnp.cos(theta0)], axis=-1)  # (n, N, 2)

    def layer(h: jax.Array, w_self: jax.Array, w_msg: jax.Array, b: jax.Array) -> jax.Array:
        msg = jnp.einsum("nij,njh->nih", adj, h @ w_msg)  # aggregate neighbours
        return jnp.tanh(h @ w_self + msg + b)

    h1 = layer(x, params["w_self1"], params["w_msg1"], params["b1"])
    h2 = layer(h1, params["w_self2"], params["w_msg2"], params["b2"])
    readout = h2[:, READOUT_OSCILLATOR, :]  # (n, H)
    return readout @ params["w_out"] + params["b_out"]


def train_gnn(
    batch: TrialBatchV2,
    teacher_labels: NDArray[np.int64],
    hidden: int,
    config: ProbeConfigV2,
    seed: int,
    epochs: int = 2000,
    lr: float = 0.01,
) -> _Params:
    """Train the code-conditioned GNN on the training trials' teacher labels."""
    tr = ~batch.is_test
    theta0 = jnp.asarray(batch.theta0[tr])
    code = jnp.asarray(batch.code[tr])
    labels = jnp.asarray(np.asarray(teacher_labels)[tr])
    partners = partners_for(config.held_out, config.bridge_mode)
    base = jnp.asarray(base_coupling_matrix(config.k_ambient, config.k_bridge, partners))
    n_bins = config.n_bins
    params = gnn_init(seed, hidden, n_bins)

    @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
    def optimise() -> _Params:
        return _adam_descent(
            params,
            lambda p: _softmax_cross_entropy(gnn_logits(p, theta0, code, base), labels, n_bins),
            lr=lr,
            epochs=epochs,
        )

    return cast(_Params, optimise())


def gnn_predict(
    params: _Params, batch_slice: TrialBatchV2, config: ProbeConfigV2
) -> NDArray[np.int64]:
    """Predicted class per trial = argmax of the GNN logits."""
    partners = partners_for(config.held_out, config.bridge_mode)
    base = jnp.asarray(base_coupling_matrix(config.k_ambient, config.k_bridge, partners))
    logits = np.asarray(
        gnn_logits(params, jnp.asarray(batch_slice.theta0), jnp.asarray(batch_slice.code), base)
    )
    pred: NDArray[np.int64] = np.argmax(logits, axis=1).astype(np.int64)
    return pred
