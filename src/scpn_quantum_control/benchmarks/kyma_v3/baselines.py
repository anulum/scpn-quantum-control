# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v3 non-oscillator baselines
"""Non-oscillator baselines for the symbolic KYMA v3 task.

Contract baselines (parameter count within ±10 % of the substrate):

* **MLP** — one tanh hidden layer over ``sin/cos`` register phases and one-hot
  first operation, second operation (with "none") and query.
* **Staged GNN** — the registers are graph nodes; each operation owns a learned
  3×3 adjacency and node bias, and one message-passing round is applied per
  operation in program order (a code-conditioned relational model with the same
  staging as the substrate but no oscillator dynamics).
* **Transformer** — one single-head attention layer over the tokens
  ``[first op, second op, query, a, b, c]``, read out at the query token.

Diagnostic baseline (reported, not part of the contract):

* **Sequential MLP** — one small MLP per operation maps ``sin/cos`` register
  phases to ``sin/cos`` phases and is applied in program order; the label is the
  queried register's angle rounded to the lattice. It isolates what staged
  operator application alone achieves.

Every model is trained with full-batch Adam on the training items only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ..kyma_v2.models import _adam_descent
from .substrate import PHASE_STEP, phase_to_label
from .task import N_OPERATIONS, N_REGISTERS, N_VALUES, NO_OPERATION, SymbolicDataset

_Params = dict[str, jax.Array]
N_OP_CODES = N_OPERATIONS + 1  # the three operations plus "none"
FEATURE_DIM = 2 * N_REGISTERS + N_OPERATIONS + N_OP_CODES + N_REGISTERS  # 16
N_TOKENS = 3 + N_REGISTERS


def features(dataset: SymbolicDataset, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Flat input features of the MLP baseline.

    Parameters
    ----------
    dataset
        The frozen split.
    mask
        Items to encode.

    Returns
    -------
    numpy.ndarray
        ``(n, 16)``: ``sin`` and ``cos`` of the three register phases, one-hot
        first operation, one-hot second operation (with "none") and one-hot query.

    """
    phase = dataset.states[mask] * PHASE_STEP
    eye_op, eye_code, eye_query = np.eye(N_OPERATIONS), np.eye(N_OP_CODES), np.eye(N_REGISTERS)
    return np.concatenate(
        [
            np.sin(phase),
            np.cos(phase),
            eye_op[dataset.first_op[mask]],
            eye_code[dataset.second_op[mask]],
            eye_query[dataset.query[mask]],
        ],
        axis=1,
    )


def _glorot(rng: np.random.Generator, shape: tuple[int, ...]) -> jax.Array:
    return jnp.asarray(rng.normal(0.0, np.sqrt(1.0 / shape[0]), size=shape))


def _cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    logp = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(labels, N_VALUES) * logp, axis=1))


# --------------------------------------------------------------------------- MLP
def mlp_param_count(hidden: int) -> int:
    """Parameters of the one-hidden-layer MLP of width ``hidden``."""
    return FEATURE_DIM * hidden + hidden + hidden * N_VALUES + N_VALUES


def mlp_init(seed: int, hidden: int) -> _Params:
    """Initialise the MLP."""
    rng = np.random.default_rng(seed + 7919)
    return {
        "w1": _glorot(rng, (FEATURE_DIM, hidden)),
        "b1": jnp.zeros(hidden),
        "w2": _glorot(rng, (hidden, N_VALUES)),
        "b2": jnp.zeros(N_VALUES),
    }


def mlp_logits(params: _Params, feats: jax.Array) -> jax.Array:
    """Class logits of the MLP."""
    return jnp.tanh(feats @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]


# --------------------------------------------------------------------- staged GNN
def gnn_param_count(hidden: int) -> int:
    """Parameters of the staged GNN of width ``hidden``."""
    adjacency = N_OPERATIONS * N_REGISTERS * N_REGISTERS
    embed = 2 * hidden + hidden
    rounds = 2 * hidden * hidden + N_OPERATIONS * hidden
    head = hidden * N_VALUES + N_VALUES
    return adjacency + embed + rounds + head


def gnn_init(seed: int, hidden: int) -> _Params:
    """Initialise the staged GNN."""
    rng = np.random.default_rng(seed + 271828)
    return {
        "adjacency": jnp.asarray(rng.normal(0.0, 0.3, size=(N_OPERATIONS, 3, 3))),
        "w_in": _glorot(rng, (2, hidden)),
        "b_in": jnp.zeros(hidden),
        "w_msg": _glorot(rng, (hidden, hidden)),
        "w_self": _glorot(rng, (hidden, hidden)),
        "b_op": jnp.zeros((N_OPERATIONS, hidden)),
        "w_out": _glorot(rng, (hidden, N_VALUES)),
        "b_out": jnp.zeros(N_VALUES),
    }


def gnn_logits(
    params: _Params, states: jax.Array, first_op: jax.Array, second_op: jax.Array, query: jax.Array
) -> jax.Array:
    """Class logits of the staged GNN, read at the queried node."""
    phase = states.astype(jnp.float32) * PHASE_STEP
    x = jnp.stack([jnp.sin(phase), jnp.cos(phase)], axis=-1)  # (n, 3, 2)
    h = jnp.tanh(x @ params["w_in"] + params["b_in"])

    def round_(h: jax.Array, operation: jax.Array) -> jax.Array:
        adjacency = params["adjacency"][operation]  # (n, 3, 3)
        message = jnp.einsum("nij,njh->nih", adjacency, h @ params["w_msg"])
        return jnp.tanh(message + h @ params["w_self"] + params["b_op"][operation][:, None, :])

    h = round_(h, first_op)
    second = jnp.where(second_op == NO_OPERATION, 0, second_op)
    h = jnp.where((second_op == NO_OPERATION)[:, None, None], h, round_(h, second))
    node = jnp.take_along_axis(h, query[:, None, None], axis=1)[:, 0, :]
    return node @ params["w_out"] + params["b_out"]


# -------------------------------------------------------------------- transformer
def transformer_param_count(width: int) -> int:
    """Parameters of the one-layer single-head transformer of width ``width``."""
    embeddings = N_OP_CODES * width + N_REGISTERS * width + 2 * width + width + N_TOKENS * width
    attention = 4 * width * width
    head = width * N_VALUES + N_VALUES
    return embeddings + attention + head


def transformer_init(seed: int, width: int) -> _Params:
    """Initialise the transformer."""
    rng = np.random.default_rng(seed + 314159)
    return {
        "op_embed": jnp.asarray(rng.normal(0.0, 0.3, size=(N_OP_CODES, width))),
        "query_embed": jnp.asarray(rng.normal(0.0, 0.3, size=(N_REGISTERS, width))),
        "w_reg": _glorot(rng, (2, width)),
        "b_reg": jnp.zeros(width),
        "position": jnp.asarray(rng.normal(0.0, 0.3, size=(N_TOKENS, width))),
        "w_q": _glorot(rng, (width, width)),
        "w_k": _glorot(rng, (width, width)),
        "w_v": _glorot(rng, (width, width)),
        "w_o": _glorot(rng, (width, width)),
        "w_out": _glorot(rng, (width, N_VALUES)),
        "b_out": jnp.zeros(N_VALUES),
    }


def transformer_logits(
    params: _Params, states: jax.Array, first_op: jax.Array, second_op: jax.Array, query: jax.Array
) -> jax.Array:
    """Class logits of the transformer, read at the query token."""
    phase = states.astype(jnp.float32) * PHASE_STEP
    registers = jnp.stack([jnp.sin(phase), jnp.cos(phase)], axis=-1) @ params["w_reg"]
    registers = registers + params["b_reg"]
    tokens = jnp.concatenate(
        [
            params["op_embed"][first_op][:, None, :],
            params["op_embed"][second_op][:, None, :],
            params["query_embed"][query][:, None, :],
            registers,
        ],
        axis=1,
    )
    tokens = tokens + params["position"][None, :, :]
    width = tokens.shape[-1]
    scores = (tokens @ params["w_q"]) @ jnp.swapaxes(tokens @ params["w_k"], 1, 2) / np.sqrt(width)
    attended = jax.nn.softmax(scores, axis=-1) @ (tokens @ params["w_v"])
    out = tokens + attended @ params["w_o"]
    return out[:, 2, :] @ params["w_out"] + params["b_out"]


# ---------------------------------------------------------------- sequential MLP
def sequential_param_count(hidden: int) -> int:
    """Parameters of the diagnostic sequential MLP (one MLP per operation)."""
    per_operation = 2 * N_REGISTERS * hidden + hidden + hidden * 2 * N_REGISTERS + 2 * N_REGISTERS
    return N_OPERATIONS * per_operation


def sequential_init(seed: int, hidden: int) -> _Params:
    """Initialise the per-operation MLPs."""
    rng = np.random.default_rng(seed + 161803)
    d = 2 * N_REGISTERS
    return {
        "w1": jnp.asarray(rng.normal(0.0, np.sqrt(1.0 / d), size=(N_OPERATIONS, d, hidden))),
        "b1": jnp.zeros((N_OPERATIONS, hidden)),
        "w2": jnp.asarray(rng.normal(0.0, np.sqrt(1.0 / hidden), size=(N_OPERATIONS, hidden, d))),
        "b2": jnp.zeros((N_OPERATIONS, d)),
    }


def sequential_angles(
    params: _Params, states: jax.Array, first_op: jax.Array, second_op: jax.Array
) -> jax.Array:
    """Return the final register angles after the per-operation maps in order."""
    phase = states.astype(jnp.float32) * PHASE_STEP
    z = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=1)  # (n, 6)

    def apply(z: jax.Array, operation: jax.Array) -> jax.Array:
        hidden = jnp.tanh(
            jnp.einsum("nd,ndh->nh", z, params["w1"][operation]) + params["b1"][operation]
        )
        return jnp.einsum("nh,nhd->nd", hidden, params["w2"][operation]) + params["b2"][operation]

    z = apply(z, first_op)
    second = jnp.where(second_op == NO_OPERATION, 0, second_op)
    z = jnp.where((second_op == NO_OPERATION)[:, None], z, apply(z, second))
    return jnp.arctan2(z[:, :N_REGISTERS], z[:, N_REGISTERS:])


# ----------------------------------------------------------------- common driver
@dataclass(frozen=True)
class BaselineSpec:
    """One baseline: name, width, parameter count, and whether it is in the contract."""

    name: str
    width: int
    params: int
    contract: bool


def closest_width(count: Callable[[int], int], target: int) -> int:
    """Width whose parameter count is closest to ``target`` (smaller width on ties).

    Parameters
    ----------
    count
        Parameter-count function of the width.
    target
        Parameter count to match.

    Returns
    -------
    int
        The chosen width in ``1 … 64``.

    """
    return min(range(1, 65), key=lambda width: (abs(count(width) - target), width))


def _seconds_per_item(infer: Callable[[], jax.Array], dataset: SymbolicDataset) -> float:
    """Wall-clock seconds per test item of a second (warm) inference pass."""
    infer().block_until_ready()
    start = time.perf_counter()
    infer().block_until_ready()
    return (time.perf_counter() - start) / int(np.sum(dataset.is_test))


def _arrays(dataset: SymbolicDataset, mask: NDArray[np.bool_]) -> tuple[jax.Array, ...]:
    return (
        jnp.asarray(dataset.states[mask]),
        jnp.asarray(dataset.first_op[mask]),
        jnp.asarray(dataset.second_op[mask]),
        jnp.asarray(dataset.query[mask]),
    )


def train_and_predict(
    name: str,
    width: int,
    dataset: SymbolicDataset,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], float]:
    """Train one baseline, predict the training and test items, and time inference.

    Parameters
    ----------
    name
        ``"mlp"``, ``"gnn"``, ``"transformer"`` or ``"sequential"``.
    width
        Hidden width.
    dataset
        The frozen split.
    seed
        Initialisation seed.
    epochs
        Adam steps.
    learning_rate
        Adam step size.

    Returns
    -------
    tuple
        Predicted labels for the training items, for the test items, and the
        measured wall-clock seconds per test item of a compiled inference pass.

    Raises
    ------
    ValueError
        If ``name`` is unknown.

    """
    train_mask = ~dataset.is_test
    labels = jnp.asarray(dataset.label[train_mask])
    if name == "mlp":
        params = mlp_init(seed, width)

        def logits(p: _Params, mask: NDArray[np.bool_]) -> jax.Array:
            return mlp_logits(p, jnp.asarray(features(dataset, mask)))

    elif name == "gnn":
        params = gnn_init(seed, width)

        def logits(p: _Params, mask: NDArray[np.bool_]) -> jax.Array:
            return gnn_logits(p, *_arrays(dataset, mask))

    elif name == "transformer":
        params = transformer_init(seed, width)

        def logits(p: _Params, mask: NDArray[np.bool_]) -> jax.Array:
            return transformer_logits(p, *_arrays(dataset, mask))

    elif name == "sequential":
        params = sequential_init(seed, width)
        target = labels.astype(jnp.float32) * PHASE_STEP

        def angle(p: _Params, mask: NDArray[np.bool_]) -> jax.Array:
            states, first_op, second_op, query = _arrays(dataset, mask)
            angles = sequential_angles(p, states, first_op, second_op)
            return jnp.take_along_axis(angles, query[:, None], axis=1)[:, 0]

        @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
        def optimise_angle() -> _Params:
            def loss(p: _Params) -> jax.Array:
                return jnp.mean(1.0 - jnp.cos(angle(p, train_mask) - target))

            return _adam_descent(params, loss, lr=learning_rate, epochs=epochs)

        trained = cast(_Params, optimise_angle())
        return (
            np.asarray(phase_to_label(angle(trained, train_mask)), dtype=np.int64),
            np.asarray(phase_to_label(angle(trained, dataset.is_test)), dtype=np.int64),
            _seconds_per_item(lambda: angle(trained, dataset.is_test), dataset),
        )
    else:
        raise ValueError(f"unknown baseline {name!r}")

    @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
    def optimise() -> _Params:
        return _adam_descent(
            params,
            lambda p: _cross_entropy(logits(p, train_mask), labels),
            lr=learning_rate,
            epochs=epochs,
        )

    trained = cast(_Params, optimise())
    return (
        np.asarray(jnp.argmax(logits(trained, train_mask), axis=1), dtype=np.int64),
        np.asarray(jnp.argmax(logits(trained, dataset.is_test), axis=1), dtype=np.int64),
        _seconds_per_item(lambda: logits(trained, dataset.is_test), dataset),
    )
