# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2 models: gated student, MLP baseline, chance floor
"""The trainable gated **student** substrate, a parameter-matched MLP baseline,
and the chance floor.

* **Student** — a gated Kuramoto substrate (same architecture as the teacher)
  whose per-``(relation, pair)`` motif gates are *learned*. It is trained to
  reproduce the teacher's achieved final phases (a differentiable circular loss
  through the RK4 solver); the class prediction is the same quantised readout the
  teacher uses. Because co-activated gates compose *dynamically* through the fixed
  ambient coupling, a student that recovers the single-relation motifs realises
  the held-out conjunction it never trained on jointly — the compositional claim.
* **MLP baseline** — a parameter-matched (±10 %) one-hidden-layer softmax network
  reading ``sin/cos(θ0)`` and the code, predicting the class directly. It has the
  same data but no compositional dynamics, so on the held-out conjunction it must
  extrapolate a non-separable map it never saw — the load-bearing comparison.
* **Chance floor** — the most-frequent training class, scored on the test set.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .coupling import assemble_coupling, base_coupling_matrix, partners_for
from .dynamics import integrate_kuramoto_batched, phase_label
from .task import (
    N_OSC,
    N_PAIRS,
    N_RELATIONS,
    READOUT_OSCILLATOR,
    ProbeConfigV2,
    TrialBatchV2,
    pair_members,
)

_Params = dict[str, jax.Array]

# --------------------------------------------------------------------------- #
# Compact motif basis: each (relation, pair) motif is a combination of the      #
# 28 = C(8,2) symmetric unit edge-matrices over the pair's eight oscillators.   #
# --------------------------------------------------------------------------- #
EDGES_PER_PAIR = len(list(combinations(range(8), 2)))  # 28


def _edge_basis() -> NDArray[np.float64]:
    """``(N_PAIRS, EDGES_PER_PAIR, N_OSC, N_OSC)`` symmetric unit-edge matrices."""
    basis = np.zeros((N_PAIRS, EDGES_PER_PAIR, N_OSC, N_OSC), dtype=np.float64)
    for p in range(N_PAIRS):
        members = pair_members(p)
        for e, (a, b) in enumerate(combinations(members.tolist(), 2)):
            basis[p, e, a, b] = 1.0
            basis[p, e, b, a] = 1.0
    return basis


_BASIS = _edge_basis()  # constant


def substrate_param_count() -> int:
    """Trainable-parameter count of the gated student (free motif edges)."""
    return N_RELATIONS * N_PAIRS * EDGES_PER_PAIR


# --------------------------------------------------------------------------- #
# Gated student substrate                                                      #
# --------------------------------------------------------------------------- #
def student_init(seed: int) -> _Params:
    """Small random motif gates ``(N_RELATIONS, N_PAIRS, EDGES_PER_PAIR)``."""
    rng = np.random.default_rng(seed)
    free = rng.normal(0.0, 0.3, size=(N_RELATIONS, N_PAIRS, EDGES_PER_PAIR))
    return {"gates_free": jnp.asarray(free)}


def _student_gates(free: jax.Array) -> jax.Array:
    """Expand free edge weights to symmetric ``(N_RELATIONS, N_PAIRS, N, N)`` gates."""
    basis = jnp.asarray(_BASIS)
    return jnp.einsum("rpe,peij->rpij", free, basis)


def student_final_phases(
    params: _Params, theta0: jax.Array, code: jax.Array, base: jax.Array, config: ProbeConfigV2
) -> jax.Array:
    """Integrate the student and return the ``(batch, N_OSC)`` achieved phases."""
    gates = _student_gates(params["gates_free"])
    coupling = assemble_coupling(code, gates, base)
    omega = jnp.zeros((theta0.shape[0], N_OSC))
    return integrate_kuramoto_batched(theta0, omega, coupling, config.dt, config.steps)


def _circular_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean ``1 − cos(pred − target)`` over trials and oscillators (phase MSE-analogue)."""
    return jnp.mean(1.0 - jnp.cos(pred - target))


# --------------------------------------------------------------------------- #
# Fully jitted Adam training loop (shared)                                     #
# --------------------------------------------------------------------------- #
_ADAM_B1, _ADAM_B2, _ADAM_EPS = 0.9, 0.999, 1e-8


def _adam_descent(
    params: _Params, loss_fn: Callable[[_Params], jax.Array], lr: float, epochs: int
) -> _Params:
    """Run ``epochs`` Adam steps of ``loss_fn`` as a single jitted fori_loop."""
    grad_fn = jax.grad(loss_fn)
    m0 = jax.tree_util.tree_map(jnp.zeros_like, params)
    v0 = jax.tree_util.tree_map(jnp.zeros_like, params)

    def body(t: jax.Array, carry: tuple) -> tuple:  # type: ignore[type-arg]
        p, m, v = carry
        g = grad_fn(p)
        m = jax.tree_util.tree_map(lambda mm, gg: _ADAM_B1 * mm + (1 - _ADAM_B1) * gg, m, g)
        v = jax.tree_util.tree_map(lambda vv, gg: _ADAM_B2 * vv + (1 - _ADAM_B2) * gg * gg, v, g)
        step = t + 1
        bc1 = 1 - _ADAM_B1**step
        bc2 = 1 - _ADAM_B2**step
        p = jax.tree_util.tree_map(
            lambda pp, mm, vv: pp - lr * (mm / bc1) / (jnp.sqrt(vv / bc2) + _ADAM_EPS),
            p,
            m,
            v,
        )
        return p, m, v

    final, _, _ = jax.lax.fori_loop(0, epochs, body, (params, m0, v0))
    return cast(_Params, final)


def train_student(
    batch: TrialBatchV2,
    teacher_finals: NDArray[np.float64],
    config: ProbeConfigV2,
    seed: int,
    epochs: int = 1500,
    lr: float = 0.05,
) -> _Params:
    """Train the student to reproduce the teacher's final phases on training trials."""
    tr = ~batch.is_test
    theta0 = jnp.asarray(batch.theta0[tr])
    code = jnp.asarray(batch.code[tr])
    target = jnp.asarray(np.asarray(teacher_finals)[tr])
    partners = partners_for(config.held_out, config.bridge_mode)
    base = jnp.asarray(base_coupling_matrix(config.k_ambient, config.k_bridge, partners))
    params = student_init(seed)

    @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
    def optimise() -> _Params:
        def loss_fn(p: _Params) -> jax.Array:
            pred = student_final_phases(p, theta0, code, base, config)
            return _circular_loss(pred, target)

        return _adam_descent(params, loss_fn, lr=lr, epochs=epochs)

    return cast(_Params, optimise())


def student_predict(
    params: _Params, batch_slice: TrialBatchV2, config: ProbeConfigV2
) -> NDArray[np.int64]:
    """Predicted class per trial = quantised student readout phase."""
    partners = partners_for(config.held_out, config.bridge_mode)
    base = jnp.asarray(base_coupling_matrix(config.k_ambient, config.k_bridge, partners))
    final = student_final_phases(
        params, jnp.asarray(batch_slice.theta0), jnp.asarray(batch_slice.code), base, config
    )
    labels = phase_label(final, READOUT_OSCILLATOR, config.n_bins)
    return np.asarray(labels, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Parameter-matched MLP baseline (softmax over classes)                       #
# --------------------------------------------------------------------------- #
def _mlp_in_dim() -> int:
    return 2 * N_OSC + N_RELATIONS * N_PAIRS


def mlp_hidden_for_match(target_params: int, n_bins: int) -> int:
    """Hidden width making a 1-layer MLP's param count closest to ``target``."""
    in_dim = _mlp_in_dim()
    best_h, best_gap = 1, None
    for h in range(1, 128):
        count = in_dim * h + h + h * n_bins + n_bins
        gap = abs(count - target_params)
        if best_gap is None or gap < best_gap:
            best_h, best_gap = h, gap
    return best_h


def mlp_param_count(hidden: int, n_bins: int) -> int:
    """Parameter count of the 1-hidden-layer MLP baseline."""
    in_dim = _mlp_in_dim()
    return in_dim * hidden + hidden + hidden * n_bins + n_bins


def _mlp_features(theta0: NDArray[np.float64], code: NDArray[np.float64]) -> jax.Array:
    """Raw initial phases (sin/cos) concatenated with the flattened code."""
    feats = np.concatenate(
        [np.sin(theta0), np.cos(theta0), code.reshape(code.shape[0], -1)], axis=1
    )
    return jnp.asarray(feats)


def mlp_init(seed: int, hidden: int, n_bins: int) -> _Params:
    """Initialise MLP parameters (Glorot-ish)."""
    rng = np.random.default_rng(seed + 7919)
    in_dim = _mlp_in_dim()
    scale1 = np.sqrt(1.0 / in_dim)
    scale2 = np.sqrt(1.0 / hidden)
    return {
        "w1": jnp.asarray(rng.normal(0, scale1, size=(in_dim, hidden))),
        "b1": jnp.zeros(hidden),
        "w2": jnp.asarray(rng.normal(0, scale2, size=(hidden, n_bins))),
        "b2": jnp.zeros(n_bins),
    }


def mlp_logits(params: _Params, feats: jax.Array) -> jax.Array:
    """Class logits ``(batch, n_bins)``."""
    h = jnp.tanh(feats @ params["w1"] + params["b1"])
    return h @ params["w2"] + params["b2"]


def _mlp_loss(params: _Params, feats: jax.Array, labels: jax.Array, n_bins: int) -> jax.Array:
    """Softmax cross-entropy to the teacher class labels."""
    logits = mlp_logits(params, feats)
    logp = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
    onehot = jax.nn.one_hot(labels, n_bins)
    return -jnp.mean(jnp.sum(onehot * logp, axis=1))


def train_mlp(
    batch: TrialBatchV2,
    teacher_labels: NDArray[np.int64],
    hidden: int,
    config: ProbeConfigV2,
    seed: int,
    epochs: int = 2000,
    lr: float = 0.02,
) -> _Params:
    """Train the MLP baseline on the training trials' teacher labels."""
    tr = ~batch.is_test
    feats = _mlp_features(batch.theta0[tr], batch.code[tr])
    labels = jnp.asarray(np.asarray(teacher_labels)[tr])
    params = mlp_init(seed, hidden, config.n_bins)
    n_bins = config.n_bins

    @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
    def optimise() -> _Params:
        return _adam_descent(
            params, lambda p: _mlp_loss(p, feats, labels, n_bins), lr=lr, epochs=epochs
        )

    return cast(_Params, optimise())


def mlp_predict(params: _Params, batch_slice: TrialBatchV2) -> NDArray[np.int64]:
    """Predicted class per trial = argmax of the MLP logits."""
    feats = _mlp_features(batch_slice.theta0, batch_slice.code)
    logits = np.asarray(mlp_logits(params, feats))
    pred: NDArray[np.int64] = np.argmax(logits, axis=1).astype(np.int64)
    return pred


def chance_floor_accuracy(
    teacher_labels: NDArray[np.int64], is_test: NDArray[np.bool_], n_bins: int
) -> float:
    """Accuracy of the most-frequent *training* class, scored on the test set."""
    train_labels = teacher_labels[~is_test]
    test_labels = teacher_labels[is_test]
    majority = int(np.bincount(train_labels, minlength=n_bins).argmax())
    return float(np.mean(test_labels == majority))
