# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA models: motif substrate, MLP baseline, chance floor
"""The motif substrate, the parameter-matched MLP baseline, and the chance floor.

* **Substrate** — a shared trainable coupling matrix ``K`` plus an
  input-conditioned natural-frequency drive (``ω(input) = Σ`` learned per
  ``(relation, pair)`` embeddings). It must *physically realise* the target
  order parameters by integrating the Kuramoto dynamics; the readout is the
  achieved order parameter of each active cluster pair.
* **MLP baseline** — a parameter-matched (±10 %) network that reads the raw
  initial phases (as sin/cos) plus the input code and *predicts the readout
  directly*, with no coupling-motif dynamics. It has a strictly easier job
  (emit a number vs. realise a state); a substrate win over it is therefore
  strong evidence that motif dynamics enable composition.
* **Chance floor** — a structure-blind constant policy (training-marginal
  order parameters), scored the same way. Measured, not assumed.
"""

from __future__ import annotations

from typing import Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .dynamics import integrate_kuramoto
from .task import N_OSC, PAIRS, ProbeConfig, TrialBatch, pair_members

_Params = dict[str, jax.Array]

_N_PAIRS = len(PAIRS)
_TRIU = np.triu_indices(N_OSC, k=1)
_N_K = _TRIU[0].size  # 120 free upper-triangular couplings
_MEMBERS = np.stack([pair_members(p) for p in range(_N_PAIRS)])  # (n_pairs, 8)


def substrate_param_count() -> int:
    """Trainable-parameter count of the motif substrate (K + drive)."""
    return _N_K + 2 * _N_PAIRS * N_OSC


# --------------------------------------------------------------------------- #
# Readout bookkeeping                                                          #
# --------------------------------------------------------------------------- #
def _member_tables(batch: TrialBatch) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-trial member indices + active masks for the R1 and R2 pairs."""
    r1_idx = np.clip(batch.r1_pair, 0, None)
    r2_idx = np.clip(batch.r2_pair, 0, None)
    r1_members = jnp.asarray(_MEMBERS[r1_idx])  # (n, 8)
    r2_members = jnp.asarray(_MEMBERS[r2_idx])
    r1_active = jnp.asarray(batch.r1_pair >= 0, dtype=jnp.float32)
    r2_active = jnp.asarray(batch.r2_pair >= 0, dtype=jnp.float32)
    return r1_members, r2_members, r1_active, r2_active


def _gathered_order_parameter(theta: jax.Array, members: jax.Array) -> jax.Array:
    """Order parameter per trial over its own ``(8,)`` member indices."""
    rows = jnp.arange(theta.shape[0])[:, None]
    selected = theta[rows, members]  # (n, 8)
    return jnp.abs(jnp.mean(jnp.exp(1j * selected), axis=1))


# --------------------------------------------------------------------------- #
# Motif substrate                                                             #
# --------------------------------------------------------------------------- #
def substrate_init(seed: int) -> dict[str, jax.Array]:
    """Initialise substrate parameters (moderate random K, small drive).

    K is seeded strong enough that the sync timescale ``~1/K`` is well within
    the fixed horizon, so gradient descent tunes an already-active substrate
    rather than having to grow coupling from ~0.
    """
    rng = np.random.default_rng(seed)
    k_raw = jnp.asarray(rng.normal(0.0, 0.5, size=_N_K))
    drive = jnp.asarray(rng.normal(0.0, 0.3, size=(2, _N_PAIRS, N_OSC)))
    return {"k_raw": k_raw, "drive": drive}


def _coupling(k_raw: jax.Array) -> jax.Array:
    """Symmetric zero-diagonal coupling from the free upper-triangular vector."""
    k = jnp.zeros((N_OSC, N_OSC))
    k = k.at[_TRIU].set(k_raw)
    return k + k.T


def _drive_for(drive: jax.Array, code: jax.Array) -> jax.Array:
    """``ω(input)`` — additive per-``(relation, pair)`` embedding, ``(n, N_OSC)``."""
    # code: (n, 2, n_pairs); drive: (2, n_pairs, N_OSC)
    return jnp.einsum("nrp,rpo->no", code, drive)


def substrate_readout(
    params: dict[str, jax.Array],
    theta0: jax.Array,
    code: jax.Array,
    r1_members: jax.Array,
    r2_members: jax.Array,
    config: ProbeConfig,
) -> tuple[jax.Array, jax.Array]:
    """Achieved ``(R1_order, R2_order)`` after integrating the substrate."""
    omega = _drive_for(params["drive"], code)
    coupling = _coupling(params["k_raw"])
    final = integrate_kuramoto(theta0, omega, coupling, config.dt, config.steps)
    return (
        _gathered_order_parameter(final, r1_members),
        _gathered_order_parameter(final, r2_members),
    )


def _substrate_loss(
    params: dict[str, jax.Array],
    theta0: jax.Array,
    code: jax.Array,
    r1_members: jax.Array,
    r2_members: jax.Array,
    r1_active: jax.Array,
    r2_active: jax.Array,
    config: ProbeConfig,
) -> jax.Array:
    """MSE to targets (R1→1, R2→0) on the active pairs only."""
    r1, r2 = substrate_readout(params, theta0, code, r1_members, r2_members, config)
    loss_r1 = jnp.sum(r1_active * (r1 - 1.0) ** 2)
    loss_r2 = jnp.sum(r2_active * (r2 - 0.0) ** 2)
    return (loss_r1 + loss_r2) / theta0.shape[0]


# --------------------------------------------------------------------------- #
# Fully jitted Adam training loop                                             #
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


def train_substrate(
    batch: TrialBatch,
    config: ProbeConfig,
    seed: int,
    epochs: int = 1200,
    lr: float = 0.05,
) -> dict[str, jax.Array]:
    """Train the substrate on the training trials of ``batch``."""
    tr = ~batch.is_test
    theta0 = jnp.asarray(batch.theta0[tr])
    code = jnp.asarray(batch.code[tr])
    r1m, r2m, r1a, r2a = _member_tables(
        TrialBatch(
            batch.theta0[tr],
            batch.code[tr],
            batch.r1_pair[tr],
            batch.r2_pair[tr],
            batch.is_test[tr],
        )
    )
    params = substrate_init(seed)

    @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
    def optimise() -> _Params:
        loss_fn = lambda p: _substrate_loss(p, theta0, code, r1m, r2m, r1a, r2a, config)  # noqa: E731
        return _adam_descent(params, loss_fn, lr=lr, epochs=epochs)

    result: _Params = optimise()
    return result


# --------------------------------------------------------------------------- #
# Parameter-matched MLP baseline                                              #
# --------------------------------------------------------------------------- #
def mlp_hidden_for_match(target_params: int) -> int:
    """Hidden width making a 1-layer MLP's param count closest to ``target``."""
    in_dim = 2 * N_OSC + 2 * _N_PAIRS  # sin/cos phases + flattened code
    # params(h) = in_dim*h + h + h*2 + 2
    best_h, best_gap = 1, None
    for h in range(1, 64):
        count = in_dim * h + h + h * 2 + 2
        gap = abs(count - target_params)
        if best_gap is None or gap < best_gap:
            best_h, best_gap = h, gap
    return best_h


def mlp_param_count(hidden: int) -> int:
    """Parameter count of the 1-hidden-layer MLP baseline."""
    in_dim = 2 * N_OSC + 2 * _N_PAIRS
    return in_dim * hidden + hidden + hidden * 2 + 2


def _mlp_features(theta0: NDArray[np.float64], code: NDArray[np.float64]) -> jax.Array:
    """Raw initial phases (sin/cos) concatenated with the flattened code."""
    feats = np.concatenate(
        [np.sin(theta0), np.cos(theta0), code.reshape(code.shape[0], -1)], axis=1
    )
    return jnp.asarray(feats)


def mlp_init(seed: int, hidden: int) -> dict[str, jax.Array]:
    """Initialise MLP parameters (Glorot-ish)."""
    rng = np.random.default_rng(seed + 7919)
    in_dim = 2 * N_OSC + 2 * _N_PAIRS
    scale1 = np.sqrt(1.0 / in_dim)
    scale2 = np.sqrt(1.0 / hidden)
    return {
        "w1": jnp.asarray(rng.normal(0, scale1, size=(in_dim, hidden))),
        "b1": jnp.zeros(hidden),
        "w2": jnp.asarray(rng.normal(0, scale2, size=(hidden, 2))),
        "b2": jnp.zeros(2),
    }


def mlp_forward(params: dict[str, jax.Array], feats: jax.Array) -> jax.Array:
    """Predict ``(R1_order, R2_order) ∈ [0, 1]`` per trial."""
    h = jnp.tanh(feats @ params["w1"] + params["b1"])
    return jax.nn.sigmoid(h @ params["w2"] + params["b2"])


def _mlp_loss(
    params: dict[str, jax.Array],
    feats: jax.Array,
    r1_active: jax.Array,
    r2_active: jax.Array,
) -> jax.Array:
    """MSE to targets (R1→1, R2→0) on active slots."""
    pred = mlp_forward(params, feats)
    loss_r1 = jnp.sum(r1_active * (pred[:, 0] - 1.0) ** 2)
    loss_r2 = jnp.sum(r2_active * (pred[:, 1] - 0.0) ** 2)
    return (loss_r1 + loss_r2) / feats.shape[0]


def train_mlp(batch: TrialBatch, hidden: int, seed: int, epochs: int = 2000) -> _Params:
    """Train the MLP baseline on the training trials."""
    tr = ~batch.is_test
    feats = _mlp_features(batch.theta0[tr], batch.code[tr])
    r1a = jnp.asarray(batch.r1_pair[tr] >= 0, dtype=jnp.float32)
    r2a = jnp.asarray(batch.r2_pair[tr] >= 0, dtype=jnp.float32)
    params = mlp_init(seed, hidden)

    @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
    def optimise() -> _Params:
        return _adam_descent(
            params, lambda p: _mlp_loss(p, feats, r1a, r2a), lr=0.02, epochs=epochs
        )

    result: _Params = optimise()
    return result


def chance_floor_accuracy(batch: TrialBatch, epsilon: float, seed: int) -> float:
    """Measured success rate of a structure-blind random policy on the test set.

    The floor emits ``(R1_order, R2_order) ~ U[0, 1]²`` per test trial — it uses
    neither the input code nor the initial phases — and is scored with the exact
    frozen success criterion. Measured (not the ``0.15² ≈ 0.0225`` closed form),
    so a finite-sample floor is reported honestly.

    (A degenerate *constant* ``(1, 0)`` policy would score 100 % because the
    target is a trivial function of "which relations are active"; that is why
    the load-bearing comparator is the parameter-matched MLP, not this floor —
    the probe tests realisation/composition, not target knowledge.)
    """
    from .task import success_mask

    test = batch.is_test
    n = int(test.sum())
    rng = np.random.default_rng(seed + 104729)
    r1 = rng.uniform(0.0, 1.0, size=n)
    r2 = rng.uniform(0.0, 1.0, size=n)
    return float(np.mean(success_mask(r1, r2, epsilon)))
