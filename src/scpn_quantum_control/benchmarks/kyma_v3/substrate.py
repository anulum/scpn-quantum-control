# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v3 staged gated-coupling oscillator substrate
"""Staged gated-coupling oscillator substrate for the symbolic KYMA v3 task.

Each register is one oscillator whose phase relative to a fixed reference encodes
its value (``v ↦ v·π/2``). The substrate has two banks of three oscillators. One
operation is one *write stage*: the destination bank is reset to the off-lattice
phase ``π/4`` and integrated while the source bank is held (it receives no
coupling). The operation code gates which couplings drive the destination bank::

    dθ_i/dt = Σ_j K[o,i,j] sin(x_j − θ_i + α[o,i,j])
            + Σ_p T[o,i,p] sin(x_{p1} + x_{p2} − θ_i + β[o,i,p])

with directed pairwise couplings ``K`` and phase lags ``α`` and triadic couplings
``T`` over the three source-register pairs with lags ``β`` (the reference phase is
zero, so ``x_{p1} + x_{p2} − θ_i`` is the rotation-invariant triadic term with the
reference subtracted). The banks then swap roles, so a two-operation program is
two write stages with the gates of each operation in turn. The label is the
queried register's final phase rounded to the nearest lattice point.

The gates are the trainable parameters (``3 × 36 = 108``). :func:`hand_gates`
sets one exact solution by hand; it is used only for the teacher-free
realisability check, never as a training target.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ..kyma_v2.models import _adam_descent
from .task import N_OPERATIONS, N_REGISTERS, N_VALUES, NO_OPERATION, SymbolicDataset

PHASE_STEP = float(np.pi / 2)
RESET_PHASE = float(np.pi / 4)
DT = 0.05
STEPS_PER_STAGE = 60
HAND_COUPLING = 2.0
PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))

_Params = dict[str, jax.Array]
_GATE_NAMES = ("coupling", "lag", "triadic", "triadic_lag")


def substrate_param_count() -> int:
    """Return the number of trainable gate parameters.

    Returns
    -------
    int
        ``N_OPERATIONS × 4 × N_REGISTERS × 3`` = 108.

    """
    return N_OPERATIONS * len(_GATE_NAMES) * N_REGISTERS * len(PAIRS)


def hand_gates(coupling: float = HAND_COUPLING) -> _Params:
    """Return one hand-set exact solution of the three operations.

    ``R0`` copies ``a ← b, b ← c, c ← a``; ``R1`` drives ``a`` to ``a + b`` through a
    triadic term and copies ``b`` and ``c``; ``R2`` copies ``a`` and ``c`` and drives
    ``b`` to ``b + π/2`` through a phase lag.

    Parameters
    ----------
    coupling
        Strength of every active coupling.

    Returns
    -------
    dict
        Gate tensors ``coupling``, ``lag``, ``triadic``, ``triadic_lag``, each of
        shape ``(3, 3, 3)`` indexed ``[operation, target, source-or-pair]``.

    """
    shape = (N_OPERATIONS, N_REGISTERS, N_REGISTERS)
    pairwise = np.zeros(shape)
    lag = np.zeros(shape)
    triadic = np.zeros(shape)
    for target, source in ((0, 1), (1, 2), (2, 0)):
        pairwise[0, target, source] = coupling
    triadic[1, 0, 0] = coupling
    pairwise[1, 1, 1] = pairwise[1, 2, 2] = coupling
    pairwise[2, 0, 0] = pairwise[2, 1, 1] = pairwise[2, 2, 2] = coupling
    lag[2, 1, 1] = PHASE_STEP
    return {
        "coupling": jnp.asarray(pairwise),
        "lag": jnp.asarray(lag),
        "triadic": jnp.asarray(triadic),
        "triadic_lag": jnp.zeros(shape),
    }


def init_gates(seed: int) -> _Params:
    """Draw small random gates for training.

    Parameters
    ----------
    seed
        Seed of the NumPy generator.

    Returns
    -------
    dict
        Gate tensors drawn from ``N(0, 0.3²)``.

    """
    rng = np.random.default_rng(seed)
    shape = (N_OPERATIONS, N_REGISTERS, N_REGISTERS)
    return {name: jnp.asarray(rng.normal(0.0, 0.3, size=shape)) for name in _GATE_NAMES}


def _write_stage(gates: _Params, operation: jax.Array, source: jax.Array) -> jax.Array:
    """Integrate one write stage for a batch; returns the destination bank phases."""
    coupling = gates["coupling"][operation]  # (n, target, source)
    lag = gates["lag"][operation]
    triadic = gates["triadic"][operation]  # (n, target, pair)
    triadic_lag = gates["triadic_lag"][operation]
    pair_sum = jnp.stack([source[:, p] + source[:, q] for p, q in PAIRS], axis=1)  # (n, pair)

    def rhs(theta: jax.Array) -> jax.Array:
        pairwise = coupling * jnp.sin(source[:, None, :] - theta[:, :, None] + lag)
        three = triadic * jnp.sin(pair_sum[:, None, :] - theta[:, :, None] + triadic_lag)
        return jnp.sum(pairwise, axis=2) + jnp.sum(three, axis=2)

    def step(theta: jax.Array, _: None) -> tuple[jax.Array, None]:
        k1 = rhs(theta)
        k2 = rhs(theta + 0.5 * DT * k1)
        k3 = rhs(theta + 0.5 * DT * k2)
        k4 = rhs(theta + DT * k3)
        return theta + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    start = jnp.full_like(source, RESET_PHASE)
    final, _ = jax.lax.scan(step, start, None, length=STEPS_PER_STAGE)
    return final


def final_phases(
    gates: _Params, states: jax.Array, first_op: jax.Array, second_op: jax.Array
) -> jax.Array:
    """Run each item's program and return the final register phases.

    Parameters
    ----------
    gates
        Gate tensors.
    states
        ``(n, 3)`` initial register values.
    first_op, second_op
        ``(n,)`` operation indices; ``second_op == NO_OPERATION`` skips stage two.

    Returns
    -------
    jax.Array
        ``(n, 3)`` final phases.

    """
    start = states.astype(jnp.float32) * PHASE_STEP
    after_first = _write_stage(gates, first_op, start)
    second = jnp.where(second_op == NO_OPERATION, 0, second_op)
    after_second = _write_stage(gates, second, after_first)
    return jnp.where((second_op == NO_OPERATION)[:, None], after_first, after_second)


def phase_to_label(phase: jax.Array) -> jax.Array:
    """Round phases to the nearest lattice value in ``0 … 3``.

    Parameters
    ----------
    phase
        Phases relative to the reference.

    Returns
    -------
    jax.Array
        Integer labels.

    """
    shifted = jnp.mod(phase + PHASE_STEP / 2, 2.0 * jnp.pi)
    return jnp.floor(shifted / PHASE_STEP).astype(jnp.int32) % N_VALUES


def _queried(phases: jax.Array, query: jax.Array) -> jax.Array:
    return jnp.take_along_axis(phases, query[:, None], axis=1)[:, 0]


def predict(
    gates: _Params, dataset: SymbolicDataset, mask: NDArray[np.bool_]
) -> NDArray[np.int64]:
    """Predict labels for the masked items.

    Parameters
    ----------
    gates
        Gate tensors.
    dataset
        The frozen split.
    mask
        Items to predict.

    Returns
    -------
    numpy.ndarray
        Predicted labels.

    """
    phases = final_phases(
        gates,
        jnp.asarray(dataset.states[mask]),
        jnp.asarray(dataset.first_op[mask]),
        jnp.asarray(dataset.second_op[mask]),
    )
    labels = phase_to_label(_queried(phases, jnp.asarray(dataset.query[mask])))
    return np.asarray(labels, dtype=np.int64)


def train(dataset: SymbolicDataset, seed: int, epochs: int, learning_rate: float) -> _Params:
    """Train the gates on the training items with full-batch Adam.

    The loss is the mean circular distance ``1 − cos(φ_q − label·π/2)`` of the
    queried register's final phase; nothing but the final queried value is
    supervised.

    Parameters
    ----------
    dataset
        The frozen split; only ``~is_test`` items are used.
    seed
        Initialisation seed.
    epochs
        Adam steps.
    learning_rate
        Adam step size.

    Returns
    -------
    dict
        Trained gate tensors.

    """
    train_mask = ~dataset.is_test
    states = jnp.asarray(dataset.states[train_mask])
    first_op = jnp.asarray(dataset.first_op[train_mask])
    second_op = jnp.asarray(dataset.second_op[train_mask])
    query = jnp.asarray(dataset.query[train_mask])
    target = jnp.asarray(dataset.label[train_mask]).astype(jnp.float32) * PHASE_STEP
    params = init_gates(seed)

    def loss(p: _Params) -> jax.Array:
        phases = _queried(final_phases(p, states, first_op, second_op), query)
        return jnp.mean(1.0 - jnp.cos(phases - target))

    @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
    def optimise() -> _Params:
        return _adam_descent(params, loss, lr=learning_rate, epochs=epochs)

    return cast(_Params, optimise())
