# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for KYMA v2 models (student, MLP, chance)
"""Tests for the gated student, the parameter-matched MLP, and the chance floor.

Kept small (tiny config, few epochs) — these exercise the training mechanics, not
the frozen research budget.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from scpn_quantum_control.benchmarks.kyma_v2 import models, task, teacher  # noqa: E402


def _tiny_cfg() -> task.ProbeConfigV2:
    return task.ProbeConfigV2(
        g_sync=0.5,
        steps=20,
        dt=0.1,
        k_bridge=0.8,
        trials_per_single=2,
        trials_per_conjunction=2,
        test_trials=24,
    )


def test_substrate_param_count() -> None:
    assert (
        models.substrate_param_count() == task.N_RELATIONS * task.N_PAIRS * models.EDGES_PER_PAIR
    )
    assert models.EDGES_PER_PAIR == 28


def test_mlp_param_match_within_ten_percent() -> None:
    target = models.substrate_param_count()
    h = models.mlp_hidden_for_match(target, n_bins=4)
    count = models.mlp_param_count(h, n_bins=4)
    assert abs(count - target) / target <= 0.10


def test_student_gates_symmetric_zero_diagonal() -> None:
    params = models.student_init(0)
    assert params["gates_free"].shape == (task.N_RELATIONS, task.N_PAIRS, models.EDGES_PER_PAIR)
    gates = np.asarray(models._student_gates(params["gates_free"]))
    assert np.allclose(gates, np.swapaxes(gates, -1, -2))
    assert np.allclose(np.diagonal(gates, axis1=-2, axis2=-1), 0.0)


def test_student_gates_are_masked_to_pair_edges() -> None:
    # A student gate for (relation, pair p) must be zero outside pair p's edges.
    free = np.zeros((task.N_RELATIONS, task.N_PAIRS, models.EDGES_PER_PAIR))
    free[0, 0, :] = 1.0  # activate all edges of pair 0's motif
    gates = np.asarray(models._student_gates(jnp.asarray(free)))
    outside = ~task.in_phase_mask(0)
    assert np.allclose(gates[0, 0][outside], 0.0)
    assert np.all(gates[0, 0][task.in_phase_mask(0)] != 0.0)


def test_student_trains_and_predicts_in_range() -> None:
    cfg = _tiny_cfg()
    batch = task.build_trials(cfg, seed=0)
    finals = np.asarray(teacher.teacher_final_phases(batch.theta0, batch.code, cfg))
    params = models.train_student(batch, finals, cfg, seed=0, epochs=50)
    test = task.TrialBatchV2(
        batch.theta0[batch.is_test],
        batch.code[batch.is_test],
        batch.r1_pair[batch.is_test],
        batch.r2_pair[batch.is_test],
        batch.is_test[batch.is_test],
    )
    pred = models.student_predict(params, test, cfg)
    assert pred.shape == (int(batch.is_test.sum()),)
    assert pred.min() >= 0 and pred.max() < cfg.n_bins


def test_student_loss_decreases() -> None:
    cfg = _tiny_cfg()
    batch = task.build_trials(cfg, seed=0)
    finals = jnp.asarray(np.asarray(teacher.teacher_final_phases(batch.theta0, batch.code, cfg)))
    tr = ~batch.is_test
    theta0, code = jnp.asarray(batch.theta0[tr]), jnp.asarray(batch.code[tr])
    base = jnp.asarray(models.base_coupling_matrix(cfg.k_ambient, cfg.k_bridge))
    target = finals[tr]

    def loss(p: dict) -> float:
        pred = models.student_final_phases(p, theta0, code, base, cfg)
        return float(models._circular_loss(pred, target))

    init = models.student_init(0)
    trained = models.train_student(batch, np.asarray(finals), cfg, seed=0, epochs=200)
    assert loss(trained) < loss(init)


def test_mlp_trains_and_predicts_in_range() -> None:
    cfg = _tiny_cfg()
    batch = task.build_trials(cfg, seed=0)
    labels = teacher.label_batch(batch, cfg)
    hidden = models.mlp_hidden_for_match(models.substrate_param_count(), cfg.n_bins)
    params = models.train_mlp(batch, labels, hidden, cfg, seed=0, epochs=50)
    test = task.TrialBatchV2(
        batch.theta0[batch.is_test],
        batch.code[batch.is_test],
        batch.r1_pair[batch.is_test],
        batch.r2_pair[batch.is_test],
        batch.is_test[batch.is_test],
    )
    pred = models.mlp_predict(params, test)
    assert pred.shape == (int(batch.is_test.sum()),)
    assert pred.min() >= 0 and pred.max() < cfg.n_bins


def test_chance_floor_is_majority_class_accuracy() -> None:
    labels = np.array([0, 0, 0, 1, 2, 2, 3])  # train majority = 0 among train slots
    is_test = np.array([False, False, True, False, True, True, True])
    # train labels = [0,0,1] → majority 0; test labels = [0,2,2,3] → acc(pred 0) = 1/4
    acc = models.chance_floor_accuracy(labels, is_test, n_bins=4)
    assert acc == pytest.approx(0.25)
