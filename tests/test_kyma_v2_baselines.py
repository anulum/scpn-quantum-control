# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for KYMA v2.1 stronger baselines (deep MLP, GNN)
"""Tests for the deep-MLP capacity control and the code-conditioned GNN."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from scpn_quantum_control.benchmarks.kyma_v2 import baselines, task, teacher  # noqa: E402


def _tiny_cfg() -> task.ProbeConfigV2:
    return task.ProbeConfigV2(
        g_sync=0.5,
        steps=20,
        dt=0.1,
        k_bridge=0.8,
        trials_per_single=3,
        trials_per_conjunction=3,
        test_trials=24,
    )


def _test_slice(batch: task.TrialBatchV2) -> task.TrialBatchV2:
    m = batch.is_test
    return task.TrialBatchV2(
        batch.theta0[m], batch.code[m], batch.r1_pair[m], batch.r2_pair[m], batch.is_test[m]
    )


def test_deep_mlp_param_count_matches_init() -> None:
    params = baselines.deep_mlp_init(0, 24, 4)
    counted = sum(int(np.asarray(v).size) for v in params.values())
    assert counted == baselines.deep_mlp_param_count(24, 4)


def test_gnn_param_count_matches_init() -> None:
    params = baselines.gnn_init(0, 8, 4)
    counted = sum(int(np.asarray(v).size) for v in params.values())
    assert counted == baselines.gnn_param_count(8, 4)


def test_deep_mlp_logits_shape() -> None:
    cfg = _tiny_cfg()
    batch = task.build_trials(cfg, 0)
    feats = baselines._mlp_features(batch.theta0, batch.code)
    params = baselines.deep_mlp_init(0, 16, cfg.n_bins)
    logits = np.asarray(baselines.deep_mlp_logits(params, feats))
    assert logits.shape == (len(batch), cfg.n_bins)


def test_gnn_adjacency_is_code_conditioned() -> None:
    import jax.numpy as jnp

    from scpn_quantum_control.benchmarks.kyma_v2 import coupling

    cfg = _tiny_cfg()
    params = baselines.gnn_init(0, 8, cfg.n_bins)
    base = jnp.asarray(coupling.base_coupling_matrix(cfg.k_ambient, cfg.k_bridge))
    code_a = jnp.asarray(task.encode(0, 5)[None])
    code_b = jnp.asarray(task.encode(1, 4)[None])
    adj_a = np.asarray(baselines._gnn_adjacency(params, code_a, base))
    adj_b = np.asarray(baselines._gnn_adjacency(params, code_b, base))
    # different active pairs → different learned adjacency
    assert not np.allclose(adj_a, adj_b)


def test_deep_mlp_trains_and_predicts_in_range() -> None:
    cfg = _tiny_cfg()
    batch = task.build_trials(cfg, 0)
    labels = teacher.label_batch(batch, cfg)
    params = baselines.train_deep_mlp(batch, labels, 16, cfg, 0, epochs=40)
    pred = baselines.deep_mlp_predict(params, _test_slice(batch))
    assert pred.shape == (int(batch.is_test.sum()),)
    assert pred.min() >= 0 and pred.max() < cfg.n_bins


def test_gnn_trains_and_predicts_in_range() -> None:
    cfg = _tiny_cfg()
    batch = task.build_trials(cfg, 0)
    labels = teacher.label_batch(batch, cfg)
    params = baselines.train_gnn(batch, labels, 8, cfg, 0, epochs=40)
    pred = baselines.gnn_predict(params, _test_slice(batch), cfg)
    assert pred.shape == (int(batch.is_test.sum()),)
    assert pred.min() >= 0 and pred.max() < cfg.n_bins


def test_softmax_cross_entropy_matches_manual() -> None:
    import jax.numpy as jnp

    logits = jnp.asarray([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]])
    labels = jnp.asarray([0, 3])
    ce = float(baselines._softmax_cross_entropy(logits, labels, 4))
    # manual: -mean(log softmax at the true class)
    lg = np.asarray(logits)
    logp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
    manual = -np.mean([logp[0, 0], logp[1, 3]])
    assert ce == pytest.approx(manual, rel=1e-5)
