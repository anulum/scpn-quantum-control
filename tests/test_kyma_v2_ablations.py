# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for KYMA v2.1 ablation models
"""Tests for the shared-K (no-gating) ablation and the separable-readout setting."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from scpn_quantum_control.benchmarks.kyma_v2 import ablations, coupling, task, teacher  # noqa: E402


def _tiny_cfg(**kw: object) -> task.ProbeConfigV2:
    base = dict(
        g_sync=0.5,
        steps=20,
        dt=0.1,
        k_bridge=0.8,
        trials_per_single=3,
        trials_per_conjunction=3,
        test_trials=24,
    )
    base.update(kw)
    return task.ProbeConfigV2(**base)  # type: ignore[arg-type]


def test_shared_param_count_and_basis() -> None:
    # 120 shared cluster edges + 2*6*17 drive
    assert ablations._N_SHARED == task.N_CLUSTER_OSC * (task.N_CLUSTER_OSC - 1) // 2
    assert ablations.shared_param_count() == 120 + task.N_RELATIONS * task.N_PAIRS * task.N_OSC


def test_shared_coupling_is_symmetric_and_excludes_readout_node() -> None:
    params = ablations.shared_init(0)
    k = np.asarray(ablations._shared_coupling(params["k_shared"]))
    assert np.allclose(k, k.T)
    # the readout node (index 16) is not part of any shared cluster edge
    assert np.allclose(k[task.READOUT_OSCILLATOR, :], 0.0)
    assert np.allclose(k[:, task.READOUT_OSCILLATOR], 0.0)


def test_shared_coupling_is_the_same_for_every_trial() -> None:
    # The no-gating ablation applies ONE coupling regardless of the code; the code
    # enters only through the additive frequency drive.
    import jax.numpy as jnp

    params = ablations.shared_init(1)
    code_a = jnp.asarray(task.encode(0, 5)[None])
    code_b = jnp.asarray(task.encode(1, 4)[None])
    # coupling term is code-independent
    ca = np.asarray(ablations._shared_coupling(params["k_shared"]))
    cb = np.asarray(ablations._shared_coupling(params["k_shared"]))
    assert np.allclose(ca, cb)
    # but the drive does depend on the code
    assert not np.allclose(
        np.asarray(ablations._drive_for(params["drive"], code_a)),
        np.asarray(ablations._drive_for(params["drive"], code_b)),
    )


def test_shared_trains_and_predicts_in_range() -> None:
    cfg = _tiny_cfg()
    batch = task.build_trials(cfg, 0)
    finals = np.asarray(teacher.teacher_final_phases(batch.theta0, batch.code, cfg))
    params = ablations.train_shared(batch, finals, cfg, 0, epochs=40)
    test = task.TrialBatchV2(
        batch.theta0[batch.is_test],
        batch.code[batch.is_test],
        batch.r1_pair[batch.is_test],
        batch.r2_pair[batch.is_test],
        batch.is_test[batch.is_test],
    )
    pred = ablations.shared_predict(params, test, cfg)
    assert pred.shape == (int(batch.is_test.sum()),)
    assert pred.min() >= 0 and pred.max() < cfg.n_bins


def test_separable_readout_uses_one_partner() -> None:
    # A2: bridge_mode="r1_only" bridges the readout to a single cluster.
    both = coupling.partners_for((0, 5), "both")
    r1_only = coupling.partners_for((0, 5), "r1_only")
    assert len(both) == 2 and len(r1_only) == 1
    assert r1_only[0] == both[0]


def test_separable_readout_label_independent_of_r2() -> None:
    # With a single-relation bridge, dropping R2 should not change the readout label.
    cfg = _tiny_cfg(bridge_mode="r1_only")
    rng = np.random.default_rng(0)
    theta0 = rng.uniform(-np.pi, np.pi, size=(48, task.N_OSC))
    code_full = np.repeat(task.encode(0, 5)[None], 48, axis=0)
    code_no_r2 = code_full.copy()
    code_no_r2[:, 1, :] = 0.0
    lab_full = teacher.teacher_labels(theta0, code_full, cfg)
    lab_no_r2 = teacher.teacher_labels(theta0, code_no_r2, cfg)
    # readout bridged to R1 cluster only → R2 barely moves the readout node
    assert np.mean(lab_full == lab_no_r2) > 0.85
