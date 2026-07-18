# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the KYMA models
"""Structural and contract tests for the motif substrate, MLP, and chance floor."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")  # KYMA probe requires the optional [jax] extra, absent from the CI lane


import jax.numpy as jnp
import numpy as np

from scpn_quantum_control.benchmarks.kyma import models
from scpn_quantum_control.benchmarks.kyma.task import (
    N_OSC,
    ProbeConfig,
    build_trials,
)

_TINY = ProbeConfig(steps=10, trials_per_single=2, trials_per_conjunction=2, test_trials=8)


def test_symmetric_zero_diagonal_coupling() -> None:
    k_raw = jnp.asarray(np.arange(models._N_K, dtype=float))
    coupling = np.asarray(models._coupling(k_raw))
    assert np.allclose(coupling, coupling.T)
    assert np.allclose(np.diag(coupling), 0.0)


def test_mlp_param_match_within_ten_percent() -> None:
    target = models.substrate_param_count()
    hidden = models.mlp_hidden_for_match(target)
    count = models.mlp_param_count(hidden)
    assert abs(count - target) / target <= 0.10


def test_drive_is_additive_over_active_relations() -> None:
    drive = jnp.asarray(np.arange(2 * models._N_PAIRS * N_OSC, dtype=float)).reshape(
        2, models._N_PAIRS, N_OSC
    )
    from scpn_quantum_control.benchmarks.kyma.task import encode

    code_r1 = jnp.asarray(encode(0, -1)[None])
    code_r2 = jnp.asarray(encode(-1, 5)[None])
    code_both = jnp.asarray(encode(0, 5)[None])
    w1 = models._drive_for(drive, code_r1)
    w2 = models._drive_for(drive, code_r2)
    wboth = models._drive_for(drive, code_both)
    assert np.allclose(np.asarray(wboth), np.asarray(w1 + w2))


def test_mlp_forward_in_unit_interval() -> None:
    batch = build_trials(_TINY, seed=0)
    params = models.mlp_init(0, hidden=4)
    feats = models._mlp_features(batch.theta0, batch.code)
    pred = np.asarray(models.mlp_forward(params, feats))
    assert np.all(pred >= 0.0) and np.all(pred <= 1.0)


def test_substrate_readout_shapes_and_bounds() -> None:
    batch = build_trials(_TINY, seed=1)
    params = models.substrate_init(1)
    r1m, r2m, _, _ = models._member_tables(batch)
    r1, r2 = models.substrate_readout(
        params, jnp.asarray(batch.theta0), jnp.asarray(batch.code), r1m, r2m, _TINY
    )
    for r in (np.asarray(r1), np.asarray(r2)):
        assert r.shape == (len(batch),)
        assert np.all(r >= -1e-9) and np.all(r <= 1.0 + 1e-9)


def test_substrate_training_reduces_loss() -> None:
    batch = build_trials(_TINY, seed=2)
    train = ~batch.is_test
    from scpn_quantum_control.benchmarks.kyma.task import TrialBatch

    trb = TrialBatch(
        batch.theta0[train],
        batch.code[train],
        batch.r1_pair[train],
        batch.r2_pair[train],
        batch.is_test[train],
    )
    r1m, r2m, r1a, r2a = models._member_tables(trb)
    args = (jnp.asarray(trb.theta0), jnp.asarray(trb.code), r1m, r2m, r1a, r2a, _TINY)
    init = models.substrate_init(2)
    trained = models.train_substrate(batch, _TINY, seed=2, epochs=30)
    loss0 = float(models._substrate_loss(init, *args))
    loss1 = float(models._substrate_loss(trained, *args))
    assert loss1 < loss0


def test_chance_floor_is_low_and_measured() -> None:
    batch = build_trials(_TINY, seed=0)
    floor = models.chance_floor_accuracy(batch, epsilon=0.15, seed=0)
    # Random (R1,R2)~U[0,1]²; success needs both in a 0.15-wide band → ~0.0225.
    assert 0.0 <= floor <= 0.15


def test_substrate_training_deterministic_for_seed() -> None:
    batch = build_trials(_TINY, seed=4)
    a = models.train_substrate(batch, _TINY, seed=4, epochs=20)
    b = models.train_substrate(batch, _TINY, seed=4, epochs=20)
    assert np.allclose(np.asarray(a["k_raw"]), np.asarray(b["k_raw"]))
