# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for the KYMA v2 ground-truth teacher
"""Tests for the fixed gated-oscillator teacher and its labels."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from scpn_quantum_control.benchmarks.kyma_v2 import task, teacher  # noqa: E402
from scpn_quantum_control.benchmarks.kyma_v2.dynamics import order_parameter  # noqa: E402


def _cfg() -> task.ProbeConfigV2:
    return task.ProbeConfigV2(g_sync=0.5, steps=40, dt=0.1, k_bridge=0.8)


def test_teacher_gates_shape_and_symmetry() -> None:
    gates = teacher.teacher_gates(0.5)
    assert gates.shape == (task.N_RELATIONS, task.N_PAIRS, task.N_OSC, task.N_OSC)
    for r in range(task.N_RELATIONS):
        for p in range(task.N_PAIRS):
            assert np.allclose(gates[r, p], gates[r, p].T)
            assert not np.diagonal(gates[r, p]).any()


def test_r1_gate_is_attractive_intra_pair() -> None:
    gates = teacher.teacher_gates(0.5)
    mask = task.in_phase_mask(0)
    assert np.all(gates[0, 0][mask] > 0)  # attractive on every intra-pair edge


def test_r2_gate_attracts_within_and_frustrates_between() -> None:
    gates = teacher.teacher_gates(0.5)
    within, between = task.anti_phase_masks(0)
    assert np.all(gates[1, 0][within] > 0)  # attractive inside each cluster
    assert np.all(gates[1, 0][between] < 0)  # frustrated across the two clusters


def test_teacher_realises_single_relation_motifs() -> None:
    cfg = _cfg()
    batch = task.build_trials(cfg, seed=0)
    finals = teacher.teacher_final_phases(batch.theta0, batch.code, cfg)
    # R1-alone on pair 0 → in-phase (R high); R2-alone on pair 0 → anti-phase (R low)
    r1_idx = np.nonzero((batch.r1_pair == 0) & (batch.r2_pair < 0))[0][:20]
    r2_idx = np.nonzero((batch.r2_pair == 0) & (batch.r1_pair < 0))[0][:20]
    members = task.pair_members(0)
    r1_R = np.asarray(order_parameter(finals[r1_idx], members))
    r2_R = np.asarray(order_parameter(finals[r2_idx], members))
    assert r1_R.mean() > 0.9
    assert r2_R.mean() < 0.1


def test_teacher_labels_in_range_and_deterministic() -> None:
    cfg = _cfg()
    batch = task.build_trials(cfg, seed=1)
    a = teacher.label_batch(batch, cfg)
    b = teacher.label_batch(batch, cfg)
    assert a.shape == (len(batch),)
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < cfg.n_bins


def test_teacher_label_depends_on_theta0() -> None:
    # Same code, different θ0 → labels not all identical (data-dependent readout).
    cfg = _cfg()
    code = task.encode(0, 5)[None].repeat(64, axis=0)
    rng = np.random.default_rng(2)
    theta0 = rng.uniform(-np.pi, np.pi, size=(64, task.N_OSC))
    labels = teacher.teacher_labels(theta0, code, cfg)
    assert len(set(labels.tolist())) > 1
