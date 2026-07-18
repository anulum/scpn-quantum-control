# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the KYMA task encoding and split
"""The compositional split holds out the target conjunction and nothing else."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")  # KYMA probe requires the optional [jax] extra, absent from the CI lane


import numpy as np

from scpn_quantum_control.benchmarks.kyma.task import (
    CLUSTERS,
    N_OSC,
    PAIRS,
    ProbeConfig,
    build_trials,
    disjoint_conjunctions,
    encode,
    pair_members,
    success_mask,
)


def test_cluster_partition_covers_all_oscillators_disjointly() -> None:
    flat = [q for cluster in CLUSTERS for q in cluster]
    assert sorted(flat) == list(range(N_OSC))
    assert len(set(flat)) == N_OSC


def test_six_pairs_and_six_disjoint_conjunctions() -> None:
    assert len(PAIRS) == 6
    conj = disjoint_conjunctions()
    assert len(conj) == 6
    # Each conjunction's two cluster pairs must be disjoint.
    for r1, r2 in conj:
        assert not (set(PAIRS[r1]) & set(PAIRS[r2]))


def test_pair_members_are_eight_distinct_oscillators() -> None:
    members = pair_members(0)
    assert members.shape == (8,)
    assert len(set(members.tolist())) == 8


def test_encode_marks_active_relation_slots() -> None:
    code = encode(0, 5)
    assert code[0, 0] == 1.0 and code[1, 5] == 1.0
    assert code.sum() == 2.0
    single = encode(2, -1)
    assert single[0, 2] == 1.0 and single.sum() == 1.0


def test_held_out_conjunction_is_test_only_and_absent_from_training() -> None:
    cfg = ProbeConfig()
    batch = build_trials(cfg, seed=0)
    train = ~batch.is_test
    # Held-out conjunction never appears jointly in training.
    joint_train = [
        (int(r1), int(r2))
        for r1, r2 in zip(batch.r1_pair[train], batch.r2_pair[train])
        if r1 >= 0 and r2 >= 0
    ]
    assert cfg.held_out not in joint_train
    # Every test trial IS the held-out conjunction.
    test = batch.is_test
    assert np.all(batch.r1_pair[test] == cfg.held_out[0])
    assert np.all(batch.r2_pair[test] == cfg.held_out[1])


def test_constituent_single_relations_are_in_training() -> None:
    cfg = ProbeConfig()
    batch = build_trials(cfg, seed=3)
    train = ~batch.is_test
    r1_singles = {
        int(p) for p, q in zip(batch.r1_pair[train], batch.r2_pair[train]) if p >= 0 and q < 0
    }
    r2_singles = {
        int(q) for p, q in zip(batch.r1_pair[train], batch.r2_pair[train]) if q >= 0 and p < 0
    }
    assert cfg.held_out[0] in r1_singles  # R1-on-P* seen alone
    assert cfg.held_out[1] in r2_singles  # R2-on-Q* seen alone


def test_success_mask_frozen_criterion() -> None:
    r1 = np.array([1.0, 0.9, 0.8, 1.0])
    r2 = np.array([0.0, 0.1, 0.05, 0.2])
    mask = success_mask(r1, r2, epsilon=0.15)
    # (1.0, 0.0) pass; (0.9, 0.1) pass; (0.8, 0.05) fail R1; (1.0, 0.2) fail R2.
    assert mask.tolist() == [True, True, False, False]


def test_determinism_same_seed_same_trials() -> None:
    cfg = ProbeConfig()
    a = build_trials(cfg, seed=7)
    b = build_trials(cfg, seed=7)
    assert np.array_equal(a.theta0, b.theta0)
    assert np.array_equal(a.code, b.code)
