# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for KYMA v2 task encoding and compositional split
"""Tests for clusters, gating masks, encoding, and the held-out split."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.benchmarks.kyma_v2 import task


def test_readout_node_is_outside_every_cluster() -> None:
    all_cluster_osc = {o for c in task.CLUSTERS for o in c}
    assert task.READOUT_OSCILLATOR not in all_cluster_osc
    assert task.N_OSC == task.N_CLUSTER_OSC + 1
    assert task.READOUT_OSCILLATOR == task.N_CLUSTER_OSC


def test_pairs_are_the_six_disjoint_free_cluster_pairs() -> None:
    assert task.N_PAIRS == 6
    assert task.PAIRS[task.HELD_OUT_R1_PAIR] == (0, 1)
    assert task.PAIRS[task.HELD_OUT_R2_PAIR] == (2, 3)


def test_pair_members_are_the_two_clusters() -> None:
    members = task.pair_members(0)  # clusters A + B
    assert set(members.tolist()) == {0, 1, 2, 3, 4, 5, 6, 7}


def test_in_phase_mask_is_symmetric_zero_diagonal_over_members() -> None:
    mask = task.in_phase_mask(0)
    assert mask.shape == (task.N_OSC, task.N_OSC)
    assert np.array_equal(mask, mask.T)
    assert not mask.diagonal().any()
    # every intra-pair edge present; no edge outside the pair
    members = set(task.pair_members(0).tolist())
    for i in range(task.N_OSC):
        for j in range(task.N_OSC):
            if i != j and i in members and j in members:
                assert mask[i, j]
            elif not (i in members and j in members):
                assert not mask[i, j]


def test_anti_phase_masks_partition_the_pair_edges() -> None:
    within, between = task.anti_phase_masks(0)  # clusters A, B
    both = task.in_phase_mask(0)
    # within ∪ between == all pair edges, and they are disjoint
    assert np.array_equal(within | between, both)
    assert not (within & between).any()
    # within couples inside a cluster only; between couples across the two clusters
    assert within[0, 1] and not within[0, 4]
    assert between[0, 4] and not between[0, 1]


def test_disjoint_conjunctions_are_all_complementary_pairs() -> None:
    conj = task.disjoint_conjunctions()
    for r1, r2 in conj:
        assert not (set(task.PAIRS[r1]) & set(task.PAIRS[r2]))
    assert (task.HELD_OUT_R1_PAIR, task.HELD_OUT_R2_PAIR) in conj


def test_encode_sets_one_hot_rows() -> None:
    code = task.encode(2, 5)
    assert code.shape == (task.N_RELATIONS, task.N_PAIRS)
    assert code[0, 2] == 1.0 and code[1, 5] == 1.0
    assert code.sum() == 2.0
    # single-relation: −1 leaves that row empty
    assert task.encode(3, -1).sum() == 1.0


def test_build_trials_holds_out_only_the_target_conjunction() -> None:
    cfg = task.ProbeConfigV2()
    batch = task.build_trials(cfg, seed=0)
    # test trials are exactly the held-out conjunction
    test_r1 = batch.r1_pair[batch.is_test]
    test_r2 = batch.r2_pair[batch.is_test]
    assert set(test_r1.tolist()) == {cfg.held_out[0]}
    assert set(test_r2.tolist()) == {cfg.held_out[1]}
    assert int(batch.is_test.sum()) == cfg.test_trials
    # the held-out conjunction never appears in training
    train = ~batch.is_test
    joint = (batch.r1_pair[train] == cfg.held_out[0]) & (batch.r2_pair[train] == cfg.held_out[1])
    assert not joint.any()


def test_build_trials_contains_both_single_relations_of_the_held_out() -> None:
    cfg = task.ProbeConfigV2()
    batch = task.build_trials(cfg, seed=0)
    train = ~batch.is_test
    r1_alone = (batch.r1_pair[train] == cfg.held_out[0]) & (batch.r2_pair[train] < 0)
    r2_alone = (batch.r2_pair[train] == cfg.held_out[1]) & (batch.r1_pair[train] < 0)
    assert r1_alone.any() and r2_alone.any()


def test_build_trials_shapes_and_determinism() -> None:
    cfg = task.ProbeConfigV2()
    a = task.build_trials(cfg, seed=3)
    b = task.build_trials(cfg, seed=3)
    assert a.theta0.shape == (len(a), task.N_OSC)
    assert np.array_equal(a.theta0, b.theta0)
    assert not np.array_equal(a.theta0, task.build_trials(cfg, seed=4).theta0)
