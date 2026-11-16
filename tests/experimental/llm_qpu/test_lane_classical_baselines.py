# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental classical baselines
"""Baselines tune on train groups only, keep a declared fit budget and score each dev row once."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.experimental.llm_qpu.learning.classical_baselines import (
    BANDWIDTH_GRID,
    FAMILIES,
    L2_GRID,
    LabelledSplit,
    _cv_loss,
    family_features,
    group_folds,
    run_baseline,
    run_baseline_suite,
)


def _split(rows: int, tag: str, seed: int) -> LabelledSplit:
    """Return a synthetic split whose labels depend on a product of angle functions."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-math.pi, math.pi, size=(rows, 4))
    labels = (np.sin(angles[:, 0]) * np.cos(angles[:, 1]) > 0).astype(np.int64) * 2 + (
        np.cos(angles[:, 2]) > 0
    ).astype(np.int64)
    latents = rng.normal(size=(rows, 16))
    latents[:, :4] += np.eye(4)[labels] * 2.0
    return LabelledSplit(
        tuple(f"{tag}-{index:03d}" for index in range(rows)),
        tuple(f"{tag}-group-{index // 2:03d}" for index in range(rows)),
        latents,
        angles,
        labels,
    )


_TRAIN, _DEV = _split(96, "train", 1), _split(32, "dev", 2)


def test_the_suite_runs_every_family_with_its_budget_and_dev_outcomes() -> None:
    """Six results in family order; fits, widths and parameters follow the declared grids."""
    results = run_baseline_suite(_TRAIN, _DEV, classes=4, max_fits_per_family=80, seed=5)
    assert [result.family for result in results] == [
        *FAMILIES[:-1],
        "random_fourier",
        "random_fourier",
    ]
    widths = [result.feature_dim for result in results]
    assert widths == [16, 4, 8, 81, 21, 81]
    for result in results:
        assert result.parameter_count == (result.feature_dim + 1) * 4
        grid = len(L2_GRID) * (len(BANDWIDTH_GRID) if result.family == "random_fourier" else 1)
        assert result.fits == grid * 4 + 1
        assert result.l2 in L2_GRID and result.converged
        assert (result.bandwidth in BANDWIDTH_GRID) == (result.family == "random_fourier")
        assert [outcome.sample_id for outcome in result.outcomes] == list(_DEV.sample_ids)
        losses = [outcome.log_loss for outcome in result.outcomes]
        assert result.mean_log_loss == pytest.approx(float(np.mean(losses)))
        assert result.accuracy == pytest.approx(
            np.mean([outcome.correct for outcome in result.outcomes])
        )
        assert result.fit_seconds >= 0.0 and result.predict_seconds >= 0.0
    by_family = {(r.family, r.feature_dim): r.mean_log_loss for r in results}
    assert by_family[("trig_product_basis", 81)] < by_family[("compressed_logistic", 4)]


def test_tuning_never_reads_dev_labels() -> None:
    """Permuting dev labels changes the dev scores but not a single tuned hyperparameter."""
    permuted = replace(_DEV, labels=_DEV.labels[::-1].copy())
    for family in FAMILIES:
        first = run_baseline(family, _TRAIN, _DEV, classes=4, max_fits=80, dimension=21, seed=5)
        second = run_baseline(
            family, _TRAIN, permuted, classes=4, max_fits=80, dimension=21, seed=5
        )
        assert (first.l2, first.bandwidth) == (second.l2, second.bandwidth)
        assert [o.log_loss for o in first.outcomes] != [o.log_loss for o in second.outcomes]


def test_tuning_picks_the_train_cross_validation_minimum_and_standardises_on_train() -> None:
    """The chosen l2 minimises the train-CV loss; a dev row's score never depends on other dev rows."""
    chosen = run_baseline("latent_logistic", _TRAIN, _DEV, classes=4, max_fits=80, seed=5)
    splits = group_folds(_TRAIN.groups, 4)
    losses = {l2: _cv_loss(_TRAIN.latents, _TRAIN.labels, splits, 4, l2) for l2 in L2_GRID}
    assert chosen.l2 == min(losses, key=lambda l2: losses[l2])
    half = LabelledSplit(
        _DEV.sample_ids[:16],
        _DEV.groups[:16],
        _DEV.latents[:16],
        _DEV.angles[:16],
        _DEV.labels[:16],
    )
    alone = run_baseline("latent_logistic", _TRAIN, half, classes=4, max_fits=80, seed=5)
    assert [o.log_loss for o in alone.outcomes] == pytest.approx(
        [o.log_loss for o in chosen.outcomes[:16]], abs=1e-12
    )


def test_group_folds_keep_each_source_group_in_one_fold() -> None:
    """Every row of a group shares one held fold, and every row is held exactly once."""
    folds = group_folds(_TRAIN.groups, 4)
    held = np.concatenate([held_rows for _, held_rows in folds])
    assert sorted(held.tolist()) == list(range(len(_TRAIN.groups)))
    for fit_rows, held_rows in folds:
        held_groups = {_TRAIN.groups[row] for row in held_rows}
        assert not held_groups & {_TRAIN.groups[row] for row in fit_rows}
    for folds_count in (1, len(set(_TRAIN.groups)) + 1):
        with pytest.raises(
            ValueError, match="^folds must be between two and the number of source groups$"
        ):
            group_folds(_TRAIN.groups, folds_count)


def test_refusals_cover_the_budget_unknown_families_and_misaligned_splits() -> None:
    """A grid larger than the budget, an unknown family and malformed splits are refused."""
    with pytest.raises(ValueError, match="^the tuning grid exceeds the declared fit budget$"):
        run_baseline("random_fourier", _TRAIN, _DEV, classes=4, max_fits=72)
    assert (
        run_baseline("random_fourier", _TRAIN, _DEV, classes=4, max_fits=73, dimension=21).fits
        == 73
    )
    for family, bandwidth in (("kernel_ridge", None), ("random_fourier", None)):
        with pytest.raises(ValueError, match="^unknown baseline family$"):
            family_features(family, _TRAIN, bandwidth=bandwidth, dimension=21, seed=0)
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"sample_ids": ()}, "^a split needs unique sample IDs with one group each$"),
        ({"sample_ids": ("a",) * 32}, "^a split needs unique sample IDs with one group each$"),
        ({"groups": ("g",)}, "^a split needs unique sample IDs with one group each$"),
        ({"latents": np.zeros((31, 16))}, "^latents must be one finite row per sample$"),
        ({"angles": np.full((32, 4), np.nan)}, "^angles must be one finite row per sample$"),
        ({"angles": np.zeros(32)}, "^angles must be one finite row per sample$"),
        ({"labels": np.zeros(31, dtype=np.int64)}, "^labels must be one integer per sample$"),
    )
    for changes, reason in cases:
        with pytest.raises(ValueError, match=reason):
            replace(_DEV, **changes)
