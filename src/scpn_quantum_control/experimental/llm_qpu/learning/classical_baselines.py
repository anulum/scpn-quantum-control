# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU classical baselines
"""Strong classical baselines tuned by group cross-validation on train and scored once on dev."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .feature_maps import (
    fit_standardiser,
    random_fourier_features,
    sin_cos_basis,
    trig_product_basis,
)
from .logistic_readout import fit_logistic, mean_log_loss

_FloatArray = np.ndarray[Any, np.dtype[np.float64]]
_IntArray = np.ndarray[Any, np.dtype[np.int64]]

FAMILIES = (
    "latent_logistic",
    "compressed_logistic",
    "sin_cos_encoding",
    "trig_product_basis",
    "random_fourier",
)
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
BANDWIDTH_GRID = (0.5, 1.0, 2.0)


@dataclass(frozen=True, slots=True)
class LabelledSplit:
    """One split's rows: identities, source groups, original latents, compressed angles and labels."""

    sample_ids: tuple[str, ...]
    groups: tuple[str, ...]
    latents: _FloatArray
    angles: _FloatArray
    labels: _IntArray

    def __post_init__(self) -> None:
        """Refuse misaligned rows, duplicated samples and non-finite inputs."""
        rows = len(self.sample_ids)
        if not rows or len(set(self.sample_ids)) != rows or len(self.groups) != rows:
            raise ValueError("a split needs unique sample IDs with one group each")
        for name in ("latents", "angles"):
            array = getattr(self, name)
            if array.ndim != 2 or array.shape[0] != rows or not np.isfinite(array).all():
                raise ValueError(f"{name} must be one finite row per sample")
        if self.labels.shape != (rows,):
            raise ValueError("labels must be one integer per sample")


@dataclass(frozen=True, slots=True)
class DevOutcome:
    """One dev sample's log-loss and whether the most probable class was right."""

    sample_id: str
    source_group: str
    log_loss: float
    correct: bool


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """One family's tuned readout, its budget and its dev outcomes."""

    family: str
    feature_dim: int
    parameter_count: int
    l2: float
    bandwidth: float | None
    fits: int
    fit_seconds: float
    predict_seconds: float
    converged: bool
    outcomes: tuple[DevOutcome, ...]
    mean_log_loss: float
    accuracy: float


def family_features(
    family: str, split: LabelledSplit, *, bandwidth: float | None, dimension: int, seed: int
) -> _FloatArray:
    """Return a family's raw features of a split before train-fitted standardisation."""
    if family == "latent_logistic":
        return split.latents
    if family == "compressed_logistic":
        return split.angles
    if family == "sin_cos_encoding":
        return sin_cos_basis(split.angles)
    if family == "trig_product_basis":
        return trig_product_basis(split.angles)
    if family == "random_fourier" and bandwidth is not None:
        return random_fourier_features(
            split.angles, dimension=dimension, bandwidth=bandwidth, seed=seed
        )
    raise ValueError("unknown baseline family")


def group_folds(groups: tuple[str, ...], folds: int) -> tuple[tuple[_IntArray, _IntArray], ...]:
    """Return (fit rows, held rows) per fold; every source group falls wholly in one fold."""
    names = sorted(set(groups))
    if type(folds) is not int or not 2 <= folds <= len(names):
        raise ValueError("folds must be between two and the number of source groups")
    fold_of = {name: index % folds for index, name in enumerate(names)}
    assigned = np.array([fold_of[group] for group in groups])
    return tuple(
        (np.flatnonzero(assigned != fold), np.flatnonzero(assigned == fold))
        for fold in range(folds)
    )


def _cv_loss(
    features: _FloatArray,
    labels: _IntArray,
    splits: tuple[tuple[_IntArray, _IntArray], ...],
    classes: int,
    l2: float,
) -> float:
    """Return the mean held-fold log-loss with a standardiser fitted on each fold's fit rows."""
    losses = []
    for fit_rows, held_rows in splits:
        scaler = fit_standardiser(features[fit_rows])
        readout = fit_logistic(
            scaler.apply(features[fit_rows]), labels[fit_rows], classes=classes, l2=l2
        )
        losses.append(mean_log_loss(readout, scaler.apply(features[held_rows]), labels[held_rows]))
    return float(np.mean(losses))


def _tune(
    features_for: Callable[[float | None], _FloatArray],
    bandwidths: tuple[float | None, ...],
    train: LabelledSplit,
    classes: int,
    folds: int,
) -> tuple[float, float | None, int]:
    """Return the best (l2, bandwidth) by train group cross-validation and the fits it took."""
    splits = group_folds(train.groups, folds)
    scored = []
    for bandwidth in bandwidths:
        features = features_for(bandwidth)
        for l2 in L2_GRID:
            scored.append((_cv_loss(features, train.labels, splits, classes, l2), l2, bandwidth))
    _, l2, bandwidth = min(scored, key=lambda item: item[0])
    return l2, bandwidth, len(scored) * folds


def _refit_and_score(
    name: str,
    raw_train: _FloatArray,
    raw_dev: _FloatArray,
    train: LabelledSplit,
    dev: LabelledSplit,
    choice: tuple[int, float, float | None, int],
) -> BaselineResult:
    """Refit the chosen readout on all train rows with a train-only standardiser and score each dev row."""
    classes, l2, bandwidth, fits = choice
    started = time.perf_counter()
    scaler = fit_standardiser(raw_train)
    readout = fit_logistic(scaler.apply(raw_train), train.labels, classes=classes, l2=l2)
    fitted = time.perf_counter()
    logp = readout.log_probabilities(scaler.apply(raw_dev))
    predicted = time.perf_counter()
    losses = -logp[np.arange(len(dev.labels)), dev.labels]
    correct = logp.argmax(axis=1) == dev.labels
    outcomes = tuple(
        DevOutcome(sample, group, float(loss), bool(hit))
        for sample, group, loss, hit in zip(
            dev.sample_ids, dev.groups, losses, correct, strict=True
        )
    )
    return BaselineResult(
        name,
        raw_train.shape[1],
        readout.parameter_count,
        l2,
        bandwidth,
        fits + 1,
        fitted - started,
        predicted - fitted,
        readout.converged,
        outcomes,
        float(losses.mean()),
        float(correct.mean()),
    )


def _check_budget(grid: int, folds: int, max_fits: int) -> None:
    """Refuse a tuning grid whose fits, plus the final refit, exceed the declared budget."""
    if grid * folds + 1 > max_fits:
        raise ValueError("the tuning grid exceeds the declared fit budget")


def run_baseline(
    family: str,
    train: LabelledSplit,
    dev: LabelledSplit,
    *,
    classes: int,
    max_fits: int,
    folds: int = 4,
    dimension: int = 81,
    seed: int = 0,
) -> BaselineResult:
    """Tune on train only, refit on all train rows, then score every dev row once."""
    bandwidths: tuple[float | None, ...] = (
        BANDWIDTH_GRID if family == "random_fourier" else (None,)
    )
    _check_budget(len(L2_GRID) * len(bandwidths), folds, max_fits)

    def features_for(bandwidth: float | None) -> _FloatArray:
        """Return the family's train features at one bandwidth."""
        return family_features(family, train, bandwidth=bandwidth, dimension=dimension, seed=seed)

    l2, bandwidth, fits = _tune(features_for, bandwidths, train, classes, folds)
    raw_train = features_for(bandwidth)
    raw_dev = family_features(family, dev, bandwidth=bandwidth, dimension=dimension, seed=seed)
    return _refit_and_score(family, raw_train, raw_dev, train, dev, (classes, l2, bandwidth, fits))


def run_feature_readout(
    name: str,
    train_features: object,
    dev_features: object,
    train: LabelledSplit,
    dev: LabelledSplit,
    *,
    classes: int,
    max_fits: int,
    folds: int = 4,
) -> BaselineResult:
    """Tune, refit and score the same readout on given per-row features, for example quantum features."""
    raw_train = np.array(train_features, dtype=np.float64)
    raw_dev = np.array(dev_features, dtype=np.float64)
    for features, split in ((raw_train, train), (raw_dev, dev)):
        if (
            features.ndim != 2
            or features.shape[0] != len(split.sample_ids)
            or not np.isfinite(features).all()
        ):
            raise ValueError("given features must be one finite row per sample")
    if raw_train.shape[1] != raw_dev.shape[1]:
        raise ValueError("train and dev features must have one width")
    _check_budget(len(L2_GRID), folds, max_fits)
    l2, _, fits = _tune(lambda _: raw_train, (None,), train, classes, folds)
    return _refit_and_score(name, raw_train, raw_dev, train, dev, (classes, l2, None, fits))


def run_baseline_suite(
    train: LabelledSplit,
    dev: LabelledSplit,
    *,
    classes: int,
    max_fits_per_family: int,
    seed: int,
    rff_dimensions: tuple[int, ...] = (21, 81),
) -> tuple[BaselineResult, ...]:
    """Run every family; random Fourier features run once per dimension (21 = n4 quantum, 81 = trig basis)."""
    fixed = tuple(
        run_baseline(family, train, dev, classes=classes, max_fits=max_fits_per_family, seed=seed)
        for family in FAMILIES
        if family != "random_fourier"
    )
    fourier = tuple(
        run_baseline(
            "random_fourier",
            train,
            dev,
            classes=classes,
            max_fits=max_fits_per_family,
            dimension=dimension,
            seed=seed,
        )
        for dimension in rff_dimensions
    )
    return fixed + fourier
