# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synthetic multimodal oscillator datasets
"""Deterministic simulation-only datasets for multimodal-forecasting forecast certificates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..phase.coupling_time_series_recovery import simulate_kuramoto_phase_time_series
from .multimodal_schema import (
    DatasetSplit,
    MultimodalObservationBatch,
    SyntheticDomainTag,
    assert_disjoint_batches,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SYNTHETIC_MULTIMODAL_SOURCE = "scpn.forecasting.synthetic_kuramoto.v1"


@dataclass(frozen=True, slots=True)
class SyntheticMultimodalConfig:
    """Configuration for independent simulation-only trajectory splits.

    Parameters
    ----------
    train_samples
        Number of independent training trajectories.
    calibration_samples
        Number of independent uncertainty-calibration trajectories.
    test_samples
        Number of independent held-out test trajectories.
    history_steps
        Historical phase samples provided to the forecaster.
    horizon_steps
        Future phase samples predicted by the forecaster.
    n_nodes
        Number of oscillators. The bounded domain templates require four.
    dt
        Fixed RK4 integration step.
    missing_fraction
        Requested random missing fraction for phase/event input entries.
    seed
        Root deterministic seed; split seeds are derived without reuse.

    """

    train_samples: int = 48
    calibration_samples: int = 20
    test_samples: int = 24
    history_steps: int = 12
    horizon_steps: int = 4
    n_nodes: int = 4
    dt: float = 0.04
    missing_fraction: float = 0.20
    seed: int = 3701

    def __post_init__(self) -> None:
        """Validate bounded sample, shape, time, and missingness controls."""
        counts = (self.train_samples, self.calibration_samples, self.test_samples)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 4 for value in counts
        ):
            raise ValueError("each split must contain at least four samples")
        if self.history_steps < 3 or self.horizon_steps < 1:
            raise ValueError("history_steps must be >= 3 and horizon_steps must be positive")
        if self.n_nodes != 4:
            raise ValueError("the bounded synthetic domain templates require n_nodes=4")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(self.missing_fraction) or not 0.0 <= self.missing_fraction < 0.8:
            raise ValueError("missing_fraction must be finite and in [0, 0.8)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SyntheticMultimodalDataset:
    """Disjoint train, calibration, and test multimodal forecasting custody."""

    train: MultimodalObservationBatch
    calibration: MultimodalObservationBatch
    test: MultimodalObservationBatch
    config: SyntheticMultimodalConfig

    def __post_init__(self) -> None:
        """Require ordered split labels and disjoint trajectory identifiers."""
        if (self.train.split, self.calibration.split, self.test.split) != (
            "train",
            "calibration",
            "test",
        ):
            raise ValueError("dataset batches must be train, calibration, and test in order")
        assert_disjoint_batches(self.train, self.calibration, self.test)

    def content_digest(self) -> str:
        """Return a SHA256 binding the three immutable split digests."""
        digest = hashlib.sha256()
        digest.update(b"scpn.synthetic_multimodal_dataset.v1\0")
        for batch in (self.train, self.calibration, self.test):
            digest.update(batch.content_digest().encode("ascii"))
        return digest.hexdigest()

    def to_summary_dict(self) -> dict[str, object]:
        """Return JSON-ready split summaries and the dataset digest."""
        return {
            "source": SYNTHETIC_MULTIMODAL_SOURCE,
            "content_digest": self.content_digest(),
            "train": self.train.to_summary_dict(),
            "calibration": self.calibration.to_summary_dict(),
            "test": self.test.to_summary_dict(),
        }


_TAGS = tuple(SyntheticDomainTag)


def _coupling_template(tag: SyntheticDomainTag) -> FloatArray:
    matrix = np.zeros((4, 4), dtype=np.float64)
    if tag is SyntheticDomainTag.SYNTHETIC:
        for left, right in ((0, 1), (1, 2), (2, 3), (3, 0)):
            matrix[left, right] = matrix[right, left] = 0.20
    elif tag is SyntheticDomainTag.GRID_LIKE_SIM:
        for left, right, weight in ((0, 1, 0.32), (1, 2, 0.24), (2, 3, 0.29)):
            matrix[left, right] = matrix[right, left] = weight
    elif tag is SyntheticDomainTag.EEG_LIKE_SIM:
        matrix[0, 1] = matrix[1, 0] = 0.34
        matrix[2, 3] = matrix[3, 2] = 0.31
        matrix[1, 2] = matrix[2, 1] = 0.06
    else:
        matrix.fill(0.13)
        np.fill_diagonal(matrix, 0.0)
        matrix[0, 2] = matrix[2, 0] = 0.24
    return matrix


def _frequency_template(tag: SyntheticDomainTag) -> FloatArray:
    if tag is SyntheticDomainTag.SYNTHETIC:
        return np.array([-0.22, -0.05, 0.08, 0.25], dtype=np.float64)
    if tag is SyntheticDomainTag.GRID_LIKE_SIM:
        return np.array([-0.13, -0.04, 0.05, 0.12], dtype=np.float64)
    if tag is SyntheticDomainTag.EEG_LIKE_SIM:
        return np.array([-0.30, -0.24, 0.23, 0.29], dtype=np.float64)
    return np.array([-0.18, 0.02, 0.07, 0.16], dtype=np.float64)


def _masked_inputs(
    values: FloatArray,
    *,
    missing_fraction: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, BoolArray]:
    mask = rng.random(values.shape) >= missing_fraction
    mask[:, 0, :] = True
    masked = values.copy()
    masked[~mask] = np.nan
    return masked, mask


def _make_batch(
    split: DatasetSplit,
    *,
    n_samples: int,
    config: SyntheticMultimodalConfig,
    split_seed: int,
) -> MultimodalObservationBatch:
    rng = np.random.default_rng(split_seed)
    total_steps = config.history_steps + config.horizon_steps - 1
    series = np.empty((n_samples, config.history_steps, config.n_nodes), dtype=np.float64)
    graphs = np.empty((n_samples, config.n_nodes, config.n_nodes), dtype=np.float64)
    events = np.empty((n_samples, config.history_steps, 2), dtype=np.float64)
    targets = np.empty((n_samples, config.horizon_steps, config.n_nodes), dtype=np.float64)
    frequencies = np.empty((n_samples, config.n_nodes), dtype=np.float64)
    sample_ids: list[str] = []
    tags: list[SyntheticDomainTag] = []
    times = config.dt * np.arange(config.history_steps, dtype=np.float64)

    for index in range(n_samples):
        tag = _TAGS[index % len(_TAGS)]
        coupling_scale = float(rng.uniform(0.85, 1.15))
        coupling = coupling_scale * _coupling_template(tag)
        omega = _frequency_template(tag) + rng.normal(0.0, 0.015, size=config.n_nodes)
        theta0 = rng.uniform(-np.pi, np.pi, size=config.n_nodes)
        trajectory = simulate_kuramoto_phase_time_series(
            coupling,
            omega,
            theta0,
            dt=config.dt,
            n_steps=total_steps,
        )
        series[index] = trajectory[: config.history_steps]
        targets[index] = trajectory[config.history_steps :]
        graphs[index] = coupling
        frequencies[index] = omega
        events[index, :, 0] = coupling_scale
        events[index, :, 1] = times
        sample_ids.append(f"forecasting-{split}-{split_seed}-{index:04d}")
        tags.append(tag)

    masked_series, series_mask = _masked_inputs(
        series,
        missing_fraction=config.missing_fraction,
        rng=rng,
    )
    masked_events, event_mask = _masked_inputs(
        events,
        missing_fraction=0.5 * config.missing_fraction,
        rng=rng,
    )
    graph_mask = np.ones_like(graphs, dtype=np.bool_)
    masked_graphs = graphs.copy()
    target_mask = np.ones_like(targets, dtype=np.bool_)
    return MultimodalObservationBatch(
        series=masked_series,
        series_mask=series_mask,
        graphs=masked_graphs,
        graph_mask=graph_mask,
        events=masked_events,
        event_mask=event_mask,
        targets=targets,
        target_mask=target_mask,
        frequencies=frequencies,
        sample_ids=tuple(sample_ids),
        domain_tags=tuple(tags),
        split=split,
        dt=config.dt,
        source=SYNTHETIC_MULTIMODAL_SOURCE,
    )


def generate_synthetic_multimodal_dataset(
    config: SyntheticMultimodalConfig | None = None,
) -> SyntheticMultimodalDataset:
    """Generate deterministic disjoint forecasting trajectory custody."""
    resolved = SyntheticMultimodalConfig() if config is None else config
    seed_sequence = np.random.SeedSequence(resolved.seed)
    child_seeds = seed_sequence.spawn(3)
    split_seeds = tuple(int(child.generate_state(1, dtype=np.uint32)[0]) for child in child_seeds)
    train = _make_batch(
        "train",
        n_samples=resolved.train_samples,
        config=resolved,
        split_seed=split_seeds[0],
    )
    calibration = _make_batch(
        "calibration",
        n_samples=resolved.calibration_samples,
        config=resolved,
        split_seed=split_seeds[1],
    )
    test = _make_batch(
        "test",
        n_samples=resolved.test_samples,
        config=resolved,
        split_seed=split_seeds[2],
    )
    return SyntheticMultimodalDataset(train, calibration, test, resolved)


__all__ = [
    "SYNTHETIC_MULTIMODAL_SOURCE",
    "SyntheticMultimodalConfig",
    "SyntheticMultimodalDataset",
    "generate_synthetic_multimodal_dataset",
]
