# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multimodal forecasting schema
"""Immutable multimodal observation custody for bounded synthetic forecasts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
DatasetSplit = Literal["train", "calibration", "test"]


class SyntheticDomainTag(str, Enum):
    """Allowed simulation-only domain tags for multimodal forecasting."""

    SYNTHETIC = "synthetic"
    GRID_LIKE_SIM = "grid_like_sim"
    EEG_LIKE_SIM = "eeg_like_sim"
    PLASMA_LIKE_SIM = "plasma_like_sim"


def _immutable_float(values: object, *, name: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    array.setflags(write=False)
    return array


def _immutable_mask(values: object, *, name: str) -> BoolArray:
    array = np.array(values, dtype=np.bool_, copy=True)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    array.setflags(write=False)
    return array


def _normalise_masked_values(values: FloatArray, mask: BoolArray, *, name: str) -> FloatArray:
    if values.shape != mask.shape:
        raise ValueError(f"{name} and its mask must have identical shapes")
    if not np.all(np.isfinite(values[mask])):
        raise ValueError(f"{name} observed entries must be finite")
    normalised = np.array(values, dtype=np.float64, copy=True)
    normalised[~mask] = np.nan
    normalised.setflags(write=False)
    return normalised


@dataclass(frozen=True, slots=True)
class MultimodalObservationBatch:
    """One immutable train, calibration, or test batch.

    Parameters
    ----------
    series
        Phase histories with shape ``(samples, history, nodes)`` in radians.
    series_mask
        Boolean observation mask with the same shape as ``series``.
    graphs
        Sample-specific oscillator couplings with shape
        ``(samples, nodes, nodes)``.
    graph_mask
        Boolean graph-entry mask with the same shape as ``graphs``.
    events
        Exogenous synthetic features with shape
        ``(samples, history, event_channels)``.
    event_mask
        Boolean event mask with the same shape as ``events``.
    targets
        Future phase targets with shape ``(samples, horizon, nodes)``.
    target_mask
        Boolean target-observation mask with the same shape as ``targets``.
    frequencies
        Natural frequencies with shape ``(samples, nodes)`` in radians per
        unit time.
    sample_ids
        Unique custody identifiers for every independent trajectory row.
    domain_tags
        Simulation-only generator tags, one per sample.
    split
        Explicit custody split: ``train``, ``calibration``, or ``test``.
    dt
        Integration time step.
    source
        Non-empty provenance label for the synthetic generator.
    """

    series: FloatArray
    series_mask: BoolArray
    graphs: FloatArray
    graph_mask: BoolArray
    events: FloatArray
    event_mask: BoolArray
    targets: FloatArray
    target_mask: BoolArray
    frequencies: FloatArray
    sample_ids: tuple[str, ...]
    domain_tags: tuple[SyntheticDomainTag, ...]
    split: DatasetSplit
    dt: float
    source: str

    def __post_init__(self) -> None:
        """Copy, validate, normalise, and freeze all batch data."""
        series_mask = _immutable_mask(self.series_mask, name="series_mask")
        graph_mask = _immutable_mask(self.graph_mask, name="graph_mask")
        event_mask = _immutable_mask(self.event_mask, name="event_mask")
        target_mask = _immutable_mask(self.target_mask, name="target_mask")
        series = _normalise_masked_values(
            _immutable_float(self.series, name="series"), series_mask, name="series"
        )
        graphs = _normalise_masked_values(
            _immutable_float(self.graphs, name="graphs"), graph_mask, name="graphs"
        )
        events = _normalise_masked_values(
            _immutable_float(self.events, name="events"), event_mask, name="events"
        )
        targets = _normalise_masked_values(
            _immutable_float(self.targets, name="targets"), target_mask, name="targets"
        )
        frequencies = _immutable_float(self.frequencies, name="frequencies")

        if series.ndim != 3 or graphs.ndim != 3 or events.ndim != 3 or targets.ndim != 3:
            raise ValueError("series, graphs, events, and targets must be rank-three")
        n_samples, history_steps, n_nodes = series.shape
        if history_steps < 2 or n_nodes < 2:
            raise ValueError("batch requires history >= 2 and nodes >= 2")
        if graphs.shape != (n_samples, n_nodes, n_nodes):
            raise ValueError("graphs must have shape (samples, nodes, nodes)")
        if events.shape[0] != n_samples or events.shape[1] != history_steps:
            raise ValueError("events must align with sample and history dimensions")
        if targets.shape[0] != n_samples or targets.shape[2] != n_nodes:
            raise ValueError("targets must align with sample and node dimensions")
        if frequencies.shape != (n_samples, n_nodes):
            raise ValueError("frequencies must have shape (samples, nodes)")
        if not np.all(np.isfinite(frequencies)):
            raise ValueError("frequencies must be finite")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError("split must be train, calibration, or test")
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if len(self.sample_ids) != n_samples or len(set(self.sample_ids)) != n_samples:
            raise ValueError("sample_ids must be unique and match the sample count")
        if any(not sample_id.strip() for sample_id in self.sample_ids):
            raise ValueError("sample_ids must be non-empty")
        if len(self.domain_tags) != n_samples or any(
            not isinstance(tag, SyntheticDomainTag) for tag in self.domain_tags
        ):
            raise ValueError("domain_tags must contain one SyntheticDomainTag per sample")
        if np.any(np.sum(series_mask, axis=(0, 1)) == 0):
            raise ValueError("every phase channel needs at least one observed history value")
        if np.any(np.sum(target_mask, axis=(0, 1)) == 0):
            raise ValueError("every target node needs at least one observed target value")

        object.__setattr__(self, "series", series)
        object.__setattr__(self, "series_mask", series_mask)
        object.__setattr__(self, "graphs", graphs)
        object.__setattr__(self, "graph_mask", graph_mask)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "event_mask", event_mask)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "target_mask", target_mask)
        object.__setattr__(self, "frequencies", frequencies)

    @property
    def n_samples(self) -> int:
        """Return the number of independent trajectory rows."""
        return int(self.series.shape[0])

    @property
    def history_steps(self) -> int:
        """Return the number of historical time steps."""
        return int(self.series.shape[1])

    @property
    def n_nodes(self) -> int:
        """Return the oscillator-node count."""
        return int(self.series.shape[2])

    @property
    def horizon_steps(self) -> int:
        """Return the forecast-horizon length."""
        return int(self.targets.shape[1])

    @property
    def event_channels(self) -> int:
        """Return the number of exogenous event channels."""
        return int(self.events.shape[2])

    @property
    def missing_fraction(self) -> float:
        """Return the missing fraction across input modalities."""
        observed = int(np.count_nonzero(self.series_mask))
        observed += int(np.count_nonzero(self.graph_mask))
        observed += int(np.count_nonzero(self.event_mask))
        possible = self.series_mask.size + self.graph_mask.size + self.event_mask.size
        return 1.0 - observed / possible

    def content_digest(self) -> str:
        """Return a SHA256 binding values, masks, tags, IDs, and custody split."""
        digest = hashlib.sha256()
        digest.update(b"scpn.multimodal_observation_batch.v1\0")
        digest.update(self.split.encode("utf-8") + b"\0")
        digest.update(self.source.encode("utf-8") + b"\0")
        digest.update(np.float64(self.dt).tobytes())
        for sample_id, tag in zip(self.sample_ids, self.domain_tags, strict=True):
            digest.update(sample_id.encode("utf-8") + b"\0")
            digest.update(tag.value.encode("utf-8") + b"\0")
        for label, values in (
            ("series", np.nan_to_num(self.series, nan=0.0)),
            ("series_mask", self.series_mask),
            ("graphs", np.nan_to_num(self.graphs, nan=0.0)),
            ("graph_mask", self.graph_mask),
            ("events", np.nan_to_num(self.events, nan=0.0)),
            ("event_mask", self.event_mask),
            ("targets", np.nan_to_num(self.targets, nan=0.0)),
            ("target_mask", self.target_mask),
            ("frequencies", self.frequencies),
        ):
            digest.update(label.encode("utf-8"))
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes(order="C"))
        return digest.hexdigest()

    def to_summary_dict(self) -> dict[str, object]:
        """Return a JSON-ready custody and shape summary without bulk arrays."""
        counts = {tag.value: self.domain_tags.count(tag) for tag in SyntheticDomainTag}
        return {
            "split": self.split,
            "samples": self.n_samples,
            "history_steps": self.history_steps,
            "horizon_steps": self.horizon_steps,
            "nodes": self.n_nodes,
            "event_channels": self.event_channels,
            "missing_fraction": self.missing_fraction,
            "domain_counts": counts,
            "source": self.source,
            "content_digest": self.content_digest(),
        }


def assert_disjoint_batches(*batches: MultimodalObservationBatch) -> None:
    """Raise when any sample identifier appears in more than one batch."""
    seen: set[str] = set()
    for batch in batches:
        overlap = seen.intersection(batch.sample_ids)
        if overlap:
            raise ValueError(f"sample-id leakage across batches: {sorted(overlap)!r}")
        seen.update(batch.sample_ids)


__all__ = [
    "BoolArray",
    "DatasetSplit",
    "FloatArray",
    "MultimodalObservationBatch",
    "SyntheticDomainTag",
    "assert_disjoint_batches",
]
