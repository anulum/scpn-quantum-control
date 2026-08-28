# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multimodal forecasting schema tests
"""Production-surface tests for immutable multimodal-forecasting observation custody."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.forecasting as forecasting
from scpn_quantum_control.forecasting.multimodal_schema import (
    DatasetSplit,
    MultimodalObservationBatch,
    SyntheticDomainTag,
    assert_disjoint_batches,
)


def _batch(
    *, sample_id: str = "train-0", split: DatasetSplit = "train"
) -> MultimodalObservationBatch:
    series = np.arange(12, dtype=np.float64).reshape(1, 4, 3)
    series_mask = np.ones_like(series, dtype=np.bool_)
    graphs = np.ones((1, 3, 3), dtype=np.float64) - np.eye(3)[None, :, :]
    graph_mask = np.ones_like(graphs, dtype=np.bool_)
    events = np.ones((1, 4, 2), dtype=np.float64)
    event_mask = np.ones_like(events, dtype=np.bool_)
    targets = np.arange(6, dtype=np.float64).reshape(1, 2, 3)
    target_mask = np.ones_like(targets, dtype=np.bool_)
    return MultimodalObservationBatch(
        series=series,
        series_mask=series_mask,
        graphs=graphs,
        graph_mask=graph_mask,
        events=events,
        event_mask=event_mask,
        targets=targets,
        target_mask=target_mask,
        frequencies=np.array([[0.1, 0.2, 0.3]]),
        sample_ids=(sample_id,),
        domain_tags=(SyntheticDomainTag.SYNTHETIC,),
        split=split,
        dt=0.1,
        source="unit.synthetic",
    )


def test_public_forecasting_facade_exports_product_entry_points() -> None:
    """The documented facade reaches every multimodal forecasting stage."""
    assert forecasting.MultimodalObservationBatch is MultimodalObservationBatch
    assert forecasting.SyntheticDomainTag is SyntheticDomainTag
    assert callable(forecasting.generate_synthetic_multimodal_dataset)
    assert callable(forecasting.fit_multimodal_ridge_forecaster)
    assert callable(forecasting.evaluate_partial_observation_batch)
    assert callable(forecasting.fit_residual_interval_calibrator)
    assert callable(forecasting.plan_forecast_active_sensing)
    assert callable(forecasting.forecast_to_controller_initialisation)
    assert callable(forecasting.write_multimodal_forecasting_evidence)


def test_batch_normalises_missing_values_and_freezes_custody() -> None:
    baseline = _batch()
    series = baseline.series.copy()
    mask = baseline.series_mask.copy()
    series[0, 1, 2] = 999.0
    mask[0, 1, 2] = False
    batch = MultimodalObservationBatch(
        series=series,
        series_mask=mask,
        graphs=baseline.graphs,
        graph_mask=baseline.graph_mask,
        events=baseline.events,
        event_mask=baseline.event_mask,
        targets=baseline.targets,
        target_mask=baseline.target_mask,
        frequencies=baseline.frequencies,
        sample_ids=baseline.sample_ids,
        domain_tags=baseline.domain_tags,
        split="train",
        dt=baseline.dt,
        source=baseline.source,
    )

    assert np.isnan(batch.series[0, 1, 2])
    assert not batch.series.flags.writeable
    assert batch.n_samples == 1
    assert batch.history_steps == 4
    assert batch.horizon_steps == 2
    assert batch.n_nodes == 3
    assert batch.event_channels == 2
    assert batch.missing_fraction > 0.0
    with pytest.raises(ValueError, match="read-only"):
        batch.series[0, 0, 0] = 2.0


def test_digest_and_summary_bind_masks_ids_tags_and_split() -> None:
    first = _batch()
    same = _batch()
    changed_id = _batch(sample_id="train-1")
    changed_split = _batch(split="test")

    assert first.content_digest() == same.content_digest()
    assert first.content_digest() != changed_id.content_digest()
    assert first.content_digest() != changed_split.content_digest()
    summary = first.to_summary_dict()
    assert summary["content_digest"] == first.content_digest()
    assert summary["domain_counts"] == {
        "synthetic": 1,
        "grid_like_sim": 0,
        "eeg_like_sim": 0,
        "plasma_like_sim": 0,
    }


def test_disjoint_batch_gate_rejects_cross_split_leakage() -> None:
    train = _batch(sample_id="row-0", split="train")
    test = _batch(sample_id="row-1", split="test")
    assert_disjoint_batches(train, test)
    with pytest.raises(ValueError, match="sample-id leakage"):
        assert_disjoint_batches(train, _batch(sample_id="row-0", split="calibration"))


def test_schema_rejects_invalid_shapes_and_metadata() -> None:
    baseline = _batch()
    with pytest.raises(ValueError, match="rank-three"):
        replace(
            baseline,
            series=np.zeros((1, 2)),
            series_mask=np.ones((1, 2), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="graphs must have shape"):
        replace(
            baseline,
            graphs=np.zeros((1, 2, 2)),
            graph_mask=np.ones((1, 2, 2), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="frequencies must have shape"):
        replace(baseline, frequencies=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        replace(baseline, dt=0.0)
    with pytest.raises(ValueError, match="source must be non-empty"):
        replace(baseline, source="")
    with pytest.raises(ValueError, match="sample_ids must be non-empty"):
        replace(baseline, sample_ids=("",))
    with pytest.raises(ValueError, match="domain_tags"):
        replace(baseline, domain_tags=())


def test_schema_rejects_mask_shape_nonfinite_and_unobserved_channels() -> None:
    baseline = _batch()
    with pytest.raises(ValueError, match="identical shapes"):
        MultimodalObservationBatch(
            series=baseline.series,
            series_mask=np.ones((1, 3, 3), dtype=np.bool_),
            graphs=baseline.graphs,
            graph_mask=baseline.graph_mask,
            events=baseline.events,
            event_mask=baseline.event_mask,
            targets=baseline.targets,
            target_mask=baseline.target_mask,
            frequencies=baseline.frequencies,
            sample_ids=baseline.sample_ids,
            domain_tags=baseline.domain_tags,
            split="train",
            dt=0.1,
            source="unit.synthetic",
        )
    series = baseline.series.copy()
    series[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="observed entries must be finite"):
        MultimodalObservationBatch(
            series=series,
            series_mask=baseline.series_mask,
            graphs=baseline.graphs,
            graph_mask=baseline.graph_mask,
            events=baseline.events,
            event_mask=baseline.event_mask,
            targets=baseline.targets,
            target_mask=baseline.target_mask,
            frequencies=baseline.frequencies,
            sample_ids=baseline.sample_ids,
            domain_tags=baseline.domain_tags,
            split="train",
            dt=0.1,
            source="unit.synthetic",
        )
    mask = baseline.series_mask.copy()
    mask[:, :, 1] = False
    with pytest.raises(ValueError, match="every phase channel"):
        MultimodalObservationBatch(
            series=baseline.series,
            series_mask=mask,
            graphs=baseline.graphs,
            graph_mask=baseline.graph_mask,
            events=baseline.events,
            event_mask=baseline.event_mask,
            targets=baseline.targets,
            target_mask=baseline.target_mask,
            frequencies=baseline.frequencies,
            sample_ids=baseline.sample_ids,
            domain_tags=baseline.domain_tags,
            split="train",
            dt=0.1,
            source="unit.synthetic",
        )
    target_mask = baseline.target_mask.copy()
    target_mask[:, :, 1] = False
    with pytest.raises(ValueError, match="every target node"):
        replace(baseline, target_mask=target_mask)


def test_schema_rejects_empty_arrays_and_all_shape_metadata_branches() -> None:
    """Every rank, alignment, and finite-metadata boundary fails explicitly."""
    baseline = _batch()
    with pytest.raises(ValueError, match="series_mask must be non-empty"):
        replace(baseline, series_mask=np.array([], dtype=np.bool_))
    with pytest.raises(ValueError, match="series must be non-empty"):
        replace(baseline, series=np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="rank-three"):
        replace(
            baseline,
            graphs=np.zeros((3, 3)),
            graph_mask=np.ones((3, 3), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="rank-three"):
        replace(
            baseline,
            events=np.zeros((4, 2)),
            event_mask=np.ones((4, 2), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="rank-three"):
        replace(
            baseline,
            targets=np.zeros((2, 3)),
            target_mask=np.ones((2, 3), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="history >= 2"):
        replace(
            baseline,
            series=np.zeros((1, 1, 3)),
            series_mask=np.ones((1, 1, 3), dtype=np.bool_),
            events=np.zeros((1, 1, 2)),
            event_mask=np.ones((1, 1, 2), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="nodes >= 2"):
        replace(
            baseline,
            series=np.zeros((1, 4, 1)),
            series_mask=np.ones((1, 4, 1), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="events must align"):
        replace(
            baseline,
            events=np.zeros((1, 3, 2)),
            event_mask=np.ones((1, 3, 2), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="events must align"):
        replace(
            baseline,
            events=np.zeros((2, 4, 2)),
            event_mask=np.ones((2, 4, 2), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="targets must align"):
        replace(
            baseline,
            targets=np.zeros((1, 2, 2)),
            target_mask=np.ones((1, 2, 2), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="targets must align"):
        replace(
            baseline,
            targets=np.zeros((2, 2, 3)),
            target_mask=np.ones((2, 2, 3), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="frequencies must be finite"):
        replace(baseline, frequencies=np.array([[0.1, np.inf, 0.3]]))
    with pytest.raises(ValueError, match="split must be"):
        replace(baseline, split=cast(DatasetSplit, "validation"))
    with pytest.raises(ValueError, match="unique and match"):
        replace(baseline, sample_ids=())
    with pytest.raises(ValueError, match="domain_tags"):
        replace(
            baseline,
            domain_tags=cast(tuple[SyntheticDomainTag, ...], ("not-a-tag",)),
        )

    duplicate_ids = ("duplicate", "duplicate")
    with pytest.raises(ValueError, match="unique and match"):
        replace(
            baseline,
            series=np.repeat(baseline.series, 2, axis=0),
            series_mask=np.repeat(baseline.series_mask, 2, axis=0),
            graphs=np.repeat(baseline.graphs, 2, axis=0),
            graph_mask=np.repeat(baseline.graph_mask, 2, axis=0),
            events=np.repeat(baseline.events, 2, axis=0),
            event_mask=np.repeat(baseline.event_mask, 2, axis=0),
            targets=np.repeat(baseline.targets, 2, axis=0),
            target_mask=np.repeat(baseline.target_mask, 2, axis=0),
            frequencies=np.repeat(baseline.frequencies, 2, axis=0),
            sample_ids=duplicate_ids,
            domain_tags=(SyntheticDomainTag.SYNTHETIC,) * 2,
        )
