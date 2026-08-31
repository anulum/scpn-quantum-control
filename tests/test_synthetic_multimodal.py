# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synthetic multimodal dataset tests
"""Production-surface tests for multimodal-forecasting simulation-only trajectory custody."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from scpn_quantum_control.forecasting.multimodal_schema import SyntheticDomainTag
from scpn_quantum_control.forecasting.synthetic_multimodal import (
    SYNTHETIC_MULTIMODAL_SOURCE,
    SyntheticMultimodalConfig,
    SyntheticMultimodalDataset,
    generate_synthetic_multimodal_dataset,
)
from scpn_quantum_control.phase.coupling_time_series_recovery import (
    simulate_kuramoto_phase_time_series,
)


def _config(
    *,
    train_samples: int = 8,
    calibration_samples: int = 8,
    test_samples: int = 8,
    history_steps: int = 6,
    horizon_steps: int = 3,
    n_nodes: int = 4,
    dt: float = 0.04,
    missing_fraction: float = 0.25,
    seed: int = 37,
) -> SyntheticMultimodalConfig:
    return SyntheticMultimodalConfig(
        train_samples=train_samples,
        calibration_samples=calibration_samples,
        test_samples=test_samples,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        n_nodes=n_nodes,
        dt=dt,
        missing_fraction=missing_fraction,
        seed=seed,
    )


def test_generator_is_deterministic_disjoint_and_domain_balanced() -> None:
    """Keep seeded split custody deterministic, disjoint, and domain-complete."""
    first = generate_synthetic_multimodal_dataset(_config())
    second = generate_synthetic_multimodal_dataset(_config())
    changed = generate_synthetic_multimodal_dataset(_config(seed=38))

    assert first.content_digest() == second.content_digest()
    assert first.content_digest() != changed.content_digest()
    assert not set(first.train.sample_ids).intersection(first.calibration.sample_ids)
    assert not set(first.train.sample_ids).intersection(first.test.sample_ids)
    assert set(first.train.domain_tags) == set(SyntheticDomainTag)
    assert first.train.source == SYNTHETIC_MULTIMODAL_SOURCE
    assert first.train.missing_fraction > 0.0
    assert np.isnan(first.train.series[~first.train.series_mask]).all()
    assert np.isnan(first.train.events[~first.train.event_mask]).all()
    assert np.isnan(first.train.graphs[~first.train.graph_mask]).all()
    summary = first.to_summary_dict()
    assert summary["content_digest"] == first.content_digest()
    assert summary["train"] == first.train.to_summary_dict()


def test_zero_missing_dataset_replays_exact_kuramoto_trajectory() -> None:
    """Replay every unmasked phase target from the exact Kuramoto trajectory."""
    config = _config(missing_fraction=0.0)
    dataset = generate_synthetic_multimodal_dataset(config)
    batch = dataset.test
    expected = simulate_kuramoto_phase_time_series(
        batch.graphs[0],
        batch.frequencies[0],
        batch.series[0, 0],
        dt=batch.dt,
        n_steps=config.history_steps + config.horizon_steps - 1,
    )

    np.testing.assert_allclose(batch.series[0], expected[: config.history_steps])
    np.testing.assert_allclose(batch.targets[0], expected[config.history_steps :])
    assert np.all(batch.series_mask)
    assert np.all(batch.event_mask)
    assert np.all(batch.graph_mask)


def test_domain_tags_change_only_simulation_templates() -> None:
    """Separate synthetic domain tags through their declared graph templates."""
    dataset = generate_synthetic_multimodal_dataset(_config(missing_fraction=0.0))
    batch = dataset.train
    by_tag = {tag: index for index, tag in enumerate(batch.domain_tags[:4])}

    grid = batch.graphs[by_tag[SyntheticDomainTag.GRID_LIKE_SIM]]
    eeg = batch.graphs[by_tag[SyntheticDomainTag.EEG_LIKE_SIM]]
    plasma = batch.graphs[by_tag[SyntheticDomainTag.PLASMA_LIKE_SIM]]
    assert grid[0, 3] == 0.0
    assert eeg[0, 1] > eeg[1, 2]
    assert plasma[0, 2] > plasma[0, 1]
    assert all("sim" in tag.value or tag is SyntheticDomainTag.SYNTHETIC for tag in by_tag)


@pytest.mark.parametrize(
    ("builder", "message"),
    [
        (lambda: _config(train_samples=3), "at least four"),
        (lambda: _config(history_steps=2), "history_steps"),
        (lambda: _config(horizon_steps=0), "history_steps"),
        (lambda: _config(n_nodes=5), "n_nodes=4"),
        (lambda: _config(dt=0.0), "dt must"),
        (lambda: _config(missing_fraction=0.8), "missing_fraction"),
        (lambda: _config(missing_fraction=float("nan")), "missing_fraction"),
        (lambda: _config(seed=-1), "seed must"),
    ],
)
def test_config_rejects_unbounded_or_invalid_inputs(
    builder: Callable[[], SyntheticMultimodalConfig], message: str
) -> None:
    """Reject invalid bounded-domain sample, shape, time, and seed controls."""
    with pytest.raises(ValueError, match=message):
        builder()


def test_dataset_rejects_wrong_split_order_and_leakage() -> None:
    """Reject reordered custody splits and repeated trajectory identifiers."""
    dataset = generate_synthetic_multimodal_dataset(_config())
    with pytest.raises(ValueError, match="train, calibration, and test"):
        SyntheticMultimodalDataset(
            dataset.calibration,
            dataset.train,
            dataset.test,
            dataset.config,
        )
    with pytest.raises(ValueError, match="sample-id leakage"):
        SyntheticMultimodalDataset(
            dataset.train,
            replace(dataset.calibration, sample_ids=dataset.train.sample_ids),
            dataset.test,
            dataset.config,
        )
