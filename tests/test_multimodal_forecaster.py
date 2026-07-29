# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multimodal ridge forecaster tests
"""Production-surface tests for the BL-37 classical reference forecaster."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scpn_quantum_control.forecasting.multimodal_forecaster import (
    MultimodalPointForecast,
    MultimodalRidgeForecaster,
    _canonical_digest_array_bytes,
    evaluate_point_forecast,
    fit_multimodal_ridge_forecaster,
)
from scpn_quantum_control.forecasting.multimodal_schema import SyntheticDomainTag
from scpn_quantum_control.forecasting.synthetic_multimodal import (
    SyntheticMultimodalConfig,
    SyntheticMultimodalDataset,
    generate_synthetic_multimodal_dataset,
)


def test_model_digest_numeric_custody_ignores_subprecision_and_signed_zero() -> None:
    """Digest bytes are stable across insignificant BLAS/runtime drift."""
    left = np.asarray([0.12345678901231, -0.0, 1.0e-14], dtype=np.float64)
    right = np.asarray([0.12345678901229, 0.0, -1.0e-14], dtype=np.float64)
    assert _canonical_digest_array_bytes(left) == _canonical_digest_array_bytes(right)


def _dataset(*, history_steps: int = 8) -> SyntheticMultimodalDataset:
    return generate_synthetic_multimodal_dataset(
        SyntheticMultimodalConfig(
            train_samples=32,
            calibration_samples=8,
            test_samples=8,
            history_steps=history_steps,
            horizon_steps=3,
            missing_fraction=0.2,
            seed=3702,
        )
    )


def _model_record(
    *,
    history_steps: int = 2,
    horizon_steps: int = 1,
    n_nodes: int = 2,
    event_channels: int = 1,
) -> MultimodalRidgeForecaster:
    """Build shape-consistent model arrays for metadata-boundary tests."""
    raw_features = history_steps * n_nodes + n_nodes**2 + history_steps * event_channels + n_nodes
    design_features = 2 * raw_features + 5
    return MultimodalRidgeForecaster(
        feature_means=np.zeros(raw_features),
        feature_scales=np.ones(raw_features),
        coefficients=np.zeros((design_features, horizon_steps * n_nodes)),
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        n_nodes=n_nodes,
        event_channels=event_channels,
        dt=0.1,
        ridge=1.0,
        training_batch_digest="a" * 64,
        training_sample_ids=("train-row",),
        model_digest="b" * 64,
    )


def test_fit_predict_and_evaluate_are_deterministic_and_immutable() -> None:
    dataset = _dataset()
    first = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    second = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    forecast = first.predict(dataset.test)
    certificate = evaluate_point_forecast(forecast, dataset.test)

    assert first.model_digest == second.model_digest
    assert first.training_batch_digest == dataset.train.content_digest()
    assert first.training_sample_ids == dataset.train.sample_ids
    np.testing.assert_allclose(forecast.values, second.predict(dataset.test).values)
    assert forecast.sample_ids == dataset.test.sample_ids
    assert not forecast.values.flags.writeable
    assert certificate.samples == dataset.test.n_samples
    assert certificate.wrapped_mse >= 0.0
    assert certificate.wrapped_mae >= 0.0
    assert certificate.persistence_wrapped_mse >= 0.0
    assert len(certificate.domains) == 4
    assert certificate.domains[0].to_dict()["domain_tag"] == "synthetic"


def test_design_matrix_preserves_masks_and_domain_tags() -> None:
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    design = model.design_matrix(dataset.test)

    assert design.shape[0] == dataset.test.n_samples
    assert design.shape[1] == model.coefficients.shape[0]
    assert np.all(np.isfinite(design))
    assert np.all(design[:, -1] == 1.0)
    domain_block = design[:, -5:-1]
    np.testing.assert_allclose(np.sum(domain_block, axis=1), 1.0)


def test_fit_refuses_wrong_custody_hyperparameters_and_dense_budget() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="requires a train batch"):
        fit_multimodal_ridge_forecaster(replace(dataset.train, split="calibration"))
    with pytest.raises(ValueError, match="ridge must"):
        fit_multimodal_ridge_forecaster(dataset.train, ridge=0.0)
    with pytest.raises(ValueError, match="max_dense_gib"):
        fit_multimodal_ridge_forecaster(dataset.train, max_dense_gib=0.0)
    with pytest.raises(MemoryError, match="above"):
        fit_multimodal_ridge_forecaster(dataset.train, max_dense_gib=1.0e-12)


def test_fit_refuses_unobserved_features_and_underobserved_targets() -> None:
    dataset = _dataset()
    graph_mask = dataset.train.graph_mask.copy()
    graph_mask[:, 0, 1] = False
    with pytest.raises(ValueError, match="every raw feature"):
        fit_multimodal_ridge_forecaster(replace(dataset.train, graph_mask=graph_mask))

    target_mask = dataset.train.target_mask.copy()
    target_mask[:, 0, 0] = False
    target_mask[0, 0, 0] = True
    with pytest.raises(ValueError, match="at least two"):
        fit_multimodal_ridge_forecaster(replace(dataset.train, target_mask=target_mask))


def test_predict_refuses_incompatible_shape_time_and_feature_width() -> None:
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    incompatible = _dataset(history_steps=9)
    with pytest.raises(ValueError, match="shape is incompatible"):
        model.predict(incompatible.test)
    with pytest.raises(ValueError, match="dt is incompatible"):
        model.predict(replace(dataset.test, dt=0.05))
    with pytest.raises(ValueError, match="feature statistics"):
        replace(
            model,
            feature_means=model.feature_means[:-1],
            feature_scales=model.feature_scales[:-1],
        )


def test_point_forecast_and_evaluation_refuse_custody_drift() -> None:
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    forecast = model.predict(dataset.test)
    with pytest.raises(ValueError, match="sample_ids must match"):
        MultimodalPointForecast(forecast.values, forecast.sample_ids[:-1], model.model_digest)
    with pytest.raises(ValueError, match="must be unique"):
        MultimodalPointForecast(
            forecast.values,
            (forecast.sample_ids[0],) * dataset.test.n_samples,
            model.model_digest,
        )
    with pytest.raises(ValueError, match="model_digest"):
        MultimodalPointForecast(forecast.values, forecast.sample_ids, "")
    with pytest.raises(ValueError, match="custody must match"):
        evaluate_point_forecast(
            replace(forecast, sample_ids=tuple(reversed(forecast.sample_ids))),
            dataset.test,
        )
    with pytest.raises(ValueError, match="target shapes"):
        evaluate_point_forecast(
            MultimodalPointForecast(
                forecast.values[:, :-1],
                forecast.sample_ids,
                forecast.model_digest,
            ),
            dataset.test,
        )


def test_prediction_refuses_sample_node_without_history_anchor() -> None:
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    mask = dataset.test.series_mask.copy()
    mask[0, :, 0] = False
    damaged = replace(dataset.test, series_mask=mask)
    with pytest.raises(ValueError, match="persistence requires"):
        model.predict(damaged)


def test_model_record_rejects_every_invalid_shape_and_custody_field() -> None:
    """The fitted-model record fails closed on arrays, metadata, and identities."""
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    with pytest.raises(ValueError, match="rank-1"):
        replace(model, feature_means=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="rank-1"):
        replace(model, feature_means=np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="rank-1"):
        replace(model, feature_means=np.array([np.nan]))
    with pytest.raises(ValueError, match="align and scales"):
        replace(model, feature_scales=-np.ones_like(model.feature_scales))
    with pytest.raises(ValueError, match="align and scales"):
        replace(model, feature_scales=model.feature_scales[:-1])
    with pytest.raises(ValueError, match="coefficient shape"):
        replace(model, coefficients=model.coefficients[:-1])
    with pytest.raises(ValueError, match="shape metadata is invalid"):
        _model_record(history_steps=1)
    with pytest.raises(ValueError, match="shape metadata is invalid"):
        _model_record(n_nodes=1)
    with pytest.raises(ValueError, match="event_channels"):
        _model_record(event_channels=0)
    with pytest.raises(ValueError, match="dt must"):
        replace(model, dt=0.0)
    with pytest.raises(ValueError, match="ridge must"):
        replace(model, ridge=0.0)
    with pytest.raises(ValueError, match="digests must"):
        replace(model, model_digest="")
    with pytest.raises(ValueError, match="digests must"):
        replace(model, training_batch_digest="")
    with pytest.raises(ValueError, match="non-empty and unique"):
        replace(model, training_sample_ids=())
    with pytest.raises(ValueError, match="non-empty and unique"):
        replace(model, training_sample_ids=(model.training_sample_ids[0],) * 2)


def test_point_forecast_rejects_nonfinite_or_wrong_rank_values() -> None:
    """Point forecasts require a finite, non-empty rank-three tensor."""
    with pytest.raises(ValueError, match="rank-3"):
        MultimodalPointForecast(np.array([np.nan]), ("row",), "f" * 64)
    with pytest.raises(ValueError, match="rank-3"):
        MultimodalPointForecast(np.empty((0, 1, 1)), (), "f" * 64)
    with pytest.raises(ValueError, match="rank-3"):
        MultimodalPointForecast(np.full((1, 1, 1), np.nan), ("row",), "f" * 64)


def test_evaluation_skips_domain_tags_absent_from_batch() -> None:
    """Domain accuracy rows cover represented tags without inventing empty rows."""
    dataset = _dataset()
    model: MultimodalRidgeForecaster = fit_multimodal_ridge_forecaster(
        dataset.train,
        ridge=10.0,
    )
    single_tag = replace(
        dataset.test,
        domain_tags=(SyntheticDomainTag.SYNTHETIC,) * dataset.test.n_samples,
    )
    certificate = evaluate_point_forecast(model.predict(single_tag), single_tag)

    assert [row.domain_tag for row in certificate.domains] == [SyntheticDomainTag.SYNTHETIC]
