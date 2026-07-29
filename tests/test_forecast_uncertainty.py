# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multimodal forecast uncertainty tests
"""Production-surface tests for BL-37 split residual interval custody."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scpn_quantum_control.forecasting.multimodal_forecaster import (
    fit_multimodal_ridge_forecaster,
)
from scpn_quantum_control.forecasting.multimodal_schema import SyntheticDomainTag
from scpn_quantum_control.forecasting.synthetic_multimodal import (
    SyntheticMultimodalConfig,
    SyntheticMultimodalDataset,
    generate_synthetic_multimodal_dataset,
)
from scpn_quantum_control.forecasting.uncertainty import (
    MultimodalIntervalForecast,
    apply_residual_interval,
    certify_interval_coverage,
    fit_residual_interval_calibrator,
)


def _dataset() -> SyntheticMultimodalDataset:
    return generate_synthetic_multimodal_dataset(
        SyntheticMultimodalConfig(
            train_samples=40,
            calibration_samples=12,
            test_samples=12,
            history_steps=8,
            horizon_steps=3,
            missing_fraction=0.2,
            seed=3704,
        )
    )


def test_calibration_and_test_coverage_are_deterministic_and_disjoint() -> None:
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    calibration_forecast = model.predict(dataset.calibration)
    first = fit_residual_interval_calibrator(
        model, calibration_forecast, dataset.calibration, alpha=0.2
    )
    second = fit_residual_interval_calibrator(
        model, calibration_forecast, dataset.calibration, alpha=0.2
    )
    interval = apply_residual_interval(first, model.predict(dataset.test))
    certificate = certify_interval_coverage(model, first, interval, dataset.test)

    assert first.calibrator_digest == second.calibrator_digest
    assert first.radius == second.radius
    assert first.target_coverage == 0.8
    assert first.order_statistic_rank == 11
    assert first.to_dict()["calibrator_digest"] == first.calibrator_digest
    assert not interval.point.flags.writeable
    np.testing.assert_allclose(interval.upper - interval.lower, 2.0 * first.radius)
    assert certificate.samples == dataset.test.n_samples
    assert 0.0 <= certificate.sample_coverage <= 1.0
    assert 0.0 <= certificate.value_coverage <= 1.0
    assert certificate.mean_interval_width == 2.0 * first.radius
    assert len(certificate.domains) == 4
    assert "not sequential EnbPI" in certificate.claim_boundary
    assert certificate.domains[0].to_dict()["domain_tag"] == "synthetic"


def test_calibrator_refuses_wrong_split_digest_custody_alpha_and_empty_sample() -> None:
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    forecast = model.predict(dataset.calibration)
    with pytest.raises(ValueError, match="requires a calibration"):
        fit_residual_interval_calibrator(model, model.predict(dataset.test), dataset.test)
    with pytest.raises(ValueError, match="model digest"):
        fit_residual_interval_calibrator(
            model,
            replace(forecast, model_digest="different"),
            dataset.calibration,
        )
    with pytest.raises(ValueError, match="custody/shape"):
        fit_residual_interval_calibrator(
            model,
            replace(forecast, sample_ids=tuple(reversed(forecast.sample_ids))),
            dataset.calibration,
        )
    leaked = replace(dataset.calibration, sample_ids=model.training_sample_ids[:12])
    with pytest.raises(ValueError, match="must be disjoint"):
        fit_residual_interval_calibrator(model, model.predict(leaked), leaked)
    with pytest.raises(ValueError, match="alpha must"):
        fit_residual_interval_calibrator(model, forecast, dataset.calibration, alpha=1.0)
    with pytest.raises(ValueError, match="too small"):
        fit_residual_interval_calibrator(model, forecast, dataset.calibration, alpha=0.01)
    target_mask = dataset.calibration.target_mask.copy()
    target_mask[0] = False
    empty = replace(dataset.calibration, target_mask=target_mask)
    with pytest.raises(ValueError, match="each calibration sample"):
        fit_residual_interval_calibrator(model, model.predict(empty), empty, alpha=0.2)


def test_apply_interval_and_interval_contract_refuse_digest_or_shape_drift() -> None:
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    calibrator = fit_residual_interval_calibrator(
        model,
        model.predict(dataset.calibration),
        dataset.calibration,
        alpha=0.2,
    )
    point = model.predict(dataset.test)
    interval = apply_residual_interval(calibrator, point)
    with pytest.raises(ValueError, match="does not match"):
        apply_residual_interval(calibrator, replace(point, model_digest="other"))
    with pytest.raises(ValueError, match="share shape"):
        MultimodalIntervalForecast(
            point.values,
            point.values[:, :-1],
            point.values,
            point.sample_ids,
            point.model_digest,
            calibrator.calibrator_digest,
        )
    with pytest.raises(ValueError, match="contain"):
        MultimodalIntervalForecast(
            point.values,
            point.values + 1.0,
            point.values + 2.0,
            point.sample_ids,
            point.model_digest,
            calibrator.calibrator_digest,
        )
    with pytest.raises(ValueError, match="sample_ids"):
        MultimodalIntervalForecast(
            point.values,
            point.values - 1.0,
            point.values + 1.0,
            point.sample_ids[:-1],
            point.model_digest,
            calibrator.calibrator_digest,
        )
    with pytest.raises(ValueError, match="rank-three"):
        MultimodalIntervalForecast(
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
            ("row",),
            point.model_digest,
            calibrator.calibrator_digest,
        )
    with pytest.raises(ValueError, match="rank-three"):
        MultimodalIntervalForecast(
            np.empty((0, 1, 1)),
            np.empty((0, 1, 1)),
            np.empty((0, 1, 1)),
            (),
            point.model_digest,
            calibrator.calibrator_digest,
        )
    with pytest.raises(ValueError, match="rank-three"):
        MultimodalIntervalForecast(
            np.full((1, 1, 1), np.nan),
            np.zeros((1, 1, 1)),
            np.ones((1, 1, 1)),
            ("row",),
            point.model_digest,
            calibrator.calibrator_digest,
        )
    with pytest.raises(ValueError, match="digests must be non-empty"):
        replace(interval, calibrator_digest="")


def test_coverage_skips_domain_tags_absent_from_a_test_batch() -> None:
    """Per-domain rows are emitted only for tags represented in test custody."""
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    calibrator = fit_residual_interval_calibrator(
        model,
        model.predict(dataset.calibration),
        dataset.calibration,
        alpha=0.2,
    )
    single_tag_test = replace(
        dataset.test,
        domain_tags=(SyntheticDomainTag.SYNTHETIC,) * dataset.test.n_samples,
    )
    interval = apply_residual_interval(calibrator, model.predict(single_tag_test))
    certificate = certify_interval_coverage(
        model,
        calibrator,
        interval,
        single_tag_test,
    )

    assert [row.domain_tag for row in certificate.domains] == [SyntheticDomainTag.SYNTHETIC]


def test_coverage_refuses_wrong_test_and_digest_chains_and_leakage() -> None:
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    calibrator = fit_residual_interval_calibrator(
        model,
        model.predict(dataset.calibration),
        dataset.calibration,
        alpha=0.2,
    )
    test_point = model.predict(dataset.test)
    interval = apply_residual_interval(calibrator, test_point)
    with pytest.raises(ValueError, match="requires a test"):
        certify_interval_coverage(
            model,
            calibrator,
            apply_residual_interval(calibrator, model.predict(dataset.calibration)),
            dataset.calibration,
        )
    with pytest.raises(ValueError, match="custody/shape"):
        certify_interval_coverage(
            model,
            calibrator,
            replace(interval, sample_ids=tuple(reversed(interval.sample_ids))),
            dataset.test,
        )
    with pytest.raises(ValueError, match="model digest chain"):
        certify_interval_coverage(
            model,
            calibrator,
            replace(interval, model_digest="other"),
            dataset.test,
        )
    with pytest.raises(ValueError, match="calibrator digest"):
        certify_interval_coverage(
            model,
            calibrator,
            replace(interval, calibrator_digest="other"),
            dataset.test,
        )
    leaked_test = replace(dataset.test, sample_ids=model.training_sample_ids[:12])
    with pytest.raises(ValueError, match="must be disjoint"):
        certify_interval_coverage(
            model,
            calibrator,
            replace(interval, sample_ids=leaked_test.sample_ids),
            leaked_test,
        )
    target_mask = dataset.test.target_mask.copy()
    target_mask[0] = False
    empty = replace(dataset.test, target_mask=target_mask)
    with pytest.raises(ValueError, match="each test sample"):
        certify_interval_coverage(model, calibrator, interval, empty)


def test_calibrator_record_refuses_invalid_fields_and_leakage() -> None:
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    calibrator = fit_residual_interval_calibrator(
        model,
        model.predict(dataset.calibration),
        dataset.calibration,
        alpha=0.2,
    )
    with pytest.raises(ValueError, match="alpha must"):
        replace(calibrator, alpha=0.0)
    with pytest.raises(ValueError, match="radius"):
        replace(calibrator, radius=-1.0)
    with pytest.raises(ValueError, match="positive"):
        replace(calibrator, calibration_samples=0)
    with pytest.raises(ValueError, match="order_statistic_rank"):
        replace(calibrator, order_statistic_rank=0)
    with pytest.raises(ValueError, match="custody must be disjoint"):
        replace(calibrator, calibration_sample_ids=calibrator.training_sample_ids)
    with pytest.raises(ValueError, match="digests must"):
        replace(calibrator, calibrator_digest="")
