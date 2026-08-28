# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synthetic forecast uncertainty calibration
"""Split sample-level residual intervals for independent multimodal-forecasting trajectories."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .multimodal_forecaster import MultimodalPointForecast, MultimodalRidgeForecaster
from .multimodal_schema import MultimodalObservationBatch, SyntheticDomainTag

FloatArray = NDArray[np.float64]
_NUMERIC_CUSTODY_DECIMALS = 12


def _canonical_digest_float_bytes(value: float) -> bytes:
    """Return platform-stable bytes at the forecasting custody precision."""
    rounded = round(float(value), _NUMERIC_CUSTODY_DECIMALS)
    canonical = 0.0 if rounded == 0.0 else rounded
    return np.asarray(canonical, dtype="<f8").tobytes()


def _immutable(values: object, *, name: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 3 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite rank-three array")
    array.setflags(write=False)
    return array


def _wrapped_absolute_error(left: FloatArray, right: FloatArray) -> FloatArray:
    return np.abs((left - right + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True, slots=True)
class ResidualIntervalCalibrator:
    """Sample-level split residual radius and custody."""

    alpha: float
    radius: float
    calibration_samples: int
    order_statistic_rank: int
    model_digest: str
    training_sample_ids: tuple[str, ...]
    calibration_sample_ids: tuple[str, ...]
    calibration_batch_digest: str
    calibrator_digest: str
    claim_boundary: str = (
        "sample-level split residual interval over independent synthetic trajectories; "
        "not sequential EnbPI, conditional coverage, domain transfer, or deployment evidence"
    )

    def __post_init__(self) -> None:
        """Validate finite calibration controls and disjoint custody."""
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be finite and in (0, 1)")
        if not np.isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("radius must be finite and non-negative")
        if self.calibration_samples < 1:
            raise ValueError("calibration_samples must be positive")
        if not 1 <= self.order_statistic_rank <= self.calibration_samples:
            raise ValueError("order_statistic_rank must index the calibration scores")
        if set(self.training_sample_ids).intersection(self.calibration_sample_ids):
            raise ValueError("training/calibration sample custody must be disjoint")
        if not all(
            value.strip()
            for value in (
                self.model_digest,
                self.calibration_batch_digest,
                self.calibrator_digest,
            )
        ):
            raise ValueError("model, batch, and calibrator digests must be non-empty")

    @property
    def target_coverage(self) -> float:
        """Return the nominal independent-sample coverage target."""
        return 1.0 - self.alpha

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready calibration record without duplicate ID arrays."""
        return {
            "alpha": self.alpha,
            "target_coverage": self.target_coverage,
            "radius": self.radius,
            "calibration_samples": self.calibration_samples,
            "order_statistic_rank": self.order_statistic_rank,
            "model_digest": self.model_digest,
            "calibration_batch_digest": self.calibration_batch_digest,
            "calibrator_digest": self.calibrator_digest,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MultimodalIntervalForecast:
    """Point, lower, and upper forecasts bound to one calibrator."""

    point: FloatArray
    lower: FloatArray
    upper: FloatArray
    sample_ids: tuple[str, ...]
    model_digest: str
    calibrator_digest: str

    def __post_init__(self) -> None:
        """Validate aligned finite interval tensors and immutable custody."""
        point = _immutable(self.point, name="point")
        lower = _immutable(self.lower, name="lower")
        upper = _immutable(self.upper, name="upper")
        if point.shape != lower.shape or point.shape != upper.shape:
            raise ValueError("point, lower, and upper forecasts must share shape")
        if np.any(lower > point) or np.any(point > upper):
            raise ValueError("interval bounds must contain their point forecasts")
        if point.shape[0] != len(self.sample_ids) or len(set(self.sample_ids)) != len(
            self.sample_ids
        ):
            raise ValueError("sample_ids must uniquely match the interval sample dimension")
        if not self.model_digest.strip() or not self.calibrator_digest.strip():
            raise ValueError("model and calibrator digests must be non-empty")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class DomainIntervalCoverage:
    """Empirical held-out interval coverage for one synthetic tag."""

    domain_tag: SyntheticDomainTag
    samples: int
    sample_coverage: float
    value_coverage: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready tag coverage row."""
        return asdict(self) | {"domain_tag": self.domain_tag.value}


@dataclass(frozen=True, slots=True)
class IntervalCoverageCertificate:
    """Empirical test coverage under disjoint synthetic trajectory custody."""

    samples: int
    target_coverage: float
    sample_coverage: float
    value_coverage: float
    mean_interval_width: float
    sample_target_met: bool
    domains: tuple[DomainIntervalCoverage, ...]
    model_digest: str
    calibrator_digest: str
    test_batch_digest: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready empirical coverage certificate."""
        payload = asdict(self)
        payload["domains"] = [row.to_dict() for row in self.domains]
        return payload


def fit_residual_interval_calibrator(
    model: MultimodalRidgeForecaster,
    forecast: MultimodalPointForecast,
    calibration: MultimodalObservationBatch,
    *,
    alpha: float = 0.10,
) -> ResidualIntervalCalibrator:
    """Fit a sample-max split residual radius on calibration-only custody."""
    if calibration.split != "calibration":
        raise ValueError("interval fitting requires a calibration batch")
    if forecast.model_digest != model.model_digest:
        raise ValueError("forecast model digest does not match the fitted model")
    if (
        forecast.sample_ids != calibration.sample_ids
        or forecast.values.shape != calibration.targets.shape
    ):
        raise ValueError("forecast and calibration custody/shape must match exactly")
    if set(model.training_sample_ids).intersection(calibration.sample_ids):
        raise ValueError("training and calibration sample IDs must be disjoint")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    rank = int(np.ceil((calibration.n_samples + 1) * (1.0 - alpha)))
    if rank > calibration.n_samples:
        raise ValueError("alpha is too small for a finite calibration order statistic")
    errors = _wrapped_absolute_error(forecast.values, calibration.targets)
    scores = np.empty(calibration.n_samples, dtype=np.float64)
    for sample in range(calibration.n_samples):
        selected = errors[sample][calibration.target_mask[sample]]
        if selected.size == 0:
            raise ValueError("each calibration sample requires an observed target")
        scores[sample] = float(np.max(selected))
    radius = float(np.sort(scores)[rank - 1])
    digest = hashlib.sha256()
    digest.update(b"scpn.residual_interval_calibrator.v1\0")
    digest.update(model.model_digest.encode("ascii"))
    digest.update(calibration.content_digest().encode("ascii"))
    digest.update(_canonical_digest_float_bytes(alpha))
    digest.update(_canonical_digest_float_bytes(radius))
    return ResidualIntervalCalibrator(
        alpha=alpha,
        radius=radius,
        calibration_samples=calibration.n_samples,
        order_statistic_rank=rank,
        model_digest=model.model_digest,
        training_sample_ids=model.training_sample_ids,
        calibration_sample_ids=calibration.sample_ids,
        calibration_batch_digest=calibration.content_digest(),
        calibrator_digest=digest.hexdigest(),
    )


def apply_residual_interval(
    calibrator: ResidualIntervalCalibrator,
    forecast: MultimodalPointForecast,
) -> MultimodalIntervalForecast:
    """Apply a frozen symmetric local-phase residual radius."""
    if forecast.model_digest != calibrator.model_digest:
        raise ValueError("forecast model digest does not match the calibrator")
    return MultimodalIntervalForecast(
        point=forecast.values,
        lower=forecast.values - calibrator.radius,
        upper=forecast.values + calibrator.radius,
        sample_ids=forecast.sample_ids,
        model_digest=forecast.model_digest,
        calibrator_digest=calibrator.calibrator_digest,
    )


def certify_interval_coverage(
    model: MultimodalRidgeForecaster,
    calibrator: ResidualIntervalCalibrator,
    interval: MultimodalIntervalForecast,
    test: MultimodalObservationBatch,
) -> IntervalCoverageCertificate:
    """Measure disjoint held-out sample and phase-value coverage."""
    if test.split != "test":
        raise ValueError("coverage certification requires a test batch")
    if interval.sample_ids != test.sample_ids or interval.point.shape != test.targets.shape:
        raise ValueError("interval and test custody/shape must match exactly")
    if (
        interval.model_digest != model.model_digest
        or interval.model_digest != calibrator.model_digest
    ):
        raise ValueError("model digest chain is inconsistent")
    if interval.calibrator_digest != calibrator.calibrator_digest:
        raise ValueError("interval calibrator digest is inconsistent")
    occupied = set(model.training_sample_ids).union(calibrator.calibration_sample_ids)
    if occupied.intersection(test.sample_ids):
        raise ValueError("train, calibration, and test sample IDs must be disjoint")
    errors = _wrapped_absolute_error(interval.point, test.targets)
    covered = errors <= calibrator.radius
    selected_covered = covered[test.target_mask]
    sample_rows = np.ones(test.n_samples, dtype=np.bool_)
    for sample in range(test.n_samples):
        selected = covered[sample][test.target_mask[sample]]
        if selected.size == 0:
            raise ValueError("each test sample requires an observed target")
        sample_rows[sample] = bool(np.all(selected))
    domains: list[DomainIntervalCoverage] = []
    for tag in SyntheticDomainTag:
        tag_rows = np.fromiter((value is tag for value in test.domain_tags), dtype=np.bool_)
        if not np.any(tag_rows):
            continue
        tag_mask = np.logical_and(test.target_mask, tag_rows[:, None, None])
        domains.append(
            DomainIntervalCoverage(
                domain_tag=tag,
                samples=int(np.count_nonzero(tag_rows)),
                sample_coverage=float(np.mean(sample_rows[tag_rows])),
                value_coverage=float(np.mean(covered[tag_mask])),
            )
        )
    sample_coverage = float(np.mean(sample_rows))
    return IntervalCoverageCertificate(
        samples=test.n_samples,
        target_coverage=calibrator.target_coverage,
        sample_coverage=sample_coverage,
        value_coverage=float(np.mean(selected_covered)),
        mean_interval_width=2.0 * calibrator.radius,
        sample_target_met=sample_coverage >= calibrator.target_coverage,
        domains=tuple(domains),
        model_digest=model.model_digest,
        calibrator_digest=calibrator.calibrator_digest,
        test_batch_digest=test.content_digest(),
        claim_boundary=calibrator.claim_boundary,
    )


__all__ = [
    "DomainIntervalCoverage",
    "IntervalCoverageCertificate",
    "MultimodalIntervalForecast",
    "ResidualIntervalCalibrator",
    "apply_residual_interval",
    "certify_interval_coverage",
    "fit_residual_interval_calibrator",
]
