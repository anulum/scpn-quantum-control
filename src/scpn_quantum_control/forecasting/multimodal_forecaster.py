# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Missingness-aware multimodal ridge forecaster
"""Classical reference forecasting over immutable multimodal-forecasting multimodal batches."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .multimodal_schema import MultimodalObservationBatch, SyntheticDomainTag

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_NUMERIC_CUSTODY_DECIMALS = 12


def _canonical_digest_array_bytes(values: FloatArray) -> bytes:
    """Return platform-stable bytes at the forecasting custody precision."""
    rounded = np.round(np.asarray(values, dtype=np.float64), _NUMERIC_CUSTODY_DECIMALS)
    canonical = np.where(rounded == 0.0, 0.0, rounded).astype("<f8", copy=False)
    return canonical.tobytes(order="C")


def _immutable(values: object, *, name: str, ndim: int) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != ndim or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite rank-{ndim} array")
    array.setflags(write=False)
    return array


def _require_matrix_budget(rows: int, columns: int, *, max_dense_gib: float) -> None:
    if not np.isfinite(max_dense_gib) or max_dense_gib <= 0.0:
        raise ValueError("max_dense_gib must be finite and positive")
    bytes_required = rows * columns * np.dtype(np.float64).itemsize
    if bytes_required > max_dense_gib * 1024**3:
        gib = bytes_required / 1024**3
        raise MemoryError(
            f"multimodal dense matrix requires {gib:.6f} GiB, above {max_dense_gib:.6f} GiB"
        )


def _raw_features(batch: MultimodalObservationBatch) -> tuple[FloatArray, BoolArray]:
    count = batch.n_samples
    values = np.concatenate(
        (
            batch.series.reshape(count, -1),
            batch.graphs.reshape(count, -1),
            batch.events.reshape(count, -1),
            batch.frequencies,
        ),
        axis=1,
    )
    mask = np.concatenate(
        (
            batch.series_mask.reshape(count, -1),
            batch.graph_mask.reshape(count, -1),
            batch.event_mask.reshape(count, -1),
            np.ones_like(batch.frequencies, dtype=np.bool_),
        ),
        axis=1,
    )
    return values, mask


def _domain_features(batch: MultimodalObservationBatch) -> FloatArray:
    tags = tuple(SyntheticDomainTag)
    encoded = np.zeros((batch.n_samples, len(tags)), dtype=np.float64)
    for row, tag in enumerate(batch.domain_tags):
        encoded[row, tags.index(tag)] = 1.0
    return encoded


@dataclass(frozen=True, slots=True)
class MultimodalPointForecast:
    """Point forecasts bound to model and sample custody."""

    values: FloatArray
    sample_ids: tuple[str, ...]
    model_digest: str

    def __post_init__(self) -> None:
        """Validate and freeze forecast values and identifiers."""
        values = _immutable(self.values, name="values", ndim=3)
        if values.shape[0] != len(self.sample_ids):
            raise ValueError("forecast sample_ids must match the sample dimension")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("forecast sample_ids must be unique")
        if not self.model_digest.strip():
            raise ValueError("model_digest must be non-empty")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class DomainForecastAccuracy:
    """Held-out accuracy for one simulation-only domain tag."""

    domain_tag: SyntheticDomainTag
    samples: int
    wrapped_mse: float
    wrapped_mae: float
    persistence_wrapped_mse: float
    lower_mse_than_persistence: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready domain accuracy row."""
        return asdict(self) | {"domain_tag": self.domain_tag.value}


@dataclass(frozen=True, slots=True)
class ForecastAccuracyCertificate:
    """Held-out point-forecast comparison against phase persistence."""

    samples: int
    wrapped_mse: float
    wrapped_mae: float
    persistence_wrapped_mse: float
    lower_mse_than_persistence: bool
    input_missing_fraction: float
    domains: tuple[DomainForecastAccuracy, ...]
    model_digest: str
    batch_digest: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready held-out accuracy certificate."""
        payload = asdict(self)
        payload["domains"] = [row.to_dict() for row in self.domains]
        return payload


@dataclass(frozen=True, slots=True)
class MultimodalRidgeForecaster:
    """Missingness-aware linear ridge reference model.

    Missing values are replaced by training-only feature means. The complete
    observation mask and synthetic-domain one-hot encoding remain explicit
    model inputs; calibration and test values never determine imputation or
    scaling statistics.
    """

    feature_means: FloatArray
    feature_scales: FloatArray
    coefficients: FloatArray
    history_steps: int
    horizon_steps: int
    n_nodes: int
    event_channels: int
    dt: float
    ridge: float
    training_batch_digest: str
    training_sample_ids: tuple[str, ...]
    model_digest: str

    def __post_init__(self) -> None:
        """Validate and freeze fitted model custody."""
        means = _immutable(self.feature_means, name="feature_means", ndim=1)
        scales = _immutable(self.feature_scales, name="feature_scales", ndim=1)
        coefficients = _immutable(self.coefficients, name="coefficients", ndim=2)
        if means.shape != scales.shape or np.any(scales <= 0.0):
            raise ValueError("feature means/scales must align and scales must be positive")
        expected_raw = (
            self.history_steps * self.n_nodes
            + self.n_nodes**2
            + self.history_steps * self.event_channels
            + self.n_nodes
        )
        if means.size != expected_raw:
            raise ValueError("feature statistics do not match model shape metadata")
        expected_design = 2 * means.size + len(SyntheticDomainTag) + 1
        if coefficients.shape != (expected_design, self.horizon_steps * self.n_nodes):
            raise ValueError("coefficient shape does not match feature and output dimensions")
        if self.history_steps < 2 or self.n_nodes < 2:
            raise ValueError("model shape metadata is invalid")
        if self.event_channels < 1:
            raise ValueError("event_channels must be positive")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        if not self.training_batch_digest.strip() or not self.model_digest.strip():
            raise ValueError("training and model digests must be non-empty")
        if not self.training_sample_ids or len(set(self.training_sample_ids)) != len(
            self.training_sample_ids
        ):
            raise ValueError("training_sample_ids must be non-empty and unique")
        object.__setattr__(self, "feature_means", means)
        object.__setattr__(self, "feature_scales", scales)
        object.__setattr__(self, "coefficients", coefficients)

    def _validate_batch(self, batch: MultimodalObservationBatch) -> None:
        if (
            batch.history_steps,
            batch.horizon_steps,
            batch.n_nodes,
            batch.event_channels,
        ) != (self.history_steps, self.horizon_steps, self.n_nodes, self.event_channels):
            raise ValueError("batch shape is incompatible with the fitted forecaster")
        if not np.isclose(batch.dt, self.dt, rtol=0.0, atol=0.0):
            raise ValueError("batch dt is incompatible with the fitted forecaster")

    def design_matrix(self, batch: MultimodalObservationBatch) -> FloatArray:
        """Build a finite inference design matrix from frozen training statistics."""
        self._validate_batch(batch)
        raw, mask = _raw_features(batch)
        imputed = np.where(mask, raw, self.feature_means[None, :])
        standardised = (imputed - self.feature_means[None, :]) / self.feature_scales[None, :]
        return np.concatenate(
            (
                standardised,
                mask.astype(np.float64),
                _domain_features(batch),
                np.ones((batch.n_samples, 1), dtype=np.float64),
            ),
            axis=1,
        )

    def predict(self, batch: MultimodalObservationBatch) -> MultimodalPointForecast:
        """Forecast one compatible batch without changing model state."""
        design = self.design_matrix(batch)
        increments = (design @ self.coefficients).reshape(
            batch.n_samples, self.horizon_steps, self.n_nodes
        )
        values = _last_observed(batch)[:, None, :] + increments
        return MultimodalPointForecast(values, batch.sample_ids, self.model_digest)


def fit_multimodal_ridge_forecaster(
    train: MultimodalObservationBatch,
    *,
    ridge: float = 1.0e-3,
    max_dense_gib: float = 0.25,
) -> MultimodalRidgeForecaster:
    """Fit the classical forecasting reference model from train-only custody."""
    if train.split != "train":
        raise ValueError("forecaster fitting requires a train batch")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    raw, mask = _raw_features(train)
    observed_counts = np.sum(mask, axis=0)
    if np.any(observed_counts == 0):
        raise ValueError("every raw feature requires at least one observed training value")
    feature_means = np.sum(np.where(mask, raw, 0.0), axis=0) / observed_counts
    imputed = np.where(mask, raw, feature_means[None, :])
    feature_scales = np.std(imputed, axis=0)
    feature_scales = np.where(feature_scales > np.finfo(np.float64).eps, feature_scales, 1.0)
    standardised = (imputed - feature_means[None, :]) / feature_scales[None, :]
    design = np.concatenate(
        (
            standardised,
            mask.astype(np.float64),
            _domain_features(train),
            np.ones((train.n_samples, 1), dtype=np.float64),
        ),
        axis=1,
    )
    anchor = _last_observed(train)[:, None, :]
    output = _wrapped_errors(train.targets, anchor).reshape(train.n_samples, -1)
    output_mask = train.target_mask.reshape(train.n_samples, -1)
    _require_matrix_budget(
        train.n_samples + design.shape[1], design.shape[1], max_dense_gib=max_dense_gib
    )
    coefficients = np.empty((design.shape[1], output.shape[1]), dtype=np.float64)
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    for column in range(output.shape[1]):
        valid = output_mask[:, column]
        if np.count_nonzero(valid) < 2:
            raise ValueError(
                "every target coordinate requires at least two observed training rows"
            )
        selected = design[valid]
        gram = selected.T @ selected + penalty
        coefficients[:, column] = np.linalg.solve(gram, selected.T @ output[valid, column])
    digest = hashlib.sha256()
    digest.update(b"scpn.multimodal_ridge_forecaster.v1\0")
    digest.update(train.content_digest().encode("ascii"))
    digest.update(_canonical_digest_array_bytes(np.asarray([ridge], dtype=np.float64)))
    digest.update(_canonical_digest_array_bytes(feature_means))
    digest.update(_canonical_digest_array_bytes(feature_scales))
    digest.update(_canonical_digest_array_bytes(coefficients))
    model_digest = digest.hexdigest()
    return MultimodalRidgeForecaster(
        feature_means=feature_means,
        feature_scales=feature_scales,
        coefficients=coefficients,
        history_steps=train.history_steps,
        horizon_steps=train.horizon_steps,
        n_nodes=train.n_nodes,
        event_channels=train.event_channels,
        dt=train.dt,
        ridge=ridge,
        training_batch_digest=train.content_digest(),
        training_sample_ids=train.sample_ids,
        model_digest=model_digest,
    )


def _wrapped_errors(forecast: FloatArray, target: FloatArray) -> FloatArray:
    return (forecast - target + np.pi) % (2.0 * np.pi) - np.pi


def _last_observed(batch: MultimodalObservationBatch) -> FloatArray:
    last = np.empty((batch.n_samples, batch.n_nodes), dtype=np.float64)
    for sample in range(batch.n_samples):
        for node in range(batch.n_nodes):
            observed = np.flatnonzero(batch.series_mask[sample, :, node])
            if observed.size == 0:
                raise ValueError("persistence requires one observed history value per sample/node")
            last[sample, node] = batch.series[sample, observed[-1], node]
    return last


def _persistence(batch: MultimodalObservationBatch) -> FloatArray:
    return np.repeat(_last_observed(batch)[:, None, :], batch.horizon_steps, axis=1)


def evaluate_point_forecast(
    forecast: MultimodalPointForecast,
    batch: MultimodalObservationBatch,
) -> ForecastAccuracyCertificate:
    """Evaluate held-out wrapped phase errors against persistence."""
    if forecast.sample_ids != batch.sample_ids:
        raise ValueError("forecast and batch sample custody must match exactly")
    if forecast.values.shape != batch.targets.shape:
        raise ValueError("forecast and target shapes must match")
    mask = batch.target_mask
    errors = _wrapped_errors(forecast.values, batch.targets)
    persistence_errors = _wrapped_errors(_persistence(batch), batch.targets)
    wrapped_mse = float(np.mean(np.square(errors[mask])))
    wrapped_mae = float(np.mean(np.abs(errors[mask])))
    persistence_mse = float(np.mean(np.square(persistence_errors[mask])))
    domains: list[DomainForecastAccuracy] = []
    for tag in SyntheticDomainTag:
        sample_mask = np.fromiter((value is tag for value in batch.domain_tags), dtype=np.bool_)
        selected_mask = np.logical_and(mask, sample_mask[:, None, None])
        if not np.any(selected_mask):
            continue
        domain_mse = float(np.mean(np.square(errors[selected_mask])))
        domain_persistence = float(np.mean(np.square(persistence_errors[selected_mask])))
        domains.append(
            DomainForecastAccuracy(
                domain_tag=tag,
                samples=int(np.count_nonzero(sample_mask)),
                wrapped_mse=domain_mse,
                wrapped_mae=float(np.mean(np.abs(errors[selected_mask]))),
                persistence_wrapped_mse=domain_persistence,
                lower_mse_than_persistence=domain_mse < domain_persistence,
            )
        )
    return ForecastAccuracyCertificate(
        samples=batch.n_samples,
        wrapped_mse=wrapped_mse,
        wrapped_mae=wrapped_mae,
        persistence_wrapped_mse=persistence_mse,
        lower_mse_than_persistence=wrapped_mse < persistence_mse,
        input_missing_fraction=batch.missing_fraction,
        domains=tuple(domains),
        model_digest=forecast.model_digest,
        batch_digest=batch.content_digest(),
    )


__all__ = [
    "DomainForecastAccuracy",
    "ForecastAccuracyCertificate",
    "MultimodalPointForecast",
    "MultimodalRidgeForecaster",
    "evaluate_point_forecast",
    "fit_multimodal_ridge_forecaster",
]
