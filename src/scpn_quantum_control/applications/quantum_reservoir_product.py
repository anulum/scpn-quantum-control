# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum reservoir product certificates
"""Held-out synthetic QRC certificates and exact reservoir objectives."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .qrc_baseline import QRCHoldoutComparison, compare_quantum_reservoir_to_esn_holdout
from .quantum_reservoir import reservoir_features

FloatArray = NDArray[np.float64]

QRC_PRODUCT_CLAIM_BOUNDARY = (
    "Deterministic synthetic tasks and local exact-statevector features only; "
    "not hardware QRC, unseen-domain generalisation, deployment, or quantum advantage."
)


def _array_digest(values: FloatArray) -> str:
    """Return a shape-bound SHA-256 identity for one float array."""
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class ReservoirTaskKind(str, Enum):
    """Synthetic task families admitted by the reservoir certificate suite."""

    CLASSIFICATION = "classification"
    FORECAST = "forecast"


@dataclass(frozen=True, slots=True)
class SyntheticReservoirDataset:
    """Disjoint synthetic train/validation data for one reservoir task."""

    task_id: str
    task_kind: ReservoirTaskKind
    X_train: FloatArray
    y_train: FloatArray
    X_validation: FloatArray
    y_validation: FloatArray
    domain_tag: str = "synthetic"

    def __post_init__(self) -> None:
        """Validate shapes, finiteness, domain, and disjoint sample rows."""
        x_train = np.asarray(self.X_train, dtype=np.float64)
        y_train = np.asarray(self.y_train, dtype=np.float64)
        x_validation = np.asarray(self.X_validation, dtype=np.float64)
        y_validation = np.asarray(self.y_validation, dtype=np.float64)
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty.")
        if self.domain_tag != "synthetic":
            raise ValueError("reservoir datasets must use the synthetic domain tag.")
        if x_train.ndim != 2 or x_validation.ndim != 2:
            raise ValueError("reservoir inputs must be 2-D arrays.")
        if x_train.shape[0] < 2 or x_validation.shape[0] < 2:
            raise ValueError("train and validation inputs must each contain at least two rows.")
        if x_train.shape[1] == 0 or x_train.shape[1] != x_validation.shape[1]:
            raise ValueError("train and validation inputs must have equal non-zero width.")
        if y_train.shape != (x_train.shape[0],) or y_validation.shape != (x_validation.shape[0],):
            raise ValueError("targets must match their respective input rows.")
        arrays = (x_train, y_train, x_validation, y_validation)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("reservoir dataset arrays must contain only finite values.")
        training_rows = {tuple(float(value) for value in row) for row in x_train}
        validation_rows = {tuple(float(value) for value in row) for row in x_validation}
        if training_rows & validation_rows:
            raise ValueError("train and validation inputs must be disjoint.")
        object.__setattr__(self, "X_train", x_train)
        object.__setattr__(self, "y_train", y_train)
        object.__setattr__(self, "X_validation", x_validation)
        object.__setattr__(self, "y_validation", y_validation)


@dataclass(frozen=True, slots=True)
class ReservoirTrainingCertificate:
    """Digest-bound held-out QRC and matched-feature ESN metrics."""

    task_id: str
    task_kind: str
    n_train: int
    n_validation: int
    n_quantum_features: int
    n_esn_features: int
    quantum_train_mse: float
    quantum_validation_mse: float
    esn_train_mse: float
    esn_validation_mse: float
    validation_mse_delta: float
    lower_validation_mse: str
    training_input_digest: str
    training_target_digest: str
    validation_input_digest: str
    validation_target_digest: str
    domain_tag: str = "synthetic"
    simulator: str = "local_exact_statevector"
    matched_feature_count: bool = True
    hardware_execution: bool = False
    claim_boundary: str = QRC_PRODUCT_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready certificate mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReservoirLinearObjective:
    """Weighted Pauli-feature objective evaluated by the exact QRC path."""

    K: FloatArray
    feature_labels: tuple[str, ...]
    feature_weights: tuple[float, ...]
    omega: FloatArray | None = None
    t: float = 1.0
    max_weight: int = 1
    max_dense_gib: float | None = None
    claim_boundary: str = QRC_PRODUCT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate coupling, objective terms, and simulator parameters."""
        coupling = np.asarray(self.K, dtype=np.float64)
        if coupling.ndim != 2 or coupling.shape[0] == 0 or coupling.shape[0] != coupling.shape[1]:
            raise ValueError("K must be a non-empty square matrix.")
        if not np.all(np.isfinite(coupling)):
            raise ValueError("K must contain only finite values.")
        if not self.feature_labels or len(self.feature_labels) != len(self.feature_weights):
            raise ValueError("feature labels and weights must be matching and non-empty.")
        if len(set(self.feature_labels)) != len(self.feature_labels):
            raise ValueError("feature labels must be unique.")
        if not all(np.isfinite(weight) for weight in self.feature_weights):
            raise ValueError("feature weights must be finite.")
        if self.omega is not None:
            frequencies = np.asarray(self.omega, dtype=np.float64)
            if frequencies.shape != (coupling.shape[0],) or not np.all(np.isfinite(frequencies)):
                raise ValueError("omega must be a finite vector matching K.")
            object.__setattr__(self, "omega", frequencies)
        object.__setattr__(self, "K", coupling)

    def evaluate(self, parameters: FloatArray) -> float:
        """Evaluate the weighted objective through exact statevector features."""
        result = reservoir_features(
            parameters,
            self.K,
            omega=self.omega,
            t=self.t,
            max_weight=self.max_weight,
            max_dense_gib=self.max_dense_gib,
        )
        feature_map = dict(zip(result.feature_labels, result.features, strict=True))
        missing = [label for label in self.feature_labels if label not in feature_map]
        if missing:
            raise ValueError(f"objective feature labels are unavailable: {', '.join(missing)}")
        return float(
            sum(
                weight * float(feature_map[label])
                for label, weight in zip(
                    self.feature_labels,
                    self.feature_weights,
                    strict=True,
                )
            )
        )

    def __call__(self, parameters: FloatArray) -> float:
        """Evaluate the exact local objective."""
        return self.evaluate(parameters)


def generate_synthetic_reservoir_task(
    task_kind: ReservoirTaskKind,
    *,
    n_train: int,
    n_validation: int,
    seed: int,
) -> SyntheticReservoirDataset:
    """Generate a deterministic synthetic forecast or classification task.

    The tasks are small functional certificates, not domain benchmarks. They
    contain no clinical, grid, plasma, private, or operational data.
    """
    if not isinstance(n_train, int) or not isinstance(n_validation, int):
        raise TypeError("n_train and n_validation must be integers.")
    if n_train < 2 or n_validation < 2:
        raise ValueError("n_train and n_validation must each be at least two.")
    if not isinstance(task_kind, ReservoirTaskKind):
        raise TypeError("task_kind must be a ReservoirTaskKind.")

    total = n_train + n_validation
    if task_kind is ReservoirTaskKind.CLASSIFICATION:
        rng = np.random.default_rng(seed)
        inputs = rng.uniform(0.05, 0.95, size=(total, 2)).astype(np.float64)
        score = np.sin(2.0 * np.pi * inputs[:, 0]) + 0.5 * np.cos(2.0 * np.pi * inputs[:, 1])
        targets = (score >= 0.0).astype(np.float64)
        task_id = "synthetic_nonlinear_classification_v1"
    else:
        times = np.linspace(0.0, 5.0, total + 1, dtype=np.float64)
        inputs = np.column_stack((np.sin(times[:-1]), np.cos(0.7 * times[:-1])))
        targets = (0.65 * np.sin(times[1:]) + 0.35 * np.cos(0.7 * times[1:])).astype(np.float64)
        task_id = "synthetic_one_step_forecast_v1"
    return SyntheticReservoirDataset(
        task_id=task_id,
        task_kind=task_kind,
        X_train=inputs[:n_train],
        y_train=targets[:n_train],
        X_validation=inputs[n_train:],
        y_validation=targets[n_train:],
    )


def certify_reservoir_training(
    dataset: SyntheticReservoirDataset,
    K: FloatArray,
    *,
    omega: FloatArray | None = None,
    alpha: float = 0.1,
    max_weight: int = 1,
    t: float = 1.0,
    seed: int = 0,
    max_dense_gib: float | None = None,
) -> ReservoirTrainingCertificate:
    """Fit QRC/ESN readouts and certify their disjoint held-out metrics."""
    comparison: QRCHoldoutComparison = compare_quantum_reservoir_to_esn_holdout(
        dataset.X_train,
        dataset.y_train,
        dataset.X_validation,
        dataset.y_validation,
        K,
        omega=omega,
        alpha=alpha,
        max_weight=max_weight,
        t=t,
        seed=seed,
        max_dense_gib=max_dense_gib,
    )
    tolerance = np.finfo(np.float64).eps * 16.0
    if abs(comparison.validation_mse_delta) <= tolerance:
        lower = "tie_within_float_tolerance"
    elif comparison.validation_mse_delta < 0.0:
        lower = "qrc"
    else:
        lower = "esn"
    return ReservoirTrainingCertificate(
        task_id=dataset.task_id,
        task_kind=dataset.task_kind.value,
        n_train=comparison.n_train,
        n_validation=comparison.n_validation,
        n_quantum_features=comparison.n_quantum_features,
        n_esn_features=comparison.n_esn_features,
        quantum_train_mse=comparison.quantum_train_mse,
        quantum_validation_mse=comparison.quantum_validation_mse,
        esn_train_mse=comparison.esn_train_mse,
        esn_validation_mse=comparison.esn_validation_mse,
        validation_mse_delta=comparison.validation_mse_delta,
        lower_validation_mse=lower,
        training_input_digest=_array_digest(dataset.X_train),
        training_target_digest=_array_digest(dataset.y_train),
        validation_input_digest=_array_digest(dataset.X_validation),
        validation_target_digest=_array_digest(dataset.y_validation),
        matched_feature_count=comparison.n_quantum_features == comparison.n_esn_features,
    )


__all__ = [
    "QRC_PRODUCT_CLAIM_BOUNDARY",
    "ReservoirLinearObjective",
    "ReservoirTaskKind",
    "ReservoirTrainingCertificate",
    "SyntheticReservoirDataset",
    "certify_reservoir_training",
    "generate_synthetic_reservoir_task",
]
