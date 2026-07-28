# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum reservoir product tests
"""Tests for held-out QRC certificates and exact reservoir objectives."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.applications import (
    ReservoirLinearObjective,
    ReservoirTaskKind,
    ReservoirTrainingCertificate,
    SyntheticReservoirDataset,
    certify_reservoir_training,
    generate_synthetic_reservoir_task,
)


def _coupling() -> NDArray[np.float64]:
    """Return the bounded two-qubit product coupling."""
    return np.array([[0.0, 0.6], [0.6, 0.0]], dtype=np.float64)


@pytest.mark.parametrize(
    "task_kind",
    [ReservoirTaskKind.CLASSIFICATION, ReservoirTaskKind.FORECAST],
)
def test_synthetic_task_certifies_real_qrc_and_esn_surfaces(
    task_kind: ReservoirTaskKind,
) -> None:
    """Both admitted task families produce disjoint held-out certificates."""
    dataset = generate_synthetic_reservoir_task(
        task_kind,
        n_train=8,
        n_validation=4,
        seed=17,
    )
    certificate = certify_reservoir_training(
        dataset,
        _coupling(),
        omega=np.array([0.1, -0.1]),
        alpha=0.2,
        max_weight=1,
        t=0.5,
        seed=17,
    )

    assert isinstance(certificate, ReservoirTrainingCertificate)
    assert certificate.task_kind == task_kind.value
    assert certificate.n_train == 8
    assert certificate.n_validation == 4
    assert certificate.n_quantum_features == certificate.n_esn_features == 6
    assert certificate.matched_feature_count
    assert certificate.lower_validation_mse in {"qrc", "esn", "tie_within_float_tolerance"}
    assert len(certificate.training_input_digest) == 64
    assert len(certificate.validation_target_digest) == 64
    assert certificate.to_dict()["hardware_execution"] is False


def test_synthetic_dataset_rejects_non_synthetic_and_overlapping_data() -> None:
    """The product dataset cannot admit domain-labelled or leaked validation rows."""
    inputs = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    targets = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="synthetic domain"):
        SyntheticReservoirDataset(
            task_id="bad-domain",
            task_kind=ReservoirTaskKind.CLASSIFICATION,
            X_train=inputs,
            y_train=targets,
            X_validation=inputs + 0.1,
            y_validation=targets,
            domain_tag="eeg_like_sim",
        )
    with pytest.raises(ValueError, match="disjoint"):
        SyntheticReservoirDataset(
            task_id="leaked",
            task_kind=ReservoirTaskKind.CLASSIFICATION,
            X_train=inputs,
            y_train=targets,
            X_validation=inputs,
            y_validation=targets,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task_id": ""}, "task_id"),
        ({"X_train": np.array([0.1, 0.2])}, "2-D"),
        ({"X_validation": np.array([[0.1, 0.2]])}, "at least two rows"),
        ({"X_validation": np.ones((2, 3))}, "equal non-zero width"),
        ({"y_train": np.array([0.0])}, "targets"),
        ({"y_validation": np.array([0.0, np.inf])}, "finite"),
    ],
)
def test_synthetic_dataset_rejects_malformed_arrays(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Dataset construction validates every public shape and finiteness boundary."""
    base: dict[str, object] = {
        "task_id": "valid",
        "task_kind": ReservoirTaskKind.CLASSIFICATION,
        "X_train": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "y_train": np.array([0.0, 1.0]),
        "X_validation": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "y_validation": np.array([1.0, 0.0]),
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SyntheticReservoirDataset(**cast(Any, base))


def test_synthetic_task_generation_rejects_invalid_requests() -> None:
    """Task generation validates sizes, integer types, and the task enum."""
    with pytest.raises(TypeError, match="must be integers"):
        generate_synthetic_reservoir_task(
            ReservoirTaskKind.FORECAST,
            n_train=cast(int, 3.5),
            n_validation=2,
            seed=0,
        )
    with pytest.raises(ValueError, match="at least two"):
        generate_synthetic_reservoir_task(
            ReservoirTaskKind.FORECAST,
            n_train=1,
            n_validation=2,
            seed=0,
        )
    with pytest.raises(TypeError, match="ReservoirTaskKind"):
        generate_synthetic_reservoir_task(
            cast(ReservoirTaskKind, "forecast"),
            n_train=2,
            n_validation=2,
            seed=0,
        )


def test_exact_reservoir_linear_objective_uses_named_pauli_features() -> None:
    """The objective evaluates the real exact-statevector reservoir path."""
    objective = ReservoirLinearObjective(
        K=_coupling(),
        omega=np.array([0.1, -0.1]),
        feature_labels=("IZ", "ZI"),
        feature_weights=(0.6, -0.4),
        t=0.5,
        max_weight=1,
    )
    point = np.array([0.25, 0.7], dtype=np.float64)

    assert objective(point) == pytest.approx(objective.evaluate(point))
    assert np.isfinite(objective(point))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"K": np.ones((2, 3))}, "square"),
        ({"K": np.array([[0.0, np.nan], [0.0, 0.0]])}, "finite"),
        ({"feature_labels": (), "feature_weights": ()}, "matching and non-empty"),
        ({"feature_labels": ("IZ", "IZ"), "feature_weights": (0.5, 0.5)}, "unique"),
        ({"feature_weights": (np.nan, -0.4)}, "finite"),
        ({"omega": np.ones(3)}, "matching K"),
    ],
)
def test_exact_reservoir_linear_objective_rejects_invalid_contracts(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """The exact objective fails closed on malformed configuration."""
    base: dict[str, object] = {
        "K": _coupling(),
        "feature_labels": ("IZ", "ZI"),
        "feature_weights": (0.6, -0.4),
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ReservoirLinearObjective(**cast(Any, base))


def test_exact_reservoir_linear_objective_rejects_unavailable_feature() -> None:
    """A requested feature outside the configured weight fails closed."""
    objective = ReservoirLinearObjective(
        K=_coupling(),
        feature_labels=("XX",),
        feature_weights=(1.0,),
        max_weight=1,
    )
    with pytest.raises(ValueError, match="unavailable"):
        objective(np.array([0.2, 0.4]))


def test_reservoir_certificate_records_an_exact_float_tie() -> None:
    """Equal zero-target MSE is labelled as a tie without inventing a winner."""
    dataset = SyntheticReservoirDataset(
        task_id="zero_target_tie",
        task_kind=ReservoirTaskKind.FORECAST,
        X_train=np.array([[0.1, 0.2], [0.3, 0.4]]),
        y_train=np.zeros(2),
        X_validation=np.array([[0.5, 0.6], [0.7, 0.8]]),
        y_validation=np.zeros(2),
    )
    certificate = certify_reservoir_training(dataset, _coupling(), max_weight=1)
    assert certificate.validation_mse_delta == 0.0
    assert certificate.lower_validation_mse == "tie_within_float_tolerance"
