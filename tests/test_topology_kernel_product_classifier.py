# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kernel classifier tests
"""Fit, predict, evaluation, and custody mismatch tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.topology_kernel_product import (
    KernelRidgeClassifier,
    TopologyKernelMatrix,
    evaluate_kernel_ridge,
    fit_kernel_ridge,
    predict_kernel_ridge,
)

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def _kernel(
    values: NDArray[np.float64] | None = None,
    *,
    row_ids: tuple[str, ...] = ("a", "b"),
    column_ids: tuple[str, ...] = ("a", "b"),
    topology_digest: str = DIGEST,
) -> TopologyKernelMatrix:
    return TopologyKernelMatrix(
        values=np.eye(2) if values is None else values,
        row_ids=row_ids,
        column_ids=column_ids,
        topology_digest=topology_digest,
        content_digest=DIGEST,
    )


def _model() -> KernelRidgeClassifier:
    return KernelRidgeClassifier(
        train_ids=("a", "b"),
        coefficients=np.array([1.0, -1.0]),
        alpha=0.1,
        topology_digest=DIGEST,
        training_kernel_digest=DIGEST,
        content_digest=DIGEST,
    )


def test_kernel_ridge_fit_predict_and_evaluate() -> None:
    """Fit, predict, evaluate, and preserve read-only result custody."""
    labels = np.array([1, -1])
    model = fit_kernel_ridge(_kernel(), labels, alpha=0.1)
    predictions = predict_kernel_ridge(model, _kernel())
    result = evaluate_kernel_ridge("identity", model, _kernel(), labels)
    assert predictions.tolist() == [1, -1]
    assert not model.coefficients.flags.writeable
    assert not predictions.flags.writeable
    assert result.correct == 2
    assert result.accuracy == pytest.approx(1.0)


def test_prediction_tie_maps_deterministically_to_positive() -> None:
    """Map an exactly zero decision score deterministically to positive."""
    model = replace(_model(), coefficients=np.zeros(2))
    assert predict_kernel_ridge(model, _kernel()).tolist() == [1, 1]


@pytest.mark.parametrize(
    "changes",
    [
        {"coefficients": np.ones((1, 2))},
        {"coefficients": np.array([1.0])},
        {"train_ids": (), "coefficients": np.array([])},
        {"train_ids": ("a", "a")},
        {"coefficients": np.array([np.nan, 1.0])},
        {"alpha": 0.0},
        {"topology_digest": "bad"},
        {"training_kernel_digest": "bad"},
        {"content_digest": "bad"},
    ],
)
def test_model_rejects_invalid_contract(changes: dict[str, Any]) -> None:
    """Reject malformed immutable classifier fields."""
    with pytest.raises(ValueError):
        replace(_model(), **changes)


def test_fit_rejects_wrong_type_nonsquare_misalignment_labels_and_alpha() -> None:
    """Reject invalid kernel, label, alignment, and regularization inputs."""
    with pytest.raises(ValueError):
        fit_kernel_ridge(object(), np.array([1]), alpha=0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fit_kernel_ridge(
            _kernel(np.ones((2, 1)), column_ids=("a",)),
            np.array([1, -1]),
            alpha=0.1,
        )
    with pytest.raises(ValueError):
        fit_kernel_ridge(
            _kernel(column_ids=("b", "a")),
            np.array([1, -1]),
            alpha=0.1,
        )
    for labels in (np.array([1]), np.array([1, 0])):
        with pytest.raises(ValueError):
            fit_kernel_ridge(_kernel(), labels, alpha=0.1)
    for alpha in (0.0, np.inf):
        with pytest.raises(ValueError):
            fit_kernel_ridge(_kernel(), np.array([1, -1]), alpha=alpha)


def test_fit_rejects_nonfinite_solver_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a non-finite coefficient vector from the numerical solver."""
    monkeypatch.setattr(np.linalg, "solve", lambda *_args: np.array([np.nan, 1.0]))
    with pytest.raises(ValueError, match="non-finite"):
        fit_kernel_ridge(_kernel(), np.array([1, -1]), alpha=0.1)


def test_predict_rejects_wrong_types_identifiers_and_topology() -> None:
    """Reject prediction inputs with broken type or custody alignment."""
    with pytest.raises(ValueError):
        predict_kernel_ridge(object(), _kernel())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        predict_kernel_ridge(_model(), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        predict_kernel_ridge(_model(), _kernel(column_ids=("b", "a")))
    with pytest.raises(ValueError):
        predict_kernel_ridge(_model(), _kernel(topology_digest=OTHER_DIGEST))


def test_evaluate_rejects_bad_test_labels() -> None:
    """Reject evaluation labels with the wrong shape or vocabulary."""
    for labels in (np.array([1]), np.array([1, 0])):
        with pytest.raises(ValueError):
            evaluate_kernel_ridge("bad", _model(), _kernel(), labels)
