# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Gaussian RBF surrogate model tests
"""Tests for the differentiable Gaussian-RBF model surface."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.surrogates import (
    GaussianRBFSurrogate,
    SurrogateFidelityThresholds,
    certify_surrogate_fidelity,
    fit_gaussian_rbf_surrogate,
    input_row_digests,
)


def _model() -> GaussianRBFSurrogate:
    """Return a real fitted-shape model with known analytic behaviour."""
    centres = np.array([[0.0], [1.0]], dtype=np.float64)
    return GaussianRBFSurrogate(
        centres=centres,
        weights=np.array([0.7, -0.2], dtype=np.float64),
        length_scale=0.5,
        intercept=0.1,
        training_input_digests=input_row_digests(centres),
        training_target_digest="a" * 64,
    )


def test_gaussian_rbf_predict_value_and_gradient_are_consistent() -> None:
    """Batch, scalar, and analytic-gradient routes agree by finite difference."""
    model = _model()
    point = np.array([0.35], dtype=np.float64)
    step = np.array([1.0e-6], dtype=np.float64)
    finite_difference = (model.value(point + step) - model.value(point - step)) / 2.0e-6

    assert model.n_parameters == 1
    assert model.predict(np.array([[0.35], [0.55]])).shape == (2,)
    assert model.value(point) == pytest.approx(model.predict(point[None, :])[0])
    assert model.gradient(point)[0] == pytest.approx(finite_difference, abs=1.0e-8)
    assert not model.centres.flags["W"]
    assert not model.weights.flags["W"]
    assert model.to_dict()["model"] == "gaussian_rbf"


@pytest.mark.parametrize("identity_source", ["foreign_rows", "reordered_rows"])
def test_gaussian_rbf_refuses_inconsistent_training_identity(identity_source: str) -> None:
    """Refuse inconsistent centre identities before a held-out score is possible.

    Parameters
    ----------
    identity_source
        Foreign or differently ordered rows whose identities do not bind the centres.

    """
    inputs = np.linspace(-1.0, 1.0, 9, dtype=np.float64)[:, None]
    targets = np.sin(inputs[:, 0])
    model = fit_gaussian_rbf_surrogate(inputs, targets)
    before = model.to_dict()
    predictions = model.predict(inputs)
    with pytest.raises(ValueError, match="overlap"):
        certify_surrogate_fidelity(
            model,
            inputs,
            targets,
            thresholds=SurrogateFidelityThresholds(0.01, 0.02, 0.99),
        )
    rows = inputs + 3.0 if identity_source == "foreign_rows" else inputs[::-1]
    with pytest.raises(ValueError, match="match.*centres"):
        replace(model, training_input_digests=input_row_digests(rows))
    assert model.to_dict() == before
    np.testing.assert_array_equal(model.predict(inputs), predictions)


def test_gaussian_rbf_refuses_nonhex_target_identity() -> None:
    """A length-correct nonhex target identity cannot label a fitted model."""
    model = _model()
    with pytest.raises(ValueError, match="SHA-256"):
        replace(model, training_target_digest="g" * 64)


def test_gaussian_rbf_normalises_matching_hexadecimal_row_identities() -> None:
    """Equivalent uppercase row identities retain the overlap refusal."""
    model = _model()
    uppercase = replace(
        model,
        training_input_digests=tuple(digest.upper() for digest in model.training_input_digests),
        training_target_digest=model.training_target_digest.upper(),
    )
    assert uppercase.training_input_digests == model.training_input_digests
    with pytest.raises(ValueError, match="overlap"):
        certify_surrogate_fidelity(
            uppercase,
            model.centres,
            model.predict(model.centres),
            thresholds=SurrogateFidelityThresholds(0.01, 0.02, 0.99),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("centres", np.array([0.0, 1.0]), "2-D"),
        ("centres", np.empty((0, 1)), "non-empty"),
        ("centres", np.array([[np.nan]]), "finite"),
        ("weights", np.array([[0.1], [0.2]]), "1-D"),
        ("weights", np.array([0.1]), "one value per centre"),
        ("weights", np.array([0.1, np.inf]), "finite"),
        ("length_scale", 0.0, "positive"),
        ("intercept", np.nan, "intercept"),
        ("training_input_digests", ("a" * 64,), "identify every"),
        ("training_input_digests", ("a" * 64, "a" * 64), "unique"),
        ("training_input_digests", ("short", "b" * 64), "SHA-256"),
        ("training_target_digest", "short", "SHA-256"),
        ("claim_boundary", "", "non-empty"),
    ],
)
def test_gaussian_rbf_rejects_invalid_model_contracts(
    field: str,
    value: object,
    message: str,
) -> None:
    """Model construction fails closed for malformed dimensions or provenance.

    Parameters
    ----------
    field
        Public model field whose contract is violated.
    value
        Malformed replacement for that field.
    message
        Expected public validation error.

    """
    values: dict[str, object] = {
        "centres": np.array([[0.0], [1.0]], dtype=np.float64),
        "weights": np.array([0.1, 0.2], dtype=np.float64),
        "length_scale": 0.5,
        "intercept": 0.0,
        "training_input_digests": input_row_digests(np.array([[0.0], [1.0]])),
        "training_target_digest": "c" * 64,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        GaussianRBFSurrogate(**cast(Any, values))


@pytest.mark.parametrize(
    ("method", "values", "message"),
    [
        ("predict", np.array([0.2]), "2-D"),
        ("predict", np.empty((0, 1)), "non-empty"),
        ("predict", np.array([[np.inf]]), "finite"),
        ("predict", np.ones((2, 2)), "parameter dimension"),
        ("value", np.array([[0.2]]), "1-D"),
        ("value", np.array([0.2, 0.3]), "parameter dimension"),
        ("gradient", np.array([np.nan]), "finite"),
        ("gradient", np.array([0.2, 0.3]), "parameter dimension"),
    ],
)
def test_gaussian_rbf_rejects_invalid_evaluation_inputs(
    method: str,
    values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    message: str,
) -> None:
    """Every public evaluation route validates rank, finiteness, and width.

    Parameters
    ----------
    method
        Public evaluation method.
    values
        Malformed parameter vector or input matrix.
    message
        Expected public validation error.

    """
    with pytest.raises(ValueError, match=message):
        getattr(_model(), method)(values)
