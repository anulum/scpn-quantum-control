# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Gaussian RBF surrogate training tests
"""Tests for deterministic Gaussian-RBF surrogate fitting."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.surrogates import (
    SurrogateFitConfig,
    fit_gaussian_rbf_surrogate,
    input_row_digests,
)


def test_fit_gaussian_rbf_surrogate_interpolates_training_surface() -> None:
    """The real ridge-fit path closely reproduces a smooth training objective."""
    inputs = np.linspace(-1.0, 1.0, 7, dtype=np.float64)[:, None]
    targets = np.sin(inputs[:, 0])
    model = fit_gaussian_rbf_surrogate(inputs, targets)

    np.testing.assert_allclose(model.predict(inputs), targets, atol=2.0e-6)
    assert model.length_scale > 0.0
    assert len(model.training_input_digests) == inputs.shape[0]
    assert len(model.training_target_digest) == 64


def test_fit_gaussian_rbf_surrogate_honours_explicit_configuration() -> None:
    """Explicit regularisation and length scale remain deterministic."""
    inputs = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)
    targets = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    config = SurrogateFitConfig(regularisation=1.0e-5, length_scale=0.3)
    first = fit_gaussian_rbf_surrogate(inputs, targets, config=config)
    second = fit_gaussian_rbf_surrogate(inputs, targets, config=config)

    assert first.length_scale == 0.3
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.training_target_digest == second.training_target_digest


def test_input_row_digests_bind_values_shape_and_order() -> None:
    """Row identities are stable and sensitive to content changes."""
    inputs = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    first = input_row_digests(inputs)
    second = input_row_digests(inputs.copy())
    changed = input_row_digests(inputs + 0.1)

    assert first == second
    assert first != changed
    assert all(len(digest) == 64 for digest in first)


def test_input_row_digests_normalise_subprecision_and_signed_zero() -> None:
    """Provenance hashes ignore insignificant runtime drift."""
    left = np.array([[-0.0, 0.12345678901231]], dtype=np.float64)
    right = np.array([[1.0e-14, 0.12345678901229]], dtype=np.float64)
    assert input_row_digests(left) == input_row_digests(right)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"regularisation": 0.0}, "regularisation"),
        ({"regularisation": np.inf}, "regularisation"),
        ({"length_scale": 0.0}, "length_scale"),
        ({"length_scale": np.nan}, "length_scale"),
    ],
)
def test_surrogate_fit_config_rejects_invalid_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    """Fit configuration requires finite positive hyperparameters."""
    with pytest.raises(ValueError, match=message):
        SurrogateFitConfig(**kwargs)


@pytest.mark.parametrize(
    ("inputs", "targets", "message"),
    [
        (np.array([0.0, 1.0]), np.array([0.0, 1.0]), "at least two rows"),
        (np.empty((2, 0)), np.array([0.0, 1.0]), "one parameter"),
        (np.array([[0.0], [np.nan]]), np.array([0.0, 1.0]), "finite"),
        (np.array([[0.0], [1.0]]), np.array([0.0]), "matching"),
        (np.array([[0.0], [1.0]]), np.array([0.0, np.inf]), "finite"),
        (np.array([[0.0], [0.0]]), np.array([0.0, 1.0]), "duplicate"),
    ],
)
def test_fit_gaussian_rbf_surrogate_rejects_malformed_training_data(
    inputs: NDArray[np.float64],
    targets: NDArray[np.float64],
    message: str,
) -> None:
    """Training rejects malformed, non-finite, and duplicate samples."""
    with pytest.raises(ValueError, match=message):
        fit_gaussian_rbf_surrogate(inputs, targets)


@pytest.mark.parametrize(
    "inputs",
    [np.array([0.0, 1.0]), np.empty((0, 1)), np.empty((2, 0)), np.array([[np.inf]])],
)
def test_input_row_digests_reject_malformed_arrays(inputs: NDArray[np.float64]) -> None:
    """Provenance row hashing accepts only finite non-empty matrices."""
    with pytest.raises(ValueError):
        input_row_digests(inputs)
