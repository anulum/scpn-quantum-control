# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental feature maps and logistic readout
"""Feature maps span the single-encoding observables; the readout reaches its penalised optimum."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts.exact import (
    _pauli_expectation,
    _prepared_state,
)
from scpn_quantum_control.experimental.llm_qpu.learning.feature_maps import (
    fit_standardiser,
    random_fourier_features,
    sin_cos_basis,
    trig_product_basis,
)
from scpn_quantum_control.experimental.llm_qpu.learning.logistic_readout import (
    LogisticReadout,
    fit_logistic,
    mean_log_loss,
)

_RNG = np.random.default_rng(20261116)


def _observables(
    angles: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return Z, X, Y single-site and Z, X adjacent-pair expectations after RY encoding and a fixed XY block."""
    fixed: list[tuple[str, tuple[int, ...], str]] = [
        ("rz", (wire,), (0.3 * (wire + 1)).hex()) for wire in range(4)
    ]
    for wires in ((0, 1), (1, 2), (2, 3)):
        fixed += [("rxx", wires, (0.7).hex()), ("ryy", wires, (0.7).hex())]
    rows = []
    for row in angles:
        encoding: list[tuple[str, tuple[int, ...], str]] = [
            ("ry", (wire,), float(row[wire]).hex()) for wire in range(4)
        ]
        state = _prepared_state(tuple(encoding + fixed), 4)
        values = [_pauli_expectation(state, axis, (wire,)) for axis in "ZXY" for wire in range(4)]
        values += [_pauli_expectation(state, axis, (w, w + 1)) for axis in "ZX" for w in range(3)]
        rows.append(values)
    return np.array(rows)


def test_the_trig_product_basis_spans_every_single_encoding_observable() -> None:
    """A least-squares fit on the 81 products recovers every coupled expectation; sin/cos alone does not."""
    angles = _RNG.uniform(-math.pi, math.pi, size=(300, 4))
    targets = _observables(angles)
    basis = trig_product_basis(angles)
    assert basis.shape == (300, 81)
    assert np.allclose(basis[:, 0], 1.0)
    assert np.allclose(basis[:, 27], np.sin(angles[:, 0]))
    assert np.allclose(basis[:, 2 * 27 + 2 * 3], np.cos(angles[:, 0]) * np.cos(angles[:, 2]))
    coefficients, *_ = np.linalg.lstsq(basis, targets, rcond=None)
    assert np.abs(basis @ coefficients - targets).max() < 1e-9
    plain = np.concatenate([np.ones((300, 1)), sin_cos_basis(angles)], axis=1)
    coefficients, *_ = np.linalg.lstsq(plain, targets, rcond=None)
    assert np.abs(plain @ coefficients - targets).max() > 1e-2
    pair = sin_cos_basis(angles)
    assert np.array_equal(pair[:, :4], np.sin(angles)) and np.array_equal(
        pair[:, 4:], np.cos(angles)
    )
    with pytest.raises(ValueError, match="^the product basis is bounded at eight angles"):
        trig_product_basis(np.zeros((2, 9)))


def test_standardiser_random_features_and_input_refusals() -> None:
    """Train statistics standardise; constant columns keep scale one; random features are seeded."""
    train = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    scaler = fit_standardiser(train)
    scaled = scaler.apply(train)
    assert np.allclose(scaled[:, 0].mean(), 0.0) and np.allclose(scaled[:, 0].std(), 1.0)
    assert np.allclose(scaled[:, 1], 0.0) and scaler.scale[1] == 1.0
    with pytest.raises(ValueError, match="^features do not have the fitted width$"):
        scaler.apply(np.zeros((1, 3)))
    for bad in (np.zeros(3), np.zeros((0, 2)), np.array([[np.nan, 1.0]])):
        with pytest.raises(ValueError, match="must be a nonempty finite matrix$"):
            fit_standardiser(bad)
    inputs = _RNG.normal(size=(5, 4))
    first = random_fourier_features(inputs, dimension=21, bandwidth=1.0, seed=3)
    assert first.shape == (5, 21)
    assert np.array_equal(
        first, random_fourier_features(inputs, dimension=21, bandwidth=1.0, seed=3)
    )
    assert not np.array_equal(
        first, random_fourier_features(inputs, dimension=21, bandwidth=1.0, seed=4)
    )
    assert np.abs(first).max() <= math.sqrt(2.0 / 21) + 1e-12
    points = np.array([[0.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0]])
    wide = random_fourier_features(points, dimension=40_000, bandwidth=0.8, seed=9)
    assert np.allclose((wide**2).sum(axis=1), 1.0, atol=0.03)
    assert float(wide[0] @ wide[1]) == pytest.approx(math.exp(-(0.6**2) / (2 * 0.8**2)), abs=0.03)
    for dimension, bandwidth, reason in (
        (0, 1.0, "^feature dimension must be a bounded positive integer$"),
        (65_537, 1.0, "^feature dimension must be a bounded positive integer$"),
        (21, 0.0, "^bandwidth must be a positive finite float$"),
        (21, 1, "^bandwidth must be a positive finite float$"),
    ):
        with pytest.raises(ValueError, match=reason):
            random_fourier_features(inputs, dimension=dimension, bandwidth=bandwidth, seed=0)


def test_the_readout_reaches_the_penalised_optimum_and_its_limits() -> None:
    """The analytic gradient vanishes at the fit; separable data nears zero loss; heavy l2 nears uniform."""
    features = _RNG.normal(size=(60, 3))
    labels = (features[:, 0] > 0).astype(np.int64) + 2 * (features[:, 1] > 0).astype(np.int64)
    readout = fit_logistic(features, labels, classes=4, l2=0.05)
    assert readout.converged and readout.parameter_count == 3 * 4 + 4
    probabilities = np.exp(readout.log_probabilities(features))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    residual = (probabilities - np.eye(4)[labels]) / len(labels)
    gradient = np.concatenate(
        [(features.T @ residual + 0.05 * readout.weights).ravel(), residual.sum(axis=0)]
    )
    assert np.abs(gradient).max() < 1e-6
    sharp = fit_logistic(features * 20.0, labels, classes=4, l2=1e-6)
    assert mean_log_loss(sharp, features * 20.0, labels) < 0.05
    flat = fit_logistic(features, labels, classes=4, l2=1e6)
    assert abs(mean_log_loss(flat, features, labels) - math.log(4)) < 0.05
    with pytest.raises(ValueError, match="^features do not have the fitted width$"):
        readout.log_probabilities(np.zeros((2, 2)))
    huge = LogisticReadout(np.array([[1000.0, 0.0]]), np.zeros(2), 0.0, True, 0)
    assert np.allclose(huge.log_probabilities(np.array([[1.0]])), [[0.0, -1000.0]])


def test_the_readout_refuses_malformed_inputs() -> None:
    """Features, classes, l2 and labels are each checked before fitting."""
    features = np.zeros((4, 2))
    labels = np.array([0, 1, 0, 1])
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"features": np.zeros((0, 2))}, "^features must be a nonempty finite matrix$"),
        (
            {"features": np.array([[np.inf, 0.0]] * 4)},
            "^features must be a nonempty finite matrix$",
        ),
        ({"classes": 1}, "^a readout needs at least two classes$"),
        ({"l2": -1.0}, "^l2 must be a non-negative finite float$"),
        ({"l2": 1}, "^l2 must be a non-negative finite float$"),
        ({"labels": np.array([0.0, 1.0, 0.0, 1.0])}, "^labels must be one integer per row$"),
        ({"labels": np.array([0, 1, 0])}, "^labels must be one integer per row$"),
        ({"labels": np.array([0, 2, 0, 1])}, "^labels must name a declared class$"),
        ({"labels": np.array([0, -1, 0, 1])}, "^labels must name a declared class$"),
    )
    for changes, reason in cases:
        arguments: dict[str, Any] = {
            "features": features,
            "labels": labels,
            "classes": 2,
            "l2": 0.1,
            **changes,
        }
        with pytest.raises(ValueError, match=reason):
            fit_logistic(
                arguments["features"],
                arguments["labels"],
                classes=arguments["classes"],
                l2=arguments["l2"],
            )
