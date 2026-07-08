# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QRC Baseline Tests
"""Tests for matched QRC and classical ESN baseline comparisons."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.applications import (
    ClassicalESNReadoutResult,
    QRCBaselineComparison,
    classical_esn_feature_matrix,
    classical_esn_ridge_regression,
    compare_quantum_reservoir_to_esn,
)
from scpn_quantum_control.applications import qrc_baseline as qrc_baseline_module
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27


def _training_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(42)
    x_train = rng.uniform(size=(10, 3)).astype(np.float64)
    y_train = np.sin(2.0 * x_train[:, 0]) + 0.25 * x_train[:, 1]
    return x_train, y_train.astype(np.float64)


def test_classical_esn_feature_matrix_is_deterministic() -> None:
    """The same seed and samples produce identical ESN states."""
    x_train, _ = _training_data()

    first = classical_esn_feature_matrix(x_train, reservoir_size=7, seed=11)
    second = classical_esn_feature_matrix(x_train, reservoir_size=7, seed=11)

    assert first.shape == (10, 7)
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_classical_esn_feature_matrix_rejects_malformed_inputs() -> None:
    """The ESN baseline validates the real input matrix and hyperparameters."""
    x_train, _ = _training_data()

    with pytest.raises(ValueError, match="2-D"):
        classical_esn_feature_matrix(x_train[0], reservoir_size=3)
    with pytest.raises(ValueError, match="at least one sample"):
        classical_esn_feature_matrix(np.empty((0, 3), dtype=np.float64), reservoir_size=3)
    with pytest.raises(ValueError, match="at least one feature"):
        classical_esn_feature_matrix(np.empty((3, 0), dtype=np.float64), reservoir_size=3)
    with pytest.raises(ValueError, match="finite"):
        broken = x_train.copy()
        broken[0, 0] = np.nan
        classical_esn_feature_matrix(broken, reservoir_size=3)
    with pytest.raises(ValueError, match="positive"):
        classical_esn_feature_matrix(x_train, reservoir_size=0)
    with pytest.raises(TypeError, match="integer"):
        classical_esn_feature_matrix(x_train, reservoir_size=3.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="leak_rate"):
        classical_esn_feature_matrix(x_train, reservoir_size=3, leak_rate=1.5)


def test_classical_esn_ridge_regression_returns_finite_readout() -> None:
    """A deterministic ESN readout returns finite predictions and MSE."""
    x_train, y_train = _training_data()

    result = classical_esn_ridge_regression(
        x_train,
        y_train,
        reservoir_size=9,
        alpha=0.2,
        seed=3,
    )

    assert isinstance(result, ClassicalESNReadoutResult)
    assert result.features.shape == (10, 9)
    assert result.weights.shape == (9,)
    assert result.predictions.shape == (10,)
    assert np.isfinite(result.mse)
    assert result.mse >= 0.0


def test_classical_esn_ridge_regression_rejects_bad_targets() -> None:
    """The baseline cannot silently train on malformed targets."""
    x_train, y_train = _training_data()

    with pytest.raises(ValueError, match="matching"):
        classical_esn_ridge_regression(x_train, y_train[:-1], reservoir_size=5)
    with pytest.raises(ValueError, match="finite"):
        broken = y_train.copy()
        broken[0] = np.inf
        classical_esn_ridge_regression(x_train, broken, reservoir_size=5)
    with pytest.raises(ValueError, match="alpha"):
        classical_esn_ridge_regression(x_train, y_train, reservoir_size=5, alpha=0.0)


def test_quantum_reservoir_comparison_matches_feature_dimensions() -> None:
    """The named QRC comparison uses the live QRC path and a matched ESN size."""
    x_train, y_train = _training_data()
    k_matrix = build_knm_paper27(L=3)

    comparison = compare_quantum_reservoir_to_esn(
        x_train,
        y_train,
        k_matrix,
        alpha=0.5,
        max_weight=1,
        seed=5,
    )

    assert isinstance(comparison, QRCBaselineComparison)
    assert comparison.n_quantum_features == 9
    assert comparison.n_esn_features == comparison.n_quantum_features
    assert comparison.quantum_predictions.shape == y_train.shape
    assert comparison.esn_predictions.shape == y_train.shape
    assert np.isfinite(comparison.quantum_mse)
    assert np.isfinite(comparison.esn_mse)
    assert comparison.mse_delta == pytest.approx(comparison.quantum_mse - comparison.esn_mse)


def test_quantum_reservoir_comparison_allows_explicit_esn_size() -> None:
    """Callers may deliberately compare against a different ESN capacity."""
    x_train, y_train = _training_data()
    k_matrix = build_knm_paper27(L=3)

    comparison = compare_quantum_reservoir_to_esn(
        x_train,
        y_train,
        k_matrix,
        max_weight=1,
        reservoir_size=4,
        seed=7,
    )

    assert comparison.n_quantum_features == 9
    assert comparison.n_esn_features == 4


def test_quantum_reservoir_comparison_arithmetic_with_injected_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison arithmetic stays covered when coverage reloads NumPy/Qiskit."""
    x_train, y_train = _training_data()

    def feature_matrix(
        X: NDArray[np.float64],
        K: NDArray[np.float64],
        *,
        omega: NDArray[np.float64] | None = None,
        max_weight: int = 1,
    ) -> NDArray[np.float64]:
        assert K.shape == (3, 3)
        assert omega is None
        assert max_weight == 1
        return np.column_stack(
            [
                np.ones(X.shape[0]),
                X[:, 0],
                X[:, 1],
                X[:, 2],
            ]
        ).astype(np.float64)

    monkeypatch.setattr(qrc_baseline_module, "reservoir_feature_matrix", feature_matrix)

    comparison = compare_quantum_reservoir_to_esn(
        x_train,
        y_train,
        np.eye(3, dtype=np.float64),
        reservoir_size=4,
        seed=9,
    )

    assert comparison.n_quantum_features == 4
    assert comparison.n_esn_features == 4
    assert comparison.quantum_predictions.shape == y_train.shape
    assert comparison.esn_predictions.shape == y_train.shape
    assert comparison.mse_delta == pytest.approx(comparison.quantum_mse - comparison.esn_mse)


def test_qrc_baseline_exports_through_applications_namespace() -> None:
    """The QRC baseline comparison is wired through the applications facade."""
    from scpn_quantum_control import applications

    assert applications.compare_quantum_reservoir_to_esn is compare_quantum_reservoir_to_esn
    assert applications.classical_esn_feature_matrix is classical_esn_feature_matrix
