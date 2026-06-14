# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Stochastic Gradient Rust Parity Tests
"""Parity tests for stochastic-gradient PyO3 kernels."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.differentiable import SPSAObjectiveSample, spsa_gradient_estimate

engine = pytest.importorskip("scpn_quantum_engine")


def test_rust_spsa_gradient_matches_python_materialised_records() -> None:
    def objective(values: np.ndarray, shots: int | None) -> SPSAObjectiveSample:
        assert shots == 400
        return SPSAObjectiveSample(
            value=float(0.5 * values[0] - 0.25 * values[1]),
            variance=0.04,
            shots=shots,
        )

    python_result = spsa_gradient_estimate(
        objective,
        np.array([0.4, -0.2], dtype=np.float64),
        perturbation_radius=0.25,
        repetitions=4,
        seed=11,
        shots=400,
        confidence_z=2.0,
    )
    plus_values = np.array([record.plus.value for record in python_result.records])
    minus_values = np.array([record.minus.value for record in python_result.records])
    plus_variances = np.array([record.plus.variance for record in python_result.records])
    minus_variances = np.array([record.minus.variance for record in python_result.records])
    plus_shots = np.array(
        [record.plus.shots for record in python_result.records], dtype=np.float64
    )
    minus_shots = np.array(
        [record.minus.shots for record in python_result.records], dtype=np.float64
    )
    perturbations = np.vstack([record.perturbation for record in python_result.records])
    trainable = np.array(python_result.trainable, dtype=np.bool_)

    gradient, standard_error, covariance, confidence_radius = engine.spsa_gradient_rust(
        plus_values,
        minus_values,
        perturbations,
        plus_variances,
        minus_variances,
        plus_shots,
        minus_shots,
        trainable,
        python_result.perturbation_radius,
        python_result.confidence_z,
    )

    np.testing.assert_allclose(gradient, python_result.gradient, atol=1e-12)
    np.testing.assert_allclose(standard_error, python_result.standard_error, atol=1e-12)
    np.testing.assert_allclose(covariance, python_result.covariance, atol=1e-12)
    np.testing.assert_allclose(confidence_radius, python_result.confidence_radius, atol=1e-12)
