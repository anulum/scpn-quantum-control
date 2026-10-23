# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Bounded phase classifier NumPy model
"""Evaluate the phase classifier model without training or report composition.

Each feature contributes a phase-shifted one-qubit probability. A sample's
prediction is their mean, clipped to the physical probability interval before
the full-batch mean squared label error. The caller owns input admission,
parameter-shift execution, optimisation, certificates and result objects.
"""

from __future__ import annotations

from typing import cast

from .phase_qnn_model_numerics import FloatArray, PhaseModelNumerics, PhaseModelNumpy


def bounded_phase_qnn_probabilities(
    features: FloatArray,
    parameters: FloatArray,
    *,
    numerics: PhaseModelNumerics,
) -> FloatArray:
    """Evaluate bounded probabilities for admitted phase-classifier inputs.

    Parameters
    ----------
    features
        Finite real sample-by-feature matrix with nonempty dimensions.
    parameters
        Finite real phase vector with one parameter per feature.
    numerics
        Live numerical reads supplied by the owning entry point.

    Returns
    -------
    FloatArray
        One double-precision probability per sample. Inputs are not mutated.
        The operation order retains clipping before the final dtype conversion.

    """
    probabilities = 0.5 * (
        1.0 - cast(PhaseModelNumpy, numerics.numpy()).cos(features + parameters[None, :])
    )
    averaged = cast(PhaseModelNumpy, numerics.numpy()).mean(probabilities, axis=1)
    return (
        cast(PhaseModelNumpy, numerics.numpy())
        .clip(averaged, 0.0, 1.0)
        .astype(cast(PhaseModelNumpy, numerics.numpy()).float64, copy=False)
    )


def phase_qnn_mean_squared_error(
    probabilities: FloatArray,
    labels: FloatArray,
    *,
    numerics: PhaseModelNumerics,
) -> float:
    """Compute the classifier's full-batch mean squared label error.

    Parameters
    ----------
    probabilities
        Admitted probability vector produced by the bounded phase model.
    labels
        Admitted real label vector with one value per sample.
    numerics
        Numerical namespace and scalar-constructor reads, preserving evaluation
        of the constructor before its reduction argument.

    Returns
    -------
    float
        Mean squared residual, without modifying either vector.

    """
    residual = probabilities - labels
    return numerics.scalar()(cast(PhaseModelNumpy, numerics.numpy()).mean(residual * residual))


__all__ = ["bounded_phase_qnn_probabilities", "phase_qnn_mean_squared_error"]
