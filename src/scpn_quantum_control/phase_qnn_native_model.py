# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native bounded phase classifier model
"""Native model arithmetic for admitted real tensors, separate from report policy.

The bounded classifier averages single-feature phase probabilities and minimises
mean squared label error. Inputs have nonempty sample and feature dimensions,
one label per sample and one phase parameter per feature; the caller owns input
admission, SDK loading and all differentiation contexts. These kernels preserve
native operations; trainable variables can produce ordinary tensors. They do
not select models, create reports or convert values to a
host array.
"""

from __future__ import annotations

from collections.abc import Callable

from .phase_qnn_native_contracts import (
    JAXPhaseNamespace,
    PhaseTensor,
    TensorMean,
    TorchPhaseNamespace,
    TorchPhaseTensor,
)


def jax_phase_classifier_loss(
    namespace: JAXPhaseNamespace,
    features: PhaseTensor,
    labels: PhaseTensor,
    parameters: PhaseTensor,
) -> PhaseTensor:
    """Return the bounded classifier mean squared loss using native JAX values.

    Parameters
    ----------
    namespace
        Actual JAX array operations; no loader or model selector.
    features
        Admitted real sample-by-feature tensor.
    labels
        Admitted real sample label vector.
    parameters
        Native phase vector with one entry per feature.

    Returns
    -------
    PhaseTensor
        Native scalar loss retaining the caller's differentiation context.

    """
    probabilities = 0.5 * (1.0 - namespace.cos(features + parameters[None, :]))
    predictions = namespace.mean(probabilities, axis=1)
    residual = predictions - labels
    return namespace.mean(residual * residual)


def jax_phase_classifier_gradient(
    namespace: JAXPhaseNamespace,
    features: PhaseTensor,
    labels: PhaseTensor,
    parameters: PhaseTensor,
) -> PhaseTensor:
    """Return the model's analytic native parameter derivative.

    Parameters
    ----------
    namespace
        Actual JAX array cosine, sine and reduction operations.
    features
        Admitted real sample-by-feature tensor.
    labels
        Admitted real sample label vector.
    parameters
        Native phase vector, preserving tracer and dtype.

    Returns
    -------
    PhaseTensor
        Native phase gradient including coupling through the averaged prediction.

    """
    feature_count = features.shape[1]
    probabilities = 0.5 * (1.0 - namespace.cos(features + parameters[None, :]))
    predictions = namespace.mean(probabilities, axis=1)
    residual = predictions - labels
    probability_derivative = 0.5 * namespace.sin(features + parameters[None, :])
    prediction_derivative = probability_derivative / feature_count
    return namespace.mean(2.0 * residual[:, None] * prediction_derivative, axis=0)


def torch_phase_classifier_loss(
    namespace: TorchPhaseNamespace,
    features: PhaseTensor,
    labels: PhaseTensor,
    parameters: TorchPhaseTensor,
) -> PhaseTensor:
    """Return the same bounded model loss without leaving native PyTorch.

    Parameters
    ----------
    namespace
        Actual PyTorch cosine and mean operations.
    features
        Admitted real sample-by-feature tensor.
    labels
        Admitted real sample label vector.
    parameters
        Native trainable phase vector.

    Returns
    -------
    PhaseTensor
        Scalar tensor preserving the caller's autograd and tracing context.

    """
    shifted = features + parameters.unsqueeze(0)
    probabilities = 0.5 * (1.0 - namespace.cos(shifted))
    predictions = namespace.mean(probabilities, dim=1)
    residual = predictions - labels
    return namespace.mean(residual * residual)


def tensorflow_phase_classifier_loss(
    cosine: Callable[[PhaseTensor], PhaseTensor],
    reduce_mean: TensorMean,
    features: PhaseTensor,
    labels: PhaseTensor,
    parameters: PhaseTensor,
) -> PhaseTensor:
    """Compute the bounded model using already resolved TensorFlow operations.

    Parameters
    ----------
    cosine
        Actual cosine resolved by the caller before tensor arithmetic.
    reduce_mean
        Actual mean operation resolved by the caller in the same order.
    features
        Admitted real sample-by-feature tensor.
    labels
        Admitted real sample label vector.
    parameters
        Native phase vector watched by the caller's tape.

    Returns
    -------
    PhaseTensor
        Scalar native loss retaining the original tape and graph lifetime.

    """
    shifted = features + parameters
    probabilities = 0.5 * (1.0 - cosine(shifted))
    predictions = reduce_mean(probabilities, axis=1)
    residual = predictions - labels
    return reduce_mean(residual * residual)


__all__ = [
    "jax_phase_classifier_gradient",
    "jax_phase_classifier_loss",
    "tensorflow_phase_classifier_loss",
    "torch_phase_classifier_loss",
]
