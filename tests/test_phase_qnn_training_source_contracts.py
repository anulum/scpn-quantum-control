# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QNN original-source compatibility contracts
"""Preserve genuine QNN result owners, input normalisation and callable serialisation."""

from __future__ import annotations

import inspect
import pickle

import numpy as np
import pytest

from scpn_quantum_control.phase import param_shift
from scpn_quantum_control.phase import qnn_training as qnn


@pytest.mark.parametrize(
    "name",
    [
        "ParameterShiftQNNPredictionResult",
        "ParameterShiftQNNTrainingResult",
        "ParameterShiftQNNExternalGradientAgreement",
        "ParameterShiftQNNGradientVerificationResult",
    ],
)
def test_qnn_result_methods_keep_original_defining_globals(name: str) -> None:
    """Keep protected result methods in their real original module dictionary.

    Parameters
    ----------
    name
        Protected result class exported by the original QNN module.

    """
    definition = getattr(qnn, name)
    assert definition.__module__ == qnn.__name__
    for member_name, descriptor in vars(definition).items():
        if member_name == "to_dict" or isinstance(descriptor, property):
            method = descriptor.fget if isinstance(descriptor, property) else descriptor
            assert inspect.isfunction(method)
            assert method.__module__ == qnn.__name__
            assert method.__globals__ is vars(qnn)


def test_qnn_public_prediction_accepts_and_copies_column_labels() -> None:
    """Normalise a real public column-label input without retaining caller storage."""
    features = np.array([[0.2, -0.4], [1.1, 0.7], [-0.8, 0.3]], dtype=np.float64)
    labels = np.array([[0.0], [1.0], [0.25]], dtype=np.float64)
    params = np.array([0.4, -0.2], dtype=np.float64)
    prediction = qnn.predict_parameter_shift_qnn_classifier(features, params, labels=labels)
    assert prediction.labels is not None
    assert prediction.labels.shape == (3,)
    np.testing.assert_array_equal(prediction.labels, np.array([0.0, 1.0, 0.25]))
    assert not np.shares_memory(prediction.labels, labels)
    np.testing.assert_array_equal(labels, np.array([[0.0], [1.0], [0.25]]))


def test_qnn_public_prediction_refuses_parameter_broadcast() -> None:
    """Reject one phase for two features through the original public prediction API."""
    features = np.array([[0.2, -0.4], [1.1, 0.7]], dtype=np.float64)
    with pytest.raises(ValueError, match="one trainable phase per feature"):
        qnn.predict_parameter_shift_qnn_classifier(features, np.array([0.4], dtype=np.float64))


def test_qnn_imported_gradient_keeps_serializable_owner() -> None:
    """Keep the actual original supplier identity and its original pickle path."""
    gradient = vars(qnn)["parameter_shift_gradient"]
    assert gradient is param_shift.parameter_shift_gradient
    assert gradient.__module__ == "scpn_quantum_control.phase.param_shift"
    assert pickle.dumps(gradient, protocol=4) == pickle.dumps(
        param_shift.parameter_shift_gradient, protocol=4
    )
