# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — surrogate numeric identity boundary tests
"""Exercise shared data identities through public fitting and certification."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.surrogates import (
    SurrogateFidelityThresholds,
    certify_surrogate_fidelity,
    fit_gaussian_rbf_surrogate,
    input_row_digests,
)


def test_fitting_copies_and_binds_actual_training_rows() -> None:
    """A fitted model retains the identities and values of its own training rows."""
    inputs = np.array([[-0.6, 0.2], [0.1, 0.8], [0.9, -0.4]])
    targets = np.sum(inputs**2, axis=1)
    before = inputs.copy()
    expected = input_row_digests(inputs)
    model = fit_gaussian_rbf_surrogate(inputs, targets)
    predictions = model.predict(before)
    inputs[:] = 10.0

    assert model.training_input_digests == expected == input_row_digests(model.centres)
    np.testing.assert_array_equal(model.centres, before)
    np.testing.assert_array_equal(model.predict(before), predictions)
    with pytest.raises(ValueError, match="overlap"):
        certify_surrogate_fidelity(
            model,
            before,
            targets,
            thresholds=SurrogateFidelityThresholds(0.01, 0.02, 0.99),
        )


def test_numeric_custody_refuses_near_duplicate_fit_and_validation_rows() -> None:
    """Subprecision identity collisions cannot be counted as independent samples."""
    inputs = np.array([[-1.0], [0.0], [1.0]])
    targets = np.sin(inputs[:, 0])
    model = fit_gaussian_rbf_surrogate(inputs, targets)
    near_training = np.array([[-1.0 + 1.0e-8], [0.4]])
    with pytest.raises(ValueError, match="overlap"):
        certify_surrogate_fidelity(
            model,
            near_training,
            np.sin(near_training[:, 0]),
            thresholds=SurrogateFidelityThresholds(0.01, 0.02, 0.99),
        )
    repeated = np.array([[0.25], [0.25 + 1.0e-8]])
    with pytest.raises(ValueError, match="duplicate"):
        fit_gaussian_rbf_surrogate(repeated, np.sin(repeated[:, 0]))
