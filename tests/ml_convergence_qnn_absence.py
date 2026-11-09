# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — isolated QNN dependency-absence qualification
"""Exercise real native training when optional SDKs cannot be imported.

Run only through the SDK-absent quality gate, whose same-ABI package view
contains the actual base libraries and omits optional framework distributions.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from oscillatools.neural_operator import (
    simulate_operator_dataset,
    train_kuramoto_neural_operator,
)
from scpn_quantum_control.ml_examples import (
    FrameworkStatus,
    evidence_payload,
    run_ml_convergence_suite,
    run_qnn_framework_rows,
    validate_ml_convergence_evidence,
)
from scpn_quantum_control.ml_examples.decoding import decode_convergence_evidence


def test_sdk_absence_keeps_native_training_and_required_failure_real() -> None:
    """Train the real native model and refuse every missing required SDK."""
    for module in ("jax", "jaxlib", "torch", "tensorflow"):
        assert importlib.util.find_spec(module) is None

    rows = run_qnn_framework_rows(required_frameworks=("jax", "pytorch", "tensorflow"))

    assert rows[0].status is FrameworkStatus.RAN
    assert rows[0].executed and rows[0].passed and rows[0].gate_passed
    for row in rows[1:4]:
        assert row.status is FrameworkStatus.UNAVAILABLE
        assert row.required and not row.executed and row.passed is None
        assert not row.gate_passed and row.max_abs_error is None

    suite = run_ml_convergence_suite(required_qnn_frameworks=("jax", "pytorch", "tensorflow"))
    payload = evidence_payload(suite)
    decoded = decode_convergence_evidence(payload)

    assert not decoded.passed
    assert decoded.to_payload() == suite.to_payload()
    assert validate_ml_convergence_evidence(payload)

    dataset = simulate_operator_dataset(
        np.array([1.0, 1.2]),
        np.array([[0.0, 0.1], [0.1, 0.0]]),
        n_trajectories=2,
        dt=0.01,
        n_steps=2,
        seed=73,
    )
    with pytest.raises(ImportError, match="requires PyTorch"):
        train_kuramoto_neural_operator(dataset, epochs=1)
