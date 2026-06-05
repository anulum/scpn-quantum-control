# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Framework Parity
"""Tests for phase/qnode_framework_parity.py framework parity evidence."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.phase.qnode_framework_parity import (
    run_phase_qnode_framework_parity_suite,
)


def test_phase_qnode_framework_parity_executes_or_classifies_every_local_framework() -> None:
    suite = run_phase_qnode_framework_parity_suite()

    assert suite.frameworks == ("scpn", "jax", "torch", "tensorflow", "pennylane")
    assert suite.record_count == 5
    assert suite.record_by_framework("scpn").status == "passed"
    assert suite.record_by_framework("scpn").value is not None
    assert suite.record_by_framework("scpn").gradient is not None
    assert suite.dependency_sparse in {True, False}
    assert not suite.hardware_execution
    assert "provider" in suite.claim_boundary

    for record in suite.records:
        assert record.status in {"passed", "dependency_missing", "failed"}
        assert record.failure_class in {
            "none",
            "dependency_missing",
            "value_mismatch",
            "gradient_mismatch",
            "runtime_error",
        }
        if record.status == "passed":
            assert record.value_abs_error is not None
            assert record.gradient_max_abs_error is not None
            assert record.value_abs_error <= suite.tolerance
            assert record.gradient_max_abs_error <= suite.tolerance
            assert record.dtype
            assert record.device
            np.testing.assert_allclose(
                record.gradient, suite.reference_gradient, atol=suite.tolerance
            )
