# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Differentiable Model Training Evidence
"""Tests for phase/model_training_evidence.py registered model suites."""

from __future__ import annotations

from scpn_quantum_control.phase.model_training_evidence import (
    run_differentiable_model_training_evidence_suite,
)


def test_model_training_evidence_suite_covers_registered_medium_cases() -> None:
    suite = run_differentiable_model_training_evidence_suite()

    assert suite.passed
    assert suite.model_names == (
        "qnn_medium_phase_classifier",
        "qgnn_registered_phase_graph",
        "qsnn_medium_batch",
        "kuramoto_xy_vqe_medium",
    )
    assert suite.unsuitable_scenarios
    assert not suite.hardware_execution
    assert "not arbitrary architecture" in suite.claim_boundary

    for record in suite.records:
        assert record.passed
        assert record.loss_reduction > 0.0
        assert record.final_loss < record.initial_loss
        assert record.gradient_max_abs_error <= record.gradient_tolerance
        assert record.seed is not None
        assert record.training_steps > 0
