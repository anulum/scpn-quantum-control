# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Differentiable Model Training Evidence
"""Tests for phase/model_training_evidence.py registered model suites."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase.model_training_evidence import (
    run_differentiable_model_training_evidence_suite,
    run_registered_differentiable_training_suite_audit,
)


def test_model_training_evidence_suite_covers_registered_medium_cases() -> None:
    suite = run_differentiable_model_training_evidence_suite()

    assert suite.passed
    assert suite.model_names == (
        "qnn_medium_phase_classifier",
        "qgnn_registered_phase_graph",
        "qsnn_medium_batch",
        "kuramoto_xy_vqe_medium",
        "open_system_control_noise_aware",
        "inverse_coupling_recovery_identifiable",
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


def test_open_system_control_training_case_records_noise_aware_evidence() -> None:
    suite = run_differentiable_model_training_evidence_suite()

    record = next(item for item in suite.records if item.name == "open_system_control_noise_aware")
    assert record.model_family == "open_system_control"
    assert record.passed
    assert record.loss_reduction > 0.0
    assert record.final_loss < record.initial_loss
    assert record.gradient_max_abs_error <= record.gradient_tolerance


def test_inverse_coupling_recovery_case_records_identifiable_knm_evidence() -> None:
    suite = run_differentiable_model_training_evidence_suite()

    record = next(
        item for item in suite.records if item.name == "inverse_coupling_recovery_identifiable"
    )
    assert record.model_family == "inverse_coupling_recovery"
    assert record.passed
    assert record.loss_reduction > 0.0
    assert record.best_loss < 1.0e-3
    assert record.final_loss < record.initial_loss
    assert record.gradient_max_abs_error <= record.gradient_tolerance


def test_registered_training_suite_audit_closes_only_evidenced_lanes() -> None:
    audit = run_registered_differentiable_training_suite_audit()

    assert audit.ready_for_training_suite_promotion
    assert audit.passed_model_families == (
        "qnn",
        "qgnn",
        "qsnn",
        "kuramoto_xy",
        "open_system_control",
        "inverse_coupling_recovery",
    )
    assert audit.blocked_model_families == ()
    assert audit.evidence_suite_passed
    assert not audit.hardware_execution
    assert "registered local training-suite readiness" in audit.claim_boundary

    records = {record.model_family: record for record in audit.records}
    assert records["qnn"].ready
    assert records["qgnn"].ready
    assert records["qsnn"].ready
    assert records["kuramoto_xy"].ready
    assert records["open_system_control"].ready
    assert records["inverse_coupling_recovery"].ready

    payload = audit.to_dict()
    assert payload["ready_for_training_suite_promotion"] is True
    assert payload["passed_model_families"] == [
        "qnn",
        "qgnn",
        "qsnn",
        "kuramoto_xy",
        "open_system_control",
        "inverse_coupling_recovery",
    ]
    assert payload["blocked_model_families"] == []


def test_model_training_evidence_rejects_invalid_tolerance() -> None:
    """The registered suite refuses non-positive gradient tolerances."""
    with pytest.raises(ValueError, match="gradient_tolerance"):
        run_differentiable_model_training_evidence_suite(gradient_tolerance=0.0)


def test_inverse_coupling_recovery_rejects_rank_deficient_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inverse-coupling evidence gate fails closed on a rank-deficient design."""

    def fake_svd(_matrix: object, *, compute_uv: bool = True) -> NDArray[np.float64]:
        return np.array([1.0, 0.5, 0.0], dtype=np.float64)

    monkeypatch.setattr(np.linalg, "svd", fake_svd)
    with pytest.raises(RuntimeError, match="full rank"):
        run_differentiable_model_training_evidence_suite()
