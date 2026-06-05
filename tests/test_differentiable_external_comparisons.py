# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable External Comparison Tests
"""Tests for richer external differentiable framework comparison rows."""

from __future__ import annotations

import numpy as np

import scpn_quantum_control.benchmarks.differentiable_external_comparison as comparison
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    ExternalComparisonRow,
    run_differentiable_external_comparison_suite,
)


def test_external_comparison_suite_records_success_rows_and_enzyme_hard_gap(
    monkeypatch,
) -> None:
    monkeypatch.setattr(comparison, "is_phase_jax_available", lambda: True)
    monkeypatch.setattr(comparison, "is_phase_torch_available", lambda: True)
    monkeypatch.setattr(comparison, "is_phase_tensorflow_available", lambda: True)
    monkeypatch.setattr(comparison, "is_phase_pennylane_available", lambda: True)
    monkeypatch.setattr(comparison, "_enzyme_tooling_available", lambda: False)

    rows = run_differentiable_external_comparison_suite()
    by_backend = {row.backend: row for row in rows}

    assert set(by_backend) == {"jax", "pytorch", "tensorflow", "pennylane", "enzyme"}
    for backend in ("jax", "pytorch", "tensorflow", "pennylane"):
        assert by_backend[backend].status == "success"
        assert by_backend[backend].failure_class is None
        assert by_backend[backend].value_error <= 1e-12
        assert by_backend[backend].gradient_error <= 1e-12
        assert by_backend[backend].artifact_fields_ready
        assert by_backend[backend].source_of_truth == "scpn_reference"
    assert by_backend["jax"].batching_support == "vmap"
    assert by_backend["pytorch"].batching_support == "torch.func.vmap"
    assert by_backend["tensorflow"].transform_support == "GradientTape"
    assert by_backend["pennylane"].transform_support == "QNode"
    assert by_backend["enzyme"].status == "hard_gap"
    assert by_backend["enzyme"].failure_class == "dependency_missing"
    assert "LLVM/Enzyme" in str(by_backend["enzyme"].setup_instructions)


def test_external_comparison_row_requires_hard_gap_fields() -> None:
    row = ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend="enzyme",
        status="hard_gap",
        failure_class="dependency_missing",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        memory_peak_bytes=None,
        batching_support="not_evaluated",
        transform_support="LLVM Enzyme",
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions="Install LLVM/Enzyme tooling.",
        claim_boundary="recorded hard gap only",
    )

    assert row.artifact_fields_ready
    assert row.to_dict()["failure_class"] == "dependency_missing"


def test_external_comparison_row_rejects_success_without_numeric_evidence() -> None:
    try:
        ExternalComparisonRow(
            case_id="bounded_phase_objective",
            backend="jax",
            status="success",
            failure_class=None,
            value_error=0.0,
            gradient_error=None,
            runtime_seconds=0.0,
            memory_peak_bytes=1024,
            batching_support="vmap",
            transform_support="value_and_grad",
            dtype="float64",
            device="cpu",
            source_of_truth="scpn_reference",
            setup_instructions=None,
            claim_boundary="diagnostic comparison only",
        )
    except ValueError as exc:
        assert "success rows require numeric" in str(exc)
    else:
        raise AssertionError("success row without gradient evidence was accepted")


def test_external_comparison_suite_records_dependency_missing_rows(monkeypatch) -> None:
    monkeypatch.setattr(comparison, "is_phase_jax_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_torch_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_tensorflow_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_pennylane_available", lambda: False)
    monkeypatch.setattr(comparison, "_enzyme_tooling_available", lambda: False)

    rows = run_differentiable_external_comparison_suite()

    assert all(row.status == "hard_gap" for row in rows)
    assert all(row.failure_class == "dependency_missing" for row in rows)
    assert np.isfinite(len(rows))
