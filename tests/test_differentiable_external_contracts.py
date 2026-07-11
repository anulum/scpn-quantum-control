# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — external comparison contract tests
"""Validation and facade-identity tests for external comparison records."""

from __future__ import annotations

from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES as FACADE_PERMANENT_BOUNDARIES,
)
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS as FACADE_REQUIRED_FIELDS,
)
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    ExternalComparisonArtifact as FacadeExternalComparisonArtifact,
)
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    ExternalComparisonRow as FacadeExternalComparisonRow,
)
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    IdenticalCircuitGradientComparisonArtifact as FacadeIdenticalCircuitArtifact,
)
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    IdenticalCircuitGradientComparisonRow as FacadeIdenticalCircuitRow,
)
from scpn_quantum_control.benchmarks.differentiable_external_contracts import (
    PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES,
    REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS,
    ExternalComparisonArtifact,
    ExternalComparisonRow,
    IdenticalCircuitGradientComparisonArtifact,
    IdenticalCircuitGradientComparisonRow,
)


def test_identical_circuit_gradient_comparison_row_requires_success_evidence() -> None:
    """Success rows should reject missing backend value or gradient evidence."""
    try:
        IdenticalCircuitGradientComparisonRow(
            case_id="case",
            backend="qiskit",
            status="success",
            failure_class=None,
            circuit_fingerprint="abc",
            operations=(("ry", (0,), 0),),
            observable="Z0",
            parameter_values=(0.4,),
            execution_mode="exact_state",
            shots=None,
            scpn_value=1.0,
            backend_value=None,
            value_error=0.0,
            scpn_gradient=(0.0,),
            backend_gradient=None,
            gradient_error=None,
            evaluations=3,
            dependency_versions={"qiskit": "test"},
            claim_boundary="bounded comparison",
        )
    except ValueError as exc:
        assert "success rows require numeric" in str(exc)
    else:
        raise AssertionError("success row without backend gradient was accepted")


def test_external_comparison_row_requires_hard_gap_fields() -> None:
    """Hard-gap rows should preserve required setup and failure metadata."""
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
    assert row.to_dict()["dependency_versions"] is None


def test_external_comparison_row_rejects_success_without_numeric_evidence() -> None:
    """Success rows should reject incomplete numeric evidence."""
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


def test_external_comparison_row_rejects_empty_dependency_metadata() -> None:
    """Dependency metadata should reject empty keys or values."""
    try:
        ExternalComparisonRow(
            case_id="bounded_phase_objective",
            backend="jax",
            status="hard_gap",
            failure_class="unsupported_dtype",
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            memory_peak_bytes=None,
            batching_support="vmap",
            transform_support="value_and_grad",
            dtype="complex128",
            device="cpu",
            source_of_truth="scpn_reference",
            setup_instructions="Use real float64 controls.",
            claim_boundary="unsupported route only",
            dependency_versions={"jax": ""},
        )
    except ValueError as exc:
        assert "dependency version metadata" in str(exc)
    else:
        raise AssertionError("empty dependency metadata was accepted")


def test_external_comparison_contracts_are_exact_facade_aliases() -> None:
    """The compatibility facade should re-export the exact contract objects."""

    assert FacadeExternalComparisonRow is ExternalComparisonRow
    assert FacadeExternalComparisonArtifact is ExternalComparisonArtifact
    assert FacadeIdenticalCircuitRow is IdenticalCircuitGradientComparisonRow
    assert FacadeIdenticalCircuitArtifact is IdenticalCircuitGradientComparisonArtifact
    assert FACADE_PERMANENT_BOUNDARIES is PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES
    assert FACADE_REQUIRED_FIELDS is REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS
