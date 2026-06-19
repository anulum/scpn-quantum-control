# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable hardening gate tests
"""Tests for the differentiable hardening-slice verification gate."""

from __future__ import annotations

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control.benchmarks import run_differentiable_hardening_slice_gate
from scpn_quantum_control.benchmarks.differentiable_hardening_gate import (
    DifferentiableHardeningGateCheck,
)


def test_hardening_slice_gate_accepts_full_focused_verification_plan() -> None:
    """Verify that a complete focused verification plan passes the gate."""
    result = run_differentiable_hardening_slice_gate(
        changed_python_targets=(
            "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
        ),
        module_specific_pytest_targets=("tests/test_differentiable_hardening_gate.py",),
    )

    assert result.passed
    assert {check.check_id for check in result.checks} == {
        "ruff_format",
        "ruff_check",
        "mypy",
        "module_specific_pytest",
        "test_quality_audit",
        "claim_ledger_validation",
    }
    assert result.to_dict()["passed"] is True
    assert "does not execute" in result.claim_boundary
    assert "isolated_affinity" in result.claim_boundary


def test_hardening_slice_gate_rejects_bucket_pytest_targets() -> None:
    """Verify that broad pytest bucket targets fail before gate creation."""
    with pytest.raises(ValueError, match="bucket target"):
        run_differentiable_hardening_slice_gate(
            changed_python_targets=(
                "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
            ),
            module_specific_pytest_targets=("tests",),
        )


def test_hardening_gate_check_rejects_empty_fields() -> None:
    """Verify that individual gate-check rows reject empty evidence fields."""
    with pytest.raises(ValueError, match="check_id"):
        DifferentiableHardeningGateCheck(
            check_id="",
            command=("./.venv/bin/ruff", "check"),
            passed=False,
            evidence="scoped lint command",
        )
    with pytest.raises(ValueError, match="command"):
        DifferentiableHardeningGateCheck(
            check_id="ruff",
            command=(),
            passed=False,
            evidence="scoped lint command",
        )
    with pytest.raises(ValueError, match="evidence"):
        DifferentiableHardeningGateCheck(
            check_id="ruff",
            command=("./.venv/bin/ruff", "check"),
            passed=False,
            evidence="",
        )


def test_hardening_slice_gate_rejects_empty_required_targets() -> None:
    """Verify that required pytest and claim-ledger targets fail closed."""
    with pytest.raises(ValueError, match="module_specific_pytest_targets"):
        run_differentiable_hardening_slice_gate(
            changed_python_targets=(
                "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
            ),
            module_specific_pytest_targets=(),
        )
    with pytest.raises(ValueError, match="claim_ledger_validation_target"):
        run_differentiable_hardening_slice_gate(
            changed_python_targets=(
                "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
            ),
            module_specific_pytest_targets=("tests/test_differentiable_hardening_gate.py",),
            claim_ledger_validation_target="",
        )


def test_hardening_slice_gate_marks_missing_source_targets_as_incomplete() -> None:
    """Verify that test-only slices cannot satisfy the strict typing check."""
    result = run_differentiable_hardening_slice_gate(
        module_specific_pytest_targets=("tests/test_differentiable_hardening_gate.py",),
    )

    mypy_check = next(check for check in result.checks if check.check_id == "mypy")
    assert not result.passed
    assert not mypy_check.passed
    assert mypy_check.command == ("./.venv/bin/mypy",)


def test_hardening_slice_gate_preserves_benchmark_classification_contracts() -> None:
    """Verify benchmark classification cases preserve promotion boundaries."""
    result = run_differentiable_hardening_slice_gate(
        changed_python_targets=(
            "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
        ),
        module_specific_pytest_targets=("tests/test_differentiable_hardening_gate.py",),
    )

    cases = {case.case_id: case for case in result.benchmark_classification_cases}
    assert cases["github_hosted_functional_non_isolated"].metadata.classification == (
        "functional_non_isolated"
    )
    assert cases["github_hosted_functional_non_isolated"].metadata.failure_class == (
        "non_isolated_runner"
    )
    assert cases["self_hosted_isolated_missing_context_hard_gap"].metadata.classification == (
        "hard_gap"
    )
    assert cases["self_hosted_isolated_missing_context_hard_gap"].metadata.failure_class == (
        "insufficient_isolation_metadata"
    )
    assert cases["self_hosted_isolated_affinity"].metadata.classification == "isolated_affinity"
    assert cases["self_hosted_isolated_affinity"].metadata.production_eligible
    assert cases["requested_accelerator_fallback_hard_gap"].metadata.failure_class == (
        "silent_accelerator_fallback"
    )
    assert all(case.passed for case in cases.values())


def test_hardening_slice_gate_is_exported_from_public_namespaces() -> None:
    """Verify that the hardening gate is available from public namespaces."""
    assert scpn.run_differentiable_hardening_slice_gate is (
        run_differentiable_hardening_slice_gate
    )
    assert "run_differentiable_hardening_slice_gate" in scpn.__all__
