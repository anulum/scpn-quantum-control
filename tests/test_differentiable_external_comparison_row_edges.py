# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable external comparison row edges.
"""Row and writer edge tests for differentiable external comparison evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, cast

import pytest
from _differentiable_external_comparison_edges import (
    gap_external_row,
    gap_identical_row,
    success_external_row,
    success_identical_row,
)

from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    ExternalComparisonRow,
    IdenticalCircuitGradientComparisonRow,
    write_differentiable_external_comparison,
    write_identical_circuit_gradient_comparison,
)


def test_external_comparison_row_rejects_identity_status_and_claim_edges() -> None:
    """External rows should reject invalid identity, status, source, and claim metadata."""
    row = success_external_row()
    invalid_cases: tuple[tuple[Callable[[], ExternalComparisonRow], str], ...] = (
        (lambda: replace(row, case_id=""), "case_id"),
        (lambda: replace(row, backend=""), "backend"),
        (lambda: replace(row, status=cast(Any, "maybe")), "status"),
        (lambda: replace(row, source_of_truth="external"), "source_of_truth"),
        (lambda: replace(row, claim_boundary=""), "claim_boundary"),
    )

    for factory, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            factory()


def test_external_comparison_row_rejects_negative_success_errors() -> None:
    """Success rows should reject negative value or gradient errors."""
    with pytest.raises(ValueError, match="non-negative"):
        replace(success_external_row(), value_error=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        replace(success_external_row(), gradient_error=-1.0)


def test_external_comparison_row_rejects_incomplete_hard_gap_and_toolchain() -> None:
    """Hard gaps and toolchain metadata should remain complete and non-empty."""
    gap = gap_external_row()

    with pytest.raises(ValueError, match="hard_gap rows require"):
        replace(gap, failure_class=None)
    with pytest.raises(ValueError, match="hard_gap rows require"):
        replace(gap, setup_instructions="")
    with pytest.raises(ValueError, match="toolchain metadata"):
        replace(gap, toolchain={"clang": ""})
    with pytest.raises(ValueError, match="toolchain metadata"):
        replace(gap, toolchain={"": "17.0"})


def test_identical_circuit_row_rejects_identity_and_execution_edges() -> None:
    """Same-circuit rows should reject invalid identity and execution metadata."""
    row = success_identical_row()
    invalid_cases: tuple[
        tuple[Callable[[], IdenticalCircuitGradientComparisonRow], str],
        ...,
    ] = (
        (lambda: replace(row, case_id=""), "case_id"),
        (lambda: replace(row, backend="tensorflow"), "backend"),
        (lambda: replace(row, status=cast(Any, "maybe")), "status"),
        (lambda: replace(row, circuit_fingerprint=""), "circuit_fingerprint"),
        (lambda: replace(row, operations=()), "operations"),
        (lambda: replace(row, observable=""), "observable"),
        (lambda: replace(row, execution_mode="finite_shot"), "execution_mode"),
        (lambda: replace(row, shots=100), "shots"),
        (lambda: replace(row, claim_boundary=""), "claim_boundary"),
    )

    for factory, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            factory()


def test_identical_circuit_row_rejects_negative_success_and_incomplete_gap() -> None:
    """Same-circuit success and hard-gap rows should keep strict evidence boundaries."""
    with pytest.raises(ValueError, match="non-negative"):
        replace(success_identical_row(), value_error=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        replace(success_identical_row(), gradient_error=-1.0)
    with pytest.raises(ValueError, match="hard_gap rows require"):
        replace(gap_identical_row(), failure_class=None)


def test_identical_circuit_artifact_writer_rejects_invalid_inputs(tmp_path: Path) -> None:
    """The same-circuit writer should reject invalid destinations and empty evidence."""
    row = gap_identical_row()

    with pytest.raises(ValueError, match="must end with .json"):
        write_identical_circuit_gradient_comparison(tmp_path / "comparison.txt", (row,))
    with pytest.raises(ValueError, match="artifact_id"):
        write_identical_circuit_gradient_comparison(
            tmp_path / "comparison.json", (row,), artifact_id=""
        )
    with pytest.raises(ValueError, match="at least one"):
        write_identical_circuit_gradient_comparison(tmp_path / "comparison.json", ())


def test_external_comparison_writer_rejects_empty_evidence(tmp_path: Path) -> None:
    """The external writer should reject empty evidence after validating the artifact id."""
    with pytest.raises(ValueError, match="at least one"):
        write_differentiable_external_comparison(
            tmp_path / "comparison.json",
            (),
            artifact_id="empty-external-comparison",
        )


def test_identical_circuit_artifact_to_dict_reports_summary(tmp_path: Path) -> None:
    """The same-circuit artifact summary should be JSON-ready."""
    artifact = write_identical_circuit_gradient_comparison(
        tmp_path / "comparison.json",
        (gap_identical_row("qiskit"), gap_identical_row("pennylane")),
        artifact_id="same-circuit-gap",
    )

    payload = artifact.to_dict()

    assert payload["artifact_id"] == "same-circuit-gap"
    assert payload["row_count"] == 2
    assert payload["hard_gap_count"] == 2
    assert payload["identical_circuit_ready"] is False
    assert "Exact-state" in payload["claim_boundary"]
