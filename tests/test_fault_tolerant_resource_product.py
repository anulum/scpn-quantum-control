# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — fault-tolerant-resource fault-tolerant resource product tests
"""Contract tests for conservative future-resource reporting."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import pytest

from scpn_quantum_control.fault_tolerant_resource_product import (
    FORMULA_REFERENCES,
    FT_RESOURCE_PRODUCT_SCHEMA,
    FaultTolerantResourceBoundaryError,
    FaultTolerantResourceProduct,
    FormulaReference,
    RegimeComparisonRow,
    SensitivityPoint,
    SyncProblemResourceRequest,
    build_fault_tolerant_resource_product,
    build_ft_sensitivity,
    build_regime_comparison,
    estimate_ft_resources,
    render_ft_resource_markdown,
)


def _request(**overrides: object) -> SyncProblemResourceRequest:
    values: dict[str, object] = {
        "n_oscillators": 4,
        "evolution_time": 1.0,
        "target_precision": 0.01,
        "coupling_density": 0.5,
        "trotter_steps": 8,
    }
    values.update(overrides)
    return SyncProblemResourceRequest(**values)  # type: ignore[arg-type]


def test_product_is_deterministic_complete_and_claim_bounded() -> None:
    """Keep repeated reports deterministic, complete, and claim bounded."""
    product = build_fault_tolerant_resource_product(_request())
    repeated = build_fault_tolerant_resource_product(_request())
    assert isinstance(product, FaultTolerantResourceProduct)
    assert product.schema == FT_RESOURCE_PRODUCT_SCHEMA
    assert product.payload_sha256 == repeated.payload_sha256
    assert len(product.payload_sha256) == 64
    assert len(product.regimes) == 6
    assert product.estimate.hardware_availability_claim_allowed is False
    assert product.estimate.fault_tolerant_execution_claim_allowed is False
    json.dumps(product.to_dict())


def test_estimate_arithmetic_and_existing_qec_formula_reuse() -> None:
    """Reuse the QEC formulas while preserving exact resource arithmetic."""
    request = _request(coupling_density=0.5, trotter_steps=10)
    estimate = estimate_ft_resources(request)
    assert estimate.logical_qubits == 4
    assert estimate.interacting_pairs == 3
    assert estimate.arbitrary_rotation_count == 70
    assert (
        estimate.total_t_count == estimate.arbitrary_rotation_count * estimate.t_count_per_rotation
    )
    assert estimate.surface_code_physical_qubits == 4 * (2 * estimate.code_distance**2 - 1)
    assert estimate.repetition_scaffold_physical_qubits == 4 * (2 * estimate.code_distance - 1)
    assert estimate.logical_failure_union_bound <= request.target_precision / 3.0
    assert sum(estimate.precision_allocation.values()) == pytest.approx(request.target_precision)


def test_zero_density_still_counts_local_rotations() -> None:
    """Count local rotations even when pairwise coupling density is zero."""
    request = _request(coupling_density=0.0, trotter_steps=3)
    estimate = estimate_ft_resources(request)
    assert estimate.interacting_pairs == 0
    assert estimate.arbitrary_rotation_count == 12


def test_sensitivity_has_estimates_and_threshold_refusal() -> None:
    """Report bounded sensitivity estimates and refuse the threshold edge."""
    rows = build_ft_sensitivity(_request())
    assert [row.status for row in rows] == ["estimated", "estimated", "estimated", "refused"]
    assert rows[-1].code_distance is None
    assert "at or above" in rows[-1].reason
    json.dumps([row.to_dict() for row in rows])


def test_distance_and_empty_sensitivity_fail_closed() -> None:
    """Refuse infeasible code distances and empty sensitivity requests."""
    with pytest.raises(FaultTolerantResourceBoundaryError, match="at or above"):
        estimate_ft_resources(_request(physical_error_rate=0.02))
    with pytest.raises(FaultTolerantResourceBoundaryError, match="no code distance"):
        estimate_ft_resources(
            _request(target_precision=1e-12, physical_error_rate=0.009, max_code_distance=3)
        )
    with pytest.raises(ValueError, match="must not be empty"):
        build_ft_sensitivity(_request(), ())


def test_regime_table_keeps_non_equivalent_boundaries() -> None:
    """Keep the six resource regimes explicitly non-equivalent."""
    request = _request()
    estimate = estimate_ft_resources(request)
    rows = build_regime_comparison(request, estimate)
    assert [row.regime for row in rows] == [
        "classical_reference",
        "nisq_sampling",
        "repetition_code_scaffold",
        "surface_code_scaffold",
        "analog_mapping",
        "fault_tolerant_planning_model",
    ]
    assert rows[0].physical_qubits is None
    assert rows[1].physical_qubits == request.n_oscillators
    assert rows[4].physical_qubits is None
    assert "No measured syndrome" in rows[3].claim_boundary


def test_markdown_contains_digest_boundaries_and_sources() -> None:
    """Render the digest, claim boundaries, and primary-source pins."""
    product = build_fault_tolerant_resource_product(_request())
    markdown = render_ft_resource_markdown(product)
    assert product.payload_sha256 in markdown
    assert "Syndrome-time floor" in markdown
    assert "refused" in markdown
    assert "https://arxiv.org/abs/1212.6253" in markdown
    assert "available FT hardware" in markdown


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_oscillators": 1}, "n_oscillators"),
        ({"n_oscillators": True}, "n_oscillators"),
        ({"evolution_time": 0.0}, "evolution_time"),
        ({"target_precision": 1.0}, "target_precision"),
        ({"coupling_density": 2.0}, "coupling_density"),
        ({"trotter_steps": 0}, "trotter_steps"),
        ({"trotter_steps": True}, "trotter_steps"),
        ({"physical_error_rate": -0.1}, "physical_error_rate"),
        ({"syndrome_cycle_seconds": 0.0}, "syndrome_cycle_seconds"),
        ({"nisq_shots": 0}, "nisq_shots"),
        ({"nisq_shots": True}, "nisq_shots"),
        ({"max_code_distance": 4}, "max_code_distance"),
    ],
)
def test_request_validation(overrides: dict[str, object], message: str) -> None:
    """Reject each invalid bounded-request field."""
    with pytest.raises(ValueError, match=message):
        _request(**overrides)


def test_record_invariants_reject_inconsistent_evidence() -> None:
    """Reject inconsistent immutable records and stale product schemas."""
    with pytest.raises(ValueError):
        FormulaReference("", "title", "authors", 2020, "https://example.test", "now")
    with pytest.raises(ValueError):
        FormulaReference("id", "title", "authors", 1800, "http://example.test", "now")
    estimate = estimate_ft_resources(_request())
    with pytest.raises(ValueError):
        replace(estimate, total_t_count=1)
    with pytest.raises(ValueError):
        replace(estimate, logical_qubits=0)
    with pytest.raises(ValueError):
        replace(estimate, hardware_availability_claim_allowed=True)
    with pytest.raises(ValueError):
        SensitivityPoint(0.001, "estimated", None, None, "reason")
    with pytest.raises(ValueError):
        SensitivityPoint(0.01, "refused", 3, 17, "reason")
    with pytest.raises(ValueError):
        SensitivityPoint(0.01, "refused", None, None, "")
    with pytest.raises(ValueError):
        RegimeComparisonRow("", "boundary", None, "label", "boundary")
    with pytest.raises(ValueError):
        RegimeComparisonRow("x", "boundary", 0, "label", "boundary")
    product = build_fault_tolerant_resource_product(_request())
    with pytest.raises(ValueError, match="unknown product schema"):
        replace(product, schema="fault_tolerant_resource_product.v1")
    with pytest.raises(ValueError):
        replace(product, sensitivity=())
    with pytest.raises(ValueError):
        replace(product, payload_sha256="short")
    assert [row.to_dict()["formula_id"] for row in FORMULA_REFERENCES]
    assert cast(Any, estimate).to_dict()["formula_ids"]
