# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for resource budget gate (BL-94)
"""Real-surface tests for ``scpn_quantum_control.resource_budget_gate``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.resource_budget_gate as resource_budget_gate
from scpn_quantum_control.compile_budget import estimate_pauli_operator
from scpn_quantum_control.dense_budget import estimate_dense_allocation
from scpn_quantum_control.resource_budget_gate import (
    RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY,
    RESOURCE_BUDGET_GATE_SCHEMA,
    BudgetDimension,
    ResourceBudgetDecision,
    ResourceBudgetEstimate,
    ResourceBudgetExceededError,
    assert_resource_budget_integrity,
    build_resource_budget_registry,
    check_resource_budget,
    enforce_resource_budget,
    estimate_resource_budget,
    get_budget_dimension,
    iter_budget_dimensions,
    list_budget_dimension_ids,
)


def test_list_and_families() -> None:
    ids = list_budget_dimension_ids()
    assert "compile_pauli_default" in ids
    assert "dense_hilbert_default" in ids
    assert ids == list_budget_dimension_ids()
    compile_rows = iter_budget_dimensions(family="compile_pauli")
    assert compile_rows
    assert all(row.family == "compile_pauli" for row in compile_rows)
    dense_rows = iter_budget_dimensions(family="dense_hilbert")
    assert dense_rows
    assert all(row.family == "dense_hilbert" for row in dense_rows)


def test_get_known_and_unknown_fail_closed() -> None:
    row = get_budget_dimension("compile_pauli_default")
    assert row.budget_id == "compile_pauli_default"
    assert row.claim_boundary == RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY
    assert row.default_max_gib > 0
    with pytest.raises(ValueError, match="non-empty"):
        get_budget_dimension("  ")
    with pytest.raises(ValueError, match="unknown budget_id"):
        get_budget_dimension("not_a_budget")


def test_estimate_composes_low_level_pauli() -> None:
    product = estimate_resource_budget("compile_pauli_default", n_qubits=4, include_zz=True)
    low = estimate_pauli_operator(
        4,
        include_zz=True,
        max_gib=get_budget_dimension("compile_pauli_default").default_max_gib,
    )
    assert product.bytes_required == low.bytes_required
    assert product.budget_bytes == low.budget_bytes
    assert product.within_budget is True
    assert product.detail["low_level"] == "estimate_pauli_operator"
    assert product.detail["term_count"] == low.term_count


def test_estimate_composes_low_level_dense() -> None:
    product = estimate_resource_budget("dense_hilbert_default", n_qubits=3, dense_rank=2)
    low = estimate_dense_allocation(
        3,
        rank=2,
        max_gib=get_budget_dimension("dense_hilbert_default").default_max_gib,
    )
    assert product.bytes_required == low.bytes_required
    assert product.budget_bytes == low.budget_bytes
    assert product.within_budget is True
    assert product.detail["low_level"] == "estimate_dense_allocation"
    assert product.detail["dimension"] == low.dimension


def test_within_budget_allowed() -> None:
    decision = check_resource_budget("compile_pauli_default", n_qubits=2)
    assert decision.allowed is True
    assert decision.outcome == "allowed"
    assert decision.blockers == ()
    assert "within budget" in decision.reason


def test_exceed_budget_refused() -> None:
    # Explicit tiny cap forces exceed for modest n (tight catalogue still
    # may fit small constructions depending on term math).
    decision = check_resource_budget(
        "compile_pauli_tight",
        n_qubits=8,
        max_gib=1e-9,
    )
    assert decision.allowed is False
    assert decision.outcome == "refused"
    assert decision.blockers
    assert "exceeds budget" in decision.reason
    dense = check_resource_budget(
        "dense_hilbert_tight",
        n_qubits=6,
        max_gib=1e-9,
    )
    assert dense.allowed is False
    assert dense.blockers


def test_enforce_raises_typed_error() -> None:
    with pytest.raises(ResourceBudgetExceededError) as excinfo:
        enforce_resource_budget("dense_hilbert_tight", n_qubits=8, max_gib=1e-9)
    err = excinfo.value
    assert err.budget_id == "dense_hilbert_tight"
    assert err.n_qubits == 8
    assert err.bytes_required > err.budget_bytes
    payload = err.to_dict()
    assert payload["error"] == "ResourceBudgetExceededError"
    assert payload["claim_boundary"] == RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY

    ok = enforce_resource_budget("dense_hilbert_default", n_qubits=2)
    assert ok.within_budget is True


def test_build_registry_and_integrity() -> None:
    registry = build_resource_budget_registry()
    assert registry["schema"] == RESOURCE_BUDGET_GATE_SCHEMA
    assert registry["blank_entry_count"] == 0
    count = registry["dimension_count"]
    assert isinstance(count, int)
    assert count == len(list_budget_dimension_ids())
    validated = assert_resource_budget_integrity(registry)
    assert validated["dimension_count"] == count
    assert assert_resource_budget_integrity()["blank_entry_count"] == 0


def test_module_exports() -> None:
    assert "check_resource_budget" in resource_budget_gate.__all__
    assert "enforce_resource_budget" in resource_budget_gate.__all__
    assert "estimate_resource_budget" in resource_budget_gate.__all__


def test_invalid_n_qubits() -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        estimate_resource_budget("compile_pauli_default", n_qubits=0)
    with pytest.raises(TypeError, match="integer"):
        estimate_resource_budget("compile_pauli_default", n_qubits=cast(Any, 1.5))
    with pytest.raises(ValueError, match="max_gib"):
        estimate_resource_budget("compile_pauli_default", n_qubits=1, max_gib=0)


def test_dimension_validation() -> None:
    base: dict[str, Any] = {
        "budget_id": "x",
        "family": "compile_pauli",
        "summary": "s",
        "default_max_gib": 1.0,
        "label": "lab",
    }
    assert BudgetDimension(**base).budget_id == "x"
    with pytest.raises(ValueError, match="budget_id"):
        BudgetDimension(**{**base, "budget_id": ""})
    with pytest.raises(ValueError, match="family"):
        BudgetDimension(**{**base, "family": cast(Any, "nope")})
    with pytest.raises(ValueError, match="summary"):
        BudgetDimension(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="label"):
        BudgetDimension(**{**base, "label": ""})
    with pytest.raises(ValueError, match="default_max_gib"):
        BudgetDimension(**{**base, "default_max_gib": 0})
    with pytest.raises(ValueError, match="as_of"):
        BudgetDimension(**{**base, "as_of": ""})


def test_estimate_and_decision_invariants() -> None:
    with pytest.raises(ValueError, match="budget_id"):
        ResourceBudgetEstimate(
            budget_id="",
            family="compile_pauli",
            n_qubits=1,
            bytes_required=1,
            budget_bytes=2,
            gib_required=0.0,
            budget_gib=0.0,
            within_budget=True,
            detail={},
        )
    with pytest.raises(ValueError, match="within_budget inconsistent"):
        ResourceBudgetEstimate(
            budget_id="x",
            family="compile_pauli",
            n_qubits=1,
            bytes_required=10,
            budget_bytes=2,
            gib_required=1.0,
            budget_gib=0.1,
            within_budget=True,
            detail={},
        )
    with pytest.raises(ValueError, match="outcome"):
        ResourceBudgetDecision(
            budget_id="x",
            outcome=cast(Any, "nope"),
            allowed=False,
            n_qubits=1,
            bytes_required=1,
            budget_bytes=1,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="require blockers"):
        ResourceBudgetDecision(
            budget_id="x",
            outcome="refused",
            allowed=False,
            n_qubits=1,
            bytes_required=1,
            budget_bytes=1,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        ResourceBudgetDecision(
            budget_id="x",
            outcome="allowed",
            allowed=True,
            n_qubits=1,
            bytes_required=1,
            budget_bytes=2,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="must use outcome=allowed"):
        ResourceBudgetDecision(
            budget_id="x",
            outcome="refused",
            allowed=True,
            n_qubits=1,
            bytes_required=1,
            budget_bytes=2,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="must use outcome=refused"):
        ResourceBudgetDecision(
            budget_id="x",
            outcome="allowed",
            allowed=False,
            n_qubits=1,
            bytes_required=1,
            budget_bytes=1,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="reason"):
        ResourceBudgetDecision(
            budget_id="x",
            outcome="allowed",
            allowed=True,
            n_qubits=1,
            bytes_required=1,
            budget_bytes=2,
            reason="",
            blockers=(),
        )
    with pytest.raises(ValueError, match="blockers entries"):
        ResourceBudgetDecision(
            budget_id="x",
            outcome="refused",
            allowed=False,
            n_qubits=1,
            bytes_required=1,
            budget_bytes=1,
            reason="r",
            blockers=(" ",),
        )
    with pytest.raises(ValueError, match="n_qubits"):
        ResourceBudgetDecision(
            budget_id="x",
            outcome="allowed",
            allowed=True,
            n_qubits=0,
            bytes_required=1,
            budget_bytes=2,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="family"):
        ResourceBudgetEstimate(
            budget_id="x",
            family=cast(Any, "nope"),
            n_qubits=1,
            bytes_required=1,
            budget_bytes=2,
            gib_required=0.0,
            budget_gib=0.0,
            within_budget=True,
            detail={},
        )
    with pytest.raises(ValueError, match="bytes fields"):
        ResourceBudgetEstimate(
            budget_id="x",
            family="compile_pauli",
            n_qubits=1,
            bytes_required=-1,
            budget_bytes=2,
            gib_required=0.0,
            budget_gib=0.0,
            within_budget=True,
            detail={},
        )


def test_to_dict_paths() -> None:
    dim = get_budget_dimension("dense_hilbert_default")
    assert dim.to_dict()["family"] == "dense_hilbert"
    est = estimate_resource_budget("compile_pauli_default", n_qubits=2)
    assert est.to_dict()["within_budget"] is True
    dec = check_resource_budget("compile_pauli_default", n_qubits=2)
    assert dec.to_dict()["allowed"] is True


def test_integrity_rejects_drift() -> None:
    good = build_resource_budget_registry()
    assert_resource_budget_integrity(good)

    bad_blank = dict(good)
    bad_blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_resource_budget_integrity(bad_blank)

    empty = dict(good)
    empty["dimensions"] = []
    with pytest.raises(ValueError, match="non-empty dimensions"):
        assert_resource_budget_integrity(empty)

    not_map = dict(good)
    not_map["dimensions"] = [123]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_resource_budget_integrity(not_map)

    raw = good["dimensions"]
    assert isinstance(raw, list)
    dims = [dict(cast(dict[str, object], row)) for row in raw]

    blank_id = dict(good)
    blank_row = dict(dims[0])
    blank_row["budget_id"] = ""
    blank_id["dimensions"] = [blank_row, *dims[1:]]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_resource_budget_integrity(blank_id)

    bad_family = dict(good)
    bad = dict(dims[0])
    bad["family"] = "nope"
    bad_family["dimensions"] = [bad if r["budget_id"] == bad["budget_id"] else r for r in dims]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_resource_budget_integrity(bad_family)

    bad_gib = dict(good)
    bad2 = dict(dims[0])
    bad2["default_max_gib"] = 0
    bad_gib["dimensions"] = [bad2 if r["budget_id"] == bad2["budget_id"] else r for r in dims]
    with pytest.raises(ValueError, match="default_max_gib"):
        assert_resource_budget_integrity(bad_gib)

    missing_family = dict(good)
    only_compile = [dict(r) for r in dims if r["family"] == "compile_pauli"]
    missing_family["dimensions"] = only_compile
    missing_family["dimension_count"] = len(only_compile)
    with pytest.raises(ValueError, match="both families|drift"):
        assert_resource_budget_integrity(missing_family)

    bad_count = dict(good)
    bad_count["dimension_count"] = 0
    with pytest.raises(ValueError, match="dimension_count"):
        assert_resource_budget_integrity(bad_count)

    duplicate = dict(good)
    duplicate["dimensions"] = [dims[0], dims[0]]
    duplicate["dimension_count"] = 2
    with pytest.raises(ValueError, match="duplicate"):
        assert_resource_budget_integrity(duplicate)


def test_catalogue_map_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = resource_budget_gate
    with pytest.raises(RuntimeError, match="non-empty"):
        monkeypatch.setattr(mod, "_CANONICAL_DIMENSIONS", ())
        mod._catalogue_map()
    good = get_budget_dimension("compile_pauli_default")
    blank = BudgetDimension(
        budget_id="tmp",
        family="compile_pauli",
        summary="s",
        default_max_gib=1.0,
        label="l",
    )
    object.__setattr__(blank, "budget_id", "  ")
    with pytest.raises(RuntimeError, match="blank budget_id"):
        monkeypatch.setattr(mod, "_CANONICAL_DIMENSIONS", (blank,))
        mod._catalogue_map()
    with pytest.raises(RuntimeError, match="duplicate"):
        monkeypatch.setattr(mod, "_CANONICAL_DIMENSIONS", (good, good))
        mod._catalogue_map()


def test_iter_budget_dimensions_without_filter_returns_full_catalogue() -> None:
    """Unfiltered dimension iter returns every catalogue row."""
    rows = iter_budget_dimensions()
    assert len(rows) == len(list_budget_dimension_ids())
    assert {row.budget_id for row in rows} == set(list_budget_dimension_ids())


def test_estimate_rejects_n_qubits_below_one() -> None:
    """ResourceBudgetEstimate refuses n_qubits < 1."""
    with pytest.raises(ValueError, match="n_qubits must be >= 1"):
        ResourceBudgetEstimate(
            budget_id="x",
            family="compile_pauli",
            n_qubits=0,
            bytes_required=1,
            budget_bytes=2,
            gib_required=0.0,
            budget_gib=0.0,
            within_budget=True,
            detail={},
        )


def test_decision_rejects_blank_budget_id() -> None:
    """ResourceBudgetDecision refuses blank budget_id."""
    with pytest.raises(ValueError, match="budget_id must be non-empty"):
        ResourceBudgetDecision(
            budget_id="  ",
            outcome="allowed",
            allowed=True,
            n_qubits=1,
            bytes_required=1,
            budget_bytes=2,
            reason="r",
            blockers=(),
        )


def test_integrity_rejects_budget_set_drift() -> None:
    """Registry budget_id set must match the live catalogue exactly."""
    good = build_resource_budget_registry()
    raw = good["dimensions"]
    assert isinstance(raw, list)
    dimensions = [dict(cast(dict[str, object], row)) for row in raw]
    drifted = dict(good)
    ghost = dict(dimensions[0])
    ghost["budget_id"] = "ghost_extra_budget"
    drifted["dimensions"] = dimensions + [ghost]
    drifted["dimension_count"] = len(dimensions) + 1
    with pytest.raises(ValueError, match="drift"):
        assert_resource_budget_integrity(drifted)
