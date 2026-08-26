# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for PL/Qiskit migration guides product
"""Real-surface tests for ``scpn_quantum_control.migration_guides_product``."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.migration_guides_product as migration_guides_product
from scpn_quantum_control.migration_guides_product import (
    MIGRATION_GUIDES_CLAIM_BOUNDARY,
    MIGRATION_GUIDES_PRODUCT_SCHEMA,
    MaterialisedPennyLaneRoundTrip,
    MaterialisedQiskitLocalGradient,
    MigrationConceptRow,
    PathEligibilityDecision,
    assert_migration_guides_product_integrity,
    build_migration_guides_product_registry,
    decide_migration_path,
    get_migration_concept,
    iter_migration_concepts,
    list_migration_concept_ids,
    map_migration_guides_public_surfaces,
    materialise_demo_pennylane_round_trip,
    materialise_demo_qiskit_local_gradient,
)


def _registry_concepts(registry: dict[str, object]) -> list[dict[str, object]]:
    raw = registry["concepts"]
    assert isinstance(raw, list)
    return cast(list[dict[str, object]], raw)


def test_list_concepts_and_filters() -> None:
    """Expose the stable concept catalogue and typed filters."""
    ids = list_migration_concept_ids()
    assert "pl_parameter_shift_to_phase_qnode" in ids
    assert "qk_statevector_parameter_shift" in ids
    assert "refuse_full_runtime_parity" in ids
    assert ids == list_migration_concept_ids()
    pl = iter_migration_concepts(framework="pennylane")
    assert pl
    assert all(row.framework == "pennylane" for row in pl)
    refuse = iter_migration_concepts(support_posture="refuse_only")
    assert refuse


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known concepts while rejecting blank and unknown identifiers."""
    row = get_migration_concept("pl_parameter_shift_to_phase_qnode")
    assert row.claim_boundary == MIGRATION_GUIDES_CLAIM_BOUNDARY
    assert row.allows_live_runtime is False
    assert row.allows_full_parity_claim is False
    with pytest.raises(ValueError, match="non-empty"):
        get_migration_concept("  ")
    with pytest.raises(ValueError, match="unknown concept_id"):
        get_migration_concept("not_a_concept")


def test_path_eligibility_refuse_and_allow() -> None:
    """Allow supported local subsets and refuse unsupported migration paths."""
    allowed = decide_migration_path(local_supported_subset=True)
    assert allowed.allowed is True
    assert allowed.outcome == "allowed"

    runtime = decide_migration_path(request_live_runtime=True)
    assert runtime.allowed is False
    assert any("runtime" in b.lower() or "qpu" in b.lower() for b in runtime.blockers)

    parity = decide_migration_path(request_full_parity=True)
    assert parity.allowed is False
    assert any("parity" in b.lower() for b in parity.blockers)

    no_local = decide_migration_path(local_supported_subset=False)
    assert no_local.allowed is False


def test_materialise_demo_pennylane_round_trip() -> None:
    """Materialise the bounded PennyLane to Phase-QNode round trip."""
    probe = materialise_demo_pennylane_round_trip(theta=0.4)
    assert probe.demo_label.startswith("pl_rx_z_expval")
    assert probe.n_parameters == 1
    assert probe.value_match is True
    assert probe.gradient_match is True
    assert probe.max_value_difference < 1e-6
    assert probe.max_gradient_difference < 1e-6
    assert abs(probe.phase_value - np.cos(0.4)) < 1e-6
    assert abs(probe.pennylane_value - np.cos(0.4)) < 1e-6
    payload = probe.to_dict()
    assert payload["value_match"] is True


def test_materialise_demo_qiskit_local_gradient() -> None:
    """Materialise the local Qiskit statevector gradient comparison."""
    probe = materialise_demo_qiskit_local_gradient(theta=0.4)
    assert probe.demo_label.startswith("qk_rx_z")
    assert abs(probe.value - np.cos(0.4)) < 1e-9
    assert abs(probe.gradient[0] - (-np.sin(0.4))) < 1e-9
    assert probe.max_value_difference < 1e-9
    assert probe.max_gradient_difference < 1e-9
    assert "parameter_shift" in probe.method or "phase_qnode" in probe.method
    payload = probe.to_dict()
    assert payload["method"] == probe.method


def test_public_surfaces_and_registry() -> None:
    """Map ambient owners and validate the complete migration registry."""
    surfaces = map_migration_guides_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.phase.pennylane_import" in paths
    assert "scpn_quantum_control.phase.qiskit_gradients" in paths

    registry = build_migration_guides_product_registry()
    assert registry["schema"] == MIGRATION_GUIDES_PRODUCT_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_concept_id"] == "pl_parameter_shift_to_phase_qnode"
    validated = assert_migration_guides_product_integrity(registry)
    assert validated["concept_count"] == len(list_migration_concept_ids())
    assert assert_migration_guides_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_ghost_concept_drift() -> None:
    """Verify integrity rejects ghost concept drift."""
    registry = build_migration_guides_product_registry()
    concepts = _registry_concepts(registry)
    broken = dict(registry)
    broken["concepts"] = concepts + [
        {
            "concept_id": "ghost",
            "framework": "pennylane",
            "external_concept": "e",
            "scpn_api": "a",
            "support_posture": "local_materialised",
            "summary": "s",
            "module_path": "m",
            "symbol_name": "x",
            "allows_live_runtime": False,
            "allows_full_parity_claim": False,
            "as_of": "2026-07-24",
            "claim_boundary": MIGRATION_GUIDES_CLAIM_BOUNDARY,
        }
    ]
    broken["concept_count"] = len(cast(list[object], broken["concepts"]))
    with pytest.raises(ValueError, match="drift"):
        assert_migration_guides_product_integrity(broken)


def test_integrity_rejects_empty_concepts() -> None:
    """Verify integrity rejects empty concepts."""
    empty: dict[str, object] = {
        "concepts": [],
        "blank_entry_count": 0,
        "concept_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty concepts"):
        assert_migration_guides_product_integrity(empty)


def test_integrity_rejects_invent_green_live_runtime() -> None:
    """Verify integrity rejects invent green live runtime."""
    registry = build_migration_guides_product_registry()
    concepts = _registry_concepts(registry)
    live = dict(registry)
    live_rows = [dict(row) for row in concepts]
    live_rows[0]["allows_live_runtime"] = True
    live["concepts"] = live_rows
    with pytest.raises(ValueError, match="live Runtime|allows_live"):
        assert_migration_guides_product_integrity(live)


def test_integrity_rejects_invent_green_full_parity() -> None:
    """Verify integrity rejects invent green full parity."""
    registry = build_migration_guides_product_registry()
    concepts = _registry_concepts(registry)
    parity = dict(registry)
    parity_rows = [dict(row) for row in concepts]
    parity_rows[0]["allows_full_parity_claim"] = True
    parity["concepts"] = parity_rows
    with pytest.raises(ValueError, match="full parity|allows_full"):
        assert_migration_guides_product_integrity(parity)


def test_integrity_rejects_non_mapping_row() -> None:
    """Verify integrity rejects non mapping row."""
    registry = build_migration_guides_product_registry()
    non_map = dict(registry)
    non_map["concepts"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_migration_guides_product_integrity(non_map)


def test_integrity_rejects_blank_concept_id() -> None:
    """Verify integrity rejects blank concept id."""
    registry = build_migration_guides_product_registry()
    concepts = _registry_concepts(registry)
    blank_id = dict(registry)
    rows = [dict(row) for row in concepts]
    rows[0]["concept_id"] = "  "
    blank_id["concepts"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_migration_guides_product_integrity(blank_id)


def test_integrity_rejects_invalid_framework() -> None:
    """Verify integrity rejects invalid framework."""
    registry = build_migration_guides_product_registry()
    concepts = _registry_concepts(registry)
    bad_fw = dict(registry)
    frows = [dict(row) for row in concepts]
    frows[1]["framework"] = "nope"
    bad_fw["concepts"] = frows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_migration_guides_product_integrity(bad_fw)


def test_integrity_rejects_blank_symbol_name() -> None:
    """Verify integrity rejects blank symbol name."""
    registry = build_migration_guides_product_registry()
    concepts = _registry_concepts(registry)
    no_symbol = dict(registry)
    srows = [dict(row) for row in concepts]
    srows[0]["symbol_name"] = ""
    no_symbol["concepts"] = srows
    with pytest.raises(ValueError, match="symbol_name"):
        assert_migration_guides_product_integrity(no_symbol)


def test_integrity_rejects_missing_default_concept() -> None:
    """Verify integrity rejects missing default concept."""
    registry = build_migration_guides_product_registry()
    concepts = _registry_concepts(registry)
    no_default = dict(registry)
    renamed = [dict(row) for row in concepts]
    for row in renamed:
        if row.get("concept_id") == "pl_parameter_shift_to_phase_qnode":
            row["concept_id"] = "renamed"
    no_default["concepts"] = renamed
    with pytest.raises(ValueError, match="missing pl_parameter_shift|drift"):
        assert_migration_guides_product_integrity(no_default)


def test_integrity_rejects_missing_refuse_concept() -> None:
    """Verify integrity rejects missing refuse concept."""
    registry = build_migration_guides_product_registry()
    concepts = _registry_concepts(registry)
    no_refuse = dict(registry)
    without = [
        dict(row) for row in concepts if row.get("concept_id") != "refuse_full_runtime_parity"
    ]
    no_refuse["concepts"] = without
    no_refuse["concept_count"] = len(without)
    with pytest.raises(ValueError, match="missing refuse_full_runtime|drift"):
        assert_migration_guides_product_integrity(no_refuse)


def test_integrity_rejects_duplicate_concept_id() -> None:
    """Verify integrity rejects duplicate concept id."""
    registry = build_migration_guides_product_registry()
    concepts = _registry_concepts(registry)
    dup = dict(registry)
    drows = [dict(row) for row in concepts]
    drows.append(dict(drows[0]))
    dup["concepts"] = drows
    dup["concept_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate concept_id"):
        assert_migration_guides_product_integrity(dup)


def test_integrity_rejects_nonzero_blank_entry_count() -> None:
    """Verify integrity rejects nonzero blank entry count."""
    registry = build_migration_guides_product_registry()
    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_migration_guides_product_integrity(blank_count)


def test_integrity_rejects_concept_count_mismatch() -> None:
    """Verify integrity rejects concept count mismatch."""
    registry = build_migration_guides_product_registry()
    count_mismatch = dict(registry)
    count_mismatch["concept_count"] = 0
    with pytest.raises(ValueError, match="concept_count"):
        assert_migration_guides_product_integrity(count_mismatch)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("schema", "migration_guides_product.v1", "product schema"),
        ("claim_boundary", "drifted claim", "claim_boundary"),
        ("policy_note", "drifted policy", "policy_note"),
        ("default_concept_id", "qk_statevector_parameter_shift", "default_concept_id"),
    ),
)
def test_integrity_rejects_governed_metadata_drift(
    field: str,
    value: str,
    error: str,
) -> None:
    """Reject stale schemas and drifted governed registry metadata."""
    registry = build_migration_guides_product_registry()
    broken = dict(registry)
    broken[field] = value
    with pytest.raises(ValueError, match=error):
        assert_migration_guides_product_integrity(broken)


def test_integrity_rejects_canonical_row_and_surface_drift() -> None:
    """Reject transported concept and public-surface mutations exactly."""
    registry = build_migration_guides_product_registry()
    concepts = [dict(row) for row in _registry_concepts(registry)]
    concepts[0]["summary"] = "drifted summary"
    row_drift = dict(registry)
    row_drift["concepts"] = concepts
    with pytest.raises(ValueError, match="concept row 0 drift"):
        assert_migration_guides_product_integrity(row_drift)

    surface_drift = dict(registry)
    surface_drift["public_surfaces"] = []
    with pytest.raises(ValueError, match="public_surfaces"):
        assert_migration_guides_product_integrity(surface_drift)


def test_module_exports() -> None:
    """Keep every documented migration product entry point public."""
    assert "materialise_demo_pennylane_round_trip" in migration_guides_product.__all__
    assert "materialise_demo_qiskit_local_gradient" in migration_guides_product.__all__
    assert "list_migration_concept_ids" in migration_guides_product.__all__


def _valid_concept_row_kwargs() -> dict[str, Any]:
    return {
        "concept_id": "x",
        "framework": "pennylane",
        "external_concept": "e",
        "scpn_api": "a",
        "support_posture": "local_materialised",
        "summary": "s",
        "module_path": "m",
        "symbol_name": "fn",
    }


def test_concept_row_accepts_valid_fields() -> None:
    """Verify concept row accepts valid fields."""
    row = MigrationConceptRow(**_valid_concept_row_kwargs())
    assert row.concept_id == "x"
    payload = row.to_dict()
    assert payload["concept_id"] == "x"
    assert payload["allows_live_runtime"] is False


def test_records_reject_claim_boundary_drift() -> None:
    """Require the exact governed claim on every serialisable record type."""
    with pytest.raises(ValueError, match="claim_boundary"):
        MigrationConceptRow(
            **_valid_concept_row_kwargs(),
            claim_boundary="drifted claim",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="bounded local path",
            blockers=(),
            claim_boundary="drifted claim",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        MaterialisedPennyLaneRoundTrip(
            value_match=True,
            gradient_match=True,
            phase_value=0.0,
            pennylane_value=0.0,
            max_value_difference=0.0,
            max_gradient_difference=0.0,
            n_parameters=1,
            demo_label="bounded_round_trip",
            claim_boundary="drifted claim",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        MaterialisedQiskitLocalGradient(
            value=0.0,
            gradient=(0.0,),
            analytic_value=0.0,
            analytic_gradient=(0.0,),
            max_value_difference=0.0,
            max_gradient_difference=0.0,
            method="parameter_shift",
            demo_label="bounded_gradient",
            claim_boundary="drifted claim",
        )


def test_concept_row_rejects_blank_concept_id() -> None:
    """Verify concept row rejects blank concept id."""
    with pytest.raises(ValueError, match="concept_id"):
        MigrationConceptRow(**{**_valid_concept_row_kwargs(), "concept_id": ""})


def test_concept_row_rejects_unknown_framework() -> None:
    """Verify concept row rejects unknown framework."""
    with pytest.raises(ValueError, match="framework"):
        MigrationConceptRow(**{**_valid_concept_row_kwargs(), "framework": cast(Any, "nope")})


def test_concept_row_rejects_invent_green_live_runtime() -> None:
    """Verify concept row rejects invent green live runtime."""
    with pytest.raises(ValueError, match="allows_live_runtime"):
        MigrationConceptRow(**{**_valid_concept_row_kwargs(), "allows_live_runtime": True})


def test_concept_row_rejects_invent_green_full_parity() -> None:
    """Verify concept row rejects invent green full parity."""
    with pytest.raises(ValueError, match="allows_full_parity"):
        MigrationConceptRow(**{**_valid_concept_row_kwargs(), "allows_full_parity_claim": True})


def test_concept_row_rejects_blank_external_concept() -> None:
    """Verify concept row rejects blank external concept."""
    with pytest.raises(ValueError, match="external_concept"):
        MigrationConceptRow(**{**_valid_concept_row_kwargs(), "external_concept": ""})


def test_concept_row_rejects_blank_scpn_api() -> None:
    """Verify concept row rejects blank scpn api."""
    with pytest.raises(ValueError, match="scpn_api"):
        MigrationConceptRow(**{**_valid_concept_row_kwargs(), "scpn_api": ""})


def test_concept_row_rejects_unknown_support_posture() -> None:
    """Verify concept row rejects unknown support posture."""
    with pytest.raises(ValueError, match="support_posture"):
        MigrationConceptRow(
            **{**_valid_concept_row_kwargs(), "support_posture": cast(Any, "nope")}
        )


def test_concept_row_rejects_blank_summary() -> None:
    """Verify concept row rejects blank summary."""
    with pytest.raises(ValueError, match="summary"):
        MigrationConceptRow(**{**_valid_concept_row_kwargs(), "summary": ""})


def test_concept_row_rejects_blank_module_path() -> None:
    """Verify concept row rejects blank module path."""
    with pytest.raises(ValueError, match="module_path"):
        MigrationConceptRow(**{**_valid_concept_row_kwargs(), "module_path": ""})


def test_concept_row_rejects_blank_symbol_name() -> None:
    """Verify concept row rejects blank symbol name."""
    with pytest.raises(ValueError, match="symbol_name"):
        MigrationConceptRow(**{**_valid_concept_row_kwargs(), "symbol_name": ""})


def test_concept_row_rejects_blank_as_of() -> None:
    """Verify concept row rejects blank as of."""
    with pytest.raises(ValueError, match="as_of"):
        MigrationConceptRow(**{**_valid_concept_row_kwargs(), "as_of": ""})


def test_path_decision_refused_requires_blockers() -> None:
    """Verify path decision refused requires blockers."""
    with pytest.raises(ValueError, match="require blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
        )


def test_path_decision_rejects_unknown_outcome() -> None:
    """Verify path decision rejects unknown outcome."""
    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            outcome=cast(Any, "nope"),
            allowed=False,
            reason="r",
            blockers=("b",),
        )


def test_path_decision_rejects_blank_reason() -> None:
    """Verify path decision rejects blank reason."""
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="",
            blockers=("b",),
        )


def test_path_decision_allowed_flag_must_match_outcome() -> None:
    """Verify path decision allowed flag must match outcome."""
    with pytest.raises(ValueError, match="outcome=allowed"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=True,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="outcome=refused"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=False,
            reason="r",
            blockers=("b",),
        )


def test_path_decision_allowed_cannot_list_blockers() -> None:
    """Verify path decision allowed cannot list blockers."""
    with pytest.raises(ValueError, match="cannot list blockers"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="r",
            blockers=("x",),
        )


def test_path_decision_rejects_blank_blocker_entries() -> None:
    """Verify path decision rejects blank blocker entries."""
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("",),
        )


def test_path_decision_to_dict_allowed() -> None:
    """Verify path decision to dict allowed."""
    assert decide_migration_path().to_dict()["allowed"] is True


def test_pennylane_probe_rejects_non_finite_values() -> None:
    """Verify pennylane probe rejects non finite values."""
    with pytest.raises(ValueError, match="finite"):
        MaterialisedPennyLaneRoundTrip(
            value_match=True,
            gradient_match=True,
            phase_value=float("nan"),
            pennylane_value=0.0,
            max_value_difference=0.0,
            max_gradient_difference=0.0,
            n_parameters=1,
            demo_label="d",
        )


def test_pennylane_probe_rejects_negative_value_difference() -> None:
    """Verify pennylane probe rejects negative value difference."""
    with pytest.raises(ValueError, match="max_value_difference"):
        MaterialisedPennyLaneRoundTrip(
            value_match=True,
            gradient_match=True,
            phase_value=0.0,
            pennylane_value=0.0,
            max_value_difference=-0.1,
            max_gradient_difference=0.0,
            n_parameters=1,
            demo_label="d",
        )


def test_pennylane_probe_rejects_negative_gradient_difference() -> None:
    """Verify pennylane probe rejects negative gradient difference."""
    with pytest.raises(ValueError, match="max_gradient_difference"):
        MaterialisedPennyLaneRoundTrip(
            value_match=True,
            gradient_match=True,
            phase_value=0.0,
            pennylane_value=0.0,
            max_value_difference=0.0,
            max_gradient_difference=-0.1,
            n_parameters=1,
            demo_label="d",
        )


def test_pennylane_probe_rejects_negative_n_parameters() -> None:
    """Verify pennylane probe rejects negative n parameters."""
    with pytest.raises(ValueError, match="n_parameters"):
        MaterialisedPennyLaneRoundTrip(
            value_match=True,
            gradient_match=True,
            phase_value=0.0,
            pennylane_value=0.0,
            max_value_difference=0.0,
            max_gradient_difference=0.0,
            n_parameters=-1,
            demo_label="d",
        )


def test_pennylane_probe_rejects_blank_demo_label() -> None:
    """Verify pennylane probe rejects blank demo label."""
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedPennyLaneRoundTrip(
            value_match=True,
            gradient_match=True,
            phase_value=0.0,
            pennylane_value=0.0,
            max_value_difference=0.0,
            max_gradient_difference=0.0,
            n_parameters=0,
            demo_label="",
        )


def test_qiskit_probe_rejects_empty_gradient() -> None:
    """Verify qiskit probe rejects empty gradient."""
    with pytest.raises(ValueError, match="gradient must be non-empty"):
        MaterialisedQiskitLocalGradient(
            value=0.0,
            gradient=(),
            analytic_value=0.0,
            analytic_gradient=(),
            max_value_difference=0.0,
            max_gradient_difference=0.0,
            method="m",
            demo_label="d",
        )


def test_qiskit_probe_rejects_gradient_length_mismatch() -> None:
    """Verify qiskit probe rejects gradient length mismatch."""
    with pytest.raises(ValueError, match="gradient length"):
        MaterialisedQiskitLocalGradient(
            value=0.0,
            gradient=(0.0,),
            analytic_value=0.0,
            analytic_gradient=(0.0, 1.0),
            max_value_difference=0.0,
            max_gradient_difference=0.0,
            method="m",
            demo_label="d",
        )


def test_qiskit_probe_rejects_non_finite_value() -> None:
    """Verify qiskit probe rejects non finite value."""
    with pytest.raises(ValueError, match="finite"):
        MaterialisedQiskitLocalGradient(
            value=float("nan"),
            gradient=(0.0,),
            analytic_value=0.0,
            analytic_gradient=(0.0,),
            max_value_difference=0.0,
            max_gradient_difference=0.0,
            method="m",
            demo_label="d",
        )


def test_qiskit_probe_rejects_negative_value_difference() -> None:
    """Verify qiskit probe rejects negative value difference."""
    with pytest.raises(ValueError, match="max_value_difference"):
        MaterialisedQiskitLocalGradient(
            value=0.0,
            gradient=(0.0,),
            analytic_value=0.0,
            analytic_gradient=(0.0,),
            max_value_difference=-1.0,
            max_gradient_difference=0.0,
            method="m",
            demo_label="d",
        )


def test_qiskit_probe_rejects_negative_gradient_difference() -> None:
    """Verify qiskit probe rejects negative gradient difference."""
    with pytest.raises(ValueError, match="max_gradient_difference"):
        MaterialisedQiskitLocalGradient(
            value=0.0,
            gradient=(0.0,),
            analytic_value=0.0,
            analytic_gradient=(0.0,),
            max_value_difference=0.0,
            max_gradient_difference=-1.0,
            method="m",
            demo_label="d",
        )


def test_qiskit_probe_rejects_blank_method() -> None:
    """Verify qiskit probe rejects blank method."""
    with pytest.raises(ValueError, match="method"):
        MaterialisedQiskitLocalGradient(
            value=0.0,
            gradient=(0.0,),
            analytic_value=0.0,
            analytic_gradient=(0.0,),
            max_value_difference=0.0,
            max_gradient_difference=0.0,
            method="",
            demo_label="d",
        )


def test_qiskit_probe_rejects_blank_demo_label() -> None:
    """Verify qiskit probe rejects blank demo label."""
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedQiskitLocalGradient(
            value=0.0,
            gradient=(0.0,),
            analytic_value=0.0,
            analytic_gradient=(0.0,),
            max_value_difference=0.0,
            max_gradient_difference=0.0,
            method="m",
            demo_label="",
        )


def test_materialise_pennylane_rejects_non_finite_theta() -> None:
    """Verify materialise pennylane rejects non finite theta."""
    with pytest.raises(ValueError, match="theta"):
        materialise_demo_pennylane_round_trip(theta=float("nan"))


def test_materialise_qiskit_rejects_non_finite_theta() -> None:
    """Verify materialise qiskit rejects non finite theta."""
    with pytest.raises(ValueError, match="theta"):
        materialise_demo_qiskit_local_gradient(theta=float("nan"))


def test_materialise_pennylane_rejects_negative_value_tolerance() -> None:
    """Verify materialise pennylane rejects negative value tolerance."""
    with pytest.raises(ValueError, match="value_tolerance"):
        materialise_demo_pennylane_round_trip(value_tolerance=-0.1)


def test_materialise_pennylane_rejects_negative_gradient_tolerance() -> None:
    """Verify materialise pennylane rejects negative gradient tolerance."""
    with pytest.raises(ValueError, match="gradient_tolerance"):
        materialise_demo_pennylane_round_trip(gradient_tolerance=-0.1)


def test_materialise_refuses_when_path_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse both materialisers when the migration path policy blocks."""

    def _refuse(**_kwargs: Any) -> PathEligibilityDecision:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="forced refuse for coverage",
            blockers=("forced",),
        )

    monkeypatch.setattr(migration_guides_product, "decide_migration_path", _refuse)
    with pytest.raises(ValueError, match="demo path refused"):
        materialise_demo_pennylane_round_trip()
    with pytest.raises(ValueError, match="demo path refused"):
        materialise_demo_qiskit_local_gradient()


def test_catalogue_map_runtime_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover blank / duplicate / empty catalogue fail-closed RuntimeError paths."""
    good = get_migration_concept("pl_parameter_shift_to_phase_qnode")
    blank = MigrationConceptRow(**_valid_concept_row_kwargs())
    object.__setattr__(blank, "concept_id", "  ")
    monkeypatch.setattr(migration_guides_product, "_CANONICAL_CONCEPTS", (blank,))
    with pytest.raises(RuntimeError, match="blank concept_id"):
        migration_guides_product._catalogue_map()

    monkeypatch.setattr(migration_guides_product, "_CANONICAL_CONCEPTS", (good, good))
    with pytest.raises(RuntimeError, match="duplicate concept_id"):
        migration_guides_product._catalogue_map()

    monkeypatch.setattr(migration_guides_product, "_CANONICAL_CONCEPTS", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        migration_guides_product._catalogue_map()

    monkeypatch.setattr(migration_guides_product, "_CANONICAL_CONCEPTS", (good,))
    assert migration_guides_product._catalogue_map()[good.concept_id].concept_id == good.concept_id


def test_phase_qnode_empty_gradient_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase-QNode local path refuses empty parameter-shift gradients."""
    monkeypatch.setattr(
        "scpn_quantum_control.phase.pennylane_import.is_pennylane_import_available",
        lambda: False,
    )

    class _EmptyGrad:
        gradient = np.asarray([], dtype=np.float64)

    monkeypatch.setattr(
        "scpn_quantum_control.phase.qnode_circuit.parameter_shift_phase_qnode_gradient",
        lambda *_a, **_k: _EmptyGrad(),
    )
    with pytest.raises(ValueError, match="empty gradient"):
        materialise_demo_pennylane_round_trip(theta=0.4)


def test_materialise_pennylane_uses_import_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose PL import success path on ambient check surface (cov-stable)."""
    qml = types.ModuleType("pennylane")
    tape_mod = types.ModuleType("pennylane.tape")

    class QuantumScript:
        def __init__(self, ops: object, measurements: object) -> None:
            self.ops = ops
            self.measurements = measurements

    tape_mod.QuantumScript = QuantumScript  # type: ignore[attr-defined]
    qml.tape = tape_mod  # type: ignore[attr-defined]

    class RX:
        def __init__(self, theta: float, wires: int = 0) -> None:
            self.theta = theta
            self.wires = wires

    class PauliZ:
        def __init__(self, wire: int) -> None:
            self.wire = wire

    class expval:
        def __init__(self, obs: object) -> None:
            self.obs = obs

    qml.RX = RX  # type: ignore[attr-defined]
    qml.PauliZ = PauliZ  # type: ignore[attr-defined]
    qml.expval = expval  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pennylane", qml)
    monkeypatch.setitem(sys.modules, "pennylane.tape", tape_mod)

    ambient = SimpleNamespace(
        value_match=True,
        gradient_match=True,
        phase_value=float(np.cos(0.4)),
        pennylane_value=float(np.cos(0.4)),
        max_value_difference=0.0,
        max_gradient_difference=0.0,
        n_parameters=1,
    )
    monkeypatch.setattr(
        "scpn_quantum_control.phase.pennylane_import.is_pennylane_import_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "scpn_quantum_control.phase.pennylane_import.check_pennylane_phase_qnode_import_round_trip",
        lambda *_a, **_k: ambient,
    )

    probe = materialise_demo_pennylane_round_trip(theta=0.4)
    assert probe.demo_label == "pl_rx_z_expval_import"
    assert probe.value_match is True
    assert probe.gradient_match is True
    assert probe.n_parameters == 1


def test_materialise_pennylane_falls_back_when_import_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When PL import reports unavailable, use Phase-QNode local subset."""
    monkeypatch.setattr(
        "scpn_quantum_control.phase.pennylane_import.is_pennylane_import_available",
        lambda: False,
    )
    probe = materialise_demo_pennylane_round_trip(theta=0.4)
    assert probe.demo_label == "pl_rx_z_expval_phase_qnode_local"
    assert probe.value_match is True


def test_materialise_pennylane_falls_back_when_import_probe_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover through the local subset when optional PennyLane probing raises."""

    def _raise() -> bool:
        raise RuntimeError("optional PennyLane probe failed")

    monkeypatch.setattr(
        "scpn_quantum_control.phase.pennylane_import.is_pennylane_import_available",
        _raise,
    )
    probe = materialise_demo_pennylane_round_trip(theta=0.4)
    assert probe.demo_label == "pl_rx_z_expval_phase_qnode_local"
    assert probe.gradient_match is True


def test_materialise_qiskit_falls_back_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover through Phase-QNode when the optional Qiskit import is incomplete."""
    monkeypatch.setitem(sys.modules, "qiskit", types.ModuleType("qiskit"))
    probe = materialise_demo_qiskit_local_gradient(theta=0.4)
    assert probe.demo_label == "qk_rx_z_phase_qnode_local"
    assert probe.method == "phase_qnode_parameter_shift_local_subset"
    assert probe.max_value_difference < 1e-12
    assert probe.max_gradient_difference < 1e-12


def test_materialise_qiskit_uses_statevector_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose Qiskit local gradient success path on ambient execute surface."""
    qiskit_mod = types.ModuleType("qiskit")

    class QuantumCircuit:
        def __init__(self, n_qubits: int) -> None:
            self.n_qubits = n_qubits
            self.ops: list[tuple[object, int]] = []

        def rx(self, param: object, qubit: int) -> None:
            self.ops.append((param, qubit))

    qiskit_mod.QuantumCircuit = QuantumCircuit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit", qiskit_mod)

    circuit_mod = types.ModuleType("qiskit.circuit")

    class Parameter:
        def __init__(self, name: str) -> None:
            self.name = name

    circuit_mod.Parameter = Parameter  # type: ignore[attr-defined]
    qiskit_mod.circuit = circuit_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit.circuit", circuit_mod)

    qi_mod = types.ModuleType("qiskit.quantum_info")

    class SparsePauliOp:
        def __init__(self, label: str) -> None:
            self.label = label

    qi_mod.SparsePauliOp = SparsePauliOp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit.quantum_info", qi_mod)

    ambient = SimpleNamespace(
        value=float(np.cos(0.4)),
        gradient=np.asarray([-float(np.sin(0.4))], dtype=np.float64),
        method="qiskit_statevector_parameter_shift",
    )
    monkeypatch.setattr(
        "scpn_quantum_control.phase.qiskit_gradients.execute_qiskit_statevector_parameter_shift",
        lambda *_a, **_k: ambient,
    )

    probe = materialise_demo_qiskit_local_gradient(theta=0.4)
    assert probe.demo_label == "qk_rx_z_statevector_parameter_shift"
    assert probe.method == "qiskit_statevector_parameter_shift"
    assert abs(probe.value - np.cos(0.4)) < 1e-12
    assert abs(probe.gradient[0] - (-np.sin(0.4))) < 1e-12


def test_materialise_qiskit_empty_gradient_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qiskit ambient empty gradient is refuse-closed on the product path."""
    qiskit_mod = types.ModuleType("qiskit")

    class QuantumCircuit:
        def __init__(self, n_qubits: int) -> None:
            self.n_qubits = n_qubits

        def rx(self, param: object, qubit: int) -> None:
            return None

    qiskit_mod.QuantumCircuit = QuantumCircuit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit", qiskit_mod)

    circuit_mod = types.ModuleType("qiskit.circuit")

    class Parameter:
        def __init__(self, name: str) -> None:
            self.name = name

    circuit_mod.Parameter = Parameter  # type: ignore[attr-defined]
    qiskit_mod.circuit = circuit_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit.circuit", circuit_mod)

    qi_mod = types.ModuleType("qiskit.quantum_info")

    class SparsePauliOp:
        def __init__(self, label: str) -> None:
            self.label = label

    qi_mod.SparsePauliOp = SparsePauliOp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit.quantum_info", qi_mod)

    ambient = SimpleNamespace(
        value=0.0,
        gradient=np.asarray([], dtype=np.float64),
        method="qiskit_statevector_parameter_shift",
    )
    monkeypatch.setattr(
        "scpn_quantum_control.phase.qiskit_gradients.execute_qiskit_statevector_parameter_shift",
        lambda *_a, **_k: ambient,
    )

    # Empty gradient raises inside try; product falls through to Phase-QNode.
    # Force Phase-QNode empty too so product refuse is observable if both fail;
    # the Qiskit branch raise is swallowed by except Exception — fallthrough is
    # the designed recovery. Assert fallthrough still produces a valid probe,
    # and separately that the Qiskit empty-gradient ValueError is raised when
    # Phase-QNode is not available as recovery (by making fallthrough also empty).
    class _EmptyGrad:
        gradient = np.asarray([], dtype=np.float64)

    monkeypatch.setattr(
        "scpn_quantum_control.phase.qnode_circuit.parameter_shift_phase_qnode_gradient",
        lambda *_a, **_k: _EmptyGrad(),
    )
    with pytest.raises(ValueError, match="empty gradient"):
        materialise_demo_qiskit_local_gradient(theta=0.4)
