# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for PL/Qiskit migration guides product (BL-41)
"""Real-surface tests for ``scpn_quantum_control.migration_guides_product``."""

from __future__ import annotations

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


def test_list_concepts_and_filters() -> None:
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
    row = get_migration_concept("pl_parameter_shift_to_phase_qnode")
    assert row.claim_boundary == MIGRATION_GUIDES_CLAIM_BOUNDARY
    assert row.allows_live_runtime is False
    assert row.allows_full_parity_claim is False
    with pytest.raises(ValueError, match="non-empty"):
        get_migration_concept("  ")
    with pytest.raises(ValueError, match="unknown concept_id"):
        get_migration_concept("not_a_concept")


def test_path_eligibility_refuse_and_allow() -> None:
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


def test_integrity_rejects_drift_and_invent_green() -> None:
    registry = build_migration_guides_product_registry()
    concepts = cast(list[dict[str, object]], list(registry["concepts"]))

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

    empty: dict[str, object] = {
        "concepts": [],
        "blank_entry_count": 0,
        "concept_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty concepts"):
        assert_migration_guides_product_integrity(empty)

    live = dict(registry)
    live_rows = [dict(row) for row in concepts]
    live_rows[0]["allows_live_runtime"] = True
    live["concepts"] = live_rows
    with pytest.raises(ValueError, match="live Runtime|allows_live"):
        assert_migration_guides_product_integrity(live)

    parity = dict(registry)
    parity_rows = [dict(row) for row in concepts]
    parity_rows[0]["allows_full_parity_claim"] = True
    parity["concepts"] = parity_rows
    with pytest.raises(ValueError, match="full parity|allows_full"):
        assert_migration_guides_product_integrity(parity)


def test_integrity_rejects_blank_invalid() -> None:
    registry = build_migration_guides_product_registry()
    concepts = cast(list[dict[str, object]], list(registry["concepts"]))

    non_map = dict(registry)
    non_map["concepts"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_migration_guides_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in concepts]
    rows[0]["concept_id"] = "  "
    blank_id["concepts"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_migration_guides_product_integrity(blank_id)

    bad_fw = dict(registry)
    frows = [dict(row) for row in concepts]
    frows[1]["framework"] = "nope"
    bad_fw["concepts"] = frows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_migration_guides_product_integrity(bad_fw)

    no_symbol = dict(registry)
    srows = [dict(row) for row in concepts]
    srows[0]["symbol_name"] = ""
    no_symbol["concepts"] = srows
    with pytest.raises(ValueError, match="symbol_name"):
        assert_migration_guides_product_integrity(no_symbol)

    no_default = dict(registry)
    renamed = [dict(row) for row in concepts]
    for row in renamed:
        if row.get("concept_id") == "pl_parameter_shift_to_phase_qnode":
            row["concept_id"] = "renamed"
    no_default["concepts"] = renamed
    with pytest.raises(ValueError, match="missing pl_parameter_shift|drift"):
        assert_migration_guides_product_integrity(no_default)

    no_refuse = dict(registry)
    without = [
        dict(row) for row in concepts if row.get("concept_id") != "refuse_full_runtime_parity"
    ]
    no_refuse["concepts"] = without
    no_refuse["concept_count"] = len(without)
    with pytest.raises(ValueError, match="missing refuse_full_runtime|drift"):
        assert_migration_guides_product_integrity(no_refuse)

    dup = dict(registry)
    drows = [dict(row) for row in concepts]
    drows.append(dict(drows[0]))
    dup["concepts"] = drows
    dup["concept_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate concept_id"):
        assert_migration_guides_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_migration_guides_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["concept_count"] = 0
    with pytest.raises(ValueError, match="concept_count"):
        assert_migration_guides_product_integrity(count_mismatch)


def test_module_exports() -> None:
    assert "materialise_demo_pennylane_round_trip" in migration_guides_product.__all__
    assert "materialise_demo_qiskit_local_gradient" in migration_guides_product.__all__
    assert "list_migration_concept_ids" in migration_guides_product.__all__


def test_row_decision_and_probe_validation() -> None:
    base: dict[str, Any] = {
        "concept_id": "x",
        "framework": "pennylane",
        "external_concept": "e",
        "scpn_api": "a",
        "support_posture": "local_materialised",
        "summary": "s",
        "module_path": "m",
        "symbol_name": "fn",
    }
    assert MigrationConceptRow(**base).concept_id == "x"
    with pytest.raises(ValueError, match="concept_id"):
        MigrationConceptRow(**{**base, "concept_id": ""})
    with pytest.raises(ValueError, match="framework"):
        MigrationConceptRow(**{**base, "framework": cast(Any, "nope")})
    with pytest.raises(ValueError, match="allows_live_runtime"):
        MigrationConceptRow(**{**base, "allows_live_runtime": True})
    with pytest.raises(ValueError, match="allows_full_parity"):
        MigrationConceptRow(**{**base, "allows_full_parity_claim": True})
    with pytest.raises(ValueError, match="external_concept"):
        MigrationConceptRow(**{**base, "external_concept": ""})
    with pytest.raises(ValueError, match="scpn_api"):
        MigrationConceptRow(**{**base, "scpn_api": ""})
    with pytest.raises(ValueError, match="support_posture"):
        MigrationConceptRow(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="summary"):
        MigrationConceptRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="module_path"):
        MigrationConceptRow(**{**base, "module_path": ""})
    with pytest.raises(ValueError, match="symbol_name"):
        MigrationConceptRow(**{**base, "symbol_name": ""})
    with pytest.raises(ValueError, match="as_of"):
        MigrationConceptRow(**{**base, "as_of": ""})

    with pytest.raises(ValueError, match="require blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            outcome=cast(Any, "nope"),
            allowed=False,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="",
            blockers=("b",),
        )
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
    with pytest.raises(ValueError, match="cannot list blockers"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="r",
            blockers=("x",),
        )
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("",),
        )
    assert decide_migration_path().to_dict()["allowed"] is True

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
    with pytest.raises(ValueError, match="theta"):
        materialise_demo_pennylane_round_trip(theta=float("nan"))
    with pytest.raises(ValueError, match="theta"):
        materialise_demo_qiskit_local_gradient(theta=float("nan"))
    with pytest.raises(ValueError, match="value_tolerance"):
        materialise_demo_pennylane_round_trip(value_tolerance=-0.1)
    with pytest.raises(ValueError, match="gradient_tolerance"):
        materialise_demo_pennylane_round_trip(gradient_tolerance=-0.1)


def test_materialise_refuses_when_path_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
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
