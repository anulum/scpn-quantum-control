# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for thermo readiness product (BL-100)
"""Real-surface tests for ``thermo_readiness_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.thermo_readiness_product as thermo_product
from scpn_quantum_control.thermo_readiness_product import (
    THERMO_READINESS_CLAIM_BOUNDARY,
    THERMO_READINESS_PRODUCT_SCHEMA,
    FepInventoryRow,
    MaterialisedKSweepProbe,
    PathEligibilityDecision,
    ReadinessCapabilityRow,
    assert_thermo_readiness_product_integrity,
    build_thermo_readiness_product_registry,
    compute_k_sweep_request_digest,
    decide_readiness_path,
    get_fep_inventory_row,
    get_readiness_capability,
    iter_fep_inventory,
    iter_readiness_capabilities,
    list_fep_module_ids,
    list_readiness_capability_ids,
    map_thermo_readiness_public_surfaces,
    materialise_demo_k_sweep_probe,
    materialise_k_sweep_probe,
    materialise_quantum_thermo_payload_probe,
    verify_ambient_claim_boundary,
)
from scpn_quantum_control.thermodynamics.readiness import (
    CLAIM_BOUNDARY as AMBIENT_CLAIM_BOUNDARY,
)
from scpn_quantum_control.thermodynamics.readiness import (
    QUANTUM_THERMO_SCHEMA,
    ThermodynamicSweepConfig,
    run_k_sweep_protocol,
)


def test_list_capability_ids_includes_k_sweep_and_entropy() -> None:
    ids = list_readiness_capability_ids()
    assert "k_sweep_protocol" in ids
    assert "entropy_production" in ids
    assert "work_identity" in ids
    assert "heat_dissipation" in ids
    assert "claim_boundary_gate" in ids
    assert len(ids) == 5


def test_list_fep_module_ids_is_research_inventory() -> None:
    fep_ids = list_fep_module_ids()
    assert "predictive_coding" in fep_ids
    assert "variational_free_energy" in fep_ids
    assert len(fep_ids) == 2


def test_iter_readiness_capabilities_unfiltered_returns_all() -> None:
    rows = iter_readiness_capabilities()
    assert len(rows) == len(list_readiness_capability_ids())
    assert {row.capability_id for row in rows} == set(list_readiness_capability_ids())


def test_iter_readiness_capabilities_filters_by_kind() -> None:
    sweeps = iter_readiness_capabilities(kind="k_sweep_protocol")
    assert len(sweeps) == 1
    assert sweeps[0].capability_id == "k_sweep_protocol"


def test_iter_fep_inventory_unfiltered_returns_all() -> None:
    rows = iter_fep_inventory()
    assert len(rows) == len(list_fep_module_ids())
    assert {row.module_id for row in rows} == set(list_fep_module_ids())


def test_iter_fep_inventory_filters_by_status() -> None:
    research = iter_fep_inventory(status="research_only")
    assert len(research) == 2
    assert all(row.status == "research_only" for row in research)


def test_get_known_and_unknown_fail_closed() -> None:
    row = get_readiness_capability("k_sweep_protocol")
    assert row.claim_boundary == THERMO_READINESS_CLAIM_BOUNDARY
    assert row.hardware_submission_allowed is False
    assert row.thermodynamic_peak_claim_allowed is False
    fep = get_fep_inventory_row("predictive_coding")
    assert fep.status == "research_only"
    assert fep.product_hook_proven is False
    assert "bl84" in fep.bl84_pointer
    with pytest.raises(ValueError, match="non-empty"):
        get_readiness_capability("  ")
    with pytest.raises(ValueError, match="unknown capability_id"):
        get_readiness_capability("not_a_capability")
    with pytest.raises(ValueError, match="non-empty"):
        get_fep_inventory_row("")
    with pytest.raises(ValueError, match="unknown module_id"):
        get_fep_inventory_row("not_a_module")


def test_ambient_claim_boundary_machine_checked() -> None:
    boundary = verify_ambient_claim_boundary()
    assert boundary == AMBIENT_CLAIM_BOUNDARY
    assert "no thermodynamic peak" in boundary
    assert "no hardware submission" in boundary


def test_decide_readiness_path() -> None:
    ok = decide_readiness_path("k_sweep_protocol")
    assert ok.allowed is True
    assert ok.outcome == "allowed"
    assert ok.blockers == ()

    peak = decide_readiness_path("k_sweep_protocol", invent_green_peak_claim=True)
    assert peak.allowed is False
    assert any("peak" in b.lower() for b in peak.blockers)

    hw = decide_readiness_path("k_sweep_protocol", invent_green_hardware_submit=True)
    assert hw.allowed is False
    assert any("hardware" in b.lower() or "submission" in b.lower() for b in hw.blockers)

    fep = decide_readiness_path("entropy_production", invent_green_fep_product=True)
    assert fep.allowed is False
    assert any("fep" in b.lower() for b in fep.blockers)

    multi = decide_readiness_path(
        "k_sweep_protocol",
        invent_green_peak_claim=True,
        invent_green_hardware_submit=True,
    )
    assert multi.allowed is False
    assert len(multi.blockers) >= 2


def test_k_sweep_probe_real_ambient_path() -> None:
    probe = materialise_demo_k_sweep_probe()
    assert probe.capability_id == "k_sweep_protocol"
    assert len(probe.probe_digest) == 64
    assert probe.row_count >= 3
    assert probe.hardware_submission_allowed is False
    assert probe.thermodynamic_peak_claim_allowed is False
    assert "no thermodynamic peak" in probe.ambient_claim_boundary
    assert probe.falsifier
    # Cross-check against ambient directly (not re-implemented).
    ambient = run_k_sweep_protocol()
    assert probe.peak_k == ambient.peak_k
    assert probe.row_count == len(ambient.rows)
    assert probe.schema == ambient.schema
    payload = probe.to_dict()
    assert payload["thermodynamic_peak_claim_allowed"] is False

    again = materialise_k_sweep_probe("k_sweep_protocol")
    assert again.probe_digest == probe.probe_digest

    custom = materialise_k_sweep_probe(
        "k_sweep_protocol",
        config=ThermodynamicSweepConfig(
            k_values=(0.5, 0.8, 1.1),
            transition_k=0.8,
        ),
        demo_label="custom_grid",
    )
    assert custom.row_count == 3
    assert custom.demo_label == "custom_grid"
    assert custom.probe_digest != probe.probe_digest


def test_k_sweep_probe_refuses_invent_green() -> None:
    with pytest.raises(ValueError, match="peak"):
        materialise_k_sweep_probe("k_sweep_protocol", invent_green_peak_claim=True)
    with pytest.raises(ValueError, match="hardware|submission"):
        materialise_k_sweep_probe("k_sweep_protocol", invent_green_hardware_submit=True)
    with pytest.raises(ValueError, match="k_sweep_protocol"):
        materialise_k_sweep_probe("entropy_production")


def test_quantum_thermo_payload_probe() -> None:
    wrapped = materialise_quantum_thermo_payload_probe()
    assert wrapped["product_schema"] == THERMO_READINESS_PRODUCT_SCHEMA
    assert wrapped["hardware_submission_allowed"] is False
    assert wrapped["thermodynamic_peak_claim_allowed"] is False
    ambient = cast(dict[str, Any], wrapped["payload"])
    assert ambient["no_qpu_submission"] is True
    assert ambient["schema"]
    assert "k_sweep" in ambient


def test_request_digest() -> None:
    d1 = compute_k_sweep_request_digest(
        k_values=(0.4, 0.6, 0.8, 1.0, 1.2),
        transition_k=0.8,
    )
    d2 = compute_k_sweep_request_digest(
        k_values=(0.4, 0.6, 0.8, 1.0, 1.2),
        transition_k=0.8,
    )
    assert d1 == d2
    assert len(d1) == 64
    d3 = compute_k_sweep_request_digest(
        k_values=(0.5, 0.8, 1.1),
        transition_k=0.8,
    )
    assert d3 != d1
    with pytest.raises(ValueError, match="capability_id"):
        compute_k_sweep_request_digest(
            k_values=(0.4, 0.6, 0.8), transition_k=0.6, capability_id=""
        )
    with pytest.raises(ValueError, match="at least three"):
        compute_k_sweep_request_digest(k_values=(0.4, 0.6), transition_k=0.4)
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_k_sweep_request_digest(k_values=(0.8, 0.6, 0.4), transition_k=0.6)
    with pytest.raises(ValueError, match="transition_k"):
        compute_k_sweep_request_digest(k_values=(0.4, 0.6, 0.8), transition_k=0.5)
    with pytest.raises(ValueError, match="unknown capability_id"):
        compute_k_sweep_request_digest(
            k_values=(0.4, 0.6, 0.8),
            transition_k=0.6,
            capability_id="ghost",
        )


def test_public_surfaces_and_registry() -> None:
    surfaces = map_thermo_readiness_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.thermo_readiness_product" in paths
    assert "scpn_quantum_control.thermodynamics.readiness" in paths
    assert "scpn_quantum_control.fep.predictive_coding" in paths
    assert "scpn_quantum_control.fep.variational_free_energy" in paths

    registry = build_thermo_readiness_product_registry()
    assert registry["schema"] == THERMO_READINESS_PRODUCT_SCHEMA
    assert registry["hardware_submission_allowed_policy"] is False
    assert registry["thermodynamic_peak_claim_allowed_policy"] is False
    assert registry["fep_product_promotion_allowed_policy"] is False
    validated = assert_thermo_readiness_product_integrity(registry)
    assert validated["capability_count"] == 5
    assert validated["fep_inventory_count"] == 2
    assert assert_thermo_readiness_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_policy() -> None:
    registry = build_thermo_readiness_product_registry()
    capabilities = cast(list[dict[str, object]], registry["capabilities"])

    broken = dict(registry)
    broken["capabilities"] = capabilities + [
        {
            "capability_id": "ghost",
            "kind": "k_sweep_protocol",
            "title": "t",
            "summary": "s",
            "ambient_symbol": "x",
            "hardware_submission_allowed": False,
            "thermodynamic_peak_claim_allowed": False,
            "support_posture": "local_research",
            "as_of": "2026-07-24",
            "claim_boundary": THERMO_READINESS_CLAIM_BOUNDARY,
        }
    ]
    broken["capability_count"] = len(cast(list[object], broken["capabilities"]))
    with pytest.raises(ValueError, match="drift"):
        assert_thermo_readiness_product_integrity(broken)

    empty: dict[str, object] = {
        "capabilities": [],
        "fep_inventory": registry["fep_inventory"],
        "blank_entry_count": 0,
        "capability_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty capabilities"):
        assert_thermo_readiness_product_integrity(empty)

    no_fep = dict(registry)
    no_fep["fep_inventory"] = []
    no_fep["fep_inventory_count"] = 0
    with pytest.raises(ValueError, match="non-empty fep_inventory"):
        assert_thermo_readiness_product_integrity(no_fep)

    peak_ok = dict(registry)
    peak_ok["thermodynamic_peak_claim_allowed_policy"] = True
    with pytest.raises(ValueError, match="thermodynamic_peak_claim_allowed_policy"):
        assert_thermo_readiness_product_integrity(peak_ok)

    hw_ok = dict(registry)
    hw_ok["hardware_submission_allowed_policy"] = True
    with pytest.raises(ValueError, match="hardware_submission_allowed_policy"):
        assert_thermo_readiness_product_integrity(hw_ok)

    fep_promo = dict(registry)
    fep_promo["fep_product_promotion_allowed_policy"] = True
    with pytest.raises(ValueError, match="fep_product_promotion_allowed_policy"):
        assert_thermo_readiness_product_integrity(fep_promo)

    caps = cast(list[dict[str, object]], registry["capabilities"])
    mutated_caps = [dict(row) for row in caps]
    mutated_caps[0]["hardware_submission_allowed"] = True
    hw_cap = dict(registry)
    hw_cap["capabilities"] = mutated_caps
    with pytest.raises(ValueError, match="hardware_submission_allowed"):
        assert_thermo_readiness_product_integrity(hw_cap)

    fep_rows = cast(list[dict[str, object]], registry["fep_inventory"])
    mutated_fep = [dict(row) for row in fep_rows]
    mutated_fep[0]["status"] = "product_hook_open"
    fep_status = dict(registry)
    fep_status["fep_inventory"] = mutated_fep
    with pytest.raises(ValueError, match="research_only"):
        assert_thermo_readiness_product_integrity(fep_status)

    proven_fep = [dict(row) for row in fep_rows]
    proven_fep[0]["product_hook_proven"] = True
    proven_reg = dict(registry)
    proven_reg["fep_inventory"] = proven_fep
    with pytest.raises(ValueError, match="product_hook_proven"):
        assert_thermo_readiness_product_integrity(proven_reg)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_thermo_readiness_product_integrity(blank_count)

    bad_boundary = dict(registry)
    bad_boundary["ambient_claim_boundary"] = "promotional peak achieved"
    with pytest.raises(ValueError, match="ambient_claim_boundary"):
        assert_thermo_readiness_product_integrity(bad_boundary)


def _valid_capability_row(**overrides: Any) -> ReadinessCapabilityRow:
    kwargs: dict[str, Any] = {
        "capability_id": "x",
        "kind": "k_sweep_protocol",
        "title": "t",
        "summary": "s",
        "ambient_symbol": "x",
    }
    kwargs.update(overrides)
    return ReadinessCapabilityRow(**kwargs)


def _valid_probe(**overrides: Any) -> MaterialisedKSweepProbe:
    kwargs: dict[str, Any] = {
        "capability_id": "k_sweep_protocol",
        "schema": "s",
        "peak_k": 0.8,
        "row_count": 5,
        "hardware_submission_allowed": False,
        "thermodynamic_peak_claim_allowed": False,
        "ambient_claim_boundary": AMBIENT_CLAIM_BOUNDARY,
        "falsifier": "f",
        "probe_digest": "a" * 64,
        "demo_label": "d",
    }
    kwargs.update(overrides)
    return MaterialisedKSweepProbe(**kwargs)


def test_capability_row_rejects_blank_capability_id() -> None:
    with pytest.raises(ValueError, match="capability_id"):
        _valid_capability_row(capability_id="")


def test_capability_row_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown capability kind"):
        _valid_capability_row(kind=cast(Any, "not_a_kind"))


def test_capability_row_rejects_blank_title() -> None:
    with pytest.raises(ValueError, match="title"):
        _valid_capability_row(title="")


def test_capability_row_rejects_blank_summary() -> None:
    with pytest.raises(ValueError, match="summary"):
        _valid_capability_row(summary="  ")


def test_capability_row_rejects_blank_ambient_symbol() -> None:
    with pytest.raises(ValueError, match="ambient_symbol"):
        _valid_capability_row(ambient_symbol="")


def test_capability_row_rejects_hardware_submission_allowed() -> None:
    with pytest.raises(ValueError, match="hardware_submission_allowed"):
        _valid_capability_row(hardware_submission_allowed=True)


def test_capability_row_rejects_thermodynamic_peak_claim_allowed() -> None:
    with pytest.raises(ValueError, match="thermodynamic_peak_claim_allowed"):
        _valid_capability_row(thermodynamic_peak_claim_allowed=True)


def test_capability_row_rejects_unknown_support_posture() -> None:
    with pytest.raises(ValueError, match="support_posture"):
        _valid_capability_row(support_posture=cast(Any, "bogus"))


def test_capability_row_rejects_blank_as_of() -> None:
    with pytest.raises(ValueError, match="as_of"):
        _valid_capability_row(as_of="")


def test_capability_row_to_dict_preserves_capability_id() -> None:
    ok_cap = _valid_capability_row()
    assert ok_cap.to_dict()["capability_id"] == "x"


def test_fep_row_rejects_blank_module_id() -> None:
    with pytest.raises(ValueError, match="module_id"):
        FepInventoryRow(module_id="", module_path="m", title="t", summary="s")


def test_fep_row_rejects_blank_module_path() -> None:
    with pytest.raises(ValueError, match="module_path"):
        FepInventoryRow(module_id="x", module_path="", title="t", summary="s")


def test_fep_row_rejects_blank_title() -> None:
    with pytest.raises(ValueError, match="title"):
        FepInventoryRow(module_id="x", module_path="m", title="", summary="s")


def test_fep_row_rejects_blank_summary() -> None:
    with pytest.raises(ValueError, match="summary"):
        FepInventoryRow(module_id="x", module_path="m", title="t", summary="")


def test_fep_row_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="status"):
        FepInventoryRow(
            module_id="x",
            module_path="m",
            title="t",
            summary="s",
            status=cast(Any, "bogus"),
        )


def test_fep_row_rejects_blank_bl84_pointer() -> None:
    with pytest.raises(ValueError, match="bl84_pointer"):
        FepInventoryRow(
            module_id="x",
            module_path="m",
            title="t",
            summary="s",
            bl84_pointer="",
        )


def test_fep_row_rejects_research_only_product_hook_proven() -> None:
    with pytest.raises(ValueError, match="research_only rows cannot set product_hook_proven"):
        FepInventoryRow(
            module_id="x",
            module_path="m",
            title="t",
            summary="s",
            status="research_only",
            product_hook_proven=True,
        )


def test_fep_row_rejects_product_hook_proven_on_open_status() -> None:
    with pytest.raises(ValueError, match="product_hook_proven must be False on product surface"):
        FepInventoryRow(
            module_id="x",
            module_path="m",
            title="t",
            summary="s",
            status="product_hook_open",
            product_hook_proven=True,
        )


def test_fep_row_to_dict_defaults_research_only() -> None:
    ok_fep = FepInventoryRow(module_id="x", module_path="m", title="t", summary="s")
    assert ok_fep.to_dict()["status"] == "research_only"


def test_path_decision_rejects_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            outcome=cast(Any, "maybe"),
            allowed=True,
            reason="r",
            blockers=(),
        )


def test_path_decision_rejects_blank_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="",
            blockers=(),
        )


def test_path_decision_rejects_allowed_with_refused_outcome() -> None:
    with pytest.raises(ValueError, match="outcome=allowed"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=True,
            reason="r",
            blockers=(),
        )


def test_path_decision_rejects_refused_flag_with_allowed_outcome() -> None:
    with pytest.raises(ValueError, match="outcome=refused"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=False,
            reason="r",
            blockers=("b",),
        )


def test_path_decision_rejects_blockers_on_allowed_path() -> None:
    with pytest.raises(ValueError, match="cannot list blockers"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="r",
            blockers=("b",),
        )


def test_path_decision_rejects_empty_blockers_when_refused() -> None:
    with pytest.raises(ValueError, match="blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="no",
            blockers=(),
        )


def test_path_decision_rejects_blank_blocker_entries() -> None:
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="no",
            blockers=("",),
        )


def test_path_decision_to_dict_reports_allowed() -> None:
    ok_dec = PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason="ok",
        blockers=(),
    )
    assert ok_dec.to_dict()["allowed"] is True


def test_probe_rejects_blank_capability_id() -> None:
    with pytest.raises(ValueError, match="capability_id"):
        _valid_probe(capability_id="")


def test_probe_rejects_blank_schema() -> None:
    with pytest.raises(ValueError, match="schema"):
        _valid_probe(schema="")


def test_probe_rejects_row_count_below_three() -> None:
    with pytest.raises(ValueError, match="row_count"):
        _valid_probe(row_count=2)


def test_probe_rejects_hardware_submission_allowed() -> None:
    with pytest.raises(ValueError, match="hardware_submission_allowed"):
        _valid_probe(hardware_submission_allowed=True)


def test_probe_rejects_thermodynamic_peak_claim_allowed() -> None:
    with pytest.raises(ValueError, match="thermodynamic_peak_claim_allowed"):
        _valid_probe(thermodynamic_peak_claim_allowed=True)


def test_probe_rejects_blank_ambient_claim_boundary() -> None:
    with pytest.raises(ValueError, match="ambient_claim_boundary"):
        _valid_probe(ambient_claim_boundary="")


def test_probe_rejects_promotional_ambient_claim_boundary() -> None:
    with pytest.raises(ValueError, match="no-thermodynamic-peak|ambient_claim_boundary"):
        _valid_probe(ambient_claim_boundary="peak claim allowed")


def test_probe_rejects_blank_falsifier() -> None:
    with pytest.raises(ValueError, match="falsifier"):
        _valid_probe(falsifier="")


def test_probe_rejects_blank_probe_digest() -> None:
    with pytest.raises(ValueError, match="probe_digest"):
        _valid_probe(probe_digest="")


def test_probe_rejects_non_hex_length_digest() -> None:
    with pytest.raises(ValueError, match="64-char"):
        _valid_probe(probe_digest="abc")


def test_probe_rejects_blank_demo_label() -> None:
    with pytest.raises(ValueError, match="demo_label"):
        _valid_probe(demo_label="")


def test_probe_to_dict_preserves_row_count() -> None:
    ok_probe = _valid_probe()
    assert ok_probe.to_dict()["row_count"] == 5


def test_materialise_k_sweep_refuses_ambient_hardware_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadSweep:
        schema = "bad"
        k_values = (0.4, 0.6, 0.8)
        rows = (object(), object(), object())
        peak_k = 0.8
        falsifier = "f"
        hardware_submission_allowed = True
        hardware_claim_allowed = False

    monkeypatch.setattr(thermo_product, "run_k_sweep_protocol", lambda config=None: _BadSweep())
    with pytest.raises(ValueError, match="hardware"):
        materialise_k_sweep_probe("k_sweep_protocol")


def test_materialise_k_sweep_refuses_empty_ambient_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EmptySweep:
        schema = "s"
        k_values = ()
        rows = ()
        peak_k = 0.0
        falsifier = "f"
        hardware_submission_allowed = False
        hardware_claim_allowed = False

    monkeypatch.setattr(thermo_product, "run_k_sweep_protocol", lambda config=None: _EmptySweep())
    with pytest.raises(ValueError, match="empty"):
        materialise_k_sweep_probe("k_sweep_protocol")


def test_quantum_payload_probe_refuses_wrong_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _bad_payload() -> dict[str, Any]:
        return {
            "schema": "wrong",
            "claim_boundary": AMBIENT_CLAIM_BOUNDARY,
            "hardware_submission_allowed": False,
            "thermodynamic_peak_claim_allowed": False,
            "no_qpu_submission": True,
        }

    monkeypatch.setattr(thermo_product, "quantum_thermo_payload", _bad_payload)
    with pytest.raises(ValueError, match="schema"):
        materialise_quantum_thermo_payload_probe()


def test_quantum_payload_probe_refuses_peak_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _peak_payload() -> dict[str, Any]:
        return {
            "schema": QUANTUM_THERMO_SCHEMA,
            "claim_boundary": AMBIENT_CLAIM_BOUNDARY,
            "hardware_submission_allowed": False,
            "thermodynamic_peak_claim_allowed": True,
            "no_qpu_submission": True,
        }

    monkeypatch.setattr(thermo_product, "quantum_thermo_payload", _peak_payload)
    with pytest.raises(ValueError, match="thermodynamic_peak_claim_allowed"):
        materialise_quantum_thermo_payload_probe()


def test_quantum_payload_probe_refuses_hardware_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _hw_payload() -> dict[str, Any]:
        return {
            "schema": QUANTUM_THERMO_SCHEMA,
            "claim_boundary": AMBIENT_CLAIM_BOUNDARY,
            "hardware_submission_allowed": True,
            "thermodynamic_peak_claim_allowed": False,
            "no_qpu_submission": True,
        }

    monkeypatch.setattr(thermo_product, "quantum_thermo_payload", _hw_payload)
    with pytest.raises(ValueError, match="hardware_submission_allowed"):
        materialise_quantum_thermo_payload_probe()


def test_quantum_payload_probe_refuses_qpu_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _qpu_payload() -> dict[str, Any]:
        return {
            "schema": QUANTUM_THERMO_SCHEMA,
            "claim_boundary": AMBIENT_CLAIM_BOUNDARY,
            "hardware_submission_allowed": False,
            "thermodynamic_peak_claim_allowed": False,
            "no_qpu_submission": False,
        }

    monkeypatch.setattr(thermo_product, "quantum_thermo_payload", _qpu_payload)
    with pytest.raises(ValueError, match="no_qpu_submission"):
        materialise_quantum_thermo_payload_probe()


def test_quantum_payload_probe_refuses_promotional_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boundary_payload() -> dict[str, Any]:
        return {
            "schema": QUANTUM_THERMO_SCHEMA,
            "claim_boundary": "promotional peak",
            "hardware_submission_allowed": False,
            "thermodynamic_peak_claim_allowed": False,
            "no_qpu_submission": True,
        }

    monkeypatch.setattr(thermo_product, "quantum_thermo_payload", _boundary_payload)
    with pytest.raises(ValueError, match="claim_boundary"):
        materialise_quantum_thermo_payload_probe()


def test_verify_ambient_boundary_rejects_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(thermo_product, "AMBIENT_CLAIM_BOUNDARY", "")
    with pytest.raises(ValueError, match="non-empty"):
        verify_ambient_claim_boundary()


def test_verify_ambient_boundary_rejects_missing_peak_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        thermo_product,
        "AMBIENT_CLAIM_BOUNDARY",
        "readiness only; no hardware submission",
    )
    with pytest.raises(ValueError, match="peak"):
        verify_ambient_claim_boundary()


def test_verify_ambient_boundary_rejects_missing_hardware_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        thermo_product,
        "AMBIENT_CLAIM_BOUNDARY",
        "readiness only; no thermodynamic peak claim",
    )
    with pytest.raises(ValueError, match="hardware submission"):
        verify_ambient_claim_boundary()


def test_integrity_rejects_non_mapping_capability_row() -> None:
    registry = build_thermo_readiness_product_registry()
    not_map = dict(registry)
    not_map["capabilities"] = cast(list[dict[str, object]], ["not-a-mapping"])
    with pytest.raises(ValueError, match="mapping"):
        assert_thermo_readiness_product_integrity(not_map)


def test_integrity_rejects_blank_capability_id() -> None:
    registry = build_thermo_readiness_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    blank_id = dict(registry)
    blank_caps = [dict(row) for row in caps]
    blank_caps[0]["capability_id"] = "  "
    blank_id["capabilities"] = blank_caps
    with pytest.raises(ValueError, match="blank"):
        assert_thermo_readiness_product_integrity(blank_id)


def test_integrity_rejects_duplicate_capability_id() -> None:
    registry = build_thermo_readiness_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    dup = dict(registry)
    dup_caps = [dict(row) for row in caps]
    dup_caps[1] = dict(dup_caps[0])
    dup["capabilities"] = dup_caps
    with pytest.raises(ValueError, match="duplicate capability_id"):
        assert_thermo_readiness_product_integrity(dup)


def test_integrity_rejects_missing_k_sweep_protocol() -> None:
    registry = build_thermo_readiness_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    no_ksweep = dict(registry)
    no_k_caps = [dict(row) for row in caps if row["capability_id"] != "k_sweep_protocol"]
    no_ksweep["capabilities"] = no_k_caps
    no_ksweep["capability_count"] = len(no_k_caps)
    with pytest.raises(ValueError, match="k_sweep_protocol|drift"):
        assert_thermo_readiness_product_integrity(no_ksweep)


def test_integrity_rejects_capability_peak_claim_flag() -> None:
    registry = build_thermo_readiness_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    peak_cap = dict(registry)
    peak_caps = [dict(row) for row in caps]
    peak_caps[0]["thermodynamic_peak_claim_allowed"] = True
    peak_cap["capabilities"] = peak_caps
    with pytest.raises(ValueError, match="thermodynamic_peak_claim_allowed"):
        assert_thermo_readiness_product_integrity(peak_cap)


def test_integrity_rejects_blank_ambient_symbol() -> None:
    registry = build_thermo_readiness_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    no_symbol = dict(registry)
    sym_caps = [dict(row) for row in caps]
    sym_caps[0]["ambient_symbol"] = ""
    no_symbol["capabilities"] = sym_caps
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_thermo_readiness_product_integrity(no_symbol)


def test_integrity_rejects_non_mapping_fep_row() -> None:
    registry = build_thermo_readiness_product_registry()
    fep_not_map = dict(registry)
    fep_not_map["fep_inventory"] = cast(list[dict[str, object]], [123])
    with pytest.raises(ValueError, match="mapping"):
        assert_thermo_readiness_product_integrity(fep_not_map)


def test_integrity_rejects_blank_fep_module_id() -> None:
    registry = build_thermo_readiness_product_registry()
    fep_rows = cast(list[dict[str, object]], registry["fep_inventory"])
    blank_fep = dict(registry)
    blank_fep_rows = [dict(row) for row in fep_rows]
    blank_fep_rows[0]["module_id"] = ""
    blank_fep["fep_inventory"] = blank_fep_rows
    with pytest.raises(ValueError, match="module_id"):
        assert_thermo_readiness_product_integrity(blank_fep)


def test_integrity_rejects_duplicate_fep_module_id() -> None:
    registry = build_thermo_readiness_product_registry()
    fep_rows = cast(list[dict[str, object]], registry["fep_inventory"])
    dup_fep = dict(registry)
    dup_fep_rows = [dict(row) for row in fep_rows]
    dup_fep_rows[1] = dict(dup_fep_rows[0])
    dup_fep["fep_inventory"] = dup_fep_rows
    with pytest.raises(ValueError, match="duplicate module_id"):
        assert_thermo_readiness_product_integrity(dup_fep)


def test_integrity_rejects_fep_set_drift() -> None:
    registry = build_thermo_readiness_product_registry()
    fep_rows = cast(list[dict[str, object]], registry["fep_inventory"])
    drifted = dict(registry)
    reduced = [dict(row) for row in fep_rows if row["module_id"] != "predictive_coding"]
    drifted["fep_inventory"] = reduced
    drifted["fep_inventory_count"] = len(reduced)
    with pytest.raises(ValueError, match="registry FEP set drift"):
        assert_thermo_readiness_product_integrity(drifted)


def test_integrity_rejects_capability_count_mismatch() -> None:
    registry = build_thermo_readiness_product_registry()
    count_mismatch = dict(registry)
    count_mismatch["capability_count"] = 99
    with pytest.raises(ValueError, match="capability_count"):
        assert_thermo_readiness_product_integrity(count_mismatch)


def test_integrity_rejects_fep_inventory_count_mismatch() -> None:
    registry = build_thermo_readiness_product_registry()
    fep_count = dict(registry)
    fep_count["fep_inventory_count"] = 99
    with pytest.raises(ValueError, match="fep_inventory_count"):
        assert_thermo_readiness_product_integrity(fep_count)


def test_request_digest_rejects_non_finite_k_value() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_k_sweep_request_digest(
            k_values=(0.4, float("nan"), 0.8),
            transition_k=0.4,
        )


def test_request_digest_rejects_non_finite_transition_k() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_k_sweep_request_digest(
            k_values=(0.4, 0.6, 0.8),
            transition_k=float("inf"),
        )


def test_capability_map_rejects_blank_id(monkeypatch: pytest.MonkeyPatch) -> None:
    blank = _valid_capability_row(capability_id="tmp")
    object.__setattr__(blank, "capability_id", "  ")
    monkeypatch.setattr(thermo_product, "_CAPABILITIES", (blank,))
    with pytest.raises(RuntimeError, match="blank capability_id"):
        thermo_product._capability_map()


def test_capability_map_rejects_duplicate_id(monkeypatch: pytest.MonkeyPatch) -> None:
    row = get_readiness_capability("k_sweep_protocol")
    monkeypatch.setattr(thermo_product, "_CAPABILITIES", (row, row))
    with pytest.raises(RuntimeError, match="duplicate capability_id"):
        thermo_product._capability_map()


def test_capability_map_rejects_empty_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(thermo_product, "_CAPABILITIES", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        thermo_product._capability_map()


def test_fep_map_rejects_blank_module_id(monkeypatch: pytest.MonkeyPatch) -> None:
    blank = FepInventoryRow(module_id="tmp", module_path="m", title="t", summary="s")
    object.__setattr__(blank, "module_id", "  ")
    monkeypatch.setattr(thermo_product, "_FEP_INVENTORY", (blank,))
    with pytest.raises(RuntimeError, match="blank module_id"):
        thermo_product._fep_map()


def test_fep_map_rejects_duplicate_module_id(monkeypatch: pytest.MonkeyPatch) -> None:
    row = get_fep_inventory_row("predictive_coding")
    monkeypatch.setattr(thermo_product, "_FEP_INVENTORY", (row, row))
    with pytest.raises(RuntimeError, match="duplicate module_id"):
        thermo_product._fep_map()


def test_fep_map_rejects_empty_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(thermo_product, "_FEP_INVENTORY", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        thermo_product._fep_map()


def test_module_exports_stable() -> None:
    assert "assert_thermo_readiness_product_integrity" in thermo_product.__all__
    assert "materialise_demo_k_sweep_probe" in thermo_product.__all__
    assert THERMO_READINESS_PRODUCT_SCHEMA == "thermo_readiness_product.v1"
