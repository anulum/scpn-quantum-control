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


def test_list_and_filters() -> None:
    ids = list_readiness_capability_ids()
    assert "k_sweep_protocol" in ids
    assert "entropy_production" in ids
    assert "work_identity" in ids
    assert "heat_dissipation" in ids
    assert "claim_boundary_gate" in ids
    assert len(ids) == 5
    fep_ids = list_fep_module_ids()
    assert "predictive_coding" in fep_ids
    assert "variational_free_energy" in fep_ids
    assert len(fep_ids) == 2
    sweeps = iter_readiness_capabilities(kind="k_sweep_protocol")
    assert len(sweeps) == 1
    assert sweeps[0].capability_id == "k_sweep_protocol"
    research = iter_fep_inventory(status="research_only")
    assert len(research) == 2


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


def test_dataclass_invariants() -> None:
    with pytest.raises(ValueError, match="capability_id"):
        ReadinessCapabilityRow(
            capability_id="",
            kind="k_sweep_protocol",
            title="t",
            summary="s",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="unknown capability kind"):
        ReadinessCapabilityRow(
            capability_id="x",
            kind=cast(Any, "not_a_kind"),
            title="t",
            summary="s",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="title"):
        ReadinessCapabilityRow(
            capability_id="x",
            kind="k_sweep_protocol",
            title="",
            summary="s",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="summary"):
        ReadinessCapabilityRow(
            capability_id="x",
            kind="k_sweep_protocol",
            title="t",
            summary="  ",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="ambient_symbol"):
        ReadinessCapabilityRow(
            capability_id="x",
            kind="k_sweep_protocol",
            title="t",
            summary="s",
            ambient_symbol="",
        )
    with pytest.raises(ValueError, match="hardware_submission_allowed"):
        ReadinessCapabilityRow(
            capability_id="x",
            kind="k_sweep_protocol",
            title="t",
            summary="s",
            ambient_symbol="x",
            hardware_submission_allowed=True,
        )
    with pytest.raises(ValueError, match="thermodynamic_peak_claim_allowed"):
        ReadinessCapabilityRow(
            capability_id="x",
            kind="k_sweep_protocol",
            title="t",
            summary="s",
            ambient_symbol="x",
            thermodynamic_peak_claim_allowed=True,
        )
    with pytest.raises(ValueError, match="support_posture"):
        ReadinessCapabilityRow(
            capability_id="x",
            kind="k_sweep_protocol",
            title="t",
            summary="s",
            ambient_symbol="x",
            support_posture=cast(Any, "bogus"),
        )
    with pytest.raises(ValueError, match="as_of"):
        ReadinessCapabilityRow(
            capability_id="x",
            kind="k_sweep_protocol",
            title="t",
            summary="s",
            ambient_symbol="x",
            as_of="",
        )
    ok_cap = ReadinessCapabilityRow(
        capability_id="x",
        kind="k_sweep_protocol",
        title="t",
        summary="s",
        ambient_symbol="x",
    )
    assert ok_cap.to_dict()["capability_id"] == "x"

    with pytest.raises(ValueError, match="module_id"):
        FepInventoryRow(module_id="", module_path="m", title="t", summary="s")
    with pytest.raises(ValueError, match="module_path"):
        FepInventoryRow(module_id="x", module_path="", title="t", summary="s")
    with pytest.raises(ValueError, match="title"):
        FepInventoryRow(module_id="x", module_path="m", title="", summary="s")
    with pytest.raises(ValueError, match="summary"):
        FepInventoryRow(module_id="x", module_path="m", title="t", summary="")
    with pytest.raises(ValueError, match="status"):
        FepInventoryRow(
            module_id="x",
            module_path="m",
            title="t",
            summary="s",
            status=cast(Any, "bogus"),
        )
    with pytest.raises(ValueError, match="bl84_pointer"):
        FepInventoryRow(
            module_id="x",
            module_path="m",
            title="t",
            summary="s",
            bl84_pointer="",
        )
    with pytest.raises(ValueError, match="product_hook_proven"):
        FepInventoryRow(
            module_id="x",
            module_path="m",
            title="t",
            summary="s",
            product_hook_proven=True,
        )
    ok_fep = FepInventoryRow(module_id="x", module_path="m", title="t", summary="s")
    assert ok_fep.to_dict()["status"] == "research_only"

    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            outcome=cast(Any, "maybe"),
            allowed=True,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="",
            blockers=(),
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
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="no",
            blockers=(),
        )
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="no",
            blockers=("",),
        )
    ok_dec = PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason="ok",
        blockers=(),
    )
    assert ok_dec.to_dict()["allowed"] is True

    with pytest.raises(ValueError, match="capability_id"):
        MaterialisedKSweepProbe(
            capability_id="",
            schema="s",
            peak_k=0.8,
            row_count=5,
            hardware_submission_allowed=False,
            thermodynamic_peak_claim_allowed=False,
            ambient_claim_boundary=AMBIENT_CLAIM_BOUNDARY,
            falsifier="f",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="schema"):
        MaterialisedKSweepProbe(
            capability_id="k_sweep_protocol",
            schema="",
            peak_k=0.8,
            row_count=5,
            hardware_submission_allowed=False,
            thermodynamic_peak_claim_allowed=False,
            ambient_claim_boundary=AMBIENT_CLAIM_BOUNDARY,
            falsifier="f",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="row_count"):
        MaterialisedKSweepProbe(
            capability_id="k_sweep_protocol",
            schema="s",
            peak_k=0.8,
            row_count=2,
            hardware_submission_allowed=False,
            thermodynamic_peak_claim_allowed=False,
            ambient_claim_boundary=AMBIENT_CLAIM_BOUNDARY,
            falsifier="f",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="hardware_submission_allowed"):
        MaterialisedKSweepProbe(
            capability_id="k_sweep_protocol",
            schema="s",
            peak_k=0.8,
            row_count=5,
            hardware_submission_allowed=True,
            thermodynamic_peak_claim_allowed=False,
            ambient_claim_boundary=AMBIENT_CLAIM_BOUNDARY,
            falsifier="f",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="thermodynamic_peak_claim_allowed"):
        MaterialisedKSweepProbe(
            capability_id="k_sweep_protocol",
            schema="s",
            peak_k=0.8,
            row_count=5,
            hardware_submission_allowed=False,
            thermodynamic_peak_claim_allowed=True,
            ambient_claim_boundary=AMBIENT_CLAIM_BOUNDARY,
            falsifier="f",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="ambient_claim_boundary"):
        MaterialisedKSweepProbe(
            capability_id="k_sweep_protocol",
            schema="s",
            peak_k=0.8,
            row_count=5,
            hardware_submission_allowed=False,
            thermodynamic_peak_claim_allowed=False,
            ambient_claim_boundary="",
            falsifier="f",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="no-thermodynamic-peak|ambient_claim_boundary"):
        MaterialisedKSweepProbe(
            capability_id="k_sweep_protocol",
            schema="s",
            peak_k=0.8,
            row_count=5,
            hardware_submission_allowed=False,
            thermodynamic_peak_claim_allowed=False,
            ambient_claim_boundary="peak claim allowed",
            falsifier="f",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="falsifier"):
        MaterialisedKSweepProbe(
            capability_id="k_sweep_protocol",
            schema="s",
            peak_k=0.8,
            row_count=5,
            hardware_submission_allowed=False,
            thermodynamic_peak_claim_allowed=False,
            ambient_claim_boundary=AMBIENT_CLAIM_BOUNDARY,
            falsifier="",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="probe_digest"):
        MaterialisedKSweepProbe(
            capability_id="k_sweep_protocol",
            schema="s",
            peak_k=0.8,
            row_count=5,
            hardware_submission_allowed=False,
            thermodynamic_peak_claim_allowed=False,
            ambient_claim_boundary=AMBIENT_CLAIM_BOUNDARY,
            falsifier="f",
            probe_digest="",
            demo_label="d",
        )
    with pytest.raises(ValueError, match="64-char"):
        MaterialisedKSweepProbe(
            capability_id="k_sweep_protocol",
            schema="s",
            peak_k=0.8,
            row_count=5,
            hardware_submission_allowed=False,
            thermodynamic_peak_claim_allowed=False,
            ambient_claim_boundary=AMBIENT_CLAIM_BOUNDARY,
            falsifier="f",
            probe_digest="abc",
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedKSweepProbe(
            capability_id="k_sweep_protocol",
            schema="s",
            peak_k=0.8,
            row_count=5,
            hardware_submission_allowed=False,
            thermodynamic_peak_claim_allowed=False,
            ambient_claim_boundary=AMBIENT_CLAIM_BOUNDARY,
            falsifier="f",
            probe_digest="a" * 64,
            demo_label="",
        )
    ok_probe = MaterialisedKSweepProbe(
        capability_id="k_sweep_protocol",
        schema="s",
        peak_k=0.8,
        row_count=5,
        hardware_submission_allowed=False,
        thermodynamic_peak_claim_allowed=False,
        ambient_claim_boundary=AMBIENT_CLAIM_BOUNDARY,
        falsifier="f",
        probe_digest="a" * 64,
        demo_label="d",
    )
    assert ok_probe.to_dict()["row_count"] == 5


def test_ambient_probe_monkeypatch_honesty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse when ambient invent-green flags appear (fail-closed)."""

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


def test_verify_ambient_boundary_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(thermo_product, "AMBIENT_CLAIM_BOUNDARY", "")
    with pytest.raises(ValueError, match="non-empty"):
        verify_ambient_claim_boundary()
    monkeypatch.setattr(
        thermo_product,
        "AMBIENT_CLAIM_BOUNDARY",
        "readiness only; no hardware submission",
    )
    with pytest.raises(ValueError, match="peak"):
        verify_ambient_claim_boundary()
    monkeypatch.setattr(
        thermo_product,
        "AMBIENT_CLAIM_BOUNDARY",
        "readiness only; no thermodynamic peak claim",
    )
    with pytest.raises(ValueError, match="hardware submission"):
        verify_ambient_claim_boundary()


def test_integrity_more_edge_cases() -> None:
    registry = build_thermo_readiness_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    not_map = dict(registry)
    not_map["capabilities"] = cast(list[dict[str, object]], ["not-a-mapping"])
    with pytest.raises(ValueError, match="mapping"):
        assert_thermo_readiness_product_integrity(not_map)

    blank_id = dict(registry)
    blank_caps = [dict(row) for row in caps]
    blank_caps[0]["capability_id"] = "  "
    blank_id["capabilities"] = blank_caps
    with pytest.raises(ValueError, match="blank"):
        assert_thermo_readiness_product_integrity(blank_id)

    dup = dict(registry)
    dup_caps = [dict(row) for row in caps]
    dup_caps[1] = dict(dup_caps[0])
    dup["capabilities"] = dup_caps
    with pytest.raises(ValueError, match="duplicate capability_id"):
        assert_thermo_readiness_product_integrity(dup)

    no_ksweep = dict(registry)
    no_k_caps = [dict(row) for row in caps if row["capability_id"] != "k_sweep_protocol"]
    # keep count consistent so we hit missing k_sweep rather than count mismatch first
    no_ksweep["capabilities"] = no_k_caps
    no_ksweep["capability_count"] = len(no_k_caps)
    with pytest.raises(ValueError, match="k_sweep_protocol|drift"):
        assert_thermo_readiness_product_integrity(no_ksweep)

    peak_cap = dict(registry)
    peak_caps = [dict(row) for row in caps]
    peak_caps[0]["thermodynamic_peak_claim_allowed"] = True
    peak_cap["capabilities"] = peak_caps
    with pytest.raises(ValueError, match="thermodynamic_peak_claim_allowed"):
        assert_thermo_readiness_product_integrity(peak_cap)

    no_symbol = dict(registry)
    sym_caps = [dict(row) for row in caps]
    sym_caps[0]["ambient_symbol"] = ""
    no_symbol["capabilities"] = sym_caps
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_thermo_readiness_product_integrity(no_symbol)

    fep_rows = cast(list[dict[str, object]], registry["fep_inventory"])
    fep_not_map = dict(registry)
    fep_not_map["fep_inventory"] = cast(list[dict[str, object]], [123])
    with pytest.raises(ValueError, match="mapping"):
        assert_thermo_readiness_product_integrity(fep_not_map)

    blank_fep = dict(registry)
    blank_fep_rows = [dict(row) for row in fep_rows]
    blank_fep_rows[0]["module_id"] = ""
    blank_fep["fep_inventory"] = blank_fep_rows
    with pytest.raises(ValueError, match="module_id"):
        assert_thermo_readiness_product_integrity(blank_fep)

    dup_fep = dict(registry)
    dup_fep_rows = [dict(row) for row in fep_rows]
    dup_fep_rows[1] = dict(dup_fep_rows[0])
    dup_fep["fep_inventory"] = dup_fep_rows
    with pytest.raises(ValueError, match="duplicate module_id"):
        assert_thermo_readiness_product_integrity(dup_fep)

    count_mismatch = dict(registry)
    count_mismatch["capability_count"] = 99
    with pytest.raises(ValueError, match="capability_count"):
        assert_thermo_readiness_product_integrity(count_mismatch)

    fep_count = dict(registry)
    fep_count["fep_inventory_count"] = 99
    with pytest.raises(ValueError, match="fep_inventory_count"):
        assert_thermo_readiness_product_integrity(fep_count)

    # finite digest edges
    with pytest.raises(ValueError, match="finite"):
        compute_k_sweep_request_digest(
            k_values=(0.4, float("nan"), 0.8),
            transition_k=0.4,
        )
    with pytest.raises(ValueError, match="finite"):
        compute_k_sweep_request_digest(
            k_values=(0.4, 0.6, 0.8),
            transition_k=float("inf"),
        )


def test_module_exports_stable() -> None:
    assert "assert_thermo_readiness_product_integrity" in thermo_product.__all__
    assert "materialise_demo_k_sweep_probe" in thermo_product.__all__
    assert THERMO_READINESS_PRODUCT_SCHEMA == "thermo_readiness_product.v1"
