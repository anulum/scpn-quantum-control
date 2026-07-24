# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for control stack compose product (BL-67)
"""Real-surface tests for ``scpn_quantum_control.control_stack_compose_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.control_stack_compose_product as control_stack_compose_product
from scpn_quantum_control.control_stack_compose_product import (
    CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY,
    CONTROL_STACK_COMPOSE_PRODUCT_SCHEMA,
    AdapterPortRow,
    MaterialisedClosedLoopTelemetryProbe,
    OwnershipRow,
    PathEligibilityDecision,
    assert_control_stack_compose_product_integrity,
    build_control_stack_compose_product_registry,
    decide_control_compose_path,
    get_adapter_port,
    get_ownership_row,
    iter_ownership_rows,
    list_adapter_port_ids,
    list_ownership_module_ids,
    map_control_stack_compose_public_surfaces,
    materialise_closed_loop_telemetry_probe,
    materialise_demo_closed_loop_telemetry_probe,
)


def test_list_and_filters() -> None:
    modules = list_ownership_module_ids()
    assert "realtime_feedback" in modules
    assert "execution_policy_gate" in modules
    assert "realtime_runtime" in modules
    assert len(modules) == 8
    assert modules == list_ownership_module_ids()
    ports = list_adapter_port_ids()
    assert "closed_loop_telemetry" in ports
    assert "execution_policy_gate" in ports
    assert len(ports) == 6
    policy_only = iter_ownership_rows(support_posture="policy_only")
    assert policy_only
    assert all(row.support_posture == "policy_only" for row in policy_only)
    empty = iter_ownership_rows(support_posture="live_hardware_gated")
    assert any(row.module_id == "hardware_feedback_dryrun" for row in empty)


def test_get_known_and_unknown_fail_closed() -> None:
    row = get_ownership_row("realtime_feedback")
    assert row.claim_boundary == CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY
    assert row.rewrites_forbidden is True
    assert row.adapter_port == "realtime_feedback"
    rt = get_ownership_row("realtime_runtime")
    assert rt.rewrites_forbidden is True
    assert rt.adapter_port is None
    port = get_adapter_port("closed_loop_telemetry")
    assert port.requires_execution_policy is True
    assert port.invent_green_pcs is False
    with pytest.raises(ValueError, match="non-empty"):
        get_ownership_row("  ")
    with pytest.raises(ValueError, match="unknown module_id"):
        get_ownership_row("not_a_module")
    with pytest.raises(ValueError, match="non-empty"):
        get_adapter_port("")
    with pytest.raises(ValueError, match="unknown port_id"):
        get_adapter_port("ghost_port")


def test_decide_control_compose_path() -> None:
    refused = decide_control_compose_path("realtime_feedback", policy_present=False)
    assert refused.allowed is False
    assert any("ClosedLoopExecutionPolicy" in b for b in refused.blockers)

    allowed = decide_control_compose_path("realtime_feedback", policy_present=True)
    assert allowed.allowed is True

    pcs = decide_control_compose_path(
        "execution_policy_gate",
        policy_present=True,
        invent_green_pcs=True,
    )
    assert pcs.allowed is False
    assert any("PCS" in b for b in pcs.blockers)

    rewrite = decide_control_compose_path(
        "closed_loop_telemetry",
        policy_present=True,
        rewrite_realtime_runtime=True,
    )
    assert rewrite.allowed is False
    assert any("rewrite" in b.lower() for b in rewrite.blockers)


def test_closed_loop_telemetry_probe() -> None:
    probe = materialise_demo_closed_loop_telemetry_probe()
    assert probe.authorised is True
    assert probe.mode == "simulation"
    assert probe.invent_green_pcs is False
    assert probe.allow_hardware is False
    assert probe.live_ticket_present is False
    assert probe.requested_rounds == 1
    payload = probe.to_dict()
    assert payload["invent_green_pcs"] is False

    hw = materialise_closed_loop_telemetry_probe(
        allow_hardware=True,
        live_ticket=None,
        requested_rounds=1,
    )
    # Ambient fail-closed: hardware without ticket is not invent-green authorised
    assert hw.invent_green_pcs is False
    assert hw.live_ticket_present is False


def test_public_surfaces_and_registry() -> None:
    surfaces = map_control_stack_compose_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.control_stack_compose_product" in paths
    assert "scpn_quantum_control.control.closed_loop_analysis" in paths
    assert "scpn_quantum_control.control.realtime_runtime" in paths

    registry = build_control_stack_compose_product_registry()
    assert registry["schema"] == CONTROL_STACK_COMPOSE_PRODUCT_SCHEMA
    assert registry["invent_green_pcs_policy"] is False
    assert registry["rewrites_forbidden_policy"] is True
    validated = assert_control_stack_compose_product_integrity(registry)
    assert validated["ownership_count"] == 8
    assert validated["port_count"] == 6
    assert assert_control_stack_compose_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_policy() -> None:
    registry = build_control_stack_compose_product_registry()
    ownership = cast(list[dict[str, object]], list(registry["ownership"]))
    ports = cast(list[dict[str, object]], list(registry["adapter_ports"]))

    broken = dict(registry)
    broken["ownership"] = ownership + [
        {
            "module_id": "ghost",
            "module_path": "x",
            "owner_kind": "control_realtime",
            "title": "t",
            "summary": "s",
            "adapter_port": None,
            "support_posture": "local_research",
            "rewrites_forbidden": True,
            "as_of": "2026-07-24",
            "claim_boundary": CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY,
        }
    ]
    broken["ownership_count"] = len(cast(list[object], broken["ownership"]))
    with pytest.raises(ValueError, match="drift"):
        assert_control_stack_compose_product_integrity(broken)

    empty: dict[str, object] = {
        "ownership": [],
        "adapter_ports": ports,
        "blank_entry_count": 0,
        "ownership_count": 0,
        "port_count": len(ports),
    }
    with pytest.raises(ValueError, match="non-empty ownership"):
        assert_control_stack_compose_product_integrity(empty)

    no_ports = dict(registry)
    no_ports["adapter_ports"] = []
    no_ports["port_count"] = 0
    with pytest.raises(ValueError, match="non-empty adapter_ports"):
        assert_control_stack_compose_product_integrity(no_ports)

    policy = dict(registry)
    policy["invent_green_pcs_policy"] = True
    with pytest.raises(ValueError, match="invent_green_pcs_policy"):
        assert_control_stack_compose_product_integrity(policy)

    rewrites = dict(registry)
    rewrites["rewrites_forbidden_policy"] = False
    with pytest.raises(ValueError, match="rewrites_forbidden_policy"):
        assert_control_stack_compose_product_integrity(rewrites)


def test_integrity_rejects_blank_invalid() -> None:
    registry = build_control_stack_compose_product_registry()
    ownership = cast(list[dict[str, object]], list(registry["ownership"]))
    ports = cast(list[dict[str, object]], list(registry["adapter_ports"]))

    non_map = dict(registry)
    non_map["ownership"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_control_stack_compose_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in ownership]
    rows[0]["module_id"] = "  "
    blank_id["ownership"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_control_stack_compose_product_integrity(blank_id)

    no_path = dict(registry)
    brows = [dict(row) for row in ownership]
    brows[0]["module_path"] = ""
    no_path["ownership"] = brows
    with pytest.raises(ValueError, match="module_path"):
        assert_control_stack_compose_product_integrity(no_path)

    rewrites_off = dict(registry)
    rrows = [dict(row) for row in ownership]
    rrows[0]["rewrites_forbidden"] = False
    rewrites_off["ownership"] = rrows
    with pytest.raises(ValueError, match="rewrites_forbidden"):
        assert_control_stack_compose_product_integrity(rewrites_off)

    no_gate = dict(registry)
    without = [dict(row) for row in ownership if row.get("module_id") != "execution_policy_gate"]
    no_gate["ownership"] = without
    no_gate["ownership_count"] = len(without)
    with pytest.raises(ValueError, match="missing execution_policy_gate|drift"):
        assert_control_stack_compose_product_integrity(no_gate)

    dup = dict(registry)
    drows = [dict(row) for row in ownership]
    drows.append(dict(drows[0]))
    dup["ownership"] = drows
    dup["ownership_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate module_id"):
        assert_control_stack_compose_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_control_stack_compose_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["ownership_count"] = 0
    with pytest.raises(ValueError, match="ownership_count"):
        assert_control_stack_compose_product_integrity(count_mismatch)

    port_non_map = dict(registry)
    port_non_map["adapter_ports"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_control_stack_compose_product_integrity(port_non_map)

    port_blank = dict(registry)
    prows = [dict(row) for row in ports]
    prows[0]["port_id"] = ""
    port_blank["adapter_ports"] = prows
    with pytest.raises(ValueError, match="blank or invalid port_id"):
        assert_control_stack_compose_product_integrity(port_blank)

    port_no_policy = dict(registry)
    pn = [dict(row) for row in ports]
    pn[0]["requires_execution_policy"] = False
    port_no_policy["adapter_ports"] = pn
    with pytest.raises(ValueError, match="require_execution_policy"):
        assert_control_stack_compose_product_integrity(port_no_policy)

    port_pcs = dict(registry)
    pc = [dict(row) for row in ports]
    pc[0]["invent_green_pcs"] = True
    port_pcs["adapter_ports"] = pc
    with pytest.raises(ValueError, match="invent_green_pcs"):
        assert_control_stack_compose_product_integrity(port_pcs)

    port_dup = dict(registry)
    pd = [dict(row) for row in ports]
    pd.append(dict(pd[0]))
    port_dup["adapter_ports"] = pd
    port_dup["port_count"] = len(pd)
    with pytest.raises(ValueError, match="duplicate port_id"):
        assert_control_stack_compose_product_integrity(port_dup)

    port_count_bad = dict(registry)
    port_count_bad["port_count"] = 0
    with pytest.raises(ValueError, match="port_count"):
        assert_control_stack_compose_product_integrity(port_count_bad)

    port_drift = dict(registry)
    pruned = [dict(row) for row in ports if row.get("port_id") != "qaoa_mpc_optional"]
    port_drift["adapter_ports"] = pruned
    port_drift["port_count"] = len(pruned)
    with pytest.raises(ValueError, match="port set drift"):
        assert_control_stack_compose_product_integrity(port_drift)


def test_module_exports() -> None:
    assert "materialise_demo_closed_loop_telemetry_probe" in control_stack_compose_product.__all__
    assert "decide_control_compose_path" in control_stack_compose_product.__all__
    assert "list_ownership_module_ids" in control_stack_compose_product.__all__


def test_row_decision_probe_validation() -> None:
    base_own: dict[str, Any] = {
        "module_id": "x",
        "module_path": "pkg.x",
        "owner_kind": "control_realtime",
        "title": "t",
        "summary": "s",
        "adapter_port": "realtime_feedback",
        "support_posture": "local_research",
        "rewrites_forbidden": True,
    }
    assert OwnershipRow(**base_own).module_id == "x"
    assert OwnershipRow(**base_own).to_dict()["module_id"] == "x"
    with pytest.raises(ValueError, match="module_id"):
        OwnershipRow(**{**base_own, "module_id": ""})
    with pytest.raises(ValueError, match="module_path"):
        OwnershipRow(**{**base_own, "module_path": ""})
    with pytest.raises(ValueError, match="title"):
        OwnershipRow(**{**base_own, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        OwnershipRow(**{**base_own, "summary": ""})
    with pytest.raises(ValueError, match="owner_kind"):
        OwnershipRow(**{**base_own, "owner_kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="support_posture"):
        OwnershipRow(**{**base_own, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="adapter_port"):
        OwnershipRow(**{**base_own, "adapter_port": cast(Any, "nope")})
    with pytest.raises(ValueError, match="as_of"):
        OwnershipRow(**{**base_own, "as_of": ""})

    base_port: dict[str, Any] = {
        "port_id": "realtime_feedback",
        "title": "t",
        "ambient_modules": ("pkg.x",),
        "bl47_pointer": "p",
        "support_posture": "local_research",
        "requires_execution_policy": True,
    }
    assert AdapterPortRow(**base_port).port_id == "realtime_feedback"
    assert AdapterPortRow(**base_port).to_dict()["port_id"] == "realtime_feedback"
    with pytest.raises(ValueError, match="port_id"):
        AdapterPortRow(**{**base_port, "port_id": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        AdapterPortRow(**{**base_port, "title": ""})
    with pytest.raises(ValueError, match="ambient_modules"):
        AdapterPortRow(**{**base_port, "ambient_modules": ()})
    with pytest.raises(ValueError, match="ambient_modules entries"):
        AdapterPortRow(**{**base_port, "ambient_modules": ("ok", "  ")})
    with pytest.raises(ValueError, match="bl47_pointer"):
        AdapterPortRow(**{**base_port, "bl47_pointer": ""})
    with pytest.raises(ValueError, match="support_posture"):
        AdapterPortRow(**{**base_port, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="invent_green_pcs"):
        AdapterPortRow(**{**base_port, "invent_green_pcs": True})
    with pytest.raises(ValueError, match="as_of"):
        AdapterPortRow(**{**base_port, "as_of": ""})

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
            reason="  ",
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
    with pytest.raises(ValueError, match="require blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
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
            blockers=("ok", "  "),
        )
    assert (
        decide_control_compose_path("execution_policy_gate", policy_present=True).to_dict()[
            "allowed"
        ]
        is True
    )

    with pytest.raises(ValueError, match="requested_rounds must be positive"):
        materialise_closed_loop_telemetry_probe(requested_rounds=0)

    with pytest.raises(ValueError, match="requested_rounds must be positive"):
        MaterialisedClosedLoopTelemetryProbe(
            authorised=True,
            mode="simulation",
            reason="r",
            requested_rounds=0,
            invent_green_pcs=False,
            allow_hardware=False,
            live_ticket_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="mode"):
        MaterialisedClosedLoopTelemetryProbe(
            authorised=True,
            mode="",
            reason="r",
            requested_rounds=1,
            invent_green_pcs=False,
            allow_hardware=False,
            live_ticket_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="reason"):
        MaterialisedClosedLoopTelemetryProbe(
            authorised=True,
            mode="simulation",
            reason="  ",
            requested_rounds=1,
            invent_green_pcs=False,
            allow_hardware=False,
            live_ticket_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_pcs"):
        MaterialisedClosedLoopTelemetryProbe(
            authorised=True,
            mode="simulation",
            reason="r",
            requested_rounds=1,
            invent_green_pcs=True,
            allow_hardware=False,
            live_ticket_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedClosedLoopTelemetryProbe(
            authorised=True,
            mode="simulation",
            reason="r",
            requested_rounds=1,
            invent_green_pcs=False,
            allow_hardware=False,
            live_ticket_present=False,
            demo_label="",
        )
    with pytest.raises(ValueError, match="unknown mode"):
        MaterialisedClosedLoopTelemetryProbe(
            authorised=True,
            mode="telepathy",
            reason="r",
            requested_rounds=1,
            invent_green_pcs=False,
            allow_hardware=False,
            live_ticket_present=False,
            demo_label="d",
        )


def test_catalogue_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive catalogue RuntimeError paths."""
    monkeypatch.setattr(control_stack_compose_product, "_OWNERSHIP", ())
    with pytest.raises(RuntimeError, match="ownership catalogue must be non-empty"):
        control_stack_compose_product._ownership_map()

    blank = OwnershipRow(
        module_id="tmp",
        module_path="pkg.tmp",
        owner_kind="control_realtime",
        title="t",
        summary="s",
        adapter_port=None,
        support_posture="local_research",
        rewrites_forbidden=True,
    )
    object.__setattr__(blank, "module_id", "  ")
    monkeypatch.setattr(control_stack_compose_product, "_OWNERSHIP", (blank,))
    with pytest.raises(RuntimeError, match="blank module_id"):
        control_stack_compose_product._ownership_map()

    good = OwnershipRow(
        module_id="dup",
        module_path="pkg.dup",
        owner_kind="control_realtime",
        title="t",
        summary="s",
        adapter_port=None,
        support_posture="local_research",
        rewrites_forbidden=True,
    )
    monkeypatch.setattr(control_stack_compose_product, "_OWNERSHIP", (good, good))
    with pytest.raises(RuntimeError, match="duplicate module_id"):
        control_stack_compose_product._ownership_map()

    monkeypatch.setattr(control_stack_compose_product, "_PORTS", ())
    with pytest.raises(RuntimeError, match="adapter port catalogue must be non-empty"):
        control_stack_compose_product._port_map()

    port = AdapterPortRow(
        port_id="realtime_feedback",
        title="t",
        ambient_modules=("pkg.x",),
        bl47_pointer="p",
        support_posture="local_research",
        requires_execution_policy=True,
    )
    monkeypatch.setattr(control_stack_compose_product, "_PORTS", (port, port))
    with pytest.raises(RuntimeError, match="duplicate port_id"):
        control_stack_compose_product._port_map()


def test_iter_ownership_rows_without_filter_returns_full_catalogue() -> None:
    """Unfiltered ownership iter returns every catalogue row."""
    rows = iter_ownership_rows()
    assert len(rows) == len(list_ownership_module_ids())
    assert {row.module_id for row in rows} == set(list_ownership_module_ids())


def test_port_map_rejects_blank_port_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank adapter port_id is refused at catalogue build."""
    blank = AdapterPortRow(
        port_id="realtime_feedback",
        title="t",
        ambient_modules=("pkg.x",),
        bl47_pointer="p",
        support_posture="local_research",
        requires_execution_policy=True,
    )
    object.__setattr__(blank, "port_id", "  ")
    monkeypatch.setattr(control_stack_compose_product, "_PORTS", (blank,))
    with pytest.raises(RuntimeError, match="blank port_id"):
        control_stack_compose_product._port_map()
