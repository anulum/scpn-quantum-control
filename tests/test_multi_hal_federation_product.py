# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for multi-HAL federation product (BL-75)
"""Real-surface tests for ``scpn_quantum_control.multi_hal_federation_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.multi_hal_federation_product as multi_hal_federation_product
from scpn_quantum_control.multi_hal_federation_product import (
    MULTI_HAL_FEDERATION_CLAIM_BOUNDARY,
    MULTI_HAL_FEDERATION_PRODUCT_SCHEMA,
    HalCapabilityRecord,
    MaterialisedFederationDryRunProbe,
    PathEligibilityDecision,
    assert_multi_hal_federation_product_integrity,
    build_federation_matrix,
    build_multi_hal_federation_product_registry,
    decide_federation_route,
    get_hal_capability,
    iter_hal_capabilities,
    list_hal_backend_ids,
    list_hal_providers,
    map_multi_hal_federation_public_surfaces,
    materialise_demo_federation_dry_run_probe,
    materialise_federation_dry_run_probe,
)


def test_list_and_filters() -> None:
    ids = list_hal_backend_ids()
    assert len(ids) >= 10
    assert ids == list_hal_backend_ids()
    providers = list_hal_providers()
    assert len(providers) >= 5
    assert len(providers) == len(set(providers))
    matrix = build_federation_matrix()
    assert len(matrix) == len(ids)
    assert all(row["no_submit_default"] is True for row in matrix)

    first = get_hal_capability(ids[0])
    filtered = iter_hal_capabilities(provider=first.provider)
    assert filtered
    assert all(row.provider == first.provider for row in filtered)
    empty = iter_hal_capabilities(provider="__no_such_provider__")
    assert empty == ()
    pulse = iter_hal_capabilities(supports_pulse=True)
    assert all(row.supports_pulse is True for row in pulse)
    posture = iter_hal_capabilities(support_posture="metadata_only")
    assert posture


def test_get_known_and_unknown_fail_closed() -> None:
    backend_id = list_hal_backend_ids()[0]
    row = get_hal_capability(backend_id)
    assert row.claim_boundary == MULTI_HAL_FEDERATION_CLAIM_BOUNDARY
    assert row.no_submit_default is True
    assert row.adapter_module
    assert row.ir_formats
    with pytest.raises(ValueError, match="non-empty"):
        get_hal_capability("  ")
    with pytest.raises(ValueError, match="unknown backend_id"):
        get_hal_capability("not_a_backend")


def test_decide_federation_route() -> None:
    backend_id = list_hal_backend_ids()[0]
    dry = decide_federation_route(backend_id, mode="dry_run")
    assert dry.allowed is True

    network = decide_federation_route(backend_id, mode="dry_run", allow_network=True)
    assert network.allowed is False
    assert any("network" in b.lower() for b in network.blockers)

    invent = decide_federation_route(backend_id, mode="dry_run", invent_green_live_submit=True)
    assert invent.allowed is False
    assert any("invent-green" in b.lower() or "live submit" in b.lower() for b in invent.blockers)

    no_ticket = decide_federation_route(backend_id, mode="ticketed_prep")
    assert no_ticket.allowed is False
    assert any("ticket" in b.lower() for b in no_ticket.blockers)

    ticketed = decide_federation_route(backend_id, mode="ticketed_prep", owner_ticket_present=True)
    assert ticketed.allowed is True

    would_live = decide_federation_route(backend_id, mode="would_live", owner_ticket_present=True)
    assert would_live.allowed is False
    assert any(
        "would_live" in b.lower() or "auto-submit" in b.lower() for b in would_live.blockers
    )


def test_dry_run_probe() -> None:
    probe = materialise_demo_federation_dry_run_probe()
    assert probe.no_submit is True
    assert probe.invent_green_live_submit is False
    assert probe.status in {"ready", "blocked", "unknown"}
    assert probe.backend_id in list_hal_backend_ids()
    payload = probe.to_dict()
    assert payload["invent_green_live_submit"] is False

    backend_id = list_hal_backend_ids()[0]
    row = get_hal_capability(backend_id)
    required = row.ir_formats[0]
    with_ir = materialise_federation_dry_run_probe(
        backend_id, required_ir_format=required, min_qubits=1
    )
    assert with_ir.no_submit is True
    missing_ir = materialise_federation_dry_run_probe(
        backend_id, required_ir_format="__no_such_ir__"
    )
    assert missing_ir.status == "blocked"
    assert any("IR" in b or "ir" in b.lower() for b in missing_ir.blockers)


def test_public_surfaces_and_registry() -> None:
    surfaces = map_multi_hal_federation_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.multi_hal_federation_product" in paths
    assert "scpn_quantum_control.hardware.backends" in paths
    assert "scpn_quantum_control.hardware.provider_capability_core" in paths

    registry = build_multi_hal_federation_product_registry()
    assert registry["schema"] == MULTI_HAL_FEDERATION_PRODUCT_SCHEMA
    assert registry["no_submit_default_policy"] is True
    assert registry["invent_green_live_submit_policy"] is False
    validated = assert_multi_hal_federation_product_integrity(registry)
    assert validated["backend_count"] == len(list_hal_backend_ids())
    assert validated["provider_count"] == len(list_hal_providers())
    assert assert_multi_hal_federation_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_policy() -> None:
    registry = build_multi_hal_federation_product_registry()
    matrix = cast(list[dict[str, object]], list(registry["federation_matrix"]))

    broken = dict(registry)
    broken["federation_matrix"] = matrix + [
        {
            "backend_id": "ghost",
            "provider": "p",
            "broker": "b",
            "adapter_module": "m",
            "modality": "x",
            "supports_shots": True,
            "supports_mid_circuit_measurement": False,
            "supports_pulse": False,
            "supports_statevector": False,
            "submit_requires_approval": True,
            "can_submit": False,
            "is_cloud": True,
            "ir_formats": ["openqasm3"],
            "max_qubits": None,
            "no_submit_default": True,
            "support_posture": "metadata_only",
            "as_of": "2026-07-24",
            "claim_boundary": MULTI_HAL_FEDERATION_CLAIM_BOUNDARY,
        }
    ]
    broken["backend_count"] = len(cast(list[object], broken["federation_matrix"]))
    with pytest.raises(ValueError, match="drift"):
        assert_multi_hal_federation_product_integrity(broken)

    empty: dict[str, object] = {
        "federation_matrix": [],
        "blank_entry_count": 0,
        "backend_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty federation_matrix"):
        assert_multi_hal_federation_product_integrity(empty)

    policy = dict(registry)
    policy["no_submit_default_policy"] = False
    with pytest.raises(ValueError, match="no_submit_default_policy"):
        assert_multi_hal_federation_product_integrity(policy)

    invent = dict(registry)
    invent["invent_green_live_submit_policy"] = True
    with pytest.raises(ValueError, match="invent_green_live_submit_policy"):
        assert_multi_hal_federation_product_integrity(invent)


def test_integrity_rejects_blank_invalid() -> None:
    registry = build_multi_hal_federation_product_registry()
    matrix = cast(list[dict[str, object]], list(registry["federation_matrix"]))

    non_map = dict(registry)
    non_map["federation_matrix"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_multi_hal_federation_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in matrix]
    rows[0]["backend_id"] = "  "
    blank_id["federation_matrix"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_multi_hal_federation_product_integrity(blank_id)

    no_provider = dict(registry)
    brows = [dict(row) for row in matrix]
    brows[0]["provider"] = ""
    no_provider["federation_matrix"] = brows
    with pytest.raises(ValueError, match="provider"):
        assert_multi_hal_federation_product_integrity(no_provider)

    no_adapter = dict(registry)
    arows = [dict(row) for row in matrix]
    arows[0]["adapter_module"] = ""
    no_adapter["federation_matrix"] = arows
    with pytest.raises(ValueError, match="adapter_module"):
        assert_multi_hal_federation_product_integrity(no_adapter)

    submit = dict(registry)
    srows = [dict(row) for row in matrix]
    srows[0]["no_submit_default"] = False
    submit["federation_matrix"] = srows
    with pytest.raises(ValueError, match="no_submit_default"):
        assert_multi_hal_federation_product_integrity(submit)

    no_ir = dict(registry)
    irows = [dict(row) for row in matrix]
    irows[0]["ir_formats"] = []
    no_ir["federation_matrix"] = irows
    with pytest.raises(ValueError, match="ir_formats"):
        assert_multi_hal_federation_product_integrity(no_ir)

    dup = dict(registry)
    drows = [dict(row) for row in matrix]
    drows.append(dict(drows[0]))
    dup["federation_matrix"] = drows
    dup["backend_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate backend_id"):
        assert_multi_hal_federation_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_multi_hal_federation_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["backend_count"] = 0
    with pytest.raises(ValueError, match="backend_count"):
        assert_multi_hal_federation_product_integrity(count_mismatch)


def test_module_exports() -> None:
    assert "materialise_demo_federation_dry_run_probe" in multi_hal_federation_product.__all__
    assert "decide_federation_route" in multi_hal_federation_product.__all__
    assert "build_federation_matrix" in multi_hal_federation_product.__all__


def test_row_decision_probe_validation() -> None:
    base: dict[str, Any] = {
        "backend_id": "x",
        "provider": "p",
        "broker": "b",
        "adapter_module": "m",
        "modality": "gate",
        "supports_shots": True,
        "supports_mid_circuit_measurement": False,
        "supports_pulse": False,
        "supports_statevector": False,
        "submit_requires_approval": True,
        "can_submit": False,
        "is_cloud": True,
        "ir_formats": ("openqasm3",),
        "max_qubits": 2,
    }
    assert HalCapabilityRecord(**base).backend_id == "x"
    assert HalCapabilityRecord(**base).to_dict()["backend_id"] == "x"
    with pytest.raises(ValueError, match="backend_id"):
        HalCapabilityRecord(**{**base, "backend_id": ""})
    with pytest.raises(ValueError, match="provider"):
        HalCapabilityRecord(**{**base, "provider": ""})
    with pytest.raises(ValueError, match="broker"):
        HalCapabilityRecord(**{**base, "broker": ""})
    with pytest.raises(ValueError, match="adapter_module"):
        HalCapabilityRecord(**{**base, "adapter_module": ""})
    with pytest.raises(ValueError, match="modality"):
        HalCapabilityRecord(**{**base, "modality": ""})
    with pytest.raises(ValueError, match="ir_formats"):
        HalCapabilityRecord(**{**base, "ir_formats": ()})
    with pytest.raises(ValueError, match="ir_formats entries"):
        HalCapabilityRecord(**{**base, "ir_formats": ("ok", "  ")})
    with pytest.raises(ValueError, match="max_qubits"):
        HalCapabilityRecord(**{**base, "max_qubits": 0})
    with pytest.raises(ValueError, match="no_submit_default"):
        HalCapabilityRecord(**{**base, "no_submit_default": False})
    with pytest.raises(ValueError, match="support_posture"):
        HalCapabilityRecord(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="as_of"):
        HalCapabilityRecord(**{**base, "as_of": ""})

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
        decide_federation_route(list_hal_backend_ids()[0], mode="dry_run").to_dict()["allowed"]
        is True
    )

    with pytest.raises(ValueError, match="backend_id"):
        MaterialisedFederationDryRunProbe(
            backend_id="",
            provider="p",
            status="blocked",
            no_submit=True,
            invent_green_live_submit=False,
            blockers=("b",),
            warnings=(),
            demo_label="d",
        )
    with pytest.raises(ValueError, match="provider"):
        MaterialisedFederationDryRunProbe(
            backend_id="x",
            provider="",
            status="blocked",
            no_submit=True,
            invent_green_live_submit=False,
            blockers=("b",),
            warnings=(),
            demo_label="d",
        )
    with pytest.raises(ValueError, match="status"):
        MaterialisedFederationDryRunProbe(
            backend_id="x",
            provider="p",
            status="green",
            no_submit=True,
            invent_green_live_submit=False,
            blockers=("b",),
            warnings=(),
            demo_label="d",
        )
    with pytest.raises(ValueError, match="no_submit"):
        MaterialisedFederationDryRunProbe(
            backend_id="x",
            provider="p",
            status="blocked",
            no_submit=False,
            invent_green_live_submit=False,
            blockers=("b",),
            warnings=(),
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_live_submit"):
        MaterialisedFederationDryRunProbe(
            backend_id="x",
            provider="p",
            status="blocked",
            no_submit=True,
            invent_green_live_submit=True,
            blockers=("b",),
            warnings=(),
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedFederationDryRunProbe(
            backend_id="x",
            provider="p",
            status="blocked",
            no_submit=True,
            invent_green_live_submit=False,
            blockers=("b",),
            warnings=(),
            demo_label="",
        )
    with pytest.raises(ValueError, match="blockers entries"):
        MaterialisedFederationDryRunProbe(
            backend_id="x",
            provider="p",
            status="blocked",
            no_submit=True,
            invent_green_live_submit=False,
            blockers=("ok", "  "),
            warnings=(),
            demo_label="d",
        )
    with pytest.raises(ValueError, match="warnings entries"):
        MaterialisedFederationDryRunProbe(
            backend_id="x",
            provider="p",
            status="blocked",
            no_submit=True,
            invent_green_live_submit=False,
            blockers=("b",),
            warnings=("ok", "  "),
            demo_label="d",
        )


def test_catalogue_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive catalogue RuntimeError paths."""
    monkeypatch.setattr(multi_hal_federation_product, "_CANONICAL_HALS", ())
    with pytest.raises(RuntimeError, match="federation catalogue must be non-empty"):
        multi_hal_federation_product._catalogue_map()

    blank = HalCapabilityRecord(
        backend_id="tmp",
        provider="p",
        broker="b",
        adapter_module="m",
        modality="gate",
        supports_shots=True,
        supports_mid_circuit_measurement=False,
        supports_pulse=False,
        supports_statevector=False,
        submit_requires_approval=True,
        can_submit=False,
        is_cloud=True,
        ir_formats=("openqasm3",),
        max_qubits=None,
    )
    object.__setattr__(blank, "backend_id", "  ")
    monkeypatch.setattr(multi_hal_federation_product, "_CANONICAL_HALS", (blank,))
    with pytest.raises(RuntimeError, match="blank backend_id"):
        multi_hal_federation_product._catalogue_map()

    good = HalCapabilityRecord(
        backend_id="dup",
        provider="p",
        broker="b",
        adapter_module="m",
        modality="gate",
        supports_shots=True,
        supports_mid_circuit_measurement=False,
        supports_pulse=False,
        supports_statevector=False,
        submit_requires_approval=True,
        can_submit=False,
        is_cloud=True,
        ir_formats=("openqasm3",),
        max_qubits=None,
    )
    monkeypatch.setattr(multi_hal_federation_product, "_CANONICAL_HALS", (good, good))
    with pytest.raises(RuntimeError, match="duplicate backend_id"):
        multi_hal_federation_product._catalogue_map()
