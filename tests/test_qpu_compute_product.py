# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for qpu_compute product (BL-95)
"""Real-surface tests for ``scpn_quantum_control.qpu_compute_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.qpu_compute_product as qpu_compute_product
from scpn_quantum_control.qpu_compute_product import (
    QPU_COMPUTE_PRODUCT_CLAIM_BOUNDARY,
    QPU_COMPUTE_PRODUCT_SCHEMA,
    ComputePlanDecision,
    ComputePlanKind,
    ComputePlanRecord,
    assert_qpu_compute_product_integrity,
    audit_compute_plan_decision,
    build_qpu_compute_product_registry,
    construct_compute_plan,
    dry_run_compute_plan,
    get_plan_kind,
    iter_plan_kinds,
    list_plan_kind_ids,
    list_supported_backend_policies,
    list_supported_kernels,
)
from scpn_quantum_control.qpu_compute_types import SUPPORTED_KERNELS


def test_list_kinds_and_kernels() -> None:
    ids = list_plan_kind_ids()
    assert "dry_run_simulator" in ids
    assert ids == list_plan_kind_ids()
    assert set(list_supported_kernels()) == set(SUPPORTED_KERNELS)
    assert "simulator_statevector" in list_supported_backend_policies()
    dry = iter_plan_kinds(mode="dry_run")
    assert dry
    assert all(row.mode == "dry_run" for row in dry)


def test_get_known_and_unknown_fail_closed() -> None:
    kind = get_plan_kind("dry_run_simulator")
    assert kind.no_submit is True
    assert kind.default_hardware_enabled is False
    assert kind.claim_boundary == QPU_COMPUTE_PRODUCT_CLAIM_BOUNDARY
    with pytest.raises(ValueError, match="non-empty"):
        get_plan_kind("  ")
    with pytest.raises(ValueError, match="unknown plan_kind_id"):
        get_plan_kind("not_a_plan")


def test_construct_and_dry_run_allowed() -> None:
    plan = construct_compute_plan("dry_run_simulator", kernel="sync_dla", shots=128)
    assert plan.hardware_enabled is False
    assert plan.backend_policy == "simulator_statevector"
    decision = dry_run_compute_plan("dry_run_simulator", kernel="sync_witness", shots=64)
    assert decision.allowed is True
    assert decision.outcome == "allowed_plan"
    assert decision.blockers == ()
    assert "no provider submission" in decision.reason
    assert decision.bl47_policy_id == "default_no_submit"


def test_refuse_would_live_and_hardware() -> None:
    live = dry_run_compute_plan("live_would_submit")
    assert live.allowed is False
    assert live.blockers
    assert any("would_live" in item or "hardware" in item for item in live.blockers)

    forced = dry_run_compute_plan(
        "dry_run_simulator",
        hardware_enabled=True,
    )
    assert forced.allowed is False


def test_ticketed_prep_requires_ticket() -> None:
    missing = dry_run_compute_plan("ticketed_prep_plan")
    assert missing.allowed is False
    assert any("ticket" in item for item in missing.blockers)

    ok = dry_run_compute_plan(
        "ticketed_prep_plan",
        live_execution_ticket="ticket-demo-001",
    )
    assert ok.allowed is True
    assert ok.bl47_policy_id == "owner_ticketed_prep"


def test_unsupported_backend_and_kernel() -> None:
    with pytest.raises(ValueError, match="kernel must be one of"):
        construct_compute_plan("dry_run_simulator", kernel="not_a_kernel")

    bad_backend = dry_run_compute_plan(
        "dry_run_simulator",
        backend_policy="ibm_qpu_live",
    )
    assert bad_backend.allowed is False
    assert any("backend_policy" in item for item in bad_backend.blockers)


def test_registry_and_integrity() -> None:
    registry = build_qpu_compute_product_registry()
    assert registry["schema"] == QPU_COMPUTE_PRODUCT_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_plan_kind_id"] == "dry_run_simulator"
    count = registry["plan_kind_count"]
    assert isinstance(count, int)
    assert count == len(list_plan_kind_ids())
    validated = assert_qpu_compute_product_integrity(registry)
    assert validated["plan_kind_count"] == count
    assert assert_qpu_compute_product_integrity()["blank_entry_count"] == 0


def test_audit_secret_free() -> None:
    decision = dry_run_compute_plan("dry_run_simulator")
    audit = audit_compute_plan_decision(decision)
    assert audit["contains_secrets"] is False
    assert audit["audit_id"] == decision.audit_id
    assert "bl47_audit" in audit


def test_module_exports() -> None:
    assert "dry_run_compute_plan" in qpu_compute_product.__all__
    assert "construct_compute_plan" in qpu_compute_product.__all__


def test_plan_kind_validation() -> None:
    base: dict[str, Any] = {
        "plan_kind_id": "x",
        "mode": "dry_run",
        "summary": "s",
        "default_backend_policy": "simulator_statevector",
        "default_hardware_enabled": False,
        "no_submit": True,
    }
    assert ComputePlanKind(**base).plan_kind_id == "x"
    with pytest.raises(ValueError, match="plan_kind_id"):
        ComputePlanKind(**{**base, "plan_kind_id": ""})
    with pytest.raises(ValueError, match="mode"):
        ComputePlanKind(**{**base, "mode": cast(Any, "nope")})
    with pytest.raises(ValueError, match="summary"):
        ComputePlanKind(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="default_backend_policy"):
        ComputePlanKind(**{**base, "default_backend_policy": ""})
    with pytest.raises(ValueError, match="default_hardware_enabled=False"):
        ComputePlanKind(**{**base, "default_hardware_enabled": True})
    with pytest.raises(ValueError, match="no_submit=True"):
        ComputePlanKind(**{**base, "no_submit": False})
    with pytest.raises(ValueError, match="as_of"):
        ComputePlanKind(**{**base, "as_of": ""})
    with pytest.raises(ValueError, match="no_submit=True"):
        ComputePlanKind(
            plan_kind_id="y",
            mode="would_live",
            summary="s",
            default_backend_policy="simulator_statevector",
            default_hardware_enabled=True,
            no_submit=False,
        )


def test_record_and_decision_invariants() -> None:
    with pytest.raises(ValueError, match="plan_kind_id"):
        ComputePlanRecord(
            plan_kind_id="",
            mode="dry_run",
            kernel="sync_dla",
            backend_policy="simulator_statevector",
            shots=1,
            hardware_enabled=False,
            live_execution_ticket="",
            no_submit=True,
        )
    with pytest.raises(ValueError, match="mode"):
        ComputePlanRecord(
            plan_kind_id="x",
            mode=cast(Any, "nope"),
            kernel="sync_dla",
            backend_policy="simulator_statevector",
            shots=1,
            hardware_enabled=False,
            live_execution_ticket="",
            no_submit=True,
        )
    with pytest.raises(ValueError, match="backend_policy"):
        ComputePlanRecord(
            plan_kind_id="x",
            mode="dry_run",
            kernel="sync_dla",
            backend_policy="",
            shots=1,
            hardware_enabled=False,
            live_execution_ticket="",
            no_submit=True,
        )
    with pytest.raises(ValueError, match="shots"):
        ComputePlanRecord(
            plan_kind_id="x",
            mode="dry_run",
            kernel="sync_dla",
            backend_policy="simulator_statevector",
            shots=0,
            hardware_enabled=False,
            live_execution_ticket="",
            no_submit=True,
        )
    with pytest.raises(ValueError, match="kernel"):
        ComputePlanRecord(
            plan_kind_id="x",
            mode="dry_run",
            kernel="bad",
            backend_policy="simulator_statevector",
            shots=1,
            hardware_enabled=False,
            live_execution_ticket="",
            no_submit=True,
        )
    with pytest.raises(ValueError, match="whitespace"):
        ComputePlanRecord(
            plan_kind_id="x",
            mode="dry_run",
            kernel="sync_dla",
            backend_policy="simulator_statevector",
            shots=1,
            hardware_enabled=False,
            live_execution_ticket="  ticket  ",
            no_submit=True,
        )
    with pytest.raises(ValueError, match="plan_kind_id"):
        ComputePlanDecision(
            plan_kind_id="",
            outcome="refused",
            allowed=False,
            mode="dry_run",
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="outcome"):
        ComputePlanDecision(
            plan_kind_id="x",
            outcome=cast(Any, "nope"),
            allowed=False,
            mode="dry_run",
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="mode"):
        ComputePlanDecision(
            plan_kind_id="x",
            outcome="refused",
            allowed=False,
            mode=cast(Any, "nope"),
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="reason"):
        ComputePlanDecision(
            plan_kind_id="x",
            outcome="refused",
            allowed=False,
            mode="dry_run",
            reason="",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="must use outcome=allowed_plan"):
        ComputePlanDecision(
            plan_kind_id="x",
            outcome="refused",
            allowed=True,
            mode="dry_run",
            reason="r",
            blockers=(),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="must use outcome=refused"):
        ComputePlanDecision(
            plan_kind_id="x",
            outcome="allowed_plan",
            allowed=False,
            mode="dry_run",
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="require blockers"):
        ComputePlanDecision(
            plan_kind_id="x",
            outcome="refused",
            allowed=False,
            mode="dry_run",
            reason="r",
            blockers=(),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        ComputePlanDecision(
            plan_kind_id="x",
            outcome="allowed_plan",
            allowed=True,
            mode="dry_run",
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="blockers entries"):
        ComputePlanDecision(
            plan_kind_id="x",
            outcome="refused",
            allowed=False,
            mode="dry_run",
            reason="r",
            blockers=(" ",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="audit_id"):
        ComputePlanDecision(
            plan_kind_id="x",
            outcome="refused",
            allowed=False,
            mode="dry_run",
            reason="r",
            blockers=("b",),
            audit_id="",
        )


def test_to_dict_paths() -> None:
    kind = get_plan_kind("dry_run_simulator")
    assert kind.to_dict()["no_submit"] is True
    plan = construct_compute_plan("dry_run_simulator")
    assert plan.to_dict()["kernel"] == "sync_dla"
    decision = dry_run_compute_plan("dry_run_simulator")
    assert decision.to_dict()["allowed"] is True


def test_integrity_rejects_drift() -> None:
    good = build_qpu_compute_product_registry()
    assert_qpu_compute_product_integrity(good)

    bad_blank = dict(good)
    bad_blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_qpu_compute_product_integrity(bad_blank)

    empty = dict(good)
    empty["plan_kinds"] = []
    with pytest.raises(ValueError, match="non-empty plan_kinds"):
        assert_qpu_compute_product_integrity(empty)

    not_map = dict(good)
    not_map["plan_kinds"] = [123]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_qpu_compute_product_integrity(not_map)

    raw = good["plan_kinds"]
    assert isinstance(raw, list)
    kinds = [dict(cast(dict[str, object], row)) for row in raw]

    invent = dict(good)
    default_row = next(r for r in kinds if r["plan_kind_id"] == "dry_run_simulator")
    broken = dict(default_row)
    broken["no_submit"] = False
    invent["plan_kinds"] = [
        broken if r["plan_kind_id"] == "dry_run_simulator" else r for r in kinds
    ]
    with pytest.raises(ValueError, match="no_submit=True"):
        assert_qpu_compute_product_integrity(invent)

    hw = dict(good)
    broken_hw = dict(default_row)
    broken_hw["default_hardware_enabled"] = True
    hw["plan_kinds"] = [
        broken_hw if r["plan_kind_id"] == "dry_run_simulator" else r for r in kinds
    ]
    with pytest.raises(ValueError, match="default_hardware_enabled=False"):
        assert_qpu_compute_product_integrity(hw)

    blank_id = dict(good)
    blank_row = dict(kinds[0])
    blank_row["plan_kind_id"] = ""
    blank_id["plan_kinds"] = [blank_row, *kinds[1:]]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_qpu_compute_product_integrity(blank_id)

    missing = dict(good)
    missing_rows = [r for r in kinds if r["plan_kind_id"] != "dry_run_simulator"]
    missing["plan_kinds"] = missing_rows
    missing["plan_kind_count"] = len(missing_rows)
    with pytest.raises(ValueError, match="missing dry_run_simulator|drift"):
        assert_qpu_compute_product_integrity(missing)

    bad_count = dict(good)
    bad_count["plan_kind_count"] = 0
    with pytest.raises(ValueError, match="plan_kind_count"):
        assert_qpu_compute_product_integrity(bad_count)

    duplicate = dict(good)
    duplicate["plan_kinds"] = [kinds[0], kinds[0]]
    duplicate["plan_kind_count"] = 2
    with pytest.raises(ValueError, match="duplicate"):
        assert_qpu_compute_product_integrity(duplicate)

    invent_live = dict(good)
    other = next(r for r in kinds if r["plan_kind_id"] != "dry_run_simulator")
    live_broken = dict(other)
    live_broken["no_submit"] = False
    invent_live["plan_kinds"] = [
        live_broken if r["plan_kind_id"] == other["plan_kind_id"] else r for r in kinds
    ]
    with pytest.raises(ValueError, match="no_submit=True"):
        assert_qpu_compute_product_integrity(invent_live)


def test_catalogue_map_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = qpu_compute_product
    with pytest.raises(RuntimeError, match="non-empty"):
        monkeypatch.setattr(mod, "_CANONICAL_KINDS", ())
        mod._catalogue_map()
    good = get_plan_kind("dry_run_simulator")
    blank = ComputePlanKind(
        plan_kind_id="tmp",
        mode="dry_run",
        summary="s",
        default_backend_policy="simulator_statevector",
        default_hardware_enabled=False,
        no_submit=True,
    )
    object.__setattr__(blank, "plan_kind_id", "  ")
    with pytest.raises(RuntimeError, match="blank plan_kind_id"):
        monkeypatch.setattr(mod, "_CANONICAL_KINDS", (blank,))
        mod._catalogue_map()
    with pytest.raises(RuntimeError, match="duplicate"):
        monkeypatch.setattr(mod, "_CANONICAL_KINDS", (good, good))
        mod._catalogue_map()


def test_integrity_invalid_mode_and_no_submit_type() -> None:
    good = build_qpu_compute_product_registry()
    raw = good["plan_kinds"]
    assert isinstance(raw, list)
    kinds = [dict(cast(dict[str, object], row)) for row in raw]
    other = next(r for r in kinds if r["plan_kind_id"] != "dry_run_simulator")
    bad_mode = dict(good)
    row = dict(other)
    row["mode"] = "nope"
    bad_mode["plan_kinds"] = [
        row if r["plan_kind_id"] == other["plan_kind_id"] else r for r in kinds
    ]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_qpu_compute_product_integrity(bad_mode)

    bad_ns = dict(good)
    row2 = dict(other)
    row2["no_submit"] = "yes"
    bad_ns["plan_kinds"] = [
        row2 if r["plan_kind_id"] == other["plan_kind_id"] else r for r in kinds
    ]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_qpu_compute_product_integrity(bad_ns)


def test_audit_would_live_and_ticketed() -> None:
    live = dry_run_compute_plan("live_would_submit")
    audit_live = audit_compute_plan_decision(live)
    assert audit_live["contains_secrets"] is False
    assert audit_live["allowed"] is False

    ticketed = dry_run_compute_plan(
        "ticketed_prep_plan",
        live_execution_ticket="t-1",
    )
    audit_t = audit_compute_plan_decision(ticketed)
    assert "bl47_audit" in audit_t
