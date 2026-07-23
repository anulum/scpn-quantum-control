# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for hardware-safe execution (BL-47)
"""Real-surface tests for ``scpn_quantum_control.hardware_safe_execution``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.hardware_safe_execution as hardware_safe_execution
from scpn_quantum_control.hardware_safe_execution import (
    HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY,
    HARDWARE_SAFE_EXECUTION_SCHEMA,
    AuditRecord,
    DryRunPlan,
    EnforceDecision,
    ExecutionPolicy,
    assert_hardware_safe_execution_integrity,
    build_audit_record,
    build_hardware_safe_execution_registry,
    default_execution_policy,
    dry_run_execution_plan,
    enforce_execution_request,
    get_execution_policy,
    iter_execution_policies,
    list_execution_policy_ids,
)


def test_list_and_default_no_submit() -> None:
    ids = list_execution_policy_ids()
    assert "default_no_submit" in ids
    assert ids == list_execution_policy_ids()
    default = default_execution_policy()
    assert default.policy_id == "default_no_submit"
    assert default.no_submit is True
    assert default.owner_allow_submit is False
    assert default.claim_boundary == HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY
    assert all(row.no_submit is True for row in iter_execution_policies(no_submit=True))


def test_get_known_and_unknown_fail_closed() -> None:
    row = get_execution_policy("ci_dry_run_only")
    assert row.policy_id == "ci_dry_run_only"
    assert row.max_total_shots > 0
    with pytest.raises(ValueError, match="non-empty"):
        get_execution_policy("  ")
    with pytest.raises(ValueError, match="unknown execution policy_id"):
        get_execution_policy("not_a_policy")


def test_build_registry_and_integrity() -> None:
    registry = build_hardware_safe_execution_registry()
    assert registry["schema"] == HARDWARE_SAFE_EXECUTION_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_policy_id"] == "default_no_submit"
    count = registry["policy_count"]
    assert isinstance(count, int)
    assert count == len(list_execution_policy_ids())
    validated = assert_hardware_safe_execution_integrity(registry)
    assert validated["policy_count"] == count
    assert assert_hardware_safe_execution_integrity()["blank_entry_count"] == 0


def test_dry_run_allowed_and_over_budget() -> None:
    ok = dry_run_execution_plan(
        "default_no_submit",
        n_params=2,
        shots_per_evaluation=128,
        shift_terms=1,
        would_submit=False,
    )
    assert ok.outcome == "allowed_plan"
    assert ok.blockers == ()
    assert ok.estimated_total_shots == 2 * 2 * 1 * 128  # evaluations * shots
    assert ok.would_submit is False

    over = dry_run_execution_plan(
        "ci_dry_run_only",
        n_params=8,
        shots_per_evaluation=256,
        shift_terms=4,
        would_submit=False,
    )
    assert over.outcome == "refused"
    assert over.blockers
    assert any("exceeds" in item for item in over.blockers)


def test_dry_run_refuses_would_submit_default() -> None:
    refused = dry_run_execution_plan(
        "default_no_submit",
        n_params=1,
        shots_per_evaluation=64,
        would_submit=True,
    )
    assert refused.outcome == "refused"
    assert any("no_submit" in item for item in refused.blockers)


def test_enforce_dry_run_and_would_submit() -> None:
    dry = enforce_execution_request(
        "default_no_submit",
        mode="dry_run",
        n_params=2,
        shots_per_evaluation=64,
    )
    assert dry.allowed is True
    assert dry.outcome == "allowed_plan"
    assert dry.audit_id
    assert dry.estimated_total_shots > 0

    submit = enforce_execution_request(
        "default_no_submit",
        mode="would_submit",
        n_params=1,
        shots_per_evaluation=64,
    )
    assert submit.allowed is False
    assert submit.outcome == "refused"
    assert submit.blockers
    assert "would_submit" in submit.reason or any(
        "would_submit" in item or "no_submit" in item for item in submit.blockers
    )


def test_enforce_ticketed_prep_requires_owner_and_ticket() -> None:
    missing = enforce_execution_request(
        "owner_ticketed_prep",
        mode="ticketed_prep",
        n_params=1,
        shots_per_evaluation=64,
        live_execution_ticket="",
    )
    assert missing.allowed is False
    assert any("ticket" in item for item in missing.blockers)

    ok = enforce_execution_request(
        "owner_ticketed_prep",
        mode="ticketed_prep",
        n_params=1,
        shots_per_evaluation=64,
        live_execution_ticket="ticket-demo-001",
    )
    assert ok.allowed is True
    assert ok.outcome == "allowed_plan"

    no_owner = enforce_execution_request(
        "default_no_submit",
        mode="ticketed_prep",
        n_params=1,
        shots_per_evaluation=64,
        live_execution_ticket="ticket-demo-001",
    )
    assert no_owner.allowed is False


def test_audit_record_secret_free() -> None:
    decision = enforce_execution_request(
        "default_no_submit",
        mode="dry_run",
        n_params=1,
        shots_per_evaluation=32,
    )
    audit = build_audit_record(decision)
    payload = audit.to_dict()
    assert payload["contains_secrets"] is False
    assert payload["audit_id"] == decision.audit_id
    assert payload["policy_id"] == decision.policy_id


def test_module_exports() -> None:
    assert "dry_run_execution_plan" in hardware_safe_execution.__all__
    assert "enforce_execution_request" in hardware_safe_execution.__all__
    assert "default_execution_policy" in hardware_safe_execution.__all__


def test_invalid_dimensions_and_mode() -> None:
    with pytest.raises(ValueError, match="n_params"):
        dry_run_execution_plan("default_no_submit", n_params=0)
    with pytest.raises(ValueError, match="shift_terms"):
        dry_run_execution_plan("default_no_submit", n_params=1, shift_terms=0)
    with pytest.raises(ValueError, match="shots_per_evaluation"):
        dry_run_execution_plan(
            "default_no_submit",
            n_params=1,
            shots_per_evaluation=0,
        )
    with pytest.raises(ValueError, match="unknown mode"):
        enforce_execution_request(
            "default_no_submit",
            mode=cast(Any, "explode"),
            n_params=1,
        )


def test_policy_record_validation() -> None:
    base: dict[str, Any] = {
        "policy_id": "x",
        "summary": "s",
        "no_submit": True,
        "owner_allow_submit": False,
        "max_shots_per_evaluation": 10,
        "max_total_shots": 100,
        "max_params": 4,
        "max_shift_terms": 2,
        "cost_model_status": "unavailable",
    }
    assert ExecutionPolicy(**base).policy_id == "x"
    with pytest.raises(ValueError, match="policy_id"):
        ExecutionPolicy(**{**base, "policy_id": ""})
    with pytest.raises(ValueError, match="summary"):
        ExecutionPolicy(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="cost_model_status"):
        ExecutionPolicy(**{**base, "cost_model_status": cast(Any, "nope")})
    with pytest.raises(ValueError, match="max_shots_per_evaluation"):
        ExecutionPolicy(**{**base, "max_shots_per_evaluation": 0})
    with pytest.raises(ValueError, match="max_total_shots must be positive"):
        ExecutionPolicy(**{**base, "max_total_shots": 0})
    with pytest.raises(ValueError, match="max_params"):
        ExecutionPolicy(**{**base, "max_params": 0})
    with pytest.raises(ValueError, match="max_shift_terms"):
        ExecutionPolicy(**{**base, "max_shift_terms": 0})
    with pytest.raises(ValueError, match="max_total_shots must be >="):
        ExecutionPolicy(**{**base, "max_total_shots": 5})
    with pytest.raises(ValueError, match="cost_usd_per_shot must be non-negative"):
        ExecutionPolicy(
            **{
                **base,
                "cost_model_status": "rate_table",
                "cost_usd_per_shot": -1.0,
            }
        )
    with pytest.raises(ValueError, match="owner_allow_submit"):
        ExecutionPolicy(**{**base, "no_submit": False, "owner_allow_submit": False})
    with pytest.raises(ValueError, match="non-rate_table"):
        ExecutionPolicy(**{**base, "cost_usd_per_shot": 1.0})
    with pytest.raises(ValueError, match="as_of"):
        ExecutionPolicy(**{**base, "as_of": ""})


def test_dry_run_and_enforce_dataclass_invariants() -> None:
    with pytest.raises(ValueError, match="policy_id"):
        DryRunPlan(
            policy_id="",
            n_params=1,
            shift_terms=1,
            shots_per_evaluation=1,
            evaluations=2,
            estimated_total_shots=2,
            estimated_cost_usd=None,
            cost_model_status="unavailable",
            would_submit=False,
            outcome="allowed_plan",
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="reason"):
        DryRunPlan(
            policy_id="x",
            n_params=1,
            shift_terms=1,
            shots_per_evaluation=1,
            evaluations=2,
            estimated_total_shots=2,
            estimated_cost_usd=None,
            cost_model_status="unavailable",
            would_submit=False,
            outcome="allowed_plan",
            reason="",
            blockers=(),
        )
    with pytest.raises(ValueError, match="outcome"):
        DryRunPlan(
            policy_id="x",
            n_params=1,
            shift_terms=1,
            shots_per_evaluation=1,
            evaluations=2,
            estimated_total_shots=2,
            estimated_cost_usd=None,
            cost_model_status="unavailable",
            would_submit=False,
            outcome=cast(Any, "nope"),
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="blockers"):
        DryRunPlan(
            policy_id="x",
            n_params=1,
            shift_terms=1,
            shots_per_evaluation=1,
            evaluations=2,
            estimated_total_shots=2,
            estimated_cost_usd=None,
            cost_model_status="unavailable",
            would_submit=False,
            outcome="refused",
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="blockers entries must be non-empty"):
        DryRunPlan(
            policy_id="x",
            n_params=1,
            shift_terms=1,
            shots_per_evaluation=1,
            evaluations=2,
            estimated_total_shots=2,
            estimated_cost_usd=None,
            cost_model_status="unavailable",
            would_submit=False,
            outcome="refused",
            reason="r",
            blockers=(" ",),
        )
    with pytest.raises(ValueError, match="allowed_plan cannot list blockers"):
        DryRunPlan(
            policy_id="x",
            n_params=1,
            shift_terms=1,
            shots_per_evaluation=1,
            evaluations=2,
            estimated_total_shots=2,
            estimated_cost_usd=None,
            cost_model_status="unavailable",
            would_submit=False,
            outcome="allowed_plan",
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="dimensions must be positive"):
        DryRunPlan(
            policy_id="x",
            n_params=0,
            shift_terms=1,
            shots_per_evaluation=1,
            evaluations=2,
            estimated_total_shots=2,
            estimated_cost_usd=None,
            cost_model_status="unavailable",
            would_submit=False,
            outcome="allowed_plan",
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="evaluations and estimated_total_shots"):
        DryRunPlan(
            policy_id="x",
            n_params=1,
            shift_terms=1,
            shots_per_evaluation=1,
            evaluations=0,
            estimated_total_shots=2,
            estimated_cost_usd=None,
            cost_model_status="unavailable",
            would_submit=False,
            outcome="allowed_plan",
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="mode"):
        EnforceDecision(
            policy_id="x",
            mode=cast(Any, "nope"),
            allowed=False,
            outcome="refused",
            estimated_total_shots=0,
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="outcome"):
        EnforceDecision(
            policy_id="x",
            mode="dry_run",
            allowed=False,
            outcome=cast(Any, "nope"),
            estimated_total_shots=0,
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="reason"):
        EnforceDecision(
            policy_id="x",
            mode="dry_run",
            allowed=False,
            outcome="refused",
            estimated_total_shots=0,
            reason="",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        EnforceDecision(
            policy_id="x",
            mode="dry_run",
            allowed=True,
            outcome="allowed_plan",
            estimated_total_shots=1,
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="require blockers"):
        EnforceDecision(
            policy_id="x",
            mode="dry_run",
            allowed=False,
            outcome="refused",
            estimated_total_shots=0,
            reason="r",
            blockers=(),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="blockers entries must be non-empty"):
        EnforceDecision(
            policy_id="x",
            mode="dry_run",
            allowed=False,
            outcome="refused",
            estimated_total_shots=0,
            reason="r",
            blockers=(" ",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="must use outcome=allowed_plan"):
        EnforceDecision(
            policy_id="x",
            mode="dry_run",
            allowed=True,
            outcome="refused",
            estimated_total_shots=1,
            reason="r",
            blockers=(),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="cannot use outcome=allowed_plan"):
        EnforceDecision(
            policy_id="x",
            mode="dry_run",
            allowed=False,
            outcome="allowed_plan",
            estimated_total_shots=0,
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="estimated_total_shots must be non-negative"):
        EnforceDecision(
            policy_id="x",
            mode="dry_run",
            allowed=False,
            outcome="refused",
            estimated_total_shots=-1,
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    with pytest.raises(ValueError, match="audit_id"):
        EnforceDecision(
            policy_id="x",
            mode="dry_run",
            allowed=False,
            outcome="refused",
            estimated_total_shots=0,
            reason="r",
            blockers=("b",),
            audit_id="",
        )
    with pytest.raises(ValueError, match="audit_id"):
        AuditRecord(
            audit_id="",
            policy_id="x",
            mode="dry_run",
            outcome="refused",
            estimated_total_shots=0,
            reason="r",
        )
    with pytest.raises(ValueError, match="policy_id"):
        AuditRecord(
            audit_id="a",
            policy_id="",
            mode="dry_run",
            outcome="refused",
            estimated_total_shots=0,
            reason="r",
        )
    with pytest.raises(ValueError, match="reason"):
        AuditRecord(
            audit_id="a",
            policy_id="x",
            mode="dry_run",
            outcome="refused",
            estimated_total_shots=0,
            reason="",
        )
    with pytest.raises(ValueError, match="mode"):
        AuditRecord(
            audit_id="a",
            policy_id="x",
            mode=cast(Any, "nope"),
            outcome="refused",
            estimated_total_shots=0,
            reason="r",
        )
    with pytest.raises(ValueError, match="outcome"):
        AuditRecord(
            audit_id="a",
            policy_id="x",
            mode="dry_run",
            outcome=cast(Any, "nope"),
            estimated_total_shots=0,
            reason="r",
        )


def test_integrity_rejects_drift_and_invent_submit() -> None:
    good = build_hardware_safe_execution_registry()
    assert_hardware_safe_execution_integrity(good)

    bad_blank = dict(good)
    bad_blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_hardware_safe_execution_integrity(bad_blank)

    empty = dict(good)
    empty["policies"] = []
    with pytest.raises(ValueError, match="non-empty policies"):
        assert_hardware_safe_execution_integrity(empty)

    not_map = dict(good)
    not_map["policies"] = [123]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_hardware_safe_execution_integrity(not_map)

    raw = good["policies"]
    assert isinstance(raw, list)
    policies = [dict(cast(dict[str, object], row)) for row in raw]

    invent_default = dict(good)
    default_row = next(row for row in policies if row["policy_id"] == "default_no_submit")
    broken = dict(default_row)
    broken["no_submit"] = False
    invent_default["policies"] = [
        broken if row["policy_id"] == "default_no_submit" else row for row in policies
    ]
    with pytest.raises(ValueError, match="default_no_submit must have no_submit=True"):
        assert_hardware_safe_execution_integrity(invent_default)

    blank_id = dict(good)
    blank_row = dict(policies[0])
    blank_row["policy_id"] = ""
    blank_id["policies"] = [blank_row, *policies[1:]]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_hardware_safe_execution_integrity(blank_id)

    bad_shots = dict(good)
    bad = dict(policies[0])
    bad["max_shots_per_evaluation"] = 0
    bad_shots["policies"] = [
        bad if row["policy_id"] == bad["policy_id"] else row for row in policies
    ]
    with pytest.raises(ValueError, match="invalid max_shots"):
        assert_hardware_safe_execution_integrity(bad_shots)

    missing = dict(good)
    missing["policies"] = policies[1:]
    missing["policy_count"] = len(policies) - 1
    with pytest.raises(ValueError, match="drift|missing default"):
        assert_hardware_safe_execution_integrity(missing)

    bad_count = dict(good)
    bad_count["policy_count"] = 0
    with pytest.raises(ValueError, match="policy_count"):
        assert_hardware_safe_execution_integrity(bad_count)

    duplicate = dict(good)
    duplicate["policies"] = [policies[0], policies[0]]
    duplicate["policy_count"] = 2
    with pytest.raises(ValueError, match="duplicate"):
        assert_hardware_safe_execution_integrity(duplicate)


def test_to_dict_round_trip_fields() -> None:
    policy = get_execution_policy("owner_ticketed_prep")
    payload = policy.to_dict()
    assert payload["cost_model_status"] == "rate_table"
    plan = dry_run_execution_plan(
        "owner_ticketed_prep",
        n_params=1,
        shots_per_evaluation=10,
    )
    assert plan.to_dict()["outcome"] == "allowed_plan"
    assert plan.estimated_cost_usd == 0.0
    decision = enforce_execution_request(
        "ci_dry_run_only",
        mode="dry_run",
        n_params=1,
        shots_per_evaluation=16,
    )
    assert decision.to_dict()["allowed"] is True


def test_catalogue_map_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = hardware_safe_execution
    with pytest.raises(RuntimeError, match="non-empty"):
        monkeypatch.setattr(mod, "_CANONICAL_POLICIES", ())
        mod._catalogue_map()
    good = get_execution_policy("default_no_submit")
    blank = ExecutionPolicy(
        policy_id="tmp",
        summary="s",
        no_submit=True,
        owner_allow_submit=False,
        max_shots_per_evaluation=1,
        max_total_shots=1,
        max_params=1,
        max_shift_terms=1,
        cost_model_status="unavailable",
    )
    object.__setattr__(blank, "policy_id", "  ")
    with pytest.raises(RuntimeError, match="blank policy_id"):
        monkeypatch.setattr(mod, "_CANONICAL_POLICIES", (blank,))
        mod._catalogue_map()
    with pytest.raises(RuntimeError, match="duplicate"):
        monkeypatch.setattr(mod, "_CANONICAL_POLICIES", (good, good))
        mod._catalogue_map()


def test_dry_run_param_and_shift_limits() -> None:
    refused_params = dry_run_execution_plan(
        "ci_dry_run_only",
        n_params=32,
        shots_per_evaluation=16,
        shift_terms=1,
    )
    assert refused_params.outcome == "refused"
    assert any("max_params" in item for item in refused_params.blockers)

    refused_shift = dry_run_execution_plan(
        "ci_dry_run_only",
        n_params=1,
        shots_per_evaluation=16,
        shift_terms=8,
    )
    assert refused_shift.outcome == "refused"
    assert any("max_shift_terms" in item for item in refused_shift.blockers)

    refused_shots = dry_run_execution_plan(
        "ci_dry_run_only",
        n_params=1,
        shots_per_evaluation=10_000,
        shift_terms=1,
    )
    assert refused_shots.outcome == "refused"
    assert any("shots_per_evaluation" in item for item in refused_shots.blockers)


def test_ticketed_prep_empty_blockers_fallback() -> None:
    # owner policy within budget but empty ticket already covered; also
    # ticketed over-budget path
    over = enforce_execution_request(
        "owner_ticketed_prep",
        mode="ticketed_prep",
        n_params=32,
        shots_per_evaluation=4096,
        shift_terms=4,
        live_execution_ticket="t1",
    )
    assert over.allowed is False
    assert over.blockers


def test_integrity_invalid_no_submit_type() -> None:
    good = build_hardware_safe_execution_registry()
    raw = good["policies"]
    assert isinstance(raw, list)
    policies = [dict(cast(dict[str, object], row)) for row in raw]
    # non-default row with invalid no_submit type
    non_default = next(r for r in policies if r["policy_id"] != "default_no_submit")
    bad = dict(good)
    row = dict(non_default)
    row["no_submit"] = "yes"
    bad["policies"] = [row if r["policy_id"] == row["policy_id"] else r for r in policies]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_hardware_safe_execution_integrity(bad)

    # missing default entirely by renaming
    renamed = dict(good)
    rows = []
    for r in policies:
        item = dict(r)
        if item["policy_id"] == "default_no_submit":
            item["policy_id"] = "renamed_default"
        rows.append(item)
    renamed["policies"] = rows
    with pytest.raises(ValueError, match="missing default_no_submit|drift"):
        assert_hardware_safe_execution_integrity(renamed)


def test_more_policy_and_integrity_edges() -> None:
    # rate_table negative already covered; enforce policy_id blank path via enforce
    with pytest.raises(ValueError, match="policy_id"):
        EnforceDecision(
            policy_id="",
            mode="dry_run",
            allowed=False,
            outcome="refused",
            estimated_total_shots=0,
            reason="r",
            blockers=("b",),
            audit_id="a",
        )
    # dry-run default shots when None
    plan = dry_run_execution_plan("default_no_submit", n_params=1)
    assert (
        plan.shots_per_evaluation
        == get_execution_policy("default_no_submit").max_shots_per_evaluation
    )
    # ticketed_prep with requirements not met empty unique path: use owner policy
    # but force plan refused so unique_blockers non-empty
    refused = enforce_execution_request(
        "owner_ticketed_prep",
        mode="ticketed_prep",
        n_params=1,
        shots_per_evaluation=1_000_000,
        live_execution_ticket="t",
    )
    assert refused.allowed is False
