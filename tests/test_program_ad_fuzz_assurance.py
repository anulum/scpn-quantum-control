# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for Program AD fuzz assurance
"""Real-surface tests for ``scpn_quantum_control.program_ad_fuzz_assurance``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.program_ad_fuzz_assurance as program_ad_fuzz_assurance
from scpn_quantum_control.program_ad_fuzz_assurance import (
    DEFAULT_TIME_BOX_SECONDS,
    MAX_TIME_BOX_SECONDS,
    PROGRAM_AD_FUZZ_ASSURANCE_SCHEMA,
    PROGRAM_AD_FUZZ_CLAIM_BOUNDARY,
    FuzzPolicy,
    FuzzProbeDecision,
    FuzzTarget,
    assert_fuzz_assurance_integrity,
    build_fuzz_assurance_registry,
    corpus_governance_policy,
    crash_pipeline_policy,
    dry_run_fuzz_target,
    fuzz_assurance_policy,
    get_fuzz_target,
    iter_fuzz_targets,
    list_fuzz_target_ids,
    map_fuzz_public_surfaces,
    validate_time_box_seconds,
)


def test_list_targets_and_filters() -> None:
    """Keep target ordering deterministic and filters responsibility-scoped."""
    ids = list_fuzz_target_ids()
    assert "program_ad_ir" in ids
    assert "knm_validators" in ids
    assert "ml_dsa_ntt" in ids
    assert "studio_kuramoto_input" in ids
    assert ids == list_fuzz_target_ids()
    ci = iter_fuzz_targets(posture="ci_optional")
    assert ci
    assert all(row.posture == "ci_optional" for row in ci)
    ir = iter_fuzz_targets(kind="program_ad_ir")
    assert len(ir) == 1


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve declared targets while refusing blank and unknown identifiers."""
    target = get_fuzz_target("program_ad_ir")
    assert target.package == "scpn-quantum-engine-fuzz"
    assert "program_ad_ir.rs" in target.rust_path
    assert target.claim_boundary == PROGRAM_AD_FUZZ_CLAIM_BOUNDARY
    with pytest.raises(ValueError, match="non-empty"):
        get_fuzz_target("  ")
    with pytest.raises(ValueError, match="unknown target_id"):
        get_fuzz_target("not_a_target")


def test_policy_defaults() -> None:
    """Expose bounded, optional, and invent-green-forbidden policy defaults."""
    policy = fuzz_assurance_policy()
    assert policy.continuous_fuzz_default is False
    assert policy.invent_green_forbidden is True
    assert policy.ci_optional is True
    assert policy.default_time_box_seconds == DEFAULT_TIME_BOX_SECONDS
    assert policy.max_time_box_seconds == MAX_TIME_BOX_SECONDS
    assert policy.to_dict()["policy_id"] == "time_boxed_ci_optional_v1"


def test_validate_time_box() -> None:
    """Accept bounded integer durations and refuse invalid or continuous spans."""
    assert validate_time_box_seconds(60) == 60
    assert validate_time_box_seconds(MAX_TIME_BOX_SECONDS) == MAX_TIME_BOX_SECONDS
    with pytest.raises(ValueError, match="positive"):
        validate_time_box_seconds(0)
    with pytest.raises(ValueError, match="must be an int"):
        validate_time_box_seconds(cast(Any, 1.5))
    with pytest.raises(ValueError, match="exceeds max|continuous"):
        validate_time_box_seconds(MAX_TIME_BOX_SECONDS + 1)


def test_dry_run_allowed() -> None:
    """Plan default and custom bounded dry-runs without executing cargo-fuzz."""
    decision = dry_run_fuzz_target("program_ad_ir")
    assert decision.allowed is True
    assert decision.outcome == "allowed_dry_run"
    assert decision.time_box_seconds == DEFAULT_TIME_BOX_SECONDS
    assert "cargo-fuzz was not executed" in decision.reason
    custom = dry_run_fuzz_target("knm_validators", time_box_seconds=120)
    assert custom.allowed is True
    assert custom.time_box_seconds == 120


def test_dry_run_refuses_continuous_and_invent_green() -> None:
    """Refuse continuous and invent-green requests with typed blockers."""
    refused = dry_run_fuzz_target("program_ad_ir", request_continuous=True)
    assert refused.allowed is False
    assert refused.outcome == "refused"
    assert any("continuous" in item.lower() for item in refused.blockers)

    invent = dry_run_fuzz_target(
        "program_ad_ir",
        request_invent_green_coverage=True,
    )
    assert invent.allowed is False
    assert any("invent-green" in item.lower() for item in invent.blockers)

    both = dry_run_fuzz_target(
        "ml_dsa_ntt",
        request_continuous=True,
        request_invent_green_coverage=True,
    )
    assert both.allowed is False
    assert len(both.blockers) >= 2


def test_dry_run_invalid_time_box_raises() -> None:
    """Reject an over-bound time box on an otherwise allowed dry-run."""
    with pytest.raises(ValueError, match="exceeds max|continuous"):
        dry_run_fuzz_target("program_ad_ir", time_box_seconds=MAX_TIME_BOX_SECONDS + 5)


def test_corpus_and_crash_policies() -> None:
    """Report unimplemented corpus and crash-automation capabilities."""
    corpus = corpus_governance_policy()
    assert corpus["retention_ops_implemented"] is False
    assert corpus["open_capability"] == "multi_day_corpus_retention"
    assert "residual_slice" not in corpus
    crash = crash_pipeline_policy()
    assert crash["automated_pipeline_implemented"] is False
    assert crash["open_capability"] == "automated_crash_to_regression_conversion"
    assert "residual_slice" not in crash


def test_public_surfaces_and_registry() -> None:
    """Map product surfaces and validate explicit and default registries."""
    surfaces = map_fuzz_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.program_ad_fuzz_assurance" in paths
    assert "scpn_quantum_engine.fuzz" in paths

    registry = build_fuzz_assurance_registry()
    assert PROGRAM_AD_FUZZ_ASSURANCE_SCHEMA == "program_ad_fuzz_assurance.v2"
    assert registry["schema"] == PROGRAM_AD_FUZZ_ASSURANCE_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_target_id"] == "program_ad_ir"
    validated = assert_fuzz_assurance_integrity(registry)
    assert validated["target_count"] == len(list_fuzz_target_ids())
    assert assert_fuzz_assurance_integrity()["blank_entry_count"] == 0
    policy = cast(dict[str, object], validated["policy"])
    assert policy["continuous_fuzz_default"] is False


def test_integrity_rejects_drift_and_invent_green_policy() -> None:
    """Reject target drift, empty inventories, and unsafe policy mutations."""
    registry = build_fuzz_assurance_registry()
    targets = cast(list[dict[str, object]], registry["targets"])

    broken = dict(registry)
    broken["targets"] = targets + [
        {
            "target_id": "ghost",
            "title": "t",
            "summary": "s",
            "kind": "program_ad_ir",
            "rust_path": "x.rs",
            "package": "p",
            "posture": "time_boxed_local",
            "parity_certificate_pointer": "a",
            "api_stability_class": "experimental_workbench",
            "as_of": "2026-07-24",
            "claim_boundary": PROGRAM_AD_FUZZ_CLAIM_BOUNDARY,
        }
    ]
    broken["target_count"] = len(cast(list[object], broken["targets"]))
    with pytest.raises(ValueError, match="drift"):
        assert_fuzz_assurance_integrity(broken)

    empty: dict[str, object] = {"targets": [], "blank_entry_count": 0, "target_count": 0}
    with pytest.raises(ValueError, match="non-empty targets"):
        assert_fuzz_assurance_integrity(empty)

    bad_policy = dict(registry)
    bad_policy["policy"] = {
        **cast(dict[str, object], registry["policy"]),
        "continuous_fuzz_default": True,
    }
    with pytest.raises(ValueError, match="continuous_fuzz_default"):
        assert_fuzz_assurance_integrity(bad_policy)

    bad_invent = dict(registry)
    bad_invent["policy"] = {
        **cast(dict[str, object], registry["policy"]),
        "invent_green_forbidden": False,
    }
    with pytest.raises(ValueError, match="invent_green_forbidden"):
        assert_fuzz_assurance_integrity(bad_invent)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject malformed rows, missing paths/defaults, duplicates, and counts."""
    registry = build_fuzz_assurance_registry()
    targets = cast(list[dict[str, object]], registry["targets"])

    non_map = dict(registry)
    non_map["targets"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_fuzz_assurance_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in targets]
    rows[0]["target_id"] = "  "
    blank_id["targets"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_fuzz_assurance_integrity(blank_id)

    bad_posture = dict(registry)
    prows = [dict(row) for row in targets]
    prows[1]["posture"] = "nope"
    bad_posture["targets"] = prows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_fuzz_assurance_integrity(bad_posture)

    no_path = dict(registry)
    path_rows = [dict(row) for row in targets]
    path_rows[0]["rust_path"] = ""
    no_path["targets"] = path_rows
    with pytest.raises(ValueError, match="rust_path"):
        assert_fuzz_assurance_integrity(no_path)

    no_default = dict(registry)
    renamed = [dict(row) for row in targets]
    for row in renamed:
        if row.get("target_id") == "program_ad_ir":
            row["target_id"] = "renamed"
    no_default["targets"] = renamed
    with pytest.raises(ValueError, match="missing program_ad_ir|drift"):
        assert_fuzz_assurance_integrity(no_default)

    dup = dict(registry)
    drows = [dict(row) for row in targets]
    drows.append(dict(drows[0]))
    dup["targets"] = drows
    dup["target_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate target_id"):
        assert_fuzz_assurance_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_fuzz_assurance_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["target_count"] = 0
    with pytest.raises(ValueError, match="target_count"):
        assert_fuzz_assurance_integrity(count_mismatch)

    no_policy = dict(registry)
    no_policy["policy"] = "nope"
    with pytest.raises(ValueError, match="policy must be a mapping"):
        assert_fuzz_assurance_integrity(no_policy)


def test_module_exports() -> None:
    """Keep the documented catalogue, policy, and dry-run APIs exported."""
    assert "dry_run_fuzz_target" in program_ad_fuzz_assurance.__all__
    assert "fuzz_assurance_policy" in program_ad_fuzz_assurance.__all__
    assert "list_fuzz_target_ids" in program_ad_fuzz_assurance.__all__


def test_target_validation() -> None:
    """Reject every invalid fuzz-target invariant at construction."""
    base: dict[str, Any] = {
        "target_id": "x",
        "title": "t",
        "summary": "s",
        "kind": "program_ad_ir",
        "rust_path": "p.rs",
        "package": "pkg",
    }
    assert FuzzTarget(**base).target_id == "x"
    with pytest.raises(ValueError, match="target_id"):
        FuzzTarget(**{**base, "target_id": ""})
    with pytest.raises(ValueError, match="title"):
        FuzzTarget(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        FuzzTarget(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="kind"):
        FuzzTarget(**{**base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="rust_path"):
        FuzzTarget(**{**base, "rust_path": ""})
    with pytest.raises(ValueError, match="package"):
        FuzzTarget(**{**base, "package": ""})
    with pytest.raises(ValueError, match="posture"):
        FuzzTarget(**{**base, "posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="api_stability_class"):
        FuzzTarget(**{**base, "api_stability_class": ""})
    with pytest.raises(ValueError, match="as_of"):
        FuzzTarget(**{**base, "as_of": ""})


def test_policy_validation() -> None:
    """Reject unbounded, continuous-default, and invent-green policy records."""
    base: dict[str, Any] = dict(
        policy_id="p",
        default_time_box_seconds=60,
        max_time_box_seconds=120,
        continuous_fuzz_default=False,
        ci_optional=True,
        invent_green_forbidden=True,
    )
    assert FuzzPolicy(**base).policy_id == "p"
    with pytest.raises(ValueError, match="policy_id"):
        FuzzPolicy(**{**base, "policy_id": ""})
    with pytest.raises(ValueError, match="default_time_box_seconds"):
        FuzzPolicy(**{**base, "default_time_box_seconds": 0})
    with pytest.raises(ValueError, match="max_time_box_seconds"):
        FuzzPolicy(**{**base, "max_time_box_seconds": 0})
    with pytest.raises(ValueError, match="cannot exceed"):
        FuzzPolicy(**{**base, "default_time_box_seconds": 200, "max_time_box_seconds": 100})
    with pytest.raises(ValueError, match="continuous_fuzz_default"):
        FuzzPolicy(**{**base, "continuous_fuzz_default": True})
    with pytest.raises(ValueError, match="invent_green_forbidden"):
        FuzzPolicy(**{**base, "invent_green_forbidden": False})


def test_decision_invariants() -> None:
    """Reject inconsistent dry-run/refusal decisions and blocker metadata."""
    with pytest.raises(ValueError, match="target_id"):
        FuzzProbeDecision(
            target_id="",
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("b",),
            time_box_seconds=0,
        )
    with pytest.raises(ValueError, match="outcome"):
        FuzzProbeDecision(
            target_id="t",
            outcome=cast(Any, "nope"),
            allowed=False,
            reason="r",
            blockers=("b",),
            time_box_seconds=0,
        )
    with pytest.raises(ValueError, match="require blockers"):
        FuzzProbeDecision(
            target_id="t",
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
            time_box_seconds=0,
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        FuzzProbeDecision(
            target_id="t",
            outcome="allowed_dry_run",
            allowed=True,
            reason="r",
            blockers=("b",),
            time_box_seconds=10,
        )
    with pytest.raises(ValueError, match="must use outcome=allowed_dry_run"):
        FuzzProbeDecision(
            target_id="t",
            outcome="refused",
            allowed=True,
            reason="r",
            blockers=(),
            time_box_seconds=10,
        )
    with pytest.raises(ValueError, match="must use outcome=refused"):
        FuzzProbeDecision(
            target_id="t",
            outcome="allowed_dry_run",
            allowed=False,
            reason="r",
            blockers=("b",),
            time_box_seconds=0,
        )
    with pytest.raises(ValueError, match="reason"):
        FuzzProbeDecision(
            target_id="t",
            outcome="refused",
            allowed=False,
            reason="",
            blockers=("b",),
            time_box_seconds=0,
        )
    with pytest.raises(ValueError, match="blockers entries"):
        FuzzProbeDecision(
            target_id="t",
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("ok", "  "),
            time_box_seconds=0,
        )
    with pytest.raises(ValueError, match="time_box_seconds"):
        FuzzProbeDecision(
            target_id="t",
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("b",),
            time_box_seconds=-1,
        )
    with pytest.raises(ValueError, match="positive time_box_seconds"):
        FuzzProbeDecision(
            target_id="t",
            outcome="allowed_dry_run",
            allowed=True,
            reason="r",
            blockers=(),
            time_box_seconds=0,
        )
    ok = dry_run_fuzz_target("studio_kuramoto_input")
    assert ok.to_dict()["allowed"] is True


def test_catalogue_map_runtime_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise blank, duplicate, empty, and valid catalogue construction."""
    from scpn_quantum_control import program_ad_fuzz_assurance as mod

    good = get_fuzz_target("program_ad_ir")
    blank = FuzzTarget(
        target_id="tmp",
        title="t",
        summary="s",
        kind="program_ad_ir",
        rust_path="p.rs",
        package="pkg",
    )
    object.__setattr__(blank, "target_id", "  ")
    monkeypatch.setattr(mod, "_CANONICAL_TARGETS", (blank,))
    with pytest.raises(RuntimeError, match="blank target_id"):
        mod._catalogue_map()

    a = get_fuzz_target("program_ad_ir")
    monkeypatch.setattr(mod, "_CANONICAL_TARGETS", (a, a))
    with pytest.raises(RuntimeError, match="duplicate target_id"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_TARGETS", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_TARGETS", (good,))
    assert mod._catalogue_map()[good.target_id].target_id == good.target_id


def test_target_to_dict() -> None:
    """Serialise a target without losing kind or identifier fields."""
    target = get_fuzz_target("ml_dsa_ntt")
    payload = target.to_dict()
    assert payload["kind"] == "ml_dsa_ntt"
    assert payload["target_id"] == "ml_dsa_ntt"
