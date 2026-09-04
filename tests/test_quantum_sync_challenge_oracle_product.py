# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for quantum sync challenge oracle product
"""Real-surface tests for ``quantum_sync_challenge_oracle_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.quantum_sync_challenge_oracle_product as oracle_product
from scpn_quantum_control.quantum_sync_challenge_oracle_product import (
    QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY,
    QUANTUM_SYNC_CHALLENGE_ORACLE_PRODUCT_SCHEMA,
    BaselineCatalogueRow,
    MaterialisedOracleProbe,
    MetricCatalogueRow,
    PathEligibilityDecision,
    ProblemFamilyRow,
    assert_quantum_sync_challenge_oracle_product_integrity,
    build_quantum_sync_challenge_oracle_product_registry,
    compute_instance_digest,
    decide_challenge_path,
    get_problem_family,
    iter_problem_families,
    list_baseline_ids,
    list_metric_ids,
    list_problem_family_ids,
    map_quantum_sync_challenge_oracle_public_surfaces,
    materialise_demo_oracle_probe,
    materialise_oracle_probe,
)


def test_list_and_filters() -> None:
    """List catalogue identifiers and filter problem families by status."""
    families = list_problem_family_ids()
    assert "F1_all_to_all_kuramoto" in families
    assert "FH_hardware_gated" in families
    assert len(families) == 5
    assert len(list_metric_ids()) == 5
    assert len(list_baseline_ids()) == 3
    synthetic = iter_problem_families(support_status="synthetic_deterministic")
    assert synthetic
    assert all(row.support_status == "synthetic_deterministic" for row in synthetic)
    empty = iter_problem_families(support_status="noisy_sim")
    assert len(empty) == 1
    assert empty[0].family_id == "F4_noisy_finite_shot"


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known families and reject blank or unknown identifiers."""
    row = get_problem_family("F1_all_to_all_kuramoto")
    assert row.claim_boundary == QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY
    assert row.invent_green_advantage is False
    assert row.n_nodes >= 2
    hw = get_problem_family("FH_hardware_gated")
    assert hw.support_status == "hardware_gated"
    assert hw.support_posture == "live_hardware_gated"
    with pytest.raises(ValueError, match="non-empty"):
        get_problem_family("  ")
    with pytest.raises(ValueError, match="unknown family_id"):
        get_problem_family("not_a_family")


def test_instance_digest_stable() -> None:
    """Keep instance digests deterministic and sensitive to family and seed."""
    d1 = compute_instance_digest("F1_all_to_all_kuramoto")
    d2 = compute_instance_digest("F1_all_to_all_kuramoto")
    assert d1 == d2
    assert len(d1) == 64
    d_other = compute_instance_digest("F2_sparse_ring_xy")
    assert d_other != d1
    d_seed = compute_instance_digest("F1_all_to_all_kuramoto", seed=9999)
    assert d_seed != d1
    with pytest.raises(ValueError, match="seed"):
        compute_instance_digest("F1_all_to_all_kuramoto", seed=-1)
    with pytest.raises(ValueError, match="schema_version"):
        compute_instance_digest("F1_all_to_all_kuramoto", schema_version="  ")


def test_decide_challenge_path() -> None:
    """Allow bounded synthetic paths and refuse prohibited claim routes."""
    ok = decide_challenge_path("F1_all_to_all_kuramoto")
    assert ok.allowed is True

    adv = decide_challenge_path("F1_all_to_all_kuramoto", invent_green_advantage=True)
    assert adv.allowed is False
    assert any("advantage" in b.lower() for b in adv.blockers)

    rank = decide_challenge_path(
        "F1_all_to_all_kuramoto",
        request_leaderboard_rank=True,
        submission_validated=False,
    )
    assert rank.allowed is False
    assert any("unvalidated" in b.lower() or "leaderboard" in b.lower() for b in rank.blockers)

    ranked = decide_challenge_path(
        "F1_all_to_all_kuramoto",
        request_leaderboard_rank=True,
        submission_validated=True,
    )
    assert ranked.allowed is True

    hw = decide_challenge_path(
        "FH_hardware_gated",
        request_hardware_execution=True,
        owner_ticket_present=False,
    )
    assert hw.allowed is False
    assert any("ticket" in b.lower() for b in hw.blockers)

    hw_ticket = decide_challenge_path(
        "FH_hardware_gated",
        request_hardware_execution=True,
        owner_ticket_present=True,
    )
    assert hw_ticket.allowed is False
    assert any("residual" in b.lower() or "schema" in b.lower() for b in hw_ticket.blockers)

    non_hw = decide_challenge_path(
        "F1_all_to_all_kuramoto",
        request_hardware_execution=True,
        owner_ticket_present=True,
    )
    assert non_hw.allowed is False
    assert any("hardware_gated" in b for b in non_hw.blockers)


def test_oracle_probe() -> None:
    """Materialise the ambient witness probe and reject hardware-only rows."""
    probe = materialise_demo_oracle_probe()
    assert probe.family_id == "F1_all_to_all_kuramoto"
    assert probe.witness_case_count >= 1
    assert probe.witness_all_passed is True
    assert probe.invent_green_advantage is False
    assert probe.invent_green_hardware is False
    assert 0.0 <= probe.order_parameter <= 1.0
    assert len(probe.instance_digest) == 64
    payload = probe.to_dict()
    assert payload["invent_green_advantage"] is False

    f3 = materialise_oracle_probe("F3_cluster_sync")
    assert f3.family_id == "F3_cluster_sync"
    with pytest.raises(ValueError, match="hardware_gated"):
        materialise_oracle_probe("FH_hardware_gated")


def test_public_surfaces_and_registry() -> None:
    """Publish complete deterministic surface and registry catalogues."""
    surfaces = map_quantum_sync_challenge_oracle_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.quantum_sync_challenge_oracle_product" in paths
    assert "scpn_quantum_control.phase.synchronisation_witness" in paths

    registry = build_quantum_sync_challenge_oracle_product_registry()
    assert registry["schema"] == QUANTUM_SYNC_CHALLENGE_ORACLE_PRODUCT_SCHEMA
    assert registry["invent_green_advantage_policy"] is False
    validated = assert_quantum_sync_challenge_oracle_product_integrity(registry)
    assert validated["family_count"] == 5
    assert validated["metric_count"] == 5
    assert validated["baseline_count"] == 3
    assert assert_quantum_sync_challenge_oracle_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_policy() -> None:
    """Reject catalogue drift and any invent-green or hardware policy."""
    registry = build_quantum_sync_challenge_oracle_product_registry()
    families = cast(list[dict[str, object]], registry["families"])

    broken = dict(registry)
    broken["families"] = families + [
        {
            "family_id": "ghost",
            "title": "t",
            "summary": "s",
            "support_status": "synthetic_deterministic",
            "default_seed": 1,
            "n_nodes": 4,
            "ambient_pointer": "p",
            "route_matrix_pointer": "r",
            "unsuitable_scenario_pointer": "b",
            "invent_green_advantage": False,
            "support_posture": "local_research",
            "as_of": "2026-07-24",
            "claim_boundary": QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY,
        }
    ]
    broken["family_count"] = len(cast(list[object], broken["families"]))
    with pytest.raises(ValueError, match="drift"):
        assert_quantum_sync_challenge_oracle_product_integrity(broken)

    empty: dict[str, object] = {
        "families": [],
        "metrics": registry["metrics"],
        "baselines": registry["baselines"],
        "blank_entry_count": 0,
        "family_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty families"):
        assert_quantum_sync_challenge_oracle_product_integrity(empty)

    policy = dict(registry)
    policy["invent_green_advantage_policy"] = True
    with pytest.raises(ValueError, match="invent_green_advantage_policy"):
        assert_quantum_sync_challenge_oracle_product_integrity(policy)

    hw_policy = dict(registry)
    hw_policy["invent_green_hardware_policy"] = True
    with pytest.raises(ValueError, match="invent_green_hardware_policy"):
        assert_quantum_sync_challenge_oracle_product_integrity(hw_policy)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject blank, duplicate, invalid, and count-drifted registry rows."""
    registry = build_quantum_sync_challenge_oracle_product_registry()
    families = cast(list[dict[str, object]], registry["families"])
    metrics = cast(list[dict[str, object]], registry["metrics"])
    baselines = cast(list[dict[str, object]], registry["baselines"])

    non_map = dict(registry)
    non_map["families"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_quantum_sync_challenge_oracle_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in families]
    rows[0]["family_id"] = "  "
    blank_id["families"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_quantum_sync_challenge_oracle_product_integrity(blank_id)

    invent = dict(registry)
    irows = [dict(row) for row in families]
    irows[0]["invent_green_advantage"] = True
    invent["families"] = irows
    with pytest.raises(ValueError, match="invent_green_advantage"):
        assert_quantum_sync_challenge_oracle_product_integrity(invent)

    no_route_matrix_pointer = dict(registry)
    brows = [dict(row) for row in families]
    brows[0]["route_matrix_pointer"] = ""
    no_route_matrix_pointer["families"] = brows
    with pytest.raises(ValueError, match="route_matrix_pointer"):
        assert_quantum_sync_challenge_oracle_product_integrity(no_route_matrix_pointer)

    bad_status = dict(registry)
    srows = [dict(row) for row in families]
    srows[0]["support_status"] = "marketing"
    bad_status["families"] = srows
    with pytest.raises(ValueError, match="unknown support_status"):
        assert_quantum_sync_challenge_oracle_product_integrity(bad_status)

    no_f1 = dict(registry)
    without = [dict(row) for row in families if row.get("family_id") != "F1_all_to_all_kuramoto"]
    no_f1["families"] = without
    no_f1["family_count"] = len(without)
    with pytest.raises(ValueError, match="missing F1|drift"):
        assert_quantum_sync_challenge_oracle_product_integrity(no_f1)

    no_hw = dict(registry)
    without_hw = [dict(row) for row in families if row.get("support_status") != "hardware_gated"]
    no_hw["families"] = without_hw
    no_hw["family_count"] = len(without_hw)
    with pytest.raises(ValueError, match="hardware_gated|drift"):
        assert_quantum_sync_challenge_oracle_product_integrity(no_hw)

    dup = dict(registry)
    drows = [dict(row) for row in families]
    drows.append(dict(drows[0]))
    dup["families"] = drows
    dup["family_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate family_id"):
        assert_quantum_sync_challenge_oracle_product_integrity(dup)

    no_metrics = dict(registry)
    no_metrics["metrics"] = []
    no_metrics["metric_count"] = 0
    with pytest.raises(ValueError, match="non-empty metrics"):
        assert_quantum_sync_challenge_oracle_product_integrity(no_metrics)

    no_baselines = dict(registry)
    no_baselines["baselines"] = []
    no_baselines["baseline_count"] = 0
    with pytest.raises(ValueError, match="non-empty baselines"):
        assert_quantum_sync_challenge_oracle_product_integrity(no_baselines)

    metric_non_map = dict(registry)
    metric_non_map["metrics"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_quantum_sync_challenge_oracle_product_integrity(metric_non_map)

    metric_blank = dict(registry)
    mrows = [dict(row) for row in metrics]
    mrows[0]["metric_id"] = ""
    metric_blank["metrics"] = mrows
    with pytest.raises(ValueError, match="blank or invalid metric_id"):
        assert_quantum_sync_challenge_oracle_product_integrity(metric_blank)

    metric_dup = dict(registry)
    md = [dict(row) for row in metrics]
    md.append(dict(md[0]))
    metric_dup["metrics"] = md
    metric_dup["metric_count"] = len(md)
    with pytest.raises(ValueError, match="duplicate metric_id"):
        assert_quantum_sync_challenge_oracle_product_integrity(metric_dup)

    metric_drift = dict(registry)
    pruned = [dict(row) for row in metrics if row.get("metric_id") != "order_parameter_r1"]
    metric_drift["metrics"] = pruned
    metric_drift["metric_count"] = len(pruned)
    with pytest.raises(ValueError, match="metric set drift"):
        assert_quantum_sync_challenge_oracle_product_integrity(metric_drift)

    baseline_submit = dict(registry)
    brows_b = [dict(row) for row in baselines]
    brows_b[0]["no_submit"] = False
    baseline_submit["baselines"] = brows_b
    with pytest.raises(ValueError, match="no_submit"):
        assert_quantum_sync_challenge_oracle_product_integrity(baseline_submit)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_quantum_sync_challenge_oracle_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["family_count"] = 0
    with pytest.raises(ValueError, match="family_count"):
        assert_quantum_sync_challenge_oracle_product_integrity(count_mismatch)

    metric_count_bad = dict(registry)
    metric_count_bad["metric_count"] = 0
    with pytest.raises(ValueError, match="metric_count"):
        assert_quantum_sync_challenge_oracle_product_integrity(metric_count_bad)

    baseline_count_bad = dict(registry)
    baseline_count_bad["baseline_count"] = 0
    with pytest.raises(ValueError, match="baseline_count"):
        assert_quantum_sync_challenge_oracle_product_integrity(baseline_count_bad)


def test_module_exports() -> None:
    """Keep the documented challenge-oracle symbols publicly exported."""
    assert "materialise_demo_oracle_probe" in oracle_product.__all__
    assert "decide_challenge_path" in oracle_product.__all__
    assert "compute_instance_digest" in oracle_product.__all__


def test_row_decision_probe_validation() -> None:
    """Validate every row, decision, and materialised-probe invariant."""
    base_f: dict[str, Any] = {
        "family_id": "Fx",
        "title": "t",
        "summary": "s",
        "support_status": "synthetic_deterministic",
        "default_seed": 1,
        "n_nodes": 4,
        "ambient_pointer": "p",
        "route_matrix_pointer": "r",
        "unsuitable_scenario_pointer": "b",
    }
    assert ProblemFamilyRow(**base_f).family_id == "Fx"
    assert ProblemFamilyRow(**base_f).to_dict()["family_id"] == "Fx"
    with pytest.raises(ValueError, match="family_id"):
        ProblemFamilyRow(**{**base_f, "family_id": ""})
    with pytest.raises(ValueError, match="title"):
        ProblemFamilyRow(**{**base_f, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        ProblemFamilyRow(**{**base_f, "summary": ""})
    with pytest.raises(ValueError, match="support_status"):
        ProblemFamilyRow(**{**base_f, "support_status": cast(Any, "nope")})
    with pytest.raises(ValueError, match="default_seed"):
        ProblemFamilyRow(**{**base_f, "default_seed": -1})
    with pytest.raises(ValueError, match="n_nodes"):
        ProblemFamilyRow(**{**base_f, "n_nodes": 1})
    with pytest.raises(ValueError, match="ambient_pointer"):
        ProblemFamilyRow(**{**base_f, "ambient_pointer": ""})
    with pytest.raises(ValueError, match="route_matrix_pointer"):
        ProblemFamilyRow(**{**base_f, "route_matrix_pointer": ""})
    with pytest.raises(ValueError, match="unsuitable_scenario_pointer"):
        ProblemFamilyRow(**{**base_f, "unsuitable_scenario_pointer": ""})
    with pytest.raises(ValueError, match="invent_green_advantage"):
        ProblemFamilyRow(**{**base_f, "invent_green_advantage": True})
    with pytest.raises(ValueError, match="hardware_gated"):
        ProblemFamilyRow(
            **{
                **base_f,
                "support_status": "hardware_gated",
                "support_posture": "local_research",
            }
        )
    with pytest.raises(ValueError, match="support_posture"):
        ProblemFamilyRow(**{**base_f, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="as_of"):
        ProblemFamilyRow(**{**base_f, "as_of": ""})

    base_m: dict[str, Any] = {
        "metric_id": "m",
        "kind": "order_parameter",
        "title": "t",
        "ambient_pointer": "p",
        "required_for_leaderboard": True,
    }
    assert MetricCatalogueRow(**base_m).metric_id == "m"
    with pytest.raises(ValueError, match="metric_id"):
        MetricCatalogueRow(**{**base_m, "metric_id": ""})
    with pytest.raises(ValueError, match="kind"):
        MetricCatalogueRow(**{**base_m, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        MetricCatalogueRow(**{**base_m, "title": ""})
    with pytest.raises(ValueError, match="ambient_pointer"):
        MetricCatalogueRow(**{**base_m, "ambient_pointer": ""})
    with pytest.raises(ValueError, match="support_posture"):
        MetricCatalogueRow(**{**base_m, "support_posture": cast(Any, "nope")})

    base_b: dict[str, Any] = {
        "baseline_id": "b",
        "kind": "classical_numpy",
        "title": "t",
    }
    assert BaselineCatalogueRow(**base_b).baseline_id == "b"
    with pytest.raises(ValueError, match="baseline_id"):
        BaselineCatalogueRow(**{**base_b, "baseline_id": ""})
    with pytest.raises(ValueError, match="kind"):
        BaselineCatalogueRow(**{**base_b, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        BaselineCatalogueRow(**{**base_b, "title": ""})
    with pytest.raises(ValueError, match="no_submit"):
        BaselineCatalogueRow(**{**base_b, "no_submit": False})
    with pytest.raises(ValueError, match="owner_ticket_required"):
        BaselineCatalogueRow(
            **{
                **base_b,
                "kind": "hardware_schema_only",
                "owner_ticket_required": False,
            }
        )
    with pytest.raises(ValueError, match="support_posture"):
        BaselineCatalogueRow(**{**base_b, "support_posture": cast(Any, "nope")})

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
    assert decide_challenge_path("F1_all_to_all_kuramoto").to_dict()["allowed"] is True

    with pytest.raises(ValueError, match="family_id"):
        MaterialisedOracleProbe(
            family_id="",
            instance_digest="a" * 64,
            witness_case_count=1,
            witness_all_passed=True,
            order_parameter=0.5,
            invent_green_advantage=False,
            invent_green_hardware=False,
            ambient_witness_claim_boundary="b",
            demo_label="d",
        )
    with pytest.raises(ValueError, match="instance_digest"):
        MaterialisedOracleProbe(
            family_id="F1",
            instance_digest="",
            witness_case_count=1,
            witness_all_passed=True,
            order_parameter=0.5,
            invent_green_advantage=False,
            invent_green_hardware=False,
            ambient_witness_claim_boundary="b",
            demo_label="d",
        )
    with pytest.raises(ValueError, match="witness_case_count"):
        MaterialisedOracleProbe(
            family_id="F1",
            instance_digest="a" * 64,
            witness_case_count=0,
            witness_all_passed=True,
            order_parameter=0.5,
            invent_green_advantage=False,
            invent_green_hardware=False,
            ambient_witness_claim_boundary="b",
            demo_label="d",
        )
    with pytest.raises(ValueError, match="order_parameter"):
        MaterialisedOracleProbe(
            family_id="F1",
            instance_digest="a" * 64,
            witness_case_count=1,
            witness_all_passed=True,
            order_parameter=1.5,
            invent_green_advantage=False,
            invent_green_hardware=False,
            ambient_witness_claim_boundary="b",
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_advantage"):
        MaterialisedOracleProbe(
            family_id="F1",
            instance_digest="a" * 64,
            witness_case_count=1,
            witness_all_passed=True,
            order_parameter=0.5,
            invent_green_advantage=True,
            invent_green_hardware=False,
            ambient_witness_claim_boundary="b",
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_hardware"):
        MaterialisedOracleProbe(
            family_id="F1",
            instance_digest="a" * 64,
            witness_case_count=1,
            witness_all_passed=True,
            order_parameter=0.5,
            invent_green_advantage=False,
            invent_green_hardware=True,
            ambient_witness_claim_boundary="b",
            demo_label="d",
        )
    with pytest.raises(ValueError, match="ambient_witness_claim_boundary"):
        MaterialisedOracleProbe(
            family_id="F1",
            instance_digest="a" * 64,
            witness_case_count=1,
            witness_all_passed=True,
            order_parameter=0.5,
            invent_green_advantage=False,
            invent_green_hardware=False,
            ambient_witness_claim_boundary="",
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedOracleProbe(
            family_id="F1",
            instance_digest="a" * 64,
            witness_case_count=1,
            witness_all_passed=True,
            order_parameter=0.5,
            invent_green_advantage=False,
            invent_green_hardware=False,
            ambient_witness_claim_boundary="b",
            demo_label="",
        )


def test_catalogue_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject empty, blank, and duplicate internal catalogue definitions."""
    monkeypatch.setattr(oracle_product, "_FAMILIES", ())
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        oracle_product._family_map()

    blank = ProblemFamilyRow(
        family_id="tmp",
        title="t",
        summary="s",
        support_status="synthetic_deterministic",
        default_seed=1,
        n_nodes=4,
        ambient_pointer="p",
        route_matrix_pointer="r",
        unsuitable_scenario_pointer="b",
    )
    object.__setattr__(blank, "family_id", "  ")
    monkeypatch.setattr(oracle_product, "_FAMILIES", (blank,))
    with pytest.raises(RuntimeError, match="blank family_id"):
        oracle_product._family_map()

    good = ProblemFamilyRow(
        family_id="dup",
        title="t",
        summary="s",
        support_status="synthetic_deterministic",
        default_seed=1,
        n_nodes=4,
        ambient_pointer="p",
        route_matrix_pointer="r",
        unsuitable_scenario_pointer="b",
    )
    monkeypatch.setattr(oracle_product, "_FAMILIES", (good, good))
    with pytest.raises(RuntimeError, match="duplicate family_id"):
        oracle_product._family_map()


def test_iter_problem_families_without_filter_returns_full_catalogue() -> None:
    """Unfiltered family iter returns every catalogue row."""
    rows = iter_problem_families()
    assert len(rows) == len(list_problem_family_ids())
    assert {row.family_id for row in rows} == set(list_problem_family_ids())


def test_materialise_oracle_probe_rejects_empty_witness_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty ambient witness suite is refused with RuntimeError."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        oracle_product,
        "run_sync_witness_suite",
        lambda: SimpleNamespace(records=()),
    )
    with pytest.raises(RuntimeError, match="no records"):
        materialise_oracle_probe("F1_all_to_all_kuramoto")


def test_integrity_rejects_non_mapping_baseline_row() -> None:
    """Baseline catalogue rows must be mappings."""
    registry = build_quantum_sync_challenge_oracle_product_registry()
    broken = dict(registry)
    broken["baselines"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="baseline row 0 must be a mapping"):
        assert_quantum_sync_challenge_oracle_product_integrity(broken)
