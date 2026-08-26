# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for studio executive product
"""Real-surface tests for ``scpn_quantum_control.studio_executive_product``."""

from __future__ import annotations

import sys
from typing import Any, cast

import pytest

import scpn_quantum_control.studio_executive_product as studio_executive_product
from scpn_quantum_control.studio_executive_product import (
    STUDIO_EXECUTIVE_CLAIM_BOUNDARY,
    STUDIO_EXECUTIVE_PRODUCT_SCHEMA,
    ExecutiveVerbRow,
    MaterialisedCoverageFrontierProbe,
    PathEligibilityDecision,
    assert_studio_executive_product_integrity,
    build_studio_executive_product_registry,
    compute_coverage_frontier_score,
    decide_executive_path,
    get_executive_verb,
    iter_executive_verbs,
    list_executive_verb_ids,
    map_studio_executive_public_surfaces,
    materialise_demo_coverage_frontier_probe,
)


def _registry_verbs(registry: dict[str, object]) -> list[dict[str, object]]:
    """Narrow a validated registry verb collection for drift fixtures."""
    raw = registry["verbs"]
    assert isinstance(raw, list)
    return cast(list[dict[str, object]], raw)


def test_list_and_filters() -> None:
    """Expose the complete verb catalogue and deterministic posture filter."""
    ids = list_executive_verb_ids()
    assert "differentiate" in ids
    assert "execute" in ids
    assert "compile" in ids
    assert len(ids) == 9
    assert ids == list_executive_verb_ids()
    gated = iter_executive_verbs(support_posture="live_hardware_gated")
    assert gated
    assert all(row.verb_id == "execute" for row in gated)


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known verbs while rejecting blank and unknown identifiers."""
    row = get_executive_verb("differentiate")
    assert row.claim_boundary == STUDIO_EXECUTIVE_CLAIM_BOUNDARY
    assert row.allows_live_hardware is False
    assert "route_matrix_row" in row.route_matrix_pointer or "route" in row.route_matrix_pointer
    execute = get_executive_verb("execute")
    assert execute.allows_live_hardware is True
    assert execute.requires_approval is True
    with pytest.raises(ValueError, match="non-empty"):
        get_executive_verb("  ")
    with pytest.raises(ValueError, match="unknown verb_id"):
        get_executive_verb("not_a_verb")


def test_decide_executive_path() -> None:
    """Allow governed routes and refuse unsupported, dishonest, or ungated paths."""
    allowed = decide_executive_path("differentiate")
    assert allowed.allowed is True

    unsupported = decide_executive_path("simulate", request_unsupported_route=True)
    assert unsupported.allowed is False
    assert any(
        "unsupported" in b.lower()
        or "governed route" in b.lower()
        or "route_matrix_row" in b.lower()
        for b in unsupported.blockers
    )

    full = decide_executive_path("analyse", invent_green_full_coverage=True)
    assert full.allowed is False
    assert any("coverage" in b.lower() for b in full.blockers)

    gated = decide_executive_path("execute", approval_present=False)
    assert gated.allowed is False
    assert any("approval" in b.lower() for b in gated.blockers)

    approved = decide_executive_path("execute", approval_present=True)
    assert approved.allowed is True


def test_coverage_frontier_probe() -> None:
    """Materialise and recompute honest answer-rate frontier scores."""
    probe = materialise_demo_coverage_frontier_probe()
    assert probe.total_claims == 10
    assert probe.answered_confident == 3
    assert probe.honest_abstentions == 5
    assert abs(probe.answer_rate - 0.3) < 1e-12
    assert abs(probe.honesty_rate - 0.8) < 1e-12
    assert abs(probe.frontier_score - 0.24) < 1e-12
    assert probe.invent_green_full_coverage is False
    assert probe.off_frontier is True
    payload = probe.to_dict()
    assert payload["invent_green_full_coverage"] is False

    scored = compute_coverage_frontier_score(
        total_claims=4,
        answered_confident=2,
        honest_abstentions=2,
        improvable_candidates=0,
    )
    assert abs(scored.answer_rate - 0.5) < 1e-12
    assert abs(scored.honesty_rate - 1.0) < 1e-12
    assert scored.off_frontier is False


def test_public_surfaces_and_registry() -> None:
    """Map public owners and validate the complete executive registry."""
    surfaces = map_studio_executive_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.studio_executive_product" in paths
    assert "scpn_quantum_control.studio.verbs" in paths

    registry = build_studio_executive_product_registry()
    assert registry["schema"] == STUDIO_EXECUTIVE_PRODUCT_SCHEMA
    assert registry["invent_green_full_coverage_policy"] is False
    validated = assert_studio_executive_product_integrity(registry)
    assert validated["verb_count"] == 9
    assert assert_studio_executive_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_policy() -> None:
    """Reject verb-set drift, live-hardware leakage, and dishonest policy."""
    registry = build_studio_executive_product_registry()
    verbs = _registry_verbs(registry)

    stale_schema = dict(registry)
    stale_schema["schema"] = "studio_executive_product.v1"
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_studio_executive_product_integrity(stale_schema)

    broken = dict(registry)
    broken["verbs"] = verbs + [
        {
            "verb_id": "ghost",
            "title": "t",
            "summary": "s",
            "route_matrix_pointer": "r",
            "unsuitable_scenario_pointer": "p",
            "support_posture": "local_research",
            "requires_approval": False,
            "allows_live_hardware": False,
            "backends": ["python"],
            "as_of": "2026-07-24",
            "claim_boundary": STUDIO_EXECUTIVE_CLAIM_BOUNDARY,
        }
    ]
    broken["verb_count"] = len(cast(list[object], broken["verbs"]))
    with pytest.raises(ValueError, match="drift"):
        assert_studio_executive_product_integrity(broken)

    empty: dict[str, object] = {
        "schema": STUDIO_EXECUTIVE_PRODUCT_SCHEMA,
        "verbs": [],
        "blank_entry_count": 0,
        "verb_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty verbs"):
        assert_studio_executive_product_integrity(empty)

    live = dict(registry)
    live_rows = [dict(row) for row in verbs]
    for row in live_rows:
        if row.get("verb_id") == "simulate":
            row["allows_live_hardware"] = True
    live["verbs"] = live_rows
    with pytest.raises(ValueError, match="live hardware|execute"):
        assert_studio_executive_product_integrity(live)

    policy = dict(registry)
    policy["invent_green_full_coverage_policy"] = True
    with pytest.raises(ValueError, match="invent_green_full_coverage_policy"):
        assert_studio_executive_product_integrity(policy)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject malformed rows, missing sentinels, duplicates, and count drift."""
    registry = build_studio_executive_product_registry()
    verbs = _registry_verbs(registry)

    non_map = dict(registry)
    non_map["verbs"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_studio_executive_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in verbs]
    rows[0]["verb_id"] = "  "
    blank_id["verbs"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_studio_executive_product_integrity(blank_id)

    missing_route_pointer = dict(registry)
    brows = [dict(row) for row in verbs]
    brows[0]["route_matrix_pointer"] = ""
    missing_route_pointer["verbs"] = brows
    with pytest.raises(ValueError, match="route_matrix_pointer"):
        assert_studio_executive_product_integrity(missing_route_pointer)

    no_backends = dict(registry)
    bk = [dict(row) for row in verbs]
    bk[0]["backends"] = []
    no_backends["verbs"] = bk
    with pytest.raises(ValueError, match="backends"):
        assert_studio_executive_product_integrity(no_backends)

    no_default = dict(registry)
    renamed = [dict(row) for row in verbs]
    for row in renamed:
        if row.get("verb_id") == "differentiate":
            row["verb_id"] = "renamed"
    no_default["verbs"] = renamed
    with pytest.raises(ValueError, match="missing differentiate|drift"):
        assert_studio_executive_product_integrity(no_default)

    no_execute = dict(registry)
    without = [dict(row) for row in verbs if row.get("verb_id") != "execute"]
    no_execute["verbs"] = without
    no_execute["verb_count"] = len(without)
    with pytest.raises(ValueError, match="missing execute|drift"):
        assert_studio_executive_product_integrity(no_execute)

    dup = dict(registry)
    drows = [dict(row) for row in verbs]
    drows.append(dict(drows[0]))
    dup["verbs"] = drows
    dup["verb_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate verb_id"):
        assert_studio_executive_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_studio_executive_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["verb_count"] = 0
    with pytest.raises(ValueError, match="verb_count"):
        assert_studio_executive_product_integrity(count_mismatch)


def test_module_exports() -> None:
    """Keep every documented Studio product entry point public."""
    assert "materialise_demo_coverage_frontier_probe" in studio_executive_product.__all__
    assert "decide_executive_path" in studio_executive_product.__all__
    assert "list_executive_verb_ids" in studio_executive_product.__all__


def test_row_decision_probe_validation() -> None:
    """Enforce verb row, path decision, and coverage-probe invariants."""
    base: dict[str, Any] = {
        "verb_id": "x",
        "title": "t",
        "summary": "s",
        "route_matrix_pointer": "r",
        "unsuitable_scenario_pointer": "p",
        "support_posture": "local_research",
        "requires_approval": False,
        "allows_live_hardware": False,
        "backends": ("python",),
    }
    assert ExecutiveVerbRow(**base).verb_id == "x"
    assert ExecutiveVerbRow(**base).to_dict()["verb_id"] == "x"
    with pytest.raises(ValueError, match="verb_id"):
        ExecutiveVerbRow(**{**base, "verb_id": ""})
    with pytest.raises(ValueError, match="title"):
        ExecutiveVerbRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        ExecutiveVerbRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="route_matrix_pointer"):
        ExecutiveVerbRow(**{**base, "route_matrix_pointer": ""})
    with pytest.raises(ValueError, match="unsuitable_scenario_pointer"):
        ExecutiveVerbRow(**{**base, "unsuitable_scenario_pointer": ""})
    with pytest.raises(ValueError, match="support_posture"):
        ExecutiveVerbRow(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="only the execute"):
        ExecutiveVerbRow(**{**base, "verb_id": "simulate", "allows_live_hardware": True})
    with pytest.raises(ValueError, match="require_approval"):
        ExecutiveVerbRow(
            **{
                **base,
                "verb_id": "execute",
                "allows_live_hardware": True,
                "requires_approval": False,
            }
        )
    with pytest.raises(ValueError, match="backends"):
        ExecutiveVerbRow(**{**base, "backends": ()})
    with pytest.raises(ValueError, match="backends entries"):
        ExecutiveVerbRow(**{**base, "backends": ("python", "  ")})
    with pytest.raises(ValueError, match="as_of"):
        ExecutiveVerbRow(**{**base, "as_of": ""})

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
    assert decide_executive_path("compile").to_dict()["allowed"] is True

    with pytest.raises(ValueError, match="total_claims must be positive"):
        compute_coverage_frontier_score(
            total_claims=0,
            answered_confident=0,
            honest_abstentions=0,
        )
    with pytest.raises(ValueError, match="improvable_candidates must be non-negative"):
        compute_coverage_frontier_score(
            total_claims=2,
            answered_confident=1,
            honest_abstentions=0,
            improvable_candidates=-1,
        )
    with pytest.raises(ValueError, match="partition of claims exceeds total_claims"):
        compute_coverage_frontier_score(
            total_claims=2,
            answered_confident=2,
            honest_abstentions=1,
        )
    with pytest.raises(ValueError, match="invent_green_full_coverage"):
        MaterialisedCoverageFrontierProbe(
            total_claims=1,
            answered_confident=1,
            honest_abstentions=0,
            answer_rate=1.0,
            honesty_rate=1.0,
            frontier_score=1.0,
            invent_green_full_coverage=True,
            off_frontier=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedCoverageFrontierProbe(
            total_claims=1,
            answered_confident=0,
            honest_abstentions=1,
            answer_rate=0.0,
            honesty_rate=1.0,
            frontier_score=0.0,
            invent_green_full_coverage=False,
            off_frontier=False,
            demo_label="",
        )
    with pytest.raises(ValueError, match="total_claims must be non-negative"):
        MaterialisedCoverageFrontierProbe(
            total_claims=-1,
            answered_confident=0,
            honest_abstentions=0,
            answer_rate=0.0,
            honesty_rate=0.0,
            frontier_score=0.0,
            invent_green_full_coverage=False,
            off_frontier=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        MaterialisedCoverageFrontierProbe(
            total_claims=1,
            answered_confident=-1,
            honest_abstentions=0,
            answer_rate=0.0,
            honesty_rate=0.0,
            frontier_score=0.0,
            invent_green_full_coverage=False,
            off_frontier=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="cannot exceed total_claims"):
        MaterialisedCoverageFrontierProbe(
            total_claims=1,
            answered_confident=1,
            honest_abstentions=1,
            answer_rate=1.0,
            honesty_rate=1.0,
            frontier_score=1.0,
            invent_green_full_coverage=False,
            off_frontier=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="answer_rate"):
        MaterialisedCoverageFrontierProbe(
            total_claims=1,
            answered_confident=0,
            honest_abstentions=1,
            answer_rate=1.5,
            honesty_rate=1.0,
            frontier_score=0.0,
            invent_green_full_coverage=False,
            off_frontier=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="honesty_rate"):
        MaterialisedCoverageFrontierProbe(
            total_claims=1,
            answered_confident=0,
            honest_abstentions=1,
            answer_rate=0.0,
            honesty_rate=-0.1,
            frontier_score=0.0,
            invent_green_full_coverage=False,
            off_frontier=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="frontier_score"):
        MaterialisedCoverageFrontierProbe(
            total_claims=1,
            answered_confident=0,
            honest_abstentions=1,
            answer_rate=0.0,
            honesty_rate=1.0,
            frontier_score=2.0,
            invent_green_full_coverage=False,
            off_frontier=False,
            demo_label="d",
        )


def test_iter_unknown_posture_and_catalogue_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty posture filter + defensive catalogue RuntimeError paths."""
    empty = iter_executive_verbs(support_posture="policy_only")
    assert empty == ()

    monkeypatch.setattr(studio_executive_product, "_CANONICAL_VERBS", ())
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        studio_executive_product._catalogue_map()

    blank_row = ExecutiveVerbRow(
        verb_id="tmp",
        title="t",
        summary="s",
        route_matrix_pointer="r",
        unsuitable_scenario_pointer="p",
        support_posture="local_research",
        requires_approval=False,
        allows_live_hardware=False,
        backends=("python",),
    )
    # Force blank id past dataclass validation via object.__setattr__
    object.__setattr__(blank_row, "verb_id", "  ")
    monkeypatch.setattr(studio_executive_product, "_CANONICAL_VERBS", (blank_row,))
    with pytest.raises(RuntimeError, match="blank verb_id"):
        studio_executive_product._catalogue_map()

    good = ExecutiveVerbRow(
        verb_id="dup",
        title="t",
        summary="s",
        route_matrix_pointer="r",
        unsuitable_scenario_pointer="p",
        support_posture="local_research",
        requires_approval=False,
        allows_live_hardware=False,
        backends=("python",),
    )
    monkeypatch.setattr(studio_executive_product, "_CANONICAL_VERBS", (good, good))
    with pytest.raises(RuntimeError, match="duplicate verb_id"):
        studio_executive_product._catalogue_map()

    # Ambient empty-catalogue path requires the Studio platform import graph.
    # Base Python 3.11 CI omits scpn_studio_platform; product-local fallback still
    # returns a non-empty catalogue, so skip ambient-only emptiness coverage.
    try:
        import scpn_studio_platform  # noqa: F401
    except ImportError:
        pytest.skip("scpn_studio_platform not installed on this matrix cell")

    monkeypatch.setattr(
        "scpn_quantum_control.studio.verbs.QUANTUM_VERBS",
        (),
    )
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        studio_executive_product._build_canonical_verbs()


def test_iter_executive_verbs_without_posture_filter() -> None:
    """Unfiltered verb iter returns the full catalogue (support_posture is None)."""
    all_rows = iter_executive_verbs()
    assert len(all_rows) == len(list_executive_verb_ids())
    assert {row.verb_id for row in all_rows} == set(list_executive_verb_ids())


def test_build_canonical_verbs_uses_local_fallback_when_studio_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materialise the honest local catalogue when the Studio extra is absent."""
    monkeypatch.setitem(sys.modules, "scpn_quantum_control.studio.executive", None)
    rows = studio_executive_product._build_canonical_verbs()
    assert tuple(row.verb_id for row in rows) == tuple(
        spec[0] for spec in studio_executive_product._FALLBACK_VERB_SPECS
    )
    execute = rows[-1]
    assert execute.verb_id == "execute"
    assert execute.requires_approval is True
    assert execute.allows_live_hardware is True

    monkeypatch.setattr(studio_executive_product, "_FALLBACK_VERB_SPECS", ())
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        studio_executive_product._build_fallback_canonical_verbs()
