# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for Phase-QNode product surface
"""Real-surface tests for ``scpn_quantum_control.phase_qnode_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.phase_qnode_product as phase_qnode_product
from scpn_quantum_control.phase_qnode_product import (
    PHASE_QNODE_PRODUCT_CLAIM_BOUNDARY,
    PHASE_QNODE_PRODUCT_SCHEMA,
    PhaseQNodeJourney,
    PhaseQNodeJourneyDecision,
    assert_phase_qnode_product_integrity,
    build_phase_qnode_product_registry,
    dry_run_phase_qnode_journey,
    get_phase_qnode_journey,
    iter_phase_qnode_journeys,
    list_phase_qnode_journey_ids,
    map_phase_qnode_public_surfaces,
)


def test_list_journeys_and_filters() -> None:
    """List stable journeys and filter them by support badge."""
    ids = list_phase_qnode_journey_ids()
    assert "build_differentiate_dry_run" in ids
    assert ids == list_phase_qnode_journey_ids()
    local = iter_phase_qnode_journeys(support_badge="local_dry_run")
    assert local
    assert all(row.support_badge == "local_dry_run" for row in local)


def test_get_known_and_unknown_fail_closed() -> None:
    """Return known journeys and reject blank or unknown identifiers."""
    journey = get_phase_qnode_journey("build_differentiate_dry_run")
    assert journey.allows_hardware is False
    assert journey.api_stability_class == "experimental_workbench"
    assert journey.claim_boundary == PHASE_QNODE_PRODUCT_CLAIM_BOUNDARY
    with pytest.raises(ValueError, match="non-empty"):
        get_phase_qnode_journey("  ")
    with pytest.raises(ValueError, match="unknown journey_id"):
        get_phase_qnode_journey("not_a_journey")


def test_dry_run_allowed_canonical_journey() -> None:
    """Allow a canonical journey only as a no-hardware dry run."""
    decision = dry_run_phase_qnode_journey("build_differentiate_dry_run")
    assert decision.allowed is True
    assert decision.outcome == "allowed_dry_run"
    assert decision.blockers == ()
    assert decision.steps_completed
    assert "no QPU submission" in decision.reason


def test_dry_run_refuses_hardware() -> None:
    """Refuse hardware requests across the public journey surface."""
    refused = dry_run_phase_qnode_journey(
        "build_differentiate_dry_run",
        request_hardware=True,
    )
    assert refused.allowed is False
    assert refused.blockers
    assert any("hardware" in item.lower() or "qpu" in item.lower() for item in refused.blockers)

    provider = dry_run_phase_qnode_journey(
        "provider_transform_boundary",
        request_hardware=True,
    )
    assert provider.allowed is False


def test_public_surface_map() -> None:
    """Expose the deterministic Phase-QNode public-surface map."""
    surfaces = map_phase_qnode_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.phase.qnode_circuit" in paths
    for row in surfaces:
        assert row["api_stability_class"] == "experimental_workbench"
        assert row["role"] == "phase_qnode_product_surface"


def test_registry_and_integrity() -> None:
    """Build and validate the canonical product registry."""
    registry = build_phase_qnode_product_registry()
    assert registry["schema"] == PHASE_QNODE_PRODUCT_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_journey_id"] == "build_differentiate_dry_run"
    count = registry["journey_count"]
    assert isinstance(count, int)
    assert count == len(list_phase_qnode_journey_ids())
    validated = assert_phase_qnode_product_integrity(registry)
    assert validated["journey_count"] == count
    assert assert_phase_qnode_product_integrity()["blank_entry_count"] == 0


def test_module_exports() -> None:
    """Keep documented product functions in the export list."""
    assert "dry_run_phase_qnode_journey" in phase_qnode_product.__all__
    assert "map_phase_qnode_public_surfaces" in phase_qnode_product.__all__


def test_journey_validation() -> None:
    """Enforce every immutable journey construction invariant."""
    base: dict[str, Any] = {
        "journey_id": "x",
        "title": "t",
        "summary": "s",
        "module_path": "m",
        "support_badge": "local_dry_run",
        "steps": ("a", "b"),
    }
    assert PhaseQNodeJourney(**base).journey_id == "x"
    with pytest.raises(ValueError, match="journey_id"):
        PhaseQNodeJourney(**{**base, "journey_id": ""})
    with pytest.raises(ValueError, match="title"):
        PhaseQNodeJourney(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        PhaseQNodeJourney(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="module_path"):
        PhaseQNodeJourney(**{**base, "module_path": ""})
    with pytest.raises(ValueError, match="support_badge"):
        PhaseQNodeJourney(**{**base, "support_badge": cast(Any, "nope")})
    with pytest.raises(ValueError, match="steps"):
        PhaseQNodeJourney(**{**base, "steps": ()})
    with pytest.raises(ValueError, match="steps entries"):
        PhaseQNodeJourney(**{**base, "steps": ("ok", "")})
    with pytest.raises(ValueError, match="allows_hardware=False"):
        PhaseQNodeJourney(**{**base, "allows_hardware": True})
    with pytest.raises(ValueError, match="as_of"):
        PhaseQNodeJourney(**{**base, "as_of": ""})
    with pytest.raises(ValueError, match="api_stability_class"):
        PhaseQNodeJourney(**{**base, "api_stability_class": ""})


def test_decision_invariants() -> None:
    """Enforce consistent allowed and refused decision states."""
    with pytest.raises(ValueError, match="journey_id"):
        PhaseQNodeJourneyDecision(
            journey_id="",
            outcome="refused",
            allowed=False,
            support_badge="local_dry_run",
            reason="r",
            blockers=("b",),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="outcome"):
        PhaseQNodeJourneyDecision(
            journey_id="x",
            outcome=cast(Any, "nope"),
            allowed=False,
            support_badge="local_dry_run",
            reason="r",
            blockers=("b",),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="require blockers"):
        PhaseQNodeJourneyDecision(
            journey_id="x",
            outcome="refused",
            allowed=False,
            support_badge="local_dry_run",
            reason="r",
            blockers=(),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        PhaseQNodeJourneyDecision(
            journey_id="x",
            outcome="allowed_dry_run",
            allowed=True,
            support_badge="local_dry_run",
            reason="r",
            blockers=("b",),
            steps_completed=("a",),
        )
    with pytest.raises(ValueError, match="must use outcome=allowed_dry_run"):
        PhaseQNodeJourneyDecision(
            journey_id="x",
            outcome="refused",
            allowed=True,
            support_badge="local_dry_run",
            reason="r",
            blockers=(),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="must use outcome=refused"):
        PhaseQNodeJourneyDecision(
            journey_id="x",
            outcome="allowed_dry_run",
            allowed=False,
            support_badge="local_dry_run",
            reason="r",
            blockers=("b",),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="reason"):
        PhaseQNodeJourneyDecision(
            journey_id="x",
            outcome="allowed_dry_run",
            allowed=True,
            support_badge="local_dry_run",
            reason="",
            blockers=(),
            steps_completed=("a",),
        )
    with pytest.raises(ValueError, match="blockers entries"):
        PhaseQNodeJourneyDecision(
            journey_id="x",
            outcome="refused",
            allowed=False,
            support_badge="local_dry_run",
            reason="r",
            blockers=(" ",),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="support_badge"):
        PhaseQNodeJourneyDecision(
            journey_id="x",
            outcome="refused",
            allowed=False,
            support_badge=cast(Any, "nope"),
            reason="r",
            blockers=("b",),
            steps_completed=(),
        )


def test_to_dict_paths() -> None:
    """Materialise journey and decision records as JSON-ready mappings."""
    journey = get_phase_qnode_journey("local_transform_suite")
    assert journey.to_dict()["support_badge"] == "local_dry_run"
    decision = dry_run_phase_qnode_journey("framework_bridge_parity")
    assert decision.to_dict()["allowed"] is True


def test_integrity_rejects_drift() -> None:
    """Reject malformed rows, invented hardware, and catalogue drift."""
    good = build_phase_qnode_product_registry()
    assert_phase_qnode_product_integrity(good)

    stale_schema = dict(good)
    stale_schema["schema"] = "phase_qnode_product.v1"
    with pytest.raises(ValueError, match="registry schema must be phase_qnode_product.v2"):
        assert_phase_qnode_product_integrity(stale_schema)

    bad_blank = dict(good)
    bad_blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_phase_qnode_product_integrity(bad_blank)

    empty = dict(good)
    empty["journeys"] = []
    with pytest.raises(ValueError, match="non-empty journeys"):
        assert_phase_qnode_product_integrity(empty)

    not_map = dict(good)
    not_map["journeys"] = [123]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_phase_qnode_product_integrity(not_map)

    raw = good["journeys"]
    assert isinstance(raw, list)
    journeys = [dict(cast(dict[str, object], row)) for row in raw]

    invent_hw = dict(good)
    default_row = next(r for r in journeys if r["journey_id"] == "build_differentiate_dry_run")
    broken = dict(default_row)
    broken["allows_hardware"] = True
    invent_hw["journeys"] = [
        broken if r["journey_id"] == "build_differentiate_dry_run" else r for r in journeys
    ]
    with pytest.raises(ValueError, match="allows_hardware=False"):
        assert_phase_qnode_product_integrity(invent_hw)

    other = next(r for r in journeys if r["journey_id"] != "build_differentiate_dry_run")
    invent_other = dict(good)
    other_broken = dict(other)
    other_broken["allows_hardware"] = True
    invent_other["journeys"] = [
        other_broken if r["journey_id"] == other["journey_id"] else r for r in journeys
    ]
    with pytest.raises(ValueError, match="invent-green hardware"):
        assert_phase_qnode_product_integrity(invent_other)

    blank_id = dict(good)
    blank_row = dict(journeys[0])
    blank_row["journey_id"] = ""
    blank_id["journeys"] = [blank_row, *journeys[1:]]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_phase_qnode_product_integrity(blank_id)

    bad_badge = dict(good)
    bad = dict(journeys[0])
    bad["support_badge"] = "nope"
    # keep journey_id so not invent_hardware on allows_hardware
    if bad["journey_id"] == "build_differentiate_dry_run":
        bad = dict(other)
        bad["support_badge"] = "nope"
    bad_badge["journeys"] = [bad if r["journey_id"] == bad["journey_id"] else r for r in journeys]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_phase_qnode_product_integrity(bad_badge)

    no_steps = dict(good)
    ns = dict(other)
    ns["steps"] = []
    no_steps["journeys"] = [ns if r["journey_id"] == other["journey_id"] else r for r in journeys]
    with pytest.raises(ValueError, match="non-empty steps"):
        assert_phase_qnode_product_integrity(no_steps)

    missing = dict(good)
    missing_rows = [r for r in journeys if r["journey_id"] != "build_differentiate_dry_run"]
    missing["journeys"] = missing_rows
    missing["journey_count"] = len(missing_rows)
    with pytest.raises(ValueError, match="missing build_differentiate_dry_run|drift"):
        assert_phase_qnode_product_integrity(missing)

    bad_count = dict(good)
    bad_count["journey_count"] = 0
    with pytest.raises(ValueError, match="journey_count"):
        assert_phase_qnode_product_integrity(bad_count)

    duplicate = dict(good)
    duplicate["journeys"] = [journeys[0], journeys[0]]
    duplicate["journey_count"] = 2
    with pytest.raises(ValueError, match="duplicate"):
        assert_phase_qnode_product_integrity(duplicate)


def test_catalogue_map_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject empty, blank, and duplicate canonical catalogue entries."""
    mod = phase_qnode_product
    with pytest.raises(RuntimeError, match="non-empty"):
        monkeypatch.setattr(mod, "_CANONICAL_JOURNEYS", ())
        mod._catalogue_map()
    good = get_phase_qnode_journey("build_differentiate_dry_run")
    blank = PhaseQNodeJourney(
        journey_id="tmp",
        title="t",
        summary="s",
        module_path="m",
        support_badge="local_dry_run",
        steps=("a",),
    )
    object.__setattr__(blank, "journey_id", "  ")
    with pytest.raises(RuntimeError, match="blank journey_id"):
        monkeypatch.setattr(mod, "_CANONICAL_JOURNEYS", (blank,))
        mod._catalogue_map()
    with pytest.raises(RuntimeError, match="duplicate"):
        monkeypatch.setattr(mod, "_CANONICAL_JOURNEYS", (good, good))
        mod._catalogue_map()


def test_iter_phase_qnode_journeys_without_filter_returns_full_catalogue() -> None:
    """Unfiltered journey iter returns every catalogue row."""
    rows = iter_phase_qnode_journeys()
    assert len(rows) == len(list_phase_qnode_journey_ids())
    assert {row.journey_id for row in rows} == set(list_phase_qnode_journey_ids())


def test_dry_run_provider_boundary_allowed_without_hardware() -> None:
    """provider_boundary dry-run is allowed when no hardware is requested."""
    decision = dry_run_phase_qnode_journey("provider_transform_boundary")
    assert decision.allowed is True
    assert decision.outcome == "allowed_dry_run"
    assert decision.support_badge == "provider_boundary"
    assert decision.steps_completed
    assert decision.blockers == ()


def test_public_surface_map_skips_duplicate_module_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate module_path across journeys appears once in the public map."""
    first = PhaseQNodeJourney(
        journey_id="dup_a",
        title="t",
        summary="s",
        module_path="scpn_quantum_control.phase.shared_mod",
        support_badge="local_dry_run",
        steps=("a",),
    )
    second = PhaseQNodeJourney(
        journey_id="dup_b",
        title="t2",
        summary="s2",
        module_path="scpn_quantum_control.phase.shared_mod",
        support_badge="local_dry_run",
        steps=("b",),
    )
    monkeypatch.setattr(phase_qnode_product, "_CANONICAL_JOURNEYS", (first, second))
    surfaces = map_phase_qnode_public_surfaces()
    paths = [row["module_path"] for row in surfaces]
    assert paths.count("scpn_quantum_control.phase.shared_mod") == 1
    shared = next(
        row for row in surfaces if row["module_path"] == "scpn_quantum_control.phase.shared_mod"
    )
    assert set(cast(list[str], shared["journey_ids"])) == {"dup_a", "dup_b"}


def test_integrity_rejects_journey_set_drift() -> None:
    """Registry journey_id set must match the live catalogue exactly."""
    good = build_phase_qnode_product_registry()
    raw = good["journeys"]
    assert isinstance(raw, list)
    journeys = [dict(cast(dict[str, object], row)) for row in raw]
    drifted = dict(good)
    ghost = dict(journeys[0])
    ghost["journey_id"] = "ghost_extra_journey"
    drifted["journeys"] = journeys + [ghost]
    drifted["journey_count"] = len(journeys) + 1
    with pytest.raises(ValueError, match="drift"):
        assert_phase_qnode_product_integrity(drifted)
