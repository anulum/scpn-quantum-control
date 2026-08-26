# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for whole-program AD product surface
"""Real-surface tests for ``scpn_quantum_control.whole_program_ad_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.whole_program_ad_product as whole_program_ad_product
from scpn_quantum_control.whole_program_ad_product import (
    WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY,
    WHOLE_PROGRAM_AD_PRODUCT_SCHEMA,
    WholeProgramADJourney,
    WholeProgramADJourneyDecision,
    assert_whole_program_ad_product_integrity,
    build_whole_program_ad_product_registry,
    dry_run_whole_program_ad_journey,
    get_whole_program_ad_journey,
    iter_whole_program_ad_journeys,
    list_whole_program_ad_journey_ids,
    map_whole_program_ad_architecture_layers,
    map_whole_program_ad_public_surfaces,
)


def test_list_journeys_and_filters() -> None:
    """List the stable catalogue and apply both supported filters."""
    ids = list_whole_program_ad_journey_ids()
    assert "frontend_compile_dry_run" in ids
    assert "value_and_grad_local_dry_run" in ids
    assert "unsupported_frontend_fail_closed" in ids
    assert ids == list_whole_program_ad_journey_ids()
    local = iter_whole_program_ad_journeys(support_badge="local_dry_run")
    assert local
    assert all(row.support_badge == "local_dry_run" for row in local)
    frontend = iter_whole_program_ad_journeys(architecture_layer="frontend")
    assert frontend
    assert all(row.architecture_layer == "frontend" for row in frontend)


def test_get_known_and_unknown_fail_closed() -> None:
    """Return known journeys and reject blank or unknown identifiers."""
    journey = get_whole_program_ad_journey("frontend_compile_dry_run")
    assert journey.allows_hardware is False
    assert journey.api_stability_class == "experimental_workbench"
    assert journey.claim_boundary == WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY
    unsupported = get_whole_program_ad_journey("unsupported_frontend_fail_closed")
    assert unsupported.unsuitable_scenario_pointer
    assert (
        "scenario_row" in unsupported.unsuitable_scenario_pointer
        or "unsuitable" in unsupported.unsuitable_scenario_pointer
    )
    with pytest.raises(ValueError, match="non-empty"):
        get_whole_program_ad_journey("  ")
    with pytest.raises(ValueError, match="unknown journey_id"):
        get_whole_program_ad_journey("not_a_journey")


def test_dry_run_allowed_canonical_journey() -> None:
    """Allow canonical local journeys only in dry-run posture."""
    decision = dry_run_whole_program_ad_journey("frontend_compile_dry_run")
    assert decision.allowed is True
    assert decision.outcome == "allowed_dry_run"
    assert decision.blockers == ()
    assert decision.steps_completed
    assert "no QPU submission" in decision.reason

    value_plan = dry_run_whole_program_ad_journey("value_and_grad_local_dry_run")
    assert value_plan.allowed is True
    assert "require_frontend_ready" in value_plan.steps_completed


def test_dry_run_refuses_hardware() -> None:
    """Refuse a hardware request through the public dry-run API."""
    refused = dry_run_whole_program_ad_journey(
        "frontend_compile_dry_run",
        request_hardware=True,
    )
    assert refused.allowed is False
    assert refused.blockers
    assert any("hardware" in item.lower() or "qpu" in item.lower() for item in refused.blockers)


def test_dry_run_refuses_unsupported_frontend_execute() -> None:
    """Refuse unsupported execution while retaining its boundary map."""
    refused = dry_run_whole_program_ad_journey(
        "unsupported_frontend_fail_closed",
        request_unsupported_frontend_execute=True,
    )
    assert refused.allowed is False
    assert any(
        "bl-53" in item.lower() or "unsupported" in item.lower() for item in refused.blockers
    )

    # Boundary map dry-run without execute request remains allowed.
    mapped = dry_run_whole_program_ad_journey("unsupported_frontend_fail_closed")
    assert mapped.allowed is True
    assert "point_unsuitable_scenario_registry" in mapped.steps_completed


def test_dry_run_refuses_polyglot_and_edge_invent_green() -> None:
    """Refuse residual certificate and edge-routing completion claims."""
    poly = dry_run_whole_program_ad_journey(
        "polyglot_parity_boundary",
        request_polyglot_cert=True,
    )
    assert poly.allowed is False
    assert any("bl-49" in item.lower() or "polyglot" in item.lower() for item in poly.blockers)

    edge = dry_run_whole_program_ad_journey(
        "edge_wasm_boundary",
        request_edge_wasm=True,
    )
    assert edge.allowed is False
    assert any("bl-74" in item.lower() or "wasm" in item.lower() for item in edge.blockers)

    # Boundary dry-run without invent-green residual claim is allowed.
    assert dry_run_whole_program_ad_journey("polyglot_parity_boundary").allowed is True
    assert dry_run_whole_program_ad_journey("edge_wasm_boundary").allowed is True


def test_public_surface_and_architecture_map() -> None:
    """Expose deterministic public-surface and architecture maps."""
    surfaces = map_whole_program_ad_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.whole_program_frontend" in paths
    assert "scpn_quantum_control.whole_program_ad_api" in paths
    for row in surfaces:
        assert row["api_stability_class"] == "experimental_workbench"
        assert row["role"] == "whole_program_ad_product_surface"

    layers = map_whole_program_ad_architecture_layers()
    assert layers
    layer_names = [row["layer"] for row in layers]
    assert "frontend" in layer_names
    assert "product" in layer_names
    assert "ir" in layer_names
    ir = next(row for row in layers if row["layer"] == "ir")
    modules = ir["module_paths"]
    assert isinstance(modules, list)
    assert any("whole_program_ad_result" in str(path) for path in modules)


def test_registry_and_integrity() -> None:
    """Build and validate the canonical serialisable registry."""
    registry = build_whole_program_ad_product_registry()
    assert registry["schema"] == WHOLE_PROGRAM_AD_PRODUCT_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_journey_id"] == "frontend_compile_dry_run"
    count = registry["journey_count"]
    assert isinstance(count, int)
    assert count == len(list_whole_program_ad_journey_ids())
    validated = assert_whole_program_ad_product_integrity(registry)
    assert validated["journey_count"] == count
    assert assert_whole_program_ad_product_integrity()["blank_entry_count"] == 0
    assert isinstance(validated["architecture_layers"], list)
    assert validated["architecture_layers"]


def test_integrity_rejects_drift_and_hardware() -> None:
    """Reject catalogue drift, hardware claims, and empty registries."""
    registry = build_whole_program_ad_product_registry()
    journeys = cast(list[dict[str, object]], registry["journeys"])
    broken = dict(registry)
    broken["journeys"] = journeys + [
        {
            "journey_id": "ghost",
            "title": "t",
            "summary": "s",
            "module_path": "m",
            "support_badge": "local_dry_run",
            "steps": ["a"],
            "allows_hardware": False,
            "architecture_layer": "frontend",
            "unsuitable_scenario_pointer": "",
            "api_stability_class": "experimental_workbench",
            "as_of": "2026-07-24",
            "claim_boundary": WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY,
        }
    ]
    broken["journey_count"] = len(cast(list[object], broken["journeys"]))
    with pytest.raises(ValueError, match="drift"):
        assert_whole_program_ad_product_integrity(broken)

    hw = dict(registry)
    hw_journeys = [dict(row) for row in journeys]
    hw_journeys[0]["allows_hardware"] = True
    hw["journeys"] = hw_journeys
    with pytest.raises(ValueError, match="invent-green hardware|allows_hardware"):
        assert_whole_program_ad_product_integrity(hw)

    empty: dict[str, object] = {"journeys": [], "blank_entry_count": 0, "journey_count": 0}
    with pytest.raises(ValueError, match="non-empty journeys"):
        assert_whole_program_ad_product_integrity(empty)


def test_module_exports() -> None:
    """Keep the documented product functions in the module export list."""
    assert "dry_run_whole_program_ad_journey" in whole_program_ad_product.__all__
    assert "map_whole_program_ad_public_surfaces" in whole_program_ad_product.__all__
    assert "map_whole_program_ad_architecture_layers" in whole_program_ad_product.__all__


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
    assert WholeProgramADJourney(**base).journey_id == "x"
    with pytest.raises(ValueError, match="journey_id"):
        WholeProgramADJourney(**{**base, "journey_id": ""})
    with pytest.raises(ValueError, match="title"):
        WholeProgramADJourney(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        WholeProgramADJourney(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="module_path"):
        WholeProgramADJourney(**{**base, "module_path": ""})
    with pytest.raises(ValueError, match="support_badge"):
        WholeProgramADJourney(**{**base, "support_badge": cast(Any, "nope")})
    with pytest.raises(ValueError, match="steps"):
        WholeProgramADJourney(**{**base, "steps": ()})
    with pytest.raises(ValueError, match="steps entries"):
        WholeProgramADJourney(**{**base, "steps": ("ok", "")})
    with pytest.raises(ValueError, match="allows_hardware=False"):
        WholeProgramADJourney(**{**base, "allows_hardware": True})
    with pytest.raises(ValueError, match="architecture_layer"):
        WholeProgramADJourney(**{**base, "architecture_layer": ""})
    with pytest.raises(ValueError, match="as_of"):
        WholeProgramADJourney(**{**base, "as_of": ""})
    with pytest.raises(ValueError, match="api_stability_class"):
        WholeProgramADJourney(**{**base, "api_stability_class": ""})


def test_decision_invariants() -> None:
    """Enforce allowed and refused decision-state consistency."""
    with pytest.raises(ValueError, match="journey_id"):
        WholeProgramADJourneyDecision(
            journey_id="",
            outcome="refused",
            allowed=False,
            support_badge="local_dry_run",
            reason="r",
            blockers=("b",),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="outcome"):
        WholeProgramADJourneyDecision(
            journey_id="x",
            outcome=cast(Any, "nope"),
            allowed=False,
            support_badge="local_dry_run",
            reason="r",
            blockers=("b",),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="require blockers"):
        WholeProgramADJourneyDecision(
            journey_id="x",
            outcome="refused",
            allowed=False,
            support_badge="local_dry_run",
            reason="r",
            blockers=(),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        WholeProgramADJourneyDecision(
            journey_id="x",
            outcome="allowed_dry_run",
            allowed=True,
            support_badge="local_dry_run",
            reason="r",
            blockers=("b",),
            steps_completed=("a",),
        )
    with pytest.raises(ValueError, match="must use outcome=allowed_dry_run"):
        WholeProgramADJourneyDecision(
            journey_id="x",
            outcome="refused",
            allowed=True,
            support_badge="local_dry_run",
            reason="r",
            blockers=(),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="must use outcome=refused"):
        WholeProgramADJourneyDecision(
            journey_id="x",
            outcome="allowed_dry_run",
            allowed=False,
            support_badge="local_dry_run",
            reason="r",
            blockers=("b",),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="reason"):
        WholeProgramADJourneyDecision(
            journey_id="x",
            outcome="refused",
            allowed=False,
            support_badge="local_dry_run",
            reason="",
            blockers=("b",),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="blockers entries"):
        WholeProgramADJourneyDecision(
            journey_id="x",
            outcome="refused",
            allowed=False,
            support_badge="local_dry_run",
            reason="r",
            blockers=("ok", "  "),
            steps_completed=(),
        )
    with pytest.raises(ValueError, match="support_badge"):
        WholeProgramADJourneyDecision(
            journey_id="x",
            outcome="refused",
            allowed=False,
            support_badge=cast(Any, "nope"),
            reason="r",
            blockers=("b",),
            steps_completed=(),
        )
    ok = WholeProgramADJourneyDecision(
        journey_id="x",
        outcome="allowed_dry_run",
        allowed=True,
        support_badge="local_dry_run",
        reason="r",
        blockers=(),
        steps_completed=("a",),
    )
    assert ok.to_dict()["allowed"] is True
    journey = get_whole_program_ad_journey("frontend_compile_dry_run")
    assert journey.to_dict()["journey_id"] == "frontend_compile_dry_run"


def test_to_dict_serialisable() -> None:
    """Materialise tuple fields as JSON-ready list values."""
    decision = dry_run_whole_program_ad_journey("adjoint_replay_local_dry_run")
    payload = decision.to_dict()
    assert payload["outcome"] == "allowed_dry_run"
    assert isinstance(payload["steps_completed"], list)
    assert payload["claim_boundary"] == WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY


def test_integrity_rejects_blank_invalid_and_metadata() -> None:
    """Reject malformed rows and inconsistent registry metadata."""
    registry = build_whole_program_ad_product_registry()
    journeys = cast(list[dict[str, object]], registry["journeys"])

    non_map = dict(registry)
    non_map["journeys"] = [cast(Any, "not-a-mapping")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_whole_program_ad_product_integrity(non_map)

    blank_id = dict(registry)
    blank_rows = [dict(row) for row in journeys]
    blank_rows[0]["journey_id"] = "  "
    blank_id["journeys"] = blank_rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_whole_program_ad_product_integrity(blank_id)

    bad_badge = dict(registry)
    badge_rows = [dict(row) for row in journeys]
    badge_rows[1]["support_badge"] = "nope"
    bad_badge["journeys"] = badge_rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_whole_program_ad_product_integrity(bad_badge)

    empty_steps = dict(registry)
    step_rows = [dict(row) for row in journeys]
    step_rows[0]["steps"] = []
    empty_steps["journeys"] = step_rows
    with pytest.raises(ValueError, match="non-empty steps"):
        assert_whole_program_ad_product_integrity(empty_steps)

    no_layer = dict(registry)
    layer_rows = [dict(row) for row in journeys]
    layer_rows[0]["architecture_layer"] = ""
    no_layer["journeys"] = layer_rows
    with pytest.raises(ValueError, match="architecture_layer"):
        assert_whole_program_ad_product_integrity(no_layer)

    no_bl53 = dict(registry)
    bl53_rows = [dict(row) for row in journeys]
    for row in bl53_rows:
        if row.get("journey_id") == "unsupported_frontend_fail_closed":
            row["unsuitable_scenario_pointer"] = ""
    no_bl53["journeys"] = bl53_rows
    with pytest.raises(ValueError, match="unsuitable_scenario_pointer"):
        assert_whole_program_ad_product_integrity(no_bl53)

    no_default = dict(registry)
    renamed = [dict(row) for row in journeys]
    for row in renamed:
        if row.get("journey_id") == "frontend_compile_dry_run":
            row["journey_id"] = "renamed_default"
    no_default["journeys"] = renamed
    with pytest.raises(ValueError, match="missing frontend_compile_dry_run|drift"):
        assert_whole_program_ad_product_integrity(no_default)

    dup = dict(registry)
    dup_rows = [dict(row) for row in journeys]
    dup_rows.append(dict(dup_rows[0]))
    dup["journeys"] = dup_rows
    dup["journey_count"] = len(dup_rows)
    with pytest.raises(ValueError, match="duplicate journey_id"):
        assert_whole_program_ad_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_whole_program_ad_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["journey_count"] = 0
    with pytest.raises(ValueError, match="journey_count"):
        assert_whole_program_ad_product_integrity(count_mismatch)

    no_layers = dict(registry)
    no_layers["architecture_layers"] = []
    with pytest.raises(ValueError, match="architecture_layers"):
        assert_whole_program_ad_product_integrity(no_layers)


def test_catalogue_map_runtime_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise fail-closed catalogue construction (not load-time happy path)."""
    from scpn_quantum_control import whole_program_ad_product as mod

    good = get_whole_program_ad_journey("frontend_compile_dry_run")
    blank = WholeProgramADJourney(
        journey_id="tmp",
        title="t",
        summary="s",
        module_path="m",
        support_badge="local_dry_run",
        steps=("a",),
    )
    # blank journey_id after construction is impossible via __post_init__;
    # inject a mutated object via object.__setattr__ for the guard.
    object.__setattr__(blank, "journey_id", "  ")
    monkeypatch.setattr(mod, "_CANONICAL_JOURNEYS", (blank,))
    with pytest.raises(RuntimeError, match="blank journey_id"):
        mod._catalogue_map()

    a = get_whole_program_ad_journey("frontend_compile_dry_run")
    monkeypatch.setattr(mod, "_CANONICAL_JOURNEYS", (a, a))
    with pytest.raises(RuntimeError, match="duplicate journey_id"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_JOURNEYS", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        mod._catalogue_map()

    # restore is automatic via monkeypatch teardown; re-seed for clarity
    monkeypatch.setattr(mod, "_CANONICAL_JOURNEYS", (good,))
    assert mod._catalogue_map()[good.journey_id].journey_id == good.journey_id


def test_architecture_map_unknown_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept an unknown journey layer without inventing an ordered row."""
    from scpn_quantum_control import whole_program_ad_product as mod

    custom = WholeProgramADJourney(
        journey_id="custom_layer_row",
        title="t",
        summary="s",
        module_path="scpn_quantum_control.custom_layer_mod",
        support_badge="experimental_workbench",
        steps=("a",),
        architecture_layer="custom_extra",
    )
    monkeypatch.setattr(mod, "_CANONICAL_JOURNEYS", (custom,))
    layers = mod.map_whole_program_ad_architecture_layers()
    # custom layer is recorded in journey_ids only when iterating canonical;
    # the fixed order does not include custom_extra as a primary order key,
    # but the unknown-layer branch should still accept it without crash.
    assert isinstance(layers, tuple)


def test_architecture_map_skips_duplicate_module_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second journey with the same module_path does not re-append the path."""
    from scpn_quantum_control import whole_program_ad_product as mod

    first = WholeProgramADJourney(
        journey_id="dup_path_a",
        title="t",
        summary="s",
        module_path="scpn_quantum_control.shared_mod",
        support_badge="local_dry_run",
        steps=("a",),
        architecture_layer="frontend",
    )
    second = WholeProgramADJourney(
        journey_id="dup_path_b",
        title="t2",
        summary="s2",
        module_path="scpn_quantum_control.shared_mod",
        support_badge="local_dry_run",
        steps=("a",),
        architecture_layer="frontend",
    )
    monkeypatch.setattr(mod, "_CANONICAL_JOURNEYS", (first, second))
    layers = mod.map_whole_program_ad_architecture_layers()
    frontend = next(row for row in layers if row["layer"] == "frontend")
    paths = cast(list[str], frontend["module_paths"])
    assert paths.count("scpn_quantum_control.shared_mod") == 1


def test_architecture_map_skips_empty_ir_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty IR ownership list omits the IR layer row entirely."""
    from scpn_quantum_control import whole_program_ad_product as mod

    frontend_only = WholeProgramADJourney(
        journey_id="frontend_only",
        title="t",
        summary="s",
        module_path="scpn_quantum_control.frontend_only",
        support_badge="local_dry_run",
        steps=("a",),
        architecture_layer="frontend",
    )
    monkeypatch.setattr(mod, "_CANONICAL_JOURNEYS", (frontend_only,))
    monkeypatch.setattr(mod, "_IR_LAYER_OWNERSHIP_MODULES", ())
    layers = mod.map_whole_program_ad_architecture_layers()
    assert all(row["layer"] != "ir" for row in layers)


def test_integrity_rejects_allows_hardware_true_on_non_default_journey() -> None:
    """Non-default journeys must also set allows_hardware=False (invent-green refuse)."""
    registry = build_whole_program_ad_product_registry()
    journeys = cast(list[dict[str, object]], registry["journeys"])
    hw = dict(registry)
    hw_journeys = [dict(row) for row in journeys]
    # Prefer a non-default journey so the generic invent-green check fires.
    target = next(
        i
        for i, row in enumerate(hw_journeys)
        if row.get("journey_id") != "frontend_compile_dry_run"
    )
    hw_journeys[target]["allows_hardware"] = True
    hw["journeys"] = hw_journeys
    with pytest.raises(ValueError, match="invent-green hardware"):
        assert_whole_program_ad_product_integrity(hw)
