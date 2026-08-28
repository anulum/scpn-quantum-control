# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for unsuitable-scenario registry
"""Real-surface tests for ``scpn_quantum_control.unsuitable_scenario_registry``."""

from __future__ import annotations

import pytest

import scpn_quantum_control.unsuitable_scenario_registry as unsuitable_scenario_registry
from scpn_quantum_control.unsuitable_scenario_registry import (
    UNSUITABLE_SCENARIO_CLAIM_BOUNDARY,
    UNSUITABLE_SCENARIO_REGISTRY_SCHEMA,
    ScenarioProbeResult,
    UnsuitableScenarioRecord,
    assert_unsuitable_registry_integrity,
    build_unsuitable_scenario_registry,
    get_unsuitable_scenario,
    iter_unsuitable_scenarios,
    list_unsuitable_scenario_ids,
    probe_unsuitable_scenario,
)


def test_list_ids_stable_nonempty_unique() -> None:
    """Keep the public identifier catalogue non-empty, unique, and stable."""
    ids = list_unsuitable_scenario_ids()
    assert ids
    assert len(ids) == len(set(ids))
    assert ids == list_unsuitable_scenario_ids()
    assert all(":" in scenario_id for scenario_id in ids)


def test_get_known_unsuitable_and_anti_silent() -> None:
    """Resolve governed unsuitable and anti-silent rows by public identifier."""
    complex_row = get_unsuitable_scenario("unsuitable:complex.objective_without_wirtinger")
    assert complex_row.kind == "unsuitable_scenario"
    assert complex_row.reason
    assert complex_row.claim_boundary == UNSUITABLE_SCENARIO_CLAIM_BOUNDARY
    assert "transform:unsupported.complex_objective" in complex_row.related_route_ids

    di_jl = get_unsuitable_scenario("anti_silent:differentiation_interface.compiled_tape")
    assert di_jl.kind == "anti_silent_wrong"
    assert "silent" in di_jl.reason.lower()
    assert di_jl.citation
    assert "competitor:differentiation_interface.silent_wrong_grads" in (di_jl.related_route_ids)

    rl = get_unsuitable_scenario("unsuitable:rl.research_without_preregistration")
    assert rl.expected_outcome == "fail_closed_plan"
    assert "preregistration" in rl.trigger.lower()
    assert "research:rl.witness_discovery" in rl.related_route_ids


def test_get_rejects_blank_and_unknown() -> None:
    """Reject blank and unregistered identifiers without inventing support."""
    with pytest.raises(ValueError, match="non-empty"):
        get_unsuitable_scenario("  ")
    with pytest.raises(ValueError, match="unknown unsuitable scenario_id"):
        get_unsuitable_scenario("invent.green.success")


def test_iter_filters_by_kind_and_outcome() -> None:
    """Filter immutable catalogue rows by either supported classification."""
    anti = iter_unsuitable_scenarios(kind="anti_silent_wrong")
    assert anti
    assert all(row.kind == "anti_silent_wrong" for row in anti)

    permanent = iter_unsuitable_scenarios(expected_outcome="permanent_boundary")
    assert permanent
    assert all(row.expected_outcome == "permanent_boundary" for row in permanent)

    both = iter_unsuitable_scenarios(
        kind="unsuitable_scenario",
        expected_outcome="unsupported_transform",
    )
    assert both
    assert all(
        row.kind == "unsuitable_scenario" and row.expected_outcome == "unsupported_transform"
        for row in both
    )


def test_build_registry_zero_blanks_and_schema() -> None:
    """Build a schema-tagged registry with consistent counts and no blanks."""
    registry = build_unsuitable_scenario_registry()
    scenarios = registry["scenarios"]
    scenario_count = registry["scenario_count"]
    unsuitable_count = registry["unsuitable_scenario_count"]
    anti_silent_count = registry["anti_silent_wrong_count"]
    assert isinstance(scenarios, list)
    assert isinstance(scenario_count, int)
    assert isinstance(unsuitable_count, int)
    assert isinstance(anti_silent_count, int)
    assert registry["schema"] == UNSUITABLE_SCENARIO_REGISTRY_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert scenario_count == len(scenarios)
    assert unsuitable_count + anti_silent_count == scenario_count
    validated = assert_unsuitable_registry_integrity(registry)
    assert validated["blank_entry_count"] == 0
    for row in scenarios:
        assert isinstance(row, dict)
        assert row["kind"] in {"unsuitable_scenario", "anti_silent_wrong"}
        assert row["reason"]
        assert row["expected_error"]


def test_assert_integrity_rejects_invalid_payloads() -> None:
    """Reject malformed rows, invalid classifications, and count drift."""
    with pytest.raises(ValueError, match="non-empty scenarios"):
        assert_unsuitable_registry_integrity({"scenarios": []})
    with pytest.raises(ValueError, match="blank"):
        assert_unsuitable_registry_integrity(
            {
                "scenarios": [{"scenario_id": "", "kind": "unsuitable_scenario"}],
                "blank_entry_count": 0,
                "scenario_count": 1,
            }
        )
    with pytest.raises(ValueError, match="missing reason"):
        assert_unsuitable_registry_integrity(
            {
                "scenarios": [
                    {
                        "scenario_id": "x",
                        "kind": "unsuitable_scenario",
                        "reason": "",
                        "expected_error": "e",
                    }
                ],
                "blank_entry_count": 0,
                "scenario_count": 1,
            }
        )
    with pytest.raises(ValueError, match="blank_entry_count"):
        good = get_unsuitable_scenario("unsuitable:complex.objective_without_wirtinger").to_dict()
        assert_unsuitable_registry_integrity(
            {
                "scenarios": [good],
                "blank_entry_count": 1,
                "scenario_count": 1,
            }
        )
    with pytest.raises(ValueError, match="scenario_count"):
        assert_unsuitable_registry_integrity(
            {
                "scenarios": [good],
                "blank_entry_count": 0,
                "scenario_count": 99,
            }
        )
    with pytest.raises(ValueError, match="mapping"):
        assert_unsuitable_registry_integrity(
            {
                "scenarios": ["not-a-mapping"],
                "blank_entry_count": 0,
                "scenario_count": 1,
            }
        )
    with pytest.raises(ValueError, match="blank"):
        assert_unsuitable_registry_integrity(
            {
                "scenarios": [
                    {
                        "scenario_id": "x",
                        "kind": "not-a-kind",
                        "reason": "r",
                        "expected_error": "e",
                    }
                ],
                "blank_entry_count": 0,
                "scenario_count": 1,
            }
        )


def test_probe_known_unsuitable_refuses_with_reason() -> None:
    """Return a reasoned refusal for a known unsuitable scenario."""
    result = probe_unsuitable_scenario("unsuitable:complex.objective_without_wirtinger")
    assert isinstance(result, ScenarioProbeResult)
    assert result.refused is True
    assert result.selected.scenario_id == ("unsuitable:complex.objective_without_wirtinger")
    assert "Wirtinger" in result.message or "wirtinger" in result.message.lower()
    assert result.selected.related_route_ids
    payload = result.to_dict()
    assert payload["refused"] is True
    assert payload["claim_boundary"] == UNSUITABLE_SCENARIO_CLAIM_BOUNDARY


def test_probe_rl_without_preregistration_refuses() -> None:
    """Carry the no-preregistration research refusal explicitly."""
    result = probe_unsuitable_scenario("unsuitable:rl.research_without_preregistration")
    assert result.refused
    assert result.selected.expected_error == (
        "RLResearchGovernanceError:preregistration_id_missing"
    )
    assert "fixed seeds" in result.selected.reason


def test_probe_competitor_anti_silent_fixtures() -> None:
    """Expose competitor silent-wrong boundaries as explicit refusals."""
    di_jl = probe_unsuitable_scenario("anti_silent:differentiation_interface.compiled_tape")
    assert di_jl.refused is True
    assert di_jl.selected.kind == "anti_silent_wrong"
    assert any("anti-silent-wrong" in note for note in di_jl.notes)
    assert any("related_route_ids=" in note for note in di_jl.notes)

    catalyst = probe_unsuitable_scenario("anti_silent:catalyst.qjit_vmap_quantum")
    assert catalyst.refused is True
    assert catalyst.selected.expected_outcome == "permanent_boundary"
    assert "vmap" in catalyst.selected.reason.lower() or "batch" in (
        catalyst.selected.reason.lower()
    )

    adaptive_shots = probe_unsuitable_scenario("anti_silent:catalyst.no_broadcast_adaptive_shots")
    assert adaptive_shots.refused is True
    assert adaptive_shots.selected.citation == (
        "Catalyst adaptive finite-shot trainability boundary"
    )


def test_probe_unknown_fail_closed_policies() -> None:
    """Fail closed for unknown identifiers under both public policies."""
    with pytest.raises(ValueError, match="unknown unsuitable scenario_id"):
        probe_unsuitable_scenario("no.such.scenario")

    boundary = probe_unsuitable_scenario(
        "no.such.scenario",
        unknown_policy="boundary",
    )
    assert boundary.refused is True
    assert boundary.selected.scenario_id.startswith("unknown:")
    assert "not in the unsuitable catalogue" in boundary.selected.reason


def test_probe_rejects_blank_and_bad_unknown_policy() -> None:
    """Reject blank identifiers and unsupported unknown-ID policies."""
    with pytest.raises(ValueError, match="non-empty"):
        probe_unsuitable_scenario("")
    with pytest.raises(ValueError, match="unknown_policy"):
        probe_unsuitable_scenario(
            "missing",
            unknown_policy="invent",  # type: ignore[arg-type]
        )


def test_record_validation_edge_paths() -> None:
    """Enforce every immutable catalogue-record invariant."""
    with pytest.raises(ValueError, match="scenario_id"):
        UnsuitableScenarioRecord(
            scenario_id="",
            kind="unsuitable_scenario",
            trigger="t",
            expected_outcome="raise_value_error",
            expected_error="e",
            reason="r",
            evidence=("e",),
        )
    with pytest.raises(ValueError, match="unknown scenario kind"):
        UnsuitableScenarioRecord(
            scenario_id="x:y",
            kind="not_kind",  # type: ignore[arg-type]
            trigger="t",
            expected_outcome="raise_value_error",
            expected_error="e",
            reason="r",
            evidence=("e",),
        )
    with pytest.raises(ValueError, match="unknown expected_outcome"):
        UnsuitableScenarioRecord(
            scenario_id="x:y",
            kind="unsuitable_scenario",
            trigger="t",
            expected_outcome="maybe",  # type: ignore[arg-type]
            expected_error="e",
            reason="r",
            evidence=("e",),
        )
    with pytest.raises(ValueError, match="trigger"):
        UnsuitableScenarioRecord(
            scenario_id="x:y",
            kind="unsuitable_scenario",
            trigger="  ",
            expected_outcome="raise_value_error",
            expected_error="e",
            reason="r",
            evidence=("e",),
        )
    with pytest.raises(ValueError, match="expected_error"):
        UnsuitableScenarioRecord(
            scenario_id="x:y",
            kind="unsuitable_scenario",
            trigger="t",
            expected_outcome="raise_value_error",
            expected_error="",
            reason="r",
            evidence=("e",),
        )
    with pytest.raises(ValueError, match="reason"):
        UnsuitableScenarioRecord(
            scenario_id="x:y",
            kind="unsuitable_scenario",
            trigger="t",
            expected_outcome="raise_value_error",
            expected_error="e",
            reason="",
            evidence=("e",),
        )
    with pytest.raises(ValueError, match="evidence"):
        UnsuitableScenarioRecord(
            scenario_id="x:y",
            kind="unsuitable_scenario",
            trigger="t",
            expected_outcome="raise_value_error",
            expected_error="e",
            reason="r",
            evidence=("",),
        )
    with pytest.raises(ValueError, match="related_route_ids"):
        UnsuitableScenarioRecord(
            scenario_id="x:y",
            kind="unsuitable_scenario",
            trigger="t",
            expected_outcome="raise_value_error",
            expected_error="e",
            reason="r",
            evidence=("e",),
            related_route_ids=("  ",),
        )


def test_probe_result_validation() -> None:
    """Forbid blank, green, or message-free probe results."""
    selected = get_unsuitable_scenario("unsuitable:hardware.gradient_without_ticket")
    with pytest.raises(ValueError, match="scenario_id must be non-empty"):
        ScenarioProbeResult(
            scenario_id="",
            refused=True,
            selected=selected,
            message="refused",
        )
    with pytest.raises(ValueError, match="refused must be True"):
        ScenarioProbeResult(
            scenario_id=selected.scenario_id,
            refused=False,
            selected=selected,
            message="refused",
        )
    with pytest.raises(ValueError, match="message"):
        ScenarioProbeResult(
            scenario_id=selected.scenario_id,
            refused=True,
            selected=selected,
            message="  ",
        )


def test_catalogue_map_rejects_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject duplicate identifiers while constructing the catalogue map."""
    row = get_unsuitable_scenario("unsuitable:rust.dynamic_axes_replay")
    monkeypatch.setattr(
        unsuitable_scenario_registry,
        "_CANONICAL_SCENARIOS",
        (row, row),
    )
    with pytest.raises(RuntimeError, match="duplicate scenario_id"):
        unsuitable_scenario_registry._catalogue_map()


def test_record_to_dict_fields() -> None:
    """Serialize a catalogue row into the complete public mapping shape."""
    row = get_unsuitable_scenario("unsuitable:pennylane.hardware_plugin_gradient")
    payload = row.to_dict()
    assert payload["scenario_id"] == row.scenario_id
    assert payload["kind"] == "unsuitable_scenario"
    assert isinstance(payload["evidence"], list)
    assert payload["claim_boundary"] == UNSUITABLE_SCENARIO_CLAIM_BOUNDARY


def test_probe_rows_without_related_routes_omit_route_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omit route notes when neither synthetic nor known rows provide routes."""
    # unknown boundary path
    result = probe_unsuitable_scenario("blank.x", unknown_policy="boundary")
    assert result.refused is True
    assert not any(note.startswith("related_route_ids=") for note in result.notes)

    # known unsuitable entry with empty related_route_ids (branch coverage)
    base = get_unsuitable_scenario("unsuitable:torch.fullgraph_compile_unregistered")
    from dataclasses import replace

    patched = replace(base, related_route_ids=())
    mapping = dict(unsuitable_scenario_registry._SCENARIO_BY_ID)
    mapping[patched.scenario_id] = patched
    monkeypatch.setattr(unsuitable_scenario_registry, "_SCENARIO_BY_ID", mapping)
    known = probe_unsuitable_scenario(patched.scenario_id)
    assert known.refused is True
    assert not any(note.startswith("related_route_ids=") for note in known.notes)
