# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-84 research-lane registry tests
"""Tests for the governed deep-analysis catalogue and inventory gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

import scpn_quantum_control.analysis.research_lane_registry as registry
from scpn_quantum_control.analysis import (
    RESEARCH_LANE_REGISTRY_BOUNDARY,
    RESEARCH_LANE_REGISTRY_SCHEMA,
    ResearchLaneClaimStatus,
    ResearchLaneDiffHook,
    ResearchLaneInventoryReport,
    ResearchLaneMaturity,
    ResearchLaneRecord,
    ResearchLaneRegistryReport,
    assert_research_lane_inventory,
    build_research_lane_registry_report,
    discover_research_lane_modules,
    get_research_lane,
    list_research_lanes,
    render_research_lane_registry_markdown,
    validate_research_lane_inventory,
)

ROOT = Path(__file__).resolve().parents[1]


def _sample_record() -> ResearchLaneRecord:
    """Return a valid bounded-composition row for mutation tests."""
    return ResearchLaneRecord(
        module="scpn_quantum_control.analysis.sample",
        summary="Bounded sample diagnostic.",
        maturity=ResearchLaneMaturity.PRODUCT_CANDIDATE,
        diff_hook=ResearchLaneDiffHook.BOUNDED_COMPOSITION,
        claim_status=ResearchLaneClaimStatus.EVIDENCE_BOUNDED,
        promotion_targets=("BL-X:complete",),
        evidence_refs=("data/sample.json",),
    )


def _load_runner() -> ModuleType:
    """Load the deterministic runner without changing import paths."""
    path = ROOT / "scripts/run_research_lane_registry.py"
    spec = importlib.util.spec_from_file_location("research_lane_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registry_covers_every_analysis_and_gauge_module_exactly_once() -> None:
    """The reviewed registry and current package inventory are identical."""
    records = list_research_lanes()
    report = assert_research_lane_inventory()

    assert len(records) == 73
    assert tuple(record.module for record in records) == tuple(
        sorted(record.module for record in records)
    )
    assert len({record.module for record in records}) == len(records)
    assert report.passed
    assert report.missing_modules == ()
    assert report.orphaned_records == ()
    assert len(report.registered_modules) == len(report.discovered_modules) == 73


def test_registry_is_non_promotional_and_has_human_classifications() -> None:
    """Every row exposes required BL-84 fields and explicit negative grants."""
    records = list_research_lanes()

    assert {record.family for record in records} == {"analysis", "gauge"}
    assert {record.maturity for record in records} == set(ResearchLaneMaturity)
    assert {record.diff_hook for record in records} == set(ResearchLaneDiffHook)
    assert {record.claim_status for record in records} == set(ResearchLaneClaimStatus)
    assert all(record.summary for record in records)
    assert all(not record.registry_grants_productisation for record in records)
    assert all(not record.registry_grants_control for record in records)
    assert all(not record.registry_grants_publication_claim for record in records)


def test_selected_promotion_routes_retain_their_real_gate_status() -> None:
    """BL-50/54/72/80 links distinguish planned, complete, and deferred work."""
    qfi = get_research_lane("scpn_quantum_control.analysis.qfi")
    dla = get_research_lane("scpn_quantum_control.analysis.dla_parity_theorem")
    tcbo = get_research_lane("scpn_quantum_control.analysis.tcbo_weighted_complex")
    fim = get_research_lane("scpn_quantum_control.analysis.adaptive_fim_feedback")

    assert qfi.promotion_targets == ("BL-50:planned",)
    assert qfi.diff_hook is ResearchLaneDiffHook.CANDIDATE
    assert dla.promotion_targets == ("BL-54:complete",)
    assert dla.claim_status is ResearchLaneClaimStatus.EVIDENCE_BOUNDED
    assert tcbo.promotion_targets == ("BL-72:deferred-owner-gate",)
    assert tcbo.diff_hook is ResearchLaneDiffHook.DEFERRED
    assert fim.promotion_targets == ("BL-80:complete",)
    assert fim.diff_hook is ResearchLaneDiffHook.BOUNDED_COMPOSITION


def test_every_evidence_pointer_resolves_inside_the_repository() -> None:
    """Bounded rows never cite imaginary local custody artefacts."""
    pointers = {
        reference for record in list_research_lanes() for reference in record.evidence_refs
    }
    assert pointers
    assert all((ROOT / pointer).is_file() for pointer in pointers)


def test_record_serialization_includes_required_schema_fields_and_denials() -> None:
    """Machine output carries the four design fields and explicit non-grants."""
    payload = _sample_record().as_dict()

    assert payload["family"] == "analysis"
    assert payload["maturity"] == "product_candidate"
    assert payload["diff_hook"] == "bounded_composition"
    assert payload["claim_status"] == "evidence_bounded"
    assert payload["registry_grants_productisation"] is False
    assert payload["registry_grants_control"] is False
    assert payload["registry_grants_publication_claim"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("module", "other.sample", "analysis or gauge"),
        (
            "module",
            "scpn_quantum_control.analysis.research_lane_registry",
            "must not catalogue itself",
        ),
        ("summary", " ", "summary must be non-empty"),
        ("promotion_targets", ("",), "must not contain blank"),
        ("evidence_refs", ("same", "same"), "must not contain duplicates"),
    ],
)
def test_record_rejects_invalid_identity_and_sequences(
    field: str, value: object, message: str
) -> None:
    """Blank, duplicate, foreign, and self-referential rows fail closed."""
    with pytest.raises(ValueError, match=message):
        replace(_sample_record(), **{field: value})  # type: ignore[arg-type]


def test_record_rejects_inconsistent_claim_and_diff_states() -> None:
    """Bounded, composed, deferred, and refusal states require their gates."""
    with pytest.raises(ValueError, match="require evidence_refs"):
        replace(_sample_record(), evidence_refs=())
    with pytest.raises(ValueError, match="require a promotion target"):
        replace(_sample_record(), promotion_targets=())
    with pytest.raises(ValueError, match="require a promotion target"):
        replace(
            _sample_record(),
            diff_hook=ResearchLaneDiffHook.DEFERRED,
            promotion_targets=(),
        )
    with pytest.raises(ValueError, match="explicit non-research"):
        replace(
            _sample_record(),
            maturity=ResearchLaneMaturity.RESEARCH,
            diff_hook=ResearchLaneDiffHook.NONE,
            claim_status=ResearchLaneClaimStatus.REFUSE_ONLY,
        )


def test_unknown_lane_lookup_fails_closed() -> None:
    """Lookup never fabricates a permissive default row."""
    with pytest.raises(KeyError, match="unregistered research lane"):
        get_research_lane("scpn_quantum_control.analysis.missing")


def test_discovery_excludes_package_and_governance_modules(tmp_path: Path) -> None:
    """Discovery includes ordinary modules and only the documented exclusions."""
    for family in ("analysis", "gauge"):
        directory = tmp_path / family
        directory.mkdir()
        (directory / "__init__.py").write_text("", encoding="utf-8")
        (directory / f"{family}_lane.py").write_text("", encoding="utf-8")
    (tmp_path / "analysis/research_lane_registry.py").write_text("", encoding="utf-8")

    assert discover_research_lane_modules(tmp_path) == (
        "scpn_quantum_control.analysis.analysis_lane",
        "scpn_quantum_control.gauge.gauge_lane",
    )


@pytest.mark.parametrize("missing_family", ["analysis", "gauge"])
def test_discovery_requires_both_package_directories(tmp_path: Path, missing_family: str) -> None:
    """A partial package cannot produce invent-green inventory evidence."""
    present = "gauge" if missing_family == "analysis" else "analysis"
    (tmp_path / present).mkdir()
    with pytest.raises(FileNotFoundError, match="package directory is missing"):
        discover_research_lane_modules(tmp_path)


def test_inventory_report_identifies_new_and_orphaned_modules() -> None:
    """Drift evidence names both unreviewed source and stale registry rows."""
    registered = tuple(record.module for record in list_research_lanes())
    discovered = (*registered[1:], "scpn_quantum_control.analysis.new_lane")
    report = validate_research_lane_inventory((*discovered, discovered[-1]))
    payload = report.as_dict()

    assert not report.passed
    assert report.missing_modules == ("scpn_quantum_control.analysis.new_lane",)
    assert report.orphaned_records == (registered[0],)
    assert payload["passed"] is False
    assert payload["registered_count"] == payload["discovered_count"] == 73
    with pytest.raises(RuntimeError, match="registry drift"):
        assert_research_lane_inventory(discovered)


def test_report_is_digest_locked_complete_and_json_ready() -> None:
    """Aggregate output is stable, complete, and tied to passing discovery."""
    first = build_research_lane_registry_report()
    second = build_research_lane_registry_report()
    payload = first.as_dict()

    assert isinstance(first, ResearchLaneRegistryReport)
    assert first.schema == RESEARCH_LANE_REGISTRY_SCHEMA
    assert first.claim_boundary == RESEARCH_LANE_REGISTRY_BOUNDARY
    assert first.content_digest == second.content_digest
    assert len(first.content_digest) == 64
    assert payload["record_count"] == 73
    assert sum(payload["maturity_counts"].values()) == 73
    assert sum(payload["diff_hook_counts"].values()) == 73
    assert sum(payload["claim_status_counts"].values()) == 73
    assert payload["inventory"]["passed"] is True
    json.dumps(payload, sort_keys=True)


def test_markdown_renderer_exposes_gates_without_marketing_promotion() -> None:
    """Human evidence includes all routes, counts, and the global boundary."""
    report = build_research_lane_registry_report()
    markdown = render_research_lane_registry_markdown(report)
    implicit = render_research_lane_registry_markdown()

    assert markdown == implicit
    assert markdown.endswith("\n")
    assert "Inventory: **PASS** (73 registered / 73 discovered)" in markdown
    assert "BL-50:planned" in markdown
    assert "BL-54:complete" in markdown
    assert "BL-72:deferred-owner-gate" in markdown
    assert "BL-80:complete" in markdown
    assert RESEARCH_LANE_REGISTRY_BOUNDARY in markdown


def test_enum_counter_rejects_non_enum_record_fields() -> None:
    """Internal report aggregation fails closed on a non-enum field name."""
    with pytest.raises(TypeError, match="not an enum-valued"):
        registry._enum_counts(list_research_lanes(), "summary")


def test_manual_inventory_report_serializes_a_passing_empty_fixture() -> None:
    """Inventory record semantics remain usable without filesystem discovery."""
    report = ResearchLaneInventoryReport((), (), (), ())
    assert report.passed
    assert report.as_dict() == {
        "passed": True,
        "registered_count": 0,
        "discovered_count": 0,
        "registered_modules": [],
        "discovered_modules": [],
        "missing_modules": [],
        "orphaned_records": [],
    }


def test_runner_writes_and_checks_exact_deterministic_bytes(tmp_path: Path) -> None:
    """The writer and check mode share one canonical byte representation."""
    runner = _load_runner()
    json_path = tmp_path / "nested/evidence.json"
    markdown_path = tmp_path / "nested/evidence.md"

    assert runner.main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    expected_json, expected_markdown = runner._expected_bytes()
    assert json_path.read_bytes() == expected_json
    assert markdown_path.read_bytes() == expected_markdown
    assert (
        runner.main(["--json", str(json_path), "--markdown", str(markdown_path), "--check"]) == 0
    )


def test_runner_check_mode_rejects_missing_and_stale_outputs(tmp_path: Path) -> None:
    """Committed evidence cannot be absent or differ by even one byte."""
    runner = _load_runner()
    json_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "evidence.md"
    with pytest.raises(SystemExit, match="missing research-lane evidence"):
        runner.main(["--json", str(json_path), "--markdown", str(markdown_path), "--check"])
    assert runner.main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    json_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="stale research-lane evidence"):
        runner.main(["--json", str(json_path), "--markdown", str(markdown_path), "--check"])
