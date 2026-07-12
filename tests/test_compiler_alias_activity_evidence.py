# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — compiler alias activity evidence tests
# scpn-quantum-control -- compiler alias-activity evidence tests
"""Tests for compiler alias-activity evidence assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.compiler import (
    CompilerAliasActivityCase,
    CompilerAliasActivityEvidence,
    build_compiler_alias_activity_evidence,
    render_compiler_alias_activity_evidence_markdown,
)


def test_compiler_alias_activity_evidence_builds_real_lattice_cases() -> None:
    """Alias-activity evidence must be assembled from real Program AD reports."""
    evidence = build_compiler_alias_activity_evidence(source_commit="test-commit")

    payload = evidence.as_dict()
    statuses = {case.status for case in evidence.cases}

    assert payload["schema"] == "scpn_qc_compiler_alias_activity_evidence_v1"
    assert payload["artifact_id"] == "compiler-alias-activity-evidence-20260706"
    assert payload["source_commit"] == "test-commit"
    assert payload["alias_activity_verified"] is True
    assert payload["promotion_ready"] is False
    assert evidence.classification == "functional_non_isolated"
    assert statuses == {"blocked_lattice", "complete_lattice"}
    assert evidence.complete_lattice_case_count >= 4
    assert evidence.blocked_lattice_case_count >= 3
    assert set(evidence.observed_alias_edge_kinds) >= {
        "control_path_alias",
        "expression_rebinding_alias",
        "list_alias",
        "local_rebinding_alias",
        "loop_carried_state",
        "mutation_version",
        "object_attribute_alias",
        "view_alias",
    }
    assert (
        "tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_reports_complete_emitted_ir"
        in (evidence.test_ids)
    )
    assert "isolated benchmark" in evidence.claim_boundary
    assert "provider, hardware, GPU, or performance claim" in evidence.claim_boundary


def test_compiler_alias_activity_markdown_lists_blockers() -> None:
    """Markdown rendering must expose alias blockers and non-promotion status."""
    evidence = build_compiler_alias_activity_evidence(source_commit="test-commit")

    markdown = render_compiler_alias_activity_evidence_markdown(evidence)

    assert "# Compiler Alias-Activity Evidence" in markdown
    assert "promotion_ready: `False`" in markdown
    assert "`control_path_alias`" in markdown
    assert "non_executed_phi_inputs_require_branch_semantics" in markdown
    assert "Claim boundary:" in markdown


def test_compiler_alias_activity_case_rejects_inconsistent_status() -> None:
    """Case validation must reject promoted statuses and empty alias metadata."""
    with pytest.raises(ValueError, match="status is unsupported"):
        CompilerAliasActivityCase(
            case_id="bad",
            status="promoted",
            complete=True,
            alias_edge_kinds=("view_alias",),
            blocker_reasons=(),
            component_count=1,
            provenance_counts={"view_alias": 1},
        )

    with pytest.raises(ValueError, match="alias_edge_kinds"):
        CompilerAliasActivityCase(
            case_id="bad",
            status="complete_lattice",
            complete=True,
            alias_edge_kinds=(),
            blocker_reasons=(),
            component_count=1,
            provenance_counts={"view_alias": 1},
        )


def test_compiler_alias_activity_case_rejects_runtime_type_drift() -> None:
    """Case validation must fail closed on dynamic evidence type drift."""
    valid_fields: dict[str, object] = {
        "case_id": "complete_view_alias",
        "status": "complete_lattice",
        "complete": True,
        "alias_edge_kinds": ("view_alias",),
        "blocker_reasons": (),
        "component_count": 1,
        "provenance_counts": {"view_alias": 1},
    }
    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"case_id": "   "}, "case_id"),
        ({"component_count": cast(int, True)}, "component_count"),
        ({"provenance_counts": cast(dict[str, int], [("view_alias", 1)])}, "provenance_counts"),
        ({"provenance_counts": {"view_alias": cast(int, True)}}, "provenance_counts"),
    )

    for overrides, match in invalid_cases:
        fields = dict(valid_fields)
        fields.update(overrides)
        with pytest.raises(ValueError, match=match):
            CompilerAliasActivityCase(**fields)  # type: ignore[arg-type]  # runtime drift


def test_compiler_alias_activity_evidence_rejects_missing_case_families() -> None:
    """Evidence validation must require complete and blocked lattice cases."""
    case = CompilerAliasActivityCase(
        case_id="only-complete",
        status="complete_lattice",
        complete=True,
        alias_edge_kinds=("view_alias",),
        blocker_reasons=(),
        component_count=1,
        provenance_counts={"view_alias": 1},
    )

    with pytest.raises(ValueError, match="blocked lattice case"):
        CompilerAliasActivityEvidence(
            source_commit="test",
            cases=(case,),
            test_ids=("tests/test_program_ad_alias_effects.py::case",),
        )


def test_compiler_alias_activity_evidence_rejects_public_container_drift() -> None:
    """Evidence validation must require immutable cases and focused test nodes."""
    evidence = build_compiler_alias_activity_evidence(source_commit="test-commit")

    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"source_commit": "   "}, "source_commit"),
        ({"cases": cast(tuple[CompilerAliasActivityCase, ...], list(evidence.cases))}, "cases"),
        ({"test_ids": ()}, "test_ids"),
    )

    for overrides, match in invalid_cases:
        fields: dict[str, object] = {
            "source_commit": "test-commit",
            "cases": evidence.cases,
            "test_ids": evidence.test_ids,
        }
        fields.update(overrides)
        with pytest.raises(ValueError, match=match):
            CompilerAliasActivityEvidence(**fields)  # type: ignore[arg-type]  # runtime drift


def test_committed_compiler_alias_activity_evidence_matches_builder() -> None:
    """Committed alias-activity JSON must match the current public builder."""
    committed = json.loads(
        Path(
            "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.json"
        ).read_text(encoding="utf-8")
    )

    assert committed == build_compiler_alias_activity_evidence(source_commit="534010da").as_dict()
