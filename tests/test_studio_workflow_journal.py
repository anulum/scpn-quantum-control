# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original workflow checkpoint public tests
"""Admit actual compiler records and retain explicit incomplete graph history."""

from __future__ import annotations

import platform
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.studio.executive import ActionRegistry, ExecutiveRequest, run_action
from scpn_quantum_control.studio.executive_compile import CompileActionHandler
from scpn_quantum_control.studio.workflow_contracts import WorkflowDefinition, parse_workflow
from scpn_quantum_control.studio.workflow_journal import (
    MAX_JOURNAL_BYTES,
    create_workflow_journal,
    parse_workflow_journal,
)
from scpn_quantum_control.studio.workflow_sweep import build_workflow_cells
from scpn_quantum_control.studio_workspace.canonical import canonical_bytes, canonical_digest
from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json


def original_definition() -> WorkflowDefinition:
    """Read the shared original compiler graph metadata.

    Returns
    -------
    WorkflowDefinition
        Exact admitted synthetic metadata fixture, with no execution claim.

    """
    corpus = cast(
        dict[str, object],
        read_json(
            (Path(__file__).parent / "data/studio_workflow/contract_cases.json").read_text()
        ),
    )
    return parse_workflow(corpus["workflow"])


def original_checkpoint() -> tuple[WorkflowDefinition, dict[str, object]]:
    """Create a partial checkpoint containing a real original compiler action.

    Returns
    -------
    tuple[WorkflowDefinition, dict[str, object]]
        Current source definition and genuine emitted-not-executed compiler
        record. Remaining cells/stages have not executed.

    """
    definition = original_definition()
    cell = build_workflow_cells(definition)[0]
    source = next(stage for stage in definition.stages if stage.id == "source")
    registry = ActionRegistry()
    registry.register(CompileActionHandler())
    request = ExecutiveRequest(
        verb=source.verb,
        action_id="journal-original-source",
        parameters=source.parameters,
        backend=source.backend,
    )
    record = run_action(request, registry=registry)
    assert record.result.status == "succeeded"
    assert record.result.outputs["execution_status"] == "emitted_not_executed"
    journal = create_workflow_journal(
        definition,
        source_fingerprint=canonical_digest("fixture-source.v1", source.parameters),
        runtime_fingerprint=canonical_digest(
            "fixture-runtime.v1", {"python": platform.python_version()}
        ),
    )
    wire = journal.to_dict()
    body = cast(dict[str, object], wire["body"])
    output = record.to_dict()
    body["entries"] = [
        {
            "cell_id": cell.id,
            "stage_id": "source",
            "fingerprint": canonical_digest("fixture-stage.v1", record.plan.to_dict()),
            "dependencies": {},
            "status": "complete",
            "output": output,
            "output_digest": canonical_digest("studio.workflow-output.v1", output),
            "reason": None,
            "evaluated": True,
        }
    ]
    body["evaluations"] = 1
    return definition, wire


def test_actual_original_record_roundtrips_with_explicit_partial_state() -> None:
    """Retain the genuine original record without inventing completion of the graph."""
    definition, wire = original_checkpoint()
    parsed = parse_workflow_journal(wire, definition)
    assert parsed.state == "partial" and parsed.evaluations == 1
    assert parsed.entries[0].status == "complete"
    assert canonical_bytes(
        "journal-fixture.v1",
        parse_workflow_journal(read_json(write_json(parsed.to_dict())), definition).to_dict(),
    ) == canonical_bytes("journal-fixture.v1", wire)


@pytest.mark.parametrize(
    "fault", ["output", "premature-complete", "duplicate", "count", "version", "state"]
)
def test_original_checkpoint_rejects_tamper_or_false_completion(fault: str) -> None:
    """Refuse altered actual producer data or unsupported completion claims.

    Parameters
    ----------
    fault
        Concrete mutation of a checkpoint originally produced by a real handler.

    """
    definition, wire = original_checkpoint()
    body = cast(dict[str, object], wire["body"])
    entries = cast(list[dict[str, object]], body["entries"])
    if fault == "output":
        cast(dict[str, object], entries[0]["output"])["digest"] = "changed"
    elif fault == "premature-complete":
        body["state"] = "complete"
    elif fault == "duplicate":
        entries.append(entries[0])
        body["evaluations"] = 2
    elif fault == "count":
        body["evaluations"] = 0
    elif fault == "version":
        wire["schema"] = "experiment_workflow_journal.v2"
    else:
        body["state"] = {}
    with pytest.raises(ValueError):
        parse_workflow_journal(wire, definition)


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "runtime",
        "fields",
        "rows",
        "cell",
        "stage",
        "parents",
        "status",
        "reason",
        "evaluated",
        "output-absent",
        "extensions-array",
        "empty-reason",
        "long-reason",
        "absent-output-digest",
    ],
)
def test_actual_checkpoint_corruption_matches_browser_admission(fault: str) -> None:
    """Refuse the browser's same original record-corruption cases without mutation.

    Parameters
    ----------
    fault
        A malformed field in a checkpoint containing an actual compiler producer record.

    """
    definition, wire = original_checkpoint()
    body = cast(dict[str, object], wire["body"])
    entry = cast(list[dict[str, object]], body["entries"])[0]
    if fault == "source":
        body["source_fingerprint"] = "unavailable"
    elif fault == "runtime":
        body["runtime_fingerprint"] = "unavailable"
    elif fault == "fields":
        entry["approved"] = True
    elif fault == "rows":
        body["entries"] = {}
    elif fault == "cell":
        entry["cell_id"] = "absent"
    elif fault == "stage":
        entry["stage_id"] = 0
    elif fault == "parents":
        entry["dependencies"] = {"missing": None}
    elif fault == "status":
        entry["status"] = "succeeded"
    elif fault == "reason":
        entry["reason"] = "failed despite completion"
    elif fault == "evaluated":
        entry["evaluated"] = False
    elif fault == "output-absent":
        entry["output"] = None
    elif fault == "extensions-array":
        wire["extensions"] = []
    else:
        entry.update(
            status="failed",
            reason=""
            if fault == "empty-reason"
            else "x" * 2049
            if fault == "long-reason"
            else "original failure",
            output=None if fault == "absent-output-digest" else entry["output"],
        )
    before = write_json(wire)
    with pytest.raises(ValueError):
        parse_workflow_journal(wire, definition)
    assert write_json(wire) == before


def test_actual_execution_journal_preserves_complete_dependencies_and_recovery_history() -> None:
    """Roundtrip the original executive loop's actual complete twelve-stage history."""
    from scpn_quantum_control.studio.workflow_execution import run_workflow

    definition = original_definition()
    journal = run_workflow(definition)
    assert journal.state == "complete" and len(journal.entries) == 12
    admitted = parse_workflow_journal(read_json(write_json(journal.to_dict())), definition)
    assert canonical_bytes("journal-complete.v1", admitted.to_dict()) == canonical_bytes(
        "journal-complete.v1", journal.to_dict()
    )


def test_original_journal_utf8_byte_bound_matches_browser_admission() -> None:
    """Refuse oversized Unicode diagnostics attached to a genuine original record."""
    definition, wire = original_checkpoint()
    cast(dict[str, object], wire["extensions"])["source_diagnostic"] = "😀" * (
        MAX_JOURNAL_BYTES // 4
    )
    with pytest.raises(ValueError, match="workflow journal exceeds byte bound"):
        parse_workflow_journal(wire, definition)


def test_original_journal_refuses_more_than_the_supported_attempt_count() -> None:
    """Refuse an oversized imported history before accepting duplicate producer records."""
    definition, wire = original_checkpoint()
    body = cast(dict[str, object], wire["body"])
    body["entries"] = cast(list[dict[str, object]], body["entries"]) * 8193
    with pytest.raises(ValueError, match="bounded original journal entries required"):
        parse_workflow_journal(wire, definition)


def test_original_failed_parent_and_unevaluated_child_remain_visible() -> None:
    """Roundtrip actual executive input refusals and their unevaluated blocked children."""
    from scpn_quantum_control.studio.workflow_execution import run_workflow

    definition = original_definition()
    axes = (
        ("source", "program_source", ("unsupported source a", "unsupported source b")),
        definition.sweep.axes[1],
    )
    definition = parse_workflow(
        replace(definition, sweep=replace(definition.sweep, axes=axes)).to_dict()
    )
    original = run_workflow(definition)
    admitted = parse_workflow_journal(original.to_dict(), definition)
    assert admitted.state == "partial" and admitted.evaluations == 6
    assert [entry.status for entry in admitted.entries] == ["failed", "blocked"] * 6
    assert all(entry.evaluated is False for entry in admitted.entries[1::2])


def test_actual_retried_parent_invalidates_retained_completed_child() -> None:
    """Refuse false completion after a real changed host approval consumes the last unit."""
    from scpn_quantum_control.studio.workflow_execution import run_workflow

    definition = original_definition()
    definition = parse_workflow(
        replace(
            definition,
            sweep=replace(definition.sweep, axes=(), evaluation_budget=3),
        ).to_dict()
    )
    first = run_workflow(definition)
    assert first.state == "complete" and first.evaluations == 2
    retried = run_workflow(definition, journal=first, approved=True)
    assert retried.state == "partial" and retried.evaluations == 3
    assert len(retried.entries) == 3 and retried.entries[:2] == first.entries
    assert retried.entries[2].output_digest != first.entries[0].output_digest
    wire = retried.to_dict()
    cast(dict[str, object], wire["body"])["state"] = "complete"
    with pytest.raises(ValueError, match="completed journal dependency has changed"):
        parse_workflow_journal(wire, definition)


def test_original_checkpoint_refuses_a_different_current_graph() -> None:
    """Keep the original checkpoint bound to its exact saved graph identity."""
    definition, wire = original_checkpoint()
    changed = parse_workflow(replace(definition, workflow_id="another-workflow").to_dict())
    with pytest.raises(ValueError, match="journal belongs to another workflow definition"):
        parse_workflow_journal(wire, changed)


@pytest.mark.parametrize("fault", ["missing-parent", "failed-parent", "changed-parent"])
def test_actual_completed_child_refuses_corrupted_dependency_history(fault: str) -> None:
    """Reject completion after the original parent's history or digest is corrupted.

    Parameters
    ----------
    fault
        Concrete corruption of a completed history generated by the original executive loop.

    """
    from scpn_quantum_control.studio.workflow_execution import run_workflow

    definition = original_definition()
    wire = run_workflow(definition).to_dict()
    body = cast(dict[str, object], wire["body"])
    entries = cast(list[dict[str, object]], body["entries"])
    if fault == "missing-parent":
        entries.pop(0)
    elif fault == "failed-parent":
        entries[0].update(status="failed", reason="Original record was corrupted")
    else:
        cast(dict[str, object], entries[1]["dependencies"])["source"] = canonical_digest(
            "studio.workflow-output.v1", entries[2]["output"]
        )
    before = write_json(wire)
    with pytest.raises(
        ValueError, match="completed attempt lacks its original completed dependency"
    ):
        parse_workflow_journal(wire, definition)
    assert write_json(wire) == before
