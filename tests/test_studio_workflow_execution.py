# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original executive workflow public tests
"""Run genuine compiler graphs through the original spine and recover real checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.studio.executive import (
    ActionRegistry,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    build_generated_script,
    run_action,
)
from scpn_quantum_control.studio.executive_cli import build_default_registry
from scpn_quantum_control.studio.executive_compile import CompileActionHandler
from scpn_quantum_control.studio.workflow_contracts import WorkflowDefinition, parse_workflow
from scpn_quantum_control.studio.workflow_execution import run_workflow
from scpn_quantum_control.studio.workflow_journal import WorkflowJournal, parse_workflow_journal
from scpn_quantum_control.studio_workspace.canonical import canonical_bytes, canonical_digest
from scpn_quantum_control.studio_workspace.json_transport import read_json


def definition() -> WorkflowDefinition:
    """Read the shared bounded original compiler graph.

    Returns
    -------
    WorkflowDefinition
        Exact two-stage, six-coordinate compiler graph.

    """
    corpus = cast(
        dict[str, object],
        read_json(
            (Path(__file__).parent / "data/studio_workflow/contract_cases.json").read_text()
        ),
    )
    return parse_workflow(corpus["workflow"])


def test_original_compiler_executes_six_cells_and_resume_preserves_all_records() -> None:
    """Run both genuine compiler stages in six cells and avoid duplicate completed work."""
    source = definition()
    saved: list[WorkflowJournal] = []
    journal = run_workflow(source, checkpoint=saved.append)
    assert journal.state == "complete" and journal.evaluations == 12
    assert len(journal.entries) == 12
    assert len({entry.cell_id for entry in journal.entries}) == 6
    assert [entry.stage_id for entry in journal.entries] == ["source", "trace"] * 6
    assert all(entry.status == "complete" for entry in journal.entries)
    assert len(saved) == 25
    assert saved[0].entries[0].status == "running"
    assert saved[1].entries[0].status == "complete"
    for entry in journal.entries:
        output = cast(dict[str, object], entry.to_dict()["output"])
        result = cast(dict[str, object], output["result"])
        assert (
            cast(dict[str, object], result["outputs"])["execution_status"]
            == "emitted_not_executed"
        )
        assert output["script"] is not None
    resumed = run_workflow(source, journal=journal)
    assert canonical_bytes("workflow-checkpoint.v1", resumed.to_dict()) == canonical_bytes(
        "workflow-checkpoint.v1", journal.to_dict()
    )


def test_cancelled_partial_resume_keeps_the_original_completed_parent() -> None:
    """Cancel after a genuinely completed stage and resume the remaining eleven attempts."""
    source = definition()
    saved: list[WorkflowJournal] = []

    def cancelled() -> bool:
        return bool(saved and saved[-1].entries and saved[-1].entries[-1].status == "complete")

    partial = run_workflow(source, checkpoint=saved.append, cancelled=cancelled)
    assert partial.state == "cancelled" and partial.evaluations == 1
    first = partial.entries[0].to_dict()
    complete = run_workflow(source, journal=partial)
    assert complete.state == "complete" and complete.evaluations == 12
    assert complete.entries[0].to_dict() == first
    assert len(complete.entries) == 12


def test_cancel_before_execute_keeps_an_explicit_unfinished_attempt() -> None:
    """Respect cancellation during the pre-execution checkpoint without inventing output."""
    saved: list[WorkflowJournal] = []
    journal = run_workflow(definition(), checkpoint=saved.append, cancelled=lambda: bool(saved))
    assert journal.state == "cancelled" and journal.evaluations == 1
    assert journal.entries[0].status == "cancelled"
    assert journal.entries[0].output is None
    assert saved[0].entries[0].status == "running"


def test_original_refused_parent_blocks_its_children_in_every_cell() -> None:
    """Retain six original input refusals and six blocked children without child execution."""
    source = definition()
    axes = (
        ("source", "program_source", ("unsupported source a", "unsupported source b")),
        source.sweep.axes[1],
    )
    source = parse_workflow(replace(source, sweep=replace(source.sweep, axes=axes)).to_dict())
    journal = run_workflow(source)
    assert journal.state == "partial" and journal.evaluations == 6
    assert [entry.status for entry in journal.entries] == ["failed", "blocked"] * 6
    assert all(entry.output is None for entry in journal.entries)
    assert all(
        entry.dependencies["source"] is None
        for entry in journal.entries
        if entry.stage_id == "trace"
    )


def test_absent_declared_output_preserves_actual_record_and_blocks_dependants() -> None:
    """Keep the genuine compiler result when its caller's declared output path is absent."""
    source = definition()
    producer = next(stage for stage in source.stages if stage.id == "source")
    producer = replace(
        producer,
        outputs=(replace(producer.outputs[0], path=("result", "outputs", "absent")),),
    )
    source = parse_workflow(
        replace(
            source,
            stages=tuple(
                producer if stage.id == producer.id else stage for stage in source.stages
            ),
        ).to_dict()
    )
    journal = run_workflow(source)
    assert journal.state == "partial" and journal.evaluations == 6
    assert [entry.status for entry in journal.entries] == ["failed", "blocked"] * 6
    for entry in journal.entries[::2]:
        record = cast(dict[str, object], entry.to_dict()["output"])
        result = cast(dict[str, object], record["result"])
        assert result["status"] == "succeeded"
        assert (
            cast(dict[str, object], result["outputs"])["execution_status"]
            == "emitted_not_executed"
        )
        assert record["script"] is not None
        assert (
            entry.reason
            == "Original action output, script or seal refused; partial evidence retained"
        )


def test_imported_completion_cannot_replace_a_current_refused_plan() -> None:
    """Refuse a completion claim copied onto an actual failed current compiler attempt."""
    source = definition()
    producer = next(stage for stage in source.stages if stage.id == "source")
    valid = parse_workflow(
        replace(
            source,
            stages=(producer,),
            sweep=replace(source.sweep, axes=(), evaluation_budget=1),
        ).to_dict()
    )
    genuine = run_workflow(valid)
    refused = parse_workflow(
        replace(
            valid,
            stages=(replace(producer, parameters={"program_source": "unsupported source"}),),
        ).to_dict()
    )
    failed = run_workflow(refused)
    assert failed.entries[0].status == "failed" and failed.entries[0].output is None
    wire = failed.to_dict()
    body = cast(dict[str, object], wire["body"])
    entry = cast(list[dict[str, object]], body["entries"])[0]
    original_output = genuine.entries[0].to_dict()["output"]
    entry.update(
        status="complete",
        reason=None,
        output=original_output,
        output_digest=canonical_digest("studio.workflow-output.v1", original_output),
    )
    body["state"] = "complete"
    admitted = parse_workflow_journal(wire, refused)
    saved: list[WorkflowJournal] = []
    with pytest.raises(ValueError, match="matching cached stage no longer has an original plan"):
        run_workflow(refused, journal=admitted, checkpoint=saved.append)
    assert saved == []


@pytest.fixture(scope="module")
def original_runtime_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy the actual package to an owned runtime allocation with every source byte verified.

    Parameters
    ----------
    tmp_path_factory
        Existing Samsung-only allocation owner for the subprocess fixture.

    Returns
    -------
    Path
        Actual complete package source tree; canonical source files stay untouched.

    """
    original = Path(__file__).parents[1] / "src/scpn_quantum_control"
    target = tmp_path_factory.mktemp("workflow-runtime") / "workflow-runtime-source"
    package = target / "scpn_quantum_control"
    shutil.copytree(original, package, ignore=shutil.ignore_patterns("__pycache__"))
    for source in original.rglob("*"):
        if source.is_file() and "__pycache__" not in source.parts:
            assert (package / source.relative_to(original)).read_bytes() == source.read_bytes()
    return target


@pytest.mark.parametrize("change", ["mtime", "unavailable"])
def test_actual_runtime_change_stops_after_the_original_completed_stage(
    original_runtime_copy: Path, tmp_path: Path, change: str
) -> None:
    """Retain the genuine first stage and stop when its observed source becomes unavailable.

    Parameters
    ----------
    original_runtime_copy
        Verified complete byte-identical package copied into the owned allocation.
    tmp_path
        Owned graph input and subprocess output allocation.
    change
        Actual copied-file timestamp change or same-directory temporary relocation.

    """
    source = tmp_path / "workflow.json"
    source.write_text(json.dumps(definition().to_dict()))
    script = """
import json, os, sys
from pathlib import Path
from scpn_quantum_control.studio import workflow_execution
from scpn_quantum_control.studio.workflow_contracts import parse_workflow
from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json

execution_path = Path(workflow_execution.__file__).resolve()
assert execution_path == Path(sys.argv[2]) / "scpn_quantum_control/studio/workflow_execution.py"
target = execution_path.parents[1] / "__init__.py"
original_bytes = target.read_bytes()
original_stat = target.stat()
relocated = target.with_suffix(".temporarily-unavailable")
changed = False
saved = []

def checkpoint(journal):
    global changed
    saved.append(journal)
    if not changed and journal.entries[-1].status == "complete":
        changed = True
        if sys.argv[3] == "unavailable":
            target.rename(relocated)
        else:
            os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1))

try:
    result = workflow_execution.run_workflow(
        parse_workflow(read_json(Path(sys.argv[1]).read_text())), checkpoint=checkpoint
    )
    assert changed and result.state == "partial" and result.evaluations == 1
    assert len(result.entries) == 1 and result.entries[0].status == "complete"
    assert result.entries[0].to_dict()["output"]["result"]["outputs"]["execution_status"] == "emitted_not_executed"
    assert result.entries[0].to_dict() == saved[1].entries[0].to_dict()
    print(write_json(result.to_dict()))
finally:
    if relocated.exists():
        relocated.rename(target)
    os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert target.read_bytes() == original_bytes
"""
    repo = Path(__file__).parents[1]
    environment = dict(
        os.environ,
        PYTHONPATH=f"{original_runtime_copy}:{repo / 'oscillatools/src'}:{repo}",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(original_runtime_copy), change],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    journal = cast(dict[str, object], read_json(result.stdout))
    body = cast(dict[str, object], journal["body"])
    assert body["state"] == "partial" and body["evaluations"] == 1
    assert len(cast(list[object], body["entries"])) == 1


@pytest.mark.parametrize("approval", [0, 1, "true", None])
def test_current_host_approval_requires_boolean_before_checkpointing(approval: object) -> None:
    """Refuse coercible host permissions before any original reservation or action.

    Parameters
    ----------
    approval
        Malformed runtime permission, independent of imported graph metadata.

    """
    saved: list[WorkflowJournal] = []
    with pytest.raises(ValueError, match="current explicit host approval must be boolean"):
        run_workflow(definition(), approved=cast(bool, approval), checkpoint=saved.append)
    assert saved == []


def test_retry_preserves_matching_blocked_children_without_duplicate_history() -> None:
    """Retain the actual failed parents and charge retries without duplicating blocked children."""
    source = definition()
    source = parse_workflow(
        replace(
            source,
            sweep=replace(
                source.sweep,
                axes=(
                    ("source", "program_source", ("unsupported source a", "unsupported source b")),
                    source.sweep.axes[1],
                ),
            ),
        ).to_dict()
    )
    first = run_workflow(source)
    retry = run_workflow(source, journal=first)
    assert first.state == retry.state == "partial"
    assert first.evaluations == 6 and retry.evaluations == 12
    assert len(first.entries) == 12 and len(retry.entries) == 18
    assert retry.entries[:12] == first.entries
    assert [entry.status for entry in retry.entries[12:]] == ["failed"] * 6
    assert sum(entry.status == "blocked" for entry in retry.entries) == 6


def test_resume_refuses_a_changed_runtime_before_saving_or_executing() -> None:
    """Reject a structurally valid checkpoint whose actual runtime identity differs."""
    source = definition()
    complete = run_workflow(source)
    changed = replace(complete, runtime_fingerprint="f" * 64)
    saved: list[WorkflowJournal] = []
    with pytest.raises(ValueError, match="runtime differs"):
        run_workflow(source, journal=changed, checkpoint=saved.append)
    assert saved == []


def test_self_rehashed_output_cannot_replace_original_executive_seal() -> None:
    """Refuse altered original result even when the imported journal output digest matches."""
    source = definition()
    complete = run_workflow(source)
    wire = complete.to_dict()
    body = cast(dict[str, object], wire["body"])
    entries = cast(list[dict[str, object]], body["entries"])
    first = entries[0]
    output = cast(dict[str, object], first["output"])
    result = cast(dict[str, object], output["result"])
    result["outputs"] = {"execution_status": "invented"}
    original_digest = first["output_digest"]
    first["output_digest"] = canonical_digest("studio.workflow-output.v1", output)
    for row in entries:
        deps = cast(dict[str, object], row["dependencies"])
        if deps.get("source") == original_digest:
            deps["source"] = first["output_digest"]
    admitted = parse_workflow_journal(wire, source)
    saved: list[WorkflowJournal] = []
    with pytest.raises(ValueError, match="digest must seal"):
        run_workflow(source, journal=admitted, checkpoint=saved.append)
    assert saved == []


@pytest.fixture(scope="module")
def original_cached_stage() -> tuple[WorkflowDefinition, WorkflowJournal]:
    """Produce one actual immutable compiler stage for adverse cache admission.

    Returns
    -------
    tuple[WorkflowDefinition, WorkflowJournal]
        Complete original single-stage graph and its genuine producer checkpoint.

    """
    source = definition()
    source = parse_workflow(
        replace(
            source,
            stages=(next(stage for stage in source.stages if stage.id == "source"),),
            sweep=replace(source.sweep, axes=(), evaluation_budget=1),
        ).to_dict()
    )
    journal = run_workflow(source)
    assert journal.state == "complete" and journal.evaluations == 1
    return source, journal


@pytest.mark.parametrize(
    "fault,refusal",
    [
        ("record-object", "original executive object required"),
        ("request", "cached original request differs"),
        ("plan", "cached original plan differs"),
        ("result-status", "cached original record did not succeed"),
        ("result-error", "cached original record did not succeed"),
        ("result-object", "original executive object required"),
        ("outputs-object", "original executive object required"),
        ("script-language", "cached original script language differs"),
        ("script-object", "original executive object required"),
        ("script-filename", "original executive text required"),
        ("script-entrypoint", "original executive text required"),
        ("script-source", "original executive text required"),
        ("script-digest", "original executive text required"),
        ("record-digest", "original executive text required"),
        ("extra-field", "cached original record fields differ"),
        ("script-content", "cached original reproduction script differs"),
    ],
)
def test_original_completed_cache_refuses_rehashed_producer_corruption(
    original_cached_stage: tuple[WorkflowDefinition, WorkflowJournal], fault: str, refusal: str
) -> None:
    """Refuse corrupted original fields even when the imported outer journal is rehashed.

    Parameters
    ----------
    original_cached_stage
        Actual immutable compiler producer history, independently run once.
    fault
        Concrete malformed field or altered reproduction script in that original history.
    refusal
        Exact public refusal boundary required before any checkpoint write.

    """
    source, complete = original_cached_stage
    original_bytes = canonical_bytes("cache-source.v1", complete.to_dict())
    wire = complete.to_dict()
    entries = cast(list[dict[str, object]], cast(dict[str, object], wire["body"])["entries"])
    entry = entries[0]
    output = cast(dict[str, object], entry["output"])
    if fault == "record-object":
        entry["output"] = []
    elif fault == "request":
        cast(dict[str, object], output["request"])["approved"] = True
    elif fault == "plan":
        cast(dict[str, object], output["plan"])["claim_boundary"] = "altered source boundary"
    elif fault in {"result-status", "result-error"}:
        cast(dict[str, object], output["result"])[
            "status" if fault == "result-status" else "error"
        ] = "failed" if fault == "result-status" else "altered source error"
    elif fault in {"result-object", "script-object"}:
        output["result" if fault == "result-object" else "script"] = []
    elif fault == "outputs-object":
        cast(dict[str, object], output["result"])["outputs"] = []
    elif fault == "script-language":
        cast(dict[str, object], output["script"])["language"] = "bash"
    elif fault == "record-digest":
        output["digest"] = 0
    elif fault == "extra-field":
        output["untrusted_extension"] = True
    elif fault == "script-content":
        script = cast(dict[str, object], output["script"])
        output["script"] = build_generated_script(
            filename=cast(str, script["filename"]),
            entrypoint=cast(str, script["entrypoint"]),
            source=cast(str, script["source"]) + "\nprint('altered reproduction')\n",
        ).to_dict()
        payload = {key: output[key] for key in ("request", "plan", "result", "script")}
        output["digest"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
    else:
        cast(dict[str, object], output["script"])[fault.removeprefix("script-")] = 0
    entry["output_digest"] = canonical_digest("studio.workflow-output.v1", entry["output"])
    admitted = parse_workflow_journal(wire, source)
    saved: list[WorkflowJournal] = []
    with pytest.raises(ValueError, match=refusal):
        run_workflow(source, journal=admitted, checkpoint=saved.append)
    assert saved == []
    assert canonical_bytes("cache-source.v1", complete.to_dict()) == original_bytes


class ScriptFailure(CompileActionHandler):
    """Original compiler with an injected post-execution script failure."""

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Fail only after the original execution result has been obtained.

        Parameters
        ----------
        plan
            Genuine original plan.
        result
            Genuine original completed execution result.

        Returns
        -------
        GeneratedScript
            Never returned by this adverse-only injection.

        Raises
        ------
        OSError
            Injected script producer failure after genuine execution.

        """
        raise OSError("injected script failure")


def test_post_execute_script_failure_retains_genuine_partial_result() -> None:
    """Do not hide actual original output when the script or seal cannot complete."""
    registry = ActionRegistry()
    registry.register(ScriptFailure())
    journal = run_workflow(definition(), registry=registry)
    assert journal.state == "partial" and journal.evaluations == 6
    first = journal.entries[0].to_dict()
    assert first["status"] == "failed"
    partial = cast(dict[str, object], first["output"])
    result = cast(dict[str, object], partial["partial_result"])
    assert result["status"] == "succeeded"
    assert cast(dict[str, object], result["outputs"])["execution_status"] == "emitted_not_executed"
    assert partial["terminal_record"] is None
    assert journal.entries[1].status == "blocked"


def test_checkpoint_failure_stops_at_the_reserved_original_attempt() -> None:
    """A failed original persistence callback cannot silently execute the stage."""
    saved: list[WorkflowJournal] = []

    def refuse(journal: WorkflowJournal) -> None:
        saved.append(journal)
        raise OSError("owned checkpoint store unavailable")

    with pytest.raises(OSError, match="checkpoint store"):
        run_workflow(definition(), checkpoint=refuse)
    assert len(saved) == 1
    assert saved[0].entries[0].status == "running"
    assert saved[0].entries[0].output is None


def test_restart_preserves_interrupted_reservation_and_charges_the_retry_once() -> None:
    """Keep the original incomplete attempt visible and consume one additional retry unit."""
    source = definition()
    source = parse_workflow(
        replace(source, sweep=replace(source.sweep, evaluation_budget=16)).to_dict()
    )
    saved: list[WorkflowJournal] = []

    def stop(journal: WorkflowJournal) -> None:
        saved.append(journal)
        raise OSError("interrupted checkpoint owner")

    with pytest.raises(OSError):
        run_workflow(source, checkpoint=stop)
    resumed = run_workflow(source, journal=saved[0])
    assert resumed.state == "complete" and resumed.evaluations == 13
    assert resumed.entries[0].status == "interrupted"
    assert resumed.entries[0].output is None
    assert len(resumed.entries) == 13


def test_exhausted_budget_retains_all_completed_and_unfinished_original_attempts() -> None:
    """An interrupted attempt cannot extend the original twelve-evaluation budget."""
    source = definition()
    saved: list[WorkflowJournal] = []

    def stop(journal: WorkflowJournal) -> None:
        saved.append(journal)
        raise OSError("interrupted checkpoint owner")

    with pytest.raises(OSError):
        run_workflow(source, checkpoint=stop)
    resumed = run_workflow(source, journal=saved[0])
    assert resumed.state == "partial" and resumed.evaluations == 12
    assert resumed.entries[0].status == "interrupted"
    assert len(resumed.entries) == 12
    assert sum(entry.status == "complete" for entry in resumed.entries) == 11


def test_original_live_verb_stays_gated_and_blocks_its_control_dependant() -> None:
    """Imported graph stages cannot grant the original live-hardware approval."""
    source = definition()
    first = next(stage for stage in source.stages if stage.id == "source")
    deployment = replace(
        first,
        id="deployment",
        verb="execute",
        backend="provider-hal",
        parameters={
            "provider": "iqm",
            "endpoint": "declared-no-submit",
            "circuit_digest": "sha256:" + "a" * 64,
            "circuit_ref": "original-input",
            "shots": 1,
        },
        outputs=(),
        depends_on=("source",),
    )
    child = replace(first, id="after", outputs=(), depends_on=("deployment",))
    source = parse_workflow(
        replace(
            source,
            stages=(first, deployment, child),
            sweep=replace(source.sweep, axes=(source.sweep.axes[0],), evaluation_budget=6),
        ).to_dict()
    )
    journal = run_workflow(source)
    assert [entry.status for entry in journal.entries] == ["complete", "gated", "blocked"] * 2
    assert journal.state == "partial" and journal.evaluations == 4
    gated = cast(dict[str, object], journal.entries[1].to_dict()["output"])
    assert cast(dict[str, object], gated["request"])["approved"] is False
    assert gated["script"] is None


def test_freshly_resealed_import_cannot_claim_unapproved_original_live_completion() -> None:
    """An imported self-sealed record cannot bypass the original current approval gate."""
    source = definition()
    first = source.stages[0]
    stage = replace(
        first,
        id="deployment",
        verb="execute",
        backend="provider-hal",
        parameters={
            "provider": "iqm",
            "endpoint": "declared-no-submit",
            "circuit_digest": "sha256:" + "a" * 64,
            "circuit_ref": "original-input",
            "shots": 1,
        },
        inputs=(),
        outputs=(),
        depends_on=(),
    )
    source = parse_workflow(
        replace(
            source, stages=(stage,), sweep=replace(source.sweep, axes=(), evaluation_budget=1)
        ).to_dict()
    )
    gated = run_workflow(source)
    assert gated.entries[0].status == "gated"
    wire = gated.to_dict()
    body = cast(dict[str, object], wire["body"])
    row = cast(list[dict[str, object]], body["entries"])[0]
    original = cast(dict[str, object], row["output"])
    request = cast(dict[str, object], original["request"])
    # The original handler emits a no-submit dossier and never contacts a provider.
    approved = run_action(
        ExecutiveRequest(
            "execute", cast(str, request["action_id"]), stage.parameters, stage.backend, True
        ),
        registry=build_default_registry(),
    ).to_dict()
    assert cast(dict[str, object], approved["result"])["status"] == "succeeded"
    cast(dict[str, object], approved["request"])["approved"] = False
    payload = {key: approved[key] for key in ("request", "plan", "result", "script")}
    approved["digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )
    row.update(
        status="complete",
        reason=None,
        output=approved,
        output_digest=canonical_digest("studio.workflow-output.v1", approved),
    )
    body["state"] = "complete"
    admitted = parse_workflow_journal(wire, source)
    saved: list[WorkflowJournal] = []
    with pytest.raises(ValueError, match="original approval gate"):
        run_workflow(source, journal=admitted, checkpoint=saved.append)
    assert saved == []
