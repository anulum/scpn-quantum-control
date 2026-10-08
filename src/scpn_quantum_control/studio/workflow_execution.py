# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original executive workflow execution
"""Run bounded graphs through the original executive spine and checkpoint every attempt.

Cancellation is cooperative at original synchronous action boundaries. Imported
graphs cannot approve actions or install handlers. Original backend contracts,
plans, numerical routines, generated scripts and record seals remain authoritative.
"""

from __future__ import annotations

import hashlib
import inspect
import platform
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

from ..studio_workspace.canonical import canonical_bytes, canonical_digest
from ..studio_workspace.json_transport import read_json, write_json
from .executive import (
    ActionHandler,
    ActionRegistry,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRecord,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    preview_action,
    run_action,
)
from .executive_cli import build_default_registry
from .workflow_contracts import (
    WorkflowDefinition,
    WorkflowStage,
    parse_workflow,
    topological_order,
    validate_port_value,
)
from .workflow_journal import (
    WorkflowAttempt,
    WorkflowJournal,
    create_workflow_journal,
    parse_workflow_journal,
)
from .workflow_sweep import WorkflowCell, build_workflow_cells


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("original executive object required")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("original executive text required")
    return value


def _select(output: object, path: tuple[str, ...]) -> object:
    for key in path:
        if not isinstance(output, Mapping) or key not in output:
            raise ValueError("declared original output port is absent")
        output = output[key]
    return output


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    current = path.stat()
    return (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )


def _source_digest(path: Path, stamps: dict[Path, tuple[int, int, int, int, int]]) -> str:
    before = _stamp(path)
    data = path.read_bytes()
    if _stamp(path) != before:
        raise ValueError("original runtime source changed while hashing")
    stamps[path] = before
    stamps[path.parent] = _stamp(path.parent)
    return hashlib.sha256(data).hexdigest()


def _runtime_versions() -> dict[str, str | None]:
    dependencies: dict[str, str | None] = {}
    for name in ("numpy", "qiskit", "qiskit-aer", "scipy", "scpn-studio-platform"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return dependencies


def _sources_unchanged(stamps: Mapping[Path, tuple[int, int, int, int, int]]) -> bool:
    try:
        return all(_stamp(path) == original for path, original in stamps.items())
    except OSError:
        return False


def workflow_runtime_identity(
    registry: ActionRegistry,
    definition: WorkflowDefinition,
    *,
    source_stamps: dict[Path, tuple[int, int, int, int, int]] | None = None,
) -> str:
    """Hash actual package/handler sources and declared installed runtime versions.

    Parameters
    ----------
    registry
        Trusted host registry; graphs never select Python modules or callbacks.
    definition
        Admitted original graph identifying the handlers used.
    source_stamps
        Optional host-owned file/directory observation index. It receives device,
        inode, size, modification and change times after stable byte hashing, for
        cooperative change detection within this run. Each new run hashes bytes
        again; filesystem observations never attest producer execution.

    Returns
    -------
    str
        Exact identity of package and handler bytes, interpreter version and
        installed dependency versions. It is a cache boundary, not an attestation
        of all transitive installed dependency bytes or scientific equivalence.

    Raises
    ------
    OSError
        An original source file cannot be read.
    ValueError
        An unavailable or oversized source set cannot support cache reuse.

    """
    package = Path(__file__).resolve().parents[1]
    paths = sorted(package.rglob("*.py"))
    paths += sorted(package.rglob("*.so"))
    sources: dict[str, str] = {}
    stamps: dict[Path, tuple[int, int, int, int, int]] = {}
    byte_count = 0
    if len(paths) > 4096:
        raise ValueError("runtime source inventory exceeds bound")
    for path in paths:
        byte_count += path.stat().st_size
        if byte_count > 128 * 1024 * 1024:
            raise ValueError("runtime source inventory exceeds byte bound")
        sources[path.relative_to(package).as_posix()] = _source_digest(path, stamps)
    for stage in definition.stages:
        handler = registry.resolve(stage.verb)
        path = Path(inspect.getfile(type(handler)))
        sources[f"handler:{type(handler).__module__}.{type(handler).__qualname__}"] = (
            _source_digest(path, stamps)
        )
    for name, spec in (
        ("scpn_quantum_engine", find_spec("scpn_quantum_engine")),
        ("scpn_studio_platform", find_spec("scpn_studio_platform")),
        ("oscillatools", find_spec("oscillatools")),
        ("numpy", find_spec("numpy")),
        ("qiskit", find_spec("qiskit")),
    ):
        if spec is None or spec.origin is None:
            continue
        origin = Path(spec.origin)
        originals = (
            [origin]
            if spec.submodule_search_locations is None or name == "scpn_quantum_engine"
            else sorted(origin.parent.rglob("*.py")) + sorted(origin.parent.rglob("*.so"))
        )
        if len(originals) > 4096:
            raise ValueError("original native dependency source inventory exceeds bound")
        for path in originals:
            byte_count += path.stat().st_size
            if byte_count > 128 * 1024 * 1024:
                raise ValueError("original native dependency source exceeds byte bound")
            sources[f"dependency:{name}/{path.relative_to(origin.parent).as_posix()}"] = (
                _source_digest(path, stamps)
            )
    if not _sources_unchanged(stamps):
        raise ValueError("original runtime source changed during snapshot admission")
    if source_stamps is not None:
        source_stamps.update(stamps)
    return canonical_digest(
        "studio.workflow-runtime.v1",
        {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "sources": sources,
            "installed_versions": _runtime_versions(),
        },
    )


class _ObservedHandler(ActionHandler):
    def __init__(self, original: ActionHandler) -> None:
        self.original = original
        self.observed: ExecutionResult | None = None

    @property
    def verb(self) -> str:
        """Return the original trusted verb without substituting a graph label."""
        return self.original.verb

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Delegate the request and contract to the original trusted planner."""
        return self.original.plan(request, contract)

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Retain and return the original handler's actual execution result."""
        self.observed = self.original.execute(plan)
        return self.observed

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Delegate script generation using the original plan and result."""
        return self.original.generate_script(plan, result)


def _verify_record(
    output: object, request: ExecutiveRequest, plan: ExecutionPlan, handler: ActionHandler
) -> None:
    if plan.requires_approval and not request.approved:
        raise ValueError("cached record cannot bypass the original approval gate")
    wire = _object(read_json(write_json(output)))
    if canonical_bytes("studio.workflow-record-field.v1", wire.get("request")) != canonical_bytes(
        "studio.workflow-record-field.v1", request.to_dict()
    ):
        raise ValueError("cached original request differs")
    if canonical_bytes("studio.workflow-record-field.v1", wire.get("plan")) != canonical_bytes(
        "studio.workflow-record-field.v1", plan.to_dict()
    ):
        raise ValueError("cached original plan differs")
    result_wire = _object(wire.get("result"))
    if result_wire.get("status") != "succeeded" or result_wire.get("error") is not None:
        raise ValueError("cached original record did not succeed")
    result = ExecutionResult("succeeded", _object(result_wire.get("outputs")))
    script_wire = _object(wire.get("script"))
    if script_wire.get("language") != "python":
        raise ValueError("cached original script language differs")
    script = GeneratedScript(
        "python",
        _text(script_wire.get("filename")),
        _text(script_wire.get("entrypoint")),
        _text(script_wire.get("source")),
        _text(script_wire.get("digest")),
    )
    record = ExecutiveRecord(request, plan, result, script, _text(wire.get("digest")))
    if canonical_bytes("studio.workflow-record.v1", record.to_dict()) != canonical_bytes(
        "studio.workflow-record.v1", wire
    ):
        raise ValueError("cached original record fields differ")
    generated = handler.generate_script(plan, result)
    if canonical_bytes("studio.workflow-script.v1", generated.to_dict()) != canonical_bytes(
        "studio.workflow-script.v1", script.to_dict()
    ):
        raise ValueError("cached original reproduction script differs")


def _parameters(
    stage: WorkflowStage,
    cell: WorkflowCell,
    completed: Mapping[str, WorkflowAttempt],
    stages: Mapping[str, WorkflowStage],
) -> dict[str, object]:
    parameters = _object(read_json(write_json(dict(stage.parameters))))
    parameters.update(_object(read_json(write_json(dict(cell.overrides.get(stage.id, {}))))))
    for binding in stage.inputs:
        parent = completed[binding.source_stage]
        port = next(
            item
            for item in stages[binding.source_stage].outputs
            if item.name == binding.source_port
        )
        value = validate_port_value(port.type, _select(parent.output, port.path))
        parameters[binding.parameter] = read_json(write_json(value))
    return parameters


def _checkpoint(
    definition: WorkflowDefinition,
    journal: WorkflowJournal,
    entries: list[dict[str, object]],
    state: str,
    source: str,
    runtime: str,
    save: Callable[[WorkflowJournal], None] | None,
) -> WorkflowJournal:
    document = journal.to_dict()
    body = _object(document["body"])
    body.update(
        source_fingerprint=source,
        runtime_fingerprint=runtime,
        entries=entries,
        state=state,
        evaluations=sum(row["evaluated"] is True for row in entries),
    )
    admitted = parse_workflow_journal(document, definition)
    if save is not None:
        save(admitted)
    return admitted


def run_workflow(
    definition: WorkflowDefinition,
    *,
    journal: WorkflowJournal | None = None,
    registry: ActionRegistry | None = None,
    checkpoint: Callable[[WorkflowJournal], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    approved: bool = False,
) -> WorkflowJournal:
    """Execute original verbs with exact dependency/cache validation and bounded attempts.

    Parameters
    ----------
    definition
        Complete admitted graph. This host supports the original executive adapter.
    journal
        Prior original checkpoint; running entries remain interrupted on restart.
    registry
        Trusted host registry, or all nine original handlers when omitted.
    checkpoint
        Trusted persistence callback invoked before and after each original action.
        A callback failure stops execution; no completed save is invented.
    cancelled
        Trusted cooperative cancellation predicate at synchronous stage boundaries.
    approved
        Explicit current host approval passed to the original gate; imported
        documents never provide this permission. Unapproved live verbs stay gated.

    Returns
    -------
    WorkflowJournal
        Original attempt history with explicit complete, partial or cancelled state.
        Budget exhaustion leaves the prior results visible in a partial checkpoint.

    Raises
    ------
    ValueError
        Unsupported adapter, invalid original history or corrupted matching record.
    OSError
        Source identity or checkpoint storage is unavailable.

    """
    definition = parse_workflow(definition.to_dict())
    if any(stage.adapter != "executive" for stage in definition.stages):
        raise ValueError("local-kuramoto workflow requires its original browser WASM adapter")
    if not isinstance(approved, bool):
        raise ValueError("current explicit host approval must be boolean")
    registry = build_default_registry() if registry is None else registry
    source = canonical_digest("studio.workflow-source.v1", definition.to_dict())
    source_stamps: dict[Path, tuple[int, int, int, int, int]] = {}
    versions = _runtime_versions()
    runtime = workflow_runtime_identity(registry, definition, source_stamps=source_stamps)
    journal = (
        create_workflow_journal(definition, source_fingerprint=source, runtime_fingerprint=runtime)
        if journal is None
        else parse_workflow_journal(journal.to_dict(), definition)
    )
    if journal.source_fingerprint != source or journal.runtime_fingerprint != runtime:
        raise ValueError("checkpoint source or runtime differs; original history retained")
    entries = [entry.to_dict() for entry in journal.entries]
    recovered = False
    for original_entry in entries:
        if original_entry["status"] == "running":
            original_entry.update(
                status="interrupted",
                reason="Original synchronous attempt has no terminal checkpoint",
            )
            recovered = True
    if recovered:
        journal = _checkpoint(definition, journal, entries, "partial", source, runtime, checkpoint)
    stages = {stage.id: stage for stage in definition.stages}
    latest = {(entry.cell_id, entry.stage_id): entry for entry in journal.entries}
    finished = True
    for cell in build_workflow_cells(definition):
        completed: dict[str, WorkflowAttempt] = {}
        for sid in topological_order(definition):
            if cancelled is not None and cancelled():
                return _checkpoint(
                    definition, journal, entries, "cancelled", source, runtime, checkpoint
                )
            if not _sources_unchanged(source_stamps) or _runtime_versions() != versions:
                return _checkpoint(
                    definition, journal, entries, "partial", source, runtime, checkpoint
                )
            stage = stages[sid]
            parents = sorted(
                set(stage.depends_on) | {binding.source_stage for binding in stage.inputs}
            )
            dependencies = {
                parent: completed[parent].output_digest if parent in completed else None
                for parent in parents
            }
            base = {
                "cell_id": cell.id,
                "stage_id": sid,
                "source": source,
                "runtime": runtime,
                "dependencies": dependencies,
                "approved": approved,
            }
            if any(parent not in completed for parent in parents):
                fingerprint = canonical_digest("studio.workflow-stage.v1", base)
                prior = latest.get((cell.id, sid))
                if prior is None or prior.status != "blocked" or prior.fingerprint != fingerprint:
                    entries.append(
                        dict(
                            cell_id=cell.id,
                            stage_id=sid,
                            fingerprint=fingerprint,
                            dependencies=dependencies,
                            status="blocked",
                            output=None,
                            output_digest=None,
                            reason="Original parent did not complete",
                            evaluated=False,
                        )
                    )
                    journal = _checkpoint(
                        definition, journal, entries, "partial", source, runtime, checkpoint
                    )
                finished = False
                continue
            plan: ExecutionPlan | None = None
            request: ExecutiveRequest | None = None
            try:
                parameters = _parameters(stage, cell, completed, stages)
                request = ExecutiveRequest(
                    stage.verb, f"workflow-{cell.id}-{sid}", parameters, stage.backend, approved
                )
                plan = preview_action(request, registry=registry)
            except (KeyError, ValueError, TypeError):
                pass
            fingerprint = canonical_digest(
                "studio.workflow-stage.v1",
                {
                    **base,
                    "request": None if request is None else request.to_dict(),
                    "plan": None if plan is None else plan.to_dict(),
                },
            )
            prior = latest.get((cell.id, sid))
            if (
                prior is not None
                and prior.status == "complete"
                and prior.fingerprint == fingerprint
            ):
                if plan is None or request is None:
                    raise ValueError("matching cached stage no longer has an original plan")
                _verify_record(prior.output, request, plan, registry.resolve(stage.verb))
                completed[sid] = prior
                continue
            if journal.evaluations >= definition.sweep.evaluation_budget:
                return _checkpoint(
                    definition, journal, entries, "partial", source, runtime, checkpoint
                )
            row: dict[str, object] = dict(
                cell_id=cell.id,
                stage_id=sid,
                fingerprint=fingerprint,
                dependencies=dependencies,
                status="running",
                output=None,
                output_digest=None,
                reason="Original synchronous action reserved; terminal checkpoint pending",
                evaluated=True,
            )
            entries.append(row)
            journal = _checkpoint(
                definition, journal, entries, "partial", source, runtime, checkpoint
            )
            if cancelled is not None and cancelled():
                row.update(status="cancelled", reason="Cancelled before original action execution")
                return _checkpoint(
                    definition, journal, entries, "cancelled", source, runtime, checkpoint
                )
            output: object = None
            status = "failed"
            reason: str | None = "Original stage input or plan refused"
            if plan is not None and request is not None:
                observer = _ObservedHandler(registry.resolve(stage.verb))
                observed_registry = ActionRegistry()
                observed_registry.register(observer)
                try:
                    record = run_action(request, registry=observed_registry)
                    output = record.to_dict()
                    if record.result.status == "succeeded":
                        _verify_record(output, request, plan, observer.original)
                        for port in stage.outputs:
                            validate_port_value(port.type, _select(output, port.path))
                        status, reason = "complete", None
                    else:
                        status = record.result.status
                        reason = (
                            "Original action was gated"
                            if status == "gated"
                            else "Original action failed; inspect its retained record"
                        )
                except Exception:
                    if output is None and observer.observed is not None:
                        output = {
                            "request": request.to_dict(),
                            "plan": plan.to_dict(),
                            "partial_result": observer.observed.to_dict(),
                            "terminal_record": None,
                        }
                    reason = (
                        "Original action output, script or seal refused; partial evidence retained"
                    )
            row.update(
                status=status,
                reason=reason,
                output=output,
                output_digest=None
                if output is None
                else canonical_digest("studio.workflow-output.v1", output),
            )
            journal = _checkpoint(
                definition, journal, entries, "partial", source, runtime, checkpoint
            )
            current = journal.entries[-1]
            latest[(cell.id, sid)] = current
            if status == "complete":
                completed[sid] = current
            else:
                finished = False
    return _checkpoint(
        definition,
        journal,
        entries,
        "complete" if finished else "partial",
        source,
        runtime,
        checkpoint,
    )
