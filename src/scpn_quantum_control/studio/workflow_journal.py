# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original workflow attempt checkpoint admission
"""Preserve bounded original attempts without claiming a cached value executed.

The journal validates identities, dependency links and original output digests.
Runtime adapters must additionally verify the original producer record before
reuse. Structural admission is not source attestation or scientific approval.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..studio_workspace.canonical import canonical_digest
from ..studio_workspace.json_transport import read_json, write_json
from .workflow_contracts import WorkflowDefinition, topological_order
from .workflow_sweep import build_workflow_cells

JOURNAL_SCHEMA = "experiment_workflow_journal.v1"
"""Additive portable attempt journal; original result formats stay unchanged."""
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
"""Portable checkpoint byte ceiling, independent of host memory availability."""

_STATUSES = frozenset(
    {"running", "complete", "failed", "gated", "blocked", "cancelled", "interrupted"}
)


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name}: exact SHA-256 identity required")
    return value


def _object(value: object, fields: set[str] | None, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or (fields is not None and set(value) != fields):
        raise ValueError(f"{name}: complete supported object required")
    return value


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class WorkflowAttempt:
    """One original stage attempt; failed and interrupted data remain visible.

    Parameters
    ----------
    cell_id
        Admitted exact coordinate identity.
    stage_id
        Original stage identity.
    fingerprint
        Exact source/runtime/input/plan/dependency fingerprint from the adapter.
    dependencies
        Original parent output digests; None means the parent did not complete.
    status
        Explicit running, complete, failed, gated, blocked, cancelled or interrupted.
    output
        Original output or partial diagnostics; None never means a zero result.
    output_digest
        Exact canonical output digest, or None when no output was received.
    reason
        Explicit non-completion reason; completed data must carry None.
    evaluated
        Whether this attempt consumed one declared stage-evaluation unit.

    """

    cell_id: str
    stage_id: str
    fingerprint: str
    dependencies: Mapping[str, str | None]
    status: str
    output: object
    output_digest: str | None
    reason: str | None
    evaluated: bool

    def to_dict(self) -> dict[str, object]:
        """Return independent original attempt data for portable checkpointing.

        Returns
        -------
        dict[str, object]
            Original output, status and source/dependency identities.

        """
        return {
            "cell_id": self.cell_id,
            "stage_id": self.stage_id,
            "fingerprint": self.fingerprint,
            "dependencies": dict(self.dependencies),
            "status": self.status,
            "output": _thaw(self.output),
            "output_digest": self.output_digest,
            "reason": self.reason,
            "evaluated": self.evaluated,
        }


@dataclass(frozen=True)
class WorkflowJournal:
    """An immutable original attempt history with explicit partial state.

    Parameters
    ----------
    workflow_digest
        Complete admitted definition identity.
    source_fingerprint
        Host-supplied exact original source identity; never an inferred default.
    runtime_fingerprint
        Host-supplied actual runtime/build identity, independently obtained.
    entries
        Original chronological attempts, retaining every prior failed outcome.
    state
        partial, cancelled or complete; complete requires all current dependency links.
    evaluations
        Actual reserved stage-evaluation units, including failed attempts.
    extensions
        Opaque immutable metadata; no source or execution authority is inferred.

    """

    workflow_digest: str
    source_fingerprint: str
    runtime_fingerprint: str
    entries: tuple[WorkflowAttempt, ...]
    state: str
    evaluations: int
    extensions: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        """Serialize original complete and partial history without dropping attempts.

        Returns
        -------
        dict[str, object]
            Exact supported portable journal document.

        """
        return {
            "schema": JOURNAL_SCHEMA,
            "body": {
                "workflow_digest": self.workflow_digest,
                "source_fingerprint": self.source_fingerprint,
                "runtime_fingerprint": self.runtime_fingerprint,
                "entries": [entry.to_dict() for entry in self.entries],
                "state": self.state,
                "evaluations": self.evaluations,
            },
            "extensions": _thaw(self.extensions),
        }


def parse_workflow_journal(payload: object, definition: WorkflowDefinition) -> WorkflowJournal:
    """Admit original history and dependency/output identities before possible reuse.

    Parameters
    ----------
    payload
        Portable v1 journal retaining complete and incomplete original outcomes.
    definition
        Exact current workflow; different coordinates or definitions refuse.

    Returns
    -------
    WorkflowJournal
        Immutable bounded history. A runtime adapter must still verify producer
        evidence and current source/runtime fingerprints before reusing outputs.

    Raises
    ------
    ValueError
        Version, identity, status, dependency, output, budget or byte bound fails.

    """
    text = write_json(payload)
    if len(text.encode("utf-8")) > MAX_JOURNAL_BYTES:
        raise ValueError("workflow journal exceeds byte bound")
    document = _object(read_json(text), {"schema", "body", "extensions"}, "journal")
    if document["schema"] != JOURNAL_SCHEMA:
        raise ValueError("unsupported workflow journal schema")
    body = _object(
        document["body"],
        {
            "workflow_digest",
            "source_fingerprint",
            "runtime_fingerprint",
            "entries",
            "state",
            "evaluations",
        },
        "journal body",
    )
    cells = build_workflow_cells(definition)
    workflow_digest = canonical_digest("studio.workflow-definition.v1", definition.to_dict())
    if _digest(body["workflow_digest"], "workflow") != workflow_digest:
        raise ValueError("journal belongs to another workflow definition")
    source = _digest(body["source_fingerprint"], "original source")
    runtime = _digest(body["runtime_fingerprint"], "actual runtime")
    stages = {stage.id: stage for stage in definition.stages}
    cell_ids = {cell.id for cell in cells}
    raw_entries = body["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) > 8192:
        raise ValueError("bounded original journal entries required")
    entries: list[WorkflowAttempt] = []
    latest: dict[tuple[str, str], WorkflowAttempt] = {}
    evaluations = 0
    for raw in raw_entries:
        row = _object(
            raw,
            {
                "cell_id",
                "stage_id",
                "fingerprint",
                "dependencies",
                "status",
                "output",
                "output_digest",
                "reason",
                "evaluated",
            },
            "attempt",
        )
        cid, sid = row["cell_id"], row["stage_id"]
        if (
            not isinstance(cid, str)
            or cid not in cell_ids
            or not isinstance(sid, str)
            or sid not in stages
        ):
            raise ValueError("attempt coordinate or stage is absent")
        stage = stages[sid]
        parents = set(stage.depends_on) | {item.source_stage for item in stage.inputs}
        raw_dependencies = row["dependencies"]
        if not isinstance(raw_dependencies, dict) or set(raw_dependencies) != parents:
            raise ValueError("attempt dependencies differ from original graph")
        dependencies: dict[str, str | None] = {}
        for parent, digest in raw_dependencies.items():
            dependencies[parent] = None if digest is None else _digest(digest, "parent output")
        status, reason = row["status"], row["reason"]
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError("unsupported original attempt status")
        if status == "complete":
            if reason is not None or row["output"] is None:
                raise ValueError(
                    "completed attempt requires original output without failure reason"
                )
            for parent, digest in dependencies.items():
                original = latest.get((cid, parent))
                if (
                    original is None
                    or original.status != "complete"
                    or digest != original.output_digest
                ):
                    raise ValueError("completed attempt lacks its original completed dependency")
        elif not isinstance(reason, str) or not 1 <= len(reason) <= 2048:
            raise ValueError("incomplete attempt requires a bounded original reason")
        output = row["output"]
        digest = row["output_digest"]
        if output is None:
            if digest is not None:
                raise ValueError("absent output cannot carry an output digest")
        elif _digest(digest, "original output") != canonical_digest(
            "studio.workflow-output.v1", output
        ):
            raise ValueError("original attempt output digest differs")
        evaluated = row["evaluated"]
        if (
            not isinstance(evaluated, bool)
            or (status == "complete" and not evaluated)
            or (status == "blocked" and evaluated)
        ):
            raise ValueError("attempt evaluation accounting differs")
        previous = latest.get((cid, sid))
        if (
            previous is not None
            and previous.status == "complete"
            and previous.fingerprint == row["fingerprint"]
        ):
            raise ValueError("completed matching stage cannot be duplicated")
        if evaluated:
            evaluations += 1
        attempt = WorkflowAttempt(
            cid,
            sid,
            _digest(row["fingerprint"], "stage input"),
            MappingProxyType(dependencies),
            status,
            _freeze(output),
            None if digest is None else _digest(digest, "original output"),
            None if reason is None else str(reason),
            evaluated,
        )
        entries.append(attempt)
        latest[(cid, sid)] = attempt
    if (
        type(body["evaluations"]) is not int
        or body["evaluations"] != evaluations
        or evaluations > definition.sweep.evaluation_budget
    ):
        raise ValueError("journal stage evaluation budget or accounting differs")
    state = body["state"]
    if not isinstance(state, str) or state not in {"partial", "cancelled", "complete"}:
        raise ValueError("unsupported journal completion state")
    if state == "complete":
        for cell in cells:
            for sid in topological_order(definition):
                entry = latest.get((cell.id, sid))
                if entry is None or entry.status != "complete":
                    raise ValueError("journal has incomplete original coordinates")
                for parent, digest in entry.dependencies.items():
                    original = latest[(cell.id, parent)]
                    if original.status != "complete" or digest != original.output_digest:
                        raise ValueError("completed journal dependency has changed")
    extensions = _object(document["extensions"], None, "journal extensions")
    return WorkflowJournal(
        workflow_digest,
        source,
        runtime,
        tuple(entries),
        state,
        evaluations,
        MappingProxyType({key: _freeze(value) for key, value in extensions.items()}),
    )


def create_workflow_journal(
    definition: WorkflowDefinition, *, source_fingerprint: str, runtime_fingerprint: str
) -> WorkflowJournal:
    """Create a partial empty checkpoint bound to supplied original source/build evidence.

    Parameters
    ----------
    definition
        Exact admitted workflow and bounded coordinate plan.
    source_fingerprint
        Actual original source identity from the owning runtime adapter.
    runtime_fingerprint
        Actual environment/build identity; no unavailable placeholder is accepted.

    Returns
    -------
    WorkflowJournal
        Admitted empty partial history, with zero execution claim.

    Raises
    ------
    ValueError
        Definition, source/runtime identity or resource admission fails.

    """
    return parse_workflow_journal(
        {
            "schema": JOURNAL_SCHEMA,
            "body": {
                "workflow_digest": canonical_digest(
                    "studio.workflow-definition.v1", definition.to_dict()
                ),
                "source_fingerprint": source_fingerprint,
                "runtime_fingerprint": runtime_fingerprint,
                "entries": [],
                "state": "partial",
                "evaluations": 0,
            },
            "extensions": {},
        },
        definition,
    )
