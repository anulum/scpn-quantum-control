# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — private Atlas overlay of an admitted Studio workspace
"""Project an admitted Studio workspace into a private Atlas snapshot.

The public Atlas pin holds what the repository commits. A workspace holds what
one person drafted and ran in the Studio. This module joins the two into a
snapshot whose ``access_scope`` is ``private``: every public node and edge
unchanged, plus one private node per workspace object.

- an experiment revision becomes ``revision:sha256:<revision hash>``;
- a parameter specification becomes ``parameter:workspace.<project>/<key>``;
- a program its revisions name becomes ``program:workspace.<project>/<digest>``;
- a local run record becomes ``run:<run id>/<attempt id>``.

Edges use only the frozen endpoint table. A revision ``derives_from`` its
parents, a program ``consumes`` each revision that names it and each parameter
those revisions bind, and a run ``consumes`` its revision. An object that cannot
take an Atlas identity is listed as an exclusion with its reason, never renamed.

The input is a workspace view that the owner's graph admission has already
accepted (:func:`scpn_quantum_control.studio_workspace.admit_workspace` in
Python, ``admitWorkspaceArchive`` in the browser). The browser twin
``studio-web/src/modules/atlas/workspaceOverlay.ts`` must reproduce every
snapshot digest this module computes on the committed cases.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from ...studio_workspace.canonical import canonical_digest
from ..contracts import (
    RELATION_VERSION,
    AtlasSnapshot,
    atlas_snapshot_digest,
    parse_atlas_snapshot,
    snapshot_to_wire,
)
from .snapshot import EVIDENCE_SET_SCHEMA

WORKSPACE_OVERLAY_VERSION = "workspace-overlay.1"
"""Version of the projection rules, appended to the pin's extractor version."""

_HEX64 = re.compile(r"[0-9a-f]{64}")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_KEY = re.compile(r"[A-Za-z0-9_.:-]+")
_NO_TIME = "workspace documents carry no observation time"
_NO_DOMAIN = "the workspace states no validity scope"


class WorkspaceOverlayError(ValueError):
    """A workspace view or a previous overlay cannot be projected.

    Parameters
    ----------
    path
        Address of the refused field.
    message
        Why it was refused.

    """

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path


@dataclass(frozen=True, slots=True)
class WorkspaceView:
    """The admitted workspace facts the overlay reads.

    Parameters
    ----------
    project_id
        The workspace project UUID.
    workspace_hash
        Digest of the admitted workspace root document.
    updated_at
        The root's ``updated_at``; it dates tombstones of removed objects.
    documents
        Admitted documents by digest, each with its ``schema`` and ``body``.

    """

    project_id: str
    workspace_hash: str
    updated_at: str
    documents: Mapping[str, Mapping[str, object]]


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise WorkspaceOverlayError(path, "must be an object")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkspaceOverlayError(path, "must be non-empty text")
    return value


def _digest(value: object, path: str) -> str:
    text = _text(value, path)
    if not _HEX64.fullmatch(text):
        raise WorkspaceOverlayError(path, "must be a lowercase sha256 digest")
    return text


def parse_workspace_view(payload: object) -> WorkspaceView:
    """Read a workspace view from its wire form.

    Parameters
    ----------
    payload
        ``{"project_id", "workspace_hash", "updated_at", "documents"}``. Each
        document is ``{"schema", "body"}`` under its digest.

    Returns
    -------
    WorkspaceView
        The view, with read-only document mappings.

    Raises
    ------
    WorkspaceOverlayError
        A field is missing, of the wrong type or not a valid identity.

    """
    view = _mapping(payload, "$")
    project_id = _text(view.get("project_id"), "$.project_id")
    if not _UUID.fullmatch(project_id):
        raise WorkspaceOverlayError("$.project_id", "must be a lowercase UUID")
    documents: dict[str, Mapping[str, object]] = {}
    for digest, document in _mapping(view.get("documents"), "$.documents").items():
        path = f"$.documents.{digest}"
        record = _mapping(document, path)
        documents[_digest(digest, path)] = MappingProxyType(
            {
                "schema": _text(record.get("schema"), f"{path}.schema"),
                "body": _mapping(record.get("body"), f"{path}.body"),
            }
        )
    return WorkspaceView(
        project_id=project_id,
        workspace_hash=_digest(view.get("workspace_hash"), "$.workspace_hash"),
        updated_at=_text(view.get("updated_at"), "$.updated_at"),
        documents=MappingProxyType(documents),
    )


class _Fragment:
    """Collect the private nodes, edges and exclusions of one workspace."""

    def __init__(self, view: WorkspaceView) -> None:
        self.view = view
        self.source = f"workspace/{view.project_id}"
        self.nodes: dict[str, dict[str, object]] = {}
        self.edges: dict[tuple[str, str, str], dict[str, object]] = {}
        self.exclusions: set[tuple[str, str]] = set()

    def node(self, node_id: str, kind: str, label: str, digest: str | None, why: str) -> None:
        reasons = {"as_of": _NO_TIME, "validity_domain": _NO_DOMAIN}
        if digest is None:
            reasons["source_digest"] = why
        self.nodes[node_id] = {
            "id": node_id,
            "kind": kind,
            "label": label,
            "source_ref": self.source,
            "source_digest": digest,
            "validity_domain": None,
            "as_of": None,
            "access": "private",
            "status": "current",
            "evidence_refs": [self.view.workspace_hash],
            "null_reasons": reasons,
        }

    def edge(self, source: str, kind: str, target: str) -> None:
        self.edges[(source, kind, target)] = {
            "source": source,
            "kind": kind,
            "target": target,
            "relation_version": RELATION_VERSION,
            "derivation_method": "extracted",
            "source_ref": self.source,
            "evidence_digest": self.view.workspace_hash,
            "assumptions": [],
            "mapping": None,
            "as_of": None,
            "access": "private",
        }

    def exclude(self, digest: str, reason: str) -> None:
        self.exclusions.add((f"{self.source}/{digest}", reason))


def _of_schema(view: WorkspaceView, schema: str) -> list[tuple[str, Mapping[str, object]]]:
    return [
        (digest, _mapping(document["body"], f"$.documents.{digest}.body"))
        for digest, document in sorted(view.documents.items())
        if document["schema"] == schema
    ]


def _parameters(fragment: _Fragment) -> dict[str, str]:
    keys: dict[str, list[str]] = {}
    for digest, body in _of_schema(fragment.view, "parameter_spec.v1"):
        key = _text(body.get("key"), f"$.documents.{digest}.body.key")
        if _KEY.fullmatch(key):
            keys.setdefault(key, []).append(digest)
        else:
            fragment.exclude(digest, "the parameter key is not an Atlas identifier")
    nodes: dict[str, str] = {}
    for key, digests in sorted(keys.items()):
        node_id = f"parameter:workspace.{fragment.view.project_id}/{key}"
        single = digests[0] if len(digests) == 1 else None
        fragment.node(
            node_id, "parameter", key, single, "the workspace holds several specifications of it"
        )
        nodes[key] = node_id
    return nodes


def _revisions(fragment: _Fragment, parameters: Mapping[str, str]) -> None:
    view = fragment.view
    revisions = _of_schema(view, "experiment_revision.v1")
    present = {digest for digest, _ in revisions}
    for digest, body in revisions:
        path = f"$.documents.{digest}.body"
        node_id = f"revision:sha256:{digest}"
        fragment.node(node_id, "revision", f"Experiment revision {digest[:12]}", digest, "")
        parents = body.get("parent_revision_hashes")
        if not isinstance(parents, list):
            raise WorkspaceOverlayError(f"{path}.parent_revision_hashes", "must be an array")
        for index, parent in enumerate(parents):
            parent_digest = _digest(parent, f"{path}.parent_revision_hashes[{index}]")
            if parent_digest in present:
                fragment.edge(node_id, "derives_from", f"revision:sha256:{parent_digest}")
            else:
                fragment.exclude(
                    digest, f"parent revision {parent_digest} is not in the workspace"
                )
        program = _digest(
            _mapping(body.get("program_ref"), f"{path}.program_ref").get("sha256"),
            f"{path}.program_ref.sha256",
        )
        program_id = f"program:workspace.{view.project_id}/{program}"
        fragment.node(program_id, "program", f"Program {program[:12]}", program, "")
        fragment.edge(program_id, "consumes", node_id)
        for key in sorted(_mapping(body.get("parameters"), f"{path}.parameters")):
            if key in parameters:
                fragment.edge(program_id, "consumes", parameters[key])


def _runs(fragment: _Fragment) -> None:
    for digest, body in _of_schema(fragment.view, "local_run_record.v1"):
        path = f"$.documents.{digest}.body"
        run_id = _text(body.get("run_id"), f"{path}.run_id")
        attempt_id = _text(body.get("attempt_id"), f"{path}.attempt_id")
        revision = _digest(body.get("revision_hash"), f"{path}.revision_hash")
        if not (_UUID.fullmatch(run_id) and _UUID.fullmatch(attempt_id)):
            fragment.exclude(digest, "the run or attempt identifier is not a lowercase UUID")
            continue
        node_id = f"run:{run_id}/{attempt_id}"
        fragment.node(node_id, "run", f"Run {run_id[:8]} attempt {attempt_id[:8]}", digest, "")
        if f"revision:sha256:{revision}" in fragment.nodes:
            fragment.edge(node_id, "consumes", f"revision:sha256:{revision}")
        else:
            fragment.exclude(digest, f"revision {revision} is not in the workspace")


def _tombstones(
    public: AtlasSnapshot, previous: AtlasSnapshot | None, live: set[str], removed_at: str
) -> list[dict[str, object]]:
    carried: list[dict[str, object]] = [
        {
            "id": tombstone.id,
            "kind": tombstone.kind,
            "last_source_digest": tombstone.last_source_digest,
            "removed_at": tombstone.removed_at,
            "reason": tombstone.reason,
        }
        for tombstone in public.tombstones
    ]
    if previous is None:
        return carried
    if previous.access_scope != "private":
        raise WorkspaceOverlayError("previous", "must be an earlier private overlay")
    seen = {str(item["id"]) for item in carried}
    for tombstone in previous.tombstones:
        if tombstone.id not in seen and tombstone.id not in live:
            seen.add(tombstone.id)
            carried.append(
                {
                    "id": tombstone.id,
                    "kind": tombstone.kind,
                    "last_source_digest": tombstone.last_source_digest,
                    "removed_at": tombstone.removed_at,
                    "reason": tombstone.reason,
                }
            )
    for node in previous.nodes:
        if node.access == "private" and node.id not in live and node.id not in seen:
            carried.append(
                {
                    "id": node.id,
                    "kind": node.kind,
                    "last_source_digest": node.source_digest,
                    "removed_at": removed_at,
                    "reason": "no longer in the admitted workspace",
                }
            )
    return carried


def overlay_workspace(
    public: AtlasSnapshot, view: WorkspaceView, *, previous: AtlasSnapshot | None = None
) -> AtlasSnapshot:
    """Join a public snapshot and one admitted workspace into a private snapshot.

    Parameters
    ----------
    public
        The public Atlas snapshot, normally the committed pin.
    view
        The admitted workspace.
    previous
        An earlier overlay. Its private nodes that are no longer present become
        tombstones dated by ``view.updated_at``.

    Returns
    -------
    AtlasSnapshot
        A validated private snapshot holding every public node and edge unchanged.

    Raises
    ------
    WorkspaceOverlayError
        ``public`` is not public, the view is malformed, ``previous`` is not a
        private overlay, or a workspace identity collides with a public one.

    """
    if public.access_scope != "public":
        raise WorkspaceOverlayError("public", "must be a public snapshot")
    fragment = _Fragment(view)
    _revisions(fragment, _parameters(fragment))
    _runs(fragment)
    wire = snapshot_to_wire(public)
    nodes = {str(node["id"]): node for node in _records(wire, "nodes")}
    for node_id, node in fragment.nodes.items():
        if node_id in nodes:
            raise WorkspaceOverlayError(node_id, "collides with a public node")
        nodes[node_id] = node
    edges = [*_records(wire, "edges"), *fragment.edges.values()]
    exclusions = {
        (str(item["source_ref"]), str(item["reason"])) for item in _records(wire, "exclusions")
    }
    evidence = sorted(
        [
            ["atlas-pin", atlas_snapshot_digest(public)],
            [fragment.source, view.workspace_hash],
        ]
    )
    document = {
        "schema": wire["schema"],
        "extractor_version": f"{public.extractor_version}+{WORKSPACE_OVERLAY_VERSION}",
        "source_commit": public.source_commit,
        "evidence_set_hash": canonical_digest(EVIDENCE_SET_SCHEMA, evidence),
        "access_scope": "private",
        "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": sorted(
            edges,
            key=lambda e: (
                str(e["source"]),
                str(e["kind"]),
                str(e["target"]),
                str(e["evidence_digest"]),
            ),
        ),
        "tombstones": _tombstones(public, previous, set(nodes), view.updated_at),
        "exclusions": [
            {"source_ref": source_ref, "reason": reason}
            for source_ref, reason in sorted(exclusions | fragment.exclusions)
        ],
    }
    return parse_atlas_snapshot(document)


def _records(wire: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    return list(cast(list[Mapping[str, object]], wire[key]))


__all__ = [
    "WORKSPACE_OVERLAY_VERSION",
    "WorkspaceOverlayError",
    "WorkspaceView",
    "overlay_workspace",
    "parse_workspace_view",
]
