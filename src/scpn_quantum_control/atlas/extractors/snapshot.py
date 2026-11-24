# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — deterministic Atlas snapshot extraction
"""Build an Atlas snapshot from a closed list of committed producer artefacts.

The extractor reads bytes only. It never imports or runs what an artefact
describes, so discovery needs no optional quantum runtime. Four adapters are
admitted:

- ``architecture_map``: the committed differentiable architecture map
  (``scpn_qc_differentiable_architecture_map_v1``). Each layer becomes a
  capability. Its implementation files ``implements`` it, it ``contains`` its
  test files, and its documentation files ``documents`` it;
- ``claim_ledger``: the reviewed differentiable claim ledger. Each claim becomes
  a claim node that ``derives_from`` its implementation files and is
  ``documents``-ed by its documentation files;
- ``hardware_result_packs``: the committed hardware result-pack manifest. Each
  pack becomes a content-addressed artefact that ``derives_from`` each of its
  raw-count, summary and analysis files, by their recorded digests. No support
  edge is derived, because a pack's scope is not a reviewed capability
  assertion;
- ``evidence_lineage``: the committed evidence lineage
  (:mod:`scpn_quantum_control.studio.evidence_lineage`). A claim ``derives_from``
  each evidence file its identifier resolves to exactly one of. An ambiguous or
  missing identifier is recorded as an exclusion, never linked to a guessed
  file. Each replication analysis becomes an artefact whose label states its
  decision, so a failed replication stays searchable.

Every input becomes an artefact node bound to its content digest. Every
derived node and edge names that digest as evidence, so changing an input
changes the provenance of what it declared. A path that an input names but the
source does not hold, an identifier this extractor cannot resolve, and an
optional input that is absent are recorded as scoped exclusions, never as
invented nodes. A malformed input, or an absent required one, refuses the whole
extraction.

The result is validated by :func:`~scpn_quantum_control.atlas.contracts.parse_atlas_snapshot`
before it is returned. Against a previous snapshot, every node that is no
longer extracted becomes a tombstone, and earlier tombstones carry forward. An
extraction that reuses unchanged input fragments yields exactly the snapshot a
full rebuild yields.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ...studio_workspace.canonical import canonical_digest
from ..contracts import (
    RELATION_VERSION,
    SNAPSHOT_SCHEMA,
    AtlasSnapshot,
    parse_atlas_snapshot,
)

EXTRACTOR_VERSION = "atlas-extractor.2"
EVIDENCE_SET_SCHEMA = "quantum_atlas_evidence_set.v1"
ARCHITECTURE_MAP_SCHEMA = "scpn_qc_differentiable_architecture_map_v1"

SourceReader = Callable[[str], bytes | None]


@dataclass(frozen=True)
class AtlasInput:
    """One admitted extraction input.

    Parameters
    ----------
    path
        Repository-relative POSIX path of the committed artefact.
    adapter
        ``"architecture_map"``, ``"claim_ledger"`` or ``"hardware_result_packs"``.
    required
        Whether an absent input refuses the extraction instead of being excluded.

    """

    path: str
    adapter: str
    required: bool


DEFAULT_INPUTS: tuple[AtlasInput, ...] = (
    AtlasInput(
        "data/differentiable_phase_qnode/differentiable_architecture_map_20260627.json",
        "architecture_map",
        True,
    ),
    AtlasInput("data/differentiable_phase_qnode/claim_ledger.json", "claim_ledger", False),
    AtlasInput("data/hardware_result_packs/manifest.json", "hardware_result_packs", False),
    AtlasInput("data/studio/evidence_lineage_20261123.json", "evidence_lineage", False),
)

_CODE_SUFFIXES = frozenset({".py", ".rs", ".ts", ".tsx", ".js", ".jl"})
_DOC_SUFFIXES = frozenset({".md", ".json", ".txt", ".csv", ".yml", ".yaml"})
_PATH = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")
_NAME = re.compile(r"[A-Za-z0-9_.:-]+")


class AtlasExtractionError(ValueError):
    """Refusal of a whole extraction, with a stable code and the input it concerns.

    Parameters
    ----------
    code
        Stable refusal category.
    path
        Input path, or ``"$inputs"`` for the input list itself.
    message
        Human-readable reason.

    """

    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(f"{code} at {path}: {message}")
        self.code = code
        self.path = path
        self.message = message


@dataclass(frozen=True)
class AtlasFragment:
    """Nodes, edges and exclusions one input contributed, keyed by what it read.

    Parameters
    ----------
    input_digest
        SHA-256 of the input bytes.
    read_digests
        ``(path, digest or None)`` for every file the adapter read, sorted.
    nodes
        Wire nodes.
    edges
        Wire edges.
    exclusions
        Wire exclusions.

    """

    input_digest: str
    read_digests: tuple[tuple[str, str | None], ...]
    nodes: tuple[dict[str, object], ...]
    edges: tuple[dict[str, object], ...]
    exclusions: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class AtlasExtraction:
    """A validated snapshot plus the fragments that a later extraction may reuse.

    Parameters
    ----------
    snapshot
        The validated snapshot.
    fragments
        Fragment per input path.

    """

    snapshot: AtlasSnapshot
    fragments: Mapping[str, AtlasFragment]


def directory_reader(root: Path) -> SourceReader:
    """Return a reader of regular files below ``root``.

    Parameters
    ----------
    root
        Directory that holds the materialised source revision.

    Returns
    -------
    SourceReader
        Callable that returns a file's bytes, or ``None`` when no regular,
        non-symlinked file exists at that path inside ``root``.

    """
    base = root.resolve()

    def read(relative: str) -> bytes | None:
        candidate = base / relative
        if candidate.is_symlink() or not candidate.is_file():
            return None
        if base not in candidate.resolve().parents:
            return None
        return candidate.read_bytes()

    return read


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _refuse(code: str, path: str, message: str) -> AtlasExtractionError:
    return AtlasExtractionError(code, path, message)


def _json_object(data: bytes, path: str) -> Mapping[str, object]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _refuse("malformed_input", path, f"not UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise _refuse("malformed_input", path, "expected a JSON object")
    return value


def _records(document: Mapping[str, object], key: str, path: str) -> list[Mapping[str, object]]:
    value = document.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise _refuse("malformed_input", path, f"{key} must be an array of objects")
    return value


def _text(record: Mapping[str, object], key: str, path: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise _refuse("malformed_input", path, f"{key} must be a nonempty string")
    return value


def _texts(record: Mapping[str, object], key: str, path: str) -> list[str]:
    value = record.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise _refuse("malformed_input", path, f"{key} must be an array of nonempty strings")
    return value


def _name(value: str, path: str) -> str:
    if not _NAME.fullmatch(value):
        raise _refuse("malformed_input", path, f"{value!r} cannot form an Atlas identity")
    return value


class _Builder:
    """Collect one input's fragment while reading the files it names."""

    def __init__(self, item: AtlasInput, data: bytes, read: SourceReader) -> None:
        self.item = item
        self.digest = _sha256(data)
        self.read = read
        self.nodes: dict[str, dict[str, object]] = {}
        self.edges: dict[tuple[str, str, str], dict[str, object]] = {}
        self.exclusions: set[tuple[str, str]] = set()
        self.read_digests: dict[str, str | None] = {}

    def node(
        self,
        node_id: str,
        kind: str,
        label: str,
        *,
        source_ref: str,
        source_digest: str | None,
        domain: Mapping[str, str] | None,
    ) -> None:
        """Record one node once, with the reason for every field the input leaves null.

        Parameters
        ----------
        node_id
            Stable node identifier; a repeated identifier keeps its first record.
        kind
            Node kind.
        label
            Human-readable label.
        source_ref
            Where the declaration comes from.
        source_digest
            Digest of the declared file, or ``None`` for a declaration inside the input.
        domain
            Validity scope the input states, or ``None``.

        """
        reasons = {"as_of": "committed declarations carry no observation time"}
        if source_digest is None:
            reasons["source_digest"] = "a declaration inside its input, without its own content"
        if domain is None:
            reasons["validity_domain"] = "the input states no validity scope"
        if node_id in self.nodes:
            return
        self.nodes[node_id] = {
            "id": node_id,
            "kind": kind,
            "label": label,
            "source_ref": source_ref,
            "source_digest": source_digest,
            "validity_domain": None if domain is None else dict(domain),
            "as_of": None,
            "access": "public",
            "status": "current",
            "evidence_refs": [self.digest],
            "null_reasons": reasons,
        }

    def edge(self, source: str, kind: str, target: str) -> None:
        """Record one declared relation of this input.

        Parameters
        ----------
        source
            Source node identifier.
        kind
            Relation kind.
        target
            Target node identifier.

        """
        self.edges[(source, kind, target)] = {
            "source": source,
            "kind": kind,
            "target": target,
            "relation_version": RELATION_VERSION,
            "derivation_method": "declared",
            "source_ref": self.item.path,
            "evidence_digest": self.digest,
            "assumptions": [],
            "mapping": None,
            "as_of": None,
            "access": "public",
        }

    def exclude(self, source_ref: str, reason: str) -> None:
        """Record one reference the input names but the snapshot leaves out, with why.

        Parameters
        ----------
        source_ref
            The excluded reference.
        reason
            Why it was left out.

        """
        self.exclusions.add((source_ref, reason))

    def file_node(self, relative: str, role: str, declared_at: str) -> str | None:
        """Return the node ID of a named repository file, or ``None`` after excluding it."""
        if not _PATH.fullmatch(relative) or ".." in PurePosixPath(relative).parts:
            self.exclude(f"{declared_at}:{relative}", "not a safe repository-relative path")
            return None
        suffix = PurePosixPath(relative).suffix
        if role == "test":
            kind = "test"
        elif role == "documentation" or suffix in _DOC_SUFFIXES:
            kind = "documentation"
        elif suffix in _CODE_SUFFIXES:
            kind = "definition"
        else:
            self.exclude(f"{declared_at}:{relative}", "file type maps to no admitted node kind")
            return None
        data = self.read(relative)
        self.read_digests[relative] = None if data is None else _sha256(data)
        if data is None:
            self.exclude(
                f"{declared_at}:{relative}", "named by the input but absent from the source"
            )
            return None
        node_id = f"{kind}:repo/{relative.replace('/', ':')}"
        self.node(
            node_id, kind, relative, source_ref=relative, source_digest=_sha256(data), domain=None
        )
        return node_id

    def fragment(self) -> AtlasFragment:
        """Return this input's fragment in canonical order.

        Returns
        -------
        AtlasFragment
            Nodes, relations and exclusions sorted by key, with every digest the input read.

        """
        return AtlasFragment(
            input_digest=self.digest,
            read_digests=tuple(sorted(self.read_digests.items())),
            nodes=tuple(self.nodes[key] for key in sorted(self.nodes)),
            edges=tuple(self.edges[key] for key in sorted(self.edges)),
            exclusions=tuple(
                {"source_ref": source_ref, "reason": reason}
                for source_ref, reason in sorted(self.exclusions)
            ),
        )


def _input_node(builder: _Builder, label: str, domain: Mapping[str, str]) -> None:
    builder.node(
        f"artefact:sha256:{builder.digest}",
        "artefact",
        label,
        source_ref=builder.item.path,
        source_digest=builder.digest,
        domain=domain,
    )


def _architecture_map(builder: _Builder, document: Mapping[str, object]) -> None:
    path = builder.item.path
    if document.get("schema") != ARCHITECTURE_MAP_SCHEMA:
        raise _refuse("unsupported_input_schema", path, f"expected {ARCHITECTURE_MAP_SCHEMA}")
    _input_node(
        builder,
        f"Architecture map {_text(document, 'artifact_id', path)}",
        {
            "schema": ARCHITECTURE_MAP_SCHEMA,
            "claim_boundary": _text(document, "claim_boundary", path),
        },
    )
    for layer in _records(document, "layers", path):
        layer_id = _name(_text(layer, "layer_id", path), path)
        capability = f"capability:atlas.architecture/{layer_id}"
        builder.node(
            capability,
            "capability",
            _text(layer, "title", path),
            source_ref=path,
            source_digest=None,
            domain={
                "role": _text(layer, "role", path),
                "claim_boundary": _text(layer, "claim_boundary", path),
                "rustification_ready": "true"
                if layer.get("rustification_ready") is True
                else "false",
                "blockers": "; ".join(_texts(layer, "blockers", path)) or "none recorded",
            },
        )
        declared_at = f"{path}#{layer_id}"
        for key in ("owner_modules", "python_surfaces", "rust_surfaces", "polyglot_surfaces"):
            for relative in _texts(layer, key, path):
                node_id = builder.file_node(relative, "implementation", declared_at)
                if node_id is None:
                    continue
                if node_id.startswith("definition:"):
                    builder.edge(node_id, "implements", capability)
                else:
                    builder.edge(node_id, "documents", capability)
        for relative in _texts(layer, "test_surfaces", path):
            node_id = builder.file_node(relative, "test", declared_at)
            if node_id is not None:
                builder.edge(capability, "contains", node_id)
        for relative in _texts(layer, "docs_surfaces", path):
            node_id = builder.file_node(relative, "documentation", declared_at)
            if node_id is not None:
                builder.edge(node_id, "documents", capability)
        for key in ("inventory_surface_ids", "benchmark_surfaces"):
            for identifier in _texts(layer, key, path):
                builder.exclude(
                    f"{declared_at}:{identifier}",
                    "surface identifier; the inventory that resolves it is not an Atlas input",
                )


def _claim_ledger(builder: _Builder, document: Mapping[str, object]) -> None:
    path = builder.item.path
    _input_node(
        builder,
        f"Claim ledger {_text(document, 'artifact_id', path)}",
        {"schema": _text(document, "schema", path)},
    )
    for claim in _records(document, "claims", path):
        claim_id = _name(_text(claim, "claim_id", path), path)
        node_id = f"claim:differentiable.claim_ledger/{claim_id}"
        builder.node(
            node_id,
            "claim",
            _text(claim, "claim_text", path),
            source_ref=path,
            source_digest=None,
            domain={"claim_boundary": _text(claim, "claim_boundary", path)},
        )
        declared_at = f"{path}#{claim_id}"
        for relative in _texts(claim, "implementation_surface", path):
            target = builder.file_node(relative, "implementation", declared_at)
            if target is not None and target.startswith("definition:"):
                builder.edge(node_id, "derives_from", target)
            elif target is not None:
                builder.edge(target, "documents", node_id)
        for relative in _texts(claim, "docs_surface", path):
            source = builder.file_node(relative, "documentation", declared_at)
            if source is not None:
                builder.edge(source, "documents", node_id)
        for key in ("evidence_artifact_ids", "benchmark_artifact_ids"):
            for identifier in _texts(claim, key, path):
                builder.exclude(
                    f"{declared_at}:{identifier}",
                    "evidence artefact identifier; its record is not an Atlas input",
                )


def _hardware_result_packs(builder: _Builder, document: Mapping[str, object]) -> None:
    path = builder.item.path
    _input_node(
        builder,
        "Hardware result-pack manifest",
        {"claim_boundary": _text(document, "claim_boundary", path)},
    )
    for pack in _records(document, "packs", path):
        pack_id = _name(_text(pack, "id", path), path)
        digest = canonical_digest("quantum_atlas_result_pack.v1", dict(pack))
        builder.node(
            f"artefact:sha256:{digest}",
            "artefact",
            _text(pack, "title", path),
            source_ref=f"{path}#{pack_id}",
            source_digest=digest,
            domain={
                key: value
                for key in ("backend", "hardware_family", "executed_utc", "status", "claim_scope")
                if isinstance(value := pack.get(key), str) and value
            }
            or None,
        )
        listed = _records(pack, "artifacts", f"{path}#{pack_id}") if "artifacts" in pack else []
        for artefact in listed:
            role = _text(artefact, "role", path)
            relative = _text(artefact, "path", path)
            recorded = _text(artefact, "sha256", path)
            target = f"artefact:sha256:{recorded}"
            builder.node(
                target,
                "artefact",
                f"{pack_id} {role}: {relative}",
                source_ref=relative,
                source_digest=recorded,
                domain={"pack": pack_id, "role": role},
            )
            builder.edge(f"artefact:sha256:{digest}", "derives_from", target)


def _evidence_lineage(builder: _Builder, document: Mapping[str, object]) -> None:
    path = builder.item.path
    _input_node(builder, "Evidence lineage", {"schema": _text(document, "schema", path)})
    for claim in _records(document, "claims", path):
        claim_id = _name(_text(claim, "claim_id", path), path)
        node_id = f"claim:differentiable.claim_ledger/{claim_id}"
        builder.node(
            node_id,
            "claim",
            _text(claim, "claim_text", path),
            source_ref=f"{path}#{claim_id}",
            source_digest=None,
            domain={"claim_boundary": _text(claim, "claim_boundary", path)},
        )
        for item in _records(claim, "evidence", f"{path}#{claim_id}"):
            identifier = _text(item, "artifact_id", path)
            state = _text(item, "state", path)
            candidates = _records(item, "candidates", path)
            if state != "resolved" or len(candidates) != 1:
                builder.exclude(
                    f"{path}#{claim_id}:{identifier}",
                    f"evidence identifier {state} with {len(candidates)} committed candidates",
                )
                continue
            relative = _text(candidates[0], "path", path)
            recorded = _text(candidates[0], "sha256", path)
            target = f"artefact:sha256:{recorded}"
            builder.node(
                target,
                "artefact",
                identifier,
                source_ref=relative,
                source_digest=recorded,
                domain={"artifact_id": identifier, "role": _text(item, "role", path)},
            )
            builder.edge(node_id, "derives_from", target)
    for replication in _records(document, "replications", path):
        relative = _text(replication, "path", path)
        decision = replication.get("decision")
        field = _text(decision, "field", path) if isinstance(decision, dict) else ""
        value = decision.get("value") if isinstance(decision, dict) else None
        if not field or not isinstance(value, bool):
            raise _refuse("malformed_input", path, "a replication decision must be a boolean")
        verdict = "true" if value else "false"
        campaign = _text(replication, "campaign", path)
        builder.node(
            f"artefact:sha256:{_text(replication, 'sha256', path)}",
            "artefact",
            f"Replication {campaign}: {field} = {verdict}",
            source_ref=relative,
            source_digest=_text(replication, "sha256", path),
            domain={"campaign": campaign, "decision": field, "value": verdict},
        )


_ADAPTERS: Mapping[str, Callable[[_Builder, Mapping[str, object]], None]] = {
    "architecture_map": _architecture_map,
    "claim_ledger": _claim_ledger,
    "hardware_result_packs": _hardware_result_packs,
    "evidence_lineage": _evidence_lineage,
}


def _check_inputs(inputs: Sequence[AtlasInput]) -> list[AtlasInput]:
    paths = [item.path for item in inputs]
    if not inputs or len(set(paths)) != len(paths):
        raise _refuse(
            "invalid_inputs", "$inputs", "inputs must be a nonempty list of distinct paths"
        )
    for item in inputs:
        if item.adapter not in _ADAPTERS:
            raise _refuse(
                "unknown_adapter", item.path, f"{item.adapter!r} is not an admitted adapter"
            )
        if not _PATH.fullmatch(item.path) or ".." in PurePosixPath(item.path).parts:
            raise _refuse("invalid_inputs", item.path, "not a safe repository-relative path")
    return sorted(inputs, key=lambda item: item.path)


def _reusable(fragment: AtlasFragment | None, digest: str, read: SourceReader) -> bool:
    if fragment is None or fragment.input_digest != digest:
        return False
    for relative, recorded in fragment.read_digests:
        data = read(relative)
        if (None if data is None else _sha256(data)) != recorded:
            return False
    return True


def _tombstones(
    previous: AtlasSnapshot | None, live: set[str], removed_at: str | None
) -> list[dict[str, object]]:
    if previous is None:
        return []
    carried: list[dict[str, object]] = [
        {
            "id": tombstone.id,
            "kind": tombstone.kind,
            "last_source_digest": tombstone.last_source_digest,
            "removed_at": tombstone.removed_at,
            "reason": tombstone.reason,
        }
        for tombstone in previous.tombstones
    ]
    removed = [node for node in previous.nodes if node.id not in live]
    if removed and removed_at is None:
        raise _refuse("missing_removal_time", "$inputs", "removed nodes need a removal time")
    carried.extend(
        {
            "id": node.id,
            "kind": node.kind,
            "last_source_digest": node.source_digest,
            "removed_at": removed_at,
            "reason": "no longer extracted from the admitted inputs",
        }
        for node in removed
    )
    return carried


def extract_atlas_snapshot(
    read: SourceReader,
    source_commit: str,
    *,
    inputs: Sequence[AtlasInput] = DEFAULT_INPUTS,
    previous: AtlasSnapshot | None = None,
    removed_at: str | None = None,
    reuse: Mapping[str, AtlasFragment] | None = None,
) -> AtlasExtraction:
    """Extract and validate an Atlas snapshot of one source revision.

    Parameters
    ----------
    read
        Reader of the revision's files, for example :func:`directory_reader`.
    source_commit
        Full 40-hex commit the bytes belong to.
    inputs
        Closed list of admitted inputs; its order does not matter.
    previous
        Earlier snapshot whose deleted nodes become tombstones.
    removed_at
        UTC time recorded on new tombstones; required when a node was removed.
    reuse
        Fragments of an earlier extraction. A fragment is reused only when its
        input and every file it read are byte-identical.

    Returns
    -------
    AtlasExtraction
        The validated public snapshot and the fragments of this extraction.

    Raises
    ------
    AtlasExtractionError
        The input list is invalid, a required input is absent, or an input is malformed.
    AtlasContractError
        The assembled document breaks the snapshot contract.

    """
    fragments: dict[str, AtlasFragment] = {}
    exclusions: list[dict[str, str]] = []
    evidence: list[list[str]] = []
    for item in _check_inputs(inputs):
        data = read(item.path)
        if data is None:
            if item.required:
                raise _refuse(
                    "missing_required_input", item.path, "absent from the source revision"
                )
            exclusions.append(
                {
                    "source_ref": item.path,
                    "reason": "optional input absent from the source revision",
                }
            )
            continue
        digest = _sha256(data)
        evidence.append([item.path, digest])
        cached = None if reuse is None else reuse.get(item.path)
        if cached is not None and _reusable(cached, digest, read):
            fragments[item.path] = cached
            continue
        builder = _Builder(item, data, read)
        _ADAPTERS[item.adapter](builder, _json_object(data, item.path))
        fragments[item.path] = builder.fragment()
    nodes: dict[str, dict[str, object]] = {}
    refs: dict[str, set[str]] = {}
    edges: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for fragment in fragments.values():
        for node in fragment.nodes:
            node_id = str(node["id"])
            nodes.setdefault(node_id, node)
            refs.setdefault(node_id, set()).add(fragment.input_digest)
        for edge in fragment.edges:
            key = (
                str(edge["source"]),
                str(edge["kind"]),
                str(edge["target"]),
                fragment.input_digest,
            )
            edges[key] = edge
        exclusions.extend(fragment.exclusions)
    document = {
        "schema": SNAPSHOT_SCHEMA,
        "extractor_version": EXTRACTOR_VERSION,
        "source_commit": source_commit,
        "evidence_set_hash": canonical_digest(EVIDENCE_SET_SCHEMA, sorted(evidence)),
        "access_scope": "public",
        "nodes": [{**nodes[key], "evidence_refs": sorted(refs[key])} for key in sorted(nodes)],
        "edges": [edges[key] for key in sorted(edges)],
        "tombstones": _tombstones(previous, set(nodes), removed_at),
        "exclusions": [
            {"source_ref": source_ref, "reason": reason}
            for source_ref, reason in sorted({(e["source_ref"], e["reason"]) for e in exclusions})
        ],
    }
    return AtlasExtraction(snapshot=parse_atlas_snapshot(document), fragments=fragments)


__all__ = [
    "ARCHITECTURE_MAP_SCHEMA",
    "DEFAULT_INPUTS",
    "EVIDENCE_SET_SCHEMA",
    "EXTRACTOR_VERSION",
    "AtlasExtraction",
    "AtlasExtractionError",
    "AtlasFragment",
    "AtlasInput",
    "SourceReader",
    "directory_reader",
    "extract_atlas_snapshot",
]
