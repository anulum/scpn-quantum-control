# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Atlas snapshot ontology and identity contract
"""Validate Atlas snapshots: node and edge vocabulary, identity and legality.

An Atlas snapshot (``quantum_atlas_snapshot.v1``) is a deterministic graph of
what the repository holds and how it connects. This module owns the shared
contract that the browser counterpart in ``studio-web/src/shared/contracts/
atlas.ts`` implements against the same fixtures:

- twelve node kinds in three identity classes: a definition binds its qualified
  owner and name, a revision binds its content digest, and an instance binds a
  project UUID plus its own UUID;
- sixteen canonical edge kinds of relation version ``atlas_relation.v1``. The
  earlier design spellings ``produced``, ``derived_from``, ``verifies`` and
  ``depends_on`` are admitted as version-1 aliases and normalised; ``verifies``
  reverses direction into ``verified_by``. A bare ``contradicts`` is refused,
  because a contradiction is only stated under explicit assumptions;
- legality: every endpoint resolves to a live node of an admitted kind;
  ``supports``, ``verified_by`` and ``contradicts_under_assumptions`` carry
  evidence; ``supports`` and ``contradicts_under_assumptions`` carry their
  assumptions; import adjacency never creates a ``supports`` edge; a
  contradiction between claims with a different estimand or units is refused
  as non-comparable; ``maps_to`` carries its units and semantics;
- access: a public snapshot holds no private node or edge, an edge touching a
  private node is private, and the public projection drops private material
  without disclosing it through counts;
- history: a deleted object leaves a tombstone that keeps its exact identity,
  and a tombstoned ID is never reused by a live node.

Validation never fetches, imports or executes what a node describes. A refusal
raises :class:`AtlasContractError` with a stable code and a field path and
leaves the caller's data unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ..studio_workspace.canonical import canonical_digest

SNAPSHOT_SCHEMA = "quantum_atlas_snapshot.v1"
RELATION_VERSION = "atlas_relation.v1"

IdentityClass = Literal["definition", "revision", "instance"]

NODE_KINDS: tuple[str, ...] = (
    "definition",
    "instance",
    "revision",
    "parameter",
    "program",
    "backend",
    "run",
    "artefact",
    "claim",
    "test",
    "documentation",
    "capability",
)
IDENTITY_CLASSES: Mapping[str, IdentityClass] = {
    "definition": "definition",
    "parameter": "definition",
    "program": "definition",
    "backend": "definition",
    "claim": "definition",
    "test": "definition",
    "documentation": "definition",
    "capability": "definition",
    "revision": "revision",
    "artefact": "revision",
    "instance": "instance",
    "run": "instance",
}
EDGE_KINDS: tuple[str, ...] = (
    "implements",
    "consumes",
    "produces",
    "derives_from",
    "constrained_by",
    "lowers_to",
    "supports",
    "requires",
    "verified_by",
    "benchmarked_against",
    "calibrated_by",
    "supersedes",
    "contradicts_under_assumptions",
    "maps_to",
    "contains",
    "documents",
)
# Version-1 aliases of the earlier design vocabulary: alias -> (canonical kind, reversed).
EDGE_ALIASES_V1: Mapping[str, tuple[str, bool]] = {
    "produced": ("produces", False),
    "derived_from": ("derives_from", False),
    "verifies": ("verified_by", True),
    "depends_on": ("requires", False),
}
REFUSED_EDGE_KINDS: Mapping[str, str] = {
    "contradicts": "state the contradiction as contradicts_under_assumptions with its assumptions",
}
ACCESS_CLASSES: tuple[str, ...] = ("public", "private")
NODE_STATUSES: tuple[str, ...] = ("current", "stale", "incomplete")
DERIVATION_METHODS: tuple[str, ...] = (
    "extracted",
    "curated",
    "measured",
    "declared",
    "import_adjacency",
)
EVIDENCE_REQUIRED: frozenset[str] = frozenset(
    {"supports", "verified_by", "contradicts_under_assumptions"}
)
ASSUMPTIONS_REQUIRED: frozenset[str] = frozenset({"supports", "contradicts_under_assumptions"})

_DEFINITION_LIKE = ("definition", "parameter", "program", "capability")
# Admitted (source kinds, target kinds) per canonical edge kind.
EDGE_ENDPOINTS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "implements": (frozenset({"definition", "program"}), frozenset({"capability", "definition"})),
    "consumes": (
        frozenset({"run", "program", "definition", "instance"}),
        frozenset({"artefact", "revision", "parameter", "backend"}),
    ),
    "produces": (
        frozenset({"run", "program", "instance"}),
        frozenset({"artefact", "revision", "claim"}),
    ),
    "derives_from": (
        frozenset({"artefact", "revision", "claim", "definition", "instance"}),
        frozenset({"artefact", "revision", "claim", "definition"}),
    ),
    "constrained_by": (
        frozenset({"definition", "program", "parameter", "run", "capability"}),
        frozenset({"definition", "capability"}),
    ),
    "lowers_to": (frozenset({"program", "definition"}), frozenset({"program", "definition"})),
    "supports": (frozenset({"artefact", "run", "test"}), frozenset({"claim", "capability"})),
    "requires": (
        frozenset({"definition", "program", "capability", "run", "instance"}),
        frozenset({"definition", "capability", "backend", "parameter"}),
    ),
    "verified_by": (
        frozenset({"claim", "capability", "definition", "program", "artefact"}),
        frozenset({"test", "artefact", "run"}),
    ),
    "benchmarked_against": (
        frozenset({"program", "definition", "run", "artefact"}),
        frozenset({"artefact", "program", "definition"}),
    ),
    "calibrated_by": (frozenset({"backend", "run"}), frozenset({"artefact", "revision"})),
    "supersedes": (
        frozenset({"revision", "artefact", "claim", "definition", "documentation"}),
        frozenset({"revision", "artefact", "claim", "definition", "documentation"}),
    ),
    "contradicts_under_assumptions": (frozenset({"claim"}), frozenset({"claim"})),
    "maps_to": (frozenset(_DEFINITION_LIKE), frozenset(_DEFINITION_LIKE)),
    "contains": (
        frozenset({"definition", "program", "capability"}),
        frozenset(
            {
                "definition",
                "parameter",
                "program",
                "test",
                "documentation",
                "capability",
                "instance",
            }
        ),
    ),
    "documents": (frozenset({"documentation"}), frozenset(NODE_KINDS) - {"documentation"}),
}
# PROV-O concept for each provenance relation; other kinds are not provenance.
PROV_RELATIONS: Mapping[str, str] = {
    "derives_from": "prov:wasDerivedFrom",
    "produces": "prov:wasGeneratedBy",
    "consumes": "prov:used",
    "calibrated_by": "prov:used",
    "supersedes": "prov:wasRevisionOf",
}
PROV_CLASSES: Mapping[str, str] = {
    "run": "prov:Activity",
    "instance": "prov:Activity",
    "backend": "prov:SoftwareAgent",
}

_HEX64 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_DEFINITION_ID = re.compile(r"(?P<kind>[a-z]+):[A-Za-z0-9_.-]+/[A-Za-z0-9_.:-]+")
_REVISION_ID = re.compile(r"(?P<kind>[a-z]+):sha256:(?P<digest>[0-9a-f]{64})")
_INSTANCE_ID = re.compile(rf"(?P<kind>[a-z]+):{_UUID}/{_UUID}")
_ID_PATTERNS: Mapping[IdentityClass, re.Pattern[str]] = {
    "definition": _DEFINITION_ID,
    "revision": _REVISION_ID,
    "instance": _INSTANCE_ID,
}
_NODE_FIELDS = frozenset(
    {
        "id",
        "kind",
        "label",
        "source_ref",
        "source_digest",
        "validity_domain",
        "as_of",
        "access",
        "status",
        "evidence_refs",
        "null_reasons",
    }
)
_NULLABLE_NODE_FIELDS = ("source_digest", "validity_domain", "as_of")
_EDGE_FIELDS = frozenset(
    {
        "source",
        "kind",
        "target",
        "relation_version",
        "derivation_method",
        "source_ref",
        "evidence_digest",
        "assumptions",
        "mapping",
        "as_of",
        "access",
    }
)
_SNAPSHOT_FIELDS = frozenset(
    {
        "schema",
        "extractor_version",
        "source_commit",
        "evidence_set_hash",
        "access_scope",
        "nodes",
        "edges",
        "tombstones",
        "exclusions",
    }
)


class AtlasContractError(ValueError):
    """Refusal of an Atlas document, with a stable code and the offending field path.

    Parameters
    ----------
    code
        Stable refusal category, shared with the browser contract.
    path
        JSON path of the offending value, for example ``$.edges[2].kind``.
    message
        Human-readable reason.

    """

    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(f"{code} at {path}: {message}")
        self.code = code
        self.path = path
        self.message = message


@dataclass(frozen=True)
class AtlasNode:
    """One validated Atlas node.

    Parameters
    ----------
    id
        Identity string whose shape follows the kind's identity class.
    kind
        One of :data:`NODE_KINDS`.
    label
        Display label.
    source_ref
        Where the node was taken from: a repository path, symbol or record name.
    source_digest
        SHA-256 of the source content, or ``None`` with a stated reason.
    validity_domain
        Scope of validity as string pairs (for claims: estimand, units, model), or ``None``.
    as_of
        UTC observation time in ISO 8601, or ``None`` with a stated reason.
    access
        ``"public"`` or ``"private"``.
    status
        ``"current"``, ``"stale"`` or ``"incomplete"``.
    evidence_refs
        Digests of supporting evidence records, sorted.
    null_reasons
        Reason for each optional field that is ``None``.

    """

    id: str
    kind: str
    label: str
    source_ref: str
    source_digest: str | None
    validity_domain: tuple[tuple[str, str], ...] | None
    as_of: str | None
    access: str
    status: str
    evidence_refs: tuple[str, ...]
    null_reasons: tuple[tuple[str, str], ...]

    @property
    def identity_class(self) -> IdentityClass:
        """Return whether the node is identified as a definition, revision or instance."""
        return IDENTITY_CLASSES[self.kind]

    @property
    def prov_class(self) -> str:
        """Return the PROV-O class the node maps to."""
        return PROV_CLASSES.get(self.kind, "prov:Entity")


@dataclass(frozen=True)
class AtlasEdge:
    """One validated, canonical Atlas edge.

    Parameters
    ----------
    source
        Node ID the relation starts at.
    kind
        Canonical edge kind; an admitted alias is already normalised.
    target
        Node ID the relation ends at.
    relation_version
        Always :data:`RELATION_VERSION`.
    derivation_method
        How the relation was obtained; one of :data:`DERIVATION_METHODS`.
    source_ref
        Where the relation was taken from.
    evidence_digest
        Digest of the evidence for the relation, or ``None`` where none is required.
    assumptions
        Stated assumptions, in the given order.
    mapping
        Units and semantics of a ``maps_to`` relation, as sorted string pairs.
    as_of
        UTC observation time, or ``None``.
    access
        ``"public"`` or ``"private"``.

    """

    source: str
    kind: str
    target: str
    relation_version: str
    derivation_method: str
    source_ref: str
    evidence_digest: str | None
    assumptions: tuple[str, ...]
    mapping: tuple[tuple[str, str], ...] | None
    as_of: str | None
    access: str

    @property
    def prov_relation(self) -> str | None:
        """Return the PROV-O relation of a provenance edge, otherwise ``None``."""
        return PROV_RELATIONS.get(self.kind)

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        """Return the hashing order ``(source, kind, target, evidence_digest)``."""
        return (self.source, self.kind, self.target, self.evidence_digest or "")


@dataclass(frozen=True)
class AtlasTombstone:
    """Exact identity of a deleted object, kept so the ID is never reused.

    Parameters
    ----------
    id
        The deleted node's ID, unchanged.
    kind
        The deleted node's kind, unchanged.
    last_source_digest
        Content digest of the last live revision, or ``None`` when it had none.
    removed_at
        UTC time of the deletion.
    reason
        Why the object was removed.

    """

    id: str
    kind: str
    last_source_digest: str | None
    removed_at: str
    reason: str


@dataclass(frozen=True)
class AtlasExclusion:
    """A documented exclusion from extraction.

    Parameters
    ----------
    source_ref
        What was left out.
    reason
        Why it was left out.

    """

    source_ref: str
    reason: str


@dataclass(frozen=True)
class AtlasSnapshot:
    """A validated Atlas snapshot in canonical order.

    Parameters
    ----------
    extractor_version
        Version of the extractor that built the snapshot.
    source_commit
        Full 40-hex commit the snapshot describes.
    evidence_set_hash
        Digest of the evidence set the snapshot was built against.
    access_scope
        ``"public"`` or ``"private"``.
    nodes
        Nodes sorted by ID.
    edges
        Canonical edges sorted by ``(source, kind, target, evidence_digest)``.
    tombstones
        Tombstones sorted by ID.
    exclusions
        Exclusions sorted by ``(source_ref, reason)``.

    """

    extractor_version: str
    source_commit: str
    evidence_set_hash: str
    access_scope: str
    nodes: tuple[AtlasNode, ...]
    edges: tuple[AtlasEdge, ...]
    tombstones: tuple[AtlasTombstone, ...]
    exclusions: tuple[AtlasExclusion, ...]

    def node(self, node_id: str) -> AtlasNode:
        """Return the live node with this ID.

        Parameters
        ----------
        node_id
            ID of a live node.

        Returns
        -------
        AtlasNode
            The node.

        Raises
        ------
        KeyError
            No live node has this ID.

        """
        for candidate in self.nodes:
            if candidate.id == node_id:
                return candidate
        raise KeyError(node_id)


def _refuse(code: str, path: str, message: str) -> AtlasContractError:
    return AtlasContractError(code, path, message)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _refuse("invalid_type", path, "expected an object with string keys")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _refuse("invalid_type", path, "expected a nonempty string")
    return value


def _choice(value: object, path: str, choices: Sequence[str], code: str) -> str:
    text = _string(value, path)
    if text not in choices:
        raise _refuse(code, path, f"{text!r} is not one of {', '.join(choices)}")
    return text


def _digest(value: object, path: str) -> str:
    text = _string(value, path)
    if not _HEX64.fullmatch(text):
        raise _refuse("invalid_digest", path, "expected 64 lowercase hexadecimal characters")
    return text


def _strings(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _refuse("invalid_type", path, "expected an array")
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))


def _pairs(value: object, path: str) -> tuple[tuple[str, str], ...]:
    pairs = _mapping(value, path)
    return tuple(sorted((key, _string(item, f"{path}.{key}")) for key, item in pairs.items()))


def _closed(payload: Mapping[str, object], fields: frozenset[str], path: str) -> None:
    unknown = sorted(set(payload) - fields)
    if unknown:
        raise _refuse("unknown_field", f"{path}.{unknown[0]}", "field is not part of the contract")
    missing = sorted(fields - set(payload))
    if missing:
        raise _refuse("missing_field", f"{path}.{missing[0]}", "required field is absent")


def _parse_node(payload: object, path: str) -> AtlasNode:
    node = _mapping(payload, path)
    _closed(node, _NODE_FIELDS, path)
    kind = _choice(node["kind"], f"{path}.kind", NODE_KINDS, "unknown_node_kind")
    node_id = _string(node["id"], f"{path}.id")
    identity = IDENTITY_CLASSES[kind]
    match = _ID_PATTERNS[identity].fullmatch(node_id)
    if match is None or match.group("kind") != kind:
        raise _refuse(
            "invalid_identity", f"{path}.id", f"not a {identity} identity of kind {kind}"
        )
    null_reasons = _pairs(node["null_reasons"], f"{path}.null_reasons")
    reasons = dict(null_reasons)
    for key in reasons:
        if key not in _NULLABLE_NODE_FIELDS or node[key] is not None:
            raise _refuse(
                "invalid_null_reason", f"{path}.null_reasons.{key}", "no null to explain"
            )
    for key in _NULLABLE_NODE_FIELDS:
        if node[key] is None and key not in reasons:
            raise _refuse("missing_null_reason", f"{path}.{key}", "a null needs a stated reason")
    source_digest = (
        None
        if node["source_digest"] is None
        else _digest(node["source_digest"], f"{path}.source_digest")
    )
    if identity == "revision" and (match.group("digest") != source_digest):
        raise _refuse("invalid_identity", f"{path}.id", "a revision ID binds its content digest")
    return AtlasNode(
        id=node_id,
        kind=kind,
        label=_string(node["label"], f"{path}.label"),
        source_ref=_string(node["source_ref"], f"{path}.source_ref"),
        source_digest=source_digest,
        validity_domain=(
            None
            if node["validity_domain"] is None
            else _pairs(node["validity_domain"], f"{path}.validity_domain")
        ),
        as_of=None if node["as_of"] is None else _string(node["as_of"], f"{path}.as_of"),
        access=_choice(node["access"], f"{path}.access", ACCESS_CLASSES, "unknown_access"),
        status=_choice(node["status"], f"{path}.status", NODE_STATUSES, "unknown_status"),
        evidence_refs=tuple(
            sorted(
                _digest(item, f"{path}.evidence_refs[{index}]")
                for index, item in enumerate(
                    _strings(node["evidence_refs"], f"{path}.evidence_refs")
                )
            )
        ),
        null_reasons=null_reasons,
    )


def _canonical_kind(kind: str, path: str) -> tuple[str, bool]:
    if kind in REFUSED_EDGE_KINDS:
        raise _refuse("refused_edge_kind", path, REFUSED_EDGE_KINDS[kind])
    if kind in EDGE_ALIASES_V1:
        return EDGE_ALIASES_V1[kind]
    if kind not in EDGE_KINDS:
        raise _refuse("unknown_edge_kind", path, f"{kind!r} is not an Atlas relation")
    return kind, False


def _parse_edge(payload: object, path: str) -> AtlasEdge:
    edge = _mapping(payload, path)
    _closed(edge, _EDGE_FIELDS, path)
    version = _string(edge["relation_version"], f"{path}.relation_version")
    if version != RELATION_VERSION:
        raise _refuse("unknown_relation_version", f"{path}.relation_version", version)
    kind, reverse = _canonical_kind(_string(edge["kind"], f"{path}.kind"), f"{path}.kind")
    source = _string(edge["source"], f"{path}.source")
    target = _string(edge["target"], f"{path}.target")
    if reverse:
        source, target = target, source
    evidence = edge["evidence_digest"]
    return AtlasEdge(
        source=source,
        kind=kind,
        target=target,
        relation_version=version,
        derivation_method=_choice(
            edge["derivation_method"],
            f"{path}.derivation_method",
            DERIVATION_METHODS,
            "unknown_derivation_method",
        ),
        source_ref=_string(edge["source_ref"], f"{path}.source_ref"),
        evidence_digest=None if evidence is None else _digest(evidence, f"{path}.evidence_digest"),
        assumptions=_strings(edge["assumptions"], f"{path}.assumptions"),
        mapping=None if edge["mapping"] is None else _pairs(edge["mapping"], f"{path}.mapping"),
        as_of=None if edge["as_of"] is None else _string(edge["as_of"], f"{path}.as_of"),
        access=_choice(edge["access"], f"{path}.access", ACCESS_CLASSES, "unknown_access"),
    )


def _check_edge(
    edge: AtlasEdge, nodes: Mapping[str, AtlasNode], tombstoned: set[str], path: str
) -> None:
    for end, node_id in (("source", edge.source), ("target", edge.target)):
        if node_id in tombstoned:
            raise _refuse("tombstoned_endpoint", f"{path}.{end}", f"{node_id} was deleted")
        if node_id not in nodes:
            raise _refuse("dangling_endpoint", f"{path}.{end}", f"{node_id} is not a live node")
    sources, targets = EDGE_ENDPOINTS[edge.kind]
    if nodes[edge.source].kind not in sources or nodes[edge.target].kind not in targets:
        raise _refuse(
            "illegal_endpoint",
            path,
            f"{edge.kind} cannot join {nodes[edge.source].kind} to {nodes[edge.target].kind}",
        )
    if edge.kind in EVIDENCE_REQUIRED and edge.evidence_digest is None:
        raise _refuse("missing_evidence", f"{path}.evidence_digest", f"{edge.kind} needs evidence")
    if edge.kind in ASSUMPTIONS_REQUIRED and not edge.assumptions:
        raise _refuse(
            "missing_assumptions", f"{path}.assumptions", f"{edge.kind} needs assumptions"
        )
    if edge.kind == "supports" and edge.derivation_method == "import_adjacency":
        raise _refuse(
            "adjacency_is_not_support",
            f"{path}.derivation_method",
            "import adjacency never establishes support",
        )
    if edge.kind == "maps_to" and not {"units", "semantics"} <= dict(edge.mapping or ()).keys():
        raise _refuse("missing_mapping", f"{path}.mapping", "maps_to needs units and semantics")
    if edge.kind == "contradicts_under_assumptions":
        domains = [
            dict(nodes[node_id].validity_domain or ()) for node_id in (edge.source, edge.target)
        ]
        for field in ("estimand", "units"):
            if field not in domains[0] or domains[0].get(field) != domains[1].get(field):
                raise _refuse(
                    "non_comparable",
                    path,
                    f"the claims do not share one {field}; show them as non-comparable",
                )
    private = "private" in (nodes[edge.source].access, nodes[edge.target].access)
    if private and edge.access != "private":
        raise _refuse(
            "access_leak", f"{path}.access", "an edge touching a private node is private"
        )


def _parse_tombstone(payload: object, path: str) -> AtlasTombstone:
    tombstone = _mapping(payload, path)
    _closed(
        tombstone, frozenset({"id", "kind", "last_source_digest", "removed_at", "reason"}), path
    )
    kind = _choice(tombstone["kind"], f"{path}.kind", NODE_KINDS, "unknown_node_kind")
    node_id = _string(tombstone["id"], f"{path}.id")
    match = _ID_PATTERNS[IDENTITY_CLASSES[kind]].fullmatch(node_id)
    if match is None or match.group("kind") != kind:
        raise _refuse("invalid_identity", f"{path}.id", "tombstone keeps the exact prior identity")
    last = tombstone["last_source_digest"]
    return AtlasTombstone(
        id=node_id,
        kind=kind,
        last_source_digest=None if last is None else _digest(last, f"{path}.last_source_digest"),
        removed_at=_string(tombstone["removed_at"], f"{path}.removed_at"),
        reason=_string(tombstone["reason"], f"{path}.reason"),
    )


def _parse_exclusion(payload: object, path: str) -> AtlasExclusion:
    exclusion = _mapping(payload, path)
    _closed(exclusion, frozenset({"source_ref", "reason"}), path)
    return AtlasExclusion(
        source_ref=_string(exclusion["source_ref"], f"{path}.source_ref"),
        reason=_string(exclusion["reason"], f"{path}.reason"),
    )


def _items(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload[key]
    if not isinstance(value, list):
        raise _refuse("invalid_type", f"$.{key}", "expected an array")
    return value


def parse_atlas_snapshot(payload: object) -> AtlasSnapshot:
    """Validate an Atlas snapshot document and return it in canonical form.

    Parameters
    ----------
    payload
        Untrusted input; it must be a ``quantum_atlas_snapshot.v1`` document: ``schema``,
        ``extractor_version``, ``source_commit``, ``evidence_set_hash``,
        ``access_scope``, ``nodes``, ``edges``, ``tombstones`` and ``exclusions``.

    Returns
    -------
    AtlasSnapshot
        Immutable snapshot with aliases normalised and every list in canonical order.

    Raises
    ------
    AtlasContractError
        The document breaks the schema, identity, vocabulary, legality or access rules.

    """
    document = _mapping(payload, "$")
    _closed(document, _SNAPSHOT_FIELDS, "$")
    schema = _string(document["schema"], "$.schema")
    if schema != SNAPSHOT_SCHEMA:
        raise _refuse("unknown_schema", "$.schema", f"{schema!r} is not {SNAPSHOT_SCHEMA}")
    source_commit = _string(document["source_commit"], "$.source_commit")
    if not _COMMIT.fullmatch(source_commit):
        raise _refuse("invalid_commit", "$.source_commit", "expected a full 40-hex commit")
    access_scope = _choice(
        document["access_scope"], "$.access_scope", ACCESS_CLASSES, "unknown_access"
    )
    nodes = [
        _parse_node(item, f"$.nodes[{index}]")
        for index, item in enumerate(_items(document, "nodes"))
    ]
    tombstones = [
        _parse_tombstone(item, f"$.tombstones[{index}]")
        for index, item in enumerate(_items(document, "tombstones"))
    ]
    by_id: dict[str, AtlasNode] = {}
    for index, node in enumerate(nodes):
        if node.id in by_id:
            raise _refuse("duplicate_id", f"$.nodes[{index}].id", f"{node.id} appears twice")
        if access_scope == "public" and node.access == "private":
            raise _refuse(
                "access_leak", f"$.nodes[{index}].access", "a public snapshot has no private node"
            )
        by_id[node.id] = node
    tombstoned: set[str] = set()
    for index, tombstone in enumerate(tombstones):
        if tombstone.id in by_id:
            raise _refuse(
                "reused_id", f"$.tombstones[{index}].id", f"{tombstone.id} is deleted and live"
            )
        if tombstone.id in tombstoned:
            raise _refuse(
                "duplicate_id", f"$.tombstones[{index}].id", f"{tombstone.id} appears twice"
            )
        tombstoned.add(tombstone.id)
    edges = [
        _parse_edge(item, f"$.edges[{index}]")
        for index, item in enumerate(_items(document, "edges"))
    ]
    seen: set[tuple[str, str, str, str]] = set()
    for index, edge in enumerate(edges):
        _check_edge(edge, by_id, tombstoned, f"$.edges[{index}]")
        if access_scope == "public" and edge.access == "private":
            raise _refuse(
                "access_leak", f"$.edges[{index}].access", "a public snapshot has no private edge"
            )
        if edge.sort_key in seen:
            raise _refuse("duplicate_edge", f"$.edges[{index}]", "the same relation appears twice")
        seen.add(edge.sort_key)
    return AtlasSnapshot(
        extractor_version=_string(document["extractor_version"], "$.extractor_version"),
        source_commit=source_commit,
        evidence_set_hash=_digest(document["evidence_set_hash"], "$.evidence_set_hash"),
        access_scope=access_scope,
        nodes=tuple(sorted(nodes, key=lambda node: node.id)),
        edges=tuple(sorted(edges, key=lambda edge: edge.sort_key)),
        tombstones=tuple(sorted(tombstones, key=lambda tombstone: tombstone.id)),
        exclusions=tuple(
            sorted(
                (
                    _parse_exclusion(item, f"$.exclusions[{index}]")
                    for index, item in enumerate(_items(document, "exclusions"))
                ),
                key=lambda exclusion: (exclusion.source_ref, exclusion.reason),
            )
        ),
    )


def _pairs_wire(pairs: tuple[tuple[str, str], ...] | None) -> dict[str, str] | None:
    return None if pairs is None else dict(pairs)


def snapshot_to_wire(snapshot: AtlasSnapshot) -> dict[str, object]:
    """Return the canonical wire document of a validated snapshot.

    Parameters
    ----------
    snapshot
        A snapshot returned by :func:`parse_atlas_snapshot` or :func:`project_public`.

    Returns
    -------
    dict[str, object]
        The document with canonical edge kinds and canonical order; parsing it
        again yields an equal snapshot.

    """
    return {
        "schema": SNAPSHOT_SCHEMA,
        "extractor_version": snapshot.extractor_version,
        "source_commit": snapshot.source_commit,
        "evidence_set_hash": snapshot.evidence_set_hash,
        "access_scope": snapshot.access_scope,
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind,
                "label": node.label,
                "source_ref": node.source_ref,
                "source_digest": node.source_digest,
                "validity_domain": _pairs_wire(node.validity_domain),
                "as_of": node.as_of,
                "access": node.access,
                "status": node.status,
                "evidence_refs": list(node.evidence_refs),
                "null_reasons": dict(node.null_reasons),
            }
            for node in snapshot.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "kind": edge.kind,
                "target": edge.target,
                "relation_version": edge.relation_version,
                "derivation_method": edge.derivation_method,
                "source_ref": edge.source_ref,
                "evidence_digest": edge.evidence_digest,
                "assumptions": list(edge.assumptions),
                "mapping": _pairs_wire(edge.mapping),
                "as_of": edge.as_of,
                "access": edge.access,
            }
            for edge in snapshot.edges
        ],
        "tombstones": [
            {
                "id": tombstone.id,
                "kind": tombstone.kind,
                "last_source_digest": tombstone.last_source_digest,
                "removed_at": tombstone.removed_at,
                "reason": tombstone.reason,
            }
            for tombstone in snapshot.tombstones
        ],
        "exclusions": [
            {"source_ref": exclusion.source_ref, "reason": exclusion.reason}
            for exclusion in snapshot.exclusions
        ],
    }


def atlas_snapshot_digest(snapshot: AtlasSnapshot) -> str:
    """Return the canonical SHA-256 of a validated snapshot.

    Parameters
    ----------
    snapshot
        A validated snapshot.

    Returns
    -------
    str
        Lowercase hex digest of the canonical wire document under the
        ``quantum_atlas_snapshot.v1`` domain; the browser contract computes the
        same value.

    """
    return canonical_digest(SNAPSHOT_SCHEMA, snapshot_to_wire(snapshot))


def project_public(snapshot: AtlasSnapshot) -> AtlasSnapshot:
    """Return the public projection of a snapshot.

    Parameters
    ----------
    snapshot
        A validated snapshot of either access scope.

    Returns
    -------
    AtlasSnapshot
        Public-scope snapshot without private nodes and without any edge that
        is private or touches a private node. Nothing records how much was
        removed, so counts do not disclose private material.

    """
    public_ids = {node.id for node in snapshot.nodes if node.access == "public"}
    return AtlasSnapshot(
        extractor_version=snapshot.extractor_version,
        source_commit=snapshot.source_commit,
        evidence_set_hash=snapshot.evidence_set_hash,
        access_scope="public",
        nodes=tuple(node for node in snapshot.nodes if node.id in public_ids),
        edges=tuple(
            edge
            for edge in snapshot.edges
            if edge.access == "public" and {edge.source, edge.target} <= public_ids
        ),
        tombstones=snapshot.tombstones,
        exclusions=snapshot.exclusions,
    )


__all__ = [
    "ACCESS_CLASSES",
    "ASSUMPTIONS_REQUIRED",
    "DERIVATION_METHODS",
    "EDGE_ALIASES_V1",
    "EDGE_ENDPOINTS",
    "EDGE_KINDS",
    "EVIDENCE_REQUIRED",
    "IDENTITY_CLASSES",
    "NODE_KINDS",
    "NODE_STATUSES",
    "PROV_CLASSES",
    "PROV_RELATIONS",
    "REFUSED_EDGE_KINDS",
    "RELATION_VERSION",
    "SNAPSHOT_SCHEMA",
    "AtlasContractError",
    "AtlasEdge",
    "AtlasExclusion",
    "AtlasNode",
    "AtlasSnapshot",
    "AtlasTombstone",
    "IdentityClass",
    "atlas_snapshot_digest",
    "parse_atlas_snapshot",
    "project_public",
    "snapshot_to_wire",
]
