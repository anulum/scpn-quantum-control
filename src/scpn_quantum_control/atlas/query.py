# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — bounded, deterministic Atlas queries
"""Answer bounded, deterministic queries over a validated Atlas snapshot.

Three query shapes share one entry point, :func:`run_atlas_query`:

- **lookup** (no ``start``): a case-folded substring match on ID and label,
  combined conjunctively with kind and status facets;
- **neighbourhood** (``start`` only): breadth-first traversal along the
  admitted edge kinds, outgoing, incoming or both, to at most ``depth`` hops;
- **path** (``start`` and ``target``): every shortest directed path within
  ``depth`` hops, in lexicographic order of the node sequence.

The access filter applies first. A public query runs on the public projection,
so hidden material changes no result, total or omitted count, and an unknown ID
is refused with the same words whether it is private or absent. Every result
names its snapshot digest, access scope, normalised query and explanation. A
cursor binds the snapshot digest, the normalised query, the access scope and
the last returned key; it is refused anywhere else. Pages never duplicate or
skip an admitted item. Limits are explicit: depth at most
:data:`MAX_DEPTH`, page size at most :data:`MAX_PAGE_SIZE`, at most
:data:`MAX_VISIBLE_EDGES` edges per result, and snapshots within the fixture
ceilings. A truncated result says so and how much it left out.

A shortest graph path is a navigation path, not a scientific proof.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from ..studio_workspace.canonical import canonical_digest
from .contracts import (
    EDGE_KINDS,
    NODE_KINDS,
    NODE_STATUSES,
    AtlasEdge,
    AtlasNode,
    AtlasSnapshot,
    atlas_snapshot_digest,
    project_public,
)

QUERY_SCHEMA = "quantum_atlas_query.v1"
DEFAULT_DEPTH = 1
MAX_DEPTH = 6
MAX_PAGE_SIZE = 500
MAX_VISIBLE_EDGES = 2000
NODE_CEILING = 10_000
EDGE_CEILING = 100_000

Direction = Literal["out", "in", "both"]


class AtlasQueryError(ValueError):
    """Refusal of a query, with a stable code and the offending field.

    Parameters
    ----------
    code
        Stable refusal category.
    field
        Query field the refusal concerns.
    message
        Human-readable reason.

    """

    def __init__(self, code: str, field: str, message: str) -> None:
        super().__init__(f"{code} at {field}: {message}")
        self.code = code
        self.field = field
        self.message = message


@dataclass(frozen=True)
class AtlasQuery:
    """One bounded query.

    Parameters
    ----------
    text
        Case-folded substring matched against node ID and label, or ``None``.
    kinds
        Node kinds to keep; empty keeps every kind.
    statuses
        Node statuses to keep; empty keeps every status.
    edge_kinds
        Canonical edge kinds to traverse; empty traverses every kind.
    start
        Start node of a neighbourhood or path query.
    target
        Target node of a path query.
    direction
        ``"out"``, ``"in"`` or ``"both"`` for a neighbourhood query.
    depth
        Hop bound, 1 to :data:`MAX_DEPTH`.
    page_size
        Items per page, 1 to :data:`MAX_PAGE_SIZE`.
    cursor
        Cursor of the previous page of the same query, or ``None``.

    """

    text: str | None = None
    kinds: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    edge_kinds: tuple[str, ...] = ()
    start: str | None = None
    target: str | None = None
    direction: Direction = "out"
    depth: int = DEFAULT_DEPTH
    page_size: int = MAX_PAGE_SIZE
    cursor: str | None = None


@dataclass(frozen=True)
class AtlasQueryResult:
    """One page of a query answer.

    Parameters
    ----------
    snapshot_digest
        Digest of the snapshot the answer was computed on.
    access_scope
        Access scope the query ran under.
    shape
        ``"lookup"``, ``"neighbourhood"`` or ``"path"``.
    query
        The normalised query, without its cursor.
    nodes
        Nodes of this page, in result order.
    edges
        Edges among this page's nodes, in canonical order.
    paths
        Node-ID sequences of this page, for a path query.
    total
        Admitted items of the whole answer.
    omitted
        Admitted items after this page.
    omitted_edges
        Admitted edges among this page's nodes that the edge limit left out.
    next_cursor
        Cursor of the next page, or ``None`` on the last page.
    explanation
        What the answer was computed from and what it does not establish.

    """

    snapshot_digest: str
    access_scope: str
    shape: str
    query: Mapping[str, object]
    nodes: tuple[AtlasNode, ...]
    edges: tuple[AtlasEdge, ...]
    paths: tuple[tuple[str, ...], ...]
    total: int
    omitted: int
    omitted_edges: int
    next_cursor: str | None
    explanation: tuple[str, ...]

    @property
    def truncated(self) -> bool:
        """Return whether admitted items or edges were left out of this page."""
        return self.omitted > 0 or self.omitted_edges > 0


def _refuse(code: str, field: str, message: str) -> AtlasQueryError:
    return AtlasQueryError(code, field, message)


@dataclass(frozen=True)
class _Normal:
    """A validated query with sorted facets and a case-folded text."""

    text: str | None
    kinds: tuple[str, ...]
    statuses: tuple[str, ...]
    edge_kinds: tuple[str, ...]

    def matches(self, node: AtlasNode) -> bool:
        """Return whether a node passes the text and facet filters."""
        if (
            self.text is not None
            and self.text not in node.id.casefold()
            and self.text not in node.label.casefold()
        ):
            return False
        if self.kinds and node.kind not in self.kinds:
            return False
        return not self.statuses or node.status in self.statuses


def _normalise(query: AtlasQuery) -> tuple[_Normal, dict[str, object]]:
    for field, values, choices, code in (
        ("kinds", query.kinds, NODE_KINDS, "unknown_node_kind"),
        ("statuses", query.statuses, NODE_STATUSES, "unknown_status"),
        ("edge_kinds", query.edge_kinds, EDGE_KINDS, "unknown_relation"),
    ):
        unknown = sorted(set(values) - set(choices))
        if unknown:
            raise _refuse(code, field, f"{unknown[0]!r} is not admitted")
    if query.direction not in ("out", "in", "both"):
        raise _refuse("invalid_direction", "direction", f"{query.direction!r}")
    if not 1 <= query.depth <= MAX_DEPTH:
        raise _refuse("oversized_traversal", "depth", f"depth must be 1 to {MAX_DEPTH}")
    if not 1 <= query.page_size <= MAX_PAGE_SIZE:
        raise _refuse("oversized_page", "page_size", f"page size must be 1 to {MAX_PAGE_SIZE}")
    if query.target is not None and query.start is None:
        raise _refuse("missing_start", "start", "a path query needs a start node")
    normal = _Normal(
        text=None if query.text is None else query.text.casefold(),
        kinds=tuple(sorted(set(query.kinds))),
        statuses=tuple(sorted(set(query.statuses))),
        edge_kinds=tuple(sorted(set(query.edge_kinds))),
    )
    return normal, {
        "schema": QUERY_SCHEMA,
        "text": normal.text,
        "kinds": list(normal.kinds),
        "statuses": list(normal.statuses),
        "edge_kinds": list(normal.edge_kinds),
        "start": query.start,
        "target": query.target,
        "direction": query.direction,
        "depth": query.depth,
        "page_size": query.page_size,
    }


def _cursor(binding: Mapping[str, object], last: object) -> str:
    raw = json.dumps({**binding, "last": last}, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def _resume(cursor: str | None, binding: Mapping[str, object]) -> object:
    if cursor is None:
        return None
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8"))
    except (UnicodeError, ValueError, binascii.Error) as error:
        raise _refuse("invalid_cursor", "cursor", "not an Atlas cursor") from error
    if not isinstance(decoded, dict) or "last" not in decoded:
        raise _refuse("invalid_cursor", "cursor", "not an Atlas cursor")
    last = decoded.pop("last")
    if decoded != dict(binding):
        raise _refuse(
            "cursor_mismatch",
            "cursor",
            "the cursor belongs to another snapshot, query or access scope",
        )
    return last


def _adjacency(
    view: AtlasSnapshot, edge_kinds: tuple[str, ...], direction: str
) -> dict[str, list[tuple[str, AtlasEdge]]]:
    adjacency: dict[str, list[tuple[str, AtlasEdge]]] = {node.id: [] for node in view.nodes}
    for edge in view.edges:
        if edge_kinds and edge.kind not in edge_kinds:
            continue
        if direction in ("out", "both"):
            adjacency[edge.source].append((edge.target, edge))
        if direction in ("in", "both"):
            adjacency[edge.target].append((edge.source, edge))
    for neighbours in adjacency.values():
        neighbours.sort(key=lambda item: (item[0], item[1].sort_key))
    return adjacency


def _distances(
    adjacency: Mapping[str, list[tuple[str, AtlasEdge]]], start: str, depth: int
) -> dict[str, int]:
    distance = {start: 0}
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        if distance[current] == depth:
            continue
        for neighbour, _edge in adjacency[current]:
            if neighbour not in distance:
                distance[neighbour] = distance[current] + 1
                queue.append(neighbour)
    return distance


def _shortest_paths(
    adjacency: Mapping[str, list[tuple[str, AtlasEdge]]], start: str, target: str, depth: int
) -> list[tuple[str, ...]]:
    distance = _distances(adjacency, start, depth)
    if target not in distance:
        return []
    paths: list[tuple[str, ...]] = []

    def extend(path: tuple[str, ...]) -> None:
        current = path[-1]
        if current == target:
            paths.append(path)
            return
        for neighbour, _edge in adjacency[current]:
            if (
                distance.get(neighbour) == distance[current] + 1
                and distance[neighbour] <= distance[target]
            ):
                extend((*path, neighbour))

    extend((start,))
    return sorted(set(paths))


def _known(view: AtlasSnapshot, node_id: str | None, field: str) -> None:
    if node_id is not None and node_id not in {node.id for node in view.nodes}:
        raise _refuse("unknown_node", field, "the node is not in the admitted snapshot")


def run_atlas_query(
    snapshot: AtlasSnapshot | None, query: AtlasQuery, *, access: str = "public"
) -> AtlasQueryResult:
    """Answer one page of a bounded query.

    Parameters
    ----------
    snapshot
        A validated snapshot; ``None`` is refused.
    query
        The query.
    access
        ``"public"`` runs on the public projection; ``"private"`` needs a
        private-scope snapshot and sees everything in it.

    Returns
    -------
    AtlasQueryResult
        One page with its totals, omissions, explanation and next cursor.

    Raises
    ------
    AtlasQueryError
        The snapshot is absent or too large, the access scope is not admitted,
        the query names an unknown kind, relation or node, a bound is exceeded,
        or the cursor belongs elsewhere.

    """
    if snapshot is None:
        raise _refuse("absent_snapshot", "snapshot", "no snapshot to query")
    if len(snapshot.nodes) > NODE_CEILING or len(snapshot.edges) > EDGE_CEILING:
        raise _refuse("snapshot_too_large", "snapshot", "above the admitted node or edge ceiling")
    if access == "public":
        view = project_public(snapshot)
    elif access == "private" and snapshot.access_scope == "private":
        view = snapshot
    else:
        raise _refuse("access_refused", "access", f"{access!r} is not admitted for this snapshot")
    normal, wire = _normalise(query)
    _known(view, query.start, "start")
    _known(view, query.target, "target")
    digest = atlas_snapshot_digest(view)
    binding = {
        "snapshot": digest,
        "access": access,
        "query": canonical_digest(QUERY_SCHEMA, wire),
    }
    last = _resume(query.cursor, binding)
    by_id = {node.id: node for node in view.nodes}
    edge_kinds = normal.edge_kinds
    explanation = [
        f"snapshot {digest} of commit {view.source_commit}, access {access}",
        f"{len(view.exclusions)} documented exclusions are not part of the graph",
    ]
    paths: list[tuple[str, ...]] = []
    if query.start is None:
        shape = "lookup"
        ordered: list[object] = [node.id for node in view.nodes if normal.matches(node)]
        explanation.append("nodes matching every given filter, ordered by ID")
    elif query.target is None:
        shape = "neighbourhood"
        adjacency = _adjacency(view, edge_kinds, query.direction)
        distance = _distances(adjacency, query.start, query.depth)
        ordered = [
            node_id
            for node_id in sorted(distance, key=lambda key: (distance[key], key))
            if normal.matches(by_id[node_id])
        ]
        explanation.append(
            f"{query.direction} neighbourhood within {query.depth} hops, ordered by distance and ID"
        )
    else:
        shape = "path"
        adjacency = _adjacency(view, edge_kinds, "out")
        paths = _shortest_paths(adjacency, query.start, query.target, query.depth)
        ordered = [list(path) for path in paths]
        explanation.append(
            f"every shortest directed path within {query.depth} hops; a navigation path, not a proof"
        )
    explanation.append(
        "edge kinds " + (", ".join(edge_kinds) if edge_kinds else "all admitted kinds")
    )
    position = (
        0
        if last is None
        else next((index + 1 for index, key in enumerate(ordered) if key == last), None)
    )
    if position is None:
        raise _refuse("cursor_mismatch", "cursor", "the cursor's last item is not in this answer")
    page = ordered[position : position + query.page_size]
    remaining = len(ordered) - position - len(page)
    page_paths = tuple(tuple(item) for item in page if isinstance(item, list))
    page_ids = (
        sorted({node_id for path in page_paths for node_id in path})
        if shape == "path"
        else [str(item) for item in page]
    )
    members = set(page_ids)
    inner = [edge for edge in view.edges if {edge.source, edge.target} <= members]
    if edge_kinds:
        inner = [edge for edge in inner if edge.kind in edge_kinds]
    nodes = tuple(by_id[node_id] for node_id in page_ids)
    return AtlasQueryResult(
        snapshot_digest=digest,
        access_scope=access,
        shape=shape,
        query=wire,
        nodes=nodes,
        edges=tuple(inner[:MAX_VISIBLE_EDGES]),
        paths=page_paths,
        total=len(ordered),
        omitted=remaining,
        omitted_edges=max(0, len(inner) - MAX_VISIBLE_EDGES),
        next_cursor=_cursor(binding, page[-1]) if remaining > 0 and page else None,
        explanation=tuple(explanation),
    )


@dataclass(frozen=True)
class AtlasPrebuiltQuery:
    """One of the Atlas design's questions, with its current support state.

    Parameters
    ----------
    question
        The question as the design states it.
    unsupported
        Why the current snapshot cannot answer it, or ``None`` when it can.

    """

    question: str
    unsupported: str | None


_NEEDS_DOMAIN = (
    "needs problem, precision and method-support inputs that no admitted extractor provides"
)
PREBUILT_QUERIES: Mapping[str, AtlasPrebuiltQuery] = {
    "usable_local_methods": AtlasPrebuiltQuery(
        "Which currently usable local methods accept this problem and precision?", _NEEDS_DOMAIN
    ),
    "transform_rejection_owner": AtlasPrebuiltQuery(
        "Why is this JAX/Torch transform composition rejected, and which rule owns it?",
        "needs transform-composition rules as Atlas contract nodes",
    ),
    "dependents_of": AtlasPrebuiltQuery(
        "What results depend on this changed coupling, compiler pass or calibration?", None
    ),
    "parameter_influence": AtlasPrebuiltQuery(
        "Which parameter influences this observable, and how is the dependency known?",
        "needs parameter and observable nodes with their dependency kind",
    ),
    "hardware_support": AtlasPrebuiltQuery(
        "Which hardware observations support this claim, with which device and shots?",
        "needs reviewed support edges from result packs to claims",
    ),
    "matched_benchmarks": AtlasPrebuiltQuery(
        "Which benchmarks compare these methods on the same problem and total budget?",
        "needs benchmark artefacts with their problem and budget",
    ),
    "capabilities_without_tests": AtlasPrebuiltQuery(
        "Which advertised capabilities have no real tests?", None
    ),
    "capabilities_without_documentation": AtlasPrebuiltQuery(
        "Which advertised capabilities have no native documentation?", None
    ),
    "contradictions_resolved_by_scope": AtlasPrebuiltQuery(
        "Which apparent contradictions disappear when model, units or estimator differ?",
        "needs the comparability owner over claim estimands",
    ),
    "smallest_offline_example": AtlasPrebuiltQuery(
        "What is the smallest no-QPU example that demonstrates this capability?",
        "needs example nodes with their resource cost",
    ),
    "stale_saved_views": AtlasPrebuiltQuery(
        "What source or evidence changes made yesterday's saved view stale?",
        "needs the snapshot-pair difference owner",
    ),
}


def _capabilities_lacking(
    snapshot: AtlasSnapshot, access: str, has: Callable[[AtlasEdge, str], bool]
) -> AtlasQueryResult:
    every = run_atlas_query(snapshot, AtlasQuery(kinds=("capability",)), access=access)
    view = project_public(snapshot) if access == "public" else snapshot
    covered = {node.id for node in every.nodes if any(has(edge, node.id) for edge in view.edges)}
    lacking = tuple(node for node in every.nodes if node.id not in covered)
    return AtlasQueryResult(
        snapshot_digest=every.snapshot_digest,
        access_scope=access,
        shape="lookup",
        query=every.query,
        nodes=lacking,
        edges=(),
        paths=(),
        total=len(lacking),
        omitted=0,
        omitted_edges=0,
        next_cursor=None,
        explanation=(*every.explanation, "capabilities without the named relation"),
    )


def run_prebuilt_query(
    snapshot: AtlasSnapshot | None,
    name: str,
    *,
    node: str | None = None,
    depth: int = DEFAULT_DEPTH,
    access: str = "public",
) -> AtlasQueryResult:
    """Answer one of the Atlas design's questions, or refuse it with its reason.

    Parameters
    ----------
    snapshot
        A validated snapshot.
    name
        A key of :data:`PREBUILT_QUERIES`.
    node
        The node a node-bound question is about (``dependents_of``).
    depth
        Hop bound of a node-bound question.
    access
        Access scope, as for :func:`run_atlas_query`.

    Returns
    -------
    AtlasQueryResult
        The answer.

    Raises
    ------
    AtlasQueryError
        The question is unknown or unsupported, its node is missing, or the
        underlying query is refused.

    """
    prebuilt = PREBUILT_QUERIES.get(name)
    if prebuilt is None:
        raise _refuse("unknown_prebuilt_query", "name", f"{name!r} is not a prebuilt question")
    if prebuilt.unsupported is not None:
        raise _refuse("unsupported_prebuilt_query", "name", prebuilt.unsupported)
    if snapshot is None:
        raise _refuse("absent_snapshot", "snapshot", "no snapshot to query")
    if name == "dependents_of":
        if node is None:
            raise _refuse("missing_start", "node", "dependents_of needs the node that changed")
        return run_atlas_query(
            snapshot,
            AtlasQuery(
                start=node,
                direction="in",
                depth=depth,
                edge_kinds=("consumes", "derives_from", "requires"),
            ),
            access=access,
        )
    if name == "capabilities_without_tests":
        return _capabilities_lacking(
            snapshot,
            access,
            lambda edge, node_id: edge.kind == "contains" and edge.source == node_id,
        )
    return _capabilities_lacking(
        snapshot, access, lambda edge, node_id: edge.kind == "documents" and edge.target == node_id
    )


__all__ = [
    "DEFAULT_DEPTH",
    "EDGE_CEILING",
    "MAX_DEPTH",
    "MAX_PAGE_SIZE",
    "MAX_VISIBLE_EDGES",
    "NODE_CEILING",
    "PREBUILT_QUERIES",
    "QUERY_SCHEMA",
    "AtlasPrebuiltQuery",
    "AtlasQuery",
    "AtlasQueryError",
    "AtlasQueryResult",
    "Direction",
    "run_atlas_query",
    "run_prebuilt_query",
]
