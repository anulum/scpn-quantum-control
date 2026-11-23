# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — temporal diffs of Atlas snapshots and their potential impact
"""Compare two Atlas snapshots and trace the potential impact of what changed.

Objects are matched by their stable identifier. A node's typed content hash
covers every field except ``evidence_refs``, which move with any byte of the
inputs that list it, so a node counts as changed only when what it states
changed.

Impact is *potential*. It spreads from each changed, removed or tombstoned node
along typed dependency edges only, in the direction :data:`IMPACT_FLOW` declares
for each relation; relations outside that table carry none. The result says it
is complete only when nothing was excluded from either extraction, every node on
an impact path is current, and every traversed edge was observed rather than
declared. Otherwise it lists why the analysis is incomplete; it never claims to
be an exhaustive proof of impact.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from ..studio_workspace.canonical import canonical_digest
from .contracts import AtlasEdge, AtlasSnapshot, snapshot_to_wire

NODE_CONTENT_SCHEMA = "quantum_atlas_node_content.v1"
"""Domain of a node's typed content hash."""

ChangeCategory = Literal["code", "data", "contract", "documentation", "calibration"]
NodeChange = Literal["added", "removed", "tombstoned", "changed"]

CATEGORY_OF_KIND: Mapping[str, ChangeCategory] = MappingProxyType(
    {
        "definition": "code",
        "test": "code",
        "program": "code",
        "artefact": "data",
        "revision": "data",
        "run": "data",
        "instance": "data",
        "capability": "contract",
        "claim": "contract",
        "parameter": "contract",
        "documentation": "documentation",
        "backend": "calibration",
    }
)
"""Change category of each node kind; a ``calibrated_by`` target is a calibration change."""

IMPACT_FLOW: Mapping[str, Literal["to_source", "to_target"]] = MappingProxyType(
    {
        "derives_from": "to_source",
        "consumes": "to_source",
        "requires": "to_source",
        "constrained_by": "to_source",
        "documents": "to_source",
        "contains": "to_source",
        "benchmarked_against": "to_source",
        "calibrated_by": "to_source",
        "implements": "to_target",
        "lowers_to": "to_target",
        "verified_by": "to_target",
        "supports": "to_target",
        "produces": "to_target",
    }
)
"""Direction potential impact travels along each relation, from dependency to dependent.

``s derives_from t`` makes ``s`` depend on ``t``, so a change of ``t`` reaches
``s`` (``to_source``). ``s implements t`` makes the capability ``t`` depend on
its implementation ``s`` (``to_target``). ``supersedes``, ``maps_to`` and
``contradicts_under_assumptions`` carry no impact.
"""

OBSERVED_DERIVATIONS: frozenset[str] = frozenset({"extracted", "measured"})
"""Derivations that observe a relation; ``declared``, ``curated`` and ``import_adjacency`` do not."""


class AtlasDiffError(ValueError):
    """Two snapshots cannot be compared.

    Parameters
    ----------
    message
        Why the comparison was refused.

    """


@dataclass(frozen=True, slots=True)
class NodeDifference:
    """One node that differs between the snapshots.

    Parameters
    ----------
    id
        Stable node identifier.
    kind
        Node kind, from the snapshot that holds the node (``before`` for a removal).
    change
        ``added``, ``removed``, ``tombstoned`` or ``changed``.
    category
        Change category of the node.
    fields
        Fields whose values differ, for a change; empty otherwise.
    before_hash
        Typed content hash before, or ``None`` for an addition.
    after_hash
        Typed content hash after, or ``None`` for a removal.
    reason
        The tombstone's reason and removal time, for a tombstoned node.

    """

    id: str
    kind: str
    change: NodeChange
    category: ChangeCategory
    fields: tuple[str, ...]
    before_hash: str | None
    after_hash: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class EdgeDifference:
    """One relation that differs between the snapshots.

    Parameters
    ----------
    source
        Source node identifier.
    kind
        Canonical relation kind.
    target
        Target node identifier.
    change
        ``added``, ``removed`` or ``changed``.

    """

    source: str
    kind: str
    target: str
    change: Literal["added", "removed", "changed"]


@dataclass(frozen=True, slots=True)
class ImpactStep:
    """One relation an impact path follows.

    Parameters
    ----------
    source
        Source node of the relation.
    kind
        Relation kind.
    target
        Target node of the relation.

    """

    source: str
    kind: str
    target: str


@dataclass(frozen=True, slots=True)
class ImpactedNode:
    """A node a change can reach, with the shortest path that reaches it.

    Parameters
    ----------
    id
        The impacted node.
    origin
        The changed node the path starts from.
    path
        The relations followed, in order.

    """

    id: str
    origin: str
    path: tuple[ImpactStep, ...]


@dataclass(frozen=True, slots=True)
class AtlasSnapshotDiff:
    """Differences between two snapshots and the potential impact of the changes.

    Parameters
    ----------
    nodes
        Differing nodes, ordered by identifier.
    edges
        Differing relations, ordered by source, kind and target.
    unchanged
        Count of nodes present and equal in both.
    impact
        Nodes a change can reach, ordered by identifier.
    complete
        Whether the impact analysis rests only on observed relations of complete
        extractions; it is never an exhaustive proof.
    incomplete_reasons
        Why the analysis is incomplete, sorted.

    """

    nodes: tuple[NodeDifference, ...]
    edges: tuple[EdgeDifference, ...]
    unchanged: int
    impact: tuple[ImpactedNode, ...]
    complete: bool
    incomplete_reasons: tuple[str, ...]


def node_content_hash(node: Mapping[str, object]) -> str:
    """Hash what a wire node states, leaving out its evidence references.

    Parameters
    ----------
    node
        A node of :func:`~scpn_quantum_control.atlas.contracts.snapshot_to_wire`.

    Returns
    -------
    str
        Canonical digest under :data:`NODE_CONTENT_SCHEMA`.

    """
    return canonical_digest(
        NODE_CONTENT_SCHEMA, {key: value for key, value in node.items() if key != "evidence_refs"}
    )


def _wire_nodes(snapshot: AtlasSnapshot) -> dict[str, Mapping[str, object]]:
    nodes = cast(list[Mapping[str, object]], snapshot_to_wire(snapshot)["nodes"])
    return {str(node["id"]): node for node in nodes}


def _wire_edges(snapshot: AtlasSnapshot) -> dict[tuple[str, str, str], Mapping[str, object]]:
    edges = cast(list[Mapping[str, object]], snapshot_to_wire(snapshot)["edges"])
    return {(str(e["source"]), str(e["kind"]), str(e["target"])): e for e in edges}


def _calibrated(snapshots: Iterable[AtlasSnapshot]) -> set[str]:
    return {
        edge.target
        for snapshot in snapshots
        for edge in snapshot.edges
        if edge.kind == "calibrated_by"
    }


def _category(kind: str, node_id: str, calibrated: set[str]) -> ChangeCategory:
    return "calibration" if node_id in calibrated else CATEGORY_OF_KIND[kind]


def _node_differences(
    before: AtlasSnapshot, after: AtlasSnapshot
) -> tuple[list[NodeDifference], int]:
    old, new = _wire_nodes(before), _wire_nodes(after)
    calibrated = _calibrated((before, after))
    tombstones = {tombstone.id: tombstone for tombstone in after.tombstones}
    differences: list[NodeDifference] = []
    unchanged = 0
    for node_id in sorted(old.keys() | new.keys()):
        was, now = old.get(node_id), new.get(node_id)
        if was is None:
            now = cast(Mapping[str, object], now)
            kind = str(now["kind"])
            differences.append(
                NodeDifference(
                    node_id,
                    kind,
                    "added",
                    _category(kind, node_id, calibrated),
                    (),
                    None,
                    node_content_hash(now),
                    None,
                )
            )
            continue
        kind = str(was["kind"])
        category = _category(kind, node_id, calibrated)
        if now is None:
            tombstone = tombstones.get(node_id)
            differences.append(
                NodeDifference(
                    node_id,
                    kind,
                    "removed" if tombstone is None else "tombstoned",
                    category,
                    (),
                    node_content_hash(was),
                    None,
                    None
                    if tombstone is None
                    else f"{tombstone.reason} (removed at {tombstone.removed_at})",
                )
            )
            continue
        before_hash, after_hash = node_content_hash(was), node_content_hash(now)
        if before_hash == after_hash:
            unchanged += 1
            continue
        fields = tuple(
            sorted(
                key
                for key in was.keys() | now.keys()
                if key != "evidence_refs" and was.get(key) != now.get(key)
            )
        )
        differences.append(
            NodeDifference(
                node_id,
                str(now["kind"]),
                "changed",
                category,
                fields,
                before_hash,
                after_hash,
                None,
            )
        )
    return differences, unchanged


def _edge_differences(before: AtlasSnapshot, after: AtlasSnapshot) -> list[EdgeDifference]:
    old, new = _wire_edges(before), _wire_edges(after)
    differences: list[EdgeDifference] = []
    for key in sorted(old.keys() | new.keys()):
        if key not in new:
            differences.append(EdgeDifference(*key, change="removed"))
        elif key not in old:
            differences.append(EdgeDifference(*key, change="added"))
        elif old[key] != new[key]:
            differences.append(EdgeDifference(*key, change="changed"))
    return differences


def _flows(edges: Iterable[AtlasEdge]) -> dict[str, list[tuple[str, AtlasEdge]]]:
    """Map each node to the nodes its change can reach in one relation."""
    reach: dict[str, list[tuple[str, AtlasEdge]]] = {}
    for edge in edges:
        flow = IMPACT_FLOW.get(edge.kind)
        if flow is None:
            continue
        start, end = (
            (edge.target, edge.source) if flow == "to_source" else (edge.source, edge.target)
        )
        reach.setdefault(start, []).append((end, edge))
    for steps in reach.values():
        steps.sort(key=lambda step: (step[0], step[1].kind))
    return reach


def _impact(
    seeds: list[str], reach: Mapping[str, list[tuple[str, AtlasEdge]]]
) -> tuple[list[ImpactedNode], set[str], list[AtlasEdge]]:
    """Breadth-first search from the sorted seeds; the first path found is kept."""
    found: dict[str, ImpactedNode] = {}
    traversed: list[AtlasEdge] = []
    seen = set(seeds)
    queue: deque[tuple[str, str, tuple[ImpactStep, ...]]] = deque(
        (seed, seed, ()) for seed in seeds
    )
    while queue:
        node, origin, path = queue.popleft()
        for end, edge in reach.get(node, ()):
            traversed.append(edge)
            if end in seen:
                continue
            seen.add(end)
            step = path + (ImpactStep(edge.source, edge.kind, edge.target),)
            found[end] = ImpactedNode(end, origin, step)
            queue.append((end, origin, step))
    return [found[key] for key in sorted(found)], seen, traversed


def _incomplete(
    before: AtlasSnapshot,
    after: AtlasSnapshot,
    reached: set[str],
    traversed: list[AtlasEdge],
) -> list[str]:
    reasons: set[str] = set()
    for name, snapshot in (("before", before), ("after", after)):
        if snapshot.exclusions:
            reasons.add(
                f"the {name} extraction excluded {len(snapshot.exclusions)} inputs or objects"
            )
    statuses: dict[str, str] = {node.id: node.status for node in before.nodes}
    statuses.update({node.id: node.status for node in after.nodes})
    for node_id in sorted(reached):
        status = statuses.get(node_id)
        if status is not None and status != "current":
            reasons.add(f"{node_id} is {status}")
    for edge in traversed:
        if edge.derivation_method not in OBSERVED_DERIVATIONS:
            reasons.add(f"{edge.derivation_method} relations were followed, not observed ones")
    return sorted(reasons)


def diff_atlas_snapshots(before: AtlasSnapshot, after: AtlasSnapshot) -> AtlasSnapshotDiff:
    """Compare two snapshots and trace the potential impact of the changes.

    Parameters
    ----------
    before
        The earlier snapshot.
    after
        The later snapshot, of the same access scope.

    Returns
    -------
    AtlasSnapshotDiff
        Classified node and relation differences, the impacted nodes with their
        paths, and whether the impact analysis is complete.

    Raises
    ------
    AtlasDiffError
        The snapshots have different access scopes.

    """
    if before.access_scope != after.access_scope:
        raise AtlasDiffError(
            f"cannot compare a {before.access_scope} snapshot with a {after.access_scope} one"
        )
    nodes, unchanged = _node_differences(before, after)
    edges = _edge_differences(before, after)
    seeds = sorted(item.id for item in nodes if item.change != "added")
    union = {(e.source, e.kind, e.target): e for e in (*before.edges, *after.edges)}
    impacted, reached, traversed = _impact(seeds, _flows(union[key] for key in sorted(union)))
    reasons = _incomplete(before, after, reached, traversed)
    return AtlasSnapshotDiff(
        nodes=tuple(nodes),
        edges=tuple(edges),
        unchanged=unchanged,
        impact=tuple(impacted),
        complete=not reasons,
        incomplete_reasons=tuple(reasons),
    )


def diff_to_wire(diff: AtlasSnapshotDiff) -> dict[str, object]:
    """Return the JSON form of a diff, in its canonical order.

    Parameters
    ----------
    diff
        A diff from :func:`diff_atlas_snapshots`.

    Returns
    -------
    dict[str, object]
        Nodes, relations, the unchanged count, the impact with its paths, and the
        completeness verdict with its reasons.

    """
    return {
        "nodes": [
            {
                "id": item.id,
                "kind": item.kind,
                "change": item.change,
                "category": item.category,
                "fields": list(item.fields),
                "before_hash": item.before_hash,
                "after_hash": item.after_hash,
                "reason": item.reason,
            }
            for item in diff.nodes
        ],
        "edges": [
            {
                "source": item.source,
                "kind": item.kind,
                "target": item.target,
                "change": item.change,
            }
            for item in diff.edges
        ],
        "unchanged": diff.unchanged,
        "impact": [
            {
                "id": item.id,
                "origin": item.origin,
                "path": [
                    {"source": step.source, "kind": step.kind, "target": step.target}
                    for step in item.path
                ],
            }
            for item in diff.impact
        ],
        "complete": diff.complete,
        "incomplete_reasons": list(diff.incomplete_reasons),
    }


ViewState = Literal["present", "tombstoned", "absent"]


@dataclass(frozen=True, slots=True)
class ViewReference:
    """Where a node a saved view names stands in a later snapshot.

    Parameters
    ----------
    id
        The node the view names.
    state
        ``present``, ``tombstoned`` or ``absent``.
    explanation
        The tombstone's reason and time, or why the node is absent; empty when present.

    """

    id: str
    state: ViewState
    explanation: str


def resolve_saved_view(snapshot: AtlasSnapshot, ids: Iterable[str]) -> tuple[ViewReference, ...]:
    """Resolve the nodes a saved view names against a later snapshot.

    Parameters
    ----------
    snapshot
        The snapshot the view is opened on.
    ids
        Node identifiers the view names, such as its focus and selection.

    Returns
    -------
    tuple[ViewReference, ...]
        One reference per distinct identifier, in sorted order.

    """
    live = {node.id for node in snapshot.nodes}
    tombstones = {tombstone.id: tombstone for tombstone in snapshot.tombstones}
    references: list[ViewReference] = []
    for node_id in sorted(set(ids)):
        if node_id in live:
            references.append(ViewReference(node_id, "present", ""))
        elif node_id in tombstones:
            tombstone = tombstones[node_id]
            references.append(
                ViewReference(
                    node_id,
                    "tombstoned",
                    f"{tombstone.reason} (removed at {tombstone.removed_at})",
                )
            )
        else:
            references.append(
                ViewReference(
                    node_id, "absent", "the snapshot neither holds it nor records its removal"
                )
            )
    return tuple(references)


__all__ = [
    "CATEGORY_OF_KIND",
    "IMPACT_FLOW",
    "NODE_CONTENT_SCHEMA",
    "OBSERVED_DERIVATIONS",
    "AtlasDiffError",
    "AtlasSnapshotDiff",
    "ChangeCategory",
    "EdgeDifference",
    "ImpactStep",
    "ImpactedNode",
    "NodeChange",
    "NodeDifference",
    "ViewReference",
    "ViewState",
    "diff_atlas_snapshots",
    "diff_to_wire",
    "node_content_hash",
    "resolve_saved_view",
]
