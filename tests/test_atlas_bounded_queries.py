# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Atlas bounded query tests
"""Tests for bounded, deterministic and access-filtered Atlas queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from scpn_quantum_control.atlas import AtlasSnapshot, parse_atlas_snapshot
from scpn_quantum_control.atlas import query as query_module
from scpn_quantum_control.atlas.extractors import directory_reader, extract_atlas_snapshot
from scpn_quantum_control.atlas.query import (
    PREBUILT_QUERIES,
    AtlasQuery,
    AtlasQueryError,
    run_atlas_query,
    run_prebuilt_query,
)

_DIGEST = "a" * 64


def _node(
    name: str, *, kind: str = "definition", access: str = "public", status: str = "current"
) -> dict[str, Any]:
    """Return one fixture node.

    Parameters
    ----------
    name : str
        Short name used in the ID and the label.
    kind : str
        Node kind; definition-class identity.
    access : str
        Access class.
    status : str
        Freshness status.

    Returns
    -------
    dict[str, Any]
        The wire node.

    """
    return {
        "id": f"{kind}:fixture/{name}",
        "kind": kind,
        "label": f"Node {name.upper()}",
        "source_ref": f"fixture/{name}",
        "source_digest": None,
        "validity_domain": None,
        "as_of": None,
        "access": access,
        "status": status,
        "evidence_refs": [],
        "null_reasons": {
            "source_digest": "fixture",
            "validity_domain": "fixture",
            "as_of": "fixture",
        },
    }


def _edge(
    source: str,
    target: str,
    *,
    kind: str = "requires",
    access: str = "public",
    src_kind: str = "definition",
    dst_kind: str = "definition",
) -> dict[str, Any]:
    """Return one fixture edge between two fixture nodes.

    Parameters
    ----------
    source : str
        Short name of the source node.
    target : str
        Short name of the target node.
    kind : str
        Edge kind.
    access : str
        Access class.
    src_kind : str
        Source node kind.
    dst_kind : str
        Target node kind.

    Returns
    -------
    dict[str, Any]
        The wire edge.

    """
    return {
        "source": f"{src_kind}:fixture/{source}",
        "kind": kind,
        "target": f"{dst_kind}:fixture/{target}",
        "relation_version": "atlas_relation.v1",
        "derivation_method": "declared",
        "source_ref": "fixture/edges",
        "evidence_digest": None,
        "assumptions": [],
        "mapping": None,
        "as_of": None,
        "access": access,
    }


def _snapshot(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], *, commit: str = "0" * 40
) -> AtlasSnapshot:
    """Return a validated private-scope fixture snapshot.

    Parameters
    ----------
    nodes : list[dict[str, Any]]
        Wire nodes.
    edges : list[dict[str, Any]]
        Wire edges.
    commit : str
        Source commit.

    Returns
    -------
    AtlasSnapshot
        The parsed snapshot.

    """
    return parse_atlas_snapshot(
        {
            "schema": "quantum_atlas_snapshot.v1",
            "extractor_version": "fixture",
            "source_commit": commit,
            "evidence_set_hash": _DIGEST,
            "access_scope": "private",
            "nodes": nodes,
            "edges": edges,
            "tombstones": [],
            "exclusions": [],
        }
    )


def _diamond(
    *extra_edges: dict[str, Any], extra_nodes: tuple[dict[str, Any], ...] = ()
) -> AtlasSnapshot:
    """Return the five-node directed diamond a→{b,c}→d→e.

    Parameters
    ----------
    *extra_edges : dict[str, Any]
        Additional edges.
    extra_nodes : tuple[dict[str, Any], ...]
        Additional nodes.

    Returns
    -------
    AtlasSnapshot
        The snapshot.

    """
    nodes = [_node(name) for name in "abcde"] + list(extra_nodes)
    edges = [
        _edge("a", "b"),
        _edge("a", "c"),
        _edge("b", "d"),
        _edge("c", "d"),
        _edge("d", "e"),
        *extra_edges,
    ]
    return _snapshot(nodes, edges)


def _id(name: str) -> str:
    """Return the fixture ID of a definition node.

    Parameters
    ----------
    name : str
        Short name.

    Returns
    -------
    str
        The node ID.

    """
    return f"definition:fixture/{name}"


def _refused(
    snapshot: AtlasSnapshot | None, query: AtlasQuery, access: str = "public"
) -> AtlasQueryError:
    """Return the refusal that running ``query`` raises.

    Parameters
    ----------
    snapshot : AtlasSnapshot | None
        Snapshot to query.
    query : AtlasQuery
        The query.
    access : str
        Access scope.

    Returns
    -------
    AtlasQueryError
        The refusal.

    """
    with pytest.raises(AtlasQueryError) as raised:
        run_atlas_query(snapshot, query, access=access)
    return raised.value


def test_atlas_bounded_queries_01_diamond_yields_exact_shortest_paths_without_duplicates_across_pages() -> (
    None
):
    """Both shortest paths come back, in order, one per page, and never twice."""
    snapshot = _diamond()
    query = AtlasQuery(start=_id("a"), target=_id("e"), depth=4, page_size=1)

    first = run_atlas_query(snapshot, query)
    second = run_atlas_query(
        snapshot, AtlasQuery(**{**query.__dict__, "cursor": first.next_cursor})
    )

    expected = [
        tuple(_id(name) for name in "abde"),
        tuple(_id(name) for name in "acde"),
    ]
    assert [*first.paths, *second.paths] == expected
    assert (first.total, first.omitted, second.omitted, second.next_cursor) == (2, 1, 0, None)
    assert {node.id for node in first.nodes} == set(expected[0])
    assert {(edge.source, edge.target) for edge in first.edges} == {
        (_id("a"), _id("b")),
        (_id("b"), _id("d")),
        (_id("d"), _id("e")),
    }
    assert first.shape == "path" and "not a proof" in first.explanation[2]
    assert (
        run_atlas_query(snapshot, AtlasQuery(start=_id("a"), target=_id("e"), depth=2)).paths == ()
    )


def test_atlas_bounded_queries_02_cycle_terminates() -> None:
    """A cycle is traversed once per node in neighbourhoods and paths."""
    snapshot = _diamond(_edge("e", "a"), _edge("d", "b"))

    around = run_atlas_query(snapshot, AtlasQuery(start=_id("a"), direction="both", depth=6))
    path = run_atlas_query(snapshot, AtlasQuery(start=_id("b"), target=_id("a"), depth=6))

    assert [node.id for node in around.nodes] == [_id("a"), _id("b"), _id("c"), _id("e"), _id("d")]
    assert path.paths == (tuple(_id(name) for name in "bdea"),)


def test_atlas_bounded_queries_03_hidden_branch_affects_neither_count_nor_omissions() -> None:
    """A private branch leaves public totals, omissions, digests and refusals unchanged."""
    hidden = _diamond(
        _edge("a", "p", access="private"),
        _edge("p", "e", access="private"),
        extra_nodes=(_node("p", access="private"),),
    )
    plain = _diamond()
    for query in (
        AtlasQuery(page_size=2),
        AtlasQuery(start=_id("a"), depth=3, page_size=2),
        AtlasQuery(start=_id("a"), target=_id("e"), depth=4, page_size=1),
    ):
        with_branch = run_atlas_query(hidden, query)
        without = run_atlas_query(plain, query)
        assert (with_branch.total, with_branch.omitted, with_branch.snapshot_digest) == (
            without.total,
            without.omitted,
            without.snapshot_digest,
        )
        assert with_branch.nodes == without.nodes and with_branch.paths == without.paths
    private = _refused(hidden, AtlasQuery(start=_id("p")))
    absent = _refused(hidden, AtlasQuery(start=_id("zz")))
    assert (private.code, private.message) == (absent.code, absent.message)
    seen = run_atlas_query(hidden, AtlasQuery(start=_id("a"), depth=1), access="private")
    assert _id("p") in {node.id for node in seen.nodes}


def test_atlas_bounded_queries_04_cursor_from_another_snapshot_rejects() -> None:
    """A cursor works only for its own snapshot, query and access scope."""
    first = _diamond()
    other = _snapshot([_node(name) for name in "abcde"], [_edge("a", "b")], commit="1" * 40)
    query = AtlasQuery(page_size=2)
    cursor = run_atlas_query(first, query).next_cursor

    def again(snapshot: AtlasSnapshot, access: str = "public", **change: Any) -> str:
        """Return the refusal code of a reused cursor."""
        reused = AtlasQuery(**{**query.__dict__, **change, "cursor": cursor})
        return _refused(snapshot, reused, access).code

    assert again(other) == "cursor_mismatch"
    assert again(first, text="node") == "cursor_mismatch"
    assert again(first, access="private") == "cursor_mismatch"
    assert _refused(first, AtlasQuery(cursor="not base64 !")).code == "invalid_cursor"
    assert _refused(first, AtlasQuery(cursor="WzFd")).code == "invalid_cursor"


def test_atlas_bounded_queries_05_capped_query_exposes_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capped page says how many items and edges it left out, and how to continue."""
    snapshot = _diamond()
    page = run_atlas_query(snapshot, AtlasQuery(page_size=2))

    assert (page.total, len(page.nodes), page.omitted, page.truncated) == (5, 2, 3, True)
    assert page.next_cursor is not None
    collected = [node.id for node in page.nodes]
    cursor: str | None = page.next_cursor
    while cursor is not None:
        follow = run_atlas_query(snapshot, AtlasQuery(page_size=2, cursor=cursor))
        collected.extend(node.id for node in follow.nodes)
        cursor = follow.next_cursor
    assert collected == [_id(name) for name in "abcde"]
    monkeypatch.setattr(query_module, "MAX_VISIBLE_EDGES", 2)
    wide = run_atlas_query(snapshot, AtlasQuery(start=_id("a"), depth=3))
    assert (len(wide.edges), wide.omitted_edges, wide.truncated) == (2, 3, True)


def test_lookup_filters_by_text_kind_and_status() -> None:
    """Text is case-folded over ID and label; facets combine conjunctively."""
    snapshot = _snapshot(
        [_node("alpha"), _node("beta", status="stale"), _node("gamma", kind="test")],
        [],
    )

    def ids(**filters: Any) -> list[str]:
        """Return the IDs a lookup with these filters returns."""
        return [node.id for node in run_atlas_query(snapshot, AtlasQuery(**filters)).nodes]

    assert ids(text="NODE A") == [_id("alpha")]
    assert ids(text="beta") == [_id("beta")]
    assert ids(kinds=("test",)) == ["test:fixture/gamma"]
    assert ids(statuses=("stale",)) == [_id("beta")]
    assert ids(kinds=("definition",), statuses=("current",)) == [_id("alpha")]


def test_neighbourhood_direction_and_edge_kinds() -> None:
    """Incoming and outgoing traversals follow only the admitted edge kinds."""
    snapshot = _diamond(
        _edge("doc", "e", kind="documents", src_kind="documentation"),
        extra_nodes=(_node("doc", kind="documentation"),),
    )

    incoming = run_atlas_query(snapshot, AtlasQuery(start=_id("d"), direction="in", depth=1))
    restricted = run_atlas_query(
        snapshot, AtlasQuery(start=_id("a"), depth=6, edge_kinds=("contains",))
    )

    assert [node.id for node in incoming.nodes] == [_id("d"), _id("b"), _id("c")]
    assert [node.id for node in restricted.nodes] == [_id("a")]
    assert "edge kinds contains" in restricted.explanation


@pytest.mark.parametrize(
    ("query", "code", "field"),
    [
        (AtlasQuery(edge_kinds=("equivalent_to",)), "unknown_relation", "edge_kinds"),
        (AtlasQuery(kinds=("module",)), "unknown_node_kind", "kinds"),
        (AtlasQuery(statuses=("fresh",)), "unknown_status", "statuses"),
        (AtlasQuery(direction=cast(Any, "sideways")), "invalid_direction", "direction"),
        (AtlasQuery(depth=7), "oversized_traversal", "depth"),
        (AtlasQuery(depth=0), "oversized_traversal", "depth"),
        (AtlasQuery(page_size=501), "oversized_page", "page_size"),
        (AtlasQuery(page_size=0), "oversized_page", "page_size"),
        (AtlasQuery(target="definition:fixture/e"), "missing_start", "start"),
        (AtlasQuery(start="definition:fixture/zz"), "unknown_node", "start"),
        (
            AtlasQuery(start="definition:fixture/a", target="definition:fixture/zz"),
            "unknown_node",
            "target",
        ),
    ],
    ids=[
        "relation",
        "kind",
        "status",
        "direction",
        "depth-high",
        "depth-zero",
        "page-high",
        "page-zero",
        "target-alone",
        "start",
        "target",
    ],
)
def test_malformed_queries_refuse(query: AtlasQuery, code: str, field: str) -> None:
    """Every malformed or out-of-bound query is refused at its field."""
    refusal = _refused(_diamond(), query)

    assert (refusal.code, refusal.field) == (code, field)
    assert str(refusal).startswith(code)


def test_snapshot_and_access_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent or oversized snapshot and an unadmitted access scope are refused."""
    snapshot = _diamond()
    public = run_atlas_query(snapshot, AtlasQuery()).nodes

    assert _refused(None, AtlasQuery()).code == "absent_snapshot"
    assert _refused(snapshot, AtlasQuery(), access="admin").code == "access_refused"
    visible = parse_atlas_snapshot(
        {
            "schema": "quantum_atlas_snapshot.v1",
            "extractor_version": "fixture",
            "source_commit": "0" * 40,
            "evidence_set_hash": _DIGEST,
            "access_scope": "public",
            "nodes": [_node("a")],
            "edges": [],
            "tombstones": [],
            "exclusions": [],
        }
    )
    assert _refused(visible, AtlasQuery(), access="private").code == "access_refused"
    assert len(public) == 5
    monkeypatch.setattr(query_module, "NODE_CEILING", 4)
    assert _refused(snapshot, AtlasQuery()).code == "snapshot_too_large"


def test_a_cursor_whose_last_item_left_the_answer_rejects() -> None:
    """A bound cursor whose last item is no longer in the answer is refused, not silently restarted."""
    snapshot = _diamond()
    page = run_atlas_query(snapshot, AtlasQuery(page_size=2))
    import base64
    import json

    raw = json.loads(base64.urlsafe_b64decode(str(page.next_cursor)))
    raw["last"] = "definition:fixture/zz"
    forged = base64.urlsafe_b64encode(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).decode()

    assert _refused(snapshot, AtlasQuery(page_size=2, cursor=forged)).code == "cursor_mismatch"


def test_every_design_question_is_answered_or_refused_with_its_reason() -> None:
    """Supported questions answer; unsupported ones refuse with the input they still need."""
    claims = _snapshot(
        [
            _node("core"),
            _node("claim_one", kind="claim"),
            _node("cap", kind="capability"),
            _node("cap_tested", kind="capability"),
            _node("t", kind="test"),
            _node("guide", kind="documentation"),
        ],
        [
            _edge("claim_one", "core", kind="derives_from", src_kind="claim"),
            _edge("cap_tested", "t", kind="contains", src_kind="capability", dst_kind="test"),
            _edge(
                "guide", "cap", kind="documents", src_kind="documentation", dst_kind="capability"
            ),
        ],
    )
    dependents = run_prebuilt_query(claims, "dependents_of", node=_id("core"))
    untested = run_prebuilt_query(claims, "capabilities_without_tests")
    undocumented = run_prebuilt_query(claims, "capabilities_without_documentation")

    assert [node.id for node in dependents.nodes] == [_id("core"), "claim:fixture/claim_one"]
    assert [node.id for node in untested.nodes] == ["capability:fixture/cap"]
    assert [node.id for node in undocumented.nodes] == ["capability:fixture/cap_tested"]
    assert len(PREBUILT_QUERIES) == 11
    for name, prebuilt in PREBUILT_QUERIES.items():
        if prebuilt.unsupported is None:
            continue
        with pytest.raises(AtlasQueryError) as refusal:
            run_prebuilt_query(claims, name)
        assert (refusal.value.code, refusal.value.message) == (
            "unsupported_prebuilt_query",
            prebuilt.unsupported,
        )
    for name, node, code in (
        ("no_such_question", None, "unknown_prebuilt_query"),
        ("dependents_of", None, "missing_start"),
    ):
        with pytest.raises(AtlasQueryError) as refusal:
            run_prebuilt_query(claims, name, node=node)
        assert refusal.value.code == code
    with pytest.raises(AtlasQueryError) as absent:
        run_prebuilt_query(None, "capabilities_without_tests")
    assert absent.value.code == "absent_snapshot"


def test_queries_run_on_the_extracted_committed_snapshot() -> None:
    """The extracted repository answers a capability lookup and a bounded neighbourhood."""
    root = Path(__file__).resolve().parents[1]
    snapshot = extract_atlas_snapshot(directory_reader(root), "0" * 40).snapshot
    capabilities = run_atlas_query(snapshot, AtlasQuery(kinds=("capability",)))
    first = capabilities.nodes[0].id
    around = run_atlas_query(snapshot, AtlasQuery(start=first, direction="in", depth=1))

    assert capabilities.total >= 1 and capabilities.next_cursor is None
    assert around.nodes[0].id == first and around.total >= 2
