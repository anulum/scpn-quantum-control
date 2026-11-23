# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — temporal diffs of Atlas snapshots
"""Check snapshot diffs, their classification, impact paths and completeness."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from scpn_quantum_control.atlas import AtlasSnapshot, parse_atlas_snapshot
from scpn_quantum_control.atlas.diff import (
    AtlasDiffError,
    diff_atlas_snapshots,
    diff_to_wire,
    node_content_hash,
    resolve_saved_view,
)

Json = dict[str, object]
EVIDENCE = "e" * 64


def node(
    node_id: str,
    kind: str = "definition",
    *,
    digest: str | None = "1" * 64,
    status: str = "current",
    access: str = "public",
) -> Json:
    """Return a valid wire node.

    Parameters
    ----------
    node_id : str
        Node identifier.
    kind : str
        Node kind.
    digest : str | None
        Source digest.
    status : str
        Node status.
    access : str
        Access class.

    Returns
    -------
    Json
        The node.

    """
    reasons = {"as_of": "fixture", "validity_domain": "fixture"}
    if digest is None:
        reasons["source_digest"] = "fixture"
    return {
        "id": node_id,
        "kind": kind,
        "label": node_id,
        "source_ref": "fixture.json",
        "source_digest": digest,
        "validity_domain": None,
        "as_of": None,
        "access": access,
        "status": status,
        "evidence_refs": [EVIDENCE],
        "null_reasons": reasons,
    }


def edge(source: str, kind: str, target: str, *, derivation: str = "extracted") -> Json:
    """Return a valid wire edge.

    Parameters
    ----------
    source : str
        Source node identifier.
    kind : str
        Relation kind.
    target : str
        Target node identifier.
    derivation : str
        How the relation was derived.

    Returns
    -------
    Json
        The edge.

    """
    return {
        "source": source,
        "kind": kind,
        "target": target,
        "relation_version": "atlas_relation.v1",
        "derivation_method": derivation,
        "source_ref": "fixture.json",
        "evidence_digest": EVIDENCE,
        "assumptions": [],
        "mapping": {"semantics": "fixture", "units": "fixture"} if kind == "maps_to" else None,
        "as_of": None,
        "access": "public",
    }


def snapshot(
    nodes: Sequence[Json],
    edges: Sequence[Json] = (),
    *,
    tombstones: Sequence[Json] = (),
    exclusions: Sequence[Json] = (),
) -> AtlasSnapshot:
    """Parse a public snapshot of the given objects.

    Parameters
    ----------
    nodes : Sequence[Json]
        Wire nodes.
    edges : Sequence[Json]
        Wire edges.
    tombstones : Sequence[Json]
        Wire tombstones.
    exclusions : Sequence[Json]
        Wire exclusions.

    Returns
    -------
    AtlasSnapshot
        The validated snapshot.

    """
    ordered: list[Json] = sorted(nodes, key=lambda item: str(item["id"]))
    return parse_atlas_snapshot(
        {
            "schema": "quantum_atlas_snapshot.v1",
            "extractor_version": "fixture",
            "source_commit": "0" * 40,
            "evidence_set_hash": EVIDENCE,
            "access_scope": "public",
            "nodes": ordered,
            "edges": list(edges),
            "tombstones": list(tombstones),
            "exclusions": list(exclusions),
        }
    )


TOP, LEFT, RIGHT, BOTTOM = (
    f"definition:repo/{name}.py" for name in ("top", "left", "right", "bottom")
)
UNRELATED = "definition:repo/mapped.py"


def diamond(top_digest: str = "1" * 64, *, derivation: str = "extracted") -> AtlasSnapshot:
    """Return the diamond: left and right derive from top, bottom from both.

    Parameters
    ----------
    top_digest : str
        Source digest of the top node.
    derivation : str
        Derivation of every relation.

    Returns
    -------
    AtlasSnapshot
        The snapshot, plus a node the top maps to, which carries no impact.

    """
    return snapshot(
        [node(TOP, digest=top_digest), node(LEFT), node(RIGHT), node(BOTTOM), node(UNRELATED)],
        [
            edge(LEFT, "derives_from", TOP, derivation=derivation),
            edge(RIGHT, "derives_from", TOP, derivation=derivation),
            edge(BOTTOM, "derives_from", LEFT, derivation=derivation),
            edge(BOTTOM, "derives_from", RIGHT, derivation=derivation),
            edge(TOP, "maps_to", UNRELATED, derivation=derivation),
        ],
    )


def test_atlas_snapshot_diff_01() -> None:
    """Changing one diamond node impacts exactly its allowed-edge descendants."""
    diff = diff_atlas_snapshots(diamond(), diamond("2" * 64))

    assert [(item.id, item.change, item.category, item.fields) for item in diff.nodes] == [
        (TOP, "changed", "code", ("source_digest",))
    ]
    assert diff.unchanged == 4
    assert diff.edges == ()
    assert [item.id for item in diff.impact] == [BOTTOM, LEFT, RIGHT]
    bottom = diff.impact[0]
    assert bottom.origin == TOP
    assert [(step.source, step.kind, step.target) for step in bottom.path] == [
        (LEFT, "derives_from", TOP),
        (BOTTOM, "derives_from", LEFT),
    ]
    assert diff.complete is True
    assert diff.incomplete_reasons == ()


def test_atlas_snapshot_diff_02() -> None:
    """Additions, removals and tombstoned removals are classified apart."""
    kept, buried, dropped, new = (f"artefact:sha256:{digit * 64}" for digit in "abcd")
    before = snapshot(
        [
            node(kept, "artefact", digest="a" * 64),
            node(buried, "artefact", digest="b" * 64),
            node(dropped, "artefact", digest="c" * 64),
        ]
    )
    after = snapshot(
        [node(kept, "artefact", digest="a" * 64), node(new, "artefact", digest="d" * 64)],
        tombstones=[
            {
                "id": buried,
                "kind": "artefact",
                "last_source_digest": "b" * 64,
                "removed_at": "2026-11-23T00:00:00Z",
                "reason": "replaced",
            }
        ],
    )

    diff = diff_atlas_snapshots(before, after)

    assert {item.id: item.change for item in diff.nodes} == {
        buried: "tombstoned",
        dropped: "removed",
        new: "added",
    }
    tombstoned = next(item for item in diff.nodes if item.change == "tombstoned")
    assert tombstoned.reason == "replaced (removed at 2026-11-23T00:00:00Z)"
    assert tombstoned.after_hash is None and tombstoned.before_hash is not None
    added = next(item for item in diff.nodes if item.change == "added")
    assert added.before_hash is None and added.category == "data"
    assert diff.unchanged == 1


def test_atlas_snapshot_diff_03() -> None:
    """Reversing the comparison swaps additions and removals."""
    one, two = "definition:repo/one.py", "definition:repo/two.py"
    before = snapshot([node(one), node(TOP)], [edge(one, "derives_from", TOP)])
    after = snapshot([node(two), node(TOP)], [edge(two, "derives_from", TOP)])

    forward = diff_atlas_snapshots(before, after)
    backward = diff_atlas_snapshots(after, before)

    assert {(item.id, item.change) for item in forward.nodes} == {(one, "removed"), (two, "added")}
    assert {(item.id, item.change) for item in backward.nodes} == {
        (one, "added"),
        (two, "removed"),
    }
    assert {(item.source, item.change) for item in forward.edges} == {
        (one, "removed"),
        (two, "added"),
    }
    assert {(item.source, item.change) for item in backward.edges} == {
        (one, "added"),
        (two, "removed"),
    }


def test_atlas_snapshot_diff_04() -> None:
    """Unobserved relations, exclusions and non-current nodes leave the analysis incomplete."""
    declared = diff_atlas_snapshots(
        diamond(derivation="declared"), diamond("2" * 64, derivation="declared")
    )
    assert [item.id for item in declared.impact] == [BOTTOM, LEFT, RIGHT]
    assert declared.complete is False
    assert declared.incomplete_reasons == ("declared relations were followed, not observed ones",)

    excluded = snapshot(
        [node(TOP)], exclusions=[{"source_ref": "fixture.json:missing.py", "reason": "absent"}]
    )
    stale = snapshot([node(TOP, digest="2" * 64, status="stale")])
    diff = diff_atlas_snapshots(excluded, stale)
    assert diff.complete is False
    assert diff.incomplete_reasons == (
        f"{TOP} is stale",
        "the before extraction excluded 1 inputs or objects",
    )


def test_changed_relations_and_calibration_are_reported() -> None:
    """A relation whose content changed is reported; a calibrated target is a calibration change."""
    run = "run:11111111-2222-4333-8444-555555555555/66666666-7777-4888-8999-aaaaaaaaaaaa"
    calibration = f"artefact:sha256:{'a' * 64}"
    before = snapshot(
        [node(run, "run"), node(calibration, "artefact", digest="a" * 64)],
        [edge(run, "calibrated_by", calibration)],
    )
    renamed = node(calibration, "artefact", digest="a" * 64)
    renamed["label"] = "renamed"
    after = snapshot(
        [node(run, "run"), renamed],
        [edge(run, "calibrated_by", calibration, derivation="measured")],
    )

    diff = diff_atlas_snapshots(before, after)

    assert [(item.id, item.category, item.fields) for item in diff.nodes] == [
        (calibration, "calibration", ("label",))
    ]
    assert [(item.kind, item.change) for item in diff.edges] == [("calibrated_by", "changed")]
    assert [item.id for item in diff.impact] == [run]
    assert diff.complete is True


def test_a_removed_node_impacts_its_former_dependents() -> None:
    """A removal reaches the dependents its earlier relations named."""
    before = diamond()
    after = snapshot(
        [node(LEFT), node(RIGHT), node(BOTTOM), node(UNRELATED)],
        [edge(BOTTOM, "derives_from", LEFT), edge(BOTTOM, "derives_from", RIGHT)],
    )
    diff = diff_atlas_snapshots(before, after)
    assert [(item.id, item.change) for item in diff.nodes] == [(TOP, "removed")]
    assert [item.id for item in diff.impact] == [BOTTOM, LEFT, RIGHT]


def test_snapshots_of_different_scopes_are_not_compared() -> None:
    """A public snapshot is not compared with a private one."""
    public = snapshot([node(TOP)])
    wire = {
        "schema": "quantum_atlas_snapshot.v1",
        "extractor_version": "fixture",
        "source_commit": "0" * 40,
        "evidence_set_hash": EVIDENCE,
        "access_scope": "private",
        "nodes": [node(TOP)],
        "edges": [],
        "tombstones": [],
        "exclusions": [],
    }
    with pytest.raises(
        AtlasDiffError, match="cannot compare a public snapshot with a private one"
    ):
        diff_atlas_snapshots(public, parse_atlas_snapshot(wire))


def test_content_hash_ignores_evidence_references() -> None:
    """Moving evidence references does not change what a node states."""
    first, second = node(TOP), node(TOP)
    second["evidence_refs"] = ["f" * 64]
    assert node_content_hash(first) == node_content_hash(second)
    second["label"] = "other"
    assert node_content_hash(first) != node_content_hash(second)


def test_saved_views_resolve_present_tombstoned_or_absent() -> None:
    """Each node a saved view names is present, tombstoned with its reason, or absent."""
    gone = "definition:repo/gone.py"
    later = snapshot(
        [node(TOP)],
        tombstones=[
            {
                "id": gone,
                "kind": "definition",
                "last_source_digest": "1" * 64,
                "removed_at": "2026-11-23T00:00:00Z",
                "reason": "deleted",
            }
        ],
    )
    references = resolve_saved_view(later, [TOP, gone, "definition:repo/never.py", TOP])
    assert [(item.id, item.state, item.explanation) for item in references] == [
        (gone, "tombstoned", "deleted (removed at 2026-11-23T00:00:00Z)"),
        (
            "definition:repo/never.py",
            "absent",
            "the snapshot neither holds it nor records its removal",
        ),
        (TOP, "present", ""),
    ]


def test_the_wire_form_keeps_every_classification_and_path() -> None:
    """The JSON form lists the node and relation changes, the impact paths and the verdict."""
    wire = diff_to_wire(
        diff_atlas_snapshots(
            diamond(derivation="declared"), diamond("2" * 64, derivation="declared")
        )
    )

    assert wire["unchanged"] == 4
    assert wire["edges"] == []
    assert wire["nodes"] == [
        {
            "id": TOP,
            "kind": "definition",
            "change": "changed",
            "category": "code",
            "fields": ["source_digest"],
            "before_hash": node_content_hash(node(TOP)),
            "after_hash": node_content_hash(node(TOP, digest="2" * 64)),
            "reason": None,
        }
    ]
    impact = wire["impact"]
    assert isinstance(impact, list)
    assert impact[0] == {
        "id": BOTTOM,
        "origin": TOP,
        "path": [
            {"source": LEFT, "kind": "derives_from", "target": TOP},
            {"source": BOTTOM, "kind": "derives_from", "target": LEFT},
        ],
    }
    assert wire["complete"] is False
    assert wire["incomplete_reasons"] == ["declared relations were followed, not observed ones"]
