# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Atlas snapshot contract tests
"""Tests for the Atlas snapshot ontology, identity, legality and access contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.atlas import (
    AtlasContractError,
    atlas_snapshot_digest,
    parse_atlas_snapshot,
    project_public,
    snapshot_to_wire,
)
from scpn_quantum_control.atlas.contracts import (
    EDGE_ENDPOINTS,
    EDGE_KINDS,
    IDENTITY_CLASSES,
    NODE_KINDS,
)

_CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "data/atlas/contracts.json").read_text(encoding="utf-8")
)
_FIXTURES: dict[str, dict[str, Any]] = _CORPUS["fixtures"]
_CASES: dict[str, dict[str, Any]] = {case["id"]: case for case in _CORPUS["cases"]}


def _fixture(name: str) -> dict[str, Any]:
    """Return a private copy of one corpus fixture.

    Parameters
    ----------
    name : str
        Fixture name in the shared corpus.

    Returns
    -------
    dict[str, Any]
        A deep copy, so a test can mutate it freely.

    """
    return copy.deepcopy(_FIXTURES[name])


def _refusal(payload: object) -> AtlasContractError:
    """Return the refusal that parsing ``payload`` raises.

    Parameters
    ----------
    payload : object
        Candidate snapshot document.

    Returns
    -------
    AtlasContractError
        The raised refusal.

    """
    with pytest.raises(AtlasContractError) as raised:
        parse_atlas_snapshot(payload)
    return raised.value


def test_atlas_contracts_01_valid_graph_digest_matches_the_shared_corpus() -> None:
    """The Python digest of the valid graph equals the corpus value the browser checks too."""
    snapshot = parse_atlas_snapshot(_fixture("valid"))

    assert atlas_snapshot_digest(snapshot) == _CASES["valid"]["digest"]
    for name in ("valid_reordered", "valid_canonical_spelling"):
        assert (
            atlas_snapshot_digest(parse_atlas_snapshot(_fixture(name)))
            == _CASES["valid"]["digest"]
        )
    assert [edge.kind for edge in snapshot.edges] == _CASES["valid"]["edge_kinds"]


def test_atlas_contracts_02_dangling_endpoint_and_unknown_edge_version_reject() -> None:
    """An edge to a missing node and an unknown relation version are refused at their field."""
    for name, code, path in (
        ("dangling_endpoint", "dangling_endpoint", "$.edges[0].target"),
        ("unknown_relation_version", "unknown_relation_version", "$.edges[0].relation_version"),
    ):
        payload = _fixture(name)
        before = copy.deepcopy(payload)
        refusal = _refusal(payload)

        assert (refusal.code, refusal.path) == (code, path)
        assert payload == before


def test_atlas_contracts_03_import_edge_cannot_satisfy_supports_proof() -> None:
    """Import adjacency never establishes support; support needs evidence and assumptions."""
    assert _refusal(_fixture("import_adjacency_support")).code == "adjacency_is_not_support"
    assert _refusal(_fixture("support_without_evidence")).code == "missing_evidence"
    assert _refusal(_fixture("support_without_assumptions")).code == "missing_assumptions"
    supports = [
        edge for edge in parse_atlas_snapshot(_fixture("valid")).edges if edge.kind == "supports"
    ]
    assert [edge.derivation_method for edge in supports] == ["measured"]


def test_atlas_contracts_04_private_node_stays_private() -> None:
    """A public snapshot holds no private material and the projection discloses none."""
    assert _refusal(_fixture("private_node_in_public_snapshot")).code == "access_leak"
    assert _refusal(_fixture("public_edge_touching_private_node")).code == "access_leak"
    snapshot = parse_atlas_snapshot(_fixture("valid"))
    public = project_public(snapshot)
    wire_text = json.dumps(snapshot_to_wire(public))

    assert [node.id for node in public.nodes] == _CASES["valid"]["public_node_ids"]
    assert atlas_snapshot_digest(public) == _CASES["valid"]["public_digest"]
    assert "draft_notes" not in wire_text
    assert all(edge.access == "public" for edge in public.edges)
    assert parse_atlas_snapshot(snapshot_to_wire(public)) == public


def test_atlas_contracts_05_tombstone_preserves_prior_definition_identity() -> None:
    """A tombstone keeps the exact deleted identity, which no live node or edge may reuse."""
    tombstone = parse_atlas_snapshot(_fixture("valid")).tombstones[0]

    assert (tombstone.id, tombstone.kind) == (
        "definition:scpn_quantum_control.fixture/retired_energy",
        "definition",
    )
    assert _refusal(_fixture("tombstoned_id_reused")).code == "reused_id"
    assert _refusal(_fixture("edge_to_tombstone")).code == "tombstoned_endpoint"
    bad = _fixture("valid")
    bad["tombstones"][0]["id"] = "definition:no-owner-separator"
    assert _refusal(bad).code == "invalid_identity"


@pytest.mark.parametrize("case_id", sorted(_CASES))
def test_every_corpus_case_meets_its_recorded_expectation(case_id: str) -> None:
    """Each shared case accepts with its digest or rejects with its exact code and path."""
    case = _CASES[case_id]
    payload = _fixture(case["fixture"])
    if case["expectation"] == "accept":
        assert atlas_snapshot_digest(parse_atlas_snapshot(payload)) == case["digest"]
    else:
        refusal = _refusal(payload)
        assert (refusal.code, refusal.path) == (case["code"], case["path"])


def test_aliases_normalise_and_verifies_reverses_direction() -> None:
    """Version-1 aliases become canonical kinds; ``verifies`` swaps source and target."""
    snapshot = parse_atlas_snapshot(_fixture("valid"))
    verified = [edge for edge in snapshot.edges if edge.kind == "verified_by"]

    assert [(edge.source, edge.target) for edge in verified] == [
        ("definition:scpn_quantum_control.fixture/energy", "test:tests.fixture/test_energy")
    ]
    assert "produces" in {edge.kind for edge in snapshot.edges}
    assert _refusal(_fixture("refused_bare_contradiction")).code == "refused_edge_kind"
    assert _refusal(_fixture("unknown_edge_kind")).code == "unknown_edge_kind"


def test_contradiction_needs_comparable_claims() -> None:
    """Claims with different units are non-comparable, never contradictory."""
    refusal = _refusal(_fixture("non_comparable_contradiction"))

    assert (refusal.code, refusal.path) == ("non_comparable", "$.edges[5]")
    assert "units" in refusal.message


def test_identity_classes_and_prov_mapping() -> None:
    """Each kind has one identity class; provenance edges and run nodes map to PROV-O."""
    snapshot = parse_atlas_snapshot(_fixture("valid"))
    run = next(node for node in snapshot.nodes if node.kind == "run")
    artefact = next(node for node in snapshot.nodes if node.kind == "artefact")
    produces = next(edge for edge in snapshot.edges if edge.kind == "produces")
    implements = next(edge for edge in snapshot.edges if edge.kind == "implements")

    assert set(IDENTITY_CLASSES) == set(NODE_KINDS)
    assert set(EDGE_ENDPOINTS) == set(EDGE_KINDS)
    assert (run.identity_class, run.prov_class) == ("instance", "prov:Activity")
    assert (artefact.identity_class, artefact.prov_class) == ("revision", "prov:Entity")
    assert produces.prov_relation == "prov:wasGeneratedBy"
    assert implements.prov_relation is None
    assert snapshot.node(run.id) is run
    with pytest.raises(KeyError):
        snapshot.node("definition:missing/node")


def test_wire_round_trip_is_stable() -> None:
    """Parsing the canonical wire document again yields an equal snapshot and digest."""
    snapshot = parse_atlas_snapshot(_fixture("valid_reordered"))
    again = parse_atlas_snapshot(snapshot_to_wire(snapshot))

    assert again == snapshot
    assert atlas_snapshot_digest(again) == atlas_snapshot_digest(snapshot)


def _mutated(path: tuple[object, ...], value: object, *, delete: bool = False) -> dict[str, Any]:
    """Return the valid fixture with one field replaced or deleted.

    Parameters
    ----------
    path : tuple[object, ...]
        Keys and indices leading to the field.
    value : object
        Replacement value.
    delete : bool
        Delete the field instead of replacing it.

    Returns
    -------
    dict[str, Any]
        The mutated document.

    """
    document = _fixture("valid")
    holder: Any = document
    for key in path[:-1]:
        holder = holder[key]
    if delete:
        del holder[path[-1]]
    else:
        holder[path[-1]] = value
    return document


@pytest.mark.parametrize(
    ("path", "value", "code", "where"),
    [
        (("nodes",), "not a list", "invalid_type", "$.nodes"),
        (("nodes", 0), "not an object", "invalid_type", "$.nodes[0]"),
        (("nodes", 0, "label"), "", "invalid_type", "$.nodes[0].label"),
        (("nodes", 0, "kind"), "module", "unknown_node_kind", "$.nodes[0].kind"),
        (("nodes", 0, "id"), "capability:no-separator", "invalid_identity", "$.nodes[0].id"),
        (("nodes", 3, "id"), "run:not-a-uuid/also-not", "invalid_identity", "$.nodes[3].id"),
        (("nodes", 0, "access"), "secret", "unknown_access", "$.nodes[0].access"),
        (("nodes", 0, "status"), "fresh", "unknown_status", "$.nodes[0].status"),
        (("nodes", 4, "evidence_refs"), "abc", "invalid_type", "$.nodes[4].evidence_refs"),
        (("nodes", 4, "evidence_refs", 0), "ABC", "invalid_digest", "$.nodes[4].evidence_refs[0]"),
        (("nodes", 8, "validity_domain"), ["units"], "invalid_type", "$.nodes[8].validity_domain"),
        (
            ("nodes", 0, "null_reasons", "as_of"),
            "as_of is set",
            "invalid_null_reason",
            "$.nodes[0].null_reasons.as_of",
        ),
        (("nodes", 1, "as_of"), None, "missing_null_reason", "$.nodes[1].as_of"),
        (
            ("edges", 0, "derivation_method"),
            "guessed",
            "unknown_derivation_method",
            "$.edges[0].derivation_method",
        ),
        (("edges", 3, "evidence_digest"), None, "missing_evidence", "$.edges[3].evidence_digest"),
        (("edges", 0, "access"), "secret", "unknown_access", "$.edges[0].access"),
        (("edges", 0, "assumptions"), "none", "invalid_type", "$.edges[0].assumptions"),
        (("source_commit",), "abc", "invalid_commit", "$.source_commit"),
        (("evidence_set_hash",), "0" * 63, "invalid_digest", "$.evidence_set_hash"),
        (("tombstones", 0, "kind"), "module", "unknown_node_kind", "$.tombstones[0].kind"),
        (("exclusions", 0, "reason"), "", "invalid_type", "$.exclusions[0].reason"),
    ],
)
def test_malformed_fields_are_refused_at_their_path(
    path: tuple[object, ...], value: object, code: str, where: str
) -> None:
    """Every malformed field is refused with its stable code and exact JSON path."""
    refusal = _refusal(_mutated(path, value))

    assert (refusal.code, refusal.path) == (code, where)


def test_closed_documents_refuse_unknown_and_missing_fields() -> None:
    """Unknown fields and absent required fields are refused, never silently kept or defaulted."""
    unknown = _fixture("valid")
    unknown["nodes"][0]["colour"] = "red"
    assert (_refusal(unknown).code, _refusal(unknown).path) == (
        "unknown_field",
        "$.nodes[0].colour",
    )
    missing = _mutated(("edges", 0, "as_of"), None, delete=True)
    assert (_refusal(missing).code, _refusal(missing).path) == (
        "missing_field",
        "$.edges[0].as_of",
    )
    assert _refusal(["not", "an", "object"]).code == "invalid_type"
    assert _refusal({1: "non-string key"}).code == "invalid_type"


def test_duplicates_are_refused() -> None:
    """A node, tombstone or relation may appear only once."""
    nodes = _fixture("valid")
    nodes["nodes"].append(copy.deepcopy(nodes["nodes"][0]))
    tombstones = _fixture("valid")
    tombstones["tombstones"].append(copy.deepcopy(tombstones["tombstones"][0]))
    edges = _fixture("valid")
    edges["edges"].append(copy.deepcopy(edges["edges"][0]))

    assert _refusal(nodes).code == "duplicate_id"
    assert _refusal(tombstones).code == "duplicate_id"
    assert _refusal(edges).code == "duplicate_edge"


def test_public_snapshot_refuses_a_private_edge_between_public_nodes() -> None:
    """A public snapshot holds no private relation even between public nodes."""
    document = _fixture("valid")
    document["nodes"] = [node for node in document["nodes"] if node["access"] == "public"]
    document["edges"] = [edge for edge in document["edges"] if edge["access"] == "public"]
    document["access_scope"] = "public"
    document["edges"][0]["access"] = "private"

    assert (_refusal(document).code, _refusal(document).path) == (
        "access_leak",
        "$.edges[0].access",
    )


def test_maps_to_with_units_and_semantics_is_admitted() -> None:
    """A mapping carries its units and semantics and stays distinct from equivalence."""
    document = _fixture("maps_to_without_mapping")
    document["edges"][-1]["mapping"] = {
        "units": "hartree",
        "semantics": "same estimand, other API",
    }
    snapshot = parse_atlas_snapshot(document)

    mapped = next(edge for edge in snapshot.edges if edge.kind == "maps_to")
    assert dict(mapped.mapping or ()) == {
        "semantics": "same estimand, other API",
        "units": "hartree",
    }


def test_error_carries_code_path_and_message() -> None:
    """The refusal exposes its stable parts and stays a ValueError."""
    refusal = _refusal(_fixture("unknown_major_schema"))

    assert isinstance(refusal, ValueError)
    assert (refusal.code, refusal.path) == ("unknown_schema", "$.schema")
    assert "quantum_atlas_snapshot.v2" in refusal.message
    assert str(refusal).startswith("unknown_schema at $.schema")
