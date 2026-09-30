# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — workspace reference admission tests
"""Exercise complete metadata graphs with an explicit synthetic producer codec."""

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.studio_workspace.canonical import canonical_digest
from scpn_quantum_control.studio_workspace.contracts import (
    ExperimentRevision,
    LocalRunRecord,
    ParameterSpec,
    WorkspaceDocument,
    WorkspaceManifest,
    parse_document,
    parse_parameter_spec,
    parse_workspace_manifest,
)
from scpn_quantum_control.studio_workspace.graph import RawArtifact, RawIdentity, admit_workspace
from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json

_FIXTURES = json.loads(
    (Path(__file__).parent / "data/studio_workspace/documents.json").read_text()
)["fixtures"]
_Bundle = tuple[
    WorkspaceManifest,
    dict[str, WorkspaceDocument],
    dict[str, RawArtifact],
    dict[str, Callable[[bytes], RawIdentity]],
    dict[str, ParameterSpec],
    dict[str, str],
]


def _fixture_codec(content: bytes) -> RawIdentity:
    payload = cast(dict[str, object], read_json(content.decode()))
    body = cast(dict[str, object], payload["body"])
    if payload["schema"] != "review_fixture.v1" or body["synthetic"] is not True:
        raise ValueError("synthetic conformance producer required")
    return RawIdentity(
        "review_fixture.v1", str(body["role"]), canonical_digest("review_fixture.v1", payload)
    )


def _bundle() -> _Bundle:
    names = ("settings", "parameter", "revision_root", "revision_child", "run")
    records = [parse_document(_FIXTURES[name]) for name in names]
    raw: dict[str, RawArtifact] = {}
    for name in ("problem", "program", "policy", "environment", "plan"):
        content = write_json(_FIXTURES[name]).encode()
        raw[_fixture_codec(content).digest] = RawArtifact("review_fixture.v1", content)
    spec = parse_parameter_spec(_FIXTURES["parameter"])
    return (
        parse_workspace_manifest(_FIXTURES["workspace"]),
        {record.digest: record for record in records},
        raw,
        {"review_fixture.v1": _fixture_codec},
        {"theta": spec},
        {"theta": "rad"},
    )


def test_complete_graph_and_raw_custody() -> None:
    """Admit real public document objects and preserve original producer bytes."""
    bundle = _bundle()
    before = {key: artifact.content for key, artifact in bundle[2].items()}
    receipt = admit_workspace(*bundle)
    assert receipt.project_id == bundle[0].body["project_id"]
    assert receipt.workspace_hash == bundle[0].digest
    assert receipt.document_hashes == tuple(sorted(bundle[1]))
    assert receipt.raw_hashes == tuple(sorted(bundle[2]))
    assert {key: artifact.content for key, artifact in bundle[2].items()} == before


def test_no_registered_codec_never_claims_raw_support() -> None:
    """Unknown raw schemas refuse rather than using a synthetic fallback."""
    manifest, documents, raw, _, specs, units = _bundle()
    with pytest.raises(ValueError, match="unsupported raw producer"):
        admit_workspace(manifest, documents, raw, {}, specs, units)


def test_digest_index_is_not_trusted() -> None:
    """Refuse a caller index that rebinds a digest to another document."""
    bundle = _bundle()
    keys = list(bundle[1])
    bundle[1][keys[0]] = bundle[1][keys[1]]
    with pytest.raises(ValueError, match="digest mismatch"):
        admit_workspace(*bundle)


def test_raw_tampering_and_false_producer_identity() -> None:
    """Recompute identities and reject altered bytes or a mismatched receipt."""
    bundle = _bundle()
    key = next(iter(bundle[2]))
    changed = cast(dict[str, object], read_json(bundle[2][key].content.decode()))
    cast(dict[str, object], changed["body"])["purpose"] = "changed"
    bundle[2][key] = RawArtifact("review_fixture.v1", write_json(changed).encode())
    with pytest.raises(ValueError, match="raw producer identity mismatch"):
        admit_workspace(*bundle)
    bundle = _bundle()
    bundle[3]["review_fixture.v1"] = lambda content: RawIdentity(
        "wrong.v1", "problem", _fixture_codec(content).digest
    )
    with pytest.raises(ValueError, match="raw producer identity mismatch"):
        admit_workspace(*bundle)


def test_unit_context_is_snapshotted_before_callbacks() -> None:
    """A producer callback cannot change the meaning of later parameter checks."""
    bundle = _bundle()

    def producer(content: bytes) -> RawIdentity:
        bundle[5]["theta"] = "Hz"
        return _fixture_codec(content)

    bundle[3]["review_fixture.v1"] = producer
    assert admit_workspace(*bundle).project_id == bundle[0].body["project_id"]
    assert bundle[5]["theta"] == "Hz"


def test_missing_units_specs_and_raw_references() -> None:
    """Require complete context without guessing units or missing sources."""
    bundle = _bundle()
    bundle[5].clear()
    with pytest.raises(ValueError, match="keys must match"):
        admit_workspace(*bundle)
    bundle = _bundle()
    bundle[2].clear()
    with pytest.raises(ValueError, match="dangling raw"):
        admit_workspace(*bundle)
    bundle = _bundle()
    bundle[4]["renamed"] = bundle[4].pop("theta")
    bundle[5]["renamed"] = bundle[5].pop("theta")
    with pytest.raises(ValueError, match="key or indexed identity"):
        admit_workspace(*bundle)


def test_parameter_dependency_cycle_refuses_before_raw_verification() -> None:
    """Reject a genuine parameter dependency cycle without forging hash cycles."""
    bundle = _bundle()
    payload = bundle[4]["theta"].to_dict()
    cast(dict[str, object], payload["body"])["dependency_keys"] = ["theta"]
    spec = parse_parameter_spec(payload)
    bundle[4]["theta"] = spec
    bundle[1][spec.digest] = spec
    bundle[3].clear()
    with pytest.raises(ValueError, match="dependency cycle"):
        admit_workspace(*bundle)


def test_raw_artifact_requires_immutable_bytes() -> None:
    """Refuse a mutable buffer at the source custody boundary."""
    with pytest.raises(ValueError, match="immutable bytes"):
        RawArtifact("review_fixture.v1", cast(bytes, bytearray(b"x")))


def _revision_bundle(payload: dict[str, object]) -> _Bundle:
    manifest, documents, raw, codecs, specs, units = _bundle()
    revision = parse_document(payload)
    documents = {
        key: value
        for key, value in documents.items()
        if not isinstance(value, (ExperimentRevision, LocalRunRecord))
    }
    documents[revision.digest] = revision
    wire = manifest.to_dict()
    cast(dict[str, object], wire["body"])["revision_refs"] = [
        {"schema": revision.schema, "sha256": revision.digest, "media_type": "application/json"}
    ]
    return parse_workspace_manifest(wire), documents, raw, codecs, specs, units


@pytest.mark.parametrize(
    "name,reason",
    [
        ("wrong_reference_schema", "schema or kind mismatch"),
        ("cross_project_parent", "cross-project revision"),
        ("revision_child", "dangling dependency"),
    ],
)
def test_shared_graph_rejection_vectors(name: str, reason: str) -> None:
    """Reject validly addressed documents whose cross-record bindings are invalid."""
    with pytest.raises(ValueError, match=reason):
        admit_workspace(*_revision_bundle(_FIXTURES[name]))


def test_parameter_specification_cannot_be_rebound_by_ambient_context() -> None:
    """Require immutable spec references inside the revision's own input_refs."""
    payload = copy.deepcopy(_FIXTURES["revision_root"])
    payload["body"]["input_refs"] = []
    with pytest.raises(ValueError, match="immutable specification reference"):
        admit_workspace(*_revision_bundle(payload))


def test_raw_kind_and_run_bindings() -> None:
    """Keep problem/program and revision/plan roles distinct even at valid hashes."""
    payload = copy.deepcopy(_FIXTURES["revision_root"])
    payload["body"]["problem_ref"] = payload["body"]["program_ref"]
    with pytest.raises(ValueError, match="schema or kind mismatch"):
        admit_workspace(*_revision_bundle(payload))
    bundle = _bundle()
    payload = copy.deepcopy(_FIXTURES["run"])
    payload["body"]["revision_hash"] = bundle[4]["theta"].digest
    run = parse_document(payload)
    for key, record in tuple(bundle[1].items()):
        if isinstance(record, LocalRunRecord):
            del bundle[1][key]
    bundle[1][run.digest] = run
    with pytest.raises(ValueError, match="missing revision"):
        admit_workspace(*bundle)


def test_native_producer_input_retains_its_own_codec_and_bytes() -> None:
    """Admit an existing real stable-core input using its public native verifier."""
    from scpn_quantum_control.stable_core_product import (
        digest_stable_core_payload,
        unwrap_model_envelope,
    )

    source = (
        Path(__file__).parent / "data/contract_custody_corpus/raw_round_trip_preserves_digest.json"
    )
    original = source.read_bytes()

    def native_codec(content: bytes) -> RawIdentity:
        payload = cast(dict[str, object], read_json(content.decode()))
        schema, kind, _ = unwrap_model_envelope(payload)
        return RawIdentity(schema, kind, digest_stable_core_payload(payload))

    identity = native_codec(original)
    payload = copy.deepcopy(_FIXTURES["revision_root"])
    payload["body"]["input_refs"].append(
        {"schema": identity.schema, "sha256": identity.digest, "media_type": "application/json"}
    )
    bundle = _revision_bundle(payload)
    bundle[2][identity.digest] = RawArtifact(identity.schema, original)
    bundle[3][identity.schema] = native_codec
    receipt = admit_workspace(*bundle)
    assert identity.digest in receipt.raw_hashes
    assert bundle[2][identity.digest].content == original == source.read_bytes()


def test_plan_role_and_missing_parameter_dependency() -> None:
    """Reject a non-plan producer and unresolved specification prerequisites."""
    bundle = _bundle()
    payload = copy.deepcopy(_FIXTURES["run"])
    payload["body"]["plan_hash"] = canonical_digest("review_fixture.v1", _FIXTURES["problem"])
    for key, record in tuple(bundle[1].items()):
        if isinstance(record, LocalRunRecord):
            del bundle[1][key]
    run = parse_document(payload)
    bundle[1][run.digest] = run
    with pytest.raises(ValueError, match="wrong raw kind"):
        admit_workspace(*bundle)
    bundle = _bundle()
    payload = bundle[4]["theta"].to_dict()
    cast(dict[str, object], payload["body"])["dependency_keys"] = ["missing"]
    spec = parse_parameter_spec(payload)
    bundle[1][spec.digest] = spec
    bundle[4]["theta"] = spec
    with pytest.raises(ValueError, match="dangling dependency"):
        admit_workspace(*bundle)


def test_ambiguous_identity_and_unexpected_workspace() -> None:
    """Refuse overlapping source identities and additional workspace roots."""
    bundle = _bundle()
    bundle[2][next(iter(bundle[1]))] = next(iter(bundle[2].values()))
    with pytest.raises(ValueError, match="ambiguous raw/document"):
        admit_workspace(*bundle)
    bundle = _bundle()
    payload = bundle[0].to_dict()
    cast(dict[str, object], payload["extensions"])["title"] = "Another root"
    root = parse_workspace_manifest(payload)
    bundle[1][root.digest] = root
    with pytest.raises(ValueError, match="unexpected workspace root"):
        admit_workspace(*bundle)


def test_raw_verifier_cannot_bypass_workspace_document_validation() -> None:
    """Even a truthful raw digest cannot replace an indexed workspace document."""
    bundle = _bundle()
    for key, document in tuple(bundle[1].items()):
        if isinstance(document, ExperimentRevision):
            bundle[2][key] = RawArtifact(document.schema, write_json(document.to_dict()).encode())
            del bundle[1][key]
        elif isinstance(document, LocalRunRecord):
            del bundle[1][key]

    def workspace_codec(content: bytes) -> RawIdentity:
        document = parse_document(read_json(content.decode()))
        return RawIdentity(document.schema, document.schema, document.digest)

    bundle[3][ExperimentRevision.schema] = workspace_codec
    with pytest.raises(ValueError, match="workspace reference requires indexed document"):
        admit_workspace(*bundle)


def test_selected_draft_and_merging_revision_admit_complete_graph() -> None:
    """A merge waits for both genuine parents and retains its selected draft identity."""
    bundle = _bundle()
    parent = next(
        record
        for record in bundle[1].values()
        if isinstance(record, ExperimentRevision) and not record.body["parent_revision_hashes"]
    )
    sibling_wire = parent.to_dict()
    cast(dict[str, object], sibling_wire["extensions"])["branch"] = "independent"
    sibling = parse_document(sibling_wire)
    merge_wire = copy.deepcopy(_FIXTURES["revision_child"])
    merge_wire["body"]["parent_revision_hashes"] = [parent.digest, sibling.digest]
    merged = parse_document(merge_wire)
    for key, record in tuple(bundle[1].items()):
        if isinstance(record, (ExperimentRevision, LocalRunRecord)) and key != parent.digest:
            del bundle[1][key]
    bundle[1][sibling.digest] = sibling
    bundle[1][merged.digest] = merged
    refs = [
        {"schema": record.schema, "sha256": record.digest, "media_type": "application/json"}
        for record in (parent, sibling, merged)
    ]
    manifest_wire = bundle[0].to_dict()
    body = cast(dict[str, object], manifest_wire["body"])
    body["revision_refs"] = refs
    body["draft_ref"] = refs[-1]
    manifest = parse_workspace_manifest(manifest_wire)
    admission = admit_workspace(manifest, *bundle[1:])
    assert merged.digest in admission.document_hashes
    assert sibling.digest in admission.document_hashes


def test_parameter_binding_requires_its_immutable_input_reference() -> None:
    """Indexed context alone cannot supply a specification omitted by the revision."""
    payload = copy.deepcopy(_FIXTURES["revision_root"])
    payload["body"]["input_refs"] = []
    with pytest.raises(ValueError, match="missing immutable specification reference"):
        admit_workspace(*_revision_bundle(payload))


def test_run_cannot_outlive_its_indexed_revision() -> None:
    """Retaining a run receipt requires retaining the exact revision it references."""
    bundle = _bundle()
    child_hash = next(
        key
        for key, record in bundle[1].items()
        if isinstance(record, ExperimentRevision) and record.body["parent_revision_hashes"]
    )
    del bundle[1][child_hash]
    manifest_wire = bundle[0].to_dict()
    body = cast(dict[str, object], manifest_wire["body"])
    refs = cast(list[dict[str, object]], body["revision_refs"])
    body["revision_refs"] = [ref for ref in refs if ref["sha256"] != child_hash]
    manifest = parse_workspace_manifest(manifest_wire)
    with pytest.raises(ValueError, match="missing revision"):
        admit_workspace(manifest, *bundle[1:])


def test_run_outputs_resolve_original_producer_bytes() -> None:
    """An output receipt must resolve its original source before admission succeeds."""
    bundle = _bundle()
    payload = copy.deepcopy(_FIXTURES["run"])
    digest = canonical_digest("review_fixture.v1", _FIXTURES["problem"])
    payload["body"]["output_refs"] = [
        {"schema": "review_fixture.v1", "sha256": digest, "media_type": "application/json"}
    ]
    for key, record in tuple(bundle[1].items()):
        if isinstance(record, LocalRunRecord):
            del bundle[1][key]
    run = parse_document(payload)
    bundle[1][run.digest] = run
    assert digest in admit_workspace(*bundle).raw_hashes
    del bundle[2][digest]
    with pytest.raises(ValueError, match="dangling raw"):
        admit_workspace(*bundle)
