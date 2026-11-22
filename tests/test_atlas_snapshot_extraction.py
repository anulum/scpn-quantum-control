# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Atlas snapshot extraction tests
"""Tests for deterministic Atlas snapshot extraction from committed artefacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.atlas import atlas_snapshot_digest
from scpn_quantum_control.atlas.extractors import (
    DEFAULT_INPUTS,
    AtlasExtractionError,
    AtlasInput,
    directory_reader,
    extract_atlas_snapshot,
)

_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_MAP = "data/map.json"
_LEDGER = "data/ledger.json"
_PACKS = "data/packs.json"
_INPUTS = (
    AtlasInput(_MAP, "architecture_map", True),
    AtlasInput(_LEDGER, "claim_ledger", False),
    AtlasInput(_PACKS, "hardware_result_packs", False),
)


def _layer(layer_id: str, **overrides: Any) -> dict[str, Any]:
    """Return one synthetic architecture-map layer.

    Parameters
    ----------
    layer_id : str
        Layer identifier.
    **overrides : Any
        Fields that replace the defaults.

    Returns
    -------
    dict[str, Any]
        The layer record.

    """
    layer: dict[str, Any] = {
        "layer_id": layer_id,
        "title": f"Layer {layer_id}",
        "role": "Synthetic fixture layer.",
        "claim_boundary": "Fixture only.",
        "rustification_ready": False,
        "blockers": ["fixture blocker"],
        "owner_modules": ["src/pkg/core.py"],
        "python_surfaces": ["src/pkg/core.py", "src/pkg/extra.py"],
        "rust_surfaces": ["engine/lib.rs"],
        "polyglot_surfaces": ["docs/guide.md"],
        "test_surfaces": ["tests/test_core.py"],
        "docs_surfaces": ["docs/guide.md"],
        "inventory_surface_ids": ["inventory_id"],
        "benchmark_surfaces": ["benchmark_id"],
    }
    layer.update(overrides)
    return layer


def _write_tree(
    root: Path, layers: list[dict[str, Any]], *, ledger: bool = True, packs: bool = True
) -> None:
    """Write a synthetic source revision below ``root``.

    Parameters
    ----------
    root : Path
        Directory of the revision.
    layers : list[dict[str, Any]]
        Architecture-map layers.
    ledger : bool
        Write the claim ledger.
    packs : bool
        Write the result-pack manifest.

    """
    files = {
        "src/pkg/core.py": "def core():\n    return 1\n",
        "src/pkg/extra.py": "def extra():\n    return 2\n",
        "engine/lib.rs": "pub fn core() -> i32 { 1 }\n",
        "docs/guide.md": "# Guide\n",
        "tests/test_core.py": "def test_core():\n    assert True\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    document = {
        "schema": "scpn_qc_differentiable_architecture_map_v1",
        "artifact_id": "fixture-map",
        "claim_boundary": "Fixture map only.",
        "layers": layers,
    }
    (root / "data").mkdir(exist_ok=True)
    (root / _MAP).write_text(json.dumps(document), encoding="utf-8")
    if ledger:
        claims = {
            "schema": "fixture_claim_ledger_v1",
            "artifact_id": "fixture-ledger",
            "claims": [
                {
                    "claim_id": "core_returns_one",
                    "claim_text": "core() returns one.",
                    "claim_boundary": "Fixture claim.",
                    "implementation_surface": ["src/pkg/core.py", "docs/guide.md"],
                    "docs_surface": ["docs/guide.md"],
                    "evidence_artifact_ids": ["evidence-id"],
                    "benchmark_artifact_ids": [],
                }
            ],
        }
        (root / _LEDGER).write_text(json.dumps(claims), encoding="utf-8")
    if packs:
        manifest = {
            "claim_boundary": "Fixture packs.",
            "packs": [
                {
                    "id": "fixture_pack",
                    "title": "Fixture pack",
                    "backend": "fixture_backend",
                    "status": "fixture",
                    "shots": 1024,
                }
            ],
        }
        (root / _PACKS).write_text(json.dumps(manifest), encoding="utf-8")


def _extract(root: Path, **kwargs: Any) -> Any:
    """Extract the synthetic revision below ``root``.

    Parameters
    ----------
    root : Path
        Directory of the revision.
    **kwargs : Any
        Keyword arguments for :func:`extract_atlas_snapshot`.

    Returns
    -------
    Any
        The extraction result.

    """
    kwargs.setdefault("inputs", _INPUTS)
    return extract_atlas_snapshot(directory_reader(root), _COMMIT, **kwargs)


def test_atlas_snapshot_extraction_01_reordered_discovery_yields_the_same_snapshot(
    tmp_path: Path,
) -> None:
    """Discovery order never changes the snapshot; record order changes only the input's own digest."""
    first, second = tmp_path / "a", tmp_path / "b"
    _write_tree(first, [_layer("alpha"), _layer("beta", python_surfaces=["src/pkg/extra.py"])])
    _write_tree(second, [_layer("beta", python_surfaces=["src/pkg/extra.py"]), _layer("alpha")])

    one = _extract(first).snapshot
    reversed_inputs = _extract(first, inputs=tuple(reversed(_INPUTS))).snapshot
    reordered_records = _extract(second).snapshot

    assert atlas_snapshot_digest(one) == atlas_snapshot_digest(reversed_inputs)
    derived = {node.id for node in one.nodes if node.kind != "artefact" or node.source_ref != _MAP}
    assert derived == {
        node.id
        for node in reordered_records.nodes
        if node.kind != "artefact" or node.source_ref != _MAP
    }
    assert {(edge.source, edge.kind, edge.target) for edge in one.edges} == {
        (edge.source, edge.kind, edge.target) for edge in reordered_records.edges
    }


def test_atlas_snapshot_extraction_02_changing_evidence_changes_its_provenance(
    tmp_path: Path,
) -> None:
    """A changed input or source file changes the digests that record where a node came from."""
    _write_tree(tmp_path, [_layer("alpha")])
    before = _extract(tmp_path).snapshot
    (tmp_path / "src/pkg/core.py").write_text("def core():\n    return 3\n", encoding="utf-8")
    after_source = _extract(tmp_path).snapshot
    document = json.loads((tmp_path / _MAP).read_text())
    document["claim_boundary"] = "Changed boundary."
    (tmp_path / _MAP).write_text(json.dumps(document), encoding="utf-8")
    after_map = _extract(tmp_path).snapshot

    core = "definition:repo/src:pkg:core.py"
    assert before.node(core).source_digest != after_source.node(core).source_digest
    capability = "capability:atlas.architecture/alpha"
    assert before.node(capability).evidence_refs != after_map.node(capability).evidence_refs
    implements = [
        edge for edge in after_map.edges if edge.source == core and edge.kind == "implements"
    ]
    assert implements[0].evidence_digest == after_map.node(capability).evidence_refs[0]
    assert before.evidence_set_hash != after_map.evidence_set_hash


def test_atlas_snapshot_extraction_03_incremental_change_equals_full_rebuild(
    tmp_path: Path,
) -> None:
    """Reusing unchanged fragments after an add, a change and a delete equals a full rebuild."""
    _write_tree(tmp_path, [_layer("alpha"), _layer("beta")])
    base = _extract(tmp_path)
    (tmp_path / "src/pkg/extra.py").write_text("def extra():\n    return 5\n", encoding="utf-8")
    (tmp_path / "src/pkg/new.py").write_text("def new():\n    return 6\n", encoding="utf-8")
    document = json.loads((tmp_path / _MAP).read_text())
    document["layers"] = [_layer("alpha", python_surfaces=["src/pkg/core.py", "src/pkg/new.py"])]
    (tmp_path / _MAP).write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / _PACKS).unlink()
    when = "2026-11-22T00:00:00Z"

    incremental = _extract(tmp_path, previous=base.snapshot, removed_at=when, reuse=base.fragments)
    full = _extract(tmp_path, previous=base.snapshot, removed_at=when)

    assert incremental.snapshot == full.snapshot
    assert incremental.fragments[_LEDGER] is base.fragments[_LEDGER]
    assert incremental.fragments[_MAP] is not base.fragments[_MAP]


def test_atlas_snapshot_extraction_04_removed_node_yields_tombstone(tmp_path: Path) -> None:
    """A node that is no longer extracted leaves a tombstone with its exact identity and last digest."""
    _write_tree(tmp_path, [_layer("alpha"), _layer("beta")])
    base = _extract(tmp_path).snapshot
    document = json.loads((tmp_path / _MAP).read_text())
    document["layers"] = [_layer("alpha")]
    (tmp_path / _MAP).write_text(json.dumps(document), encoding="utf-8")

    later = _extract(tmp_path, previous=base, removed_at="2026-11-22T00:00:00Z").snapshot
    again = _extract(tmp_path, previous=later, removed_at="2026-11-23T00:00:00Z").snapshot

    gone = {tombstone.id: tombstone for tombstone in later.tombstones}
    assert "capability:atlas.architecture/beta" in gone
    assert gone["capability:atlas.architecture/beta"].kind == "capability"
    assert {tombstone.id for tombstone in again.tombstones} >= set(gone)
    with pytest.raises(AtlasExtractionError) as missing_time:
        _extract(tmp_path, previous=base)
    assert missing_time.value.code == "missing_removal_time"


@pytest.mark.parametrize(
    ("mutate", "code", "where"),
    [
        (
            lambda root: (root / _MAP).write_text("{not json", encoding="utf-8"),
            "malformed_input",
            _MAP,
        ),
        (lambda root: (root / _MAP).write_text("[]", encoding="utf-8"), "malformed_input", _MAP),
        (lambda root: (root / _MAP).write_bytes(b"\xff\xfe"), "malformed_input", _MAP),
        (lambda root: (root / _MAP).unlink(), "missing_required_input", _MAP),
        (
            lambda root: (root / _LEDGER).write_text('{"claims": "x"}', encoding="utf-8"),
            "malformed_input",
            _LEDGER,
        ),
    ],
    ids=["invalid-json", "not-an-object", "not-utf8", "required-absent", "malformed-optional"],
)
def test_atlas_snapshot_extraction_05_malformed_required_input_refuses_whole_extraction(
    tmp_path: Path, mutate: Any, code: str, where: str
) -> None:
    """A malformed or absent required input, or a malformed optional one, refuses everything."""
    _write_tree(tmp_path, [_layer("alpha")])
    mutate(tmp_path)

    with pytest.raises(AtlasExtractionError) as refusal:
        _extract(tmp_path)

    assert (refusal.value.code, refusal.value.path) == (code, where)
    assert str(refusal.value).startswith(code)


@pytest.mark.parametrize(
    ("layer_fields", "code"),
    [
        ({"schema_override": "other_schema"}, "unsupported_input_schema"),
        ({"layer_id": "bad id"}, "malformed_input"),
        ({"title": ""}, "malformed_input"),
        ({"blockers": "not a list"}, "malformed_input"),
        ({"test_surfaces": [1]}, "malformed_input"),
    ],
    ids=["schema", "identity", "empty-title", "blockers-type", "path-type"],
)
def test_malformed_architecture_records_refuse(
    tmp_path: Path, layer_fields: dict[str, Any], code: str
) -> None:
    """Every malformed map field refuses the extraction instead of producing a partial graph."""
    schema = layer_fields.pop("schema_override", None)
    layer = _layer("alpha")
    layer.update(layer_fields)
    _write_tree(tmp_path, [layer])
    if schema is not None:
        document = json.loads((tmp_path / _MAP).read_text())
        document["schema"] = schema
        (tmp_path / _MAP).write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AtlasExtractionError) as refusal:
        _extract(tmp_path)

    assert refusal.value.code == code


def test_unresolvable_names_become_scoped_exclusions_not_nodes(tmp_path: Path) -> None:
    """Identifiers, absent files, unknown file types and unsafe paths are excluded, never invented."""
    _write_tree(
        tmp_path,
        [
            _layer(
                "alpha",
                python_surfaces=[
                    "src/pkg/core.py",
                    "src/pkg/missing.py",
                    "src/pkg/data.bin",
                    "../escape.py",
                ],
            )
        ],
        packs=False,
    )
    snapshot = _extract(tmp_path).snapshot
    reasons = {exclusion.source_ref: exclusion.reason for exclusion in snapshot.exclusions}

    assert (
        reasons[f"{_MAP}#alpha:src/pkg/missing.py"]
        == "named by the input but absent from the source"
    )
    assert reasons[f"{_MAP}#alpha:src/pkg/data.bin"] == "file type maps to no admitted node kind"
    assert reasons[f"{_MAP}#alpha:../escape.py"] == "not a safe repository-relative path"
    assert reasons[f"{_MAP}#alpha:inventory_id"].startswith("surface identifier")
    assert reasons[f"{_LEDGER}#core_returns_one:evidence-id"].startswith(
        "evidence artefact identifier"
    )
    assert reasons[_PACKS] == "optional input absent from the source revision"
    ids = {node.id for node in snapshot.nodes}
    assert not any("missing" in node_id or "escape" in node_id for node_id in ids)


def test_claims_and_packs_map_to_their_nodes_and_relations(tmp_path: Path) -> None:
    """Claims derive from implementations and are documented; packs are content-addressed artefacts."""
    _write_tree(tmp_path, [_layer("alpha")])
    snapshot = _extract(tmp_path).snapshot
    claim = "claim:differentiable.claim_ledger/core_returns_one"
    relations = {(edge.source, edge.kind, edge.target) for edge in snapshot.edges}

    assert (claim, "derives_from", "definition:repo/src:pkg:core.py") in relations
    assert ("documentation:repo/docs:guide.md", "documents", claim) in relations
    assert (
        "capability:atlas.architecture/alpha",
        "contains",
        "test:repo/tests:test_core.py",
    ) in relations
    assert (
        "definition:repo/engine:lib.rs",
        "implements",
        "capability:atlas.architecture/alpha",
    ) in relations
    packs = [node for node in snapshot.nodes if node.source_ref == f"{_PACKS}#fixture_pack"]
    assert len(packs) == 1
    assert dict(packs[0].validity_domain or ()) == {
        "backend": "fixture_backend",
        "status": "fixture",
    }
    assert all(edge.kind != "supports" for edge in snapshot.edges)


def test_invalid_input_lists_are_refused(tmp_path: Path) -> None:
    """Empty, duplicated, unknown or unsafe inputs refuse before anything is read."""
    _write_tree(tmp_path, [_layer("alpha")])
    cases = [
        ((), "invalid_inputs"),
        ((_INPUTS[0], _INPUTS[0]), "invalid_inputs"),
        ((AtlasInput(_MAP, "guesswork", True),), "unknown_adapter"),
        ((AtlasInput("../outside.json", "architecture_map", True),), "invalid_inputs"),
    ]
    for inputs, code in cases:
        with pytest.raises(AtlasExtractionError) as refusal:
            _extract(tmp_path, inputs=inputs)
        assert refusal.value.code == code


def test_directory_reader_refuses_symlinks_and_escapes(tmp_path: Path) -> None:
    """The reader returns only regular files inside its root."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "file.txt").write_text("inside", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    (root / "link.txt").symlink_to(tmp_path / "outside.txt")
    read = directory_reader(root)

    assert read("file.txt") == b"inside"
    assert read("link.txt") is None
    assert read("../outside.txt") is None
    assert read("absent.txt") is None


def test_the_committed_repository_extracts_deterministically() -> None:
    """The admitted committed inputs extract without heavy imports and reproduce the same digest."""
    root = Path(__file__).resolve().parents[1]
    first = extract_atlas_snapshot(directory_reader(root), _COMMIT, inputs=DEFAULT_INPUTS).snapshot
    second = extract_atlas_snapshot(
        directory_reader(root), _COMMIT, inputs=DEFAULT_INPUTS
    ).snapshot

    assert atlas_snapshot_digest(first) == atlas_snapshot_digest(second)
    assert {node.kind for node in first.nodes} >= {"capability", "definition", "test", "artefact"}
    assert first.access_scope == "public"


def test_missing_named_files_in_every_role_are_excluded(tmp_path: Path) -> None:
    """Absent test, documentation and claim files are excluded in every role that names them."""
    _write_tree(
        tmp_path,
        [_layer("alpha", test_surfaces=["tests/test_gone.py"], docs_surfaces=["docs/gone.md"])],
    )
    ledger = json.loads((tmp_path / _LEDGER).read_text())
    ledger["claims"][0]["implementation_surface"] = ["src/pkg/gone.py"]
    ledger["claims"][0]["docs_surface"] = ["docs/gone.md"]
    (tmp_path / _LEDGER).write_text(json.dumps(ledger), encoding="utf-8")
    snapshot = _extract(tmp_path).snapshot
    excluded = {exclusion.source_ref for exclusion in snapshot.exclusions}

    assert {
        f"{_MAP}#alpha:tests/test_gone.py",
        f"{_MAP}#alpha:docs/gone.md",
        f"{_LEDGER}#core_returns_one:src/pkg/gone.py",
        f"{_LEDGER}#core_returns_one:docs/gone.md",
    } <= excluded
    claim = "claim:differentiable.claim_ledger/core_returns_one"
    assert not [edge for edge in snapshot.edges if claim in (edge.source, edge.target)]


def test_layers_that_are_not_records_refuse(tmp_path: Path) -> None:
    """A map whose layers are not an array of objects refuses the extraction."""
    _write_tree(tmp_path, [_layer("alpha")])
    document = json.loads((tmp_path / _MAP).read_text())
    document["layers"] = "not records"
    (tmp_path / _MAP).write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AtlasExtractionError) as refusal:
        _extract(tmp_path)

    assert (refusal.value.code, refusal.value.path) == ("malformed_input", _MAP)


def test_a_fragment_is_rebuilt_when_a_file_it_read_changes(tmp_path: Path) -> None:
    """An unchanged input whose named file changed is re-extracted, and still equals a full rebuild."""
    _write_tree(tmp_path, [_layer("alpha")])
    base = _extract(tmp_path)
    (tmp_path / "tests/test_core.py").write_text(
        "def test_core():\n    assert 1\n", encoding="utf-8"
    )

    incremental = _extract(tmp_path, reuse=base.fragments)

    assert incremental.fragments[_MAP] is not base.fragments[_MAP]
    assert incremental.fragments[_PACKS] is base.fragments[_PACKS]
    assert incremental.snapshot == _extract(tmp_path).snapshot


def test_extraction_imports_no_optional_runtime() -> None:
    """Extracting the committed inputs in a fresh interpreter loads no numerical or quantum runtime."""
    root = Path(__file__).resolve().parents[1]
    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "from scpn_quantum_control.atlas.extractors import directory_reader, extract_atlas_snapshot\n"
        "extract_atlas_snapshot(directory_reader(Path(sys.argv[1])), '0' * 40)\n"
        "names = ('numpy', 'scipy', 'qiskit', 'torch', 'jax', 'pennylane', 'tensorflow', 'scpn_quantum_engine')\n"
        "print(','.join(name for name in names if name in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(root)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env={"PYTHONPATH": f"{root / 'src'}", "PATH": "/usr/bin:/bin"},
    )

    assert completed.stdout.strip() == ""
