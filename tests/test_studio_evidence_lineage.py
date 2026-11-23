# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — evidence lineage tests
"""Tests for the evidence lineage of claims, hardware result packs and replications."""

from __future__ import annotations

import importlib.util
import json
import runpy
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from scpn_quantum_control.studio.evidence_lineage import (
    ABSENT_CERTIFICATION_REASON,
    EVIDENCE_LINEAGE_SCHEMA,
    LINEAGE_RECORD_SCHEMA,
    ReplicationSource,
    artifact_index,
    build_evidence_lineage,
    portable_bundle,
    sealed_record,
)
from scpn_quantum_control.studio_workspace.canonical import canonical_digest

_ROOT = Path(__file__).resolve().parents[1]
_LEDGER = "data/differentiable_phase_qnode/claim_ledger.json"
_REGISTRY = "data/differentiable_phase_qnode/reference_validation_certifications.json"
_MANIFEST = "data/hardware_result_packs/manifest.json"
_REPLICATION = "data/iqm_paper_replication/iqm_dla_d10_retest_primary_analysis_2026-07-22.json"

Record = dict[str, Any]


def _tool() -> ModuleType:
    """Load the committed-data tool from its file.

    Returns
    -------
    ModuleType
        The loaded ``tools/studio_evidence_lineage.py`` module.

    """
    spec = importlib.util.spec_from_file_location(
        "studio_evidence_lineage_for_tests", _ROOT / "tools/studio_evidence_lineage.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TOOL = _tool()


def _records(lineage: dict[str, object], key: str) -> list[Record]:
    """Return one part of a lineage as typed records.

    Parameters
    ----------
    lineage
        A built lineage.
    key
        ``"claims"``, ``"packs"`` or ``"replications"``.

    Returns
    -------
    list[Record]
        The records of that part.

    """
    return cast(list[Record], lineage[key])


def _repository_copy(tmp_path: Path) -> Path:
    """Copy the sources the lineage reads, with every pack artefact, into a new root.

    Parameters
    ----------
    tmp_path
        Directory that receives the copy.

    Returns
    -------
    Path
        The new repository root.

    """
    root = tmp_path / "repo"
    manifest = json.loads((_ROOT / _MANIFEST).read_text(encoding="utf-8"))
    paths = {_LEDGER, _REGISTRY, _MANIFEST, _REPLICATION}
    paths.update(item["path"] for pack in manifest["packs"] for item in pack["artifacts"])
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / relative, target)
    return root


def _rewrite(path: Path, change: Callable[[Record], None]) -> None:
    """Load a JSON object, apply ``change`` to it and write it back.

    Parameters
    ----------
    path
        File to rewrite.
    change
        Edits the parsed document in place.

    """
    document = cast(Record, json.loads(path.read_text(encoding="utf-8")))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_the_repository_lineage_keeps_every_status_verbatim() -> None:
    """Claims stay unvalidated, the negative pack keeps both statuses, the replication fails."""
    lineage = build_evidence_lineage(_ROOT)
    claims = {claim["claim_id"]: claim for claim in _records(lineage, "claims")}
    packs = {pack["id"]: pack for pack in _records(lineage, "packs")}

    assert lineage["schema"] == EVIDENCE_LINEAGE_SCHEMA
    assert len(claims) == 16
    for claim in claims.values():
        assert claim["certification"] == {"state": "absent", "reason": ABSENT_CERTIFICATION_REASON}
        assert claim["bundle"]["claim_boundary"]["status"] != "reference-validated"
        assert claim["bundle"]["freshness"] == "traceable-unchecked"
    negative = packs["scpn_fim_negative_ibm_kingston_2026_05_05"]
    assert negative["manifest_status"] == "promoted_negative_result"
    assert negative["bundle"]["claim_boundary"]["status"] == "bounded-support"
    custody = {item["custody"] for pack in packs.values() for item in pack["artefacts"]}
    assert custody == {"digest-matches"}
    (replication,) = _records(lineage, "replications")
    assert replication["decision"] == {"field": "negative_sign_replicates", "value": False}
    assert replication["alpha"] == 0.05


def test_evidence_identifiers_resolve_without_choosing_a_candidate() -> None:
    """One file resolves, several stay ambiguous with every candidate, none is missing."""
    lineage = build_evidence_lineage(_ROOT)
    states = {
        item["artifact_id"]: item
        for claim in _records(lineage, "claims")
        for item in cast(list[Record], claim["evidence"])
    }

    resolved = states["diff-baseline-scorecard-20260620"]
    assert resolved["state"] == "resolved"
    assert [item["path"] for item in resolved["candidates"]] == [
        "data/differentiable_phase_qnode/differentiable_baseline_scorecard_20260620.json"
    ]
    assert states["diff-qnode-ci-evidence-schema-v1"]["state"] == "ambiguous"
    assert len(states["diff-qnode-ci-evidence-schema-v1"]["candidates"]) == 4
    missing = states["diff-hardening-slice-gate-v1"]
    assert (missing["state"], missing["candidates"]) == ("missing", [])


def test_portable_bundles_drop_only_the_emitting_host() -> None:
    """Host, Studio version and Python version go; every verdict axis stays."""
    bundle: Record = {
        "prov": {
            "entity": {"id": "e", "digest": "d"},
            "activity": {"host": "h", "started": "s"},
            "agent": {"studio_version": "v", "operator": "o"},
        },
        "recompute_environment": {"toolchain": {"python": "3", "command": "c"}},
        "claim_boundary": {"status": "bounded-model"},
    }
    assert portable_bundle(bundle) == {
        "prov": {
            "entity": {"id": "e", "digest": "d"},
            "activity": {"started": "s"},
            "agent": {"operator": "o"},
        },
        "recompute_environment": {"toolchain": {"command": "c"}},
        "claim_boundary": {"status": "bounded-model"},
    }
    assert portable_bundle({"prov": "opaque"}) == {"prov": "opaque"}
    assert bundle["prov"]["activity"] == {"host": "h", "started": "s"}


def test_a_certified_promoted_claim_reaches_reference_validated(tmp_path: Path) -> None:
    """A valid certification of a promoted claim is reported with its record and bundle.

    Parameters
    ----------
    tmp_path
        Directory that receives the repository copy.

    """
    root = _repository_copy(tmp_path)
    claim_id = json.loads((root / _LEDGER).read_text(encoding="utf-8"))["claims"][0]["claim_id"]
    _rewrite(root / _LEDGER, lambda doc: doc["claims"][0].update(promotion_status="promoted"))
    certificate = {
        "claim_id": claim_id,
        "status": "reference-validated",
        "certificate_ref": "fixture-certificate",
        "reference_artifact_digest": "sha256:" + "a" * 64,
        "adjudicated_at": "2026-11-23T00:00:00Z",
    }
    _rewrite(root / _REGISTRY, lambda doc: doc["certifications"].append(certificate))

    first = _records(build_evidence_lineage(root), "claims")[0]

    assert first["certification"] == {"state": "certified", "record": certificate}
    assert first["bundle"]["claim_boundary"]["status"] == "reference-validated"


def test_an_invalid_registry_refuses_the_build(tmp_path: Path) -> None:
    """A certification of an unknown claim fails closed instead of laundering a status.

    Parameters
    ----------
    tmp_path
        Directory that receives the repository copy.

    """
    root = _repository_copy(tmp_path)
    certificate = {
        "claim_id": "no-such-claim",
        "certificate_ref": "fixture-certificate",
        "reference_artifact_digest": "sha256:" + "b" * 64,
        "adjudicated_at": "2026-11-23T00:00:00Z",
    }
    _rewrite(root / _REGISTRY, lambda doc: doc["certifications"].append(certificate))

    with pytest.raises(ValueError, match="absent from ledger"):
        build_evidence_lineage(root)


def test_pack_custody_reports_a_changed_or_missing_artefact(tmp_path: Path) -> None:
    """An edited artefact differs from its recorded digest; a deleted one is missing.

    Parameters
    ----------
    tmp_path
        Directory that receives the repository copy.

    """
    root = _repository_copy(tmp_path)
    manifest = json.loads((root / _MANIFEST).read_text(encoding="utf-8"))
    first, second = (item["path"] for item in manifest["packs"][0]["artifacts"][:2])
    (root / first).write_text("{}", encoding="utf-8")
    (root / second).unlink()

    pack = _records(build_evidence_lineage(root), "packs")[0]
    custody = {item["path"]: item["custody"] for item in pack["artefacts"]}

    assert custody[first] == "digest-differs"
    assert custody[second] == "missing"


@pytest.mark.parametrize(
    ("section", "decision"),
    [("absent_section", "negative_sign_replicates"), ("primary_sign_replication", "alpha")],
)
def test_a_replication_without_its_boolean_decision_is_refused(
    tmp_path: Path, section: str, decision: str
) -> None:
    """A declared analysis must hold its record and a boolean decision.

    Parameters
    ----------
    tmp_path
        Directory that receives the repository copy.
    section
        Record key the declaration names.
    decision
        Decision key the declaration names.

    """
    root = _repository_copy(tmp_path)
    source = ReplicationSource(_REPLICATION, section, decision)

    with pytest.raises(ValueError, match="must be a boolean"):
        build_evidence_lineage(root, replication_sources=(source,))


def test_a_declared_replication_must_be_a_json_object(tmp_path: Path) -> None:
    """A replication file that is not a JSON object refuses the build.

    Parameters
    ----------
    tmp_path
        Directory that receives the repository copy.

    """
    root = _repository_copy(tmp_path)
    (root / _REPLICATION).write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON object"):
        build_evidence_lineage(root)


def test_the_artefact_index_skips_unreadable_and_unidentified_files(tmp_path: Path) -> None:
    """Invalid JSON, arrays and records without an identifier are not indexed.

    Parameters
    ----------
    tmp_path
        Repository root that receives the data files.

    """
    data = tmp_path / "data"
    data.mkdir()
    (data / "broken.json").write_text("{", encoding="utf-8")
    (data / "array.json").write_text("[]", encoding="utf-8")
    (data / "plain.json").write_text('{"artifact_id": ""}', encoding="utf-8")
    (data / "named.json").write_text('{"artifact_id": "x"}', encoding="utf-8")

    index = artifact_index(tmp_path)

    assert list(index) == ["x"]
    assert index["x"][0]["path"] == "data/named.json"


def test_every_record_carries_the_digest_of_its_own_content() -> None:
    """Each record's digest recomputes from the record without it, and changes with it."""
    lineage = build_evidence_lineage(_ROOT)
    for key in ("claims", "packs", "replications"):
        for record in _records(lineage, key):
            body = {name: value for name, value in record.items() if name != "record_digest"}
            assert record["record_digest"] == canonical_digest(LINEAGE_RECORD_SCHEMA, body)
    assert sealed_record({"a": 1})["record_digest"] != sealed_record({"a": 2})["record_digest"]


def test_the_committed_lineage_is_current() -> None:
    """The committed file equals a fresh build."""
    assert _TOOL.main(["--check"]) == 0
    committed = json.loads(_TOOL.DEFAULT_PATH.read_text(encoding="utf-8"))
    assert committed["schema"] == EVIDENCE_LINEAGE_SCHEMA


def test_the_script_entry_point_checks_the_committed_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the tool as a script exits with the check status.

    Parameters
    ----------
    monkeypatch
        Replaces the process arguments.

    """
    monkeypatch.setattr(sys, "argv", ["studio_evidence_lineage.py", "--check"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(_ROOT / "tools/studio_evidence_lineage.py"), run_name="__main__")
    assert stopped.value.code == 0
