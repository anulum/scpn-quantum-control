# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- hardware result-pack verifier tests
"""Regression tests for offline hardware result-pack verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.hardware_result_packs import sha256, verify_manifest


def write_pack_fixture(tmp_path: Path, *, job_id: str = "ibm-run-test") -> Path:
    """Create a minimal result-pack fixture and return its manifest path."""

    artifact_path = tmp_path / "data" / "pack" / "raw.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps({"job_ids": [job_id], "counts": {"0": 16}}), encoding="utf-8"
    )
    manifest_path = tmp_path / "data" / "hardware_result_packs" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    rel_artifact = artifact_path.relative_to(tmp_path).as_posix()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": "test/repo",
                "packs": [
                    {
                        "id": "fixture_pack",
                        "status": "promoted_raw_count_evidence",
                        "required_job_ids": [job_id],
                        "artifacts": [
                            {
                                "path": rel_artifact,
                                "role": "raw_counts",
                                "sha256": sha256(artifact_path),
                                "bytes": artifact_path.stat().st_size,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_verify_manifest_accepts_complete_pack(tmp_path: Path) -> None:
    """A complete pack verifies and reports stable counts."""

    manifest_path = write_pack_fixture(tmp_path)

    summary = verify_manifest(manifest_path, repo_root=tmp_path)

    assert summary["pack_count"] == 1
    assert summary["artifact_count"] == 1
    assert summary["packs"] == [
        {
            "id": "fixture_pack",
            "artifact_count": 1,
            "job_id_count": 1,
            "status": "promoted_raw_count_evidence",
        }
    ]


def test_verify_manifest_rejects_digest_mismatch(tmp_path: Path) -> None:
    """A changed artefact digest fails closed."""

    manifest_path = write_pack_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["packs"][0]["artifacts"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest(manifest_path, repo_root=tmp_path)


def test_verify_manifest_rejects_missing_job_id(tmp_path: Path) -> None:
    """Declared IBM job IDs must appear in pack artefacts."""

    manifest_path = write_pack_fixture(tmp_path, job_id="ibm-run-present")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["packs"][0]["required_job_ids"] = ["ibm-run-absent"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing declared job IDs"):
        verify_manifest(manifest_path, repo_root=tmp_path)


def test_verify_manifest_rejects_unsafe_artifact_path(tmp_path: Path) -> None:
    """Artefact paths must stay repository-relative."""

    manifest_path = write_pack_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["packs"][0]["artifacts"][0]["path"] = "../outside.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe artifact path"):
        verify_manifest(manifest_path, repo_root=tmp_path)


def test_verify_manifest_filters_pack_ids(tmp_path: Path) -> None:
    """Pack selection verifies only requested known packs."""

    manifest_path = write_pack_fixture(tmp_path)

    summary = verify_manifest(manifest_path, repo_root=tmp_path, pack_ids={"fixture_pack"})

    assert summary["pack_count"] == 1
    assert summary["packs"][0]["id"] == "fixture_pack"


def test_verify_manifest_rejects_unknown_pack_id(tmp_path: Path) -> None:
    """Unknown pack filters fail closed."""

    manifest_path = write_pack_fixture(tmp_path)

    with pytest.raises(ValueError, match="unknown hardware result-pack IDs"):
        verify_manifest(manifest_path, repo_root=tmp_path, pack_ids={"missing_pack"})


def test_export_result_packs_writes_deterministic_archive(tmp_path: Path) -> None:
    """Pack export writes deterministic archive contents and digest."""

    import tarfile

    from scpn_quantum_control.hardware_result_packs import export_result_packs

    manifest_path = write_pack_fixture(tmp_path)
    export_dir = tmp_path / "exports"

    first = export_result_packs(manifest_path, repo_root=tmp_path, export_dir=export_dir)
    first_digest = first[0]["sha256"]
    second = export_result_packs(manifest_path, repo_root=tmp_path, export_dir=export_dir)

    assert second[0]["sha256"] == first_digest
    archive_path = Path(first[0]["path"])
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
    assert names == [
        "fixture_pack/PACK_MANIFEST.json",
        "fixture_pack/data/pack/raw.json",
    ]
