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


def _rewrite_manifest(manifest_path: Path, mutate: object) -> None:
    """Load, mutate, and re-write a manifest JSON payload in place."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_repo_root_prefers_cwd_with_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the working directory holds the manifest, it is the repo root."""

    from scpn_quantum_control.hardware_result_packs import (
        MANIFEST_RELATIVE_PATH,
        default_repo_root,
    )

    (tmp_path / MANIFEST_RELATIVE_PATH).parent.mkdir(parents=True)
    (tmp_path / MANIFEST_RELATIVE_PATH).write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert default_repo_root() == tmp_path


def test_default_repo_root_falls_back_to_source_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a manifest in the cwd, the source-tree parent layout is used."""

    from scpn_quantum_control.hardware_result_packs import default_repo_root

    monkeypatch.chdir(tmp_path)

    root = default_repo_root()

    assert (root / "src" / "scpn_quantum_control").exists()


def test_digest_bytes_matches_hashlib() -> None:
    """digest_bytes mirrors hashlib.sha256 over the same payload."""

    import hashlib

    from scpn_quantum_control.hardware_result_packs import digest_bytes

    payload = b"hardware-result-pack"
    assert digest_bytes(payload) == hashlib.sha256(payload).hexdigest()


def test_load_manifest_rejects_bad_schema_version(tmp_path: Path) -> None:
    """A manifest with an unsupported schema_version fails closed."""

    from scpn_quantum_control.hardware_result_packs import load_manifest

    manifest_path = write_pack_fixture(tmp_path)
    _rewrite_manifest(manifest_path, lambda p: p.__setitem__("schema_version", 99))

    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_empty_packs(tmp_path: Path) -> None:
    """A manifest without packs fails closed."""

    from scpn_quantum_control.hardware_result_packs import load_manifest

    manifest_path = write_pack_fixture(tmp_path)
    _rewrite_manifest(manifest_path, lambda p: p.__setitem__("packs", []))

    with pytest.raises(ValueError, match="at least one pack"):
        load_manifest(manifest_path)


def test_verify_rejects_missing_pack_id(tmp_path: Path) -> None:
    """A pack without an id fails closed."""

    manifest_path = write_pack_fixture(tmp_path)
    _rewrite_manifest(manifest_path, lambda p: p["packs"][0].__setitem__("id", ""))

    with pytest.raises(ValueError, match="missing id"):
        verify_manifest(manifest_path, repo_root=tmp_path)


def test_verify_rejects_pack_without_artifacts(tmp_path: Path) -> None:
    """A pack with an empty artefact list fails closed."""

    manifest_path = write_pack_fixture(tmp_path)
    _rewrite_manifest(manifest_path, lambda p: p["packs"][0].__setitem__("artifacts", []))

    with pytest.raises(ValueError, match="no artifacts"):
        verify_manifest(manifest_path, repo_root=tmp_path)


def test_verify_rejects_missing_artifact_file(tmp_path: Path) -> None:
    """A manifest referencing an absent artefact fails closed."""

    manifest_path = write_pack_fixture(tmp_path)
    (tmp_path / "data" / "pack" / "raw.json").unlink()

    with pytest.raises(FileNotFoundError, match="missing artifact"):
        verify_manifest(manifest_path, repo_root=tmp_path)


def test_verify_rejects_size_mismatch(tmp_path: Path) -> None:
    """A declared byte size that disagrees with the file fails closed."""

    manifest_path = write_pack_fixture(tmp_path)
    _rewrite_manifest(
        manifest_path, lambda p: p["packs"][0]["artifacts"][0].__setitem__("bytes", 999999)
    )

    with pytest.raises(ValueError, match="size mismatch"):
        verify_manifest(manifest_path, repo_root=tmp_path)


def test_verify_rejects_non_list_job_ids(tmp_path: Path) -> None:
    """required_job_ids must be a list."""

    manifest_path = write_pack_fixture(tmp_path)
    _rewrite_manifest(
        manifest_path, lambda p: p["packs"][0].__setitem__("required_job_ids", "not-a-list")
    )

    with pytest.raises(ValueError, match="required_job_ids must be a list"):
        verify_manifest(manifest_path, repo_root=tmp_path)


def test_verify_labels_manifest_outside_repo_root(tmp_path: Path) -> None:
    """A manifest outside the repo root is labelled by its absolute path."""

    repo_root = tmp_path / "repo"
    artifact_path = repo_root / "data" / "pack" / "raw.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(json.dumps({"job_ids": ["j1"]}), encoding="utf-8")
    manifest_path = tmp_path / "external_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "packs": [
                    {
                        "id": "p",
                        "status": "ok",
                        "required_job_ids": ["j1"],
                        "artifacts": [
                            {
                                "path": "data/pack/raw.json",
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

    summary = verify_manifest(manifest_path, repo_root=repo_root)

    assert summary["manifest"] == str(manifest_path.resolve())


def test_parse_pack_ids_variants() -> None:
    """Pack-id parsing handles empty, comma-separated, and blank-only inputs."""

    from scpn_quantum_control.hardware_result_packs import parse_pack_ids

    assert parse_pack_ids([]) is None
    assert parse_pack_ids(["a,b", "c"]) == {"a", "b", "c"}
    assert parse_pack_ids([" , "]) is None


def test_main_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI emits a machine-readable summary and returns 0."""

    from scpn_quantum_control import hardware_result_packs as hrp

    manifest_path = write_pack_fixture(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--repo-root", str(tmp_path), "--manifest", str(manifest_path), "--json"],
    )

    assert hrp.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pack_count"] == 1


def test_main_text_output_with_export_and_default_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI default-roots from the cwd, prints a text summary, and exports."""

    from scpn_quantum_control import hardware_result_packs as hrp

    write_pack_fixture(tmp_path)
    export_dir = tmp_path / "exports"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["prog", "--export-dir", str(export_dir)])

    assert hrp.main() == 0
    out = capsys.readouterr().out
    assert "verification passed" in out
    assert "Exported deterministic archives" in out
    assert (export_dir / "fixture_pack.tar.gz").exists()
