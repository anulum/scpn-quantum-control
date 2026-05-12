# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM public artefact anonymisation tests
"""Tests for the IBM public artefact anonymisation helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_ANON_TOOL = ROOT / "tools" / "anonymize_ibm_public_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("anonymize_ibm_public_artifacts", _ANON_TOOL)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MANIFEST_SCHEMA = _MODULE.MANIFEST_SCHEMA
build_private_manifest = _MODULE.build_private_manifest
looks_like_raw_ibm_job_id = _MODULE.looks_like_raw_ibm_job_id
main = _MODULE.main
public_run_label = _MODULE.public_run_label
sanitise_json_files = _MODULE.sanitise_json_files
sanitise_json_payload = _MODULE.sanitise_json_payload

RAW_EXAMPLE_ID = "d" + "6h3e2f3o3rs73caglmg"
RAW_FIM_PILOT_ID = "d" + "7t53ofljm6s73bc6bj0"
RAW_FIM_REPEAT_ID = "d" + "7t5gtaudops7397ikn0"


def test_public_run_label_is_deterministic_hmac_label() -> None:
    label = public_run_label(RAW_EXAMPLE_ID, "private-salt")

    assert label == public_run_label(RAW_EXAMPLE_ID, "private-salt")
    assert label.startswith("ibm-run-")
    assert RAW_EXAMPLE_ID not in label


def test_public_run_label_rejects_empty_salt() -> None:
    with pytest.raises(ValueError, match="salt"):
        public_run_label(RAW_EXAMPLE_ID, "")


def test_raw_job_id_detector_accepts_observed_ibm_shape_only() -> None:
    assert looks_like_raw_ibm_job_id(RAW_EXAMPLE_ID)
    assert not looks_like_raw_ibm_job_id("ibm-run-1234567890abcdef")
    assert not looks_like_raw_ibm_job_id("job-1")


def test_sanitise_json_payload_replaces_job_ids_and_preserves_counts() -> None:
    payload = {
        "backend": "ibm_fez",
        "job_id": RAW_EXAMPLE_ID,
        "counts": {"0000": 12, "1111": 4},
        "submitted_at_utc": "2026-05-05T20:24:00Z",
    }

    result = sanitise_json_payload(payload, source_path=Path("results/example.json"), salt="salt")

    assert result.changed
    assert result.public_payload["backend"] == "ibm_fez"
    assert result.public_payload["counts"] == payload["counts"]
    assert result.public_payload["job_id"].startswith("ibm-run-")
    assert "submitted_at_utc" not in result.public_payload
    assert {entry.kind for entry in result.entries} == {
        "operational_metadata",
        "raw_ibm_job_id",
    }


def test_sanitise_json_payload_handles_job_id_lists_and_nested_rows() -> None:
    payload = {
        "job_ids": [RAW_FIM_PILOT_ID, "non-provider-id"],
        "circuits": [
            {"job_id": RAW_FIM_REPEAT_ID, "counts": {"0": 1}},
            {"job_id": "local_simulated", "counts": {"1": 1}},
        ],
    }

    result = sanitise_json_payload(payload, source_path=Path("data/example.json"), salt="salt")

    assert result.public_payload["job_ids"][0].startswith("ibm-run-")
    assert result.public_payload["job_ids"][1] == "non-provider-id"
    assert result.public_payload["circuits"][0]["job_id"].startswith("ibm-run-")
    assert result.public_payload["circuits"][1]["job_id"] == "local_simulated"
    assert len([entry for entry in result.entries if entry.kind == "raw_ibm_job_id"]) == 2


def test_private_manifest_records_raw_values_but_not_salt() -> None:
    result = sanitise_json_payload(
        {"job_id": RAW_EXAMPLE_ID},
        source_path=Path("results/example.json"),
        salt="private-secret-salt",
    )

    manifest = build_private_manifest(
        result.entries,
        salt="private-secret-salt",
        created_utc=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["created_utc"] == "2026-05-13T00:00:00Z"
    assert manifest["entry_count"] == 1
    assert manifest["entries"][0]["raw_value"] == RAW_EXAMPLE_ID
    assert manifest["entries"][0]["public_label"].startswith("ibm-run-")
    assert "private-secret-salt" not in json.dumps(manifest)


def test_sanitise_json_files_dry_run_does_not_modify_public_file(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    artefact = data / "run.json"
    artefact.write_text(f'{{"job_id": "{RAW_EXAMPLE_ID}"}}\n', encoding="utf-8")

    results = sanitise_json_files(tmp_path, (Path("data/run.json"),), salt="salt", write=False)

    assert results[0].changed
    assert json.loads(artefact.read_text(encoding="utf-8"))["job_id"] == RAW_EXAMPLE_ID


def test_sanitise_json_files_write_updates_public_file(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    artefact = data / "run.json"
    artefact.write_text(
        f'{{"job_id": "{RAW_EXAMPLE_ID}", "created_utc": "2026-05-05T00:00:00Z"}}\n',
        encoding="utf-8",
    )

    sanitise_json_files(tmp_path, (Path("data/run.json"),), salt="salt", write=True)
    decoded = json.loads(artefact.read_text(encoding="utf-8"))

    assert decoded["job_id"].startswith("ibm-run-")
    assert "created_utc" not in decoded


def test_cli_writes_manifest_without_rewriting_by_default(tmp_path: Path, capsys: object) -> None:
    data = tmp_path / "data"
    data.mkdir()
    artefact = data / "run.json"
    manifest = tmp_path / "private_manifest.json"
    artefact.write_text(f'{{"job_id": "{RAW_EXAMPLE_ID}"}}\n', encoding="utf-8")

    assert (
        main(["--project-root", str(tmp_path), "--manifest", str(manifest), "--salt", "salt"]) == 0
    )

    summary = json.loads(capsys.readouterr().out)
    decoded_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert summary["changed_files"] == 1
    assert summary["write"] is False
    assert decoded_manifest["entry_count"] == 1
    assert json.loads(artefact.read_text(encoding="utf-8"))["job_id"] == RAW_EXAMPLE_ID


def test_cli_requires_exactly_one_salt_source(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="exactly one"):
        main(["--project-root", str(tmp_path), "--manifest", str(manifest)])
