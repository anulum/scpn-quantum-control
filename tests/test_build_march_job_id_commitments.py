# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the March job-id commitment builder
"""Tests for scripts/build_march_job_id_commitments.py.

Exercises mapping validation, scope filtering, label conflict detection,
raw-id publicisation, nonce sidecar lifecycle (mint, reuse, extend,
malformed), commitment arithmetic, payload assembly, the fail-closed leak
check, and the CLI including idempotent re-runs — all on synthetic private
mappings; no real identifiers appear anywhere in this file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import build_march_job_id_commitments as script

RAW_A = "fakejobid00000000aaa"
RAW_B = "fakejobid00000000bbb"
RAW_OUT = "fakejobid00000000ccc"
LABEL_A = "ibm-run-00000000000000a1"
LABEL_B = "ibm-run-00000000000000b2"
LABEL_OUT = "ibm-run-00000000000000c3"
SCOPE = ("results/ibm_hardware_2026-03-28/", "results/march_2026/")


def entry(raw: str, label: str, path: str, pointer: str = "/job_id") -> dict[str, str]:
    return {
        "kind": "raw_ibm_job_id",
        "raw_value": raw,
        "public_label": label,
        "path": path,
        "json_pointer": pointer,
        "field": "job_id",
    }


def make_manifest() -> dict[str, Any]:
    return {
        "schema": script.MAPPING_SCHEMA,
        "created_utc": "2026-05-12T00:00:00Z",
        "label_prefix": "ibm-run",
        "salt_sha256": "ab" * 32,
        "entry_count": 5,
        "entries": [
            entry(RAW_A, LABEL_A, "results/ibm_hardware_2026-03-28/bell_test_4q.json"),
            entry(RAW_A, LABEL_A, "results/march_2026/campaign_manifest.json", "/job_ids/0"),
            entry(RAW_B, LABEL_B, f"results/march_2026/job_{RAW_B}.json"),
            entry(RAW_OUT, LABEL_OUT, "data/phase1_dla_parity/other.json"),
            {"kind": "operational_metadata", "path": "results/march_2026/x.json"},
        ],
    }


def write_manifest(tmp_path: Path, manifest: dict[str, Any]) -> Path:
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class TestLoadPrivateMapping:
    def test_happy_path(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path, make_manifest())
        manifest = script.load_private_mapping(path)
        assert manifest["label_prefix"] == "ibm-run"

    def test_rejects_non_object(self, tmp_path: Path) -> None:
        path = tmp_path / "mapping.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="not a JSON object"):
            script.load_private_mapping(path)

    def test_rejects_wrong_schema(self, tmp_path: Path) -> None:
        manifest = make_manifest()
        manifest["schema"] = "other.v9"
        with pytest.raises(ValueError, match="has schema"):
            script.load_private_mapping(write_manifest(tmp_path, manifest))

    def test_rejects_missing_entries(self, tmp_path: Path) -> None:
        manifest = make_manifest()
        manifest["entries"] = None
        with pytest.raises(ValueError, match="no entries list"):
            script.load_private_mapping(write_manifest(tmp_path, manifest))

    @pytest.mark.parametrize("key", ["salt_sha256", "label_prefix"])
    def test_rejects_missing_metadata(self, tmp_path: Path, key: str) -> None:
        manifest = make_manifest()
        manifest[key] = ""
        with pytest.raises(ValueError, match=f"missing {key}"):
            script.load_private_mapping(write_manifest(tmp_path, manifest))


class TestEntrySelection:
    def test_raw_entries_skip_operational_metadata(self) -> None:
        entries = script.raw_job_id_entries(make_manifest())
        assert len(entries) == 4
        assert all(e["kind"].startswith("raw_ibm_job_id") for e in entries)

    def test_raw_entries_reject_missing_field(self) -> None:
        manifest = make_manifest()
        del manifest["entries"][0]["json_pointer"]
        with pytest.raises(ValueError, match="json_pointer"):
            script.raw_job_id_entries(manifest)

    def test_scope_filters_out_of_scope_paths(self) -> None:
        entries = script.raw_job_id_entries(make_manifest())
        scoped = script.scoped_entries(entries, SCOPE)
        assert {e["raw_value"] for e in scoped} == {RAW_A, RAW_B}

    def test_scope_with_no_match_fails_closed(self) -> None:
        entries = script.raw_job_id_entries(make_manifest())
        with pytest.raises(ValueError, match="no raw job-id entries matched"):
            script.scoped_entries(entries, ("results/none/",))


class TestLabels:
    def test_label_by_raw_maps_all(self) -> None:
        labels = script.label_by_raw(script.raw_job_id_entries(make_manifest()))
        assert labels == {RAW_A: LABEL_A, RAW_B: LABEL_B, RAW_OUT: LABEL_OUT}

    def test_label_conflict_is_rejected(self) -> None:
        manifest = make_manifest()
        manifest["entries"][1]["public_label"] = LABEL_B
        with pytest.raises(ValueError, match="conflicting public labels"):
            script.label_by_raw(script.raw_job_id_entries(manifest))

    def test_publicise_replaces_longest_first(self) -> None:
        long_raw = RAW_A + "x"
        labels = {RAW_A: LABEL_A, long_raw: LABEL_B}
        assert script.publicise(f"job_{long_raw}.json", labels) == f"job_{LABEL_B}.json"
        assert script.publicise(f"job_{RAW_A}.json", labels) == f"job_{LABEL_A}.json"


class TestNonceSidecar:
    def test_fresh_mint_and_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "nonces.json"
        nonces, minted = script.load_or_extend_nonces(path, [RAW_A, RAW_B])
        assert minted == 2
        assert all(len(n) == 32 for n in nonces.values())
        script.write_nonce_sidecar(path, nonces)
        reloaded, minted_again = script.load_or_extend_nonces(path, [RAW_A, RAW_B])
        assert minted_again == 0
        assert reloaded == nonces

    def test_extends_with_new_raw_id(self, tmp_path: Path) -> None:
        path = tmp_path / "nonces.json"
        first, _ = script.load_or_extend_nonces(path, [RAW_A])
        script.write_nonce_sidecar(path, first)
        extended, minted = script.load_or_extend_nonces(path, [RAW_A, RAW_B])
        assert minted == 1
        assert extended[RAW_A] == first[RAW_A]

    def test_malformed_sidecar_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "nonces.json"
        path.write_text(json.dumps({"schema": "wrong", "nonces": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            script.load_or_extend_nonces(path, [RAW_A])


class TestCommitments:
    def test_commitment_digest_matches_direct_arithmetic(self) -> None:
        expected = hashlib.sha256(f"{RAW_A}:00ff".encode()).hexdigest()
        assert script.commitment_digest(RAW_A, "00ff") == expected

    def test_build_payload_shape(self) -> None:
        manifest = make_manifest()
        nonces = {RAW_A: "aa" * 16, RAW_B: "bb" * 16}
        payload = script.build_payload(manifest, SCOPE, nonces, "2026-07-17T00:00:00Z")
        assert payload["schema"] == script.COMMITMENT_SCHEMA
        assert payload["commitment_count"] == 2
        assert payload["salt_sha256"] == manifest["salt_sha256"]
        labels = [c["public_label"] for c in payload["commitments"]]
        assert labels == sorted(labels)
        first = payload["commitments"][0]
        assert first["public_label"] == LABEL_A
        assert first["commitment_sha256"] == script.commitment_digest(RAW_A, nonces[RAW_A])
        assert len(first["artefact_references"]) == 2

    def test_duplicate_references_are_deduplicated(self) -> None:
        manifest = make_manifest()
        manifest["entries"].append(dict(manifest["entries"][0]))
        nonces = {RAW_A: "aa" * 16, RAW_B: "bb" * 16}
        payload = script.build_payload(manifest, SCOPE, nonces, "2026-07-17T00:00:00Z")
        first = payload["commitments"][0]
        assert first["public_label"] == LABEL_A
        assert len(first["artefact_references"]) == 2

    def test_build_payload_publicises_paths_and_pointers(self) -> None:
        manifest = make_manifest()
        manifest["entries"][2]["json_pointer"] = f"/jobs/{RAW_B}/id"
        nonces = {RAW_A: "aa" * 16, RAW_B: "bb" * 16}
        payload = script.build_payload(manifest, SCOPE, nonces, "2026-07-17T00:00:00Z")
        serialised = json.dumps(payload)
        assert RAW_A not in serialised and RAW_B not in serialised
        assert f"job_{LABEL_B}.json" in serialised
        assert f"/jobs/{LABEL_B}/id" in serialised


class TestLeakCheck:
    def test_clean_payload_passes(self) -> None:
        nonces = {RAW_A: "aa" * 16}
        script.assert_no_private_leak("only labels here", make_manifest(), nonces)

    def test_raw_id_leak_is_fatal(self) -> None:
        with pytest.raises(RuntimeError, match=f"raw job id for {LABEL_A}"):
            script.assert_no_private_leak(f"text {RAW_A} text", make_manifest(), {})

    def test_nonce_leak_is_fatal(self) -> None:
        nonces = {RAW_A: "cd" * 16}
        with pytest.raises(RuntimeError, match="nonce for"):
            script.assert_no_private_leak(f"text {nonces[RAW_A]}", make_manifest(), nonces)


class TestWriteJson:
    def test_returns_sha256_of_written_text(self, tmp_path: Path) -> None:
        path = tmp_path / "out" / "payload.json"
        digest = script._write_json(path, {"k": "v"})
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


class TestMain:
    def run_main(self, tmp_path: Path) -> tuple[int, Path, Path]:
        mapping = write_manifest(tmp_path, make_manifest())
        nonces = tmp_path / "nonces.json"
        output = tmp_path / "out" / "commitments.json"
        code = script.main(
            [
                "--private-mapping",
                str(mapping),
                "--nonces",
                str(nonces),
                "--output",
                str(output),
                "--scope",
                SCOPE[0],
                "--scope",
                SCOPE[1],
            ]
        )
        return code, output, nonces

    def test_happy_path_writes_commitments_and_sidecar(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, output, nonces = self.run_main(tmp_path)
        assert code == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["commitment_count"] == 2
        sidecar = json.loads(nonces.read_text(encoding="utf-8"))
        assert set(sidecar["nonces"]) == {RAW_A, RAW_B}
        out = capsys.readouterr().out
        assert "commitments: 2 unique job ids" in out
        assert "minted nonces: 2" in out
        assert RAW_A not in output.read_text(encoding="utf-8")

    def test_rerun_is_idempotent(self, tmp_path: Path) -> None:
        _, output, _ = self.run_main(tmp_path)
        first = json.loads(output.read_text(encoding="utf-8"))
        code, output, _ = self.run_main(tmp_path)
        assert code == 0
        second = json.loads(output.read_text(encoding="utf-8"))
        assert first["commitments"] == second["commitments"]

    def test_default_scope_is_used_without_scope_args(self, tmp_path: Path) -> None:
        mapping = write_manifest(tmp_path, make_manifest())
        code = script.main(
            [
                "--private-mapping",
                str(mapping),
                "--nonces",
                str(tmp_path / "n.json"),
                "--output",
                str(tmp_path / "c.json"),
            ]
        )
        assert code == 0
        payload = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
        assert payload["scope"] == sorted(script.DEFAULT_SCOPE)
