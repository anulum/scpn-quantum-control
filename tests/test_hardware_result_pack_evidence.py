# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- hardware result-pack evidence generator tests
"""Tests for hardware result-pack evidence-packet generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control import hardware_result_pack_evidence as evidence


def test_generate_non_citing_packet(tmp_path: Path) -> None:
    """The generator can emit a valid non-citing release packet."""

    exit_code = evidence.main(
        [
            "--non-citing",
            "--reason",
            "No hardware claims in this release.",
            "--output-dir",
            str(tmp_path),
        ]
    )

    packets = list(tmp_path.glob("hardware_result_pack_evidence_*.json"))
    assert exit_code == 0
    assert len(packets) == 1
    payload = json.loads(packets[0].read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "hardware_evidence_cited": False,
        "reason": "No hardware claims in this release.",
    }


def test_generate_citing_packet_orchestrates_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The citing path writes verifier/export summaries, logs, and log digests."""

    output_dir = tmp_path / "release"
    export_dir = tmp_path / "exports"
    monkeypatch.setattr(evidence, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(evidence, "VERIFY_SCRIPT", tmp_path / "scripts" / "verify.py")
    monkeypatch.setattr(
        evidence,
        "load_manifest",
        lambda: {
            "schema_version": 1,
            "packs": [
                {
                    "id": "pack_a",
                    "reproduce_command": "python reproduce_pack_a.py",
                }
            ],
        },
    )

    def stub_run_json(command: list[str], *, cwd: Path, output_path: Path) -> dict[str, Any]:
        payload = (
            {"pack_count": 1, "packs": [{"id": "pack_a"}]}
            if "--export-dir" not in command
            else {"exports": [{"id": "pack_a", "sha256": "abc", "bytes": 10}]}
        )
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def stub_run_log(command: str, *, cwd: Path, log_path: Path) -> None:
        assert command == "python reproduce_pack_a.py"
        log_path.write_text("reproduced\n", encoding="utf-8")

    monkeypatch.setattr(evidence, "run_json_command", stub_run_json)
    monkeypatch.setattr(evidence, "run_log_command", stub_run_log)

    exit_code = evidence.main(
        [
            "--pack-id",
            "pack_a",
            "--output-dir",
            str(output_dir),
            "--export-dir",
            str(export_dir),
        ]
    )

    packets = list(output_dir.glob("hardware_result_pack_evidence_*.json"))
    assert exit_code == 0
    assert len(packets) == 1
    payload = json.loads(packets[0].read_text(encoding="utf-8"))
    assert payload["hardware_evidence_cited"] is True
    assert payload["verifier_summary_path"].endswith(".json")
    assert payload["export_summary_path"].endswith(".json")
    assert payload["reproduction_logs"][0]["pack_id"] == "pack_a"
    assert len(payload["reproduction_logs"][0]["sha256"]) == 64


def test_run_json_command_captures_stdout_and_parses(tmp_path: Path) -> None:
    """A JSON-emitting command persists stdout and returns the parsed payload."""
    output = tmp_path / "out.json"
    payload = evidence.run_json_command(
        [sys.executable, "-c", "print('{\"ok\": 1}')"],
        cwd=tmp_path,
        output_path=output,
    )
    assert payload == {"ok": 1}
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": 1}


def test_run_json_command_logs_stderr_on_success(tmp_path: Path) -> None:
    """Non-fatal stderr from a succeeding command is captured beside the output."""
    output = tmp_path / "out.json"
    evidence.run_json_command(
        [sys.executable, "-c", "import sys; sys.stderr.write('warn'); print('{}')"],
        cwd=tmp_path,
        output_path=output,
    )
    assert (tmp_path / "out.json.stderr.log").read_text(encoding="utf-8") == "warn"


def test_run_json_command_raises_on_failure(tmp_path: Path) -> None:
    """A non-zero command writes a stderr log and raises."""
    output = tmp_path / "out.json"
    with pytest.raises(RuntimeError, match="command failed"):
        evidence.run_json_command(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
            cwd=tmp_path,
            output_path=output,
        )
    assert (tmp_path / "out.json.stderr.log").read_text(encoding="utf-8") == "boom"


def test_run_log_command_captures_and_raises(tmp_path: Path) -> None:
    """The reproduction runner logs output and fails closed on non-zero exit."""
    log_path = tmp_path / "run.log"
    evidence.run_log_command(
        f"{sys.executable} -c \"print('hello')\"", cwd=tmp_path, log_path=log_path
    )
    assert "hello" in log_path.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError, match="reproduction command failed"):
        evidence.run_log_command(
            f'{sys.executable} -c "import sys; sys.exit(2)"',
            cwd=tmp_path,
            log_path=tmp_path / "fail.log",
        )


def test_load_manifest_reads_committed_manifest() -> None:
    """The committed hardware result-pack manifest loads as a mapping."""
    manifest = evidence.load_manifest()
    assert isinstance(manifest, dict)
    assert "packs" in manifest


def test_select_packs_returns_all_when_unfiltered() -> None:
    """A None filter returns every pack."""
    manifest = {"packs": [{"id": "a"}, {"id": "b"}]}
    assert evidence.select_packs(manifest, None) == manifest["packs"]


def test_select_packs_rejects_unknown_ids() -> None:
    """Unknown pack filters fail closed."""
    with pytest.raises(ValueError, match="unknown hardware result-pack IDs"):
        evidence.select_packs({"packs": [{"id": "a"}]}, {"missing"})


def test_parse_pack_ids_empty_is_none() -> None:
    """No pack filters parse to None."""
    assert evidence.parse_pack_ids([]) is None


def test_main_raises_when_no_packs_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The citing path fails closed when the manifest yields no packs."""
    monkeypatch.setattr(evidence, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(evidence, "load_manifest", lambda: {"schema_version": 1, "packs": []})
    with pytest.raises(RuntimeError, match="no hardware result packs selected"):
        evidence.main(["--output-dir", str(tmp_path / "release")])


def test_main_raises_when_pack_lacks_reproduce_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected pack without a reproduce command fails closed."""
    monkeypatch.setattr(evidence, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        evidence,
        "load_manifest",
        lambda: {"schema_version": 1, "packs": [{"id": "pack_x"}]},
    )

    def stub_run_json(command: list[str], *, cwd: Path, output_path: Path) -> dict[str, Any]:
        output_path.write_text("{}", encoding="utf-8")
        return {"packs": [{"id": "pack_x"}]}

    monkeypatch.setattr(evidence, "run_json_command", stub_run_json)
    with pytest.raises(RuntimeError, match="does not declare reproduce_command"):
        evidence.main(["--pack-id", "pack_x", "--output-dir", str(tmp_path / "release")])
