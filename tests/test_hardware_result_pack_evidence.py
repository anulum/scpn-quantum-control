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
from pathlib import Path

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


def test_generate_citing_packet_orchestrates_commands(tmp_path: Path, monkeypatch) -> None:
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

    def stub_run_json(command: list[str], *, cwd: Path, output_path: Path):
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
