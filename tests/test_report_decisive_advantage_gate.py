# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the decisive advantage-gate report script
"""Tests for the decisive advantage-gate report CLI (real script boundary)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.benchmarks.decisive_advantage_protocol import (
    default_decisive_advantage_protocol,
)
from scripts.report_decisive_advantage_gate import (
    _rows_from_payload as rows_from_payload,
)
from scripts.report_decisive_advantage_gate import (
    build_report,
    main,
)

PROTOCOL = default_decisive_advantage_protocol()


def _valid_rows() -> list[dict[str, Any]]:
    def row(baseline: str, wall: float) -> dict[str, Any]:
        return {
            "protocol_id": PROTOCOL.protocol.protocol_id,
            "n_qubits": PROTOCOL.criterion.target_size,
            "baseline": baseline,
            "status": "ok",
            "wall_time_ms": wall,
            "memory_bytes": 2048,
            "metric_payload": {"reference_error": 0.004, "order_parameter_R": 0.5},
            "command": "run",
            "machine": "ml350",
            "dependencies": {"numpy": "2.1.0"},
            "git_commit": "cafe",
            "notes": [],
        }

    return [
        row("classical_ode", 100.0),
        row("mps_tensor_network", 110.0),
        row("dense_statevector_evolution", 90.0),
        row("qpu_hardware", 5000.0),
    ]


class TestRowsFromPayload:
    def test_list_payload(self) -> None:
        assert rows_from_payload([{"a": 1}]) == [{"a": 1}]

    def test_mapping_payload(self) -> None:
        assert rows_from_payload({"rows": [{"a": 1}]}) == [{"a": 1}]

    def test_invalid_payload_raises(self) -> None:
        with pytest.raises(ValueError, match="rows payload"):
            rows_from_payload(42)


class TestBuildReport:
    def test_preregistration_only_has_no_decision(self) -> None:
        report = build_report(PROTOCOL, None)
        assert "protocol" in report
        assert "decision" not in report

    def test_with_rows_has_validation_and_decision(self) -> None:
        report = build_report(PROTOCOL, _valid_rows())
        assert report["validation"]["valid"] is True
        assert report["decision"]["label"] == "classical_wins"


class TestMain:
    def test_preregistration_only_writes_manifest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["--out-dir", str(tmp_path)])
        assert code == 0
        out = tmp_path / f"{PROTOCOL.protocol.protocol_id}.json"
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert "decision" not in payload
        assert "preregistration only" in capsys.readouterr().out

    def test_with_rows_reports_decision(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rows_path = tmp_path / "rows.json"
        rows_path.write_text(json.dumps(_valid_rows()), encoding="utf-8")
        code = main(["--rows", str(rows_path), "--out-dir", str(tmp_path)])
        assert code == 0
        assert "decision: classical_wins" in capsys.readouterr().out
        payload = json.loads(
            (tmp_path / f"{PROTOCOL.protocol.protocol_id}.json").read_text(encoding="utf-8")
        )
        assert payload["decision"]["label"] == "classical_wins"
