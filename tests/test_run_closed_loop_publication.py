# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the closed-loop publication run script
"""Tests for scripts/run_closed_loop_publication.py.

The publication run is stubbed via a canned artifact so the CLI wiring
(argument parsing, config assembly, JSON output, console summary) is exercised
without live wall-clock measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.benchmarks.closed_loop_publication_run import (
    ClosedLoopPublicationArtifact,
)
from scripts import run_closed_loop_publication as script


def _canned_artifact() -> ClosedLoopPublicationArtifact:
    return ClosedLoopPublicationArtifact(
        package={"claim_ledger_rows": []},
        latency_report={"classification": "software_in_loop_latency", "passes": False},
        dynamic_circuit_templates={"claim_note": "un-run"},
        timing_grade="advisory_shared_host",
        host={"ready": False},
        config={"seed": 0},
        provenance={"git_commit": "abc"},
        notes=("not a hardware measurement",),
        package_markdown="# Closed-Loop Quantum Control Evidence Package",
    )


class TestArgumentParsing:
    def test_defaults(self) -> None:
        args = script._parse_args([])
        assert (args.n, args.rounds, args.template_rounds, args.seed) == (4, 32, 3, 0)
        assert Path(args.out_dir) == script.DEFAULT_OUT_DIR

    def test_custom_arguments(self) -> None:
        args = script._parse_args(
            ["--n", "3", "--coupling", "0.4", "--target-r", "0.5", "--rounds", "8", "--seed", "2"]
        )
        assert (args.n, args.coupling, args.target_r, args.rounds, args.seed) == (
            3,
            0.4,
            0.5,
            8,
            2,
        )


class TestMain:
    def test_writes_artifact_and_prints_summary(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(config: Any) -> Any:
            captured["config"] = config
            return _canned_artifact()

        monkeypatch.setattr(script, "run_closed_loop_publication", fake_run)
        code = script.main(
            ["--out-dir", str(tmp_path), "--n", "3", "--rounds", "8", "--seed", "5"]
        )
        assert code == 0

        config = captured["config"]
        assert (config.n_oscillators, config.n_rounds, config.seed) == (3, 8, 5)

        artifacts = list(tmp_path.glob("closed_loop_publication_n3_seed5.json"))
        assert len(artifacts) == 1
        payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert payload["timing_grade"] == "advisory_shared_host"
        assert payload["latency_report"]["passes"] is False

        out = capsys.readouterr().out
        assert "# Closed-Loop Quantum Control Evidence Package" in out
        assert "timing_grade: advisory_shared_host" in out
        assert "not a hardware measurement" in out
        assert "written:" in out

    def test_config_carries_cli_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(config: Any) -> Any:
            captured["config"] = config
            return _canned_artifact()

        monkeypatch.setattr(script, "run_closed_loop_publication", fake_run)
        code = script.main(
            [
                "--out-dir",
                str(tmp_path),
                "--coupling",
                "0.4",
                "--target-r",
                "0.5",
                "--template-rounds",
                "2",
                "--reserved-core",
                "1",
            ]
        )
        assert code == 0
        config = captured["config"]
        assert (
            config.coupling,
            config.target_r,
            config.dynamic_circuit_rounds,
            config.reserved_core,
        ) == (0.4, 0.5, 2, 1)
