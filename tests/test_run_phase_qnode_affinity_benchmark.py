# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ordinary-import affinity benchmark CLI
"""Protect real benchmark evidence, replay commands and isolation refusals."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import pytest

from tools.run_phase_qnode_affinity_benchmark import main

ROOT = Path(__file__).resolve().parents[1]


class Metadata(TypedDict):
    """Recorded command used to reproduce an actual affinity measurement."""

    command: str


class Evidence(TypedDict):
    """Public benchmark evidence fields whose CLI behavior must remain stable."""

    evidence_label: str
    production_benchmark: bool
    metadata: Metadata
    raw_timing_rows: list[object]


@pytest.mark.parametrize("recorded", [False, True])
def test_cli_writes_real_evidence_and_preserves_replay_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: bool
) -> None:
    """Run the genuine benchmark and retain the requested outer command.

    Parameters
    ----------
    tmp_path
        Evidence output directory.
    monkeypatch
        Scoped public CLI argument vector.
    recorded
        Whether the caller supplies an exact admitted outer replay command.

    """
    output = tmp_path / "directory with spaces" / "affinity.json"
    outer = "taskset -c 2 chrt -f 1 python tools/run_phase_qnode_affinity_benchmark.py"
    arguments = ["benchmark", "--repetitions", "2", "--warmups", "1", "--output", str(output)]
    if recorded:
        arguments.extend(["--recorded-command", outer])
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    payload = cast(Evidence, json.loads(output.read_text(encoding="utf-8")))
    assert payload["evidence_label"] == "functional_non_isolated"
    assert payload["production_benchmark"] is False
    assert payload["raw_timing_rows"]
    if recorded:
        assert payload["metadata"]["command"] == outer
    else:
        assert shlex.split(payload["metadata"]["command"]) == [
            "python",
            "tools/run_phase_qnode_affinity_benchmark.py",
            "--repetitions",
            "2",
            "--warmups",
            "1",
            "--reserved-cpus",
            "",
            "--output",
            str(output),
        ]


def test_strict_cli_writes_diagnostics_and_refuses_unreserved_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persist real diagnostic evidence before rejecting absent affinity isolation.

    Parameters
    ----------
    tmp_path
        Diagnostic evidence output directory.
    monkeypatch
        Scoped public CLI argument vector requiring actual isolation.

    """
    output = tmp_path / "diagnostic.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--repetitions",
            "1",
            "--warmups",
            "0",
            "--require-isolated",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit, match="isolated_affinity evidence was required"):
        main()
    payload = cast(Evidence, json.loads(output.read_text(encoding="utf-8")))
    assert payload["evidence_label"] == "functional_non_isolated"
    assert payload["production_benchmark"] is False
    assert "--require-isolated" in shlex.split(payload["metadata"]["command"])


def test_script_entry_point_runs_with_the_real_package_initializers(tmp_path: Path) -> None:
    """Execute the actual script after retiring its package-shell loader.

    Parameters
    ----------
    tmp_path
        Output destination for the genuine subprocess measurement.

    """
    output = tmp_path / "subprocess.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_phase_qnode_affinity_benchmark.py"),
            "--repetitions",
            "1",
            "--warmups",
            "0",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = cast(Evidence, json.loads(output.read_text(encoding="utf-8")))
    assert payload["raw_timing_rows"]
    assert payload["evidence_label"] == "functional_non_isolated"
