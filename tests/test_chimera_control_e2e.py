# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera-control CLI boundary test
"""End-to-end evidence-runner test through the real command-line boundary."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_runner(repo: Path) -> ModuleType:
    path = repo / "scripts/run_chimera_multiscale_control_evidence.py"
    spec = importlib.util.spec_from_file_location("chimera_evidence_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load chimera evidence runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_runner_writes_then_byte_checks_real_outputs(tmp_path: Path) -> None:
    """Exercise write and byte-check modes through the real subprocess boundary."""
    repo = Path(__file__).resolve().parents[1]
    json_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "evidence.md"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{repo / 'src'}:{repo / 'oscillatools/src'}"
    command = [
        sys.executable,
        str(repo / "scripts/run_chimera_multiscale_control_evidence.py"),
        "--population-size",
        "4",
        "--json-path",
        str(json_path),
        "--markdown-path",
        str(markdown_path),
    ]

    written = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    checked = subprocess.run(
        [*command, "--check"],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"wrote {json_path}" in written.stdout
    assert "content_digest=" in written.stdout
    assert f"checked {json_path}" in checked.stdout
    assert json_path.stat().st_size > 1000
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Chimera and Multiscale Control Evidence"
    )


def test_evidence_runner_main_and_parser_execute_in_process(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise parser and main entry points in-process without bypassing writes."""
    repo = Path(__file__).resolve().parents[1]
    runner = _load_runner(repo)
    json_path = tmp_path / "direct.json"
    markdown_path = tmp_path / "direct.md"
    arguments = [
        "--population-size",
        "4",
        "--json-path",
        str(json_path),
        "--markdown-path",
        str(markdown_path),
    ]

    assert runner.main(arguments) == 0
    assert runner.main([*arguments, "--check"]) == 0
    parser = runner.build_parser()
    assert parser.parse_args(arguments).population_size == 4
    assert json_path.exists() and markdown_path.exists()
    output = capsys.readouterr().out
    assert f"wrote {json_path}" in output
    assert f"checked {json_path}" in output
