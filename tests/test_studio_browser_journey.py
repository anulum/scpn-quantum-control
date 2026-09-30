# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — test studio browser journey
"""The public browser runner refuses external navigation before loading a browser."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:4173/",
        "http://example.com:4173/",
        "http://192.168.1.1:4173/",
        "http://127.0.0.1:4173/?q=1",
        "http://127.0.0.1:4173/#route",
        "http://name:secret@127.0.0.1:4173/",
        "http://127.0.0.1/",
        "http://127.0.0.1:0/",
        "http://127.0.0.1:65536/",
        "file:///tmp/index.html",
    ],
)
@pytest.mark.parametrize(
    "scenario",
    [
        "capability_catalogue",
        "evidence_inspector",
        "resource_plan_projection",
        "workspace_recovery",
        "workspace_panel_refusal",
    ],
)
def test_runner_rejects_external_or_ambiguous_preview(
    url: str, tmp_path: Path, scenario: str
) -> None:
    """Invalid preview addresses yield failed evidence without browser execution."""
    output = tmp_path / "refused.json"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/studio_browser_journey.py",
            "--scenario",
            scenario,
            "--base-url",
            url,
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 1, completed.stderr
    evidence = json.loads(output.read_text())
    assert evidence["passed"] is False
    assert "ValueError" in evidence["error"]


@pytest.mark.parametrize("scenario", ["workspace_recovery", "workspace_panel_refusal"])
def test_public_dispatch_requires_matching_source_option(tmp_path: Path, scenario: str) -> None:
    """Refuse a missing or misapplied source address through the public dispatcher."""
    from tools.studio_browser_journey import main

    output = tmp_path / "source-option-refused.json"
    argv = [
        "--scenario",
        scenario,
        "--base-url",
        "http://127.0.0.1:4173/",
        "--output",
        str(output),
    ]
    if scenario == "workspace_panel_refusal":
        argv.extend(["--workspace-source-url", "http://127.0.0.1:4174/"])
    assert main(argv) == 1
    evidence = json.loads(output.read_text())
    assert evidence["passed"] is False
    assert "ValueError" in evidence["error"]
    assert "workspace-source-url" in evidence["error"]


@pytest.mark.parametrize(
    "url", ["https://127.0.0.1:4173/", "http://example.com:4173/", "http://127.0.0.1:4173/?q=1"]
)
def test_public_dispatch_records_url_refusal_before_browser_import(
    tmp_path: Path, url: str
) -> None:
    """Retain the refusal through the public dispatcher without constructing a browser."""
    from tools.studio_browser_journey import main

    output = tmp_path / "unsafe-source.json"
    assert (
        main(["--scenario", "workspace_panel_refusal", "--base-url", url, "--output", str(output)])
        == 1
    )
    evidence = json.loads(output.read_text())
    assert evidence["passed"] is False
    assert evidence["base_url"] == "rejected"
    assert "ValueError" in evidence["error"]
