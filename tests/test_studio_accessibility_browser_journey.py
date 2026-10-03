# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — accessibility public runner ownership and engine custody
"""Refuse unsafe origins and altered auditors through the actual public runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.studio_accessibility_browser import run_accessibility_journey
from tools.studio_accessibility_checks import read_auditor
from tools.studio_browser_journey import main


def test_accessibility_public_runner_refuses_unowned_preview(tmp_path: Path) -> None:
    """Record a refused origin before starting the accessibility browser.

    Parameters
    ----------
    tmp_path
        Owned destination for the actual public command's refusal receipt.

    """
    output = tmp_path / "refused.json"
    assert (
        main(
            [
                "--scenario",
                "workbench_accessibility",
                "--base-url",
                "http://example.com:4173/",
                "--workspace-source-url",
                "http://127.0.0.1:4174/",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    evidence = json.loads(output.read_text())
    assert evidence["passed"] is False
    assert evidence["base_url"] == "rejected"
    assert "ValueError" in evidence["error"]


@pytest.mark.parametrize(
    "source",
    ["http://127.0.0.1:4173/", "http://127.0.0.1:4174/prefix/", "http://example.com:4174/"],
)
def test_accessibility_refuses_ambiguous_source_before_auditor(source: str) -> None:
    """Reject coincident, prefixed or external source origins before tool loading.

    Parameters
    ----------
    source
        Proposed source violating the original two-origin ownership boundary.

    """
    with pytest.raises(ValueError):
        run_accessibility_journey("http://127.0.0.1:4173/", source)


def test_accessibility_cannot_accept_an_altered_auditor(tmp_path: Path) -> None:
    """An altered auditor is refused before a browser can produce a false pass.

    Parameters
    ----------
    tmp_path
        Owned asset that deliberately attempts to substitute the audit engine.

    """
    altered = tmp_path / "axe.min.js"
    altered.write_text("window.axe = {version:'4.13.0',run:async()=>({violations:[]})};")
    with pytest.raises(ValueError, match="locked axe-core source"):
        read_auditor(altered)
    output = tmp_path / "auditor-refused.json"
    assert (
        main(
            [
                "--scenario",
                "workbench_accessibility",
                "--base-url",
                "http://127.0.0.1:4173/",
                "--workspace-source-url",
                "http://127.0.0.1:4174/",
                "--axe-source",
                str(altered),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "locked axe-core source" in json.loads(output.read_text())["error"]


def test_accessibility_missing_auditor_refuses_before_browser(tmp_path: Path) -> None:
    """Retain the real missing-dependency diagnostic rather than skipping an audit.

    Parameters
    ----------
    tmp_path
        Owned nonexistent auditor path and failed public command destination.

    """
    with pytest.raises(FileNotFoundError):
        read_auditor(tmp_path / "missing.js")
    output = tmp_path / "missing.json"
    assert (
        main(
            [
                "--scenario",
                "workbench_accessibility",
                "--base-url",
                "http://127.0.0.1:4173/",
                "--workspace-source-url",
                "http://127.0.0.1:4174/",
                "--axe-source",
                str(tmp_path / "missing.js"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "FileNotFoundError" in json.loads(output.read_text())["error"]


def test_auditor_option_cannot_change_an_unrelated_journey(tmp_path: Path) -> None:
    """Refuse an auditor supplied to a scenario that does not own this option.

    Parameters
    ----------
    tmp_path
        Owned public command evidence destination.

    """
    output = tmp_path / "unrelated.json"
    assert (
        main(
            [
                "--scenario",
                "capability_catalogue",
                "--base-url",
                "http://127.0.0.1:4173/",
                "--axe-source",
                str(tmp_path / "missing.js"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "valid only for workbench_accessibility" in json.loads(output.read_text())["error"]


def test_accessibility_requires_its_real_source_server(tmp_path: Path) -> None:
    """Do not drop native graph coverage when the source origin is omitted.

    Parameters
    ----------
    tmp_path
        Owned actual command refusal receipt.

    """
    output = tmp_path / "missing-source.json"
    assert (
        main(
            [
                "--scenario",
                "workbench_accessibility",
                "--base-url",
                "http://127.0.0.1:4173/",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "requires --workspace-source-url" in json.loads(output.read_text())["error"]
