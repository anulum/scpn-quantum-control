# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original comparison browser authority
"""Qualify original shared CLI source refusal and bounded browser authority."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.studio_browser_journey import main
from tools.studio_comparison_browser import run_comparison_journey


@pytest.mark.parametrize(
    "url", ["https://127.0.0.1:4173/", "http://example.com:4173/", "http://127.0.0.1:4173/?q=1"]
)
def test_comparison_requires_owned_loopback(url: str) -> None:
    """Refuse external or ambiguous authority before loading a browser.

    Parameters
    ----------
    url
        Unsupported preview authority.

    """
    with pytest.raises(ValueError):
        run_comparison_journey(url)


@pytest.mark.parametrize(
    "source",
    ["http://127.0.0.1:4173/", "http://127.0.0.1:4174/nested/", "http://example.com:4174/"],
)
def test_comparison_source_requires_distinct_owned_root(source: str) -> None:
    """Refuse a shared, nested or external source before computation.

    Parameters
    ----------
    source
        Unsupported original source authority.

    """
    with pytest.raises(ValueError):
        run_comparison_journey("http://127.0.0.1:4173/", source_url=source)


def test_comparison_browser_extra_is_required() -> None:
    """A genuine bare interpreter refuses absent native browser capability."""
    command = """
import importlib.util,sys
sys.path.insert(0,sys.argv[1])
assert importlib.util.find_spec('playwright') is None
from tools.studio_comparison_browser import run_comparison_journey
try:
    run_comparison_journey('http://127.0.0.1:4173/')
except ModuleNotFoundError as error:
    assert error.name=='playwright'
else:
    raise AssertionError('Native browser acceptance bypassed')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-S",
            "-c",
            command,
            str(Path(__file__).resolve().parents[1]),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_registered_comparison_cli_records_real_source_refusal(tmp_path: Path) -> None:
    """Retain actual failed authority admission through the public shared CLI.

    Parameters
    ----------
    tmp_path
        Actual task-owned current evidence destination.

    """
    output = tmp_path / "comparison-refused.json"
    assert (
        main(
            [
                "--scenario",
                "immutable_run_comparison",
                "--base-url",
                "http://127.0.0.1:4173/",
                "--workspace-source-url",
                "http://127.0.0.1:4173/",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    evidence = json.loads(output.read_text())
    assert evidence["scenario"] == "immutable_run_comparison"
    assert evidence["passed"] is False
    assert "distinct owned root" in evidence["error"]
