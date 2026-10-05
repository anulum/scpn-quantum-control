# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — result browser source admission
"""Refuse unowned previews and qualify the genuine original CLI producer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.studio_browser_journey import main
from tools.studio_result_browser import analyse_export, run_result_journey


@pytest.mark.parametrize(
    "url", ["https://127.0.0.1:4173/", "http://example.com:4173/", "http://127.0.0.1:4173/?q=1"]
)
def test_result_preview_requires_owned_loopback(url: str) -> None:
    """Refuse ambiguous or external previews before native computation.

    Parameters
    ----------
    url
        Invalid preview authority.

    """
    with pytest.raises(ValueError):
        run_result_journey(url)


@pytest.mark.parametrize(
    "source",
    ["http://127.0.0.1:4173/", "http://127.0.0.1:4174/nested/", "http://example.com:4174/"],
)
def test_result_source_requires_distinct_owned_root(source: str) -> None:
    """Refuse a shared, nested or external source before importing a browser.

    Parameters
    ----------
    source
        Invalid native source authority.

    """
    with pytest.raises(ValueError):
        run_result_journey("http://127.0.0.1:4173/", source_url=source)


def test_result_browser_extra_is_required() -> None:
    """A genuine bare interpreter refuses missing native browser acceptance."""
    command = """
import importlib.util,sys
sys.path.insert(0,sys.argv[1])
assert importlib.util.find_spec('playwright') is None
from tools.studio_result_browser import run_result_journey
try:
    run_result_journey('http://127.0.0.1:4173/')
except ModuleNotFoundError as error:
    assert error.name == 'playwright'
else:
    raise AssertionError('Native browser acceptance bypassed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", command, str(Path(__file__).resolve().parents[1])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_original_cli_stationary_cloud_has_actual_nonuniform_thresholds() -> None:
    """Read the actual native CLI export and the independent constant-cloud oracle."""
    pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")
    record = json.loads(analyse_export())
    assert record["request"]["verb"] == "analyse"
    assert record["result"]["status"] == "succeeded"
    panels = record["result"]["outputs"]["inspection"]["panels"]
    assert [sample["coordinate"] for sample in panels[0]["samples"]] == [0.0, 0.125, 2.0]
    assert [sample["value"] for sample in panels[0]["samples"]] == [1, 1, 1]
    assert [sample["value"] for sample in panels[1]["samples"]] == [0, 0, 0]
    assert all(
        panel["coordinateUnit"] == "rad" and panel["valueDtype"] == "int64" for panel in panels
    )


def test_original_dispatcher_preserves_real_result_source_refusal(tmp_path: Path) -> None:
    """Record failed admission through the registered original public dispatcher.

    Parameters
    ----------
    tmp_path
        Current task-owned evidence destination.

    """
    output = tmp_path / "result-refused.json"
    assert (
        main(
            [
                "--scenario",
                "result_value_inspector",
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
    receipt = json.loads(output.read_text())
    assert receipt["scenario"] == "result_value_inspector"
    assert receipt["passed"] is False
    assert "distinct owned root" in receipt["error"]
