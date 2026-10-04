# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — local experiment browser admission
"""Reject unowned experiment previews before importing the native browser."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.studio_browser_journey import main
from tools.studio_local_experiment_browser import run_local_experiment_journey


@pytest.mark.parametrize(
    "url", ["https://127.0.0.1:4173/", "http://example.com:4173/", "http://127.0.0.1:4173/?q=1"]
)
def test_experiment_preview_requires_owned_loopback(url: str) -> None:
    """Reject external or ambiguous preview addresses before numerical work.

    Parameters
    ----------
    url
        Deliberately inadmissible preview address.

    """
    with pytest.raises(ValueError):
        run_local_experiment_journey(url)


@pytest.mark.parametrize(
    "source",
    ["http://127.0.0.1:4173/", "http://127.0.0.1:4174/nested/", "http://example.com:4174/"],
)
def test_source_counters_require_distinct_owned_root(source: str) -> None:
    """Refuse source counter authority that is shared, nested or external.

    Parameters
    ----------
    source
        Invalid original-source host declaration.

    """
    with pytest.raises(ValueError):
        run_local_experiment_journey("http://127.0.0.1:4173/", source_url=source)


def test_experiment_journey_requires_actual_browser_extra() -> None:
    """A real bare interpreter refuses rather than skipping native acceptance."""
    command = """
import importlib.util,sys
sys.path.insert(0,sys.argv[1])
assert importlib.util.find_spec('playwright') is None
from tools.studio_local_experiment_browser import run_local_experiment_journey
try:
    run_local_experiment_journey('http://127.0.0.1:4173/')
except ModuleNotFoundError as error:
    assert error.name == 'playwright'
else:
    raise AssertionError('Native browser acceptance was bypassed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", command, str(Path(__file__).resolve().parents[1])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_shared_dispatcher_retains_native_source_refusal(tmp_path: Path) -> None:
    """Use the actual original dispatcher and retain its invalid source receipt.

    Parameters
    ----------
    tmp_path
        Exact task-owned pytest output directory.

    """
    output = tmp_path / "experiment-refused.json"
    assert (
        main(
            [
                "--scenario",
                "local_experiment_journey",
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
    observed = json.loads(output.read_text())
    assert observed["scenario"] == "local_experiment_journey"
    assert observed["passed"] is False
    assert "distinct owned root" in observed["error"]
