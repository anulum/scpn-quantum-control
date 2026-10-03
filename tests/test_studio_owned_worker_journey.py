# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — owned browser journey public admission
"""Reject unowned worker previews before loading optional browser dependencies."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.studio_owned_worker_journey import run_owned_worker_journey


@pytest.mark.parametrize(
    "url", ["https://127.0.0.1:4173/", "http://example.com:4173/", "http://127.0.0.1:4173/?q=1"]
)
def test_owned_worker_journey_rejects_unowned_preview(url: str) -> None:
    """Unsafe addresses fail at the original public URL owner."""
    with pytest.raises(ValueError):
        run_owned_worker_journey(url)


def test_owned_worker_refusal_without_browser_extra() -> None:
    """A real bare interpreter retains refusal without silently bypassing Chromium."""
    command = """
import importlib.util, sys
sys.path.insert(0, sys.argv[1])
assert importlib.util.find_spec('playwright') is None
from tools.studio_owned_worker_journey import run_owned_worker_journey
try:
    run_owned_worker_journey('http://127.0.0.1:4173/')
except ModuleNotFoundError as error:
    assert error.name == 'playwright'
else:
    raise AssertionError('Owned kernel journey bypassed the native browser dependency')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", command, str(Path(__file__).resolve().parents[1])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_shared_worker_dispatch_rejects_misapplied_source_option(tmp_path: Path) -> None:
    """The original browser dispatcher does not admit an unrelated source server."""
    from tools.studio_browser_journey import main

    output = tmp_path / "refused.json"
    assert (
        main(
            [
                "--scenario",
                "owned_kernel_worker",
                "--base-url",
                "http://127.0.0.1:4173/",
                "--workspace-source-url",
                "http://127.0.0.1:4174/",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "workspace-source-url" in json.loads(output.read_text())["error"]
