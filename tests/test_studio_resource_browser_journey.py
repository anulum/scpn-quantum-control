# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — test studio browser journey
"""Public resource journey refuses external URLs before constructing a browser."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.studio_resource_browser_journey import run_resource_journey


@pytest.mark.parametrize(
    "url", ["http://example.com:4173/", "https://127.0.0.1:4173/", "http://127.0.0.1:4173/?q=1"]
)
def test_public_resource_journey_refuses_external_preview(url: str) -> None:
    """Preserve the original URL-owner refusal at the resource entry point."""
    with pytest.raises(ValueError):
        run_resource_journey(url)


def test_public_resource_refusal_without_browser_extra() -> None:
    """Reject unowned previews in a real interpreter without installed extras."""
    command = """
import importlib.util
import sys
sys.path.insert(0, sys.argv[1])
assert importlib.util.find_spec("playwright") is None
from tools.studio_resource_browser_journey import run_resource_journey
for url in ("http://example.com:4173/", "https://127.0.0.1:4173/", "http://127.0.0.1:4173/?q=1"):
    try:
        run_resource_journey(url)
    except ValueError:
        pass
    else:
        raise AssertionError("Unowned preview was admitted")
try:
    run_resource_journey("http://127.0.0.1:4173/")
except ModuleNotFoundError as error:
    assert error.name == "playwright"
else:
    raise AssertionError("Owned journey silently bypassed its browser dependency")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", command, str(Path(__file__).resolve().parents[1])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
