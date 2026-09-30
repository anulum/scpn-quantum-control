# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — workspace browser URL boundary tests
"""The workspace public journey refuses either unsafe address before browser construction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.studio_workspace_browser_journey import run_panel_refusal_journey, run_workspace_journey


@pytest.mark.parametrize("url", ["http://example.com:4175/", "https://127.0.0.1:4175/"])
def test_panel_journey_refuses_external_source(url: str) -> None:
    """Reject an unowned damaged-source server before importing browser extras."""
    with pytest.raises(ValueError):
        run_panel_refusal_journey(url)


@pytest.mark.parametrize(
    "url", ["http://example.com:4173/", "https://127.0.0.1:4173/", "http://127.0.0.1:4173/?q=1"]
)
def test_workspace_journey_refuses_external_preview(url: str) -> None:
    """Require the same original URL owner on the built-preview input."""
    with pytest.raises(ValueError):
        run_workspace_journey(url, "http://127.0.0.1:4174/")


@pytest.mark.parametrize(
    "url", ["http://example.com:4174/", "https://127.0.0.1:4174/", "http://127.0.0.1:4174/#test"]
)
def test_workspace_journey_refuses_external_native_source(url: str) -> None:
    """Require explicit ownership of the source-API test origin as well."""
    with pytest.raises(ValueError):
        run_workspace_journey("http://127.0.0.1:4173/", url)


def test_workspace_journey_requires_distinct_owned_servers() -> None:
    """A deployed preview cannot masquerade as a native source API server."""
    with pytest.raises(ValueError, match="distinct"):
        run_workspace_journey("http://127.0.0.1:4173/", "http://127.0.0.1:4173/")


@pytest.mark.parametrize(
    ("preview", "source"),
    [
        ("http://example.com:4173/", "http://127.0.0.1:4174/"),
        ("http://127.0.0.1:4173/", "http://example.com:4174/"),
        ("http://127.0.0.1:4173/", "http://127.0.0.1:4173/"),
    ],
)
def test_workspace_url_refusal_without_installed_runtime(preview: str, source: str) -> None:
    """Refuse unsafe origins in the real repository without site packages.

    Parameters
    ----------
    preview
        Built-preview input requiring early public URL validation.
    source
        Source-server input requiring separate ownership and origin validation.

    """
    root = Path(__file__).resolve().parents[1]
    script = """import sys
sys.path.insert(0, sys.argv[1])
from tools.studio_workspace_browser_journey import run_workspace_journey
try:
    run_workspace_journey(sys.argv[2], sys.argv[3])
except ValueError:
    pass
else:
    raise AssertionError("Unsafe workspace origins admitted without a runtime")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root), preview, source],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
