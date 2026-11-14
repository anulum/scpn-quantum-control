# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — owned-preview request custody tests
"""Check every decision of the shared preview route guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest

from tools.studio_preview_custody import preview_route_guard

if TYPE_CHECKING:
    from playwright.sync_api import Route


@dataclass
class _Request:
    """Request fields the guard reads."""

    url: str
    method: str


@dataclass
class _Route:
    """Route double that records the guard's single decision."""

    request: _Request
    decisions: list[str] = field(default_factory=list)

    def abort(self) -> None:
        """Record an abort."""
        self.decisions.append("abort")

    def continue_(self) -> None:
        """Record a continuation."""
        self.decisions.append("continue")


@pytest.mark.parametrize(
    ("url", "method", "decision", "external", "submissions"),
    [
        ("http://127.0.0.1:4173/index.html", "GET", "continue", [], []),
        ("http://example.com/x.js", "GET", "abort", ["outside-preview"], []),
        ("https://127.0.0.1:4173/", "GET", "abort", ["outside-preview"], []),
        ("http://127.0.0.1:4174/", "GET", "abort", ["outside-preview"], []),
        ("http://127.0.0.1:4173/api", "POST", "abort", [], ["POST"]),
    ],
)
def test_each_request_is_continued_or_refused_and_recorded(
    url: str, method: str, decision: str, external: list[str], submissions: list[str]
) -> None:
    """Outside requests and submissions are aborted and recorded; owned reads continue.

    Parameters
    ----------
    url
        Requested URL.
    method
        Request method.
    decision
        Expected guard decision.
    external
        Expected outside-request record.
    submissions
        Expected submission record.

    """
    seen_external: list[str] = []
    seen_submissions: list[str] = []
    guard = preview_route_guard("127.0.0.1:4173", seen_external, seen_submissions)
    route = _Route(_Request(url, method))
    guard(cast("Route", route))
    assert route.decisions == [decision]
    assert (seen_external, seen_submissions) == (external, submissions)


def test_kernel_paths_are_recorded_only_when_requested() -> None:
    """A guard with a kernel record lists each loaded ``.wasm`` path; others list nothing."""
    kernels: list[str] = []
    guard = preview_route_guard("127.0.0.1:4173", [], [], kernels)
    for path in ("/wasm/kernel.wasm", "/assets/app.js"):
        route = _Route(_Request(f"http://127.0.0.1:4173{path}", "GET"))
        guard(cast("Route", route))
        assert route.decisions == ["continue"]
    assert kernels == ["/wasm/kernel.wasm"]
    refused = _Route(_Request("http://127.0.0.1:4173/wasm/kernel.wasm", "PUT"))
    guard(cast("Route", refused))
    assert kernels == ["/wasm/kernel.wasm"]
