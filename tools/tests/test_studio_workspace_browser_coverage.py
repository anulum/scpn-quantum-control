# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real native coverage capture lifecycle
"""Exercise public CDP coverage capture in the dedicated owned browser cohort."""

from __future__ import annotations

import os
from collections.abc import Iterator
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Browser, Error, Route, expect, sync_playwright

from tools.studio_browser_journey import loopback_url
from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage


@pytest.fixture
def native_browser() -> Iterator[Browser]:
    """Own and close a genuine Chromium process for collector behavior.

    Yields
    ------
    Browser
        Real process, with every context owned by this test fixture.

    """
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            yield browser
        finally:
            browser.close()


def test_capture_actual_workspace_owners(native_browser: Browser) -> None:
    """Capture original scripts and counters from the real owned source host.

    Parameters
    ----------
    native_browser
        Real Chromium process whose test context is isolated from other cases.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Start the owned source host and supply STUDIO_WORKSPACE_SOURCE_URL")
    source = loopback_url(supplied)
    authority = urlsplit(source).netloc
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()

        def owned_request(route: Route) -> None:
            """Permit only requests to the original owned source authority."""
            parsed = urlsplit(route.request.url)
            if parsed.scheme == "http" and parsed.netloc == authority:
                route.continue_()
            else:
                route.abort()

        page.route("**/*", owned_request)
        session = start_native_coverage(page)
        page.goto(source + "browser-tests/workspace.html", wait_until="networkidle")
        records = take_native_coverage(session)
        actual: set[str] = set()
        for record in records:
            native = record["coverage"]
            assert isinstance(native, dict)
            url = native.get("url")
            assert isinstance(url, str)
            actual.add(urlsplit(url).path)
        assert actual == {
            "/src/shared/storage/workspaceStore.ts",
            "/src/shared/storage/workspaceArchive.ts",
            "/src/features/workspace/WorkspacePanel.tsx",
            "/src/features/workspace/useWorkspace.ts",
        }
    finally:
        context.close()


def test_capture_actual_workbench_navigation_owners(native_browser: Browser) -> None:
    """Retain all fourteen real source owners after public lazy-route navigation.

    Parameters
    ----------
    native_browser
        Owned Chromium process exercising the original facade and source controllers.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Start the owned source host and supply STUDIO_WORKSPACE_SOURCE_URL")
    source = loopback_url(supplied)
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        session = start_native_coverage(page)
        page.goto(source + "browser-tests/workbench.html", wait_until="networkidle")
        nav = page.get_by_role("navigation", name="Workbench views")
        for view in ("Build", "Results", "Atlas"):
            nav.get_by_role("link", name=view, exact=True).click()
            expect(
                page.get_by_role(
                    "heading", name=view if view != "Atlas" else "Atlas unavailable", exact=True
                )
            ).to_be_visible()
        records = take_native_coverage(session, include_workbench=True)
        actual: set[str] = set()
        for record in records:
            native = record["coverage"]
            assert isinstance(native, dict)
            url = native.get("url")
            assert isinstance(url, str)
            actual.add(urlsplit(url).path)
        assert actual == {
            "/src/shared/storage/workspaceStore.ts",
            "/src/shared/storage/workspaceArchive.ts",
            "/src/features/workspace/WorkspacePanel.tsx",
            "/src/features/workspace/useWorkspace.ts",
            "/src/QuantumStudioPanel.tsx",
            "/src/features/catalogue/CapabilityCatalogue.tsx",
            "/src/app/Workbench.tsx",
            "/src/app/WorkbenchInspector.tsx",
            "/src/app/RouteBoundary.tsx",
            "/src/app/routing.ts",
            "/src/app/useWorkbenchRoute.ts",
            "/src/app/routes/BuildView.tsx",
            "/src/app/routes/ResultsView.tsx",
            "/src/app/routes/UnavailableView.tsx",
        }
    finally:
        context.close()


def test_missing_owners_stops_real_profiler(native_browser: Browser) -> None:
    """Refuse an actual empty page without leaving precise coverage active.

    Parameters
    ----------
    native_browser
        Real Chromium process used for a new empty test context.

    """
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        session = start_native_coverage(page)
        records: list[dict[str, object]] = []
        with pytest.raises(AssertionError, match="Missing current-page native owners"):
            take_native_coverage(session, records)
        assert records == []
        assert not page.is_closed()
        with pytest.raises(Error):
            session.send("Profiler.takePreciseCoverage")
    finally:
        context.close()


def test_closed_page_preserves_capture_and_cleanup_failures(native_browser: Browser) -> None:
    """Retain both actual protocol failures when the profiled page closes.

    Parameters
    ----------
    native_browser
        Real Chromium process with an owned page that closes before capture.

    """
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        session = start_native_coverage(page)
        page.close()
        records: list[dict[str, object]] = []
        with pytest.raises(RuntimeError, match="Native coverage capture failed") as failure:
            take_native_coverage(session, records)
        assert "profiler stop also failed" in str(failure.value)
        assert isinstance(failure.value.__cause__, Error)
        assert records == []
    finally:
        context.close()


def test_disposal_after_capture_preserves_records_and_stop_error(native_browser: Browser) -> None:
    """Keep actual counters when their recipient closes the page before profiler cleanup.

    Parameters
    ----------
    native_browser
        Real Chromium process used for an independently disposed page.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Start the owned source host and supply STUDIO_WORKSPACE_SOURCE_URL")
    source = loopback_url(supplied)
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        session = start_native_coverage(page)
        page.goto(source + "browser-tests/workspace.html", wait_until="networkidle")

        class DisposingRecords(list[dict[str, object]]):
            """Dispose the real page after receiving its four original owners."""

            def append(self, record: dict[str, object]) -> None:
                """Retain the original SDK record before initiating actual disposal.

                Parameters
                ----------
                record
                    Original counter/source record produced by the real profiler.

                """
                super().append(record)
                if len(self) == 4:
                    page.close()

        records = DisposingRecords()
        with pytest.raises(Error, match="closed"):
            take_native_coverage(session, records)
        assert len(records) == 4
        assert page.is_closed()
        assert all(isinstance(record["coverage"], dict) for record in records)
    finally:
        context.close()
