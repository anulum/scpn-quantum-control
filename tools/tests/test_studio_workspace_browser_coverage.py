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
from pathlib import Path
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


@pytest.mark.parametrize(("include_panel", "include_workbench"), [(True, False), (False, True)])
def test_capture_actual_workbench_navigation_owners(
    native_browser: Browser, include_panel: bool, include_workbench: bool
) -> None:
    """Retain the requested facade or complete navigation cohort after real navigation.

    Parameters
    ----------
    native_browser
        Owned Chromium process exercising the original facade and source controllers.
    include_panel
        Request the original facade and four workspace owners.
    include_workbench
        Request the complete fourteen-owner navigation cohort.

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
        records = take_native_coverage(
            session, include_panel=include_panel, include_workbench=include_workbench
        )
        actual: set[str] = set()
        for record in records:
            native = record["coverage"]
            assert isinstance(native, dict)
            url = native.get("url")
            assert isinstance(url, str)
            actual.add(urlsplit(url).path)
        expected = {
            "/src/shared/storage/workspaceStore.ts",
            "/src/shared/storage/workspaceArchive.ts",
            "/src/features/workspace/WorkspacePanel.tsx",
            "/src/features/workspace/useWorkspace.ts",
            "/src/QuantumStudioPanel.tsx",
        }
        if include_workbench:
            expected.update(
                {
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
            )
        assert actual == expected
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


def test_capture_actual_parameter_editor_owners(native_browser: Browser) -> None:
    """Retain the four linked editors and their four original workspace owners.

    Parameters
    ----------
    native_browser
        Real Chromium process importing a fully admitted signed matrix archive.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host")
    source = loopback_url(supplied)
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        session = start_native_coverage(page)
        page.goto(source + "browser-tests/parameterEditor.html", wait_until="networkidle")
        archive = page.evaluate(
            "async corpus => (await import('/browser-tests/parameterEditor.native.tsx'))"
            ".createParameterConformanceArchive(corpus)",
            (
                Path(__file__).resolve().parents[2] / "tests/data/studio_workspace/documents.json"
            ).read_text(),
        )
        page.get_by_label("Workspace archive JSON", exact=True).fill(archive["json"])
        page.get_by_role("button", name="Preview archive", exact=True).click()
        expect(page.get_by_role("region", name="Linked parameter editor")).to_be_visible()
        records = take_native_coverage(session, include_parameters=True)
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
            "/src/features/parameters/parameterDraft.ts",
            "/src/features/parameters/parameterRevision.ts",
            "/src/features/parameters/ParameterEditor.tsx",
            "/src/features/parameters/ParameterWorkspace.tsx",
        }
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


def test_parameter_cohort_refuses_unavailable_editor_source(native_browser: Browser) -> None:
    """Refuse a real editor cohort whose revision source request was interrupted.

    Parameters
    ----------
    native_browser
        Real Chromium process experiencing an unavailable original revision module.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host")
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        interrupted: list[str] = []

        def unavailable_revision(route: Route) -> None:
            """Interrupt the actual original module request and retain its identity."""
            interrupted.append(route.request.url)
            route.abort()

        page.route("**/src/features/parameters/parameterRevision.ts", unavailable_revision)
        session = start_native_coverage(page)
        page.goto(
            loopback_url(supplied) + "browser-tests/workspace.html", wait_until="networkidle"
        )
        records: list[dict[str, object]] = []
        with pytest.raises(AssertionError, match="Missing current-page native owners"):
            take_native_coverage(session, records, include_parameters=True)
        assert len(interrupted) == 1
        assert all(
            isinstance(record["coverage"], dict)
            and not str(record["coverage"].get("url", "")).endswith("/parameterRevision.ts")
            for record in records
        )
        with pytest.raises(Error):
            session.send("Profiler.takePreciseCoverage")
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
