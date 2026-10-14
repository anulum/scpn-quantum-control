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
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        try:
            yield browser
        finally:
            browser.close()


def test_capture_actual_workspace_owners(native_browser: Browser) -> None:
    """Capture original scripts and counters from the real owned source host.

    Parameters
    ----------
    native_browser : Browser
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
        assert {
            "/src/shared/storage/workspaceStore.ts",
            "/src/shared/storage/workspaceArchive.ts",
            "/src/shared/workspace/WorkspaceEditor.tsx",
            "/src/WorkspacePanel.tsx",
            "/src/shared/workspace/useWorkspace.ts",
        } <= actual
        assert {
            "/src/app/moduleRegistry.tsx",
            "/src/shared/contracts/module.ts",
            "/src/shared/moduleViews/ModuleDocumentViews.tsx",
            "/src/shared/browser/ScrollRegion.tsx",
        } <= actual
    finally:
        context.close()


def test_capture_actual_workflow_owners(native_browser: Browser) -> None:
    """Capture all original graph, editor, execution and storage scripts on Experiments.

    Parameters
    ----------
    native_browser : Browser
        Real Chromium process using the actual production workflow facade.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host")
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        session = start_native_coverage(page)
        page.goto(loopback_url(supplied) + "#/experiments", wait_until="networkidle")
        expect(
            page.get_by_role("region", name="Reproducible workflows", exact=True)
        ).to_be_visible()
        records = take_native_coverage(session, include_workflows=True)
        actual = {
            urlsplit(str(record["coverage"]["url"])).path
            for record in records
            if isinstance(record["coverage"], dict)
        }
        assert {
            "/src/shared/storage/workspaceStore.ts",
            "/src/shared/storage/workspaceArchive.ts",
            "/src/shared/workspace/WorkspaceEditor.tsx",
            "/src/shared/workspace/useWorkspace.ts",
            "/src/app/Workbench.tsx",
            "/src/shared/workflows/workflowModel.ts",
            "/src/shared/workflows/workflowSweep.ts",
            "/src/shared/workflows/workflowJournal.ts",
            "/src/shared/workflows/workflowArchive.ts",
            "/src/shared/workflows/workflowExecution.ts",
            "/src/shared/workflows/useWorkflowRun.ts",
            "/src/modules/workflows/WorkflowEditor.tsx",
            "/src/modules/workflows/WorkflowRunner.tsx",
        } <= actual
        assert {
            "/src/app/moduleRegistry.tsx",
            "/src/shared/contracts/module.ts",
            "/src/shared/moduleViews/ModuleDocumentViews.tsx",
            "/src/shared/browser/ScrollRegion.tsx",
        } <= actual
    finally:
        context.close()


def test_workflow_capture_refuses_interrupted_original_owner(native_browser: Browser) -> None:
    """Retain collected counters and stop profiling after a real workflow module fails.

    Parameters
    ----------
    native_browser : Browser
        Real Chromium process with one unavailable original workflow source request.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host")
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        interrupted: list[str] = []

        def unavailable_workflow(route: Route) -> None:
            """Retain and interrupt the actual original workflow execution module request."""
            interrupted.append(route.request.url)
            route.abort()

        page.route("**/src/shared/workflows/workflowExecution.ts", unavailable_workflow)
        session = start_native_coverage(page)
        page.goto(loopback_url(supplied) + "#/experiments", wait_until="networkidle")
        records: list[dict[str, object]] = []
        with pytest.raises(AssertionError, match="Missing current-page native owners"):
            take_native_coverage(session, records, include_workflows=True)
        assert len(interrupted) == 1
        assert all(
            isinstance(record["coverage"], dict)
            and not str(record["coverage"].get("url", "")).endswith("/workflowExecution.ts")
            for record in records
        )
        with pytest.raises(Error):
            session.send("Profiler.takePreciseCoverage")
    finally:
        context.close()


def test_capture_actual_experiment_owners(native_browser: Browser) -> None:
    """Collect original experiment and compiled shell owners after real navigation.

    Parameters
    ----------
    native_browser : Browser
        Real Chromium process whose page imports the production experiment controls.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host")
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        session = start_native_coverage(page)
        page.goto(
            loopback_url(supplied) + "browser-tests/workbench.html", wait_until="networkidle"
        )
        page.get_by_role("navigation", name="Workbench views").get_by_role(
            "link", name="Experiments", exact=True
        ).click()
        expect(page.get_by_role("heading", name="Local experiment", exact=True)).to_be_visible()
        records = take_native_coverage(session, include_experiments=True)
        actual: set[str] = set()
        for record in records:
            native = record["coverage"]
            assert isinstance(native, dict)
            url = native.get("url")
            assert isinstance(url, str)
            actual.add(urlsplit(url).path)
        assert {
            "/src/shared/storage/workspaceStore.ts",
            "/src/shared/storage/workspaceArchive.ts",
            "/src/shared/workspace/WorkspaceEditor.tsx",
            "/src/shared/workspace/useWorkspace.ts",
            "/src/app/Workbench.tsx",
            "/src/shared/experiments/kuramotoArtifacts.ts",
            "/src/shared/experiments/experimentArchive.ts",
            "/src/shared/experiments/experimentPlan.ts",
            "/src/shared/experiments/useExperimentRun.ts",
            "/src/modules/experiments/ExperimentRunner.tsx",
        } <= actual
        assert {
            "/src/app/moduleRegistry.tsx",
            "/src/shared/contracts/module.ts",
            "/src/shared/moduleViews/ModuleDocumentViews.tsx",
            "/src/shared/browser/ScrollRegion.tsx",
        } <= actual
    finally:
        context.close()


def test_experiment_capture_refuses_interrupted_original_owner(native_browser: Browser) -> None:
    """Preserve actual collected scripts when an original experiment import is unavailable.

    Parameters
    ----------
    native_browser : Browser
        Real Chromium process with one deliberately aborted source-module request.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host")
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        interrupted: list[str] = []

        def unavailable_experiment(route: Route) -> None:
            """Abort the real request and retain its actual source URL."""
            if urlsplit(route.request.url).path == "/src/modules/experiments/ExperimentRunner.tsx":
                interrupted.append(route.request.url)
                route.abort()
            else:
                route.continue_()

        page.route("**/*", unavailable_experiment)
        session = start_native_coverage(page)
        page.goto(
            loopback_url(supplied) + "browser-tests/workbench.html", wait_until="networkidle"
        )
        page.get_by_role("navigation", name="Workbench views").get_by_role(
            "link", name="Experiments", exact=True
        ).click()
        expect(page.get_by_role("alert")).to_contain_text("unavailable")
        records: list[dict[str, object]] = []
        with pytest.raises(AssertionError, match="Missing current-page native owners"):
            take_native_coverage(session, records, include_experiments=True)
        assert len(interrupted) == 1
        assert all(
            isinstance(record["coverage"], dict)
            and not str(record["coverage"].get("url", "")).endswith("/ExperimentRunner.tsx")
            for record in records
        )
        with pytest.raises(Error):
            session.send("Profiler.takePreciseCoverage")
    finally:
        context.close()


@pytest.mark.parametrize(("include_panel", "include_workbench"), [(True, False), (False, True)])
def test_capture_actual_workbench_navigation_owners(
    native_browser: Browser, include_panel: bool, include_workbench: bool
) -> None:
    """Retain the requested facade or complete navigation cohort after real navigation.

    Parameters
    ----------
    native_browser : Browser
        Owned Chromium process exercising the original facade and source controllers.
    include_panel : bool
        Request the original facade and its compiled shell ownership.
    include_workbench : bool
        Request the original navigation and complete compiled shell cohort.

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
            "/src/shared/workspace/WorkspaceEditor.tsx",
            "/src/shared/workspace/useWorkspace.ts",
            "/src/QuantumStudioPanel.tsx",
        }
        if include_workbench:
            expected.update(
                {
                    "/src/modules/catalogue/CapabilityCatalogue.tsx",
                    "/src/app/Workbench.tsx",
                    "/src/app/WorkbenchInspector.tsx",
                    "/src/app/RouteBoundary.tsx",
                    "/src/app/routing.ts",
                    "/src/app/useWorkbenchRoute.ts",
                    "/src/modules/build/BuildView.tsx",
                    "/src/modules/results/ResultsView.tsx",
                    "/src/modules/atlas/UnavailableView.tsx",
                }
            )
        assert expected <= actual
        assert "/src/app/moduleRegistry.tsx" in actual
        assert "/src/shared/moduleViews/ModuleJobDock.tsx" in actual
    finally:
        context.close()


def test_missing_owners_stops_real_profiler(native_browser: Browser) -> None:
    """Refuse an actual empty page without leaving precise coverage active.

    Parameters
    ----------
    native_browser : Browser
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


def test_missing_compiled_entry_stops_real_profiler(native_browser: Browser) -> None:
    """Require a real compiled entry and retain partial counters after its request fails.

    Parameters
    ----------
    native_browser : Browser
        Owned Chromium process whose source request is deliberately unavailable.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host")
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        interrupted: list[str] = []

        def unavailable_entry(route: Route) -> None:
            """Interrupt the actual original compiled compiler entry request."""
            interrupted.append(route.request.url)
            route.abort()

        page.route("**/src/modules/compiler/index.tsx", unavailable_entry)
        session = start_native_coverage(page)
        page.goto(
            loopback_url(supplied) + "browser-tests/workspace.html", wait_until="networkidle"
        )
        records: list[dict[str, object]] = []
        with pytest.raises(AssertionError, match="Missing current-page native owners"):
            take_native_coverage(session, records)
        assert len(interrupted) == 1
        assert all(
            isinstance(record["coverage"], dict)
            and urlsplit(str(record["coverage"].get("url", ""))).path
            != "/src/modules/compiler/index.tsx"
            for record in records
        )
        with pytest.raises(Error):
            session.send("Profiler.takePreciseCoverage")
    finally:
        context.close()


def test_capture_actual_parameter_editor_owners(native_browser: Browser) -> None:
    """Retain original linked editing and complete compiled workspace composition.

    Parameters
    ----------
    native_browser : Browser
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
        assert {
            "/src/shared/storage/workspaceStore.ts",
            "/src/shared/storage/workspaceArchive.ts",
            "/src/shared/workspace/WorkspaceEditor.tsx",
            "/src/WorkspacePanel.tsx",
            "/src/shared/workspace/useWorkspace.ts",
            "/src/shared/parameters/parameterDraft.ts",
            "/src/shared/parameters/parameterRevision.ts",
            "/src/modules/parameters/ParameterEditor.tsx",
            "/src/modules/parameters/ParameterWorkspace.tsx",
        } <= actual
        assert {
            "/src/app/moduleRegistry.tsx",
            "/src/shared/contracts/module.ts",
            "/src/shared/moduleViews/ModuleDocumentViews.tsx",
            "/src/shared/browser/ScrollRegion.tsx",
        } <= actual
    finally:
        context.close()


def test_closed_page_preserves_capture_and_cleanup_failures(native_browser: Browser) -> None:
    """Retain both actual protocol failures when the profiled page closes.

    Parameters
    ----------
    native_browser : Browser
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


def test_reused_page_session_captures_original_sources_after_reload(
    native_browser: Browser,
) -> None:
    """Restart actual profiling on the same page session and retain both intervals.

    Parameters
    ----------
    native_browser : Browser
        Real Chromium process whose owned page reloads between native captures.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host")
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        session = start_native_coverage(page)
        page.goto(
            loopback_url(supplied) + "browser-tests/workspace.html", wait_until="networkidle"
        )
        first = take_native_coverage(session)
        resumed = start_native_coverage(page, session=session)
        assert resumed is session
        page.reload(wait_until="networkidle")
        second = take_native_coverage(resumed)
        assert {(record["code_sha256"], record["source_sha256"]) for record in first} == {
            (record["code_sha256"], record["source_sha256"]) for record in second
        }
        assert first and second
        with pytest.raises(Error):
            session.send("Profiler.takePreciseCoverage")
    finally:
        context.close()


def test_parameter_cohort_refuses_unavailable_editor_source(native_browser: Browser) -> None:
    """Refuse a real editor cohort whose revision source request was interrupted.

    Parameters
    ----------
    native_browser : Browser
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

        page.route("**/src/shared/parameters/parameterRevision.ts", unavailable_revision)
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
    """Keep every captured source when its recipient closes the page before cleanup.

    Parameters
    ----------
    native_browser : Browser
        Real Chromium process used for an independently disposed page.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Start the owned source host and supply STUDIO_WORKSPACE_SOURCE_URL")
    source = loopback_url(supplied)
    context = native_browser.new_context(service_workers="block")
    try:
        reference_page = context.new_page()
        reference_session = start_native_coverage(reference_page)
        reference_page.goto(source + "browser-tests/workspace.html", wait_until="networkidle")
        reference = take_native_coverage(reference_session)
        reference_page.close()
        page = context.new_page()
        session = start_native_coverage(page)
        page.goto(source + "browser-tests/workspace.html", wait_until="networkidle")

        class DisposingRecords(list[dict[str, object]]):
            """Dispose the real page after collecting its original compiled workspace owners."""

            def append(self, record: dict[str, object]) -> None:
                """Retain the original SDK record before initiating actual disposal.

                Parameters
                ----------
                record : dict[str, object]
                    Original counter/source record produced by the real profiler.

                """
                super().append(record)
                if len(self) == 4:
                    page.close()

        records = DisposingRecords()
        with pytest.raises(Error, match="closed"):
            take_native_coverage(session, records)
        assert len(records) == len(reference) > 4
        assert page.is_closed()
        assert all(isinstance(record["coverage"], dict) for record in records)
        assert {(record["code_sha256"], record["source_sha256"]) for record in records} == {
            (record["code_sha256"], record["source_sha256"]) for record in reference
        }
    finally:
        context.close()


def test_capture_actual_result_owners(native_browser: Browser) -> None:
    """Retain every result, storage and compiled owner through actual Results navigation.

    Parameters
    ----------
    native_browser : Browser
        Owned Chromium process using the original production Results route.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host")
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        session = start_native_coverage(page)
        page.goto(loopback_url(supplied) + "#/results", wait_until="networkidle")
        expect(page.get_by_role("region", name="Source result inspector")).to_be_visible()
        records = take_native_coverage(session, include_results=True)
        actual: set[str] = set()
        for record in records:
            native = record["coverage"]
            assert isinstance(native, dict)
            url = native.get("url")
            assert isinstance(url, str)
            actual.add(urlsplit(url).path)
        assert {
            "/src/shared/storage/workspaceStore.ts",
            "/src/shared/storage/workspaceArchive.ts",
            "/src/shared/workspace/WorkspaceEditor.tsx",
            "/src/shared/workspace/useWorkspace.ts",
            "/src/app/Workbench.tsx",
            "/src/modules/results/ResultsView.tsx",
            "/src/shared/results/resultModel.ts",
            "/src/shared/results/resultSources.ts",
            "/src/modules/results/resultExport.ts",
            "/src/modules/results/ResultInspector.tsx",
            "/src/modules/results/ResultLoader.tsx",
        } <= actual
        assert {
            "/src/app/moduleRegistry.tsx",
            "/src/shared/contracts/module.ts",
            "/src/shared/moduleViews/ModuleDocumentViews.tsx",
            "/src/shared/browser/ScrollRegion.tsx",
        } <= actual
    finally:
        context.close()


def test_result_capture_refuses_interrupted_original_owner(native_browser: Browser) -> None:
    """Preserve native records and stop profiling when an original result module fails.

    Parameters
    ----------
    native_browser : Browser
        Owned Chromium process with one genuinely unavailable source-module request.

    """
    supplied = os.environ.get("STUDIO_WORKSPACE_SOURCE_URL")
    if supplied is None:
        raise RuntimeError("Supply the owned source host")
    context = native_browser.new_context(service_workers="block")
    try:
        page = context.new_page()
        interrupted: list[str] = []

        def unavailable_result(route: Route) -> None:
            """Retain and interrupt the actual original result module request."""
            interrupted.append(route.request.url)
            route.abort()

        page.route("**/src/shared/results/resultModel.ts", unavailable_result)
        session = start_native_coverage(page)
        page.goto(loopback_url(supplied) + "#/results", wait_until="networkidle")
        expect(page.get_by_role("alert")).to_contain_text("View unavailable")
        records: list[dict[str, object]] = []
        with pytest.raises(AssertionError, match="Missing current-page native owners"):
            take_native_coverage(session, records, include_results=True)
        assert len(interrupted) == 1
        assert records
        assert all(
            isinstance(record["coverage"], dict)
            and not str(record["coverage"].get("url", "")).endswith("/resultModel.ts")
            for record in records
        )
        with pytest.raises(Error):
            session.send("Profiler.takePreciseCoverage")
    finally:
        context.close()
