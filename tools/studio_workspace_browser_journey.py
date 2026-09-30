# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — browser workspace recovery journey
"""Exercise actual workspace UI and native IndexedDB in isolated owned contexts."""

from __future__ import annotations

from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext


def run_panel_refusal_journey(source_url: str) -> dict[str, object]:
    """Exercise original panel refusals over an isolated damaged-source server.

    Parameters
    ----------
    source_url
        Owned loopback Vite server containing isolated damaged input files.

    Returns
    -------
    dict[str, object]
        Actual refusal observations and all original owner counters.

    Raises
    ------
    ValueError
        The source server is not an explicit owned loopback URL.
    AssertionError
        Browser errors, external requests or missing original owners occur.

    """
    from tools.studio_browser_journey import loopback_url

    source = loopback_url(source_url)
    from playwright.sync_api import sync_playwright

    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    errors: list[str] = []
    rejected: list[str] = []
    records: list[dict[str, object]] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            context = browser.new_context(service_workers="block")
            try:
                _observe_context(context, {urlsplit(source).netloc}, errors, rejected)
                page = context.new_page()
                profiler = start_native_coverage(page)
                try:
                    page.goto(source + "browser-tests/workspace.html", wait_until="networkidle")
                    observed = page.evaluate(
                        "async () => (await import('/browser-tests/workspacePanelRefusal.native.tsx')).runNativePanelRefusalCases()"
                    )
                finally:
                    take_native_coverage(profiler, records, include_panel=True)
                assert errors == [], errors
                assert rejected == [], rejected
                return {
                    "source_url": source,
                    "browser": browser.version,
                    "playwright": version("playwright"),
                    "coverage_percentage": "not_calculated",
                    "native_v8_coverage": records,
                    "panel_source_refusal": observed,
                    "page_errors": errors,
                    "external_requests": rejected,
                }
            finally:
                context.close()
        finally:
            browser.close()


def _observe_context(
    context: BrowserContext, origins: set[str], errors: list[str], rejected: list[str]
) -> None:
    """Record page failures and bound requests to the owned servers.

    Parameters
    ----------
    context
        Isolated owned browser context.
    origins
        Exact allowed loopback authorities.
    errors
        Destination for observed browser failures.
    rejected
        Destination for refused request addresses.

    """
    from playwright.sync_api import Route

    def request(route: Route) -> None:
        parsed = urlsplit(route.request.url)
        if parsed.scheme == "http" and parsed.netloc in origins:
            route.continue_()
        else:
            rejected.append(route.request.url)
            route.abort()

    context.route("**/*", request)
    context.on("page", lambda page: page.on("pageerror", lambda error: errors.append(str(error))))


def run_workspace_journey(
    base_url: str, source_url: str, observations: dict[str, object] | None = None
) -> dict[str, object]:
    """Exercise exact UI recovery and complete-index native transaction cases.

    Parameters
    ----------
    base_url
        Owned loopback preview with the built real WASM bundle.
    source_url
        Separately owned loopback Vite server for actual public source API cases.
    observations
        Caller-owned partial evidence retained if a later assertion fails.

    Returns
    -------
    dict[str, object]
        Actual UI/native observations and V8 counters; no percentage is inferred.

    Raises
    ------
    ValueError
        Either supplied server address is not an explicit owned loopback URL.
    AssertionError
        A real runtime, identity, rollback, quota or recovery assertion fails.

    """
    from tools.studio_browser_journey import loopback_url

    preview = loopback_url(base_url)
    source = loopback_url(source_url)
    if preview == source:
        raise ValueError("Built preview and owned source-test server must be distinct")
    from scpn_quantum_control.studio_workspace.canonical import canonical_digest

    observed: dict[str, object] = {} if observations is None else observations
    from playwright.sync_api import expect, sync_playwright

    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage
    from tools.studio_workspace_browser_ui import (
        run_empty_project_ui,
        run_formatting_ui,
        run_graph_ui,
    )

    origins = {urlsplit(preview).netloc, urlsplit(source).netloc}
    corpus = (
        Path(__file__).resolve().parents[1] / "tests/data/studio_workspace/documents.json"
    ).read_text(encoding="utf-8")
    errors: list[str] = []
    rejected: list[str] = []
    observed.update(
        source_url=source,
        external_requests=rejected,
        page_errors=errors,
        coverage_percentage="not_calculated",
    )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        observed.update(browser=browser.version, playwright=version("playwright"))
        try:
            with ExitStack() as contexts:
                context = contexts.enter_context(
                    browser.new_context(service_workers="block", accept_downloads=True)
                )
                clean = contexts.enter_context(
                    browser.new_context(service_workers="block", accept_downloads=True)
                )
                _observe_context(context, origins, errors, rejected)
                _observe_context(clean, origins, errors, rejected)
                page = context.new_page()
                page.goto(preview, wait_until="networkidle")
                expect(page.get_by_role("img", name="order parameter over time")).to_be_visible()
                ui = run_empty_project_ui(page)
                observed["ui_empty_project_roundtrip"] = ui
                fresh_ui = clean.new_page()
                fresh_ui.goto(preview, wait_until="networkidle")
                workspace = fresh_ui.get_by_role("region", name="Local workspace")
                expect(
                    workspace.get_by_role("button", name="Create empty project")
                ).to_be_enabled()
                workspace.get_by_label("Workspace archive file").set_input_files(
                    {
                        "name": "portable.json",
                        "mimeType": "application/json",
                        "buffer": cast(str, ui["archive"]).encode("utf-8"),
                    }
                )
                expect(workspace.get_by_role("status")).to_contain_text("Archive read locally")
                workspace.get_by_role("button", name="Preview archive", exact=True).click()
                save = workspace.get_by_role("button", name="Save draft and revision references")
                expect(save).to_be_enabled()
                save.click()
                expect(workspace.get_by_role("status")).to_contain_text(
                    "Workspace transaction committed"
                )
                expect(workspace.get_by_label("Saved workspace identity")).to_have_text(
                    cast(str, ui["identity"]), use_inner_text=True
                )
                native_page = context.new_page()
                profiler = start_native_coverage(native_page)
                native_v8: list[dict[str, object]] = []
                observed["native_v8_coverage"] = native_v8
                try:
                    native_page.goto(
                        source + "browser-tests/workspace.html", wait_until="networkidle"
                    )
                    native = cast(
                        dict[str, object],
                        native_page.evaluate(
                            "async corpus => (await import('/browser-tests/workspaceStore.native.ts')).runNativeWorkspaceCases(corpus)",
                            corpus,
                        ),
                    )
                    observed["native_synthetic_graph_roundtrip"] = native
                    native_digest = canonical_digest(
                        "quantum_workspace_archive_source.v1", native["archive"]
                    )
                    assert native_digest == native["archiveDigest"], (
                        "Native exact archive snapshot disagrees with original Python digest"
                    )
                    observed["python_archive_snapshot_digest"] = native_digest
                    observed["native_cache_recovery"] = native_page.evaluate(
                        "async corpus => (await import('/browser-tests/workspaceRecovery.native.ts')).runNativeRecoveryCases(corpus)",
                        corpus,
                    )
                    observed["native_blocked_cache_open"] = native_page.evaluate(
                        "async () => (await import('/browser-tests/workspaceConnections.native.ts')).runNativeBlockedOpenCases()"
                    )
                    observed["native_connection_lifecycle"] = native_page.evaluate(
                        "async corpus => (await import('/browser-tests/workspaceConnections.native.ts')).runNativeConnectionCases(corpus)",
                        corpus,
                    )
                    observed["native_controller_recovery"] = native_page.evaluate(
                        "async () => (await import('/browser-tests/workspaceController.native.tsx')).runNativeControllerCases()"
                    )
                    observed["native_save_recheck_races"] = native_page.evaluate(
                        "async corpus => (await import('/browser-tests/workspaceSaveRaces.native.ts')).runNativeSaveRaces(corpus)",
                        corpus,
                    )
                    formatting = cast(
                        dict[str, object],
                        native_page.evaluate(
                            "async corpus => (await import('/browser-tests/workspaceFormatting.native.ts')).runNativeFormatting(corpus)",
                            corpus,
                        ),
                    )
                    observed["native_formatting_recovery"] = formatting
                    for text_key, digest_key in (
                        ("priorArchive", "priorArchiveDigest"),
                        ("archive", "archiveDigest"),
                    ):
                        expected_digest = canonical_digest(
                            "quantum_workspace_archive_source.v1", formatting[text_key]
                        )
                        assert expected_digest == formatting[digest_key], (
                            f"Native formatting {digest_key} disagrees with original Python digest"
                        )
                    observed["source_component_empty_roundtrip"] = run_empty_project_ui(
                        native_page, navigate=False
                    )
                    observed["source_component_graph_roundtrip"] = run_graph_ui(
                        native_page, native
                    )
                    observed["source_component_formatting_recovery"] = run_formatting_ui(
                        native_page, native
                    )
                    other_tab = context.new_page()
                    tab_profiler = start_native_coverage(other_tab)
                    try:
                        other_tab.goto(
                            source + "browser-tests/workspace.html", wait_until="networkidle"
                        )
                        database_name = native_page.evaluate(
                            "() => 'workspace-two-tabs-' + crypto.randomUUID()"
                        )
                        prepared_tabs = []
                        for tab, initialize in ((native_page, True), (other_tab, False)):
                            prepared_tabs.append(
                                tab.evaluate(
                                    "async args => (await import('/browser-tests/workspaceTabs.native.ts')).prepareNativeTab(args.corpus, args.databaseName, args.initialize)",
                                    {
                                        "corpus": corpus,
                                        "databaseName": database_name,
                                        "initialize": initialize,
                                    },
                                )
                            )
                        assert prepared_tabs[0] == prepared_tabs[1], (
                            "Actual independent tabs did not start from the same original head"
                        )
                        first_tab = native_page.evaluate(
                            "async () => (await import('/browser-tests/workspaceTabs.native.ts')).savePreparedNativeTab(false)"
                        )
                        stale_tab = other_tab.evaluate(
                            "async () => (await import('/browser-tests/workspaceTabs.native.ts')).savePreparedNativeTab(true)"
                        )
                        observed["native_two_tab_reconciliation"] = {
                            "prepared": prepared_tabs,
                            "first_commit": first_tab,
                            "stale_refusal": stale_tab,
                        }
                    finally:
                        take_native_coverage(tab_profiler, native_v8)
                    origin = source.rstrip("/")
                    profiler.send(
                        "Storage.overrideQuotaForOrigin", {"origin": origin, "quotaSize": 1024}
                    )
                    try:
                        quota_admission = profiler.send(
                            "Storage.getUsageAndQuota", {"origin": origin}
                        )
                        observed["native_quota_admission"] = quota_admission
                        assert quota_admission["overrideActive"] is True
                        assert quota_admission["quota"] == 1024
                        # Chromium's IndexedDB bucket allowance decays over 30 seconds:
                        # content/browser/indexed_db/instance/bucket_context.cc:GetBucketSpaceToAllot.
                        quota_wait_started = monotonic()
                        native_page.wait_for_timeout(31_000)
                        observed["native_quota_cache_wait_seconds"] = (
                            monotonic() - quota_wait_started
                        )
                        quota = native_page.evaluate(
                            "async args => (await import('/browser-tests/workspaceQuota.native.ts')).requireNativeQuotaRefusal(args.databaseName, args.archive)",
                            native,
                        )
                        observed["native_quota"] = quota
                    finally:
                        profiler.send("Storage.overrideQuotaForOrigin", {"origin": origin})
                except Exception as error:
                    observed["native_failure"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    take_native_coverage(profiler, native_v8)
                fresh_native = clean.new_page()
                fresh_profiler = start_native_coverage(fresh_native)
                try:
                    fresh_native.goto(
                        source + "browser-tests/workspace.html", wait_until="networkidle"
                    )
                    imported = cast(
                        dict[str, object],
                        fresh_native.evaluate(
                            "async json => (await import('/browser-tests/workspaceStore.native.ts')).importNativeWorkspace(json)",
                            native["archive"],
                        ),
                    )
                    observed["native_fresh_context_import"] = imported
                    for key in (
                        "projectId",
                        "archiveDigest",
                        "workspaceHash",
                        "documentHashes",
                        "rawHashes",
                    ):
                        assert imported[key] == native[key], (
                            f"Fresh-context native import changed {key}"
                        )
                    observed["fresh_source_component_graph_roundtrip"] = run_graph_ui(
                        fresh_native, native
                    )
                except Exception as error:
                    observed["fresh_import_failure"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    take_native_coverage(fresh_profiler, native_v8)
                # Synthetic test producers are deliberately absent from the production UI.
                workspace.get_by_label("Workspace archive JSON").fill(cast(str, native["archive"]))
                workspace.get_by_role("button", name="Preview archive", exact=True).click()
                expect(workspace.get_by_role("status")).to_contain_text(
                    "unsupported raw producer/schema"
                )
                expect(save).to_be_disabled()
                expect(workspace.get_by_label("Saved workspace identity")).to_have_text(
                    cast(str, ui["identity"]), use_inner_text=True
                )
                observed["production_unknown_source_refusal"] = True
                failure_page = context.new_page()
                failure_profiler = start_native_coverage(failure_page)
                try:
                    failure_page.goto(
                        source + "browser-tests/workspace.html", wait_until="networkidle"
                    )
                    observed["native_controller_failed_open"] = failure_page.evaluate(
                        "async () => (await import('/browser-tests/workspaceController.native.tsx')).runNativeControllerFailedOpenCases()"
                    )
                finally:
                    take_native_coverage(failure_profiler, native_v8)
                assert errors == [], errors
                assert rejected == [], rejected
                return observed
        finally:
            browser.close()
