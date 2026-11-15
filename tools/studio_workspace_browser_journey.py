# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — browser workspace recovery journey
"""Exercise actual workspace UI and native IndexedDB in isolated owned contexts."""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, CDPSession, Locator, Page

_TESTS = "/browser-tests/"
_SOURCE_PAGE = "browser-tests/workspace.html"
_PANEL_CASES = (
    f"async () => (await import('{_TESTS}workspacePanelRefusal.native.tsx'))"
    ".runNativePanelRefusalCases()"
)
_NATIVE_CASES = {
    "native_cache_recovery": (
        f"async corpus => (await import('{_TESTS}workspaceRecovery.native.ts'))"
        ".runNativeRecoveryCases(corpus)",
        True,
    ),
    "native_blocked_cache_open": (
        f"async () => (await import('{_TESTS}workspaceConnections.native.ts'))"
        ".runNativeBlockedOpenCases()",
        False,
    ),
    "native_connection_lifecycle": (
        f"async corpus => (await import('{_TESTS}workspaceConnections.native.ts'))"
        ".runNativeConnectionCases(corpus)",
        True,
    ),
    "native_controller_recovery": (
        f"async () => (await import('{_TESTS}workspaceController.native.tsx'))"
        ".runNativeControllerCases()",
        False,
    ),
    "native_save_recheck_races": (
        f"async corpus => (await import('{_TESTS}workspaceSaveRaces.native.ts'))"
        ".runNativeSaveRaces(corpus)",
        True,
    ),
}
_STORE = f"{_TESTS}workspaceStore.native.ts"
_TABS = f"{_TESTS}workspaceTabs.native.ts"
_ARCHIVE_SCHEMA = "quantum_workspace_archive_source.v1"
_CORPUS = Path(__file__).resolve().parents[1] / "tests/data/studio_workspace/documents.json"
_IDENTITY_KEYS = ("projectId", "archiveDigest", "workspaceHash", "documentHashes", "rawHashes")


@dataclass(frozen=True)
class _Run:
    """Owned hosts, contexts, corpus and evidence of one workspace journey."""

    preview: str
    source: str
    context: BrowserContext
    clean: BrowserContext
    corpus: str
    observed: dict[str, object]
    native_v8: list[dict[str, object]] = field(default_factory=list)


def _observe_context(
    context: BrowserContext, origins: set[str], errors: list[str], rejected: list[str]
) -> None:
    """Record page failures and bound requests to the owned servers.

    Parameters
    ----------
    context : BrowserContext
        Isolated owned browser context.
    origins : set[str]
        Exact allowed loopback authorities.
    errors : list[str]
        Destination for observed browser failures.
    rejected : list[str]
        Destination for refused request addresses.

    """
    from playwright.sync_api import Route

    def request(route: Route) -> None:
        """Permit actual requests only to this context's recorded loopback origins.

        Parameters
        ----------
        route : Route
            Original intercepted native browser request route.

        """
        parsed = urlsplit(route.request.url)
        if parsed.scheme == "http" and parsed.netloc in origins:
            route.continue_()
        else:
            rejected.append(route.request.url)
            route.abort()

    context.route("**/*", request)
    context.on("page", lambda page: page.on("pageerror", lambda error: errors.append(str(error))))


def run_panel_refusal_journey(source_url: str) -> dict[str, object]:
    """Exercise original panel refusals over an isolated damaged-source server.

    Parameters
    ----------
    source_url : str
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

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        try:
            evidence = _panel_refusals(browser, source)
            evidence.update(browser=browser.version, playwright=version("playwright"))
            return evidence
        finally:
            browser.close()


def _panel_refusals(browser: Browser, source: str) -> dict[str, object]:
    """Run the native panel refusal cases in one owned context under a profiler.

    Parameters
    ----------
    browser : Browser
        Launched browser.
    source : str
        Owned loopback Vite server.

    Returns
    -------
    dict[str, object]
        Refusal observations, native counters and the empty failure records.

    """
    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    errors: list[str] = []
    rejected: list[str] = []
    records: list[dict[str, object]] = []
    context = browser.new_context(service_workers="block")
    try:
        _observe_context(context, {urlsplit(source).netloc}, errors, rejected)
        page = context.new_page()
        profiler = start_native_coverage(page)
        try:
            page.goto(source + _SOURCE_PAGE, wait_until="networkidle")
            observed = page.evaluate(_PANEL_CASES)
        finally:
            take_native_coverage(profiler, records, include_panel=True)
        assert errors == [], errors
        assert rejected == [], rejected
        return {
            "source_url": source,
            "coverage_percentage": "not_calculated",
            "native_v8_coverage": records,
            "panel_source_refusal": observed,
            "page_errors": errors,
            "external_requests": rejected,
        }
    finally:
        context.close()


def _ui_roundtrip(run: _Run) -> tuple[dict[str, object], Locator]:
    """Save an empty project in the built UI and import its archive in a clean context.

    Parameters
    ----------
    run : _Run
        The workspace journey.

    Returns
    -------
    tuple[dict[str, object], Locator]
        The saved project and the clean context's workspace region.

    """
    from playwright.sync_api import expect

    from tools.studio_workspace_browser_ui import run_empty_project_ui

    page = run.context.new_page()
    page.goto(run.preview, wait_until="networkidle")
    expect(page.get_by_role("img", name="order parameter over time")).to_be_visible()
    ui = run_empty_project_ui(page)
    run.observed["ui_empty_project_roundtrip"] = ui
    fresh_ui = run.clean.new_page()
    fresh_ui.goto(run.preview, wait_until="networkidle")
    workspace = fresh_ui.get_by_role("region", name="Local workspace")
    expect(workspace.get_by_role("button", name="Create empty project")).to_be_enabled()
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
    expect(workspace.get_by_role("status")).to_contain_text("Workspace transaction committed")
    identity = workspace.get_by_label("Saved workspace identity")
    expect(identity).to_have_text(cast(str, ui["identity"]), use_inner_text=True)
    return ui, workspace


def _native_cases(page: Page, run: _Run) -> dict[str, Any]:
    """Run the native store, recovery, connection, controller and race cases.

    Parameters
    ----------
    page : Page
        Owned page on the source workspace test page.
    run : _Run
        The workspace journey.

    Returns
    -------
    dict[str, Any]
        The native synthetic graph round trip.

    """
    from scpn_quantum_control.studio_workspace.canonical import canonical_digest

    store = f"async corpus => (await import('{_STORE}')).runNativeWorkspaceCases(corpus)"
    native = cast(dict[str, Any], page.evaluate(store, run.corpus))
    run.observed["native_synthetic_graph_roundtrip"] = native
    native_digest = canonical_digest(_ARCHIVE_SCHEMA, native["archive"])
    assert native_digest == native["archiveDigest"], (
        "Native exact archive snapshot disagrees with original Python digest"
    )
    run.observed["python_archive_snapshot_digest"] = native_digest
    for key, (script, takes_corpus) in _NATIVE_CASES.items():
        run.observed[key] = (
            page.evaluate(script, run.corpus) if takes_corpus else page.evaluate(script)
        )
    return native


def _check_formatting(page: Page, run: _Run) -> None:
    """Run the native formatting recovery and match both archives to Python digests.

    Parameters
    ----------
    page : Page
        Owned page on the source workspace test page.
    run : _Run
        The workspace journey.

    """
    from scpn_quantum_control.studio_workspace.canonical import canonical_digest

    script = (
        f"async corpus => (await import('{_TESTS}workspaceFormatting.native.ts'))"
        ".runNativeFormatting(corpus)"
    )
    formatting = cast(dict[str, Any], page.evaluate(script, run.corpus))
    run.observed["native_formatting_recovery"] = formatting
    for text_key, digest_key in (
        ("priorArchive", "priorArchiveDigest"),
        ("archive", "archiveDigest"),
    ):
        expected_digest = canonical_digest(_ARCHIVE_SCHEMA, formatting[text_key])
        assert expected_digest == formatting[digest_key], (
            f"Native formatting {digest_key} disagrees with original Python digest"
        )


def _source_components(page: Page, native: dict[str, Any], run: _Run) -> None:
    """Round-trip an empty project, the synthetic graph and the formatting in the source UI.

    Parameters
    ----------
    page : Page
        Owned page on the source workspace test page.
    native : dict[str, Any]
        Native synthetic graph round trip.
    run : _Run
        The workspace journey.

    """
    from tools.studio_workspace_browser_ui import (
        run_empty_project_ui,
        run_formatting_ui,
        run_graph_ui,
    )

    run.observed["source_component_empty_roundtrip"] = run_empty_project_ui(page, navigate=False)
    run.observed["source_component_graph_roundtrip"] = run_graph_ui(page, native)
    run.observed["source_component_formatting_recovery"] = run_formatting_ui(page, native)


def _two_tabs(page: Page, run: _Run) -> None:
    """Prepare two tabs on one head, commit in the first and refuse the stale second.

    Parameters
    ----------
    page : Page
        Owned page on the source workspace test page, the first tab.
    run : _Run
        The workspace journey.

    """
    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    other_tab = run.context.new_page()
    tab_profiler = start_native_coverage(other_tab)
    try:
        other_tab.goto(run.source + _SOURCE_PAGE, wait_until="networkidle")
        database_name = page.evaluate("() => 'workspace-two-tabs-' + crypto.randomUUID()")
        prepare = (
            f"async args => (await import('{_TABS}'))"
            ".prepareNativeTab(args.corpus, args.databaseName, args.initialize)"
        )
        prepared_tabs = [
            tab.evaluate(
                prepare,
                {"corpus": run.corpus, "databaseName": database_name, "initialize": initialize},
            )
            for tab, initialize in ((page, True), (other_tab, False))
        ]
        assert prepared_tabs[0] == prepared_tabs[1], (
            "Actual independent tabs did not start from the same original head"
        )
        save = "async stale => (await import('" + _TABS + "')).savePreparedNativeTab(stale)"
        run.observed["native_two_tab_reconciliation"] = {
            "prepared": prepared_tabs,
            "first_commit": page.evaluate(save, False),
            "stale_refusal": other_tab.evaluate(save, True),
        }
    finally:
        take_native_coverage(tab_profiler, run.native_v8)


def _quota_refusal(profiler: CDPSession, page: Page, native: dict[str, Any], run: _Run) -> None:
    """Override the source origin's quota to 1 KiB and require the native refusal.

    Parameters
    ----------
    profiler : CDPSession
        Session of the source page, used for the storage override.
    page : Page
        Owned page on the source workspace test page.
    native : dict[str, Any]
        Native synthetic graph round trip.
    run : _Run
        The workspace journey.

    """
    origin = run.source.rstrip("/")
    profiler.send("Storage.overrideQuotaForOrigin", {"origin": origin, "quotaSize": 1024})
    try:
        quota_admission = profiler.send("Storage.getUsageAndQuota", {"origin": origin})
        run.observed["native_quota_admission"] = quota_admission
        assert quota_admission["overrideActive"] is True
        assert quota_admission["quota"] == 1024
        # Chromium's IndexedDB bucket allowance decays over 30 seconds:
        # content/browser/indexed_db/instance/bucket_context.cc:GetBucketSpaceToAllot.
        quota_wait_started = monotonic()
        page.wait_for_timeout(31_000)
        run.observed["native_quota_cache_wait_seconds"] = monotonic() - quota_wait_started
        script = (
            f"async args => (await import('{_TESTS}workspaceQuota.native.ts'))"
            ".requireNativeQuotaRefusal(args.databaseName, args.archive)"
        )
        run.observed["native_quota"] = page.evaluate(script, native)
    finally:
        profiler.send("Storage.overrideQuotaForOrigin", {"origin": origin})


def _native_interval(run: _Run) -> dict[str, Any]:
    """Run every native source case under one profiler and record any failure.

    Parameters
    ----------
    run : _Run
        The workspace journey.

    Returns
    -------
    dict[str, Any]
        Native synthetic graph round trip.

    """
    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    page = run.context.new_page()
    profiler = start_native_coverage(page)
    try:
        page.goto(run.source + _SOURCE_PAGE, wait_until="networkidle")
        native = _native_cases(page, run)
        _check_formatting(page, run)
        _source_components(page, native, run)
        _two_tabs(page, run)
        _quota_refusal(profiler, page, native, run)
        return native
    except Exception as error:
        run.observed["native_failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        take_native_coverage(profiler, run.native_v8)


def _fresh_import(run: _Run, native: dict[str, Any]) -> None:
    """Import the native archive in the clean context and keep every identity.

    Parameters
    ----------
    run : _Run
        The workspace journey.
    native : dict[str, Any]
        Native synthetic graph round trip.

    """
    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage
    from tools.studio_workspace_browser_ui import run_graph_ui

    fresh_native = run.clean.new_page()
    fresh_profiler = start_native_coverage(fresh_native)
    try:
        fresh_native.goto(run.source + _SOURCE_PAGE, wait_until="networkidle")
        script = f"async json => (await import('{_STORE}')).importNativeWorkspace(json)"
        imported = cast(dict[str, object], fresh_native.evaluate(script, native["archive"]))
        run.observed["native_fresh_context_import"] = imported
        for key in _IDENTITY_KEYS:
            assert imported[key] == native[key], f"Fresh-context native import changed {key}"
        run.observed["fresh_source_component_graph_roundtrip"] = run_graph_ui(fresh_native, native)
    except Exception as error:
        run.observed["fresh_import_failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        take_native_coverage(fresh_profiler, run.native_v8)


def _refuse_unknown_source(
    workspace: Locator, ui: dict[str, object], native: dict[str, Any], run: _Run
) -> None:
    """Refuse the synthetic producer's archive in the production UI, keeping the save.

    Parameters
    ----------
    workspace : Locator
        Clean context's workspace region.
    ui : dict[str, object]
        Saved empty project.
    native : dict[str, Any]
        Native synthetic graph round trip.
    run : _Run
        The workspace journey.

    """
    from playwright.sync_api import expect

    # Synthetic test producers are deliberately absent from the production UI.
    workspace.get_by_label("Workspace archive JSON").fill(cast(str, native["archive"]))
    workspace.get_by_role("button", name="Preview archive", exact=True).click()
    expect(workspace.get_by_role("status")).to_contain_text("unsupported raw producer/schema")
    save = workspace.get_by_role("button", name="Save draft and revision references")
    expect(save).to_be_disabled()
    identity = workspace.get_by_label("Saved workspace identity")
    expect(identity).to_have_text(cast(str, ui["identity"]), use_inner_text=True)
    run.observed["production_unknown_source_refusal"] = True


def _failed_open(run: _Run) -> None:
    """Record the native controller's failed-open cases in a fresh page.

    Parameters
    ----------
    run : _Run
        The workspace journey.

    """
    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    failure_page = run.context.new_page()
    failure_profiler = start_native_coverage(failure_page)
    try:
        failure_page.goto(run.source + _SOURCE_PAGE, wait_until="networkidle")
        script = (
            f"async () => (await import('{_TESTS}workspaceController.native.tsx'))"
            ".runNativeControllerFailedOpenCases()"
        )
        run.observed["native_controller_failed_open"] = failure_page.evaluate(script)
    finally:
        take_native_coverage(failure_profiler, run.native_v8)


def _workspace_contexts(
    browser: Browser, hosts: tuple[str, str], observed: dict[str, object]
) -> None:
    """Run the journey in an owned context and a clean one, both bound to the two hosts.

    Parameters
    ----------
    browser : Browser
        Launched browser.
    hosts : tuple[str, str]
        Owned built preview and source-test server.
    observed : dict[str, object]
        Caller-owned evidence holding the failure records.

    """
    errors = cast(list[str], observed["page_errors"])
    rejected = cast(list[str], observed["external_requests"])
    origins = {urlsplit(host).netloc for host in hosts}
    with ExitStack() as contexts:
        context, clean = (
            contexts.enter_context(
                browser.new_context(service_workers="block", accept_downloads=True)
            )
            for _ in range(2)
        )
        for owned in (context, clean):
            _observe_context(owned, origins, errors, rejected)
        run = _Run(*hosts, context, clean, _CORPUS.read_text(encoding="utf-8"), observed)
        observed["native_v8_coverage"] = run.native_v8
        ui, workspace = _ui_roundtrip(run)
        native = _native_interval(run)
        _fresh_import(run, native)
        _refuse_unknown_source(workspace, ui, native, run)
        _failed_open(run)
        assert errors == [], errors
        assert rejected == [], rejected


def _owned_hosts(base_url: str, source_url: str) -> tuple[str, str]:
    """Validate the built preview and the distinct source-test server.

    Parameters
    ----------
    base_url : str
        Owned loopback preview with the built real WASM bundle.
    source_url : str
        Separately owned loopback Vite server for actual public source API cases.

    Returns
    -------
    tuple[str, str]
        The preview and the source-test server.

    Raises
    ------
    ValueError
        Either address is not an explicit owned loopback URL, or they coincide.

    """
    from tools.studio_browser_journey import loopback_url

    preview = loopback_url(base_url)
    source = loopback_url(source_url)
    if preview == source:
        raise ValueError("Built preview and owned source-test server must be distinct")
    return preview, source


def run_workspace_journey(
    base_url: str, source_url: str, observations: dict[str, object] | None = None
) -> dict[str, object]:
    """Exercise exact UI recovery and complete-index native transaction cases.

    Parameters
    ----------
    base_url : str
        Owned loopback preview with the built real WASM bundle.
    source_url : str
        Separately owned loopback Vite server for actual public source API cases.
    observations : dict[str, object] | None
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
    preview, source = _owned_hosts(base_url, source_url)
    from playwright.sync_api import sync_playwright

    observed: dict[str, object] = {} if observations is None else observations
    observed.update(
        source_url=source,
        external_requests=[],
        page_errors=[],
        coverage_percentage="not_calculated",
    )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        observed.update(browser=browser.version, playwright=version("playwright"))
        try:
            _workspace_contexts(browser, (preview, source), observed)
            return observed
        finally:
            browser.close()
