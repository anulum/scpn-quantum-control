# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native workbench navigation journey
"""Exercise the original workbench, genuine federation and saved-state recovery."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import Browser, CDPSession, Locator, Page, Request, Route

_CASES = ("standalone", "embedded", "chunk-error", "source")
_LAZY_VIEWS = ("BuildView", "ResultsView", "UnavailableView")
_INVALID_ROUTES = ("#/build?project=a&project=b", "#/future", "#/atlas?revision=%FF")


@dataclass(frozen=True)
class _Saved:
    """The saved empty project, its identity and the unsaved draft kept beside it."""

    saved: dict[str, object]
    editor: Locator
    identity: Locator
    project: str
    archive: str
    draft: str


def _check_landing(page: Page) -> None:
    """Find every landing instrument and the committed evidence catalogue.

    Parameters
    ----------
    page : Page
        Isolated native page already displaying the original workbench.

    """
    from playwright.sync_api import expect

    expect(page.get_by_role("navigation", name="Workbench views")).to_be_visible()
    for heading in ("Compile recompute", "Kuramoto Play", "3D Lab", "Program-AD gradient replay"):
        expect(page.get_by_role("heading", name=heading, exact=True)).to_be_visible()
    expect(page.get_by_role("region", name="Capability catalogue")).to_be_visible()
    expect(page.get_by_role("region", name="Inspect evidence JSON")).to_be_visible()
    # The committed evidence surfaces live in the Evidence catalogue destination.
    page.get_by_role("navigation", name="Continue working").get_by_role(
        "link", name="Evidence catalogue", exact=True
    ).click()
    catalogue = page.get_by_role("article", name="Committed evidence catalogue")
    for heading in (
        "Capabilities",
        "Differentiate support explorer",
        "Gradient-plan explanation",
        "Baseline scorecard",
    ):
        expect(catalogue.get_by_role("heading", name=heading, exact=True)).to_be_visible()


def _save_with_draft(page: Page, nav: Locator) -> _Saved:
    """Save an empty project and keep an unsaved draft of its archive in the editor.

    Parameters
    ----------
    page : Page
        Owned workbench page.
    nav : Locator
        Workbench views navigation.

    Returns
    -------
    _Saved
        The saved project, its identity and the kept draft.

    """
    from playwright.sync_api import expect

    from tools.studio_workspace_browser_ui import run_empty_project_ui

    nav.get_by_role("link", name="Workspace", exact=True).click()
    expect(page.get_by_role("region", name="Capability catalogue")).to_be_visible()
    saved = run_empty_project_ui(page, navigate=False)
    identity = page.get_by_label("Saved workspace identity")
    editor = page.get_by_label("Workspace archive JSON")
    draft = cast(str, saved["archive"]) + "\n"
    editor.fill(draft)
    return _Saved(
        saved,
        editor,
        identity,
        identity.locator("dd").nth(0).inner_text(),
        identity.locator("dd").nth(2).inner_text(),
        draft,
    )


def _check_model_and_analyse(page: Page, nav: Locator, kernels: bool) -> None:
    """Recompute both built WASM claims and walk history between Model and Analyse.

    Parameters
    ----------
    page : Page
        Owned workbench page on the Model route.
    nav : Locator
        Workbench views navigation.
    kernels : bool
        Require both original built WASM instruments when true.

    """
    from playwright.sync_api import expect

    expect(page.get_by_role("heading", name="Model", exact=True)).to_be_visible()
    expect(page.get_by_role("region", name="Workbench view", exact=True)).to_be_focused()
    if kernels:
        page.get_by_role("button", name="Recompute in browser", exact=True).click()
        matched = page.get_by_text("recomputed digest matches the signed claim", exact=True)
        expect(matched).to_be_visible()
    nav.get_by_role("link", name="Analyse", exact=True).click()
    expect(page.get_by_role("heading", name="Analyse", exact=True)).to_be_visible()
    if kernels:
        page.get_by_role("button", name="Recompute gradient in browser", exact=True).click()
        expect(page.locator('[data-verdict="match"]')).to_be_visible()
    page.go_back()
    expect(page.get_by_role("heading", name="Model", exact=True)).to_be_visible()
    page.go_forward()
    expect(page.get_by_role("heading", name="Analyse", exact=True)).to_be_visible()


def _check_compare_and_invalid_routes(page: Page, nav: Locator, kept: _Saved) -> None:
    """Open Compare, then recover from three invalid routes with the draft kept.

    Parameters
    ----------
    page : Page
        Owned workbench page.
    nav : Locator
        Workbench views navigation.
    kept : _Saved
        The saved project and the kept draft.

    """
    from playwright.sync_api import expect

    # Comparison is its own stage of the loop; its registered instruments load there.
    nav.get_by_role("link", name="Compare", exact=True).click()
    expect(page.get_by_role("heading", name="Compare", exact=True)).to_be_visible()
    comparison = page.get_by_role("heading", name="Method comparison and uncertainty", exact=True)
    expect(comparison).to_be_visible()
    nav.get_by_role("link", name="Workspace", exact=True).click()
    expect(kept.editor).to_have_value(kept.draft)
    for invalid in _INVALID_ROUTES:
        page.evaluate("hash => { window.location.hash = hash; }", invalid)
        expect(page.get_by_role("heading", name="Route unavailable")).to_be_visible()
        page.get_by_role("link", name="Return to Workspace").click()
        expect(kept.editor).to_have_value(kept.draft)
        expect(kept.identity.locator("dd").nth(2)).to_have_text(kept.archive)


def _check_deep_link(
    page: Page, nav: Locator, kept: _Saved, before_reload: Callable[[], None] | None
) -> None:
    """Load an Atlas deep link with encoded context, reload it and walk history.

    Parameters
    ----------
    page : Page
        Owned workbench page.
    nav : Locator
        Workbench views navigation.
    kept : _Saved
        The saved project and the kept draft.
    before_reload : Callable[[], None] | None
        Capture the complete original document before full navigation discards scripts.

    """
    from playwright.sync_api import expect

    context_nav = page.get_by_role("navigation", name="Workbench context destinations")
    context_nav.get_by_role("link", name="Atlas", exact=True).click()
    expect(page.get_by_role("heading", name="Atlas unavailable")).to_be_visible()
    if before_reload is not None:
        before_reload()
    deep = "#/atlas?project=" + kept.project + "&revision=r%2F%CE%B1&snapshot=s%2B2"
    page.goto(page.url.split("#")[0] + deep, wait_until="networkidle")
    inspector = page.get_by_role("complementary", name="Workbench inspector")
    expect(page.get_by_role("heading", name="Atlas unavailable")).to_be_visible()
    for shown in ("r/α", "s+2", kept.archive):
        expect(inspector.get_by_text(shown, exact=True)).to_be_visible()
    page.reload(wait_until="networkidle")
    expect(inspector.get_by_text("r/α", exact=True)).to_be_visible()
    nav.get_by_role("link", name="Run", exact=True).click()
    expect(page.get_by_role("heading", name="Local experiment", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Open Kuramoto sample", exact=True)).to_be_enabled()
    expect(page.get_by_label("Current experiment result", exact=True)).to_have_count(0)
    page.go_back()
    expect(page.get_by_role("heading", name="Atlas unavailable")).to_be_visible()
    page.go_forward()
    expect(page.get_by_role("heading", name="Local experiment", exact=True)).to_be_visible()
    nav.get_by_role("link", name="Workspace", exact=True).click()
    expect(kept.editor).to_have_value(cast(str, kept.saved["archive"]))
    assert "revision=r%2F%CE%B1" in page.url


def _check_keyboard_focus(page: Page, nav: Locator) -> tuple[dict[str, float], str]:
    """Reach the Evidence link by keyboard and find its focus box and outline visible.

    Parameters
    ----------
    page : Page
        Owned workbench page.
    nav : Locator
        Workbench views navigation.

    Returns
    -------
    tuple[dict[str, float], str]
        The focused link's box and its outline style.

    """
    from playwright.sync_api import expect

    link = nav.get_by_role("link", name="Evidence", exact=True)
    link.focus()
    page.keyboard.press("Shift+Tab")
    page.keyboard.press("Tab")
    expect(link).to_be_focused()
    box = link.bounding_box()
    assert box is not None and box["x"] >= 0 and box["x"] + box["width"] <= 320
    outline = cast(str, link.evaluate("element => getComputedStyle(element).outlineStyle"))
    assert outline != "none", "Keyboard focus has no visible outline"
    return cast(dict[str, float], box), outline


def _recover_chunk_error(page: Page, kept: _Saved) -> dict[str, object]:
    """Find the refused Build chunk reported and return to the kept workspace.

    Parameters
    ----------
    page : Page
        Owned workbench page after the refused chunk.
    kept : _Saved
        The saved project and the kept draft.

    Returns
    -------
    dict[str, object]
        The chunk-error observations.

    """
    from playwright.sync_api import expect

    expect(page.get_by_role("alert")).to_contain_text("Your workspace and editor are retained")
    page.get_by_role("link", name="Return to Workspace").click()
    expect(kept.editor).to_have_value(kept.draft)
    expect(kept.identity.locator("dd").nth(2)).to_have_text(kept.archive)
    return {"saved": kept.saved, "draft_retained": True, "real_chunk_status": 503}


def _navigation(
    page: Page,
    *,
    kernels: bool,
    chunk_error: bool,
    before_reload: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Navigate the public panel while retaining actual editor and archive identity.

    Parameters
    ----------
    page : Page
        Isolated native page already displaying the original workbench.
    kernels : bool
        Require both original built WASM instruments when true.
    chunk_error : bool
        Recover from an actual refused Build chunk when true.
    before_reload : Callable[[], None] | None
        Capture the complete original document before full navigation discards scripts.

    Returns
    -------
    dict[str, object]
        Saved identity, retained context and keyboard recovery observations.

    """
    nav = page.get_by_role("navigation", name="Workbench views")
    _check_landing(page)
    kept = _save_with_draft(page, nav)
    nav.get_by_role("link", name="Model", exact=True).focus()
    page.keyboard.press("Enter")
    if chunk_error:
        return _recover_chunk_error(page, kept)
    _check_model_and_analyse(page, nav, kernels)
    _check_compare_and_invalid_routes(page, nav, kept)
    _check_deep_link(page, nav, kept, before_reload)
    box, outline = _check_keyboard_focus(page, nav)
    return {
        "saved": kept.saved,
        "history_and_reload": True,
        "malformed_route_recovery": True,
        "project": kept.project,
        "revision": "r/α",
        "snapshot": "s+2",
        "focus_box": box,
        "outline": outline,
        "real_wasm": kernels,
    }


@dataclass
class _Case:
    """One navigation case's owned origin, failure records and source profiler."""

    name: str
    authority: str
    records: list[dict[str, object]]
    errors: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)
    profiler: CDPSession | None = None

    def bound(self, route: Route) -> None:
        """Refuse outside requests and, in the chunk-error case, the Build chunk.

        Parameters
        ----------
        route : Route
            Intercepted request.

        """
        target = urlsplit(route.request.url)
        if target.scheme != "http" or target.netloc != self.authority:
            self.rejected.append(route.request.url)
            route.abort()
        elif self.name == "chunk-error" and "/assets/BuildView-" in target.path:
            route.fulfill(status=503, body="Bounded route chunk unavailable")
        else:
            route.continue_()

    def record_request(self, request: Request) -> None:
        """Retain the actual request URL of this case.

        Parameters
        ----------
        request : Request
            Original native browser request.

        """
        self.requests.append(request.url)

    def capture_source(self) -> None:
        """Take the source counters once, before navigation discards the scripts."""
        from tools.studio_workspace_browser_coverage import take_native_coverage

        session, self.profiler = self.profiler, None
        if session is not None:
            take_native_coverage(session, self.records, include_workbench=True)


def _open_case(page: Page, case: _Case, hosts: tuple[str, str]) -> None:
    """Load the case's page and check its lazily loaded views and federation host.

    Parameters
    ----------
    page : Page
        Owned page of the case.
    case : _Case
        The navigation case.
    hosts : tuple[str, str]
        Owned built preview and root Vite source server.

    """
    from playwright.sync_api import expect

    preview, source = hosts
    if case.name == "source":
        url = source + "browser-tests/workbench.html"
    elif case.name == "embedded":
        url = preview + "acceptance-host/browser-tests/federation.html"
    else:
        url = preview
    page.goto(url, wait_until="networkidle")
    expect(page.get_by_role("navigation", name="Workbench views")).to_be_visible()
    if case.name != "source":
        assert not any(
            "/assets/" + view + "-" in request for request in case.requests for view in _LAZY_VIEWS
        )
    if case.name == "embedded":
        page.get_by_role("button", name="Host singleton counter").click()
        expect(page.get_by_label("Federation consumer state")).to_have_text("1")
        expect(page.get_by_text("Embedded workbench", exact=False)).to_be_visible()


def _dispose_workers(page: Page, entry: dict[str, object]) -> None:
    """Leave the workspace, await native worker closure and require none left.

    Parameters
    ----------
    page : Page
        Owned page of the case.
    entry : dict[str, object]
        Case record receiving the worker observations.

    """
    from playwright.sync_api import expect

    from tools.studio_owned_worker_journey import wait_for_worker_disposal

    entry["workers_before_final_navigation"] = [worker.url for worker in tuple(page.workers)]
    # Workspace may still be preparing automatic playback at the snapshot.
    # Always unmount its owner, then await native closure, including workers
    # created while disposal is being observed.
    page.get_by_role("navigation", name="Workbench context destinations").get_by_role(
        "link", name="Atlas", exact=True
    ).click()
    expect(page.get_by_role("heading", name="Atlas unavailable")).to_be_visible()
    wait_for_worker_disposal(page, observe_termination=False)
    entry["final_route"] = urlsplit(page.url).fragment
    entry["worker_urls"] = [worker.url for worker in page.workers]
    entry["workers"] = len(page.workers)
    assert not page.workers, "Workbench leaked an owned worker"


def _new_case(
    name: str, hosts: tuple[str, str], observed: dict[str, object]
) -> tuple[_Case, dict[str, object]]:
    """Create a case bound to its host and append its record to the evidence.

    Parameters
    ----------
    name : str
        Case name; the source case is bound to the source server.
    hosts : tuple[str, str]
        Owned built preview and root Vite source server.
    observed : dict[str, object]
        Caller-owned evidence holding the case list and the coverage records.

    Returns
    -------
    tuple[_Case, dict[str, object]]
        The case and its record.

    """
    records = cast(list[dict[str, object]], observed["native_v8_coverage"])
    case = _Case(name, urlsplit(hosts[1] if name == "source" else hosts[0]).netloc, records)
    entry: dict[str, object] = {
        "case": name,
        "page_errors": case.errors,
        "external_requests": case.rejected,
    }
    cast(list[dict[str, object]], observed["observations"]).append(entry)
    return case, entry


def _run_case(
    browser: Browser, name: str, hosts: tuple[str, str], observed: dict[str, object]
) -> None:
    """Run one navigation case in its own narrow context and record it.

    Parameters
    ----------
    browser : Browser
        Launched browser.
    name : str
        Case name.
    hosts : tuple[str, str]
        Owned built preview and root Vite source server.
    observed : dict[str, object]
        Caller-owned evidence holding the case list and the coverage records.

    """
    from tools.studio_workspace_browser_coverage import start_native_coverage

    case, entry = _new_case(name, hosts, observed)
    context = browser.new_context(
        viewport={"width": 320, "height": 760}, service_workers="block", accept_downloads=True
    )
    try:
        context.route("**/*", case.bound)
        page = context.new_page()
        page.set_default_timeout(15_000)
        page.on("pageerror", lambda error: case.errors.append(str(error)))
        page.on("request", case.record_request)
        case.profiler = start_native_coverage(page) if name == "source" else None
        try:
            _open_case(page, case, hosts)
            before_reload = case.capture_source if name == "source" else None
            entry.update(
                _navigation(
                    page,
                    kernels=name != "source",
                    chunk_error=name == "chunk-error",
                    before_reload=before_reload,
                )
            )
            for refused in (case.errors, case.rejected):
                assert not refused, refused
            _dispose_workers(page, entry)
        finally:
            case.capture_source()
    finally:
        context.close()


def _owned_hosts(base_url: str, source_url: str) -> tuple[str, str]:
    """Validate the built preview and the distinct root Vite source server.

    Parameters
    ----------
    base_url : str
        Owned built preview with actual Rust WASM and independent acceptance host.
    source_url : str
        Distinct owned root Vite server for the unchanged public source facade.

    Returns
    -------
    tuple[str, str]
        The preview and the source server.

    Raises
    ------
    ValueError
        Either URL is unowned, the servers coincide, or source uses a nonroot path.

    """
    from tools.studio_browser_journey import loopback_url

    preview = loopback_url(base_url)
    source = loopback_url(source_url)
    preview_address = urlsplit(preview)
    source_address = urlsplit(source)
    same_server = (
        ipaddress.ip_address(cast(str, preview_address.hostname))
        == ipaddress.ip_address(cast(str, source_address.hostname))
        and preview_address.port == source_address.port
    )
    if same_server or source_address.path != "/":
        raise ValueError("Workbench requires a distinct owned root source server")
    return preview, source


def run_workbench_journey(
    base_url: str, source_url: str, observations: dict[str, object] | None = None
) -> dict[str, object]:
    """Run standalone/federated/chunk-refusal cases and qualified source counters.

    Parameters
    ----------
    base_url : str
        Owned built preview with actual Rust WASM and independent acceptance host.
    source_url : str
        Distinct owned root Vite server for the unchanged public source facade.
    observations : dict[str, object] | None
        Caller-owned partial diagnostics retained when a later case fails.

    Returns
    -------
    dict[str, object]
        Actual browser outcomes and native source counters without inferred percentages.

    Raises
    ------
    ValueError
        Either URL is unowned, the servers coincide, or source uses a nonroot path.
    AssertionError
        Navigation, recovery, actual recomputation or boundary ownership fails.

    """
    hosts = _owned_hosts(base_url, source_url)
    from playwright.sync_api import sync_playwright

    observed = {} if observations is None else observations
    observed.update(
        source_url=hosts[1],
        observations=[],
        native_v8_coverage=[],
        coverage_percentage="not_calculated",
        playwright=version("playwright"),
    )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        observed["browser"] = browser.version
        try:
            for name in _CASES:
                _run_case(browser, name, hosts, observed)
            return observed
        finally:
            browser.close()
