# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real workbench accessibility acceptance
"""Audit actual routes and native keyboard behaviour with the exact locked auditor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

from tools.studio_accessibility_checks import (
    AXE_SHA256,
    AXE_VERSION,
    audit_page,
    chart_table_identity,
    graph_table_identity,
    keyboard_route,
    populated_states,
    read_auditor,
)

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Route

Theme = Literal["light", "dark"]
_THEMES: tuple[Theme, ...] = ("light", "dark")
_LOADING = "loading the WASM simulator kernel…"
_ROUTES = (
    ("build", "Workbench views", "Model"),
    ("workspace", "Workbench views", "Workspace"),
    ("results", "Workbench views", "Analyse"),
    ("compare", "Workbench views", "Compare"),
    ("evidence", "Workbench views", "Evidence"),
    ("operations", "Workbench context destinations", "Devices & Operations"),
    ("experiments", "Workbench views", "Run"),
    ("atlas", "Workbench context destinations", "Atlas"),
)
_CONTROLS = "button:visible, a:visible, input:visible, select:visible, textarea:visible"
_REDUCED_MOTION = """() => [...document.querySelectorAll('.qsp-workbench, .qsp-workbench *')].every(element => {
    const style = getComputedStyle(element);
    return style.animationDuration === '0s' && style.transitionDuration === '0s';
})"""
_PARTIAL_EVIDENCE = json.dumps(
    {"status": "failed", "claim_status": "refuted", "supports_quantum_advantage": False}
)


@dataclass
class _Transport:
    """Bound one theme's requests to the two owned origins and hold kernels on demand."""

    origins: set[str]
    rejected: list[str]
    submissions: list[str]
    held: list[Route] = field(default_factory=list)
    holding: bool = True

    def bound(self, route: Route) -> None:
        """Refuse outside requests and submissions; let owned reads through.

        Parameters
        ----------
        route
            Intercepted request.

        """
        target = urlsplit(route.request.url)
        if target.scheme != "http" or target.netloc not in self.origins:
            self.rejected.append(route.request.url)
            route.abort()
        elif route.request.method != "GET":
            self.submissions.append(route.request.method)
            route.abort()
        else:
            route.continue_()

    def __call__(self, route: Route) -> None:
        """Hold kernel transport while holding; bound every other request.

        Parameters
        ----------
        route
            Intercepted request.

        """
        if self.holding and urlsplit(route.request.url).path.endswith(".wasm"):
            self.held.append(route)
        else:
            self.bound(route)

    def release(self) -> None:
        """Stop holding and pass every held kernel request through the bound."""
        self.holding = False
        for pending in self.held:
            self.bound(pending)
        self.held.clear()


@dataclass(frozen=True)
class _Lists:
    """Caller-visible evidence lists the journey fills."""

    reports: list[dict[str, object]] = field(default_factory=list)
    keyboards: list[dict[str, object]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    submissions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Audit:
    """Audit pages with the locked auditor under one theme."""

    auditor: str
    reports: list[dict[str, object]]
    theme: Theme

    def __call__(self, page: Page, state: str) -> None:
        """Audit the page and record it under the theme-qualified state.

        Parameters
        ----------
        page
            Page in the audited state.
        state
            State identity without the theme.

        """
        audit_page(page, self.auditor, f"{self.theme}:{state}", self.reports)


def _check_loading(page: Page, preview: str, transport: _Transport, audit: _Audit) -> None:
    """Audit the declared kernel-loading state, then release the kernels.

    Parameters
    ----------
    page
        Owned page of this theme.
    preview
        Owned built preview.
    transport
        Route state of this theme, holding kernels.
    audit
        Auditor of this theme.

    """
    from playwright.sync_api import expect

    page.goto(preview, wait_until="domcontentloaded")
    loading = page.get_by_role("status").filter(has_text=_LOADING)
    expect(loading).to_have_count(2)
    expect(loading.first).to_be_visible()
    audit(page, "actual-kernel-loading")
    for refused in (transport.rejected, transport.submissions):
        assert not refused, refused
    transport.release()
    expect(page.get_by_role("table", name="Order parameter data", exact=True)).to_be_visible()
    expect(page.get_by_role("table", name="Phase trajectory data", exact=True)).to_be_visible()


def _check_zoomed_layout(page: Page) -> None:
    """Zoom to 200 % and keep every visible control inside the 1280 px viewport.

    Parameters
    ----------
    page
        Owned page on one route.

    """
    page.evaluate("document.documentElement.style.zoom = '200%'")
    assert page.evaluate("getComputedStyle(document.documentElement).zoom") == "2"
    for control in page.locator(_CONTROLS).all():
        control.scroll_into_view_if_needed()
        box = control.bounding_box()
        assert box is not None and box["width"] > 0 and box["height"] > 0
        assert box["x"] >= -1 and box["x"] + box["width"] <= 1281, box


def _check_routes(page: Page, audit: _Audit, keyboards: list[dict[str, object]]) -> None:
    """Reach every route by keyboard and audit it at normal size and at 200 % zoom.

    Parameters
    ----------
    page
        Owned page of this theme.
    audit
        Auditor of this theme.
    keyboards
        Caller-owned keyboard behaviour records.

    """
    from playwright.sync_api import expect

    for route, navigation_name, label in _ROUTES:
        navigation = page.get_by_role("navigation", name=navigation_name, exact=True)
        navigation.get_by_role("link", name=label, exact=True).focus()
        page.keyboard.press("Enter")
        view = page.get_by_role("region", name="Workbench view", exact=True)
        expect(view).to_be_focused()
        expect(page.get_by_role("status").filter(has_text="Loading")).to_have_count(0)
        keyboards.append({"state": f"{audit.theme}:{route}", **keyboard_route(page)})
        audit(page, f"{route}:normal")
        _check_zoomed_layout(page)
        audit(page, f"{route}:css-zoom-200")
        page.evaluate("document.documentElement.style.zoom = ''")


def _check_states(page: Page, source: str, audit: _Audit, observed: dict[str, object]) -> None:
    """Audit the malformed route, the populated states, the evidence states and the graph.

    Parameters
    ----------
    page
        Owned page of this theme.
    source
        Owned root Vite source server.
    audit
        Auditor of this theme.
    observed
        Caller-owned evidence.

    """
    from playwright.sync_api import expect

    theme = audit.theme
    page.evaluate("location.hash = '#/atlas?revision=%FF'")
    expect(page.get_by_role("alert")).to_contain_text("Route unavailable")
    audit(page, "malformed-route")
    observed[f"{theme}_populated_states"] = populated_states(
        page, audit.auditor, theme, audit.reports
    )
    page.evaluate("location.hash = '#/workspace'")
    viewer = page.get_by_role("region", name="Inspect evidence JSON")
    editor = viewer.get_by_label("Evidence JSON")
    inspect = viewer.get_by_role("button", name="Inspect snapshot", exact=True)
    for state, text in (("malformed-evidence", "{"), ("partial-evidence", _PARTIAL_EVIDENCE)):
        editor.fill(text)
        inspect.click()
        shown = (
            viewer.get_by_role("alert")
            if state == "malformed-evidence"
            else viewer.get_by_role("region", name="Evidence inspector")
        )
        expect(shown).to_be_visible()
        audit(page, state)
    observed[f"{theme}_graph_table"] = graph_table_identity(page, source)
    audit(page, "original-parameter-graph")
    assert not page.workers, "Accessibility must not leave an owned worker"


def _check_offline(
    context: BrowserContext,
    preview: str,
    transport: _Transport,
    audit: _Audit,
    observed: dict[str, object],
    errors: list[str],
) -> None:
    """Hold the kernel transport, go offline and audit the refusal of unverifiable results.

    Parameters
    ----------
    context
        Browser context of this theme.
    preview
        Owned built preview.
    transport
        Route state of this theme.
    audit
        Auditor of this theme.
    observed
        Caller-owned evidence.
    errors
        Caller-owned page-error record.

    """
    from playwright.sync_api import expect

    transport.holding = True
    offline = context.new_page()
    offline.on("pageerror", lambda error: errors.append(str(error)))
    with offline.expect_request(lambda request: urlsplit(request.url).path.endswith(".wasm")):
        offline.goto(preview, wait_until="domcontentloaded")
    expect(offline.get_by_role("status").filter(has_text=_LOADING)).to_have_count(2)
    assert transport.held, "Offline case did not reach the original transport"
    observed[f"{audit.theme}_offline_held_requests"] = len(transport.held)
    context.set_offline(True)
    transport.release()
    unverifiable = offline.get_by_role("alert").filter(has_text="unverifiable")
    expect(unverifiable).to_have_count(2)
    expect(offline.get_by_role("table", name="Order parameter data")).to_have_count(0)
    expect(offline.get_by_role("table", name="Phase trajectory data")).to_have_count(0)
    audit(offline, "actual-offline-kernel-refusal")
    context.set_offline(False)
    offline.close()


def _audit_theme(
    browser: Browser,
    hosts: tuple[str, str],
    audit: _Audit,
    lists: _Lists,
    observed: dict[str, object],
) -> None:
    """Run every audit of one colour scheme in its own reduced-motion context.

    Parameters
    ----------
    browser
        Launched browser.
    hosts
        Owned built preview and root Vite source server.
    audit
        Auditor of this theme.
    lists
        Evidence lists.
    observed
        Caller-owned evidence.

    """
    preview, source = hosts
    origins = {urlsplit(host).netloc for host in hosts}
    transport = _Transport(origins, lists.rejected, lists.submissions)
    with browser.new_context(
        service_workers="block",
        viewport={"width": 1280, "height": 900},
        color_scheme=audit.theme,
        reduced_motion="reduce",
    ) as context:
        context.set_default_timeout(15_000)
        context.route("**/*", transport)
        page = context.new_page()
        page.on("pageerror", lambda error: lists.errors.append(str(error)))
        _check_loading(page, preview, transport, audit)
        observed[f"{audit.theme}_chart_table"] = chart_table_identity(page)
        reduced_motion = page.evaluate(_REDUCED_MOTION)
        assert reduced_motion, "A delivered control animates under reduced motion"
        observed[f"{audit.theme}_reduced_motion"] = reduced_motion
        _check_routes(page, audit, lists.keyboards)
        _check_states(page, source, audit, observed)
        _check_offline(context, preview, transport, audit, observed, lists.errors)


def _owned_hosts(base_url: str, source_url: str) -> tuple[str, str]:
    """Validate the built preview and the distinct root Vite source server.

    Parameters
    ----------
    base_url
        Owned actual built Studio preview with its genuine WASM artifacts.
    source_url
        Distinct owned root Vite server containing verbatim production sources.

    Returns
    -------
    tuple[str, str]
        The preview and the source server.

    Raises
    ------
    ValueError
        The addresses violate the original ownership contract.

    """
    from tools.studio_browser_journey import loopback_url

    preview, source = loopback_url(base_url), loopback_url(source_url)
    if urlsplit(preview).netloc == urlsplit(source).netloc or urlsplit(source).path != "/":
        raise ValueError("Accessibility requires a distinct owned root source server")
    return preview, source


def _start_evidence(evidence: dict[str, object] | None) -> tuple[dict[str, object], _Lists]:
    """Record the scenario, the locked auditor identity and the empty evidence lists.

    Parameters
    ----------
    evidence
        Partial caller-owned report, or None for a fresh one.

    Returns
    -------
    tuple[dict[str, object], _Lists]
        The evidence mapping and the lists it exposes.

    """
    lists = _Lists()
    observed = {} if evidence is None else evidence
    observed.update(
        scenario="workbench_accessibility",
        axe_version=AXE_VERSION,
        axe_sha256=AXE_SHA256,
        audits=lists.reports,
        keyboard=lists.keyboards,
        page_errors=lists.errors,
        external_requests=lists.rejected,
        submissions=lists.submissions,
        playwright=version("playwright"),
        screen_reader="separate manual runtime evidence required",
        public_screenshots=0,
        user_workspace_exports=0,
    )
    return observed, lists


def _evidence_refusals(
    preview: str, auditor: str, reports: list[dict[str, object]], observed: dict[str, object]
) -> None:
    """Audit the shared evidence journey's refusal states under each theme.

    Parameters
    ----------
    preview
        Owned built preview.
    auditor
        Locked auditor source.
    reports
        Caller-owned audit records.
    observed
        Caller-owned evidence receiving each theme's evidence journey.

    """
    from tools.studio_browser_journey import run_evidence_journey

    for theme in _THEMES:
        observed[f"{theme}_source_evidence_refusal"] = run_evidence_journey(
            preview, color_scheme=theme, audit=_Audit(auditor, reports, theme)
        )


def _audit_themes(
    hosts: tuple[str, str], auditor: str, lists: _Lists, observed: dict[str, object]
) -> None:
    """Audit both colour schemes in one browser and refuse any recorded failure.

    Parameters
    ----------
    hosts
        Owned built preview and root Vite source server.
    auditor
        Locked auditor source.
    lists
        Evidence lists.
    observed
        Caller-owned evidence.

    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        observed["browser"] = browser.version
        try:
            for theme in _THEMES:
                audit = _Audit(auditor, lists.reports, theme)
                _audit_theme(browser, hosts, audit, lists, observed)
            for refused in (lists.errors, lists.rejected, lists.submissions):
                assert not refused, refused
            observed["workers"] = 0
        finally:
            browser.close()


def run_accessibility_journey(
    base_url: str,
    source_url: str,
    axe_source: Path | None = None,
    evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    """Qualify real route audits, source tables and old-result refusal boundaries.

    Parameters
    ----------
    base_url
        Owned actual built Studio preview with its genuine WASM artifacts.
    source_url
        Distinct owned root Vite server containing verbatim production sources.
    axe_source
        Existing locked auditor asset; no dependency installation is performed.
    evidence
        Partial caller-owned report retained when an actual assertion fails.

    Returns
    -------
    dict[str, object]
        Full audits, original source conformance and observed keyboard behaviour.

    Raises
    ------
    ValueError
        The addresses or auditor violate the original ownership contract.
    AssertionError
        A real route, accessibility, stale-result or custody requirement fails.

    """
    hosts = _owned_hosts(base_url, source_url)
    auditor = read_auditor(axe_source)
    observed, lists = _start_evidence(evidence)
    _audit_themes(hosts, auditor, lists, observed)
    _evidence_refusals(hosts[0], auditor, lists.reports, observed)
    return observed
