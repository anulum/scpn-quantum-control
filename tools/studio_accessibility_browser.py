# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real workbench accessibility acceptance
"""Audit actual routes and native keyboard behavior with the exact locked auditor."""

from __future__ import annotations

import json
from collections.abc import Callable
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
    from playwright.sync_api import Page


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
        Full audits, original source conformance and observed keyboard behavior.

    Raises
    ------
    ValueError
        The addresses or auditor violate the original ownership contract.
    AssertionError
        A real route, accessibility, stale-result or custody requirement fails.

    """
    from tools.studio_browser_journey import loopback_url, run_evidence_journey

    preview, source = loopback_url(base_url), loopback_url(source_url)
    if urlsplit(preview).netloc == urlsplit(source).netloc or urlsplit(source).path != "/":
        raise ValueError("Accessibility requires a distinct owned root source server")
    auditor = read_auditor(axe_source)
    from playwright.sync_api import Route, expect, sync_playwright

    observed = {} if evidence is None else evidence
    audits: list[dict[str, object]] = []
    keyboards: list[dict[str, object]] = []
    errors: list[str] = []
    rejected: list[str] = []
    submissions: list[str] = []
    observed.update(
        scenario="workbench_accessibility",
        axe_version=AXE_VERSION,
        axe_sha256=AXE_SHA256,
        audits=audits,
        keyboard=keyboards,
        page_errors=errors,
        external_requests=rejected,
        submissions=submissions,
        playwright=version("playwright"),
        screen_reader="separate manual runtime evidence required",
        public_screenshots=0,
        user_workspace_exports=0,
    )
    origins = {urlsplit(preview).netloc, urlsplit(source).netloc}
    themes: tuple[Literal["light", "dark"], ...] = ("light", "dark")
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        observed["browser"] = browser.version
        try:
            for theme in themes:
                with browser.new_context(
                    service_workers="block",
                    viewport={"width": 1280, "height": 900},
                    color_scheme=theme,
                    reduced_motion="reduce",
                ) as context:
                    context.set_default_timeout(15_000)

                    def request(route: Route) -> None:
                        target = urlsplit(route.request.url)
                        if target.scheme != "http" or target.netloc not in origins:
                            rejected.append(route.request.url)
                            route.abort()
                        elif route.request.method != "GET":
                            submissions.append(route.request.method)
                            route.abort()
                        else:
                            route.continue_()

                    held: list[Route] = []
                    holding = [True]
                    original_request = request

                    def loading_request(
                        route: Route,
                        *,
                        pending: list[Route] = held,
                        gating: list[bool] = holding,
                        bound: Callable[[Route], None] = original_request,
                    ) -> None:
                        """Hold actual kernel transport for this context's declared loading state."""
                        if gating[0] and urlsplit(route.request.url).path.endswith(".wasm"):
                            pending.append(route)
                        else:
                            bound(route)

                    context.route("**/*", loading_request)
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(preview, wait_until="domcontentloaded")
                    loading = page.get_by_role("status").filter(
                        has_text="loading the WASM simulator kernel…"
                    )
                    expect(loading).to_have_count(2)
                    expect(loading.first).to_be_visible()
                    audit_page(page, auditor, f"{theme}:actual-kernel-loading", audits)
                    assert not errors, errors
                    assert not rejected, rejected
                    assert not submissions, submissions
                    holding[0] = False
                    for pending in held:
                        original_request(pending)
                    held.clear()
                    expect(
                        page.get_by_role("table", name="Order parameter data", exact=True)
                    ).to_be_visible()
                    expect(
                        page.get_by_role("table", name="Phase trajectory data", exact=True)
                    ).to_be_visible()
                    observed[f"{theme}_chart_table"] = chart_table_identity(page)
                    reduced_motion = page.evaluate("""() => [...document.querySelectorAll('.qsp-workbench, .qsp-workbench *')].every(element => {
                        const style = getComputedStyle(element);
                        return style.animationDuration === '0s' && style.transitionDuration === '0s';
                    })""")
                    assert reduced_motion, "A delivered control animates under reduced motion"
                    observed[f"{theme}_reduced_motion"] = reduced_motion
                    for route in (
                        "build",
                        "workspace",
                        "results",
                        "operations",
                        "experiments",
                        "atlas",
                    ):
                        navigation = page.get_by_role(
                            "navigation",
                            name="Workbench context destinations"
                            if route == "operations"
                            else "Workbench views",
                            exact=True,
                        )
                        link = navigation.get_by_role(
                            "link",
                            name="Devices & Operations"
                            if route == "operations"
                            else route.title(),
                            exact=True,
                        )
                        link.focus()
                        page.keyboard.press("Enter")
                        expect(
                            page.get_by_role("region", name="Workbench view", exact=True)
                        ).to_be_focused()
                        expect(
                            page.get_by_role("status").filter(has_text="Loading")
                        ).to_have_count(0)
                        keyboards.append({"state": f"{theme}:{route}", **keyboard_route(page)})
                        audit_page(page, auditor, f"{theme}:{route}:normal", audits)
                        page.evaluate("document.documentElement.style.zoom = '200%'")
                        assert (
                            page.evaluate("getComputedStyle(document.documentElement).zoom") == "2"
                        )
                        for control in page.locator(
                            "button:visible, a:visible, input:visible, select:visible, textarea:visible"
                        ).all():
                            control.scroll_into_view_if_needed()
                            box = control.bounding_box()
                            assert box is not None and box["width"] > 0 and box["height"] > 0
                            assert box["x"] >= -1 and box["x"] + box["width"] <= 1281, box
                        audit_page(page, auditor, f"{theme}:{route}:css-zoom-200", audits)
                        page.evaluate("document.documentElement.style.zoom = ''")
                    page.evaluate("location.hash = '#/atlas?revision=%FF'")
                    expect(page.get_by_role("alert")).to_contain_text("Route unavailable")
                    audit_page(page, auditor, f"{theme}:malformed-route", audits)
                    observed[f"{theme}_populated_states"] = populated_states(
                        page, auditor, theme, audits
                    )
                    page.evaluate("location.hash = '#/workspace'")
                    viewer = page.get_by_role("region", name="Inspect evidence JSON")
                    editor = viewer.get_by_label("Evidence JSON")
                    inspect = viewer.get_by_role("button", name="Inspect snapshot", exact=True)
                    for state, text in (
                        ("malformed-evidence", "{"),
                        (
                            "partial-evidence",
                            json.dumps(
                                {
                                    "status": "failed",
                                    "claim_status": "refuted",
                                    "supports_quantum_advantage": False,
                                }
                            ),
                        ),
                    ):
                        editor.fill(text)
                        inspect.click()
                        expect(
                            viewer.get_by_role("alert")
                            if state == "malformed-evidence"
                            else viewer.get_by_role("region", name="Evidence inspector")
                        ).to_be_visible()
                        audit_page(page, auditor, f"{theme}:{state}", audits)
                    observed[f"{theme}_graph_table"] = graph_table_identity(page, source)
                    audit_page(page, auditor, f"{theme}:original-parameter-graph", audits)
                    assert not page.workers, "Accessibility must not leave an owned worker"
                    # Hold actual kernel transport, then let the browser fail it offline.
                    holding[0] = True
                    offline = context.new_page()
                    offline.on("pageerror", lambda error: errors.append(str(error)))
                    with offline.expect_request(
                        lambda request: urlsplit(request.url).path.endswith(".wasm")
                    ):
                        offline.goto(preview, wait_until="domcontentloaded")
                    expect(
                        offline.get_by_role("status").filter(
                            has_text="loading the WASM simulator kernel…"
                        )
                    ).to_have_count(2)
                    assert held, "Offline case did not reach the original transport"
                    observed[f"{theme}_offline_held_requests"] = len(held)
                    context.set_offline(True)
                    holding[0] = False
                    for pending in held:
                        original_request(pending)
                    held.clear()
                    expect(
                        offline.get_by_role("alert").filter(has_text="unverifiable")
                    ).to_have_count(2)
                    expect(
                        offline.get_by_role("table", name="Order parameter data")
                    ).to_have_count(0)
                    expect(
                        offline.get_by_role("table", name="Phase trajectory data")
                    ).to_have_count(0)
                    audit_page(offline, auditor, f"{theme}:actual-offline-kernel-refusal", audits)
                    context.set_offline(False)
                    offline.close()
            assert not errors, errors
            assert not rejected, rejected
            assert not submissions, submissions
            observed["workers"] = 0
        finally:
            browser.close()
    for theme in themes:

        def themed_audit(page: Page, state: str, *, scheme: str = theme) -> None:
            """Audit the actual evidence state under its captured native theme.

            Parameters
            ----------
            page
                Actual original evidence page from the shared native runner.
            state
                Original observation identity after its assertion succeeds.
            scheme
                Bound current theme, independent of later loop iterations.

            """
            audit_page(page, auditor, f"{scheme}:{state}", audits)

        observed[f"{theme}_source_evidence_refusal"] = run_evidence_journey(
            preview,
            color_scheme=theme,
            audit=themed_audit,
        )
    return observed
