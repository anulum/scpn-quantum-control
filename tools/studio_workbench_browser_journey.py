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
from collections.abc import Callable
from importlib.metadata import version
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import Page


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
    page
        Isolated native page already displaying the original workbench.
    kernels
        Require both original built WASM instruments when true.
    chunk_error
        Recover from an actual refused Build chunk when true.
    before_reload
        Capture the complete original document before full navigation discards scripts.

    Returns
    -------
    dict[str, object]
        Saved identity, retained context and keyboard recovery observations.

    """
    from playwright.sync_api import expect

    from tools.studio_workspace_browser_ui import run_empty_project_ui

    nav = page.get_by_role("navigation", name="Workbench views")
    expect(nav).to_be_visible()
    expect(page.get_by_role("heading", name="Baseline scorecard", exact=True)).to_be_visible()
    for heading in (
        "Capabilities",
        "Compile recompute",
        "Kuramoto Play",
        "3D Lab",
        "Program-AD gradient replay",
        "Differentiate support explorer",
        "Gradient-plan explanation",
    ):
        expect(page.get_by_role("heading", name=heading, exact=True)).to_be_visible()
    expect(page.get_by_role("region", name="Capability catalogue")).to_be_visible()
    expect(page.get_by_role("region", name="Inspect evidence JSON")).to_be_visible()
    saved = run_empty_project_ui(page, navigate=False)
    identity = page.get_by_label("Saved workspace identity")
    project = identity.locator("dd").nth(0).inner_text()
    archive = identity.locator("dd").nth(2).inner_text()
    editor = page.get_by_label("Workspace archive JSON")
    draft = cast(str, saved["archive"]) + "\n"
    editor.fill(draft)
    nav.get_by_role("link", name="Build", exact=True).focus()
    page.keyboard.press("Enter")
    if chunk_error:
        expect(page.get_by_role("alert")).to_contain_text("Your workspace and editor are retained")
        page.get_by_role("link", name="Return to Workspace").click()
        expect(editor).to_have_value(draft)
        expect(identity.locator("dd").nth(2)).to_have_text(archive)
        return {"saved": saved, "draft_retained": True, "real_chunk_status": 503}
    expect(page.get_by_role("heading", name="Build", exact=True)).to_be_visible()
    expect(page.get_by_role("region", name="Workbench view", exact=True)).to_be_focused()
    if kernels:
        page.get_by_role("button", name="Recompute in browser", exact=True).click()
        expect(
            page.get_by_text("recomputed digest matches the signed claim", exact=True)
        ).to_be_visible()
    nav.get_by_role("link", name="Results", exact=True).click()
    expect(page.get_by_role("heading", name="Results", exact=True)).to_be_visible()
    if kernels:
        page.get_by_role("button", name="Recompute gradient in browser", exact=True).click()
        expect(page.locator('[data-verdict="match"]')).to_be_visible()
    page.go_back()
    expect(page.get_by_role("heading", name="Build", exact=True)).to_be_visible()
    page.go_forward()
    expect(page.get_by_role("heading", name="Results", exact=True)).to_be_visible()
    nav.get_by_role("link", name="Workspace", exact=True).click()
    expect(editor).to_have_value(draft)
    for invalid in ("#/build?project=a&project=b", "#/future", "#/atlas?revision=%FF"):
        page.evaluate("hash => { window.location.hash = hash; }", invalid)
        expect(page.get_by_role("heading", name="Route unavailable")).to_be_visible()
        page.get_by_role("link", name="Return to Workspace").click()
        expect(editor).to_have_value(draft)
        expect(identity.locator("dd").nth(2)).to_have_text(archive)
    nav.get_by_role("link", name="Atlas", exact=True).click()
    expect(page.get_by_role("heading", name="Atlas unavailable")).to_be_visible()
    if before_reload is not None:
        before_reload()
    deep = "#/atlas?project=" + project + "&revision=r%2F%CE%B1&snapshot=s%2B2"
    page.goto(page.url.split("#")[0] + deep, wait_until="networkidle")
    inspector = page.get_by_role("complementary", name="Workbench inspector")
    expect(page.get_by_role("heading", name="Atlas unavailable")).to_be_visible()
    expect(inspector.get_by_text("r/α", exact=True)).to_be_visible()
    expect(inspector.get_by_text("s+2", exact=True)).to_be_visible()
    expect(inspector.get_by_text(archive, exact=True)).to_be_visible()
    page.reload(wait_until="networkidle")
    expect(inspector.get_by_text("r/α", exact=True)).to_be_visible()
    nav.get_by_role("link", name="Experiments", exact=True).click()
    expect(page.get_by_role("heading", name="Local experiment", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Open Kuramoto sample", exact=True)).to_be_enabled()
    expect(page.get_by_label("Current experiment result", exact=True)).to_have_count(0)
    page.go_back()
    expect(page.get_by_role("heading", name="Atlas unavailable")).to_be_visible()
    page.go_forward()
    expect(page.get_by_role("heading", name="Local experiment", exact=True)).to_be_visible()
    nav.get_by_role("link", name="Workspace", exact=True).click()
    expect(editor).to_have_value(cast(str, saved["archive"]))
    assert "revision=r%2F%CE%B1" in page.url
    nav.get_by_role("link", name="Atlas", exact=True).focus()
    page.keyboard.press("Shift+Tab")
    page.keyboard.press("Tab")
    link = nav.get_by_role("link", name="Atlas", exact=True)
    expect(link).to_be_focused()
    box = link.bounding_box()
    assert box is not None and box["x"] >= 0 and box["x"] + box["width"] <= 320
    outline = link.evaluate("element => getComputedStyle(element).outlineStyle")
    assert outline != "none", "Keyboard focus has no visible outline"
    return {
        "saved": saved,
        "history_and_reload": True,
        "malformed_route_recovery": True,
        "project": project,
        "revision": "r/α",
        "snapshot": "s+2",
        "focus_box": box,
        "outline": outline,
        "real_wasm": kernels,
    }


def run_workbench_journey(
    base_url: str, source_url: str, observations: dict[str, object] | None = None
) -> dict[str, object]:
    """Run standalone/federated/chunk-refusal cases and qualified source counters.

    Parameters
    ----------
    base_url
        Owned built preview with actual Rust WASM and independent acceptance host.
    source_url
        Distinct owned root Vite server for the unchanged public source facade.
    observations
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
    from playwright.sync_api import Error, Request, Route, expect, sync_playwright

    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    observed = {} if observations is None else observations
    cases: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    observed.update(
        source_url=source,
        observations=cases,
        native_v8_coverage=records,
        coverage_percentage="not_calculated",
        playwright=version("playwright"),
    )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        observed["browser"] = browser.version
        try:
            for case in ("standalone", "embedded", "chunk-error", "source"):
                context = browser.new_context(
                    viewport={"width": 320, "height": 760},
                    service_workers="block",
                    accept_downloads=True,
                )
                errors: list[str] = []
                rejected: list[str] = []
                requests: list[str] = []
                entry: dict[str, object] = {
                    "case": case,
                    "page_errors": errors,
                    "external_requests": rejected,
                }
                cases.append(entry)
                owned = urlsplit(source if case == "source" else preview).netloc

                def bound(
                    route: Route,
                    *,
                    authority: str = owned,
                    refused: list[str] = rejected,
                    fail_chunk: bool = case == "chunk-error",
                ) -> None:
                    target = urlsplit(route.request.url)
                    if target.scheme != "http" or target.netloc != authority:
                        refused.append(route.request.url)
                        route.abort()
                    elif fail_chunk and "/assets/BuildView-" in target.path:
                        route.fulfill(status=503, body="Bounded route chunk unavailable")
                    else:
                        route.continue_()

                def record_error(error: Error, *, captured: list[str] = errors) -> None:
                    captured.append(str(error))

                def record_request(request: Request, *, captured: list[str] = requests) -> None:
                    captured.append(request.url)

                try:
                    context.route("**/*", bound)
                    page = context.new_page()
                    page.set_default_timeout(15_000)
                    page.on("pageerror", record_error)
                    page.on("request", record_request)
                    profiler = start_native_coverage(page) if case == "source" else None

                    def capture_source() -> None:
                        nonlocal profiler
                        if profiler is not None:
                            session = profiler
                            profiler = None
                            take_native_coverage(session, records, include_workbench=True)

                    try:
                        if case == "source":
                            url = source + "browser-tests/workbench.html"
                        elif case == "embedded":
                            url = preview + "acceptance-host/browser-tests/federation.html"
                        else:
                            url = preview
                        page.goto(url, wait_until="networkidle")
                        expect(
                            page.get_by_role("navigation", name="Workbench views")
                        ).to_be_visible()
                        if case != "source":
                            assert not any(
                                "/assets/" + view + "-" in request
                                for request in requests
                                for view in ("BuildView", "ResultsView", "UnavailableView")
                            )
                        if case == "embedded":
                            page.get_by_role("button", name="Host singleton counter").click()
                            expect(page.get_by_label("Federation consumer state")).to_have_text(
                                "1"
                            )
                            expect(
                                page.get_by_text("Embedded workbench", exact=False)
                            ).to_be_visible()
                        entry.update(
                            _navigation(
                                page,
                                kernels=case != "source",
                                chunk_error=case == "chunk-error",
                                before_reload=capture_source if case == "source" else None,
                            )
                        )
                        assert not errors, errors
                        assert not rejected, rejected
                        active_workers = tuple(page.workers)
                        entry["workers_before_final_navigation"] = [
                            worker.url for worker in active_workers
                        ]
                        if active_workers:
                            # Returning to Workspace mounts original automatic playback.
                            # Exercise its real route cleanup before asserting native worker closure.
                            page.get_by_role("navigation", name="Workbench views").get_by_role(
                                "link", name="Atlas", exact=True
                            ).click()
                            expect(
                                page.get_by_role("heading", name="Atlas unavailable")
                            ).to_be_visible()
                        entry["worker_urls"] = [worker.url for worker in page.workers]
                        entry["workers"] = len(page.workers)
                        assert not page.workers, "Workbench leaked an owned worker"
                    finally:
                        capture_source()
                finally:
                    context.close()
            return observed
        finally:
            browser.close()
