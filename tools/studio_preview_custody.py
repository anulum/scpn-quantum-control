# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — owned-preview request custody for Studio browser journeys
"""Keep a Studio browser journey on its owned preview, with no submission.

Every instrument journey routes all page requests through one guard: a request
outside the owned literal-loopback preview is recorded and aborted, a request
with any method other than GET is recorded and aborted, and every other
request continues. Journeys that qualify the WebAssembly kernels also record
each ``.wasm`` path they load. ``run_custodied_journey`` runs one instrument
journey inside that custody and refuses it after any page error, outside
request, submission or retained worker.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.sync_api import Page, Playwright, Route

    Walk = Callable[[Page, str, list[str]], dict[str, object]]


def preview_route_guard(
    origin: str,
    external: list[str],
    submissions: list[str],
    kernels: list[str] | None = None,
) -> Callable[[Route], None]:
    """Build the route handler that keeps a page on its owned preview.

    Parameters
    ----------
    origin
        Host and port of the owned literal-loopback preview.
    external
        Caller-owned record; receives ``"outside-preview"`` per refused request.
    submissions
        Caller-owned record; receives the method of each refused non-GET request.
    kernels
        Optional caller-owned record of the ``.wasm`` paths the page loads.

    Returns
    -------
    Callable
        Handler that aborts outside requests and submissions and lets the rest through.

    """

    def bound(route: Route) -> None:
        target = urlsplit(route.request.url)
        if target.scheme != "http" or target.netloc != origin:
            external.append("outside-preview")
            route.abort()
        elif route.request.method != "GET":
            submissions.append(route.request.method)
            route.abort()
        else:
            if kernels is not None and target.path.endswith(".wasm"):
                kernels.append(target.path)
            route.continue_()

    return bound


def _start_evidence(
    evidence: dict[str, object] | None, scenario: str, kernels: list[str] | None
) -> tuple[dict[str, object], tuple[list[str], list[str], list[str], list[str]]]:
    """Record the scenario identity and the lists the journey fills.

    Parameters
    ----------
    evidence
        Caller-owned evidence mapping, or None for a fresh one.
    scenario
        Scenario name recorded in the evidence.
    kernels
        Optional kernel record, kept as ``wasm_requests``.

    Returns
    -------
    tuple
        The evidence mapping, then the observation, page-error, outside-request
        and submission lists.

    """
    observed = {} if evidence is None else evidence
    lists: tuple[list[str], list[str], list[str], list[str]] = ([], [], [], [])
    observed.update(
        scenario=scenario,
        playwright=version("playwright"),
        observations=lists[0],
        page_errors=lists[1],
        external_requests=lists[2],
        submission_requests=lists[3],
    )
    if kernels is not None:
        observed["wasm_requests"] = kernels
    return observed, lists


def _require_custody(
    page: Page, browser_version: str, observed: dict[str, object], *refusals: list[str]
) -> dict[str, object]:
    """Refuse a journey after any custody breach, then record the browser identity.

    Parameters
    ----------
    page
        Page whose retained workers are checked.
    browser_version
        Version string of the launched browser.
    observed
        Evidence mapping completed in place.
    *refusals
        Page errors, outside requests and submissions; each must be empty.

    Returns
    -------
    dict
        The completed evidence mapping.

    Raises
    ------
    AssertionError
        A list holds a breach or the page retained a worker.

    """
    for refused in (*refusals, page.workers):
        assert not refused, refused
    observed.update(browser=browser_version, workers=len(page.workers), no_submit=True)
    return observed


def _drive(
    runtime: Playwright,
    url: str,
    walk: Walk,
    observed: dict[str, object],
    lists: tuple[list[str], list[str], list[str], list[str]],
    kernels: list[str] | None,
    downloads: bool,
) -> dict[str, object]:
    """Open the custodied page, run the walk on it and check its custody."""
    observations, errors, external, submissions = lists
    browser = runtime.chromium.launch()
    context = browser.new_context(
        service_workers="block", accept_downloads=True if downloads else None
    )
    context.set_default_timeout(15000)
    try:
        guard = preview_route_guard(urlsplit(url).netloc, external, submissions, kernels)
        context.route("**/*", guard)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        observed.update(walk(page, url, observations))
        return _require_custody(page, browser.version, observed, errors, external, submissions)
    finally:
        context.close()
        browser.close()


def run_custodied_journey(
    base_url: str,
    evidence: dict[str, object] | None,
    scenario: str,
    walk: Walk,
    *,
    kernels: list[str] | None = None,
    downloads: bool = False,
) -> dict[str, object]:
    """Run one instrument journey on the owned preview and check its custody.

    Parameters
    ----------
    base_url
        Owned literal-loopback preview carrying the actual built Studio.
    evidence
        Caller-owned partial evidence retained on a later runtime refusal.
    scenario
        Scenario name recorded in the evidence.
    walk
        Journey steps: page, preview URL and observations in, extra evidence out.
    kernels
        Optional record of loaded ``.wasm`` paths, kept as ``wasm_requests``.
    downloads
        Whether the browser context accepts downloads.

    Returns
    -------
    dict
        Browser identity, observations, walk evidence and the custody record.

    Raises
    ------
    ValueError
        The preview violates the shared loopback ownership contract.
    AssertionError
        A real UI, network or runtime contract fails.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import sync_playwright

    observed, lists = _start_evidence(evidence, scenario, kernels)
    with sync_playwright() as runtime:
        return _drive(runtime, url, walk, observed, lists, kernels, downloads)
