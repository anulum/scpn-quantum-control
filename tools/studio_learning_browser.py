# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — learning, forecasting and domain benchmarks through the built Studio
"""Exercise the learning instrument in a real browser, with no submission."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.sync_api import Page, Route

INSTRUMENT = "Learning, forecasting and domain benchmarks"
AXE_SOURCE = Path(__file__).resolve().parents[1] / "studio-web/node_modules/axe-core/axe.min.js"


def _check_instrument(page: Page, observations: list[str]) -> None:
    """Assert the five learning contracts on the rendered instrument.

    Parameters
    ----------
    page
        Playwright page showing the Analyse stage.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    instrument = page.get_by_role("region", name=INSTRUMENT, exact=True)
    expect(instrument).to_be_visible()
    custody = instrument.get_by_label("Held-out custody", exact=True)
    expect(custody).to_contain_text("forecaster fitting requires a train batch")
    expect(instrument.get_by_label("QSNN held-out evaluation", exact=True)).to_contain_text(
        "train/test overlap"
    )
    observations.append("held-out-records-refused-for-fitting-and-scoring")
    expect(instrument.get_by_label("Checkpoint identity", exact=True)).to_contain_text(
        "qsnn_training_checkpoint.v2"
    )
    observations.append("checkpoint-resume-keeps-optimiser-identity")
    expect(instrument.get_by_role("list", name="Owner covariance refusals")).to_contain_text(
        "sigma_q must be positive definite"
    )
    observations.append("owner-covariance-refusal-shown")
    selector = instrument.get_by_label("Domain benchmark", exact=True)
    selector.select_option(label="iter_mhd_8mode")
    expect(
        instrument.get_by_role("img", name="iter_mhd_8mode proxy metrics, not physical outcomes")
    ).to_contain_text("(proxy)")
    href = instrument.get_by_role("link", name="Download proxy metadata").get_attribute("href")
    exported = json.loads(unquote(str(href).split(",", 1)[1]))
    assert exported["evidence_class"] == "domain proxy metric", exported
    assert "real plant control" in exported["claims_forbidden"], exported
    observations.append("proxy-labels-in-plot-and-export")
    expect(instrument).to_contain_text("global random state was left unchanged")
    observations.append("global-random-state-unchanged")


def _accessibility(page: Page) -> list[dict[str, object]]:
    """Run the pinned axe rules on the instrument at desktop and mobile width.

    Parameters
    ----------
    page
        Playwright page showing the focused instrument.

    Returns
    -------
    list of dict
        One empty violation list per checked viewport width.

    """
    page.add_script_tag(path=str(AXE_SOURCE))
    accessibility: list[dict[str, object]] = []
    for width in (1280, 390):
        page.set_viewport_size({"width": width, "height": 900})
        violations = page.evaluate("""async () => {
            const root = document.querySelector('[aria-label="Learning, forecasting and domain benchmarks"]');
            const result = await axe.run(root);
            return result.violations.map(item => ({id: item.id, nodes: item.nodes.length}));
        }""")
        assert not violations, violations
        accessibility.append({"width": width, "violations": violations})
    return accessibility


def _walk(page: Page, url: str, observations: list[str]) -> list[dict[str, object]]:
    """Open Analyse, check the instrument, its deep link and its accessibility.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    list of dict
        Accessibility results per viewport width.

    """
    from playwright.sync_api import expect

    page.goto(url, wait_until="networkidle")
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name="Analyse", exact=True
    ).click()
    _check_instrument(page, observations)
    page.goto(url + "#/results/learning", wait_until="networkidle")
    expect(page.locator('[id="/results/learning"]')).to_be_focused()
    observations.append("deep-link-focuses-the-instrument")
    accessibility = _accessibility(page)
    observations.append("accessible-at-desktop-and-mobile")
    return accessibility


def _preview_bound(
    origin: str, external: list[str], submissions: list[str]
) -> Callable[[Route], None]:
    """Build the route handler that keeps the page on the owned preview.

    Parameters
    ----------
    origin
        Host and port of the owned literal-loopback preview.
    external
        Caller-owned record of requests outside the preview.
    submissions
        Caller-owned record of request methods other than GET.

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
            route.continue_()

    return bound


def _start_evidence(observed: dict[str, object]) -> tuple[list[str], ...]:
    """Record the scenario identity and the lists the journey fills.

    Parameters
    ----------
    observed
        Caller-owned evidence mapping.

    Returns
    -------
    tuple of list
        Observations, page errors, outside requests and submissions, in that order.

    """
    lists: tuple[list[str], ...] = ([], [], [], [])
    observed.update(
        scenario="learning_domain_instrument",
        playwright=version("playwright"),
        observations=lists[0],
        page_errors=lists[1],
        external_requests=lists[2],
        submission_requests=lists[3],
    )
    return lists


def run_learning_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Inspect source learning evidence through the production Analyse stage.

    Parameters
    ----------
    base_url
        Owned literal-loopback preview carrying the actual built Studio.
    evidence
        Caller-owned partial evidence retained on a later runtime refusal.

    Returns
    -------
    dict
        Browser identity, every observed contract and the absence of workers,
        submissions, outside requests and page errors.

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

    observed = {} if evidence is None else evidence
    observations, errors, external, submissions = _start_evidence(observed)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        context = browser.new_context(service_workers="block")
        context.set_default_timeout(15000)
        try:
            context.route("**/*", _preview_bound(urlsplit(url).netloc, external, submissions))
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            observed["accessibility"] = _walk(page, url, observations)
            for refused in (errors, external, submissions, page.workers):
                assert not refused, refused
            observed.update(browser=browser.version, workers=len(page.workers), no_submit=True)
            return observed
        finally:
            context.close()
            browser.close()
