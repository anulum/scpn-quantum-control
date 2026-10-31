# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — pulse feasibility instrument through the built Studio
"""Exercise the pulse feasibility instrument in a real browser, with no submission."""

from __future__ import annotations

import json
from importlib.metadata import version
from urllib.parse import unquote, urlsplit

_DOSSIER_PREFIX = "data:application/json;charset=utf-8,"


def run_pulse_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Switch designs and read the dossier through the production Analyse stage.

    Parameters
    ----------
    base_url
        Owned literal-loopback preview carrying the actual built Studio.
    evidence
        Caller-owned partial evidence retained on a later runtime refusal.

    Returns
    -------
    dict
        Browser/runtime identity, every observed contract and the absence of
        workers, submissions, outside requests and page errors.

    Raises
    ------
    ValueError
        The preview violates the shared loopback ownership contract.
    AssertionError
        A real UI, network or runtime contract fails.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import Route, expect, sync_playwright

    observed = {} if evidence is None else evidence
    observations: list[str] = []
    errors: list[str] = []
    external: list[str] = []
    submissions: list[str] = []
    observed.update(
        scenario="pulse_feasibility",
        playwright=version("playwright"),
        observations=observations,
        page_errors=errors,
        external_requests=external,
        submission_requests=submissions,
    )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        context = browser.new_context(service_workers="block")
        context.set_default_timeout(15000)
        try:
            origin = urlsplit(url).netloc

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

            context.route("**/*", bound)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url, wait_until="networkidle")
            page.get_by_role("navigation", name="Workbench views").get_by_role(
                "link", name="Analyse", exact=True
            ).click()
            instrument = page.get_by_role("region", name="Pulse feasibility", exact=True)
            expect(instrument).to_be_visible()
            link = instrument.get_by_role("link", name="Download no-submit dossier")
            dossier = json.loads(
                unquote((link.get_attribute("href") or "").removeprefix(_DOSSIER_PREFIX))
            )
            assert dossier["submissions"] == 0 and dossier["hardware_submission"] is False
            observations.append("feasible-design-exports-a-no-submit-dossier")
            instrument.get_by_label("Design", exact=True).select_option(
                label="half-grid-and-over-amplitude"
            )
            expect(instrument.locator("rect[data-violated='true']")).to_have_count(2)
            expect(instrument.get_by_role("alert", name="Dossier refused")).to_be_visible()
            observations.append("violations-marked-and-export-refused")
            instrument.get_by_label("Design", exact=True).select_option(label="expired-snapshot")
            expect(instrument.get_by_role("note", name="Capability snapshot")).to_contain_text(
                "expired before the assessment"
            )
            expect(link).to_have_count(0)
            observations.append("expired-snapshot-refuses-the-dossier")
            assert instrument.get_by_role("button", name="Submit").count() == 0

            page.goto(url + "#/results/pulses", wait_until="networkidle")
            expect(page.locator('[id="/results/pulses"]')).to_be_focused()
            observations.append("deep-link-focuses-the-instrument")
            assert not errors, errors
            assert not external, external
            assert not submissions, submissions
            assert not page.workers, "The instrument must not retain workers"
            observed.update(browser=browser.version, workers=len(page.workers), no_submit=True)
            return observed
        finally:
            context.close()
            browser.close()
