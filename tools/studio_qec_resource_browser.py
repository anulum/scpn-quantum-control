# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QEC and future-resource instrument through the built Studio
"""Exercise the QEC and future-resource instrument in a real browser, with no submission."""

from __future__ import annotations

from importlib.metadata import version
from urllib.parse import urlsplit


def run_qec_resource_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Switch decoder settings and views through the production Analyse stage.

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
        scenario="qec_resource_instrument",
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
            instrument = page.get_by_role("region", name="QEC and future resources", exact=True)
            expect(instrument).to_be_visible()
            decoder = instrument.get_by_role("region", name="Decoder simulation")
            facts = decoder.get_by_label("Selected decoder sample")
            decoder.get_by_label("Code distance").select_option("5")
            decoder.get_by_label("Physical error rate").select_option("0.003")
            expect(facts).to_contain_text("0 of 2000")
            expect(facts).to_contain_text("not zero")
            observations.append("zero-failure-interval-states-its-trials")
            decoder.get_by_label("Physical error rate").select_option("0.1")
            expect(facts).not_to_contain_text("0 of 2000")
            observations.append("switching-noise-changes-the-source-sample")

            instrument.get_by_role("tab", name="Offline readiness").click()
            expect(
                instrument.get_by_role("status", name="Hardware logical-error reduction")
            ).to_have_text("Hardware logical-error reduction: not measured")
            observations.append("hardware-reduction-stays-unmeasured")

            instrument.get_by_role("tab", name="Future resources").click()
            excluded = instrument.get_by_role("list", name="Excluded overheads")
            expect(excluded).to_contain_text("magic-state factory")
            expect(excluded).to_contain_text("routing")
            rates = instrument.get_by_label("Assumed physical error rate")
            rates.select_option(label="0.01")
            expect(
                instrument.get_by_role("alert", name="Resource estimate refused")
            ).to_contain_text("threshold")
            observations.append("excluded-overheads-and-threshold-refusal-shown")

            page.goto(url + "#/results/qec-resources", wait_until="networkidle")
            expect(page.locator('[id="/results/qec-resources"]')).to_be_focused()
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
