# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — mitigation and readout instrument through the built Studio
"""Exercise the mitigation and readout instrument in a real browser, with no submission."""

from __future__ import annotations

from importlib.metadata import version
from urllib.parse import unquote, urlsplit

_CSV_PREFIX = "data:text/csv;charset=utf-8,"


def run_mitigation_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Switch readout cases and views through the production Analyse stage.

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
        scenario="mitigation_inspector",
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
            instrument = page.get_by_role("region", name="Mitigation and readout", exact=True)
            expect(instrument).to_be_visible()
            readout = instrument.get_by_role("region", name="Readout correction")
            readout.get_by_label("Readout case").select_option(label="permuted-partial-map")
            expect(readout.get_by_label("Measurement map")).to_have_text("c0←q2, c1←q0")
            csv = unquote(
                (
                    readout.get_by_role("link", name="Download CSV").get_attribute("href") or ""
                ).removeprefix(_CSV_PREFIX)
            )
            assert '# measurement map,"c0←q2, c1←q0"' in csv, csv
            observations.append("permuted-map-kept-in-view-and-csv")
            readout.get_by_label("Readout case").select_option(
                label="negative-quasi-probabilities"
            )
            expect(readout.get_by_role("status", name="Signed values")).to_contain_text(
                "Negative at 01, 10"
            )
            observations.append("negative-values-stay-signed")
            readout.get_by_label("Readout case").select_option(label="singular-calibration")
            expect(readout.get_by_role("alert", name="Correction refused")).to_contain_text(
                "condition"
            )
            observations.append("refused-calibration-keeps-raw-counts")

            instrument.get_by_role("tab", name="Zero-noise extrapolation").click()
            expect(instrument.get_by_label("Extrapolation", exact=True)).to_contain_text(
                "gls-delta"
            )
            observations.append("extrapolation-shows-raw-and-corrected")
            instrument.get_by_role("tab", name="Hardware pairs").click()
            expect(instrument.get_by_role("note", name="Correction caveat")).to_contain_text(
                "not all computational basis states"
            )
            observations.append("hardware-caveat-shown")

            page.goto(url + "#/results/mitigation", wait_until="networkidle")
            expect(page.locator('[id="/results/mitigation"]')).to_be_focused()
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
