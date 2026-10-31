# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — sensing and entropy instrument through the built Studio
"""Exercise the sensing and entropy instrument in a real browser, with no submission."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote, urlsplit

_METADATA_PREFIX = "data:application/json;charset=utf-8,"


def run_sensing_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Inspect source scores and export public metadata through the production Analyse stage.

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
        scenario="sensing_entropy_instrument",
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
            instrument = page.get_by_role(
                "region", name="Sensing and entropy provenance", exact=True
            )
            expect(instrument).to_be_visible()
            expect(
                instrument.get_by_role("table", name="Measurement-design candidates")
            ).to_contain_text("sync_order")
            expect(instrument.get_by_label("Entropy provenance", exact=True)).to_contain_text(
                "synthetic"
            )
            expect(instrument.get_by_role("note")).to_contain_text("Source entropy: unqualified")
            expect(instrument.get_by_label("Recorded diagnostic outcomes")).to_contain_text(
                "not independently verified"
            )
            link = instrument.get_by_role("link", name="Download public entropy metadata")
            metadata = json.loads(
                unquote((link.get_attribute("href") or "").removeprefix(_METADATA_PREFIX))
            )
            assert metadata["source_kind"] == "synthetic"
            assert metadata["source_entropy_qualified"] is False
            assert set(metadata) == {
                "case_id",
                "source_kind",
                "source",
                "extractor",
                "assumptions",
                "payload_sha256",
                "diagnostics",
                "source_entropy_qualified",
                "claim_boundary",
            }
            observations.append("synthetic-scores-and-noncertifying-metadata")
            instrument.get_by_label("Entropy record", exact=True).select_option(
                label="device-unavailable"
            )
            expect(instrument.get_by_role("status")).to_contain_text("device evidence unavailable")
            expect(instrument.get_by_role("list", name="Evidence limitations")).to_contain_text(
                "Missing source provenance"
            )
            metadata = json.loads(
                unquote((link.get_attribute("href") or "").removeprefix(_METADATA_PREFIX))
            )
            assert metadata["source_kind"] == "device" and metadata["source"] is None
            assert (
                metadata["diagnostics"] is None and metadata["source_entropy_qualified"] is False
            )
            observations.append("device-provenance-missing-without-fabricated-samples")
            page.goto(url + "#/results/sensing", wait_until="networkidle")
            expect(page.locator('[id="/results/sensing"]')).to_be_focused()
            observations.append("deep-link-focuses-the-instrument")
            axe_source = (
                Path(__file__).resolve().parents[1] / "studio-web/node_modules/axe-core/axe.min.js"
            )
            page.add_script_tag(path=str(axe_source))
            accessibility: list[dict[str, object]] = []
            for width in (1280, 390):
                page.set_viewport_size({"width": width, "height": 900})
                violations = page.evaluate("""async () => {
                    const root = document.querySelector('[aria-label="Sensing and entropy provenance"]');
                    const result = await axe.run(root);
                    return result.violations.map(item => ({id: item.id, nodes: item.nodes.length}));
                }""")
                assert not violations, violations
                accessibility.append({"width": width, "violations": violations})
            observed["accessibility"] = accessibility
            observations.append("accessible-at-desktop-and-mobile")
            assert not errors, errors
            assert not external, external
            assert not submissions, submissions
            assert not page.workers, "The instrument must not retain workers"
            observed.update(browser=browser.version, workers=len(page.workers), no_submit=True)
            return observed
        finally:
            context.close()
            browser.close()
