# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — control proposals and co-design through the built Studio
"""Exercise the control proposal instrument in a real browser, with no submission."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit


def run_control_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Inspect source control proposals through the production Analyse stage.

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
        scenario="control_proposal_instrument",
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
                "region", name="Control proposals and co-design", exact=True
            )
            expect(instrument).to_be_visible()
            targets = instrument.get_by_role("table", name="Tracking targets")
            expect(targets.get_by_role("row", name="signed target")).to_contain_text(
                "Proposed, not applied"
            )
            expect(instrument.get_by_role("note")).to_contain_text(
                "only with explicit runtime actuation evidence"
            )
            observations.append("signed-target-objective-proposed-not-applied")
            graph = instrument.get_by_role("img", name="Coupling graph constraints")
            edge = graph.locator('line[data-edge="0-1"]')
            expect(edge).to_have_attribute("data-state", "binding")
            selector = instrument.get_by_label("Co-design proposal", exact=True)
            selector.select_option(label="desired plan, edge 0-1 over its cap")
            expect(edge).to_have_attribute("data-state", "violated")
            observations.append("graph-highlights-binding-and-violated-caps")
            outcome = instrument.get_by_label("Proposal outcome", exact=True)
            for label in ("plan outside the tested domain", "hardware-mode request"):
                selector.select_option(label=label)
                expect(outcome).to_contain_text("No, no runtime actuation evidence")
            expect(outcome).to_contain_text("vetoed")
            observations.append("vetoed-and-out-of-domain-never-applied")
            trace = instrument.get_by_role("table", name="Simulated closed-loop trace")
            expect(trace).to_contain_text("0.8999999999999999")
            expect(instrument.get_by_role("status")).to_contain_text(
                "Observed hardware telemetry: unavailable"
            )
            expect(
                instrument.get_by_role("table", name="Latency budget verdicts")
            ).to_contain_text("latency claim refused")
            observations.append("integrator-timestamps-and-refused-latency-claim")
            page.goto(url + "#/results/control", wait_until="networkidle")
            expect(page.locator('[id="/results/control"]')).to_be_focused()
            observations.append("deep-link-focuses-the-instrument")
            axe_source = (
                Path(__file__).resolve().parents[1] / "studio-web/node_modules/axe-core/axe.min.js"
            )
            page.add_script_tag(path=str(axe_source))
            accessibility: list[dict[str, object]] = []
            for width in (1280, 390):
                page.set_viewport_size({"width": width, "height": 900})
                violations = page.evaluate("""async () => {
                    const root = document.querySelector('[aria-label="Control proposals and co-design"]');
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
