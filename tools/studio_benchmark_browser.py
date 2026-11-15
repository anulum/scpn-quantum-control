# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark databank through the built Studio
"""Read the verified tier benchmark databank in a real browser, with no submission."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

AXE_SOURCE = Path(__file__).resolve().parents[1] / "studio-web/node_modules/axe-core/axe.min.js"


def _check_databank(databank: Locator, observations: list[str]) -> None:
    """Check custody, provenance, the latency table and the operation choice.

    Parameters
    ----------
    databank
        The benchmark databank instrument.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    expect(databank.get_by_role("status")).to_contain_text(
        "verified in this browser", timeout=30000
    )
    observations.append("file-and-payload-digests-verified-in-the-browser")
    expect(databank.get_by_role("table", name="Benchmark provenance")).to_contain_text(
        "AMD EPYC 7763"
    )
    table = databank.get_by_role("table", name="Median latency by size")
    expect(table.locator("tbody tr")).to_have_count(4)
    expect(table.locator('[data-fastest="none"]')).to_have_count(0)
    expect(databank.get_by_role("list", name="Unavailable measurements")).to_contain_text(
        "Local julia: tier excluded by --tiers"
    )
    observations.append("provenance-latency-and-unavailable-tiers-shown")
    databank.get_by_label("Operation").select_option("kuramoto_rk4_trajectory")
    expect(table.locator("caption")).to_contain_text("kuramoto_rk4_trajectory")
    observations.append("operation-choice-changes-the-table")


def _accessibility(page: Page) -> list[dict[str, object]]:
    """Run the pinned axe rules on the databank at desktop and mobile width.

    Parameters
    ----------
    page
        Playwright page showing the focused databank.

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
            const root = document.querySelector('[aria-label="Benchmark databank"]');
            const result = await axe.run(root);
            return result.violations.map(item => ({id: item.id, nodes: item.nodes.length}));
        }""")
        assert not violations, violations
        accessibility.append({"width": width, "violations": violations})
    return accessibility


def _walk(page: Page, url: str, observations: list[str]) -> dict[str, object]:
    """Open Evidence, check the databank, its deep link and its accessibility.

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
    dict
        Accessibility results per viewport width.

    """
    from playwright.sync_api import expect

    page.goto(url, wait_until="networkidle")
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name="Evidence", exact=True
    ).click()
    _check_databank(page.get_by_role("article", name="Benchmark databank"), observations)
    page.goto(url + "#/evidence/benchmarks", wait_until="networkidle")
    expect(page.locator('[id="/evidence/benchmarks"]')).to_be_focused()
    observations.append("deep-link-focuses-the-instrument")
    expect(
        page.get_by_role("article", name="Benchmark databank").get_by_role("status")
    ).to_contain_text("verified in this browser", timeout=30000)
    accessibility = _accessibility(page)
    observations.append("accessible-at-desktop-and-mobile")
    return {"accessibility": accessibility}


def run_benchmark_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Read the verified benchmark databank through the production Evidence stage.

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
    return run_custodied_journey(base_url, evidence, "benchmark_databank", _walk)
