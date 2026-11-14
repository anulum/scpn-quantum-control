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
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

_METADATA_PREFIX = "data:application/json;charset=utf-8,"


def _exported_metadata(link: Locator) -> dict[str, object]:
    """Decode the public entropy metadata the download link carries.

    Parameters
    ----------
    link
        The instrument's metadata download link.

    Returns
    -------
    dict
        Decoded metadata fields.

    """
    href = link.get_attribute("href") or ""
    decoded: dict[str, object] = json.loads(unquote(href.removeprefix(_METADATA_PREFIX)))
    return decoded


def _check_instrument(page: Page, observations: list[str]) -> None:
    """Check every instrument contract on the Analyse stage.

    Parameters
    ----------
    page
        Owned page showing the Analyse stage.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    instrument = page.get_by_role("region", name="Sensing and entropy provenance", exact=True)
    expect(instrument).to_be_visible()
    expect(instrument.get_by_role("table", name="Measurement-design candidates")).to_contain_text(
        "sync_order"
    )
    expect(instrument.get_by_label("Entropy provenance", exact=True)).to_contain_text("synthetic")
    expect(instrument.get_by_role("note")).to_contain_text("Source entropy: unqualified")
    expect(instrument.get_by_label("Recorded diagnostic outcomes")).to_contain_text(
        "not independently verified"
    )
    link = instrument.get_by_role("link", name="Download public entropy metadata")
    metadata = _exported_metadata(link)
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
    instrument.get_by_label("Entropy record", exact=True).select_option(label="device-unavailable")
    expect(instrument.get_by_role("status")).to_contain_text("device evidence unavailable")
    expect(instrument.get_by_role("list", name="Evidence limitations")).to_contain_text(
        "Missing source provenance"
    )
    metadata = _exported_metadata(link)
    assert metadata["source_kind"] == "device" and metadata["source"] is None
    assert metadata["diagnostics"] is None and metadata["source_entropy_qualified"] is False
    observations.append("device-provenance-missing-without-fabricated-samples")


def _walk(page: Page, url: str, observations: list[str]) -> dict[str, object]:
    """Walk the instrument on the owned page and check each contract.

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
        Extra evidence fields for the journey record.

    """
    from playwright.sync_api import expect

    extra: dict[str, object] = {}
    page.goto(url, wait_until="networkidle")
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name="Analyse", exact=True
    ).click()
    _check_instrument(page, observations)
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
    extra["accessibility"] = accessibility
    observations.append("accessible-at-desktop-and-mobile")
    return extra


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
    return run_custodied_journey(base_url, evidence, "sensing_entropy_instrument", _walk)
