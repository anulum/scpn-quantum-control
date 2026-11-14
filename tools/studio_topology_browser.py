# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — symmetry, chimera and topology lenses through the built Studio
"""Exercise the topology instrument in a real browser, with no submission."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

INSTRUMENT = "Symmetry, chimera and topology lenses"
PHYSICS = (
    "Order parameter under relabelling",
    "Symmetry sectors",
    "Cluster order parameters",
    "Persistence summaries",
)
AXE_SOURCE = Path(__file__).resolve().parents[1] / "studio-web/node_modules/axe-core/axe.min.js"


def _check_derivatives(instrument: Locator, observations: list[str]) -> None:
    """Require a JVP only at the interior point and the owner's refusal elsewhere.

    Parameters
    ----------
    instrument
        The rendered instrument region.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    select = instrument.get_by_label("Primal point", exact=True)
    jvp = instrument.get_by_role("table", name="Local JVP for the fixed active set")
    expect(jvp).to_be_visible()
    for case, blocker in (
        ("coupling at the clipping bound", "uniform_bounds"),
        ("edge removed", "sign_policy, uniform_bounds"),
    ):
        select.select_option(label=case)
        expect(jvp).to_have_count(0)
        expect(instrument.get_by_role("status")).to_have_text(
            f"No derivative is shown: projection derivative is unsupported for: {blocker}."
        )
    select.select_option(label="interior point")
    expect(jvp).to_be_visible()
    observations.append("kink-and-discrete-edit-show-no-derivative")


def _check_rotation(instrument: Locator, observations: list[str]) -> None:
    """Rotate the display projection and require every physics value unchanged.

    Parameters
    ----------
    instrument
        The rendered instrument region.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    before = [instrument.get_by_label(name, exact=True).inner_text() for name in PHYSICS]
    node = instrument.locator('[data-node="0"]')
    start = node.get_attribute("cx")
    instrument.get_by_label("Rotate view (degrees)", exact=True).press("End")
    instrument.get_by_label("Tilt view (degrees)", exact=True).press("Home")
    expect(node).not_to_have_attribute("cx", str(start))
    after = [instrument.get_by_label(name, exact=True).inner_text() for name in PHYSICS]
    assert after == before, (before, after)
    observations.append("rotation-changes-no-physics-value")


def _check_instrument(page: Page, observations: list[str]) -> None:
    """Assert the four topology contracts on the rendered instrument.

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
    expect(instrument.get_by_label("Relabelling", exact=True)).to_contain_text("0→2, 1→0")
    observations.append("order-parameter-kept-under-relabelling")
    spectra = instrument.get_by_label("Sector spectra against the dense reference", exact=True)
    expect(spectra).to_contain_text("even 8, odd 8")
    observations.append("finite-sectors-match-dense-reference")
    _check_derivatives(instrument, observations)
    _check_rotation(instrument, observations)
    expect(instrument).to_contain_text("global random state")
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
            const root = document.querySelector('[aria-label="Symmetry, chimera and topology lenses"]');
            const result = await axe.run(root);
            return result.violations.map(item => ({id: item.id, nodes: item.nodes.length}));
        }""")
        assert not violations, violations
        accessibility.append({"width": width, "violations": violations})
    return accessibility


def _walk(page: Page, url: str, observations: list[str]) -> dict[str, object]:
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
    dict
        Accessibility results per viewport width.

    """
    from playwright.sync_api import expect

    page.goto(url, wait_until="networkidle")
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name="Analyse", exact=True
    ).click()
    _check_instrument(page, observations)
    page.goto(url + "#/results/topology", wait_until="networkidle")
    expect(page.locator('[id="/results/topology"]')).to_be_focused()
    observations.append("deep-link-focuses-the-instrument")
    accessibility = _accessibility(page)
    observations.append("accessible-at-desktop-and-mobile")
    return {"accessibility": accessibility}


def run_topology_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Inspect source topology evidence through the production Analyse stage.

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
    return run_custodied_journey(base_url, evidence, "topology_symmetry_instrument", _walk)
