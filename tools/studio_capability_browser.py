# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — capability states through the built Studio
"""Read every capability state on the Workspace stage of a real browser, with no submission."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

AXE_SOURCE = Path(__file__).resolve().parents[1] / "studio-web/node_modules/axe-core/axe.min.js"
#: Every capability the registered modules declare. A module that declares a new one must
#: add it here, so the list on the Workspace stage is checked by name, not only by count.
CAPABILITIES = frozenset(
    {
        "browse-atlas",
        "durable-local-jobs",
        "execute-on-hardware",
        "recompute-xy-compile",
        "replay-program-ad",
        "run-local-experiments",
        "save-projects",
        "verify-attestations",
        "verify-benchmarks",
    }
)


def _state(section: Locator, capability: str) -> str | None:
    """Return the resolved state of one listed capability."""
    return section.locator(f'[data-capability="{capability}"]').get_attribute("data-state")


def _check_before(section: Locator, observations: list[str]) -> None:
    """Check that every capability is listed and the kernels wait for the user's check.

    Parameters
    ----------
    section
        The capability list.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    listed = section.locator("[data-capability]")
    expect(listed).to_have_count(len(CAPABILITIES))
    names = listed.evaluate_all("items => items.map(item => item.dataset.capability)")
    assert set(names) == CAPABILITIES, sorted(set(names) ^ CAPABILITIES)
    for capability in ("save-projects", "browse-atlas"):
        expect(section.locator(f'[data-capability="{capability}"]')).to_have_attribute(
            "data-state", "stable"
        )
    assert _state(section, "recompute-xy-compile") == "degraded"
    assert _state(section, "execute-on-hardware") == "unavailable"
    assert _state(section, "durable-local-jobs") == "unavailable"
    expect(
        section.get_by_role("list", name="What Execute programs on quantum hardware is missing")
    ).to_contain_text("separate gated service integration")
    expect(
        section.get_by_role(
            "list", name="What Recompute the XY compile digest in the browser is missing"
        )
    ).to_contain_text("not checked yet")
    observations.append("every-capability-listed-and-kernels-wait-for-the-check")


def _check_after(section: Locator, observations: list[str]) -> None:
    """Check the runtimes and expect both shipped kernels to verify.

    Parameters
    ----------
    section
        The capability list.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    section.get_by_role("button", name="Check runtimes").click()
    for capability in ("recompute-xy-compile", "run-local-experiments", "replay-program-ad"):
        expect(section.locator(f'[data-capability="{capability}"]')).to_have_attribute(
            "data-state", "stable", timeout=30000
        )
    observations.append("runtime-check-verifies-both-kernels")


def _accessibility(page: Page) -> list[dict[str, object]]:
    """Run the pinned axe rules on the capability list at desktop and mobile width.

    Parameters
    ----------
    page
        Playwright page showing the Workspace stage.

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
            const root = document.querySelector('[aria-label="What this installation can do"]');
            const result = await axe.run(root);
            return result.violations.map(item => ({id: item.id, nodes: item.nodes.length}));
        }""")
        assert not violations, violations
        accessibility.append({"width": width, "violations": violations})
    return accessibility


def _walk(page: Page, url: str, observations: list[str]) -> dict[str, object]:
    """Open the Workspace stage, check the list before and after the runtime check.

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
    page.goto(url + "#/workspace", wait_until="networkidle")
    section = page.get_by_role("region", name="What this installation can do")
    _check_before(section, observations)
    _check_after(section, observations)
    accessibility = _accessibility(page)
    observations.append("accessible-at-desktop-and-mobile")
    return {"accessibility": accessibility}


def run_capability_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Read every capability state through the production Workspace stage.

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
    return run_custodied_journey(base_url, evidence, "capability_resolver", _walk)
