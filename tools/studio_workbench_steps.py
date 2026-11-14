# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shared workbench steps of the operations browser journeys
"""Shared steps of the Studio journeys that inspect operations panels.

The operator policy, operator dossier and provider observation journeys keep an
unsaved workspace draft, record every IndexedDB write the page attempts, open
their panel from the context destinations by keyboard, and finish by
recomputing the built WebAssembly claim and finding the draft unchanged. These
steps live here once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

_WRITE_PROBE = """() => {
    window.%s = [];
    for (const name of ['put','add','delete','clear']) {
        const native = IDBObjectStore.prototype[name];
        IDBObjectStore.prototype[name] = function(...args) {
            window.%s.push(name);
            return native.apply(this,args);
        };
    }
}"""


def open_operations_panel(
    page: Page, url: str, draft: str, record: str, panel: str, probe: str | None = None
) -> Locator:
    """Keep a draft, probe storage writes and open an operations panel by keyboard.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    draft
        Unsaved workspace text that must survive the journey.
    record
        Window property that collects the attempted IndexedDB writes.
    panel
        Accessible name of the operations panel.
    probe
        Optional probe script that fills ``record``; by default every attempted
        IndexedDB write is recorded by name.

    Returns
    -------
    Locator
        The visible panel.

    """
    from playwright.sync_api import expect

    page.goto(url, wait_until="networkidle")
    page.get_by_label("Workspace archive JSON").fill(draft)
    page.evaluate(_WRITE_PROBE % (record, record) if probe is None else probe)
    destination = page.get_by_role("navigation", name="Workbench context destinations")
    destination.get_by_role("link", name="Devices & Operations").focus()
    page.keyboard.press("Enter")
    region = page.get_by_role("region", name=panel, exact=True)
    expect(region).to_be_visible()
    return region


def storage_writes(page: Page, record: str) -> object:
    """Return the IndexedDB writes the probe recorded.

    Parameters
    ----------
    page
        Owned page carrying the probe.
    record
        Window property the probe fills.

    Returns
    -------
    object
        The recorded write operation names.

    """
    return page.evaluate(f"window.{record}")


def check_recompute_and_draft(
    page: Page,
    draft: str,
    observations: list[str],
    draft_label: str = "original-workspace-draft-preserved",
) -> None:
    """Recompute the built WASM claim in Model and find the draft kept in Workspace.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    draft
        Workspace text kept since the start of the journey.
    observations
        Caller-owned list that records each contract once it holds.
    draft_label
        Observation recorded once the draft is found unchanged.

    """
    from playwright.sync_api import expect

    views = page.get_by_role("navigation", name="Workbench views")
    views.get_by_role("link", name="Model", exact=True).click()
    recompute = page.locator('[id="/build/compile-recompute"]')
    recompute.get_by_role("button", name="Recompute in browser").click()
    expect(recompute.get_by_role("status")).to_have_text(
        "recomputed digest matches the signed claim"
    )
    observations.append("original-built-wasm-recompute")
    views.get_by_role("link", name="Workspace", exact=True).click()
    expect(page.get_by_label("Workspace archive JSON")).to_have_value(draft)
    observations.append(draft_label)
