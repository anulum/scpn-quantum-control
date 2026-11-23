# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Atlas selection linked to Studio drafts through the built Studio
"""Select a workspace revision in the Atlas, draft from it in the Studio and come back."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Final

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
AXE_SOURCE: Final[Path] = _ROOT / "studio-web/node_modules/axe-core/axe.min.js"
_REVISION: Final[re.Pattern[str]] = re.compile(r"^Experiment revision [0-9a-f]{12}$")
_SELECTED: Final[re.Pattern[str]] = re.compile(r"revision:sha256:([0-9a-f]{64})")
_PROJECT: Final[re.Pattern[str]] = re.compile(
    r'"project_id":\s*"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"'
)


def private_identifiers(archive: str, selected: str) -> list[str]:
    """Return the private identifiers no shared link may carry.

    Parameters
    ----------
    archive
        The workspace archive text.
    selected
        The text naming the selected revision.

    Returns
    -------
    list[str]
        The project UUIDs of the archive and the revision hashes the text names.

    """
    return sorted({*_PROJECT.findall(archive), *_SELECTED.findall(selected)})


def _revision_named(text: str) -> str:
    """Return the first revision hash a text names.

    Parameters
    ----------
    text
        Text naming ``revision:sha256:<hash>``.

    Returns
    -------
    str
        The hash.

    """
    found: list[str] = _SELECTED.findall(text)
    return found[0]


def _open_sample(page: Page, url: str) -> str:
    """Open the Kuramoto sample in the Studio and return its workspace archive.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.

    Returns
    -------
    str
        The workspace archive text the Atlas overlays.

    """
    from playwright.sync_api import expect

    page.goto(f"{url}#/experiments", wait_until="networkidle")
    experiment = page.get_by_label("Local experiment", exact=True)
    experiment.get_by_role("button", name="Open Kuramoto sample").click()
    experiment.get_by_role("button", name="Validate experiment archive").click()
    expect(experiment.get_by_role("button", name="Prepare numerical plan")).to_be_enabled()
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name="Workspace", exact=True
    ).click()
    return page.get_by_label("Workspace archive JSON").input_value()


def _atlas(page: Page, url: str) -> Locator:
    """Open the Atlas Workspace lens and wait for the overlaid workspace.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.

    Returns
    -------
    Locator
        The Atlas stage.

    """
    from playwright.sync_api import expect

    page.get_by_role("navigation", name="Workbench context destinations").get_by_role(
        "link", name="Atlas", exact=True
    ).click()
    atlas = page.get_by_role("article", name="Atlas")
    atlas.get_by_role("navigation", name="Atlas lenses").get_by_role(
        "link", name="Workspace", exact=True
    ).click()
    expect(atlas.get_by_role("region", name="Atlas snapshot")).to_contain_text(
        "with your workspace", timeout=30000
    )
    expect(page).to_have_url(re.compile(re.escape(f"{url}#/atlas/workspace")))
    return atlas


def _select_revision(page: Page, atlas: Locator, archive: str, observations: list[str]) -> str:
    """Select the sample's revision and check that no shared link names it.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    atlas
        The Atlas stage on the Workspace lens.
    archive
        The workspace archive text.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    str
        The selected revision hash.

    """
    from playwright.sync_api import expect

    rows = atlas.get_by_role("table", name="Atlas nodes").get_by_role("button", name=_REVISION)
    expect(rows).to_have_count(1, timeout=30000)
    rows.click()
    selected = page.get_by_role("region", name="Selected object")
    expect(selected).to_contain_text("selected in atlas")
    revision = _revision_named(selected.inner_text())
    observations.append("atlas-selection-reaches-the-studio-inspector")
    link = atlas.get_by_role("link", name="Link to this view").get_attribute("href") or ""
    for secret in private_identifiers(archive, selected.inner_text()):
        assert secret not in page.url, secret
        assert secret not in link, secret
    observations.append("private-identity-absent-from-shared-urls")
    return revision


def _draft(page: Page, atlas: Locator, revision: str, observations: list[str]) -> str:
    """Open the selected revision as a new draft, save a child and return its hash.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    atlas
        The Atlas stage with the revision selected.
    revision
        The selected revision hash.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    str
        The saved child revision hash.

    """
    from playwright.sync_api import expect

    atlas.get_by_role("region", name="Revision draft").get_by_role(
        "link", name="Open as a new parameter draft"
    ).click()
    editor = page.get_by_label("Linked parameter editor", exact=True)
    expect(
        page.get_by_text(f"Draft from revision {revision[:12]}, selected in atlas.")
    ).to_be_visible(timeout=30000)
    editor.get_by_role("button", name="coupling[scalar]", exact=False).click()
    editor.get_by_label("Selected value").fill("1.4")
    editor.get_by_role("button", name="Apply value", exact=True).click()
    editor.get_by_role("button", name="Save parameter revision").click()
    selected = page.get_by_role("region", name="Selected object")
    expect(selected).to_contain_text("selected in parameters", timeout=30000)
    child = _revision_named(selected.inner_text())
    assert child != revision
    observations.append("draft-saves-a-child-and-keeps-the-revision")
    return child


def _return(page: Page, revision: str, child: str, observations: list[str]) -> Locator:
    """Follow the inspector back to the Atlas and find the child beside its parent.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    revision
        The selected revision hash.
    child
        The saved child revision hash.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    Locator
        The Atlas stage.

    """
    from playwright.sync_api import expect

    page.get_by_role("region", name="Selected object").get_by_role(
        "link", name="Show it in the Atlas"
    ).click()
    atlas = page.get_by_role("article", name="Atlas")
    shared = atlas.get_by_role("region", name="Shared selection")
    expect(shared).to_contain_text(
        f"Selected in parameters: Experiment revision {child[:12]} (revision).", timeout=30000
    )
    relations = atlas.get_by_role("table", name="Atlas relations table")
    expect(relations).to_contain_text(
        f"revision:sha256:{child}derives_fromrevision:sha256:{revision}"
    )
    observations.append("atlas-to-studio-to-atlas-keeps-identity")
    return atlas


def _accessibility(page: Page) -> list[dict[str, object]]:
    """Run the pinned axe rules on the Atlas inspector at desktop and mobile width.

    Parameters
    ----------
    page
        Playwright page showing the Atlas with a shared selection.

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
            const root = document.querySelector('[aria-label="Atlas"]');
            const result = await axe.run(root);
            return result.violations.map(item => ({id: item.id, nodes: item.nodes.length}));
        }""")
        assert not violations, violations
        accessibility.append({"width": width, "violations": violations})
    page.set_viewport_size({"width": 1280, "height": 900})
    return accessibility


def _walk(page: Page, url: str, observations: list[str]) -> dict[str, object]:
    """Draft from an Atlas selection and return to it, then audit the Atlas.

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
        The revisions followed and the accessibility results.

    """
    archive = _open_sample(page, url)
    atlas = _atlas(page, url)
    revision = _select_revision(page, atlas, archive, observations)
    child = _draft(page, atlas, revision, observations)
    _return(page, revision, child, observations)
    accessibility = _accessibility(page)
    observations.append("accessible-at-desktop-and-mobile")
    return {"revision": revision, "child": child, "accessibility": accessibility}


def run_atlas_selection_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Link an Atlas selection to a Studio draft through the production Studio.

    Parameters
    ----------
    base_url
        Owned literal-loopback preview carrying the actual built Studio.
    evidence
        Caller-owned partial evidence retained on a later runtime refusal.

    Returns
    -------
    dict
        Browser identity, every observed contract and the absence of retained
        workers, submissions, outside requests and page errors.

    Raises
    ------
    ValueError
        The preview violates the shared loopback ownership contract.
    AssertionError
        A real UI, network or runtime contract fails.

    """
    return run_custodied_journey(base_url, evidence, "atlas_studio_selection", _walk)
