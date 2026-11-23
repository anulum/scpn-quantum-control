# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Atlas evidence lineage through the built Studio
"""Follow claims, a negative pack and a failed replication through the built Atlas stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
AXE_SOURCE: Final[Path] = _ROOT / "studio-web/node_modules/axe-core/axe.min.js"
LINEAGE_PATH: Final[Path] = _ROOT / "data/studio/evidence_lineage_20261123.json"
NEGATIVE_PACK: Final[str] = "scpn_fim_negative_ibm_kingston_2026_05_05"


def journey_subjects(path: Path = LINEAGE_PATH) -> dict[str, str]:
    """Pick the committed records the journey opens.

    Parameters
    ----------
    path
        The committed evidence lineage.

    Returns
    -------
    dict[str, str]
        The text of a claim with resolved evidence and its first file's digest, the text of
        a claim with missing evidence, the negative pack's title and the replication label.

    """
    lineage: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    claims: list[dict[str, Any]] = lineage["claims"]
    resolved = next(c for c in claims if any(e["state"] == "resolved" for e in c["evidence"]))
    missing = next(c for c in claims if any(e["state"] == "missing" for e in c["evidence"]))
    first = next(e for e in resolved["evidence"] if e["state"] == "resolved")
    pack = next(p for p in lineage["packs"] if p["id"] == NEGATIVE_PACK)
    replication = lineage["replications"][0]
    decision = replication["decision"]
    return {
        "resolved_claim": resolved["claim_text"],
        "resolved_sha256": first["candidates"][0]["sha256"],
        "missing_claim": missing["claim_text"],
        "negative_pack": pack["title"],
        "replication": (
            f"Replication {replication['campaign']}: {decision['field']} = "
            f"{'true' if decision['value'] else 'false'}"
        ),
    }


def _open(page: Page, url: str, lens: str) -> Locator:
    """Open the Atlas stage on one lens and wait for its answer.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    lens
        Instrument of the lens.

    Returns
    -------
    Locator
        The Atlas stage.

    """
    from playwright.sync_api import expect

    page.goto(f"{url}#/atlas/{lens}", wait_until="networkidle")
    atlas = page.get_by_role("article", name="Atlas")
    expect(atlas.get_by_role("status", name="Atlas summary")).to_contain_text(
        "Showing", timeout=30000
    )
    return atlas


def _select(atlas: Locator, label: str) -> Locator:
    """Select a node by its table label and return the evidence lineage panel.

    Parameters
    ----------
    atlas
        The Atlas stage.
    label
        The node's label.

    Returns
    -------
    Locator
        The lineage panel of the selection.

    """
    atlas.get_by_role("table", name="Atlas nodes").get_by_role(
        "button", name=label, exact=True
    ).click()
    return atlas.get_by_role("region", name="Evidence lineage")


def _claims(atlas: Locator, subjects: dict[str, str], observations: list[str]) -> None:
    """Check a claim's statuses, its verified record and its exact and missing evidence.

    Parameters
    ----------
    atlas
        The Atlas stage on the claims lens.
    subjects
        The records the journey opens.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    lineage = _select(atlas, subjects["resolved_claim"])
    expect(lineage.get_by_label("Claim statuses")).to_contain_text("bounded_candidate")
    expect(lineage.get_by_label("Claim statuses")).to_contain_text("Absent:")
    expect(lineage.locator('[data-axis="Claim status"]')).to_have_text("bounded-model")
    expect(lineage).not_to_contain_text("reference-validated")
    observations.append("unvalidated-claim-keeps-its-statuses")
    lineage.get_by_role("button", name="Verify lineage record").click()
    expect(lineage.locator('[data-verdict="match"]')).to_be_visible(timeout=30000)
    observations.append("lineage-record-digest-verified-in-the-browser")
    evidence = lineage.get_by_role("table", name="Declared evidence")
    expect(evidence).to_contain_text(subjects["resolved_sha256"])
    lineage = _select(atlas, subjects["missing_claim"])
    expect(lineage.get_by_role("table", name="Declared evidence")).to_contain_text(
        "Missing custody"
    )
    observations.append("claim-opens-exact-hashes-and-missing-custody")


def _evidence(atlas: Locator, subjects: dict[str, str], observations: list[str]) -> None:
    """Find the negative pack and the failed replication with their statuses verbatim.

    Parameters
    ----------
    atlas
        The Atlas stage on the evidence lens.
    subjects
        The records the journey opens.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    lineage = _select(atlas, subjects["negative_pack"])
    expect(lineage.get_by_label("Pack statuses")).to_contain_text("promoted_negative_result")
    expect(lineage.locator('[data-axis="Claim status"]')).to_have_text("bounded-support")
    observations.append("negative-pack-shows-both-statuses-verbatim")
    atlas.get_by_label("Search").fill("replicates = false")
    summary = atlas.get_by_role("status", name="Atlas summary")
    expect(summary).to_have_text("Showing 1 of 1 nodes.")
    lineage = _select(atlas, subjects["replication"])
    expect(lineage.get_by_label("Replication decision")).to_contain_text("false")
    observations.append("failed-replication-is-searchable")


def _accessibility(page: Page) -> list[dict[str, object]]:
    """Run the pinned axe rules on the lineage panel at desktop and mobile width.

    Parameters
    ----------
    page
        Playwright page showing a claim's lineage panel.

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
            const root = document.querySelector('[aria-label="Evidence lineage"]');
            const result = await axe.run(root);
            return result.violations.map(item => ({id: item.id, nodes: item.nodes.length}));
        }""")
        assert not violations, violations
        accessibility.append({"width": width, "violations": violations})
    page.set_viewport_size({"width": 1280, "height": 900})
    return accessibility


def _walk(page: Page, url: str, observations: list[str]) -> dict[str, object]:
    """Follow the claims lens, audit its lineage panel, then follow the evidence lens.

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
        The records opened and the accessibility results.

    """
    subjects = journey_subjects()
    _claims(_open(page, url, "claims"), subjects, observations)
    accessibility = _accessibility(page)
    observations.append("accessible-at-desktop-and-mobile")
    _evidence(_open(page, url, "evidence"), subjects, observations)
    return {"subjects": subjects, "accessibility": accessibility}


def run_atlas_lineage_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Follow the evidence lineage through the production Atlas stage.

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
    return run_custodied_journey(base_url, evidence, "atlas_evidence_lineage", _walk)
