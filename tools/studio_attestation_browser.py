# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — attestation verification through the built Studio
"""Verify attested result-pack envelopes in a real browser, with nothing uploaded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import FilePayload, Locator, Page

ROOT = Path(__file__).resolve().parents[1]
AXE_SOURCE = ROOT / "studio-web/node_modules/axe-core/axe.min.js"
DATA = ROOT / "data/studio/attestation_instrument_20261115.json"
FILE_LABEL = "Envelope file (attested_result_pack.v1 JSON, read only here)"


def envelope_file(*, changed: bool) -> bytes:
    """Return the sealed committed envelope as a file, optionally with one value changed.

    Parameters
    ----------
    changed
        Change one content value after sealing, keeping the carried digest.

    Returns
    -------
    bytes
        UTF-8 JSON in the owner's spelling, so ``0.0`` stays a float.

    """
    envelope = json.loads(DATA.read_text(encoding="utf-8"))["cases"][0]["envelope"]
    if changed:
        envelope["content"]["C_fim"]["mean"] = 0.9159
    return json.dumps(envelope).encode("utf-8")


def _check_cases(instrument: Locator, observations: list[str]) -> None:
    """Check that every committed case is decided here as the Python owner decided it.

    Parameters
    ----------
    instrument
        The attestation instrument.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    table = instrument.get_by_role("table", name="Attestation cases")
    expect(table.locator('[data-agrees="true"]')).to_have_count(7)
    expect(table.locator('tr[data-case="intact"] [data-status="VERIFIED"]')).to_have_count(2)
    expect(table.locator('tr[data-case="changed-value"] [data-status="FORGED"]')).to_have_count(2)
    expect(table.locator('tr[data-case="stripped-axes"] [data-status="STRIPPED"]')).to_have_count(
        2
    )
    observations.append("browser-verdicts-equal-python-verdicts")
    expect(instrument.get_by_label("Canonical form")).to_contain_text("for all 6 samples")
    observations.append("canonical-form-equals-python-text")


def _check_file(instrument: Locator, observations: list[str]) -> None:
    """Verify a local envelope file, then the same file with a changed value as FORGED.

    Parameters
    ----------
    instrument
        The attestation instrument.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    chooser = instrument.get_by_label(FILE_LABEL)
    outcome = instrument.get_by_label("Check an envelope file").get_by_role("status")
    for name, changed, status in (("sealed", False, "VERIFIED"), ("changed", True, "FORGED")):
        payload: FilePayload = {
            "name": f"{name}.json",
            "mimeType": "application/json",
            "buffer": envelope_file(changed=changed),
        }
        chooser.set_input_files(files=[payload])
        expect(outcome).to_contain_text(status)
    observations.append("local-file-verified-and-changed-file-forged")


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
            const root = document.querySelector('[aria-label="Attestation verification"]');
            const result = await axe.run(root);
            return result.violations.map(item => ({id: item.id, nodes: item.nodes.length}));
        }""")
        assert not violations, violations
        accessibility.append({"width": width, "violations": violations})
    return accessibility


def _walk(page: Page, url: str, observations: list[str]) -> dict[str, object]:
    """Open Evidence, check the instrument, its deep link and its accessibility.

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
    instrument = page.get_by_role("article", name="Attestation verification")
    expect(instrument).to_be_visible()
    _check_cases(instrument, observations)
    _check_file(instrument, observations)
    page.goto(url + "#/evidence/attestation", wait_until="networkidle")
    expect(page.locator('[id="/evidence/attestation"]')).to_be_focused()
    observations.append("deep-link-focuses-the-instrument")
    accessibility = _accessibility(page)
    observations.append("accessible-at-desktop-and-mobile")
    return {"accessibility": accessibility}


def run_attestation_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Verify attested envelopes through the production Evidence stage.

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
    return run_custodied_journey(base_url, evidence, "attestation_verification", _walk)
