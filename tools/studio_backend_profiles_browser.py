# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual offline profile browser boundary
"""Inspect exact native metadata through the built Studio and genuine WASM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from tools.studio_preview_custody import run_custodied_journey
from tools.studio_workbench_steps import check_recompute_and_draft

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page


DRAFT: Final = '{"unsaved":"exact retained α draft"}'


@dataclass(frozen=True)
class _Fixtures:
    """Committed backend-profile sources used by the journey.

    Attributes
    ----------
    original
        Committed declared-profile source text.
    offline
        Compact offline envelope text.
    envelope
        Offline envelope digest.
    binding
        Offline binding references.
    current
        Profile row bound by the offline envelope.
    other
        Second profile row for the same device.

    """

    original: str
    offline: str
    envelope: str
    binding: dict[str, Any]
    current: dict[str, Any]
    other: dict[str, Any]


def _load_fixtures() -> _Fixtures:
    """Read the committed profile sources and pick the bound and the other row.

    Returns
    -------
    _Fixtures
        The sources the journey imports and compares.

    """
    repo = Path(__file__).resolve().parents[1]
    cases = json.loads((repo / "data/studio/backend_profiles_cases.json").read_text())
    binding = cases["offline"]["body"]["binding"]
    rows = cases["offline"]["body"]["profiles"]
    return _Fixtures(
        original=(repo / "data/studio/backend_profiles.json").read_text(),
        offline=json.dumps(cases["offline"], ensure_ascii=False),
        envelope=cases["offline"]["sha256"],
        binding=binding,
        current=next(row for row in rows if row["sha256"] == binding["profile_sha256"]),
        other=next(row for row in rows if row["sha256"] != binding["profile_sha256"]),
    )


def _open_profiles(page: Page, url: str) -> Locator:
    """Open the declared profiles from the context destinations by keyboard.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.

    Returns
    -------
    Locator
        The backend profiles panel.

    """
    from playwright.sync_api import expect

    page.goto(url, wait_until="networkidle")
    page.get_by_label("Workspace archive JSON").fill(DRAFT)
    destination = page.get_by_role("navigation", name="Workbench context destinations")
    destination.get_by_role("link", name="Devices & Operations").focus()
    page.keyboard.press("Enter")
    panel = page.get_by_role("region", name="Backend profiles", exact=True)
    expect(panel).to_be_visible()
    panel.get_by_role("button", name="Open declared profiles").click()
    return panel


def _check_declared(panel: Locator, observations: list[str]) -> None:
    """Show the offline declared profile with its age and unknown observations.

    Parameters
    ----------
    panel
        The backend profiles panel.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    select = panel.get_by_label("Backend profile", exact=True)
    expect(select).to_have_value("")
    select.select_option("direct/iqm")
    expect(panel.get_by_label("Online observation")).to_have_text("unknown")
    expect(panel.get_by_label("Observed shot ceiling")).to_have_text("unknown")
    expect(panel.get_by_role("button", name="Pulse options")).to_be_disabled()
    expect(panel.get_by_role("button", name="Analog options")).to_be_disabled()
    expect(panel.get_by_text("Snapshot date:", exact=False)).to_contain_text("Snapshot age:")
    observations.append("offline-declared-age-and-unknown")


def _check_import(panel: Locator, fx: _Fixtures, observations: list[str]) -> None:
    """Import the offline envelope, then refuse a malformed one without losing it.

    Parameters
    ----------
    panel
        The backend profiles panel.
    fx
        Committed sources.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    field = panel.get_by_label("Backend profiles JSON")
    field.fill(fx.offline)
    panel.get_by_role("button", name="Inspect profiles", exact=True).click()
    expect(panel.get_by_label("Plan reference")).to_have_text(fx.binding["plan_ref"])
    expect(panel.get_by_label("Bound calibration reference")).to_have_text(
        fx.binding["calibration_ref"]
    )
    expect(panel.get_by_label("Review reference")).to_have_text(fx.binding["approval_ref"])
    expect(panel.get_by_label("Backend profile", exact=True)).to_have_value(
        fx.current["body"]["route_id"]
    )
    expect(panel.get_by_label("Observed shot ceiling")).to_have_text("18446744073709551615")
    expect(panel.get_by_label("Online observation")).to_have_text("offline (supplied observation)")
    assert fx.current["body"]["device"] == fx.other["body"]["device"]
    assert fx.current["sha256"] != fx.other["sha256"]
    observations.append("two-brokers-same-device-exact-integer")
    field.fill('{"api_key":"refused-unknown-field"}')
    panel.get_by_role("button", name="Inspect profiles", exact=True).click()
    expect(panel.get_by_role("alert")).to_contain_text("metadata refused")
    expect(panel.get_by_label("Plan reference")).to_have_text(fx.binding["plan_ref"])
    observations.append("malformed-import-preserves-prior-binding")


def _check_switch_and_export(
    page: Page, panel: Locator, fx: _Fixtures, observations: list[str]
) -> None:
    """Invalidate references on a switch, export exactly, then recover a valid source.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    panel
        The backend profiles panel.
    fx
        Committed sources.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    panel.get_by_label("Backend profile", exact=True).select_option(fx.other["body"]["route_id"])
    for label in ("Plan reference", "Bound calibration reference", "Review reference"):
        expect(panel.get_by_label(label)).to_have_text("none")
    expect(panel.get_by_role("region", name="Selected backend profile")).to_contain_text(
        fx.other["sha256"]
    )
    observations.append("switch-invalidates-all-dependent-references")
    with page.expect_download() as exported:
        panel.get_by_role("button", name="Export admitted profiles").click()
    download = exported.value
    assert download.suggested_filename == "backend-profiles.json"
    saved = download.path()
    assert saved is not None and Path(saved).read_text() == fx.offline
    observations.append("exact-admitted-export")
    # Compact transport retains native values and digests; pretty multiline
    # keyboard insertion timed out in Chromium before admission started.
    panel.get_by_label("Backend profiles JSON").fill(
        json.dumps(json.loads(fx.original), ensure_ascii=False)
    )
    panel.get_by_role("button", name="Inspect profiles", exact=True).click()
    expect(panel.get_by_role("alert")).to_have_count(0)
    expect(panel.get_by_label("Plan reference")).to_have_text("none")
    observations.append("valid-source-recovery")


def _walk(
    page: Page, url: str, observations: list[str], *, fx: _Fixtures, kernels: list[str]
) -> dict[str, object]:
    """Inspect, import, export and recover profiles, recompute WASM, keep the draft.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.
    fx
        Committed sources.
    kernels
        Record of loaded ``.wasm`` paths.

    Returns
    -------
    dict
        Workspace writes and the offline envelope digest.

    """
    panel = _open_profiles(page, url)
    _check_declared(panel, observations)
    _check_import(panel, fx, observations)
    _check_switch_and_export(page, panel, fx, observations)
    check_recompute_and_draft(page, DRAFT, observations)
    assert kernels, "Actual built WASM was not requested"
    return {"workspace_writes": 0, "profile_envelope_sha256": fx.envelope}


def run_backend_profiles_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Exercise production navigation, profile identity and no-submit custody.

    Parameters
    ----------
    base_url
        Owned literal-loopback preview carrying the actual built Studio and WASM.
    evidence
        Caller-owned partial evidence retained on a later runtime refusal.

    Returns
    -------
    dict
        Actual browser/runtime, identity, exact export, refusal, invalidation,
        saved-draft custody and genuine WASM recomputation observations.

    Raises
    ------
    ValueError
        The preview violates the shared loopback ownership contract.
    AssertionError
        A real UI, metadata, export, network or runtime contract fails.

    """
    kernels: list[str] = []
    walk = partial(_walk, fx=_load_fixtures(), kernels=kernels)
    return run_custodied_journey(
        base_url, evidence, "operator_backend_profiles", walk, kernels=kernels, downloads=True
    )
