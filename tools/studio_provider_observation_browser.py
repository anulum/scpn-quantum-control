# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — existing provider job browser boundary
"""Exercise original recorded job evidence through the built operator UI and WASM."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from tools.studio_preview_custody import run_custodied_journey
from tools.studio_workbench_steps import (
    check_recompute_and_draft,
    open_operations_panel,
    storage_writes,
)

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page


DRAFT: Final = '{"unsaved":"original job inspection draft α"}'
RECORD: Final = "providerJobWriteObservations"
FIXTURE: Final = (
    Path(__file__).resolve().parents[1] / "data/studio/provider_observation_cases.json"
)


def _admit(
    panel: Locator,
    cases: dict[str, str],
    records: list[dict[str, object]],
    case: str,
    refresh: bool = False,
) -> None:
    """Read an original native capture and compare exact source identity and state.

    Parameters
    ----------
    panel
        The provider job journal.
    cases
        Original native captures by case name.
    records
        Record of each admitted observation.
    case
        Capture to admit.
    refresh
        Whether the capture refreshes the retained job instead of importing it.

    """
    from playwright.sync_api import expect

    raw = cases[case]
    unit = json.loads(raw)
    body = unit["body"]
    panel.get_by_label("Provider observation JSON", exact=True).fill(raw)
    action = "Refresh existing job observation" if refresh else "Import existing job observation"
    button = panel.get_by_role("button", name=action, exact=True)
    button.click()
    expect(button).to_be_enabled()
    expect(panel.get_by_label("Provider observation digest")).to_have_text(unit["sha256"])
    expect(panel.get_by_label("Provider attempt identity")).to_have_text(body["attempt_id"])
    expect(panel.get_by_label("Provider payload digest")).to_have_text(body["payload_sha256"])
    expect(panel.get_by_label("Local job state")).to_have_text(body["local_state"])
    records.append({"case": case, "sha256": unit["sha256"], "attempt_id": body["attempt_id"]})


def _refresh_twice(panel: Locator, raw: str, state: str | None, digest: str) -> None:
    """Refresh the same capture twice and require the retained digest and state.

    Parameters
    ----------
    panel
        The provider job journal.
    raw
        Capture text to refresh with.
    state
        Local job state to require, or None to skip that check.
    digest
        Observation digest that must stay unchanged.

    """
    from playwright.sync_api import expect

    for _ in range(2):
        panel.get_by_label("Provider observation JSON", exact=True).fill(raw)
        button = panel.get_by_role("button", name="Refresh existing job observation")
        button.click()
        expect(button).to_be_enabled()
        if state is not None:
            expect(panel.get_by_label("Local job state")).to_have_text(state)
        expect(panel.get_by_label("Provider observation digest")).to_have_text(digest)


def _check_history(
    panel: Locator,
    cases: dict[str, str],
    records: list[dict[str, object]],
    observations: list[str],
) -> str:
    """Follow unknown dispatch, submission, cancellation intent, completion and a late partial.

    Parameters
    ----------
    panel
        The provider job journal.
    cases
        Original native captures by case name.
    records
        Record of each admitted observation.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    str
        Digest of the retained late-partial observation.

    """
    from playwright.sync_api import expect

    digest = panel.get_by_label("Provider observation digest")
    _admit(panel, cases, records, "submission_unknown")
    _refresh_twice(panel, cases["submission_unknown"], "submission_unknown", digest.inner_text())
    expect(panel.get_by_label("Provider job identity")).to_contain_text("unknown")
    observations.append("unknown-dispatch-refresh-never-invents-handle-or-retries")
    _admit(panel, cases, records, "submitted")
    _admit(panel, cases, records, "cancellation_requested", refresh=True)
    expect(panel.get_by_label("Provider job state")).to_have_text("DONE")
    _admit(panel, cases, records, "completed", refresh=True)
    history = panel.get_by_label("Original provider observation history")
    expect(history).to_contain_text("cancellation_requested")
    expect(history).to_contain_text('"counts":{"1":16}')
    observations.append("native-completion-after-cancellation-intent-preserves-history")
    _admit(panel, cases, records, "late_partial", refresh=True)
    expect(history).to_contain_text('"partial_results":[]')
    expect(history).to_contain_text('"counts":{"1":16}')
    original_digest = digest.inner_text()
    observations.append("late-partial-retains-original-native-counts-and-request")
    _refresh_twice(panel, cases["late_partial"], None, original_digest)
    observations.append("two-repeated-result-refreshes-with-zero-submit")
    return original_digest


def _check_refusals_and_export(
    page: Page, panel: Locator, cases: dict[str, str], digest: str, observations: list[str]
) -> None:
    """Refuse a stale refresh and a malformed import, then download the exact capture.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    panel
        The provider job journal.
    cases
        Original native captures by case name.
    digest
        Digest of the retained observation.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    field = panel.get_by_label("Provider observation JSON", exact=True)
    field.fill(cases["completed"])
    panel.get_by_role("button", name="Refresh existing job observation").click()
    expect(panel.get_by_role("alert")).to_contain_text("retained history differs")
    field.fill("{}")
    panel.get_by_role("button", name="Import existing job observation").click()
    expect(panel.get_by_role("alert")).to_contain_text("observation refused")
    expect(panel.get_by_label("Provider observation digest")).to_have_text(digest)
    with page.expect_download() as exported:
        panel.get_by_role("button", name="Export admitted job observation").click()
    download = exported.value
    saved = download.path()
    assert download.suggested_filename == "provider-observation.json"
    assert saved is not None and Path(saved).read_text() == cases["late_partial"]
    observations.append("exact-native-download-after-stale-and-malformed-refusals")


def _walk(
    page: Page,
    url: str,
    observations: list[str],
    *,
    evidence: dict[str, object],
    cases: dict[str, str],
    records: list[dict[str, object]],
    kernels: list[str],
) -> dict[str, object]:
    """Inspect original captures, refuse stale and malformed input, export, keep the draft.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.
    evidence
        Journey evidence; receives the storage write observations.
    cases
        Original native captures by case name.
    records
        Record of each admitted observation.
    kernels
        Record of loaded ``.wasm`` paths.

    Returns
    -------
    dict
        No further evidence fields.

    """
    from playwright.sync_api import expect

    panel = open_operations_panel(page, url, DRAFT, RECORD, "Existing provider job journal")
    export = panel.get_by_role("button", name="Export admitted job observation")
    expect(export).to_be_disabled()
    digest = _check_history(panel, cases, records, observations)
    _check_refusals_and_export(page, panel, cases, digest, observations)
    writes = storage_writes(page, RECORD)
    assert writes == [], writes
    evidence["job_storage_write_observations"] = writes
    check_recompute_and_draft(page, DRAFT, observations)
    assert kernels, "Actual built WASM was not requested"
    return {}


def run_provider_observation_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Qualify original identity, refresh, partial evidence and export through the real UI.

    Parameters
    ----------
    base_url
        Owned literal HTTP loopback preview containing current Studio and genuine WASM.
    evidence
        Caller-owned partial observations retained when the real runtime refuses.

    Returns
    -------
    dict[str, object]
        Actual UI/export identities, network and storage observations, native
        capture provenance, original WASM recomputation and workspace draft custody.
        Captured Aer observations are not new provider or physical-device runs.

    Raises
    ------
    ValueError
        If the original shared loopback ownership contract refuses the address.
    AssertionError
        If identity, append-only refresh, runtime or evidence custody disagrees.

    """
    fixture = FIXTURE.read_bytes()
    corpus = json.loads(fixture)
    observed: dict[str, object] = {} if evidence is None else evidence
    records: list[dict[str, object]] = []
    kernels: list[str] = []
    observed.update(
        native_observations=records,
        recorded_capture_sha256=hashlib.sha256(fixture).hexdigest(),
        recorded_capture_origin=corpus["origin"],
    )
    cases = cast(dict[str, str], corpus["cases"])
    walk = partial(_walk, evidence=observed, cases=cases, records=records, kernels=kernels)
    return run_custodied_journey(
        base_url, observed, "provider_observation_journal", walk, kernels=kernels, downloads=True
    )
