# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — calibration snapshot drift through the built operations view
"""Qualify the calibration inspector against an independent source build through the built UI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Final

from tools.studio_preview_custody import run_custodied_journey
from tools.studio_workbench_steps import (
    check_recompute_and_draft,
    open_operations_panel,
    storage_writes,
)

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

DRAFT: Final = '{"unsaved":"calibration inspector retained α draft"}'
RECORD: Final = "calibrationWriteObservations"


def _items(record: Mapping[str, object], key: str) -> Sequence[Mapping[str, object]]:
    """Return one list of mappings from the source record.

    Parameters
    ----------
    record
        The independently built source record.
    key
        Top-level list name.

    Returns
    -------
    Sequence[Mapping[str, object]]
        The entries.

    """
    items = record[key]
    assert isinstance(items, list) and all(isinstance(item, Mapping) for item in items)
    return items


def _check_history(panel: Locator, record: Mapping[str, object]) -> dict[str, object]:
    """Compare the newest and the first snapshot and the first drift with the source.

    Parameters
    ----------
    panel
        The calibration inspector.
    record
        The independently built source record.

    Returns
    -------
    dict[str, object]
        Calibration sets and the drift size the UI showed.

    """
    from playwright.sync_api import expect

    snapshots, drifts = _items(record, "snapshots"), _items(record, "drifts")
    expect(panel.get_by_label("Snapshot standing", exact=True)).to_have_text(
        "Current calibration of this target."
    )
    selector = panel.get_by_label("Calibration snapshot", exact=True)
    selector.select_option("0")
    expect(panel.get_by_label("Snapshot standing", exact=True)).to_have_text(
        f"Superseded by {snapshots[0]['superseded_by']}."
    )
    selector.select_option("1")
    changes = drifts[0]["changes"]
    assert isinstance(changes, list)
    table = panel.get_by_role("table", name="Changes from the previous snapshot")
    expect(table.get_by_role("row")).to_have_count(len(changes) + 1)
    readings = panel.get_by_role("table", name="Element readings")
    expect(readings.get_by_role("row")).to_have_count(
        sum(
            len(entry)
            for entry in (snapshots[1]["qubits"], snapshots[1]["couplers"])
            if isinstance(entry, list)
        )
        + 1
    )
    return {
        "first_set": snapshots[0]["calibration_set_id"],
        "current_set": snapshots[-1]["calibration_set_id"],
        "first_drift_changes": len(changes),
    }


def _check_plans_and_reference(panel: Locator, record: Mapping[str, object]) -> None:
    """Compare every plan verdict and the superseded and unknown reference verdicts.

    Parameters
    ----------
    panel
        The calibration inspector.
    record
        The independently built source record.

    """
    from playwright.sync_api import expect

    plans = _items(record, "plans")
    rows = panel.get_by_role("table", name="Plans and their calibration").locator("tbody tr")
    expect(rows).to_have_count(len(plans))
    for index, plan in enumerate(plans):
        expect(rows.nth(index)).to_have_attribute("data-status", str(plan["status"]))
    first = _items(record, "snapshots")[0]
    field = panel.get_by_label("Calibration file sha256", exact=True)
    verdict = panel.get_by_label("Calibration reference verdict", exact=True)
    field.fill(str(first["sha256"]))
    expect(verdict).to_have_text(
        f"Superseded by {first['superseded_by']}: dossiers bound to "
        f"{first['calibration_set_id']} are blocked, and no other snapshot is substituted."
    )
    field.fill("0" * 64)
    expect(verdict).to_have_text("Not an admitted snapshot: dossiers bound to it are blocked.")


def _walk(
    page: Page,
    url: str,
    observations: list[str],
    *,
    evidence: dict[str, object],
    kernels: list[str],
) -> dict[str, object]:
    """Open the inspector, compare it with the source record, then recompute WASM and keep the draft.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.
    evidence
        Journey evidence; receives the record digest and storage observations.
    kernels
        Record of loaded ``.wasm`` paths.

    Returns
    -------
    dict
        No further evidence fields.

    """
    from playwright.sync_api import expect

    from tools.studio_calibration_inspection import build_committed_calibration_inspection

    record = build_committed_calibration_inspection()
    panel = open_operations_panel(page, url, DRAFT, RECORD, "Calibration inspection")
    panel.get_by_role("button", name="Open calibration inspection").click()
    expect(
        panel.get_by_text(f"Record {str(record['record_digest'])[:12]} verified")
    ).to_be_visible()
    evidence["record_digest"] = record["record_digest"]
    observations.append("source-record-admitted-by-digest")
    evidence["history"] = _check_history(panel, record)
    observations.append("history-and-exact-drift-visible")
    _check_plans_and_reference(panel, record)
    observations.append("plan-verdicts-and-superseded-reference-blocked")
    expect(panel.get_by_role("button", name="Submit")).to_have_count(0)
    writes = storage_writes(page, RECORD)
    assert writes == [], writes
    evidence["calibration_storage_write_observations"] = writes
    check_recompute_and_draft(page, DRAFT, observations)
    assert kernels, "Actual built WASM was not requested"
    return {}


def run_calibration_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Exercise the calibration inspector through the real reachable operations view.

    Parameters
    ----------
    base_url
        Owned literal HTTP loopback preview containing genuine current Studio and WASM.
    evidence
        Caller-owned observations retained after an actual runtime refusal.

    Returns
    -------
    dict[str, object]
        The admitted record digest, the history and plan parity with an independent
        source build, storage-write and network observations, draft custody and
        original WASM recomputation.

    Raises
    ------
    ValueError
        If the shared preview ownership contract refuses the URL.
    AssertionError
        If any UI, runtime or custody observation disagrees with the source.

    """
    observed: dict[str, object] = {} if evidence is None else evidence
    kernels: list[str] = []
    walk = partial(_walk, evidence=observed, kernels=kernels)
    return run_custodied_journey(
        base_url, observed, "calibration_snapshot_diff", walk, kernels=kernels
    )
