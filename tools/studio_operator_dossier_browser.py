# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual native dossier review browser boundary
"""Exercise immutable native review, exact downloads and genuine original WASM."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from tools.studio_preview_custody import run_custodied_journey
from tools.studio_workbench_steps import (
    check_recompute_and_draft,
    open_operations_panel,
    storage_writes,
)

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page


DRAFT: Final = '{"unsaved":"operator retained α dossier draft"}'
RECORD: Final = "dossierWriteObservations"
PROBE: Final = """() => {
    window.dossierWriteObservations = [];
    for (const name of ['put','add','delete','clear']) {
        const native = IDBObjectStore.prototype[name];
        IDBObjectStore.prototype[name] = function(...args) {
            window.dossierWriteObservations.push('IDB.'+name);
            return native.apply(this,args);
        };
    }
    for (const name of ['setItem','removeItem','clear']) {
        const native = Storage.prototype[name];
        Storage.prototype[name] = function(...args) {
            window.dossierWriteObservations.push('Storage.'+name);
            return native.apply(this,args);
        };
    }
}"""
CHANGES: Final = (
    "changed_payload",
    "changed_target",
    "changed_shots",
    "changed_calibration",
    "changed_price",
    "changed_expiry",
)


def _now(offset: timedelta = timedelta(0)) -> str:
    """Return the current UTC time, shifted by ``offset``, as an ISO second."""
    return (datetime.now(UTC) + offset).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class _Inspector:
    """Admit native review examples into the dossier panel and record each one.

    Attributes
    ----------
    panel
        The operator review dossier panel.
    records
        Record of each admitted dossier.
    as_of
        Default creation time of the examples.

    """

    panel: Locator
    records: list[dict[str, object]]
    as_of: str

    def inspect(self, case: str, source_time: str | None = None) -> tuple[str, dict[str, Any]]:
        """Admit one native example and check its identities and exact integers.

        Parameters
        ----------
        case
            Native example name.
        source_time
            Example creation time; the inspector's default when None.

        Returns
        -------
        tuple
            The admitted text and its source body.

        """
        from playwright.sync_api import expect

        from scpn_quantum_control.studio.workspace import read_json, write_json
        from tools.export_operator_review_dossiers import build_operator_review_example

        created = self.as_of if source_time is None else source_time
        bundle = build_operator_review_example(case, as_of=created)
        source = cast(dict[str, Any], bundle["body"])
        dossier = cast(dict[str, Any], read_json(source["dossier_text"]))
        raw = write_json(bundle)
        panel = self.panel
        panel.get_by_label("Operator dossier JSON", exact=True).fill(raw)
        panel.get_by_role("button", name="Inspect operator dossier", exact=True).click()
        identity = dossier["sha256"]
        execution = dossier["body"]["execution_sha256"]
        expect(panel.get_by_label("Admitted dossier identity", exact=True)).to_have_text(identity)
        expect(panel.get_by_label("Admitted execution identity", exact=True)).to_have_text(
            execution
        )
        settings = panel.get_by_label("Original resolved settings", exact=True)
        expect(settings).to_contain_text("9007199254740993")
        self.records.append(
            {
                "case": case,
                "sha256": identity,
                "execution_sha256": execution,
                "created_at": created,
            }
        )
        return raw, source


def _check_invalidation(inspector: _Inspector, observations: list[str]) -> str:
    """Approve the pending review, then invalidate it on every execution change.

    Parameters
    ----------
    inspector
        Dossier inspector bound to the panel.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    str
        Identity of the originally approved dossier.

    """
    from playwright.sync_api import expect

    from scpn_quantum_control.studio.workspace import read_json

    panel = inspector.panel
    status = panel.get_by_label("Operator review status", exact=True)
    review = panel.get_by_label("Original human review reference", exact=True)
    _, original = inspector.inspect("pending")
    expect(status).to_have_text("pending")
    panel.get_by_role("button", name="Approve human review").click()
    expect(status).to_have_text("approved")
    identity = str(cast(dict[str, Any], read_json(original["dossier_text"]))["sha256"])
    for case in CHANGES:
        inspector.inspect(case)
        expect(status).to_have_text("invalidated")
        expect(review).to_contain_text(identity)
        inspector.inspect("pending")
        expect(status).to_have_text("approved")
    observations.append("all-native-execution-changes-invalidate-original-review")
    inspector.inspect("theme_light")
    expect(status).to_have_text("approved")
    expect(review).to_contain_text(identity)
    observations.append("native-display-only-change-preserves-original-human-source")
    return identity


def _check_exports(
    page: Page, inspector: _Inspector, identity: str, observations: list[str]
) -> None:
    """Download the exact dossier and verifier, then the separate human review.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    inspector
        Dossier inspector bound to the panel.
    identity
        Identity of the originally approved dossier.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from scpn_quantum_control.studio.workspace import read_json

    panel = inspector.panel
    raw, original = inspector.inspect("pending")
    for name, filename, expected in (
        ("Export admitted dossier", "operator-review-dossier.json", original["dossier_text"]),
        ("Export native verifier", "verify_operator_review.py", original["script"]["source"]),
    ):
        with page.expect_download() as exporting:
            panel.get_by_role("button", name=name).click()
        exported = exporting.value
        assert exported.suggested_filename == filename
        saved = exported.path()
        assert saved is not None and Path(saved).read_text() == expected
    with page.expect_download() as exporting_review:
        panel.get_by_role("button", name="Export human review").click()
    saved_review = exporting_review.value.path()
    assert saved_review is not None
    human = cast(dict[str, Any], read_json(Path(saved_review).read_text()))
    assert human["original_export"] == raw
    record = cast(dict[str, Any], read_json(human["review_text"]))
    assert record["body"]["dossier_sha256"] == identity and record["body"]["no_submit"] is True
    observations.append("exact-native-dossier-verifier-and-separate-original-review-downloads")


def _check_refusals(inspector: _Inspector, identity: str, observations: list[str]) -> None:
    """Refuse a malformed draft, deny then approve, and refuse unknown inputs.

    Parameters
    ----------
    inspector
        Dossier inspector bound to the panel.
    identity
        Identity of the originally approved dossier.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    from scpn_quantum_control.studio.workspace import read_json

    panel = inspector.panel
    status = panel.get_by_label("Operator review status", exact=True)
    approve = panel.get_by_role("button", name="Approve human review")
    panel.get_by_label("Operator dossier JSON", exact=True).fill("{}")
    expect(status).to_have_text("draft changed")
    expect(approve).to_be_disabled()
    panel.get_by_role("button", name="Inspect operator dossier", exact=True).click()
    expect(panel.get_by_role("alert")).to_contain_text("metadata refused")
    expect(panel.get_by_label("Admitted dossier identity", exact=True)).to_have_text(identity)
    observations.append("malformed-draft-refusal-retains-immutable-source-and-review")
    inspector.inspect("pending")
    panel.get_by_role("button", name="Deny human review").click()
    expect(status).to_have_text("denied")
    approve.click()
    expect(status).to_have_text("approved")
    inspector.inspect("unknown_price")
    expect(panel.get_by_label("Original price estimate", exact=True)).to_have_text("unknown")
    expect(approve).to_be_disabled()
    inspector.inspect("unknown_calibration")
    expect(panel.get_by_label("Original calibration", exact=True)).to_have_text("unknown")
    _, capacity_source = inspector.inspect("declared_shot_capacity")
    capacity = cast(dict[str, Any], read_json(capacity_source["dossier_text"]))
    assert capacity["body"]["profile"]["capabilities"]["max_shots"] == 2048
    expect(panel.get_by_label("Original backend profile", exact=True)).to_contain_text(
        '"max_shots":2048'
    )
    observations.append("native-declared-shot-capacity-preserved")


def _check_expiry(inspector: _Inspector, observations: list[str]) -> None:
    """Refuse an expired review and watch an approved one expire on the wall clock.

    Parameters
    ----------
    inspector
        Dossier inspector bound to the panel.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    panel = inspector.panel
    status = panel.get_by_label("Operator review status", exact=True)
    approve = panel.get_by_role("button", name="Approve human review")
    inspector.inspect("expired", _now(-timedelta(seconds=2)))
    expect(approve).to_be_disabled()
    inspector.inspect("expires_soon", _now())
    approve.click()
    expect(status).to_have_text("approved")
    expect(status).to_have_text("expired", timeout=15000)
    expect(approve).to_be_disabled()
    observations.append("denial-unknown-inputs-and-actual-wall-clock-expiry-never-approve")


def _walk(
    page: Page,
    url: str,
    observations: list[str],
    *,
    evidence: dict[str, object],
    records: list[dict[str, object]],
    kernels: list[str],
) -> dict[str, object]:
    """Review, invalidate, export, refuse and expire dossiers, then keep the draft.

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
    records
        Record of each admitted dossier.
    kernels
        Record of loaded ``.wasm`` paths.

    Returns
    -------
    dict
        No further evidence fields.

    """
    from playwright.sync_api import expect

    panel = open_operations_panel(page, url, DRAFT, RECORD, "Operator review dossier", PROBE)
    expect(panel.get_by_role("button", name="Export admitted dossier")).to_be_disabled()
    expect(panel.get_by_role("button", name="Export human review")).to_be_disabled()
    inspector = _Inspector(panel, records, _now())
    identity = _check_invalidation(inspector, observations)
    _check_exports(page, inspector, identity, observations)
    _check_refusals(inspector, identity, observations)
    _check_expiry(inspector, observations)
    writes = storage_writes(page, RECORD)
    assert writes == [], writes
    evidence["dossier_storage_write_observations"] = writes
    assert panel.get_by_role("button", name="submit", exact=False).count() == 0
    label = "original-workspace-draft-preserved-with-no-storage-writes"
    check_recompute_and_draft(page, DRAFT, observations, label)
    assert kernels, "Actual built WASM was not requested"
    return {}


def run_operator_dossier_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Qualify native source review through the original reachable production view.

    Parameters
    ----------
    base_url
        Owned literal HTTP loopback host serving actual built Studio and original WASM.
    evidence
        Caller-owned observations retained when an actual runtime boundary refuses.

    Returns
    -------
    dict[str, object]
        Exact native identities and exports, actual expiry, original draft custody,
        and genuine browser storage, network and WASM observations.

    Raises
    ------
    ValueError
        If the shared preview authority refuses the URL before browser creation.
    AssertionError
        If any native identity, observable UI state or runtime custody differs.

    """
    observed: dict[str, object] = {} if evidence is None else evidence
    records: list[dict[str, object]] = []
    kernels: list[str] = []
    observed["native_dossiers"] = records
    walk = partial(_walk, evidence=observed, records=records, kernels=kernels)
    return run_custodied_journey(
        base_url, observed, "operator_review_dossiers", walk, kernels=kernels, downloads=True
    )
