# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual native policy verdict browser boundary
"""Qualify source verdict parity, custody and original WASM through the built UI."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page


CASES: Final = {
    "unknown_price": "price_unknown",
    "at_ceiling": None,
    "over_shots": "shots_ceiling",
    "over_cost": "cost_ceiling",
    "wrong_region": "region_forbidden",
    "expired_policy": "policy_expired",
    "over_concurrency": "concurrency_ceiling",
    "over_time": "time_limit_ms_ceiling",
}
DRAFT: Final = '{"unsaved":"operator retained α draft"}'
WRITE_PROBE: Final = """() => {
    window.operatorWriteObservations = [];
    for (const name of ['put','add','delete','clear']) {
        const native = IDBObjectStore.prototype[name];
        IDBObjectStore.prototype[name] = function(...args) {
            window.operatorWriteObservations.push(name);
            return native.apply(this,args);
        };
    }
}"""


def _open_policy(page: Page, url: str, observations: list[str]) -> Locator:
    """Keep a draft, probe storage writes and open the policy example by keyboard.

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
    Locator
        The operator policy inspector.

    """
    from playwright.sync_api import expect

    page.goto(url, wait_until="networkidle")
    page.get_by_label("Workspace archive JSON").fill(DRAFT)
    page.evaluate(WRITE_PROBE)
    destination = page.get_by_role("navigation", name="Workbench context destinations")
    destination.get_by_role("link", name="Devices & Operations").focus()
    page.keyboard.press("Enter")
    panel = page.get_by_role("region", name="Operator policy inspector", exact=True)
    expect(panel).to_be_visible()
    expect(panel.get_by_role("button", name="Export admitted policy")).to_be_disabled()
    panel.get_by_role("button", name="Open policy example").click()
    expect(panel.get_by_label("Estimated cost", exact=True)).to_have_text("unknown")
    observations.append("source-unknown-price-visible")
    return panel


def _check_case(panel: Locator, case: str, independent_reason: str | None) -> dict[str, object]:
    """Admit one native policy example and compare it with its independent reason.

    Parameters
    ----------
    panel
        The operator policy inspector.
    case
        Native example name.
    independent_reason
        Refusal reason expected independently of the owner, or None when allowed.

    Returns
    -------
    dict
        The case, its envelope digest and the native decision.

    """
    from playwright.sync_api import expect

    from scpn_quantum_control.studio.workspace import write_json
    from tools.export_operator_policy_decisions import build_operator_policy_example

    envelope = build_operator_policy_example(case)
    body = envelope["body"]
    assert isinstance(body, Mapping)
    decision = body["decision"]
    assert isinstance(decision, Mapping)
    reasons = decision["reasons"]
    assert isinstance(reasons, list)
    assert (decision["allowed"] is True) == (independent_reason is None)
    assert independent_reason is None or independent_reason in reasons
    panel.get_by_label("Operator policy JSON", exact=True).fill(write_json(envelope))
    panel.get_by_role("button", name="Inspect policy decision", exact=True).click()
    expect(panel.get_by_label("Core policy verdict", exact=True)).to_have_text(
        "allowed plan" if decision["allowed"] else "refused plan"
    )
    expect(panel.get_by_label("Policy refusal reasons", exact=True)).to_have_text(
        ", ".join(str(reason) for reason in reasons) if reasons else "none"
    )
    settings = panel.get_by_role("table", name="Operator requested and effective settings")
    expect(settings).to_contain_text("9007199254740993")
    expect(panel.get_by_role("region", name="Admitted operator decision")).to_contain_text(
        str(envelope["sha256"])
    )
    return {"case": case, "sha256": envelope["sha256"], "decision": decision}


def _check_source_and_export(page: Page, panel: Locator, observations: list[str]) -> None:
    """Restore the source verdict, refuse a malformed import and export exactly.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    panel
        The operator policy inspector.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    original = (
        Path(__file__).resolve().parents[1] / "data/studio/operator_policy_decisions.json"
    ).read_text()
    field = panel.get_by_label("Operator policy JSON", exact=True)
    inspect = panel.get_by_role("button", name="Inspect policy decision", exact=True)
    field.fill(original)
    inspect.click()
    expect(panel.get_by_label("Policy refusal reasons")).to_have_text("price_unknown")
    field.fill("{}")
    inspect.click()
    expect(panel.get_by_role("alert")).to_contain_text("metadata refused")
    expect(panel.get_by_label("Policy refusal reasons")).to_have_text("price_unknown")
    observations.append("malformed-import-preserves-source-verdict")
    with page.expect_download() as exported:
        panel.get_by_role("button", name="Export admitted policy").click()
    download = exported.value
    assert download.suggested_filename == "operator-policy-decision.json"
    saved = download.path()
    assert saved is not None and Path(saved).read_text() == original
    observations.append("exact-native-policy-export")


def _check_recompute_and_draft(page: Page, observations: list[str]) -> None:
    """Recompute the built WASM claim in Model and find the draft kept in Workspace.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    observations
        Caller-owned list that records each contract once it holds.

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
    expect(page.get_by_label("Workspace archive JSON")).to_have_value(DRAFT)
    observations.append("original-workspace-draft-preserved")


def _walk(
    page: Page,
    url: str,
    observations: list[str],
    *,
    evidence: dict[str, object],
    records: list[dict[str, object]],
    kernels: list[str],
) -> dict[str, object]:
    """Admit every native verdict, export, then recompute WASM and keep the draft.

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
        Record of each admitted native decision.
    kernels
        Record of loaded ``.wasm`` paths.

    Returns
    -------
    dict
        No further evidence fields.

    """
    panel = _open_policy(page, url, observations)
    records.extend(_check_case(panel, case, reason) for case, reason in CASES.items())
    observations.append("all-eight-native-verdicts-exact-integers-and-provenance")
    _check_source_and_export(page, panel, observations)
    writes = page.evaluate("window.operatorWriteObservations")
    assert writes == [], writes
    evidence["policy_storage_write_observations"] = writes
    observations.append("actual-policy-storage-write-custody")
    _check_recompute_and_draft(page, observations)
    assert kernels, "Actual built WASM was not requested"
    return {}


def run_operator_policy_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Exercise exact source decisions through the real reachable operator view.

    Parameters
    ----------
    base_url
        Owned literal HTTP loopback preview containing genuine current Studio and WASM.
    evidence
        Caller-owned observations retained after an actual runtime refusal.

    Returns
    -------
    dict[str, object]
        Native input/decision identities, exact UI/export parity, actual storage-write
        and network observations, draft custody and original WASM recomputation.

    Raises
    ------
    ValueError
        If the shared preview ownership contract refuses the URL.
    AssertionError
        If any original native verdict, UI, runtime or custody observation disagrees.

    """
    observed: dict[str, object] = {} if evidence is None else evidence
    records: list[dict[str, object]] = []
    kernels: list[str] = []
    observed["native_decisions"] = records
    walk = partial(_walk, evidence=observed, records=records, kernels=kernels)
    return run_custodied_journey(
        base_url, observed, "operator_policy_decisions", walk, kernels=kernels, downloads=True
    )
