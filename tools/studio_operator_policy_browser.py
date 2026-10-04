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
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit


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
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import Route, expect, sync_playwright

    from scpn_quantum_control.studio.workspace import write_json
    from tools.export_operator_policy_decisions import build_operator_policy_example

    original = (
        Path(__file__).resolve().parents[1] / "data/studio/operator_policy_decisions.json"
    ).read_text()
    cases = {
        "unknown_price": "price_unknown",
        "at_ceiling": None,
        "over_shots": "shots_ceiling",
        "over_cost": "cost_ceiling",
        "wrong_region": "region_forbidden",
        "expired_policy": "policy_expired",
        "over_concurrency": "concurrency_ceiling",
        "over_time": "time_limit_ms_ceiling",
    }
    observed = {} if evidence is None else evidence
    observations: list[str] = []
    errors: list[str] = []
    external: list[str] = []
    submissions: list[str] = []
    kernels: list[str] = []
    records: list[dict[str, object]] = []
    observed.update(
        scenario="operator_policy_decisions",
        playwright=version("playwright"),
        observations=observations,
        page_errors=errors,
        external_requests=external,
        submission_requests=submissions,
        wasm_requests=kernels,
        native_decisions=records,
    )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        context = browser.new_context(service_workers="block", accept_downloads=True)
        context.set_default_timeout(15000)
        try:
            origin = urlsplit(url).netloc

            def bound(route: Route) -> None:
                target = urlsplit(route.request.url)
                if target.scheme != "http" or target.netloc != origin:
                    external.append("outside-preview")
                    route.abort()
                elif route.request.method != "GET":
                    submissions.append(route.request.method)
                    route.abort()
                else:
                    if target.path.endswith(".wasm"):
                        kernels.append(target.path)
                    route.continue_()

            context.route("**/*", bound)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url, wait_until="networkidle")
            workspace = page.get_by_label("Workspace archive JSON")
            draft = '{"unsaved":"operator retained α draft"}'
            workspace.fill(draft)
            page.evaluate("""() => {
                window.operatorWriteObservations = [];
                for (const name of ['put','add','delete','clear']) {
                    const native = IDBObjectStore.prototype[name];
                    IDBObjectStore.prototype[name] = function(...args) {
                        window.operatorWriteObservations.push(name);
                        return native.apply(this,args);
                    };
                }
            }""")
            destination = page.get_by_role(
                "navigation", name="Workbench context destinations"
            ).get_by_role("link", name="Devices & Operations")
            destination.focus()
            page.keyboard.press("Enter")
            panel = page.get_by_role("region", name="Operator policy inspector", exact=True)
            expect(panel).to_be_visible()
            expect(panel.get_by_role("button", name="Export admitted policy")).to_be_disabled()
            panel.get_by_role("button", name="Open policy example").click()
            expect(panel.get_by_label("Estimated cost", exact=True)).to_have_text("unknown")
            observations.append("source-unknown-price-visible")
            field = panel.get_by_label("Operator policy JSON", exact=True)
            for case, independent_reason in cases.items():
                envelope = build_operator_policy_example(case)
                body = envelope["body"]
                assert isinstance(body, Mapping)
                decision = body["decision"]
                assert isinstance(decision, Mapping)
                reasons = decision["reasons"]
                assert isinstance(reasons, list)
                assert (decision["allowed"] is True) == (independent_reason is None)
                assert independent_reason is None or independent_reason in reasons
                raw = write_json(envelope)
                field.fill(raw)
                panel.get_by_role("button", name="Inspect policy decision", exact=True).click()
                expect(panel.get_by_label("Core policy verdict", exact=True)).to_have_text(
                    "allowed plan" if decision["allowed"] else "refused plan"
                )
                expect(panel.get_by_label("Policy refusal reasons", exact=True)).to_have_text(
                    ", ".join(str(reason) for reason in reasons) if reasons else "none"
                )
                expect(
                    panel.get_by_role("table", name="Operator requested and effective settings")
                ).to_contain_text("9007199254740993")
                expect(
                    panel.get_by_role("region", name="Admitted operator decision")
                ).to_contain_text(str(envelope["sha256"]))
                records.append({"case": case, "sha256": envelope["sha256"], "decision": decision})
            observations.append("all-eight-native-verdicts-exact-integers-and-provenance")
            field.fill(original)
            panel.get_by_role("button", name="Inspect policy decision", exact=True).click()
            expect(panel.get_by_label("Policy refusal reasons")).to_have_text("price_unknown")
            field.fill("{}")
            panel.get_by_role("button", name="Inspect policy decision", exact=True).click()
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
            writes = page.evaluate("window.operatorWriteObservations")
            assert writes == [], writes
            observed["policy_storage_write_observations"] = writes
            observations.append("actual-policy-storage-write-custody")
            page.get_by_role("navigation", name="Workbench views").get_by_role(
                "link", name="Build", exact=True
            ).click()
            recompute = page.locator('[id="/build/compile-recompute"]')
            recompute.get_by_role("button", name="Recompute in browser").click()
            expect(recompute.get_by_role("status")).to_have_text(
                "recomputed digest matches the signed claim"
            )
            observations.append("original-built-wasm-recompute")
            page.get_by_role("navigation", name="Workbench views").get_by_role(
                "link", name="Workspace", exact=True
            ).click()
            expect(workspace).to_have_value(draft)
            observations.append("original-workspace-draft-preserved")
            assert kernels, "Actual built WASM was not requested"
            assert not errors, errors
            assert not external, external
            assert not submissions, submissions
            assert not page.workers, "Policy inspection must not retain workers"
            observed.update(browser=browser.version, workers=len(page.workers), no_submit=True)
            return observed
        finally:
            context.close()
            browser.close()
