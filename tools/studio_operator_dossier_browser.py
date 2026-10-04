# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual native dossier review browser boundary
"""Exercise immutable native review, exact downloads and genuine original WASM."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit


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
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import Route, expect, sync_playwright

    from scpn_quantum_control.studio.workspace import read_json, write_json
    from tools.export_operator_review_dossiers import build_operator_review_example

    observed = {} if evidence is None else evidence
    observations: list[str] = []
    errors: list[str] = []
    external: list[str] = []
    submissions: list[str] = []
    kernels: list[str] = []
    records: list[dict[str, object]] = []
    observed.update(
        scenario="operator_review_dossiers",
        playwright=version("playwright"),
        observations=observations,
        page_errors=errors,
        external_requests=external,
        submission_requests=submissions,
        wasm_requests=kernels,
        native_dossiers=records,
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
            draft = '{"unsaved":"operator retained α dossier draft"}'
            workspace.fill(draft)
            page.evaluate("""() => {
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
            }""")
            destination = page.get_by_role(
                "navigation", name="Workbench context destinations"
            ).get_by_role("link", name="Devices & Operations")
            destination.focus()
            page.keyboard.press("Enter")
            panel = page.get_by_role("region", name="Operator review dossier", exact=True)
            expect(panel).to_be_visible()
            expect(panel.get_by_role("button", name="Export admitted dossier")).to_be_disabled()
            expect(panel.get_by_role("button", name="Export human review")).to_be_disabled()
            field = panel.get_by_label("Operator dossier JSON", exact=True)
            status = panel.get_by_label("Operator review status", exact=True)
            as_of = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

            def inspect(case: str, source_time: str = as_of) -> tuple[str, dict[str, Any]]:
                bundle = build_operator_review_example(case, as_of=source_time)
                source = cast(dict[str, Any], bundle["body"])
                dossier = cast(dict[str, Any], read_json(source["dossier_text"]))
                raw = write_json(bundle)
                field.fill(raw)
                panel.get_by_role("button", name="Inspect operator dossier", exact=True).click()
                expect(panel.get_by_label("Admitted dossier identity", exact=True)).to_have_text(
                    dossier["sha256"]
                )
                expect(panel.get_by_label("Admitted execution identity", exact=True)).to_have_text(
                    dossier["body"]["execution_sha256"]
                )
                expect(
                    panel.get_by_label("Original resolved settings", exact=True)
                ).to_contain_text("9007199254740993")
                records.append(
                    {
                        "case": case,
                        "sha256": dossier["sha256"],
                        "execution_sha256": dossier["body"]["execution_sha256"],
                        "created_at": source_time,
                    }
                )
                return raw, source

            raw, original = inspect("pending")
            expect(status).to_have_text("pending")
            panel.get_by_role("button", name="Approve human review").click()
            expect(status).to_have_text("approved")
            original_identity = cast(dict[str, Any], read_json(original["dossier_text"]))["sha256"]
            for case in (
                "changed_payload",
                "changed_target",
                "changed_shots",
                "changed_calibration",
                "changed_price",
                "changed_expiry",
            ):
                inspect(case)
                expect(status).to_have_text("invalidated")
                expect(
                    panel.get_by_label("Original human review reference", exact=True)
                ).to_contain_text(original_identity)
                inspect("pending")
                expect(status).to_have_text("approved")
            observations.append("all-native-execution-changes-invalidate-original-review")
            inspect("theme_light")
            expect(status).to_have_text("approved")
            expect(
                panel.get_by_label("Original human review reference", exact=True)
            ).to_contain_text(original_identity)
            observations.append("native-display-only-change-preserves-original-human-source")
            raw, original = inspect("pending")
            for name, filename, expected in (
                (
                    "Export admitted dossier",
                    "operator-review-dossier.json",
                    original["dossier_text"],
                ),
                (
                    "Export native verifier",
                    "verify_operator_review.py",
                    original["script"]["source"],
                ),
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
            assert (
                record["body"]["dossier_sha256"] == original_identity
                and record["body"]["no_submit"] is True
            )
            observations.append(
                "exact-native-dossier-verifier-and-separate-original-review-downloads"
            )
            field.fill("{}")
            expect(status).to_have_text("draft changed")
            expect(panel.get_by_role("button", name="Approve human review")).to_be_disabled()
            panel.get_by_role("button", name="Inspect operator dossier", exact=True).click()
            expect(panel.get_by_role("alert")).to_contain_text("metadata refused")
            expect(panel.get_by_label("Admitted dossier identity", exact=True)).to_have_text(
                original_identity
            )
            observations.append("malformed-draft-refusal-retains-immutable-source-and-review")
            inspect("pending")
            panel.get_by_role("button", name="Deny human review").click()
            expect(status).to_have_text("denied")
            panel.get_by_role("button", name="Approve human review").click()
            expect(status).to_have_text("approved")
            inspect("unknown_price")
            expect(panel.get_by_label("Original price estimate", exact=True)).to_have_text(
                "unknown"
            )
            expect(panel.get_by_role("button", name="Approve human review")).to_be_disabled()
            inspect("unknown_calibration")
            expect(panel.get_by_label("Original calibration", exact=True)).to_have_text("unknown")
            _, capacity_source = inspect("declared_shot_capacity")
            capacity_dossier = cast(dict[str, Any], read_json(capacity_source["dossier_text"]))
            assert capacity_dossier["body"]["profile"]["capabilities"]["max_shots"] == 2048
            expect(panel.get_by_label("Original backend profile", exact=True)).to_contain_text(
                '"max_shots":2048'
            )
            observations.append("native-declared-shot-capacity-preserved")
            past = (datetime.now(UTC) - timedelta(seconds=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
            inspect("expired", past)
            expect(panel.get_by_role("button", name="Approve human review")).to_be_disabled()
            fresh = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            inspect("expires_soon", fresh)
            panel.get_by_role("button", name="Approve human review").click()
            expect(status).to_have_text("approved")
            expect(status).to_have_text("expired", timeout=15000)
            expect(panel.get_by_role("button", name="Approve human review")).to_be_disabled()
            observations.append("denial-unknown-inputs-and-actual-wall-clock-expiry-never-approve")
            writes = page.evaluate("window.dossierWriteObservations")
            assert writes == [], writes
            observed["dossier_storage_write_observations"] = writes
            assert panel.get_by_role("button", name="submit", exact=False).count() == 0
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
            observations.append("original-workspace-draft-preserved-with-no-storage-writes")
            assert kernels, "Actual built WASM was not requested"
            assert not errors, errors
            assert not external, external
            assert not submissions, submissions
            assert not page.workers, "Review must not retain workers"
            observed.update(browser=browser.version, workers=len(page.workers), no_submit=True)
            return observed
        finally:
            context.close()
            browser.close()
