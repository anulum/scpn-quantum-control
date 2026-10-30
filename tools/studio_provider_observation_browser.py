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
from importlib.metadata import version
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit


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
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import Route, expect, sync_playwright

    fixture = (
        Path(__file__).resolve().parents[1] / "data/studio/provider_observation_cases.json"
    ).read_bytes()
    corpus = json.loads(fixture)
    cases = cast(dict[str, str], corpus["cases"])
    observed = {} if evidence is None else evidence
    observations: list[str] = []
    errors: list[str] = []
    external: list[str] = []
    submissions: list[str] = []
    kernels: list[str] = []
    records: list[dict[str, object]] = []
    observed.update(
        scenario="provider_observation_journal",
        playwright=version("playwright"),
        observations=observations,
        page_errors=errors,
        external_requests=external,
        submission_requests=submissions,
        wasm_requests=kernels,
        native_observations=records,
        recorded_capture_sha256=hashlib.sha256(fixture).hexdigest(),
        recorded_capture_origin=corpus["origin"],
    )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        context = browser.new_context(service_workers="block", accept_downloads=True)
        context.set_default_timeout(15000)
        try:
            origin = urlsplit(url).netloc

            def bound(route: Route) -> None:
                """Retain actual attempted requests while permitting only owned preview reads."""
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
            draft = '{"unsaved":"original job inspection draft α"}'
            workspace.fill(draft)
            page.evaluate("""() => {
                window.providerJobWriteObservations = [];
                for (const name of ['put','add','delete','clear']) {
                    const native = IDBObjectStore.prototype[name];
                    IDBObjectStore.prototype[name] = function(...args) {
                        window.providerJobWriteObservations.push(name);
                        return native.apply(this,args);
                    };
                }
            }""")
            destination = page.get_by_role(
                "navigation", name="Workbench context destinations"
            ).get_by_role("link", name="Devices & Operations")
            destination.focus()
            page.keyboard.press("Enter")
            panel = page.get_by_role("region", name="Existing provider job journal", exact=True)
            expect(panel).to_be_visible()
            expect(
                panel.get_by_role("button", name="Export admitted job observation")
            ).to_be_disabled()
            field = panel.get_by_label("Provider observation JSON", exact=True)

            def admit(case: str, refresh: bool = False) -> None:
                """Read an original native capture and compare exact source identity and state."""
                raw = cases[case]
                unit = json.loads(raw)
                body = unit["body"]
                field.fill(raw)
                button = panel.get_by_role(
                    "button",
                    name="Refresh existing job observation"
                    if refresh
                    else "Import existing job observation",
                    exact=True,
                )
                button.click()
                expect(button).to_be_enabled()
                expect(panel.get_by_label("Provider observation digest")).to_have_text(
                    unit["sha256"]
                )
                expect(panel.get_by_label("Provider attempt identity")).to_have_text(
                    body["attempt_id"]
                )
                expect(panel.get_by_label("Provider payload digest")).to_have_text(
                    body["payload_sha256"]
                )
                expect(panel.get_by_label("Local job state")).to_have_text(body["local_state"])
                records.append(
                    {"case": case, "sha256": unit["sha256"], "attempt_id": body["attempt_id"]}
                )

            admit("submission_unknown")
            unknown_digest = panel.get_by_label("Provider observation digest").inner_text()
            for _ in range(2):
                field.fill(cases["submission_unknown"])
                button = panel.get_by_role("button", name="Refresh existing job observation")
                button.click()
                expect(button).to_be_enabled()
                expect(panel.get_by_label("Local job state")).to_have_text("submission_unknown")
                expect(panel.get_by_label("Provider observation digest")).to_have_text(
                    unknown_digest
                )
            expect(panel.get_by_label("Provider job identity")).to_contain_text("unknown")
            observations.append("unknown-dispatch-refresh-never-invents-handle-or-retries")
            admit("submitted")
            admit("cancellation_requested", refresh=True)
            expect(panel.get_by_label("Provider job state")).to_have_text("DONE")
            admit("completed", refresh=True)
            history = panel.get_by_label("Original provider observation history")
            expect(history).to_contain_text("cancellation_requested")
            expect(history).to_contain_text('"counts":{"1":16}')
            observations.append("native-completion-after-cancellation-intent-preserves-history")
            admit("late_partial", refresh=True)
            expect(history).to_contain_text('"partial_results":[]')
            expect(history).to_contain_text('"counts":{"1":16}')
            original_digest = panel.get_by_label("Provider observation digest").inner_text()
            observations.append("late-partial-retains-original-native-counts-and-request")
            for _ in range(2):
                field.fill(cases["late_partial"])
                button = panel.get_by_role("button", name="Refresh existing job observation")
                button.click()
                expect(button).to_be_enabled()
                expect(panel.get_by_label("Provider observation digest")).to_have_text(
                    original_digest
                )
            observations.append("two-repeated-result-refreshes-with-zero-submit")
            field.fill(cases["completed"])
            panel.get_by_role("button", name="Refresh existing job observation").click()
            expect(panel.get_by_role("alert")).to_contain_text("retained history differs")
            field.fill("{}")
            panel.get_by_role("button", name="Import existing job observation").click()
            expect(panel.get_by_role("alert")).to_contain_text("observation refused")
            expect(panel.get_by_label("Provider observation digest")).to_have_text(original_digest)
            with page.expect_download() as exported:
                panel.get_by_role("button", name="Export admitted job observation").click()
            download = exported.value
            saved = download.path()
            assert download.suggested_filename == "provider-observation.json"
            assert saved is not None and Path(saved).read_text() == cases["late_partial"]
            observations.append("exact-native-download-after-stale-and-malformed-refusals")
            writes = page.evaluate("window.providerJobWriteObservations")
            assert writes == [], writes
            observed["job_storage_write_observations"] = writes
            page.get_by_role("navigation", name="Workbench views").get_by_role(
                "link", name="Model", exact=True
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
            assert not page.workers, "Job observation must not retain workers"
            observed.update(browser=browser.version, workers=len(page.workers), no_submit=True)
            return observed
        finally:
            context.close()
            browser.close()
