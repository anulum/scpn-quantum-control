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
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit


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
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import Route, expect, sync_playwright

    repo = Path(__file__).resolve().parents[1]
    original = (repo / "data/studio/backend_profiles.json").read_text()
    cases = json.loads((repo / "data/studio/backend_profiles_cases.json").read_text())
    offline = json.dumps(cases["offline"], ensure_ascii=False)
    binding = cases["offline"]["body"]["binding"]
    rows = cases["offline"]["body"]["profiles"]
    current = next(row for row in rows if row["sha256"] == binding["profile_sha256"])
    other = next(row for row in rows if row["sha256"] != binding["profile_sha256"])
    observed = {} if evidence is None else evidence
    observations: list[str] = []
    errors: list[str] = []
    external: list[str] = []
    submissions: list[str] = []
    kernels: list[str] = []
    observed.update(
        scenario="operator_backend_profiles",
        playwright=version("playwright"),
        observations=observations,
        page_errors=errors,
        external_requests=external,
        submission_requests=submissions,
        wasm_requests=kernels,
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
            workspace.fill('{"unsaved":"exact retained α draft"}')
            destination = page.get_by_role(
                "navigation", name="Workbench context destinations"
            ).get_by_role("link", name="Devices & Operations")
            destination.focus()
            page.keyboard.press("Enter")
            panel = page.get_by_role("region", name="Backend profiles", exact=True)
            expect(panel).to_be_visible()
            panel.get_by_role("button", name="Open declared profiles").click()
            select = panel.get_by_label("Backend profile", exact=True)
            expect(select).to_have_value("")
            select.select_option("direct/iqm")
            expect(panel.get_by_label("Online observation")).to_have_text("unknown")
            expect(panel.get_by_label("Observed shot ceiling")).to_have_text("unknown")
            expect(panel.get_by_role("button", name="Pulse options")).to_be_disabled()
            expect(panel.get_by_role("button", name="Analog options")).to_be_disabled()
            expect(panel.get_by_text("Snapshot date:", exact=False)).to_contain_text(
                "Snapshot age:"
            )
            observations.append("offline-declared-age-and-unknown")
            field = panel.get_by_label("Backend profiles JSON")
            field.fill(offline)
            panel.get_by_role("button", name="Inspect profiles", exact=True).click()
            expect(panel.get_by_label("Plan reference")).to_have_text(binding["plan_ref"])
            expect(panel.get_by_label("Bound calibration reference")).to_have_text(
                binding["calibration_ref"]
            )
            expect(panel.get_by_label("Review reference")).to_have_text(binding["approval_ref"])
            expect(select).to_have_value(current["body"]["route_id"])
            expect(panel.get_by_label("Observed shot ceiling")).to_have_text(
                "18446744073709551615"
            )
            expect(panel.get_by_label("Online observation")).to_have_text(
                "offline (supplied observation)"
            )
            assert current["body"]["device"] == other["body"]["device"]
            assert current["sha256"] != other["sha256"]
            observations.append("two-brokers-same-device-exact-integer")
            field.fill('{"api_key":"refused-unknown-field"}')
            panel.get_by_role("button", name="Inspect profiles", exact=True).click()
            expect(panel.get_by_role("alert")).to_contain_text("metadata refused")
            expect(panel.get_by_label("Plan reference")).to_have_text(binding["plan_ref"])
            observations.append("malformed-import-preserves-prior-binding")
            select.select_option(other["body"]["route_id"])
            for label in ("Plan reference", "Bound calibration reference", "Review reference"):
                expect(panel.get_by_label(label)).to_have_text("none")
            expect(panel.get_by_role("region", name="Selected backend profile")).to_contain_text(
                other["sha256"]
            )
            observations.append("switch-invalidates-all-dependent-references")
            with page.expect_download() as exported:
                panel.get_by_role("button", name="Export admitted profiles").click()
            download = exported.value
            assert download.suggested_filename == "backend-profiles.json"
            saved = download.path()
            assert saved is not None and Path(saved).read_text() == offline
            observations.append("exact-admitted-export")
            # Compact transport retains native values and digests; pretty multiline
            # keyboard insertion timed out in Chromium before admission started.
            field.fill(json.dumps(json.loads(original), ensure_ascii=False))
            panel.get_by_role("button", name="Inspect profiles", exact=True).click()
            expect(panel.get_by_role("alert")).to_have_count(0)
            expect(panel.get_by_label("Plan reference")).to_have_text("none")
            observations.append("valid-source-recovery")
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
            expect(workspace).to_have_value('{"unsaved":"exact retained α draft"}')
            observations.append("original-workspace-draft-preserved")
            assert kernels, "Actual built WASM was not requested"
            assert not errors, errors
            assert not external, external
            assert not submissions, submissions
            assert not page.workers, "Profile inspection must not retain workers"
            observed.update(
                browser=browser.version,
                workers=len(page.workers),
                workspace_writes=0,
                profile_envelope_sha256=cases["offline"]["sha256"],
                no_submit=True,
            )
            return observed
        finally:
            context.close()
            browser.close()
