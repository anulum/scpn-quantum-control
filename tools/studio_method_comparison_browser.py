# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Original native method-comparison browser journey
"""Exercise complete native populations through the shared built/source Studio runner."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit


def run_method_comparison_journey(
    base_url: str,
    *,
    source_url: str | None = None,
    evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    """Qualify original full-population import, refusal, selection and exact export.

    Parameters
    ----------
    base_url : str
        Owned built Studio preview with its genuine original Rust WASM.
    source_url : str or None
        Distinct owned root Vite source origin for actual debugger counters.
    evidence : dict[str, object] or None
        Caller-owned partial diagnostics retained on an actual runtime failure.

    Returns
    -------
    dict[str, object]
        Actual browser versions, native source identities, all original controls,
        exact download bytes, storage/network custody and original WASM outcome.
        Native counters retain real zero ranges and infer no coverage percentage.

    Raises
    ------
    ValueError
        Preview/source ownership, root path or distinct-origin contract fails.
    AssertionError
        Original values, retention, exact export, workspace or runtime custody fails.

    """
    from tools.studio_browser_journey import loopback_url

    preview = loopback_url(base_url)
    source = None if source_url is None else loopback_url(source_url)
    if source is not None:
        a, b = urlsplit(preview), urlsplit(source)
        if (
            ipaddress.ip_address(cast(str, a.hostname))
            == ipaddress.ip_address(cast(str, b.hostname))
            and a.port == b.port
        ) or b.path != "/":
            raise ValueError("Method comparison requires a distinct owned root source server")
    from playwright.sync_api import Route, expect, sync_playwright

    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    fixture = (
        Path(__file__).resolve().parents[1] / "data/studio/method_comparison_cases.json"
    ).read_bytes()
    pack = json.loads(fixture)
    cases = cast(dict[str, str], pack["cases"])
    observed = {} if evidence is None else evidence
    records: list[dict[str, object]] = []
    native: list[dict[str, object]] = []
    errors: list[str] = []
    external: list[str] = []
    submissions: list[str] = []
    kernels: list[str] = []
    observations: list[str] = []
    transports: list[dict[str, object]] = []
    observed.update(
        scenario="method_comparison_instrument",
        playwright=version("playwright"),
        records=records,
        page_errors=errors,
        external_requests=external,
        submission_requests=submissions,
        wasm_requests=kernels,
        observations=observations,
        file_transport_checks=transports,
        source_capture_sha256=hashlib.sha256(fixture).hexdigest(),
        interval_protocol=pack["interval_protocol"],
    )
    if source is not None:
        observed.update(
            source_url=source, native_v8_coverage=native, coverage_percentage="not_calculated"
        )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        observed["browser"] = browser.version
        try:
            targets = [("built", preview)] + ([] if source is None else [("source", source)])
            for kind, target in targets:
                context = browser.new_context(service_workers="block", accept_downloads=True)
                context.set_default_timeout(15000)
                try:
                    origin = urlsplit(target).netloc

                    def bound(route: Route, *, authority: str = origin) -> None:
                        """Retain actual attempted requests; permit only reads on the owned origin."""
                        destination = urlsplit(route.request.url)
                        if destination.scheme != "http" or destination.netloc != authority:
                            external.append("outside-preview")
                            route.abort()
                        elif route.request.method != "GET":
                            submissions.append(route.request.method)
                            route.abort()
                        else:
                            if destination.path.endswith(".wasm"):
                                kernels.append(destination.path)
                            route.continue_()

                    context.route("**/*", bound)
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    profiler = start_native_coverage(page) if kind == "source" else None
                    try:
                        page.goto(target, wait_until="networkidle")
                        workspace = page.get_by_label("Workspace archive JSON")
                        draft = '{"unsaved":"native comparison draft α"}'
                        workspace.fill(draft)
                        page.evaluate("""() => {
                            window.methodComparisonWrites=[];
                            for(const name of ['put','add','delete','clear']) {
                                const original=IDBObjectStore.prototype[name];
                                IDBObjectStore.prototype[name]=function(...args) {
                                    window.methodComparisonWrites.push(name);return original.apply(this,args);
                                };
                            }
                        }""")
                        navigation = page.get_by_role("navigation", name="Workbench views")
                        navigation.get_by_role("link", name="Compare", exact=True).click()
                        panel = page.get_by_role(
                            "region", name="Method comparison and uncertainty", exact=True
                        )
                        expect(panel).to_be_visible()
                        field = panel.get_by_label("Native method comparison JSON", exact=True)
                        read = panel.get_by_role(
                            "button", name="Read native method comparison", exact=True
                        )
                        file_input = panel.get_by_label("Native comparison file", exact=True)
                        for name, raw in cases.items():
                            original = json.loads(raw)
                            if name == "integer-scalar":
                                file_input.set_input_files(
                                    {
                                        "name": "original-integer.json",
                                        "mimeType": "application/json",
                                        "buffer": raw.encode("utf-8"),
                                    }
                                )
                            else:
                                field.fill(raw)
                                read.click()
                            expect(read).to_be_enabled()
                            expect(panel.get_by_label("Method comparison digest")).to_have_text(
                                original["sha256"]
                            )
                            summary = panel.get_by_role("table", name="All-run method summaries")
                            before = summary.inner_text()
                            selection = panel.get_by_label(
                                "Inspect retained repetition", exact=True
                            )
                            selection.select_option(
                                original["body"]["comparison"]["runs"][-1]["run_id"]
                            )
                            expect(
                                panel.get_by_role("region", name="Selected original repetition")
                            ).to_be_visible()
                            assert summary.inner_text() == before
                            if name == "population":
                                expect(summary).to_contain_text("30.0")
                                expect(
                                    summary.get_by_role("row", name=re.compile(r"^reference\s"))
                                ).to_contain_text("1.0")
                                expect(
                                    summary.get_by_role("row", name=re.compile(r"^candidate\s"))
                                ).to_contain_text("2.0")
                                expect(
                                    panel.get_by_role("table", name="All retained repetitions")
                                ).to_contain_text("-1.0")
                            if name in {
                                "model-mismatch",
                                "precision-mismatch",
                                "missing-population",
                                "unknown-cost",
                                "budget-exceeded",
                                "original-solver",
                                "original-ode-only",
                            }:
                                expect(
                                    panel.get_by_role("alert", name="Native comparison refusal")
                                ).to_be_visible()
                            if name == "intervals":
                                expect(
                                    panel.get_by_role(
                                        "table", name="Captured fixed-cohort interval check"
                                    )
                                ).to_contain_text("0.91")
                                expect(
                                    panel.get_by_text(
                                        "Showing retained repetitions 1–20 of 400", exact=True
                                    )
                                ).to_be_visible()
                                panel.get_by_role(
                                    "button", name="Next retained repetitions"
                                ).click()
                                expect(
                                    panel.get_by_text(
                                        "Showing retained repetitions 21–40 of 400", exact=True
                                    )
                                ).to_be_visible()
                                panel.get_by_role(
                                    "button", name="Previous retained repetitions"
                                ).click()
                            with page.expect_download() as pending:
                                panel.get_by_role(
                                    "button", name="Download original method comparison"
                                ).click()
                            saved = pending.value.path()
                            assert pending.value.suggested_filename == "method-comparison.json"
                            assert saved is not None and Path(saved).read_bytes() == raw.encode(
                                "utf-8"
                            )
                            if name == "integer-scalar":
                                transports.append({"runtime": kind, "case": "native-integer-file"})
                            records.append(
                                {
                                    "runtime": kind,
                                    "case": name,
                                    "sha256": original["sha256"],
                                    "retained": len(original["body"]["comparison"]["runs"]),
                                }
                            )
                        last = panel.get_by_label("Method comparison digest").text_content()
                        assert last is not None
                        retained_bytes = ("\r\n" + raw + "\r\n").encode("utf-8")
                        file_input.set_input_files(
                            {
                                "name": "original-crlf.json",
                                "mimeType": "application/json",
                                "buffer": retained_bytes,
                            }
                        )
                        # HTML textarea presentation normalises CRLF; the admitted source/download must not.
                        expect(field).to_have_value("\n" + raw + "\n")
                        expect(read).to_be_enabled()
                        expect(panel.get_by_label("Method comparison digest")).to_have_text(last)
                        with page.expect_download() as formatted:
                            panel.get_by_role(
                                "button", name="Download original method comparison"
                            ).click()
                        formatted_path = formatted.value.path()
                        assert (
                            formatted_path is not None
                            and Path(formatted_path).read_bytes() == retained_bytes
                        )
                        transports.append(
                            {
                                "runtime": kind,
                                "case": "original-crlf",
                                "sha256": hashlib.sha256(retained_bytes).hexdigest(),
                            }
                        )
                        for filename, malformed in [
                            ("bom.json", b"\xef\xbb\xbf" + retained_bytes),
                            ("invalid-utf8.json", b"\xff" + retained_bytes),
                            (
                                "future-major.json",
                                raw.replace(
                                    "studio.method_comparison.v1", "studio.method_comparison.v2", 1
                                ).encode("utf-8"),
                            ),
                        ]:
                            file_input.set_input_files(
                                {
                                    "name": filename,
                                    "mimeType": "application/json",
                                    "buffer": malformed,
                                }
                            )
                            expect(
                                panel.get_by_role("alert", name="Method comparison import error")
                            ).to_contain_text("refused")
                            expect(panel.get_by_label("Method comparison digest")).to_have_text(
                                last
                            )
                            with page.expect_download() as refused:
                                panel.get_by_role(
                                    "button", name="Download original method comparison"
                                ).click()
                            refused_path = refused.value.path()
                            assert (
                                refused_path is not None
                                and Path(refused_path).read_bytes() == retained_bytes
                            )
                            transports.append({"runtime": kind, "case": filename, "refused": True})
                        field.fill("{")
                        read.click()
                        expect(
                            panel.get_by_role("alert", name="Method comparison import error")
                        ).to_contain_text("refused")
                        expect(panel.get_by_label("Method comparison digest")).to_have_text(last)
                        with page.expect_download() as retained:
                            panel.get_by_role(
                                "button", name="Download original method comparison"
                            ).click()
                        preserved = retained.value.path()
                        assert (
                            preserved is not None
                            and Path(preserved).read_bytes() == retained_bytes
                        )
                        observations.append(
                            kind
                            + ":all-native-rows-selection-negative-interval-and-exact-download"
                        )
                        writes = page.evaluate("window.methodComparisonWrites")
                        assert writes == [], writes
                        observed["storage_write_observations"] = writes
                        navigation.get_by_role("link", name="Model", exact=True).click()
                        compiler = page.locator('[id="/build/compile-recompute"]')
                        compiler.get_by_role("button", name="Recompute in browser").click()
                        expect(compiler.get_by_role("status")).to_have_text(
                            "recomputed digest matches the signed claim"
                        )
                        observations.append(kind + ":original-built-wasm-recompute")
                        # Original workbench cohort includes every route's actual lazy composition.
                        for view in [
                            "Run",
                            "Analyse",
                            "Compare",
                            "Evidence",
                            "Atlas",
                            "Workspace",
                        ]:
                            page.get_by_role("link", name=view, exact=True).click()
                        expect(workspace).to_have_value(draft)
                        assert not errors, errors
                        assert not external, external
                        assert not submissions, submissions
                        assert not page.workers, "Comparison must retain no workers"
                        observed.update(workers=len(page.workers), no_submit=True)
                    finally:
                        if profiler is not None:
                            take_native_coverage(profiler, native, include_workbench=True)
                finally:
                    context.close()
            assert kernels, "Actual original WASM was not requested"
            return observed
        finally:
            browser.close()
