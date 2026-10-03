# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native compiler trace browser boundary
"""Inspect real native snapshots beside the original built WASM source editor."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

from scpn_quantum_control.studio_workspace.canonical import canonical_digest


def run_compiler_trace_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Qualify the real Build route without executing IR or submitting jobs.

    Parameters
    ----------
    base_url
        Owned literal loopback HTTP preview carrying the genuine built WASM.
    evidence
        Caller-owned observations retained if a later browser assertion fails.

    Returns
    -------
    dict
        Browser/runtime identity and actually observed source, mapping, export,
        missing-artifact, refusal and saved-state custody boundaries.

    Raises
    ------
    ValueError
        Preview ownership is invalid before any browser is allocated.
    AssertionError
        A source identity, visible contract or runtime custody check fails.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import Route, expect, sync_playwright

    repo = Path(__file__).resolve().parents[1]
    original = (repo / "data/studio/compiler_trace_demo.json").read_text()
    cases = json.loads((repo / "data/studio/compiler_trace_cases.json").read_text())
    missing = json.loads(original)
    missing["body"]["passes"][1] = {
        "state": "missing",
        "reason": "Native pass artifact was not supplied.",
    }
    missing["body"]["complete"] = False
    missing["sha256"] = canonical_digest(
        "studio.compiler-trace.v1",
        {key: missing[key] for key in ("schema", "body", "extensions")},
    )
    observed = {} if evidence is None else evidence
    observations: list[str] = []
    errors: list[str] = []
    external: list[str] = []
    observed.update(
        scenario="compiler_trace_inspector",
        playwright=version("playwright"),
        observations=observations,
        page_errors=errors,
        external_requests=external,
        fixture_sha256=hashlib.sha256(original.encode()).hexdigest(),
    )
    origin = urlsplit(url).netloc
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        observed["browser"] = browser.version
        try:
            with browser.new_context(service_workers="block", accept_downloads=True) as context:
                context.set_default_timeout(15_000)

                def bound_request(route: Route) -> None:
                    target = urlsplit(route.request.url)
                    if target.scheme != "http" or target.netloc != origin:
                        external.append("outside-preview")
                        route.abort()
                    else:
                        route.continue_()

                context.route("**/*", bound_request)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script("""
                    window.__compilerWorkspaceWrites = [];
                    for (const name of ['put','add','delete','clear']) {
                        const original = IDBObjectStore.prototype[name];
                        IDBObjectStore.prototype[name] = function(...args) {
                            window.__compilerWorkspaceWrites.push(this.name + ':' + name);
                            return original.apply(this, args);
                        };
                    }
                """)
                page.goto(url + "#/build", wait_until="networkidle")
                before = page.evaluate("window.__compilerWorkspaceWrites.length")
                editor = page.get_by_role("region", name="Program authoring", exact=True)
                source = editor.get_by_label("Program source", exact=True)
                source.fill(cases["lowering"]["body"]["source"])
                editor.get_by_role("button", name="Compile source", exact=True).click()
                expect(
                    editor.get_by_role("region", name="Compiled program", exact=True)
                ).to_be_visible()
                expect(
                    editor.get_by_role("heading", name="Emitted — not executed")
                ).to_be_visible()
                observations.append("original-source-compiler-through-genuine-wasm")
                inspector = page.get_by_role("region", name="Compiler trace inspector", exact=True)
                expect(inspector.get_by_role("status")).to_contain_text(
                    "Missing compiler pass artifact"
                )
                inspector.get_by_role("button", name="Open native example", exact=True).click()
                expect(inspector.get_by_role("table", name="Qubit mapping")).to_contain_text(
                    "q[0]q[0]q[1]"
                )
                expect(inspector.get_by_label("Mapped readout")).to_contain_text(
                    "q[0] / c[1] → q[1] / c[1]"
                )
                pinned = inspector.get_by_label("Original compiler source", exact=True)
                expect(pinned).to_have_value(json.loads(original)["body"]["source"])
                inspector.get_by_role(
                    "button", name="Select source operation 1", exact=True
                ).click()
                selection = pinned.evaluate(
                    "(field) => [field.selectionStart,field.selectionEnd,field.value.slice(field.selectionStart,field.selectionEnd)]"
                )
                assert selection[2].startswith("ry(")
                inspector.get_by_label("Compiler pass", exact=True).select_option("1")
                assert (
                    pinned.evaluate(
                        "(field) => [field.selectionStart,field.selectionEnd,field.value.slice(field.selectionStart,field.selectionEnd)]"
                    )
                    == selection
                )
                expect(inspector.get_by_label("Selected original source")).to_have_text(
                    selection[2]
                )
                observations.append("unicode-source-selection-pinned-across-physical-passes")
                export = inspector.get_by_role("button", name="Export admitted trace", exact=True)
                with page.expect_download() as downloaded:
                    export.click()
                path = downloaded.value.path()
                assert path is not None
                assert path.read_text() == original
                observations.append("exact-native-backend-snapshot-export-without-execution")
                draft = inspector.get_by_label("Compiler trace JSON", exact=True)
                draft.fill(json.dumps(missing))
                inspector.get_by_role("button", name="Inspect trace", exact=True).click()
                expect(inspector.get_by_text("Incomplete trace", exact=True)).to_be_visible()
                inspector.get_by_label("Compiler pass", exact=True).select_option("1")
                expect(inspector.get_by_role("status")).to_contain_text(
                    "Native pass artifact was not supplied"
                )
                observations.append("missing-native-artifact-remains-explicit")
                for rejected in [
                    '{"schema":"studio.compiler-trace.v2"}',
                    '{"schema":1,"schema":2}',
                ]:
                    draft.fill(rejected)
                    inspector.get_by_role("button", name="Inspect trace", exact=True).click()
                    expect(inspector.get_by_role("alert")).to_be_visible()
                    expect(pinned).to_have_value(json.loads(original)["body"]["source"])
                    expect(export).to_be_enabled()
                observations.append("unsupported-and-duplicate-imports-preserve-admitted-source")
                draft.fill(json.dumps(cases["lowering"]))
                inspector.get_by_role("button", name="Inspect trace", exact=True).click()
                expect(
                    inspector.get_by_text("Textual MLIR — not executed", exact=True)
                ).to_be_visible()
                expect(inspector.get_by_role("table", name="Gate changes")).to_contain_text(
                    "h10-1"
                )
                expect(inspector.get_by_role("alert")).to_have_count(0)
                observations.append("actual-basis-lowering-effects-and-textual-mlir-recovery")
                draft.fill(json.dumps(cases["crlf"]))
                inspector.get_by_role("button", name="Inspect trace", exact=True).click()
                expect(inspector.get_by_role("heading", name="crlf", exact=True)).to_be_visible()
                inspector.get_by_role(
                    "button", name="Select source operation 1", exact=True
                ).click()
                assert (
                    pinned.evaluate(
                        "(field) => field.value.slice(field.selectionStart,field.selectionEnd)"
                    )
                    == "h q[0];"
                )
                expect(inspector.get_by_label("Selected original source")).to_have_text("h q[0];")
                with page.expect_download() as crlf_download:
                    export.click()
                crlf_path = crlf_download.value.path()
                assert crlf_path is not None
                assert crlf_path.read_text() == json.dumps(cases["crlf"])
                observations.append("crlf-native-source-caret-and-exact-export")
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                observations.append("compiler-source-and-mapping-fit-mobile-workbench")
                assert page.evaluate("window.__compilerWorkspaceWrites.length") == before
                assert not external, external
                assert not errors, errors
                assert not page.workers
                observed.update(
                    execution_status="emitted_not_executed", workers=0, workspace_writes=0
                )
                return observed
        finally:
            browser.close()
