# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real supported program browser journey
"""Exercise original editor, built WASM, exact export and refusal recovery."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit


def run_program_authoring_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Verify actual program authoring without execution or provider requests.

    Parameters
    ----------
    base_url
        Owned loopback preview with the genuine built source compiler WASM.
    evidence
        Caller-owned partial observations retained if a later assertion fails.

    Returns
    -------
    dict
        Actual browser versions, original compiler records and observed outcomes.

    Raises
    ------
    ValueError
        The preview is not an owned literal loopback HTTP address.
    AssertionError
        Source identity, a rendered boundary or a refusal contract fails.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import Route, expect, sync_playwright

    origin = urlsplit(url).netloc
    corpus = json.loads(
        (
            Path(__file__).resolve().parents[1] / "tests/data/program_authoring/corpus.json"
        ).read_text()
    )
    observations: list[str] = []
    records: list[dict[str, object]] = []
    rejected: list[str] = []
    errors: list[str] = []
    observed = {} if evidence is None else evidence
    observed.update(
        scenario="program_authoring",
        observations=observations,
        records=records,
        external_requests=rejected,
        page_errors=errors,
        playwright=version("playwright"),
    )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        observed["browser"] = browser.version
        try:
            with browser.new_context(service_workers="block", accept_downloads=True) as context:
                context.set_default_timeout(15_000)
                held: list[Route] = []
                hold_next = False

                def bound_request(route: Route) -> None:
                    nonlocal hold_next
                    target = urlsplit(route.request.url)
                    if target.scheme != "http" or target.netloc != origin:
                        rejected.append("outside-preview")
                        route.abort()
                    elif hold_next and target.path.endswith(
                        "scpn_quantum_studio_wasm_kernel.wasm"
                    ):
                        hold_next = False
                        held.append(route)
                    else:
                        route.continue_()

                context.route("**/*", bound_request)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script("""
                    window.__programDigests = 0;
                    const digest = crypto.subtle.digest.bind(crypto.subtle);
                    Object.defineProperty(crypto.subtle, 'digest', {value: (...args) =>
                        digest(...args).finally(() => { window.__programDigests += 1; })});
                """)
                try:
                    page.goto(url + "#/build", wait_until="networkidle")
                    editor = page.get_by_role("region", name="Program authoring", exact=True)
                    source = editor.get_by_label("Program source", exact=True)
                    compile_button = editor.get_by_role(
                        "button", name="Compile source", exact=True
                    )
                    export = editor.get_by_role("button", name="Export exact source", exact=True)
                    for case in corpus["cases"]:
                        source.fill(case["source"])
                        compile_button.click()
                        expect(compile_button).to_be_enabled()
                        if case["ok"]:
                            plan = editor.get_by_role(
                                "region", name="Compiled program", exact=True
                            )
                            expect(
                                plan.get_by_role("heading", name="Emitted — not executed")
                            ).to_be_visible()
                            record = json.loads(
                                plan.locator("details pre").text_content() or "null"
                            )
                            assert record["source"] == case["source"]
                            assert (
                                record["source_sha256"]
                                == hashlib.sha256(case["source"].encode()).hexdigest()
                            )
                            assert record["execution_status"] == "emitted_not_executed"
                            assert record["measurements"] == case["measurements"]
                            assert [op["name"] for op in record["operations"]] == case[
                                "operations"
                            ]
                            if "parameters" in case:
                                assert [op["parameters"] for op in record["operations"]] == case[
                                    "parameters"
                                ]
                            if "conditions" in case:
                                assert [op["condition"] for op in record["operations"]] == case[
                                    "conditions"
                                ]
                            records.append(record)
                            expect(export).to_be_enabled()
                        else:
                            expect(editor.get_by_role("alert")).to_contain_text(case["code"])
                            expect(editor.locator("mark")).to_have_text(case["token"])
                            expect(
                                editor.get_by_role("region", name="Compiled program")
                            ).to_have_count(0)
                            expect(export).to_be_disabled()
                        expect(source).to_have_value(case["source"])
                    observations.append("shared-original-source-corpus-through-built-wasm")
                    source.fill(corpus["cases"][0]["source"])
                    compile_button.click()
                    expect(export).to_be_enabled()
                    with page.expect_download() as download:
                        export.click()
                    downloaded = download.value.path()
                    assert downloaded is not None
                    assert downloaded.read_text() == source.input_value()
                    observations.append("exact-source-download-without-execution")
                    editor.get_by_label("Operation", exact=True).select_option("rz")
                    editor.get_by_label("Parameters (decimal radians)").fill("-0.7853981633974492")
                    editor.get_by_label("Classical condition c equals (optional)").fill("2")
                    editor.get_by_role("button", name="Append operation").click()
                    expect(export).to_be_disabled()
                    expect(editor.get_by_role("region", name="Compiled program")).to_have_count(0)
                    compile_button.click()
                    expect(export).to_be_enabled()
                    expect(editor.get_by_role("table", name="Program IR")).to_contain_text(
                        "bfe921fb54442d20"
                    )
                    expect(editor.get_by_role("table", name="Program IR")).to_contain_text(
                        "if(c==2)"
                    )
                    observations.append("structured-phase-condition-and-draft-invalidation")
                    page.set_viewport_size({"width": 390, "height": 844})
                    editor.get_by_text("Exact emitted record", exact=True).click()
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= window.innerWidth"
                    )
                    editor.get_by_text("Exact emitted record", exact=True).click()
                    page.set_viewport_size({"width": 1280, "height": 720})
                    observations.append("mobile-source-and-ir-remain-within-workbench")
                    before_trace = editor.get_by_role(
                        "region", name="Compilation trace"
                    ).inner_text()
                    hold_next = True
                    compile_button.click()
                    expect(editor.get_by_role("button", name="Compiling source…")).to_be_disabled()
                    source.fill(corpus["cases"][0]["source"] + "x q[0];")
                    expect(export).to_be_disabled()
                    assert len(held) == 1
                    digests = page.evaluate("window.__programDigests")
                    held.pop().continue_()
                    page.wait_for_function(
                        "before => window.__programDigests > before", arg=digests
                    )
                    expect(export).to_be_disabled()
                    expect(editor.get_by_role("region", name="Compiled program")).to_have_count(0)
                    assert (
                        editor.get_by_role("region", name="Compilation trace").inner_text()
                        == before_trace
                    )
                    compile_button.click()
                    expect(export).to_be_enabled()
                    observations.append("delayed-real-wasm-result-cannot-revalidate-changed-draft")
                    assert not errors, errors
                    assert not rejected, rejected
                    assert not page.workers
                finally:
                    for route in held:
                        route.abort()
            observed.update(workers=0, execution_status="emitted_not_executed")
            return observed
        finally:
            browser.close()
