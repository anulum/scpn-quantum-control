# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program authoring browser boundary
"""Exercise original editor, built WASM, exact export and refusal recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page, Route

_DIGEST_PROBE = """
    window.__programDigests = 0;
    const digest = crypto.subtle.digest.bind(crypto.subtle);
    Object.defineProperty(crypto.subtle, 'digest', {value: (...args) =>
        digest(...args).finally(() => { window.__programDigests += 1; })});
"""
_KERNEL = "scpn_quantum_studio_wasm_kernel.wasm"


@dataclass
class _KernelHold:
    """Hold the next owned kernel request until the journey releases it."""

    origin: str
    armed: bool = False
    held: list[Route] = field(default_factory=list)

    def __call__(self, route: Route) -> None:
        """Hold the armed owned kernel request; pass every other one to the guard.

        Parameters
        ----------
        route
            Intercepted kernel request.

        """
        if self.armed and urlsplit(route.request.url).netloc == self.origin:
            self.armed = False
            self.held.append(route)
        else:
            route.fallback()

    def dispose(self) -> None:
        """Abort every request still held."""
        while self.held:
            self.held.pop().abort()


@dataclass(frozen=True)
class _Editor:
    """Program authoring controls the journey drives."""

    region: Locator
    source: Locator
    compile: Locator
    export: Locator

    @classmethod
    def on(cls, page: Page) -> _Editor:
        """Locate the program authoring controls on the Build route.

        Parameters
        ----------
        page
            Owned page on the Build route.

        Returns
        -------
        _Editor
            The editor region, source field, compile and export buttons.

        """
        region = page.get_by_role("region", name="Program authoring", exact=True)
        return cls(
            region,
            region.get_by_label("Program source", exact=True),
            region.get_by_role("button", name="Compile source", exact=True),
            region.get_by_role("button", name="Export exact source", exact=True),
        )

    def compile_source(self, source: str) -> None:
        """Replace the source and compile it.

        Parameters
        ----------
        source
            Program source to compile.

        """
        self.source.fill(source)
        self.compile.click()


def _emitted_record(editor: _Editor, case: dict[str, Any]) -> dict[str, Any]:
    """Check the emitted record of an admitted case against the shared corpus.

    Parameters
    ----------
    editor
        Program authoring controls.
    case
        Admitted corpus case.

    Returns
    -------
    dict
        The exact emitted, not executed, record.

    """
    from playwright.sync_api import expect

    plan = editor.region.get_by_role("region", name="Compiled program", exact=True)
    expect(plan.get_by_role("heading", name="Emitted — not executed")).to_be_visible()
    record: dict[str, Any] = json.loads(plan.locator("details pre").text_content() or "null")
    assert record["source"] == case["source"]
    assert record["source_sha256"] == hashlib.sha256(case["source"].encode()).hexdigest()
    assert record["execution_status"] == "emitted_not_executed"
    assert record["measurements"] == case["measurements"]
    operations = record["operations"]
    assert [operation["name"] for operation in operations] == case["operations"]
    for expected, key in (("parameters", "parameters"), ("conditions", "condition")):
        if expected in case:
            assert [operation[key] for operation in operations] == case[expected]
    return record


def _check_corpus(
    editor: _Editor, corpus: dict[str, Any], records: list[dict[str, object]]
) -> None:
    """Compile every shared corpus case through the built WASM compiler.

    Parameters
    ----------
    editor
        Program authoring controls.
    corpus
        Shared original source corpus.
    records
        Caller-owned record of each emitted program.

    """
    from playwright.sync_api import expect

    for case in corpus["cases"]:
        editor.compile_source(case["source"])
        expect(editor.compile).to_be_enabled()
        if case["ok"]:
            records.append(_emitted_record(editor, case))
            expect(editor.export).to_be_enabled()
        else:
            expect(editor.region.get_by_role("alert")).to_contain_text(case["code"])
            expect(editor.region.locator("mark")).to_have_text(case["token"])
            compiled = editor.region.get_by_role("region", name="Compiled program")
            expect(compiled).to_have_count(0)
            expect(editor.export).to_be_disabled()
        expect(editor.source).to_have_value(case["source"])


def _check_export_and_append(
    page: Page, editor: _Editor, source: str, observations: list[str]
) -> None:
    """Download the exact source, then append a structured conditioned phase.

    Parameters
    ----------
    page
        Owned page that receives the download.
    editor
        Program authoring controls.
    source
        Admitted source compiled before the export.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    editor.compile_source(source)
    expect(editor.export).to_be_enabled()
    with page.expect_download() as download:
        editor.export.click()
    downloaded = download.value.path()
    assert downloaded is not None
    assert downloaded.read_text() == editor.source.input_value()
    observations.append("exact-source-download-without-execution")
    editor.region.get_by_label("Operation", exact=True).select_option("rz")
    editor.region.get_by_label("Parameters (decimal radians)").fill("-0.7853981633974492")
    editor.region.get_by_label("Classical condition c equals (optional)").fill("2")
    editor.region.get_by_role("button", name="Append operation").click()
    expect(editor.export).to_be_disabled()
    expect(editor.region.get_by_role("region", name="Compiled program")).to_have_count(0)
    editor.compile.click()
    expect(editor.export).to_be_enabled()
    program = editor.region.get_by_role("table", name="Program IR")
    expect(program).to_contain_text("bfe921fb54442d20")
    expect(program).to_contain_text("if(c==2)")
    observations.append("structured-phase-condition-and-draft-invalidation")


def _check_mobile(page: Page, editor: _Editor, observations: list[str]) -> None:
    """Keep the source and the opened emitted record within a phone-width workbench.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    editor
        Program authoring controls.
    observations
        Caller-owned list that records each contract once it holds.

    """
    page.set_viewport_size({"width": 390, "height": 844})
    editor.region.get_by_text("Exact emitted record", exact=True).click()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    editor.region.get_by_text("Exact emitted record", exact=True).click()
    page.set_viewport_size({"width": 1280, "height": 720})
    observations.append("mobile-source-and-ir-remain-within-workbench")


def _check_delayed_result(
    page: Page, editor: _Editor, hold: _KernelHold, source: str, observations: list[str]
) -> None:
    """Let a held real WASM result arrive after the draft changed and find it ignored.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    editor
        Program authoring controls.
    hold
        Holder of the next owned kernel request.
    source
        Admitted source that the pending compilation started from.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    trace = editor.region.get_by_role("region", name="Compilation trace")
    before_trace = trace.inner_text()
    hold.armed = True
    editor.compile.click()
    compiling = editor.region.get_by_role("button", name="Compiling source…")
    expect(compiling).to_be_disabled()
    editor.source.fill(source + "x q[0];")
    expect(editor.export).to_be_disabled()
    assert len(hold.held) == 1
    digests = page.evaluate("window.__programDigests")
    hold.held.pop().fallback()
    page.wait_for_function("before => window.__programDigests > before", arg=digests)
    expect(editor.export).to_be_disabled()
    expect(editor.region.get_by_role("region", name="Compiled program")).to_have_count(0)
    assert trace.inner_text() == before_trace
    editor.compile.click()
    expect(editor.export).to_be_enabled()
    observations.append("delayed-real-wasm-result-cannot-revalidate-changed-draft")


def _walk(
    page: Page,
    url: str,
    observations: list[str],
    *,
    corpus: dict[str, Any],
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Write, compile, export and invalidate programs on the Build route.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.
    corpus
        Shared original source corpus.
    records
        Caller-owned record of each emitted program.

    Returns
    -------
    dict
        The execution status.

    """
    hold = _KernelHold(urlsplit(url).netloc)
    page.route(f"**/{_KERNEL}", hold)
    page.add_init_script(_DIGEST_PROBE)
    try:
        page.goto(url + "#/build", wait_until="networkidle")
        editor = _Editor.on(page)
        _check_corpus(editor, corpus, records)
        observations.append("shared-original-source-corpus-through-built-wasm")
        first = corpus["cases"][0]["source"]
        _check_export_and_append(page, editor, first, observations)
        _check_mobile(page, editor, observations)
        _check_delayed_result(page, editor, hold, first, observations)
    finally:
        hold.dispose()
    return {"execution_status": "emitted_not_executed"}


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
    repo = Path(__file__).resolve().parents[1]
    corpus = json.loads((repo / "tests/data/program_authoring/corpus.json").read_text())
    records: list[dict[str, object]] = []
    observed = {} if evidence is None else evidence
    observed["records"] = records
    return run_custodied_journey(
        base_url,
        observed,
        "program_authoring",
        partial(_walk, corpus=corpus, records=records),
        downloads=True,
    )
