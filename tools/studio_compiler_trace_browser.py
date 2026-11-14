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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scpn_quantum_control.studio_workspace.canonical import canonical_digest
from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

_WRITE_PROBE = """
    window.__compilerWorkspaceWrites = [];
    for (const name of ['put','add','delete','clear']) {
        const original = IDBObjectStore.prototype[name];
        IDBObjectStore.prototype[name] = function(...args) {
            window.__compilerWorkspaceWrites.push(this.name + ':' + name);
            return original.apply(this, args);
        };
    }
"""
_SELECTION = (
    "(field) => [field.selectionStart,field.selectionEnd,"
    "field.value.slice(field.selectionStart,field.selectionEnd)]"
)
_WRITES = "window.__compilerWorkspaceWrites.length"


def _missing_pass_trace(original: str) -> dict[str, Any]:
    """Withdraw the second native pass artifact from the admitted trace and re-sign it.

    Parameters
    ----------
    original
        Exact admitted compiler trace text.

    Returns
    -------
    dict
        Trace whose second pass is missing, with its canonical digest.

    """
    missing: dict[str, Any] = json.loads(original)
    missing["body"]["passes"][1] = {
        "state": "missing",
        "reason": "Native pass artifact was not supplied.",
    }
    missing["body"]["complete"] = False
    missing["sha256"] = canonical_digest(
        "studio.compiler-trace.v1",
        {key: missing[key] for key in ("schema", "body", "extensions")},
    )
    return missing


def _inspect(inspector: Locator, trace: str) -> None:
    """Paste a trace into the inspector and ask it to inspect.

    Parameters
    ----------
    inspector
        The compiler trace inspector region.
    trace
        Trace text to inspect.

    """
    inspector.get_by_label("Compiler trace JSON", exact=True).fill(trace)
    inspector.get_by_role("button", name="Inspect trace", exact=True).click()


def _export_text(page: Page, export: Locator) -> str:
    """Export the admitted trace and return the downloaded text.

    Parameters
    ----------
    page
        Owned page that receives the download.
    export
        The export button.

    Returns
    -------
    str
        Exact downloaded text.

    """
    with page.expect_download() as downloaded:
        export.click()
    path = downloaded.value.path()
    assert path is not None
    return path.read_text()


def _compile_source(page: Page, source: str, observations: list[str]) -> Locator:
    """Compile the lowering case through the genuine WASM source compiler.

    Parameters
    ----------
    page
        Owned page on the Build route.
    source
        Program source of the lowering case.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    Locator
        The compiler trace inspector, still reporting its missing artifact.

    """
    from playwright.sync_api import expect

    editor = page.get_by_role("region", name="Program authoring", exact=True)
    editor.get_by_label("Program source", exact=True).fill(source)
    editor.get_by_role("button", name="Compile source", exact=True).click()
    expect(editor.get_by_role("region", name="Compiled program", exact=True)).to_be_visible()
    expect(editor.get_by_role("heading", name="Emitted — not executed")).to_be_visible()
    observations.append("original-source-compiler-through-genuine-wasm")
    inspector = page.get_by_role("region", name="Compiler trace inspector", exact=True)
    expect(inspector.get_by_role("status")).to_contain_text("Missing compiler pass artifact")
    return inspector


def _pin_native_example(
    page: Page, inspector: Locator, original: str, observations: list[str]
) -> tuple[Locator, Locator]:
    """Open the native example, pin a Unicode source selection and export it exactly.

    Parameters
    ----------
    page
        Owned page that receives the download.
    inspector
        The compiler trace inspector region.
    original
        Exact admitted compiler trace text.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    tuple
        The pinned original-source field and the export button.

    """
    from playwright.sync_api import expect

    inspector.get_by_role("button", name="Open native example", exact=True).click()
    expect(inspector.get_by_role("table", name="Qubit mapping")).to_contain_text("q[0]q[0]q[1]")
    expect(inspector.get_by_label("Mapped readout")).to_contain_text("q[0] / c[1] → q[1] / c[1]")
    pinned = inspector.get_by_label("Original compiler source", exact=True)
    expect(pinned).to_have_value(json.loads(original)["body"]["source"])
    operation = "Select source operation 1. ry (6:1)"
    inspector.get_by_role("button", name=operation, exact=True).click()
    selection = pinned.evaluate(_SELECTION)
    assert selection[2].startswith("ry(")
    inspector.get_by_label("Compiler pass", exact=True).select_option("1")
    assert pinned.evaluate(_SELECTION) == selection
    expect(inspector.get_by_label("Selected original source")).to_have_text(selection[2])
    observations.append("unicode-source-selection-pinned-across-physical-passes")
    export = inspector.get_by_role("button", name="Export admitted trace", exact=True)
    assert _export_text(page, export) == original
    observations.append("exact-native-backend-snapshot-export-without-execution")
    return pinned, export


def _check_missing_and_refusals(
    inspector: Locator,
    fields: tuple[Locator, Locator],
    missing: dict[str, Any],
    source: str,
    observations: list[str],
) -> None:
    """Keep a missing pass explicit and keep the admitted source after refused imports.

    Parameters
    ----------
    inspector
        The compiler trace inspector region.
    fields
        The pinned original-source field and the export button.
    missing
        Trace whose second pass artifact is missing.
    source
        Original source of the admitted trace.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    pinned, export = fields
    _inspect(inspector, json.dumps(missing))
    expect(inspector.get_by_text("Incomplete trace", exact=True)).to_be_visible()
    inspector.get_by_label("Compiler pass", exact=True).select_option("1")
    expect(inspector.get_by_role("status")).to_contain_text(
        "Native pass artifact was not supplied"
    )
    observations.append("missing-native-artifact-remains-explicit")
    for rejected in ('{"schema":"studio.compiler-trace.v2"}', '{"schema":1,"schema":2}'):
        _inspect(inspector, rejected)
        expect(inspector.get_by_role("alert")).to_be_visible()
        expect(pinned).to_have_value(source)
        expect(export).to_be_enabled()
    observations.append("unsupported-and-duplicate-imports-preserve-admitted-source")


def _check_lowering_and_crlf(
    page: Page,
    inspector: Locator,
    fields: tuple[Locator, Locator],
    cases: dict[str, Any],
    observations: list[str],
) -> None:
    """Recover with the lowering case, then pin and export the CRLF case exactly.

    Parameters
    ----------
    page
        Owned page that receives the download.
    inspector
        The compiler trace inspector region.
    fields
        The pinned original-source field and the export button.
    cases
        Committed native compiler trace cases.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    pinned, export = fields
    _inspect(inspector, json.dumps(cases["lowering"]))
    expect(inspector.get_by_text("Textual MLIR — not executed", exact=True)).to_be_visible()
    expect(inspector.get_by_role("table", name="Gate changes")).to_contain_text("h10-1")
    expect(inspector.get_by_role("alert")).to_have_count(0)
    observations.append("actual-basis-lowering-effects-and-textual-mlir-recovery")
    _inspect(inspector, json.dumps(cases["crlf"]))
    expect(inspector.get_by_role("heading", name="crlf", exact=True)).to_be_visible()
    operation = "Select source operation 1. h (5:1)"
    inspector.get_by_role("button", name=operation, exact=True).click()
    assert pinned.evaluate(_SELECTION)[2] == "h q[0];"
    expect(inspector.get_by_label("Selected original source")).to_have_text("h q[0];")
    assert _export_text(page, export) == json.dumps(cases["crlf"])
    observations.append("crlf-native-source-caret-and-exact-export")


def _walk(
    page: Page, url: str, observations: list[str], *, original: str, cases: dict[str, Any]
) -> dict[str, object]:
    """Compile, inspect, refuse and export on the Build route without saving state.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.
    original
        Exact admitted compiler trace text.
    cases
        Committed native compiler trace cases.

    Returns
    -------
    dict
        Execution status and the absence of workspace writes.

    """
    page.add_init_script(_WRITE_PROBE)
    page.goto(url + "#/build", wait_until="networkidle")
    before = page.evaluate(_WRITES)
    inspector = _compile_source(page, cases["lowering"]["body"]["source"], observations)
    fields = _pin_native_example(page, inspector, original, observations)
    source = json.loads(original)["body"]["source"]
    _check_missing_and_refusals(
        inspector, fields, _missing_pass_trace(original), source, observations
    )
    _check_lowering_and_crlf(page, inspector, fields, cases, observations)
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    observations.append("compiler-source-and-mapping-fit-mobile-workbench")
    assert page.evaluate(_WRITES) == before
    return {"execution_status": "emitted_not_executed", "workspace_writes": 0}


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
    repo = Path(__file__).resolve().parents[1]
    original = (repo / "data/studio/compiler_trace_demo.json").read_text()
    cases = json.loads((repo / "data/studio/compiler_trace_cases.json").read_text())
    observed = {} if evidence is None else evidence
    observed["fixture_sha256"] = hashlib.sha256(original.encode()).hexdigest()
    return run_custodied_journey(
        base_url,
        observed,
        "compiler_trace_inspector",
        partial(_walk, original=original, cases=cases),
        downloads=True,
    )
