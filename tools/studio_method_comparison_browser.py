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
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from tools.studio_preview_custody import preview_route_guard

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Locator, Page

_WRITE_PROBE = """() => {
    window.methodComparisonWrites=[];
    for(const name of ['put','add','delete','clear']) {
        const original=IDBObjectStore.prototype[name];
        IDBObjectStore.prototype[name]=function(...args) {
            window.methodComparisonWrites.push(name);return original.apply(this,args);
        };
    }
}"""
_DRAFT = '{"unsaved":"native comparison draft α"}'
_REFUSED_CASES = frozenset(
    {
        "model-mismatch",
        "precision-mismatch",
        "missing-population",
        "unknown-cost",
        "budget-exceeded",
        "original-solver",
        "original-ode-only",
    }
)
_DOWNLOAD = "Download original method comparison"
_IMPORT_ERROR = "Method comparison import error"


@dataclass
class _Records:
    """Caller-visible evidence lists the journey fills."""

    records: list[dict[str, object]] = field(default_factory=list)
    native: list[dict[str, object]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    external: list[str] = field(default_factory=list)
    submissions: list[str] = field(default_factory=list)
    kernels: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    transports: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class _Panel:
    """Method comparison controls on one page."""

    page: Page
    region: Locator
    json_text: Locator
    read: Locator
    file_input: Locator

    def digest(self) -> Locator:
        """Return the admitted comparison digest.

        Returns
        -------
        Locator
            The digest element.

        """
        return self.region.get_by_label("Method comparison digest")

    def download(self) -> bytes:
        """Download the admitted comparison and return its exact bytes.

        Returns
        -------
        bytes
            Downloaded bytes.

        """
        with self.page.expect_download() as pending:
            self.region.get_by_role("button", name=_DOWNLOAD).click()
        saved = pending.value.path()
        assert saved is not None
        return Path(saved).read_bytes()

    def upload(self, name: str, content: bytes) -> None:
        """Choose a comparison file through the native file input.

        Parameters
        ----------
        name
            File name.
        content
            Exact file bytes.

        """
        self.file_input.set_input_files(
            {"name": name, "mimeType": "application/json", "buffer": content}
        )


def _owned_targets(base_url: str, source_url: str | None) -> list[tuple[str, str]]:
    """Validate the built preview and the optional distinct root source origin.

    Parameters
    ----------
    base_url : str
        Owned built Studio preview with its genuine original Rust WASM.
    source_url : str or None
        Distinct owned root Vite source origin for actual debugger counters.

    Returns
    -------
    list[tuple[str, str]]
        Runtime kind and URL of each target, built first.

    Raises
    ------
    ValueError
        Preview/source ownership, root path or distinct-origin contract fails.

    """
    from tools.studio_browser_journey import loopback_url

    preview = loopback_url(base_url)
    source = None if source_url is None else loopback_url(source_url)
    if source is None:
        return [("built", preview)]
    a, b = urlsplit(preview), urlsplit(source)
    same_host = ipaddress.ip_address(cast(str, a.hostname)) == ipaddress.ip_address(
        cast(str, b.hostname)
    )
    if (same_host and a.port == b.port) or b.path != "/":
        raise ValueError("Method comparison requires a distinct owned root source server")
    return [("built", preview), ("source", source)]


def _open_panel(page: Page, target: str) -> _Panel:
    """Keep a workspace draft, probe storage writes and open the comparison panel.

    Parameters
    ----------
    page : Page
        Owned page bound to one target.
    target : str
        Owned target URL.

    Returns
    -------
    _Panel
        The comparison controls.

    """
    from playwright.sync_api import expect

    page.goto(target, wait_until="networkidle")
    page.get_by_label("Workspace archive JSON").fill(_DRAFT)
    page.evaluate(_WRITE_PROBE)
    navigation = page.get_by_role("navigation", name="Workbench views")
    navigation.get_by_role("link", name="Compare", exact=True).click()
    region = page.get_by_role("region", name="Method comparison and uncertainty", exact=True)
    expect(region).to_be_visible()
    return _Panel(
        page,
        region,
        region.get_by_label("Native method comparison JSON", exact=True),
        region.get_by_role("button", name="Read native method comparison", exact=True),
        region.get_by_label("Native comparison file", exact=True),
    )


def _check_population(panel: _Panel, summary: Locator) -> None:
    """Check the population case's summaries and its retained negative difference.

    Parameters
    ----------
    panel : _Panel
        The comparison controls.
    summary : Locator
        All-run method summaries table.

    """
    from playwright.sync_api import expect

    expect(summary).to_contain_text("30.0")
    reference = summary.get_by_role("row", name=re.compile(r"^reference\s"))
    expect(reference).to_contain_text("1.0")
    candidate = summary.get_by_role("row", name=re.compile(r"^candidate\s"))
    expect(candidate).to_contain_text("2.0")
    retained = panel.region.get_by_role("table", name="All retained repetitions")
    expect(retained).to_contain_text("-1.0")


def _check_intervals(panel: _Panel) -> None:
    """Check the captured interval and page the 400 retained repetitions.

    Parameters
    ----------
    panel : _Panel
        The comparison controls.

    """
    from playwright.sync_api import expect

    region = panel.region
    cohort = region.get_by_role("table", name="Captured fixed-cohort interval check")
    expect(cohort).to_contain_text("0.91")
    first = "Showing retained repetitions 1–20 of 400"
    expect(region.get_by_text(first, exact=True)).to_be_visible()
    region.get_by_role("button", name="Next retained repetitions").click()
    second = "Showing retained repetitions 21–40 of 400"
    expect(region.get_by_text(second, exact=True)).to_be_visible()
    region.get_by_role("button", name="Previous retained repetitions").click()


def _check_case(panel: _Panel, name: str, raw: str) -> dict[str, Any]:
    """Admit one native case, select its last repetition and download it exactly.

    Parameters
    ----------
    panel : _Panel
        The comparison controls.
    name : str
        Case name; ``integer-scalar`` arrives through the file input.
    raw : str
        Exact case text.

    Returns
    -------
    dict[str, Any]
        The admitted original comparison.

    """
    from playwright.sync_api import expect

    original: dict[str, Any] = json.loads(raw)
    if name == "integer-scalar":
        panel.upload("original-integer.json", raw.encode("utf-8"))
    else:
        panel.json_text.fill(raw)
        panel.read.click()
    expect(panel.read).to_be_enabled()
    expect(panel.digest()).to_have_text(original["sha256"])
    summary = panel.region.get_by_role("table", name="All-run method summaries")
    before = summary.inner_text()
    selection = panel.region.get_by_label("Inspect retained repetition", exact=True)
    selection.select_option(original["body"]["comparison"]["runs"][-1]["run_id"])
    repetition = panel.region.get_by_role("region", name="Selected original repetition")
    expect(repetition).to_be_visible()
    assert summary.inner_text() == before
    if name == "population":
        _check_population(panel, summary)
    if name in _REFUSED_CASES:
        refusal = panel.region.get_by_role("alert", name="Native comparison refusal")
        expect(refusal).to_be_visible()
    if name == "intervals":
        _check_intervals(panel)
    with panel.page.expect_download() as pending:
        panel.region.get_by_role("button", name=_DOWNLOAD).click()
    saved = pending.value.path()
    assert pending.value.suggested_filename == "method-comparison.json"
    assert saved is not None and Path(saved).read_bytes() == raw.encode("utf-8")
    return original


def _check_cases(
    panel: _Panel, cases: dict[str, str], kind: str, evidence: _Records
) -> tuple[str, str]:
    """Admit every native case and record each one.

    Parameters
    ----------
    panel : _Panel
        The comparison controls.
    cases : dict[str, str]
        Case texts by name.
    kind : str
        Runtime kind, ``built`` or ``source``.
    evidence : _Records
        Evidence lists.

    Returns
    -------
    tuple[str, str]
        The last case text and its admitted digest.

    """
    raw = ""
    for name, raw in cases.items():
        original = _check_case(panel, name, raw)
        if name == "integer-scalar":
            evidence.transports.append({"runtime": kind, "case": "native-integer-file"})
        evidence.records.append(
            {
                "runtime": kind,
                "case": name,
                "sha256": original["sha256"],
                "retained": len(original["body"]["comparison"]["runs"]),
            }
        )
    last = panel.digest().text_content()
    assert last is not None
    return raw, last


def _check_crlf(panel: _Panel, raw: str, last: str, kind: str, evidence: _Records) -> bytes:
    """Admit the last case with CRLF framing and download those exact bytes.

    Parameters
    ----------
    panel : _Panel
        The comparison controls.
    raw : str
        Last case text.
    last : str
        Its admitted digest.
    kind : str
        Runtime kind.
    evidence : _Records
        Evidence lists.

    Returns
    -------
    bytes
        The retained CRLF-framed bytes.

    """
    from playwright.sync_api import expect

    retained_bytes = ("\r\n" + raw + "\r\n").encode("utf-8")
    panel.upload("original-crlf.json", retained_bytes)
    # HTML textarea presentation normalises CRLF; the admitted source/download must not.
    expect(panel.json_text).to_have_value("\n" + raw + "\n")
    expect(panel.read).to_be_enabled()
    expect(panel.digest()).to_have_text(last)
    assert panel.download() == retained_bytes
    evidence.transports.append(
        {
            "runtime": kind,
            "case": "original-crlf",
            "sha256": hashlib.sha256(retained_bytes).hexdigest(),
        }
    )
    return retained_bytes


def _check_refusals(
    panel: _Panel, raw: str, admitted: tuple[str, bytes], kind: str, evidence: _Records
) -> None:
    """Refuse BOM, invalid UTF-8, a future major and malformed text, keeping the admitted bytes.

    Parameters
    ----------
    panel : _Panel
        The comparison controls.
    raw : str
        Last case text.
    admitted : tuple[str, bytes]
        The admitted digest and its retained CRLF-framed bytes.
    kind : str
        Runtime kind.
    evidence : _Records
        Evidence lists.

    """
    from playwright.sync_api import expect

    last, retained_bytes = admitted
    future = raw.replace("studio.method_comparison.v1", "studio.method_comparison.v2", 1)
    for filename, malformed in [
        ("bom.json", b"\xef\xbb\xbf" + retained_bytes),
        ("invalid-utf8.json", b"\xff" + retained_bytes),
        ("future-major.json", future.encode("utf-8")),
    ]:
        panel.upload(filename, malformed)
        alert = panel.region.get_by_role("alert", name=_IMPORT_ERROR)
        expect(alert).to_contain_text("refused")
        expect(panel.digest()).to_have_text(last)
        assert panel.download() == retained_bytes
        evidence.transports.append({"runtime": kind, "case": filename, "refused": True})
    panel.json_text.fill("{")
    panel.read.click()
    expect(panel.region.get_by_role("alert", name=_IMPORT_ERROR)).to_contain_text("refused")
    expect(panel.digest()).to_have_text(last)
    assert panel.download() == retained_bytes


def _check_recompute_and_views(page: Page, kind: str, evidence: _Records) -> None:
    """Recompute the built WASM claim, visit every view and find the draft kept.

    Parameters
    ----------
    page : Page
        Owned page bound to one target.
    kind : str
        Runtime kind.
    evidence : _Records
        Evidence lists.

    """
    from playwright.sync_api import expect

    navigation = page.get_by_role("navigation", name="Workbench views")
    navigation.get_by_role("link", name="Model", exact=True).click()
    compiler = page.locator('[id="/build/compile-recompute"]')
    compiler.get_by_role("button", name="Recompute in browser").click()
    expect(compiler.get_by_role("status")).to_have_text(
        "recomputed digest matches the signed claim"
    )
    evidence.observations.append(kind + ":original-built-wasm-recompute")
    # Original workbench cohort includes every route's actual lazy composition.
    for view in ["Run", "Analyse", "Compare", "Evidence", "Atlas", "Workspace"]:
        page.get_by_role("link", name=view, exact=True).click()
    expect(page.get_by_label("Workspace archive JSON")).to_have_value(_DRAFT)


def _walk(
    page: Page,
    target: tuple[str, str],
    cases: dict[str, str],
    evidence: _Records,
    observed: dict[str, object],
) -> None:
    """Admit, select, refuse and download every case, then recompute and revisit.

    Parameters
    ----------
    page : Page
        Owned page bound to one target.
    target : tuple[str, str]
        Runtime kind and owned target URL.
    cases : dict[str, str]
        Case texts by name.
    evidence : _Records
        Evidence lists.
    observed : dict[str, object]
        Caller-owned evidence receiving the recorded storage writes.

    """
    kind, url = target
    panel = _open_panel(page, url)
    raw, last = _check_cases(panel, cases, kind, evidence)
    retained_bytes = _check_crlf(panel, raw, last, kind, evidence)
    _check_refusals(panel, raw, (last, retained_bytes), kind, evidence)
    evidence.observations.append(
        kind + ":all-native-rows-selection-negative-interval-and-exact-download"
    )
    writes = cast(list[str], page.evaluate("window.methodComparisonWrites"))
    assert writes == [], writes
    observed["storage_write_observations"] = writes
    _check_recompute_and_views(page, kind, evidence)


def _compare_on(
    browser: Browser,
    target: tuple[str, str],
    cases: dict[str, str],
    evidence: _Records,
    observed: dict[str, object],
) -> None:
    """Run the journey on one target in its own context and check its custody.

    Parameters
    ----------
    browser : Browser
        Launched browser.
    target : tuple[str, str]
        Runtime kind and owned target URL; the source target records native coverage.
    cases : dict[str, str]
        Case texts by name.
    evidence : _Records
        Evidence lists.
    observed : dict[str, object]
        Caller-owned evidence.

    """
    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    context = browser.new_context(service_workers="block", accept_downloads=True)
    context.set_default_timeout(15000)
    try:
        guard = preview_route_guard(
            urlsplit(target[1]).netloc, evidence.external, evidence.submissions, evidence.kernels
        )
        context.route("**/*", guard)
        page = context.new_page()
        page.on("pageerror", lambda error: evidence.errors.append(str(error)))
        profiler = start_native_coverage(page) if target[0] == "source" else None
        try:
            _walk(page, target, cases, evidence, observed)
            for refused in (evidence.errors, evidence.external, evidence.submissions):
                assert not refused, refused
            assert not page.workers, "Comparison must retain no workers"
            observed.update(workers=len(page.workers), no_submit=True)
        finally:
            if profiler is not None:
                take_native_coverage(profiler, evidence.native, include_workbench=True)
    finally:
        context.close()


def _start_evidence(
    observed: dict[str, object], source: str | None
) -> tuple[dict[str, str], _Records]:
    """Record the scenario, the evidence lists and the captured fixture identity.

    Parameters
    ----------
    observed : dict[str, object]
        Caller-owned evidence.
    source : str | None
        Source target whose native coverage is recorded, if any.

    Returns
    -------
    tuple[dict[str, str], _Records]
        Case texts by name and the evidence lists exposed in ``observed``.

    """
    evidence = _Records()
    fixture = (
        Path(__file__).resolve().parents[1] / "data/studio/method_comparison_cases.json"
    ).read_bytes()
    pack = json.loads(fixture)
    observed.update(
        scenario="method_comparison_instrument",
        playwright=version("playwright"),
        records=evidence.records,
        page_errors=evidence.errors,
        external_requests=evidence.external,
        submission_requests=evidence.submissions,
        wasm_requests=evidence.kernels,
        observations=evidence.observations,
        file_transport_checks=evidence.transports,
        source_capture_sha256=hashlib.sha256(fixture).hexdigest(),
        interval_protocol=pack["interval_protocol"],
    )
    if source is not None:
        observed.update(
            source_url=source,
            native_v8_coverage=evidence.native,
            coverage_percentage="not_calculated",
        )
    return cast(dict[str, str], pack["cases"]), evidence


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
    targets = _owned_targets(base_url, source_url)
    from playwright.sync_api import sync_playwright

    observed = {} if evidence is None else evidence
    cases, lists = _start_evidence(observed, targets[1][1] if len(targets) == 2 else None)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        observed["browser"] = browser.version
        try:
            for target in targets:
                _compare_on(browser, target, cases, lists, observed)
            assert lists.kernels, "Actual original WASM was not requested"
            return observed
        finally:
            browser.close()
