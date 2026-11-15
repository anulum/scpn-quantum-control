# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original result inspector browser acceptance
"""Inspect real original CLI and disposed WASM results through public controls."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit
from xml.etree import ElementTree

from tools.studio_owned_worker_journey import _OBSERVE_WORKERS

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Locator, Page, Route


def analyse_export() -> str:
    """Produce an actual sealed CLI export with an independent stationary-cloud oracle.

    Returns
    -------
    str
        Original executive export containing nonuniform filtration thresholds.

    Raises
    ------
    AssertionError
        The real CLI fails or violates the independent constant-phase oracle.

    """
    from scpn_quantum_control.studio.executive_cli import run

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = run(
            [
                "analyse",
                "--action-id",
                "original-result-inspector-oracle",
                "--params",
                json.dumps(
                    {
                        "phases": [0.0, 0.0, 0.0, 0.0],
                        "thresholds": [0.0, 0.125, 2.0],
                        "reference_scale": 0.125,
                    }
                ),
            ]
        )
    assert code == 0
    result = json.loads(stream.getvalue())["result"]
    assert result["status"] == "succeeded"
    assert result["outputs"]["betti0_curve"] == [1, 1, 1]
    assert result["outputs"]["betti1_curve"] == [0, 0, 0]
    return stream.getvalue()


def _saved_archive(page: Page) -> str:
    """Read the actual saved draft through the continuously mounted Workspace editor.

    Parameters
    ----------
    page : Page
        Owned actual Workbench page whose workspace controller remains mounted.

    Returns
    -------
    str
        Original current portable archive text without executing or editing it.

    """
    # React can include the textarea's initial JSON in its wrapping label text.
    # Reading that labelled original control does not need to change the active view.
    return page.get_by_label("Workspace archive JSON").input_value()


def _exports(page: Page, inspector: Locator) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Read actual browser downloads and their complete raw source metadata.

    Parameters
    ----------
    page : Page
        Owned native page with the real result controls mounted.
    inspector : Locator
        Admitted original result inspector, independent of another source result.

    Returns
    -------
    tuple
        Raw CSV source rows and full lossless SVG metadata.

    """
    with page.expect_download() as pending:
        inspector.get_by_role("button", name="Export raw CSV", exact=True).click()
    csv_text = pending.value.path().read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(csv_text)))
    with page.expect_download() as pending:
        inspector.get_by_role("button", name="Export SVG", exact=True).click()
    svg = ElementTree.fromstring(pending.value.path().read_text(encoding="utf-8"))
    metadata = svg.find("{http://www.w3.org/2000/svg}metadata")
    assert metadata is not None and metadata.text is not None
    original = json.loads(metadata.text)
    assert rows[0] == ["source_sha256", original["sourceSha256"]]
    assert rows[1] == ["caption", original["caption"]]
    assert rows[2] == ["claim_boundary", original["claimBoundary"]]
    assert original["caption"] in "".join(svg.itertext())
    return [dict(zip(rows[4], row, strict=True)) for row in rows[5:]], original


def _owned_hosts(base_url: str, source_url: str | None) -> tuple[str, ...]:
    """Validate the built host and the optional distinct root source host.

    Parameters
    ----------
    base_url : str
        Owned literal loopback preview of the actual production build.
    source_url : str | None
        Optional distinct owned root Vite host for original-source native counters.

    Returns
    -------
    tuple[str, ...]
        The built host, then the source host when one is supplied.

    Raises
    ------
    ValueError
        A supplied URL is external, ambiguous, shared or nested.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    source = None if source_url is None else loopback_url(source_url)
    if source is None:
        return (url,)
    if urlsplit(source).path != "/" or urlsplit(source).netloc == urlsplit(url).netloc:
        raise ValueError("Result source coverage requires a distinct owned root source host")
    return (url, source)


def _run_sample(page: Page, target: str) -> tuple[dict[str, object], str, dict[str, object]]:
    """Run and save the Kuramoto sample and return its native record and saved output.

    Parameters
    ----------
    page : Page
        Owned page bound to one host.
    target : str
        Owned host URL.

    Returns
    -------
    tuple
        The owned kernel record, the saved archive and its saved Kuramoto output member.

    """
    from playwright.sync_api import expect

    page.goto(target + "#/experiments", wait_until="networkidle")
    experiment = page.get_by_label("Local experiment", exact=True)
    experiment.get_by_role("button", name="Open Kuramoto sample").click()
    experiment.get_by_role("button", name="Validate experiment archive").click()
    prepare = experiment.get_by_role("button", name="Prepare numerical plan")
    expect(prepare).to_be_enabled()
    prepare.click()
    run = experiment.get_by_role("button", name="Run experiment", exact=True)
    expect(run).to_be_enabled()
    run.click()
    expect(experiment.get_by_text("Experiment succeeded", exact=True)).to_be_visible()
    page.wait_for_function("window.__ownedKernel.active === 0")
    native = page.evaluate("window.__ownedKernel")
    experiment.get_by_role("button", name="Save experiment attempt").click()
    expect(experiment.get_by_role("status", name="Experiment operation")).to_contain_text(
        "Exact experiment attempt committed"
    )
    saved = _saved_archive(page)
    saved_output = next(
        row for row in json.loads(saved)["members"] if row["schema"] == "studio.kuramoto-output.v1"
    )
    return native, saved, saved_output


def _check_native_values(
    page: Page, native: dict[str, object], saved_output: dict[str, object]
) -> tuple[Locator, dict[str, object]]:
    """Inspect the saved native run and link its final coordinate to the selection.

    Parameters
    ----------
    page : Page
        Owned page after the saved run.
    native : dict[str, object]
        Owned kernel record carrying the run's result event.
    saved_output : dict[str, object]
        Saved Kuramoto output member.

    Returns
    -------
    tuple
        The source result loader and the observation.

    """
    from playwright.sync_api import expect

    events = cast(list[dict[str, Any]], native["events"])
    event = next(row for row in reversed(events) if row["kind"] == "result")
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name="Analyse", exact=True
    ).click()
    loader = page.get_by_role("region", name="Source result inspector", exact=True)
    inspector = loader.get_by_role("region", name="Result value inspector").first
    heading = inspector.get_by_role("heading", name="Original classical Kuramoto run")
    expect(heading).to_be_visible()
    rows, metadata = _exports(page, inspector)
    order = [row for row in rows if row["panel_id"] == "order"]
    phases = [row for row in rows if row["panel_id"] == "final-phases"]
    assert [float(row["value"]) for row in order] == event["payload"]["orderParameter"]
    assert [float(row["value"]) for row in phases] == event["payload"]["thetaFinal"]
    assert metadata["sourceSha256"] == saved_output["sha256"]
    assert all(row["interval_method"] == "Not estimated" for row in rows)
    assert "kernel does not report measured timestamps" in str(metadata["caption"])
    _check_native_selection(inspector, len(phases))
    return loader, {
        "outcome": "actual-native-raw-values-and-linked-final-coordinate",
        "source_sha256": metadata["sourceSha256"],
        "raw_rows": len(rows),
    }


def _check_native_selection(inspector: Locator, phases: int) -> None:
    """Link the last order-parameter sample to the selection and count the phase markers.

    Parameters
    ----------
    inspector : Locator
        Result value inspector of the saved native run.
    phases : int
        Number of exported final phases.

    """
    from playwright.sync_api import expect

    chart = inspector.get_by_role("img", name="Original order parameter chart")
    assert chart.locator("[data-index]").count() <= 1000
    final = chart.locator("[data-index]").last
    coordinate = final.get_attribute("data-coordinate")
    final.click()
    selection = inspector.get_by_role("status", name="Result selection")
    expect(selection).to_contain_text(str(coordinate))
    phase_chart = inspector.get_by_role("img", name="Original final phases chart")
    expect(phase_chart.locator("[data-index]")).to_have_count(phases)


def _check_cli_values(
    page: Page, loader: Locator, original: str, source_sha: str
) -> tuple[Locator, dict[str, object]]:
    """Import the CLI export and check its nonuniform coordinates and raw integer values.

    Parameters
    ----------
    page : Page
        Owned page on the Analyse route.
    loader : Locator
        Source result loader.
    original : str
        Actual sealed CLI export.
    source_sha : str
        SHA-256 of the CLI export.

    Returns
    -------
    tuple
        The imported result inspector and the observation.

    """
    from playwright.sync_api import expect

    loader.get_by_label("Result producer JSON").fill(original)
    loader.get_by_role("button", name="Inspect producer result").click()
    imported = loader.get_by_role("region", name="Result value inspector").last
    heading = imported.get_by_role("heading", name="Phase-cloud synchronisation witness")
    expect(heading).to_be_visible()
    rows, metadata = _exports(page, imported)
    assert metadata["sourceSha256"] == source_sha
    h0 = [row for row in rows if row["panel_id"] == "betti0"]
    assert [float(row["coordinate"]) for row in h0] == [0.0, 0.125, 2.0]
    assert [int(row["value"]) for row in h0] == [1, 1, 1]
    assert all(
        row["coordinate_unit"] == "rad"
        and row["value_dtype"] == "int64"
        and row["interval_method"] == "Not estimated"
        for row in rows
    )
    _check_betti_markers(imported)
    return imported, {
        "outcome": "actual-cli-nonuniform-coordinates-raw-int64-and-absent-intervals",
        "source_sha256": source_sha,
    }


def _check_betti_markers(imported: Locator) -> None:
    """Place the H0 markers at their nonuniform thresholds and select the middle one.

    Parameters
    ----------
    imported : Locator
        Result value inspector of the imported CLI export.

    """
    from playwright.sync_api import expect

    markers = imported.get_by_role("img", name="Betti H0 chart").locator("[data-index]")
    coordinates = [markers.nth(i).get_attribute("data-coordinate") for i in range(3)]
    assert coordinates == ["0", "0.125", "2"]
    x = [float(markers.nth(i).get_attribute("cx") or "nan") for i in range(3)]
    assert x[1] - x[0] < x[2] - x[1]
    imported.get_by_role("button", name="Select Betti H0 sample 0.125 H0").click()
    selection = imported.get_by_role("status", name="Result selection")
    expect(selection).to_contain_text("0.125 rad")


def _check_refusals(
    page: Page,
    loaders: tuple[Locator, Locator],
    original: str,
    saved: str,
    native: dict[str, object],
) -> dict[str, object]:
    """Refuse malformed and unsupported imports while keeping source, save and workers.

    Parameters
    ----------
    page : Page
        Owned page on the Analyse route.
    loaders : tuple[Locator, Locator]
        Source result loader and the imported result inspector.
    original : str
        Actual sealed CLI export.
    saved : str
        Saved archive, which must stay unchanged.
    native : dict[str, object]
        Owned kernel record taken after the run.

    Returns
    -------
    dict[str, object]
        Observation with the saved archive digest.

    """
    from playwright.sync_api import expect

    loader, imported = loaders
    source_sha = hashlib.sha256(original.encode()).hexdigest()
    unsupported = original.replace(
        '"analysis_schema": "studio.sync-analysis.v1"',
        '"analysis_schema": "studio.sync-analysis.v2"',
    )
    for invalid in ("{", unsupported):
        loader.get_by_label("Result producer JSON").fill(invalid)
        loader.get_by_role("button", name="Inspect producer result").click()
        expect(loader.get_by_role("status", name="Result import status")).not_to_have_text("")
        expect(imported.get_by_text(source_sha, exact=True)).to_be_visible()
    assert _saved_archive(page) == saved, "Result controls changed the saved source"
    assert page.evaluate("window.__ownedKernel.started") == native["started"], (
        "Result-only actions allocated a new numerical worker"
    )
    assert not page.workers
    return {
        "outcome": "refused-import-and-complete-exports-retain-saved-source-and-worker-count",
        "saved_sha256": hashlib.sha256(saved.encode()).hexdigest(),
    }


def _inspect(
    page: Page, target: str, original: str, observations: list[dict[str, object]]
) -> None:
    """Run the sample, inspect native and CLI values and refuse unsupported imports.

    Parameters
    ----------
    page : Page
        Owned page bound to one host.
    target : str
        Owned host URL.
    original : str
        Actual sealed CLI export.
    observations : list[dict[str, object]]
        Caller-owned list that records each outcome once it holds.

    """
    source_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
    native, saved, saved_output = _run_sample(page, target)
    loader, observation = _check_native_values(page, native, saved_output)
    observations.append(observation)
    imported, observation = _check_cli_values(page, loader, original, source_sha)
    observations.append(observation)
    observations.append(_check_refusals(page, (loader, imported), original, saved, native))


def _record_failure(
    receipt: dict[str, object], page: Page | None, target: str, cause: BaseException
) -> None:
    """Keep the failure, its traceback and what the page still shows.

    Parameters
    ----------
    receipt : dict[str, object]
        Caller-owned evidence receiving the diagnostics.
    page : Page | None
        Owned page, or None when it was never opened.
    target : str
        Owned host URL of the failed context.
    cause : BaseException
        The browser or assertion failure.

    """
    from traceback import format_exception

    receipt["original_error"] = f"{type(cause).__name__}: {cause}"
    receipt["original_traceback"] = "".join(format_exception(cause))
    receipt["failed_context"] = target
    try:
        assert page is not None, "page was never opened"
        receipt["failed_native_workers"] = page.evaluate("window.__ownedKernel")
        receipt["failed_page_url"] = page.url
        receipt["failed_page_labels"] = page.locator("label").all_text_contents()
        receipt["failed_page_textareas"] = page.locator("textarea").evaluate_all(
            "elements => elements.map(element => ({id: element.id, "
            "label: element.labels?.[0]?.textContent, value: element.value}))"
        )
    except Exception as capture_error:
        receipt["diagnostic_capture_error"] = f"{type(capture_error).__name__}: {capture_error}"


def _owned_page(
    context: BrowserContext, target: str, refused: list[str], errors: list[str]
) -> Page:
    """Restrict a context to its exact owned host and open a page.

    Parameters
    ----------
    context : BrowserContext
        Fresh browser context for one host.
    target : str
        Owned host URL.
    refused : list[str]
        Caller-owned record of refused request URLs.
    errors : list[str]
        Caller-owned record of uncaught page failures, attached by the caller.

    Returns
    -------
    Page
        The owned page.

    """
    authority = urlsplit(target).netloc

    def bound_request(route: Route) -> None:
        requested = urlsplit(route.request.url)
        if requested.scheme != "http" or requested.netloc != authority:
            refused.append(route.request.url)
            route.abort()
        else:
            route.continue_()

    context.route("**/*", bound_request)
    return context.new_page()


def _owned_context(browser: Browser) -> BrowserContext:
    """Open a download-accepting context that observes the owned kernel workers.

    Parameters
    ----------
    browser : Browser
        Launched browser.

    Returns
    -------
    BrowserContext
        The fresh context.

    """
    context = browser.new_context(service_workers="block", accept_downloads=True)
    context.set_default_timeout(15_000)
    context.add_init_script(_OBSERVE_WORKERS)
    return context


def _result_context(
    browser: Browser,
    target: str,
    source: str | None,
    receipt: dict[str, object],
    collected: tuple[str, list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """Run the whole result journey on one owned host in its own context.

    Parameters
    ----------
    browser : Browser
        Launched browser.
    target : str
        Owned host URL.
    source : str | None
        Source host whose native coverage is recorded, if any.
    receipt : dict[str, object]
        Caller-owned evidence, also receiving this context's records and failures.
    collected : tuple
        The actual CLI export, the observations list and the native coverage records.

    """
    from playwright.sync_api import Error

    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    original, observations, records = collected
    context = _owned_context(browser)
    rejected: list[str] = []
    errors: list[str] = []
    receipt.update(
        current_owned_context=target, current_page_errors=errors, current_refused_requests=rejected
    )
    page: Page | None = None
    try:
        page = _owned_page(context, target, rejected, errors)
        profiler = start_native_coverage(page) if target == source else None
        page.on("pageerror", lambda error: errors.append(str(error)))
        _inspect(page, target, original, observations)
        assert not errors, errors
        assert not rejected, rejected
        if profiler is not None:
            take_native_coverage(profiler, records, include_results=True)
    except (Error, AssertionError) as cause:
        _record_failure(receipt, page, target, cause)
        raise
    finally:
        context.close()


def _start_receipt(
    receipt: dict[str, object], source: str | None
) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    """Record the scenario and the actual CLI export identity before any browser starts.

    Parameters
    ----------
    receipt : dict[str, object]
        Caller-owned evidence.
    source : str | None
        Source host whose native coverage is recorded, if any.

    Returns
    -------
    tuple
        The actual CLI export, the observations list and the native coverage records.

    """
    observations: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    receipt.update(scenario="result_value_inspector", observations=observations)
    if source is not None:
        receipt.update(
            source_url=source, native_v8_coverage=records, coverage_percentage="not_calculated"
        )
    original = analyse_export()
    receipt["actual_original_cli_source_sha256"] = hashlib.sha256(original.encode()).hexdigest()
    return original, observations, records


def run_result_journey(
    base_url: str, *, source_url: str | None = None, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Observe native result values, source linkage, raw exports and import refusal.

    Parameters
    ----------
    base_url : str
        Owned literal loopback preview of the actual production build.
    source_url : str | None
        Optional distinct owned root Vite host for original-source native counters.
    evidence : dict[str, object] | None
        Original shared-dispatcher destination retaining completed observations.

    Returns
    -------
    dict[str, object]
        Actual runtime versions, raw producer identities, observed values and counters.

    Raises
    ------
    ValueError
        A supplied URL is external, ambiguous, shared or nested.
    AssertionError
        Original values, saved state, worker disposal or runtime boundaries fail.

    """
    hosts = _owned_hosts(base_url, source_url)
    source = hosts[1] if len(hosts) == 2 else None
    from playwright.sync_api import sync_playwright

    receipt = evidence if evidence is not None else {}
    collected = _start_receipt(receipt, source)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        receipt.update(playwright=version("playwright"), browser=browser.version)
        try:
            for target in hosts:
                _result_context(browser, target, source, receipt, collected)
            return receipt
        finally:
            browser.close()
