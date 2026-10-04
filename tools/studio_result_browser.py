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
from importlib.metadata import version
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from xml.etree import ElementTree

from tools.studio_owned_worker_journey import _OBSERVE_WORKERS

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page


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
    page
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
    page
        Owned native page with the real result controls mounted.
    inspector
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


def run_result_journey(
    base_url: str, *, source_url: str | None = None, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Observe native result values, source linkage, raw exports and import refusal.

    Parameters
    ----------
    base_url
        Owned literal loopback preview of the actual production build.
    source_url
        Optional distinct owned root Vite host for original-source native counters.
    evidence
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
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    source = None if source_url is None else loopback_url(source_url)
    if source is not None and (
        urlsplit(source).path != "/" or urlsplit(source).netloc == urlsplit(url).netloc
    ):
        raise ValueError("Result source coverage requires a distinct owned root source host")
    from playwright.sync_api import Error, Route, expect, sync_playwright

    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    receipt = evidence if evidence is not None else {}
    observations: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    receipt.update(scenario="result_value_inspector", observations=observations)
    if source is not None:
        receipt.update(
            source_url=source, native_v8_coverage=records, coverage_percentage="not_calculated"
        )
    original = analyse_export()
    source_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
    receipt["actual_original_cli_source_sha256"] = source_sha
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        receipt.update(playwright=version("playwright"), browser=browser.version)
        try:
            for target in (url,) if source is None else (url, source):
                context = browser.new_context(service_workers="block", accept_downloads=True)
                context.set_default_timeout(15_000)
                context.add_init_script(_OBSERVE_WORKERS)
                rejected: list[str] = []
                errors: list[str] = []
                origin = urlsplit(target).netloc
                receipt.update(
                    current_owned_context=target,
                    current_page_errors=errors,
                    current_refused_requests=rejected,
                )

                def bound_request(
                    route: Route, *, authority: str = origin, refused: list[str] = rejected
                ) -> None:
                    """Retain and refuse real external transport attempts.

                    Parameters
                    ----------
                    route
                        Actual native intercepted request.
                    authority
                        This context's owned loopback authority.
                    refused
                        Original destination for this context's actual refused requests.

                    """
                    requested = urlsplit(route.request.url)
                    if requested.scheme != "http" or requested.netloc != authority:
                        refused.append(route.request.url)
                        route.abort()
                    else:
                        route.continue_()

                def record_error(error: Error, *, observed: list[str] = errors) -> None:
                    """Preserve actual uncaught host failures.

                    Parameters
                    ----------
                    error
                        Native uncaught browser exception.
                    observed
                        Original destination for this context's native page errors.

                    """
                    observed.append(str(error))

                try:
                    context.route("**/*", bound_request)
                    page = context.new_page()
                    profiler = start_native_coverage(page) if target == source else None
                    page.on("pageerror", record_error)
                    page.goto(target + "#/experiments", wait_until="networkidle")
                    experiment = page.get_by_label("Local experiment", exact=True)
                    experiment.get_by_role("button", name="Open Kuramoto sample").click()
                    experiment.get_by_role("button", name="Validate experiment archive").click()
                    expect(
                        experiment.get_by_role("button", name="Prepare numerical plan")
                    ).to_be_enabled()
                    experiment.get_by_role("button", name="Prepare numerical plan").click()
                    expect(
                        experiment.get_by_role("button", name="Run experiment", exact=True)
                    ).to_be_enabled()
                    experiment.get_by_role("button", name="Run experiment", exact=True).click()
                    expect(
                        experiment.get_by_text("Experiment succeeded", exact=True)
                    ).to_be_visible()
                    page.wait_for_function("window.__ownedKernel.active === 0")
                    native = page.evaluate("window.__ownedKernel")
                    event = next(
                        row for row in reversed(native["events"]) if row["kind"] == "result"
                    )
                    experiment.get_by_role("button", name="Save experiment attempt").click()
                    expect(
                        experiment.get_by_role("status", name="Experiment operation")
                    ).to_contain_text("Exact experiment attempt committed")
                    saved = _saved_archive(page)
                    saved_output = next(
                        row
                        for row in json.loads(saved)["members"]
                        if row["schema"] == "studio.kuramoto-output.v1"
                    )
                    page.get_by_role("navigation", name="Workbench views").get_by_role(
                        "link", name="Results", exact=True
                    ).click()
                    loader = page.get_by_role("region", name="Source result inspector", exact=True)
                    inspector = loader.get_by_role("region", name="Result value inspector").first
                    expect(
                        inspector.get_by_role("heading", name="Original classical Kuramoto run")
                    ).to_be_visible()
                    rows, metadata = _exports(page, inspector)
                    order = [row for row in rows if row["panel_id"] == "order"]
                    phases = [row for row in rows if row["panel_id"] == "final-phases"]
                    assert [float(row["value"]) for row in order] == event["payload"][
                        "orderParameter"
                    ]
                    assert [float(row["value"]) for row in phases] == event["payload"][
                        "thetaFinal"
                    ]
                    assert metadata["sourceSha256"] == saved_output["sha256"]
                    assert all(row["interval_method"] == "Not estimated" for row in rows)
                    assert "kernel does not report measured timestamps" in str(metadata["caption"])
                    chart = inspector.get_by_role("img", name="Original order parameter chart")
                    assert chart.locator("[data-index]").count() <= 1000
                    final = chart.locator("[data-index]").last
                    coordinate = final.get_attribute("data-coordinate")
                    final.click()
                    expect(
                        inspector.get_by_role("status", name="Result selection")
                    ).to_contain_text(str(coordinate))
                    expect(
                        inspector.get_by_role("img", name="Original final phases chart").locator(
                            "[data-index]"
                        )
                    ).to_have_count(len(phases))
                    observations.append(
                        {
                            "outcome": "actual-native-raw-values-and-linked-final-coordinate",
                            "source_sha256": metadata["sourceSha256"],
                            "raw_rows": len(rows),
                        }
                    )
                    loader.get_by_label("Result producer JSON").fill(original)
                    loader.get_by_role("button", name="Inspect producer result").click()
                    imported = loader.get_by_role("region", name="Result value inspector").last
                    expect(
                        imported.get_by_role("heading", name="Phase-cloud synchronisation witness")
                    ).to_be_visible()
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
                    markers = imported.get_by_role("img", name="Betti H0 chart").locator(
                        "[data-index]"
                    )
                    assert [markers.nth(i).get_attribute("data-coordinate") for i in range(3)] == [
                        "0",
                        "0.125",
                        "2",
                    ]
                    x = [float(markers.nth(i).get_attribute("cx") or "nan") for i in range(3)]
                    assert x[1] - x[0] < x[2] - x[1]
                    imported.get_by_role("button", name="Select Betti H0 sample 0.125 H0").click()
                    expect(
                        imported.get_by_role("status", name="Result selection")
                    ).to_contain_text("0.125 rad")
                    observations.append(
                        {
                            "outcome": "actual-cli-nonuniform-coordinates-raw-int64-and-absent-intervals",
                            "source_sha256": source_sha,
                        }
                    )
                    for invalid in (
                        "{",
                        original.replace(
                            '"analysis_schema": "studio.sync-analysis.v1"',
                            '"analysis_schema": "studio.sync-analysis.v2"',
                        ),
                    ):
                        loader.get_by_label("Result producer JSON").fill(invalid)
                        loader.get_by_role("button", name="Inspect producer result").click()
                        expect(
                            loader.get_by_role("status", name="Result import status")
                        ).not_to_have_text("")
                        expect(imported.get_by_text(source_sha, exact=True)).to_be_visible()
                    assert _saved_archive(page) == saved, (
                        "Result controls changed the saved source"
                    )
                    assert page.evaluate("window.__ownedKernel.started") == native["started"], (
                        "Result-only actions allocated a new numerical worker"
                    )
                    assert not page.workers
                    observations.append(
                        {
                            "outcome": "refused-import-and-complete-exports-retain-saved-source-and-worker-count",
                            "saved_sha256": hashlib.sha256(saved.encode()).hexdigest(),
                        }
                    )
                    assert not errors, errors
                    assert not rejected, rejected
                    if profiler is not None:
                        take_native_coverage(profiler, records, include_results=True)
                except (Error, AssertionError):
                    receipt["failed_native_workers"] = page.evaluate("window.__ownedKernel")
                    receipt["failed_page_url"] = page.url
                    receipt["failed_page_labels"] = page.locator("label").all_text_contents()
                    receipt["failed_page_textareas"] = page.locator("textarea").evaluate_all(
                        "elements => elements.map(element => ({id: element.id, "
                        "label: element.labels?.[0]?.textContent, value: element.value}))"
                    )
                    raise
                finally:
                    context.close()
            return receipt
        finally:
            browser.close()
