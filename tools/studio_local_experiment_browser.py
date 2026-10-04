# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original local experiment browser acceptance
"""Observe real experiment archives, original workers and clean-context replay."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from tools.studio_owned_worker_journey import _OBSERVE_WORKERS

if TYPE_CHECKING:
    from playwright.sync_api import Page


def _plan_identity(page: Page) -> dict[str, str]:
    """Read the actual rendered source and plan, without inventing a request.

    Parameters
    ----------
    page
        Actual original Workbench page.

    Returns
    -------
    dict[str, str]
        Complete visible numerical plan declaration.

    """
    declaration = page.get_by_label("Numerical plan", exact=True)
    keys = declaration.locator("dt").all_text_contents()
    values = declaration.locator("dd").all_text_contents()
    assert len(keys) == len(values) and len(keys) >= 15
    return dict(zip(keys, values, strict=True))


def _archive(page: Page) -> str:
    """Read the original Workspace editor's current exact portable JSON.

    Parameters
    ----------
    page
        Page whose original workspace view is currently open.

    Returns
    -------
    str
        Current original draft bytes decoded as UTF-8.

    """
    return page.get_by_label("Workspace archive JSON").input_value()


def _navigate(page: Page, view: str) -> None:
    """Navigate through the original link without importing or executing a source.

    Parameters
    ----------
    page
        Actual Workbench browser page.
    view
        Original Workspace or Experiments view label.

    """
    from playwright.sync_api import expect

    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name=view, exact=True
    ).click()
    expect(
        page.get_by_role(
            "heading",
            name="Local workspace" if view == "Workspace" else "Local experiment",
            exact=True,
        )
    ).to_be_visible()


def _disposed(page: Page) -> None:
    """Require actual native disposal and absence of a leaked browser worker.

    Parameters
    ----------
    page
        Actual page with the original native Worker observer installed.

    """
    page.wait_for_function("window.__ownedKernel.active === 0")
    assert not page.workers


def _execute(page: Page) -> dict[str, object]:
    """Explicitly run the original worker and retain its genuine native event.

    Parameters
    ----------
    page
        Current Workbench page with an admitted source plan.

    Returns
    -------
    dict[str, object]
        Original native result envelope, never a fabricated computation.

    """
    from playwright.sync_api import expect

    experiment = page.get_by_label("Local experiment", exact=True)
    experiment.get_by_role("button", name="Run experiment", exact=True).click()
    expect(
        experiment.get_by_role("status").filter(has_text="Experiment succeeded")
    ).to_be_visible()
    _disposed(page)
    trace = page.evaluate("window.__ownedKernel")
    return cast(
        "dict[str, object]",
        next(row for row in reversed(trace["events"]) if row["kind"] == "result"),
    )


def run_local_experiment_journey(
    base_url: str, *, source_url: str | None = None, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Exercise sample, edit, plan, genuine execution, save, import and replay.

    Parameters
    ----------
    base_url
        Owned literal loopback preview of the current production build.
    source_url
        Optional distinct owned root Vite host for original-source native counters.
    evidence
        Optional original dispatcher receipt retaining partial observations.

    Returns
    -------
    dict[str, object]
        Observed versions, identities, native envelopes and refusal boundaries.

    Raises
    ------
    ValueError
        The supplied preview is not an owned plain HTTP loopback URL.
    AssertionError
        An original identity, lifecycle, save or replay contract fails.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    source = None if source_url is None else loopback_url(source_url)
    if source is not None and (
        urlsplit(source).path != "/" or urlsplit(source).netloc == urlsplit(url).netloc
    ):
        raise ValueError("Experiment source coverage requires a distinct owned root source host")
    from playwright.sync_api import Error, Route, expect, sync_playwright

    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    receipt = evidence if evidence is not None else {}
    observations: list[dict[str, object]] = []
    receipt.update(scenario="local_experiment_journey", observations=observations)
    records: list[dict[str, object]] = []
    if source is not None:
        receipt.update(
            source_url=source, native_v8_coverage=records, coverage_percentage="not_calculated"
        )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        receipt.update(playwright=version("playwright"), browser=browser.version)
        try:
            exported = ""
            original_plan: dict[str, str] = {}
            original_result: dict[str, object] = {}
            boundaries = ("original", "fresh-replay", "missing-kernel") + (
                ()
                if source is None
                else (
                    "source-original",
                    "source-fresh-replay",
                    "source-invalid-sample",
                    "source-unmounted-sample",
                )
            )
            for boundary in boundaries:
                context = browser.new_context(service_workers="block")
                context.set_default_timeout(15_000)
                context.add_init_script(_OBSERVE_WORKERS)
                held: list[Route] = []
                hold_next = False
                fail_next = False
                hold_sample_kernel = False
                errors: list[str] = []
                rejected: list[str] = []
                target_url = source if boundary.startswith("source-") else url
                assert target_url is not None
                origin = urlsplit(target_url).netloc

                damaged_sample_requests: list[str] = []

                def route_request(
                    route: Route,
                    *,
                    unavailable: bool = boundary == "missing-kernel",
                    invalid_sample: bool = boundary == "source-invalid-sample",
                    damaged: list[str] = damaged_sample_requests,
                    refused: list[str] = rejected,
                    waiting: list[Route] = held,
                    authority: str = origin,
                ) -> None:
                    """Restrict requests and hold or refuse the real worker transport."""
                    nonlocal hold_next, fail_next, hold_sample_kernel
                    target = urlsplit(route.request.url)
                    if target.scheme != "http" or target.netloc != authority:
                        refused.append(route.request.url)
                        route.abort()
                    elif hold_sample_kernel and target.path.endswith(
                        "scpn_quantum_studio_wasm_kernel.wasm"
                    ):
                        hold_sample_kernel = False
                        waiting.append(route)
                    elif unavailable and target.path.endswith(".wasm"):
                        route.abort()
                    elif invalid_sample and target.path.endswith(
                        "kuramoto_scenario_meanfield_20260708.json"
                    ):
                        damaged.append(route.request.url)
                        route.fulfill(
                            status=200,
                            content_type="application/javascript",
                            body="export default {};\n",
                        )
                    elif (
                        Path(target.path).name.startswith("kernelWorker-")
                        or Path(target.path).name == "kernelWorker.ts"
                    ):
                        if hold_next:
                            hold_next = False
                            waiting.append(route)
                        elif fail_next:
                            fail_next = False
                            route.abort()
                        else:
                            route.continue_()
                    else:
                        route.continue_()

                try:
                    context.route("**/*", route_request)
                    page = context.new_page()
                    profiler = (
                        start_native_coverage(page) if boundary.startswith("source-") else None
                    )

                    def record_error(error: Error, *, observed: list[str] = errors) -> None:
                        """Retain native page failures in this context's exact receipt."""
                        observed.append(str(error))

                    page.on("pageerror", record_error)
                    page.goto(target_url + "#/experiments", wait_until="networkidle")
                    experiment = page.get_by_label("Local experiment", exact=True)
                    expect(
                        experiment.get_by_role("heading", name="Local experiment")
                    ).to_be_visible()
                    assert page.evaluate("window.__ownedKernel.started") == 0

                    if boundary == "source-unmounted-sample":
                        original_draft = _archive(page)
                        hold_sample_kernel = True
                        with page.expect_request(
                            lambda request: urlsplit(request.url).path.endswith(
                                "scpn_quantum_studio_wasm_kernel.wasm"
                            )
                        ):
                            experiment.get_by_role("button", name="Open Kuramoto sample").click()
                        page.get_by_role("navigation", name="Workbench views").get_by_role(
                            "link", name="Atlas", exact=True
                        ).click()
                        expect(
                            page.get_by_role("heading", name="Atlas unavailable", exact=True)
                        ).to_be_visible()
                        assert len(held) == 1
                        for route in held:
                            route.continue_()
                        held.clear()
                        page.wait_for_load_state("networkidle")
                        current_draft = _archive(page)
                        receipt["late_sample_source"] = {
                            "before_sha256": hashlib.sha256(original_draft.encode()).hexdigest(),
                            "after_sha256": hashlib.sha256(current_draft.encode()).hexdigest(),
                            "before_characters": len(original_draft),
                            "after_characters": len(current_draft),
                        }
                        assert current_draft == original_draft, receipt["late_sample_source"]
                        assert page.evaluate("window.__ownedKernel.started") == 0
                        expect(
                            page.get_by_label("Current experiment result", exact=True)
                        ).to_have_count(0)
                        observations.append(
                            {"outcome": "late-native-sample-load-cannot-overwrite-unmounted-view"}
                        )
                    elif boundary == "source-invalid-sample":
                        experiment.get_by_role("button", name="Open Kuramoto sample").click()
                        expect(
                            experiment.get_by_role("status", name="Experiment operation")
                        ).to_have_text("original committed sample is unavailable")
                        expect(
                            experiment.get_by_label("Numerical plan", exact=True)
                        ).to_have_count(0)
                        expect(
                            experiment.get_by_label("Current experiment result", exact=True)
                        ).to_have_count(0)
                        _navigate(page, "Workspace")
                        assert _archive(page) == ""
                        assert page.evaluate("window.__ownedKernel.started") == 0
                        assert len(damaged_sample_requests) == 1
                        observations.append(
                            {
                                "outcome": "invalid-original-sample-refuses-without-draft-or-worker",
                                "actual_interrupted_source": damaged_sample_requests[0],
                            }
                        )
                    elif boundary == "missing-kernel":
                        experiment.get_by_role("button", name="Open Kuramoto sample").click()
                        expect(
                            experiment.get_by_role("status", name="Experiment operation")
                        ).to_have_text(
                            "Local experiment operation refused; current draft and saved archive retained"
                        )
                        expect(
                            experiment.get_by_label("Numerical plan", exact=True)
                        ).to_have_count(0)
                        expect(
                            experiment.get_by_label("Current experiment result", exact=True)
                        ).to_have_count(0)
                        _navigate(page, "Workspace")
                        assert _archive(page) == ""
                        assert page.evaluate("window.__ownedKernel.started") == 0
                        observations.append(
                            {"outcome": "missing-native-kernel-refuses-without-draft-or-worker"}
                        )
                    elif boundary.endswith("fresh-replay"):
                        _navigate(page, "Workspace")
                        assert _archive(page) == ""
                        page.get_by_label("Workspace archive file").set_input_files(
                            {
                                "name": "original-experiment.json",
                                "mimeType": "application/json",
                                "buffer": exported.encode("utf-8"),
                            }
                        )
                        page.get_by_role("button", name="Preview archive", exact=True).click()
                        expect(
                            page.get_by_role("button", name="Save draft and revision references")
                        ).to_be_enabled()
                        page.get_by_role(
                            "button", name="Save draft and revision references"
                        ).click()
                        expect(
                            page.get_by_label("Saved workspace identity", exact=True)
                        ).to_be_visible()
                        _navigate(page, "Experiments")
                        experiment.get_by_role("button", name="Prepare saved replay").click()
                        expect(
                            experiment.get_by_role("button", name="Run experiment", exact=True)
                        ).to_be_enabled()
                        replay_plan = _plan_identity(page)
                        assert replay_plan == original_plan
                        actual = _execute(page)
                        replay_payload = cast("dict[str, object]", actual["payload"])
                        original_payload = cast("dict[str, object]", original_result["payload"])
                        assert (
                            replay_payload["orderParameter"] == original_payload["orderParameter"]
                        )
                        assert replay_payload["thetaFinal"] == original_payload["thetaFinal"]
                        expect(
                            experiment.get_by_text(
                                "Experiment succeeded · Original float64 replay verified",
                                exact=True,
                            )
                        ).to_be_visible()
                        observations.append(
                            {
                                "outcome": "test_local_experiment_journey_01",
                                "source_plan": replay_plan,
                                "native_replay": actual,
                                "portable_sha256": hashlib.sha256(exported.encode()).hexdigest(),
                            }
                        )
                    else:
                        experiment.get_by_role("button", name="Open Kuramoto sample").click()
                        experiment.get_by_role(
                            "button", name="Validate experiment archive"
                        ).click()
                        expect(
                            experiment.get_by_role("button", name="Prepare numerical plan")
                        ).to_be_enabled()
                        _navigate(page, "Workspace")
                        sample_archive = _archive(page)
                        editor = page.get_by_label("Linked parameter editor", exact=True)
                        editor.get_by_role("button", name="coupling[scalar]", exact=False).click()
                        editor.get_by_label("Selected value").fill("1.4")
                        editor.get_by_role("button", name="Apply value", exact=True).click()
                        editor.get_by_role("button", name="Save parameter revision").click()
                        expect(
                            page.get_by_label("Saved workspace identity", exact=True)
                        ).to_be_visible()
                        edited = _archive(page)
                        assert edited != sample_archive
                        _navigate(page, "Experiments")
                        experiment.get_by_role("button", name="Prepare numerical plan").click()
                        expect(
                            experiment.get_by_role("button", name="Run experiment", exact=True)
                        ).to_be_enabled()
                        original_plan = _plan_identity(page)
                        original_result = _execute(page)
                        with page.expect_download() as portable_download:
                            experiment.get_by_role(
                                "button", name="Export experiment attempt"
                            ).click()
                        portable_attempt = portable_download.value.path().read_text(
                            encoding="utf-8"
                        )
                        assert any(
                            row["schema"] == "local_run_record.v1"
                            for row in json.loads(portable_attempt)["members"]
                        )
                        _navigate(page, "Workspace")
                        assert _archive(page) == edited, (
                            "Portable attempt export changed the original saved source"
                        )
                        _navigate(page, "Experiments")
                        experiment.get_by_role("button", name="Save experiment attempt").click()
                        expect(
                            experiment.get_by_role("status", name="Experiment operation")
                        ).to_contain_text("Exact experiment attempt committed")
                        expect(
                            experiment.get_by_label("Current experiment result", exact=True)
                        ).to_be_visible()
                        with page.expect_download() as download:
                            experiment.get_by_role(
                                "button", name="Export saved experiment"
                            ).click()
                        exported = download.value.path().read_text(encoding="utf-8")
                        _navigate(page, "Workspace")
                        assert _archive(page) == exported
                        saved = exported
                        _navigate(page, "Experiments")
                        before = page.evaluate("window.__ownedKernel.started")
                        experiment.get_by_label("Run numeric byte budget").fill("0")
                        expect(
                            experiment.get_by_role("button", name="Run experiment", exact=True)
                        ).to_be_disabled()
                        experiment.get_by_role("button", name="Prepare numerical plan").click()
                        expect(
                            experiment.get_by_role("status", name="Experiment lifecycle")
                        ).to_contain_text("refused")
                        assert page.evaluate("window.__ownedKernel.started") == before
                        _navigate(page, "Workspace")
                        assert _archive(page) == saved
                        _navigate(page, "Experiments")
                        observations.append(
                            {
                                "outcome": "test_local_experiment_journey_02",
                                "workers_started": before,
                            }
                        )
                        experiment.get_by_label("Run numeric byte budget").fill("")
                        experiment.get_by_role("button", name="Prepare numerical plan").click()
                        expect(
                            experiment.get_by_role("button", name="Run experiment", exact=True)
                        ).to_be_enabled()
                        hold_next = True
                        experiment.get_by_role("button", name="Run experiment", exact=True).click()
                        page.wait_for_function("window.__ownedKernel.active === 1")
                        experiment.get_by_role(
                            "button", name="Cancel experiment", exact=True
                        ).click()
                        expect(
                            experiment.get_by_role("status", name="Experiment lifecycle")
                        ).to_contain_text("cancelled")
                        _disposed(page)
                        expect(
                            experiment.get_by_label("Current experiment result", exact=True)
                        ).to_have_count(0)
                        expect(
                            experiment.get_by_label("Original attempt diagnostics", exact=True)
                        ).to_contain_text("cancelled")
                        _navigate(page, "Workspace")
                        assert _archive(page) == saved
                        _navigate(page, "Experiments")
                        observations.append(
                            {
                                "outcome": "test_local_experiment_journey_03",
                                "diagnostics": experiment.get_by_label(
                                    "Original attempt diagnostics", exact=True
                                ).inner_text(),
                            }
                        )
                        for route in held:
                            route.abort()
                        held.clear()
                        hold_next = True
                        experiment.get_by_role("button", name="Run experiment", exact=True).click()
                        page.wait_for_function("window.__ownedKernel.active === 1")
                        _navigate(page, "Workspace")
                        _disposed(page)
                        editor = page.get_by_label("Linked parameter editor", exact=True)
                        editor.get_by_role("button", name="coupling[scalar]", exact=False).click()
                        editor.get_by_label("Selected value").fill("1.6")
                        editor.get_by_role("button", name="Apply value", exact=True).click()
                        editor.get_by_role("button", name="Save parameter revision").click()
                        expect(
                            page.get_by_role("status").filter(
                                has_text="Parameter revision transaction committed"
                            )
                        ).to_be_visible()
                        expect(page.get_by_label("Workspace archive JSON")).not_to_have_value(
                            saved
                        )
                        newer = _archive(page)
                        assert newer != saved, (
                            "Actual second parameter transaction did not update the source archive"
                        )
                        for route in held:
                            route.continue_()
                        held.clear()
                        _navigate(page, "Experiments")
                        expect(
                            experiment.get_by_role("status", name="Experiment lifecycle")
                        ).to_contain_text("stale")
                        expect(
                            experiment.get_by_label("Current experiment result", exact=True)
                        ).to_have_count(0)
                        expect(
                            experiment.get_by_role("button", name="Save experiment attempt")
                        ).to_be_disabled()
                        observations.append(
                            {
                                "outcome": "test_local_experiment_journey_04",
                                "original_revision": original_plan["Revision"],
                                "diagnostics": experiment.get_by_label(
                                    "Original attempt diagnostics", exact=True
                                ).inner_text(),
                            }
                        )
                        experiment.get_by_role("button", name="Prepare numerical plan").click()
                        expect(
                            experiment.get_by_role("button", name="Run experiment", exact=True)
                        ).to_be_enabled()
                        fail_next = True
                        experiment.get_by_role("button", name="Run experiment", exact=True).click()
                        expect(
                            experiment.get_by_role("status", name="Experiment lifecycle")
                        ).to_contain_text("failed")
                        _disposed(page)
                        expect(
                            experiment.get_by_label("Current experiment result", exact=True)
                        ).to_have_count(0)
                        _navigate(page, "Workspace")
                        assert _archive(page) == newer
                        damaged = json.loads(newer)
                        member = next(row for row in damaged["members"] if row["kind"] == "raw")
                        member["content"] = (
                            "01" if member["content"][:2] != "01" else "00"
                        ) + member["content"][2:]
                        page.get_by_label("Workspace archive JSON").fill(json.dumps(damaged))
                        page.get_by_role("button", name="Preview archive", exact=True).click()
                        expect(
                            page.get_by_role("button", name="Save draft and revision references")
                        ).to_be_disabled()
                        page.get_by_role("button", name="Reload saved workspace").click()
                        expect(page.get_by_label("Workspace archive JSON")).to_have_value(newer)
                        observations.append(
                            {
                                "outcome": "native-worker-failure-and-tamper-refusal-retain-saved-source"
                            }
                        )
                    assert not errors, errors
                    assert not rejected, rejected
                    _disposed(page)
                    trace = page.evaluate("window.__ownedKernel")
                    assert trace["started"] == trace["disposed"]
                    if profiler is not None:
                        take_native_coverage(profiler, records, include_experiments=True)
                finally:
                    for route in held:
                        route.abort()
                    context.close()
            return receipt
        finally:
            browser.close()
