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
import os
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from tools.studio_owned_worker_journey import _OBSERVE_WORKERS, wait_for_worker_disposal

if TYPE_CHECKING:
    from playwright.sync_api import Browser, CDPSession, Locator, Page, Route

_KERNEL = "scpn_quantum_studio_wasm_kernel.wasm"
_SAMPLE = "kuramoto_scenario_meanfield_20260708.json"
_STARTED = "window.__ownedKernel.started"
_RESULT = "Current experiment result"
_RUN = "Run experiment"


def _plan_identity(page: Page) -> dict[str, str]:
    """Read the actual rendered source and plan, without inventing a request.

    Parameters
    ----------
    page : Page
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
    page : Page
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
    page : Page
        Actual Workbench browser page.
    view : str
        Original Workspace or Run view label.

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


def _execute(page: Page) -> dict[str, object]:
    """Explicitly run the original worker and retain its genuine native event.

    Parameters
    ----------
    page : Page
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
    wait_for_worker_disposal(page)
    trace = page.evaluate("window.__ownedKernel")
    return cast(
        "dict[str, object]",
        next(row for row in reversed(trace["events"]) if row["kind"] == "result"),
    )


def _execute_registered_saved(page: Page) -> dict[str, object]:
    """Run the compiled job against the original committed browser workspace.

    Parameters
    ----------
    page : Page
        Original Workbench with a saved archive and its prepared numerical plan.

    Returns
    -------
    dict[str, object]
        Actual original worker result after confirmed disposal.

    """
    from playwright.sync_api import expect

    before = page.evaluate("window.__ownedKernel.started")
    dock = page.get_by_role("region", name="Registered job dock", exact=True)
    dock.get_by_role("button", name="Run prepared local experiment", exact=True).click()
    expect(page.get_by_role("status", name="Registered job lifecycle")).to_contain_text("settled:")
    wait_for_worker_disposal(page)
    trace = page.evaluate("window.__ownedKernel")
    assert trace["started"] == before + 1
    assert trace["started"] == trace["disposed"]
    return cast(
        "dict[str, object]",
        next(row for row in reversed(trace["events"]) if row["kind"] == "result"),
    )


@dataclass
class _Transport:
    """Route one boundary's requests: refuse, hold, fail or replace them as armed."""

    authority: str
    unavailable: bool = False
    invalid_sample: bool = False
    refused: list[str] = field(default_factory=list)
    held: list[Route] = field(default_factory=list)
    damaged: list[str] = field(default_factory=list)
    hold_next: bool = False
    fail_next: bool = False
    hold_sample_kernel: bool = False

    def __call__(self, route: Route) -> None:
        """Restrict requests to the owned host and apply the armed interference.

        Parameters
        ----------
        route : Route
            Intercepted request.

        """
        target = urlsplit(route.request.url)
        if target.scheme != "http" or target.netloc != self.authority:
            self.refused.append(route.request.url)
            route.abort()
        elif self.hold_sample_kernel and target.path.endswith(_KERNEL):
            self.hold_sample_kernel = False
            self.held.append(route)
        elif self.unavailable and target.path.endswith(".wasm"):
            route.abort()
        elif self.invalid_sample and target.path.endswith(_SAMPLE):
            self.damaged.append(route.request.url)
            route.fulfill(
                status=200, content_type="application/javascript", body="export default {};\n"
            )
        elif Path(target.path).name.startswith("kernelWorker-") or (
            Path(target.path).name == "kernelWorker.ts"
        ):
            self.worker_entry(route)
        else:
            route.continue_()

    def worker_entry(self, route: Route) -> None:
        """Hold, fail or pass the worker entry request as armed.

        Parameters
        ----------
        route : Route
            Intercepted worker entry request.

        """
        if self.hold_next:
            self.hold_next = False
            self.held.append(route)
        elif self.fail_next:
            self.fail_next = False
            route.abort()
        else:
            route.continue_()

    def release(self, *, deliver: bool) -> None:
        """Continue or abort every held request.

        Parameters
        ----------
        deliver : bool
            Continue the held requests when true; abort them otherwise.

        """
        for route in self.held:
            if deliver:
                route.continue_()
            else:
                route.abort()
        self.held.clear()


@dataclass
class _Carried:
    """What the original boundary hands to the fresh replays."""

    exported: str = ""
    plan: dict[str, str] = field(default_factory=dict)
    result: dict[str, object] = field(default_factory=dict)


def _refusal_without_draft(page: Page, experiment: Locator, status: str) -> None:
    """Open the sample and find it refused with no plan, result, draft or worker.

    Parameters
    ----------
    page : Page
        Owned page on the Experiments route.
    experiment : Locator
        The local experiment region.
    status : str
        Exact operation status of the refusal.

    """
    from playwright.sync_api import expect

    experiment.get_by_role("button", name="Open Kuramoto sample").click()
    expect(experiment.get_by_role("status", name="Experiment operation")).to_have_text(status)
    expect(experiment.get_by_label("Numerical plan", exact=True)).to_have_count(0)
    expect(experiment.get_by_label(_RESULT, exact=True)).to_have_count(0)
    _navigate(page, "Workspace")
    assert _archive(page) == ""
    assert page.evaluate(_STARTED) == 0


def _unmounted_sample(
    page: Page, experiment: Locator, transport: _Transport, receipt: dict[str, object]
) -> dict[str, object]:
    """Leave the view while the sample kernel loads and find the late load ignored.

    Parameters
    ----------
    page : Page
        Owned page on the Experiments route.
    experiment : Locator
        The local experiment region.
    transport : _Transport
        Route state of this boundary.
    receipt : dict[str, object]
        Caller-owned evidence receiving the draft identities.

    Returns
    -------
    dict[str, object]
        The observation.

    """
    from playwright.sync_api import expect

    original_draft = _archive(page)
    transport.hold_sample_kernel = True
    with page.expect_request(lambda request: urlsplit(request.url).path.endswith(_KERNEL)):
        experiment.get_by_role("button", name="Open Kuramoto sample").click()
    page.get_by_role("navigation", name="Workbench context destinations").get_by_role(
        "link", name="Atlas", exact=True
    ).click()
    expect(page.get_by_role("heading", name="Atlas unavailable", exact=True)).to_be_visible()
    assert len(transport.held) == 1
    transport.release(deliver=True)
    page.wait_for_load_state("networkidle")
    current_draft = _archive(page)
    receipt["late_sample_source"] = {
        "before_sha256": hashlib.sha256(original_draft.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(current_draft.encode()).hexdigest(),
        "before_characters": len(original_draft),
        "after_characters": len(current_draft),
    }
    assert current_draft == original_draft, receipt["late_sample_source"]
    assert page.evaluate(_STARTED) == 0
    expect(page.get_by_label(_RESULT, exact=True)).to_have_count(0)
    return {"outcome": "late-native-sample-load-cannot-overwrite-unmounted-view"}


def _import_saved(page: Page, exported: str) -> None:
    """Import the exported archive into the empty workspace and save it.

    Parameters
    ----------
    page : Page
        Owned page of a fresh context.
    exported : str
        Saved archive exported by the original boundary.

    """
    from playwright.sync_api import expect

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
    save = page.get_by_role("button", name="Save draft and revision references")
    expect(save).to_be_enabled()
    save.click()
    expect(page.get_by_label("Saved workspace identity", exact=True)).to_be_visible()


def _fresh_replay(page: Page, experiment: Locator, carried: _Carried) -> dict[str, object]:
    """Import the exported archive in a clean context and replay it bit for bit.

    Parameters
    ----------
    page : Page
        Owned page on the Experiments route.
    experiment : Locator
        The local experiment region.
    carried : _Carried
        Export, plan and result of the original boundary.

    Returns
    -------
    dict[str, object]
        The observation.

    """
    from playwright.sync_api import expect

    _import_saved(page, carried.exported)
    _navigate(page, "Run")
    experiment.get_by_role("button", name="Prepare saved replay").click()
    expect(experiment.get_by_role("button", name=_RUN, exact=True)).to_be_enabled()
    replay_plan = _plan_identity(page)
    assert replay_plan == carried.plan
    actual = _execute(page)
    replay_payload = cast("dict[str, object]", actual["payload"])
    original_payload = cast("dict[str, object]", carried.result["payload"])
    assert replay_payload["orderParameter"] == original_payload["orderParameter"]
    assert replay_payload["thetaFinal"] == original_payload["thetaFinal"]
    verified = "Experiment succeeded · Original float64 replay verified"
    expect(experiment.get_by_text(verified, exact=True)).to_be_visible()
    return {
        "outcome": "test_local_experiment_journey_01",
        "source_plan": replay_plan,
        "native_replay": actual,
        "portable_sha256": hashlib.sha256(carried.exported.encode()).hexdigest(),
    }


def _set_coupling(page: Page, value: str) -> None:
    """Set the coupling in the linked parameter editor and save a parameter revision.

    Parameters
    ----------
    page : Page
        Owned page on the Workspace route.
    value : str
        Decimal coupling value.

    """
    editor = page.get_by_label("Linked parameter editor", exact=True)
    editor.get_by_role("button", name="coupling[scalar]", exact=False).click()
    editor.get_by_label("Selected value").fill(value)
    editor.get_by_role("button", name="Apply value", exact=True).click()
    editor.get_by_role("button", name="Save parameter revision").click()


def _edit_sample(page: Page, experiment: Locator) -> str:
    """Open and validate the sample, then save a coupling of 1.4 as a new revision.

    Parameters
    ----------
    page : Page
        Owned page on the Experiments route.
    experiment : Locator
        The local experiment region.

    Returns
    -------
    str
        The edited saved archive.

    """
    from playwright.sync_api import expect

    experiment.get_by_role("button", name="Open Kuramoto sample").click()
    experiment.get_by_role("button", name="Validate experiment archive").click()
    expect(experiment.get_by_role("button", name="Prepare numerical plan")).to_be_enabled()
    _navigate(page, "Workspace")
    sample_archive = _archive(page)
    _set_coupling(page, "1.4")
    expect(page.get_by_label("Saved workspace identity", exact=True)).to_be_visible()
    edited = _archive(page)
    assert edited != sample_archive
    return edited


def _run_and_export(
    page: Page,
    experiment: Locator,
    edited: str,
    carried: _Carried,
    observations: list[dict[str, object]],
) -> None:
    """Prepare the plan, run it as a registered job and directly, and export the attempt.

    Parameters
    ----------
    page : Page
        Owned page on the Workspace route.
    experiment : Locator
        The local experiment region.
    edited : str
        Edited saved archive, which the portable export must leave unchanged.
    carried : _Carried
        Receives the plan and the direct result.
    observations : list[dict[str, object]]
        Caller-owned list that receives the registered-job outcome.

    """
    from playwright.sync_api import expect

    _navigate(page, "Run")
    experiment.get_by_role("button", name="Prepare numerical plan").click()
    expect(experiment.get_by_role("button", name=_RUN, exact=True)).to_be_enabled()
    carried.plan = _plan_identity(page)
    registered = _execute_registered_saved(page)
    observations.append(
        {
            "outcome": "registered-job-on-original-saved-browser-source",
            "source_plan": carried.plan,
            "native_result": registered,
        }
    )
    carried.result = _execute(page)
    with page.expect_download() as portable_download:
        experiment.get_by_role("button", name="Export experiment attempt").click()
    portable_attempt = portable_download.value.path().read_text(encoding="utf-8")
    members = json.loads(portable_attempt)["members"]
    assert any(row["schema"] == "local_run_record.v1" for row in members)
    _navigate(page, "Workspace")
    assert _archive(page) == edited, "Portable attempt export changed the original saved source"


def _save_and_export(page: Page, experiment: Locator) -> str:
    """Save the attempt and export the saved experiment exactly as the workspace holds it.

    Parameters
    ----------
    page : Page
        Owned page on the Workspace route.
    experiment : Locator
        The local experiment region.

    Returns
    -------
    str
        The exported saved archive.

    """
    from playwright.sync_api import expect

    _navigate(page, "Run")
    experiment.get_by_role("button", name="Save experiment attempt").click()
    expect(experiment.get_by_role("status", name="Experiment operation")).to_contain_text(
        "Exact experiment attempt committed"
    )
    expect(experiment.get_by_label(_RESULT, exact=True)).to_be_visible()
    with page.expect_download() as download:
        experiment.get_by_role("button", name="Export saved experiment").click()
    exported = download.value.path().read_text(encoding="utf-8")
    _navigate(page, "Workspace")
    assert _archive(page) == exported
    return exported


def _refuse_budget(page: Page, experiment: Locator, saved: str) -> dict[str, object]:
    """Refuse a zero numeric byte budget before any worker starts.

    Parameters
    ----------
    page : Page
        Owned page on the Workspace route.
    experiment : Locator
        The local experiment region.
    saved : str
        Saved archive, which must stay unchanged.

    Returns
    -------
    dict[str, object]
        The observation.

    """
    from playwright.sync_api import expect

    _navigate(page, "Run")
    before = page.evaluate(_STARTED)
    experiment.get_by_label("Run numeric byte budget").fill("0")
    expect(experiment.get_by_role("button", name=_RUN, exact=True)).to_be_disabled()
    experiment.get_by_role("button", name="Prepare numerical plan").click()
    lifecycle = experiment.get_by_role("status", name="Experiment lifecycle")
    expect(lifecycle).to_contain_text("refused")
    assert page.evaluate(_STARTED) == before
    _navigate(page, "Workspace")
    assert _archive(page) == saved
    _navigate(page, "Run")
    return {"outcome": "test_local_experiment_journey_02", "workers_started": before}


def _cancel_held(
    page: Page, experiment: Locator, transport: _Transport, saved: str
) -> dict[str, object]:
    """Cancel a run whose worker entry is held and find no result and no saved change.

    Parameters
    ----------
    page : Page
        Owned page on the Run route.
    experiment : Locator
        The local experiment region.
    transport : _Transport
        Route state of this boundary.
    saved : str
        Saved archive, which must stay unchanged.

    Returns
    -------
    dict[str, object]
        The observation with the attempt diagnostics.

    """
    from playwright.sync_api import expect

    experiment.get_by_label("Run numeric byte budget").fill("")
    experiment.get_by_role("button", name="Prepare numerical plan").click()
    expect(experiment.get_by_role("button", name=_RUN, exact=True)).to_be_enabled()
    transport.hold_next = True
    experiment.get_by_role("button", name=_RUN, exact=True).click()
    page.wait_for_function("window.__ownedKernel.active === 1")
    experiment.get_by_role("button", name="Cancel experiment", exact=True).click()
    lifecycle = experiment.get_by_role("status", name="Experiment lifecycle")
    expect(lifecycle).to_contain_text("cancelled")
    wait_for_worker_disposal(page)
    expect(experiment.get_by_label(_RESULT, exact=True)).to_have_count(0)
    diagnostics = experiment.get_by_label("Original attempt diagnostics", exact=True)
    expect(diagnostics).to_contain_text("cancelled")
    _navigate(page, "Workspace")
    assert _archive(page) == saved
    _navigate(page, "Run")
    observation: dict[str, object] = {
        "outcome": "test_local_experiment_journey_03",
        "diagnostics": diagnostics.inner_text(),
    }
    transport.release(deliver=False)
    return observation


def _stale_after_edit(
    page: Page, experiment: Locator, transport: _Transport, saved: str, revision: str
) -> tuple[dict[str, object], str]:
    """Save a newer coupling while a run is held and find the late result stale.

    Parameters
    ----------
    page : Page
        Owned page on the Run route.
    experiment : Locator
        The local experiment region.
    transport : _Transport
        Route state of this boundary.
    saved : str
        Saved archive the held run started from.
    revision : str
        Revision of the original plan.

    Returns
    -------
    tuple[dict[str, object], str]
        The observation and the newer saved archive.

    """
    from playwright.sync_api import expect

    transport.hold_next = True
    experiment.get_by_role("button", name=_RUN, exact=True).click()
    page.wait_for_function("window.__ownedKernel.active === 1")
    _navigate(page, "Workspace")
    wait_for_worker_disposal(page)
    _set_coupling(page, "1.6")
    committed = page.get_by_role("status").filter(
        has_text="Parameter revision transaction committed"
    )
    expect(committed).to_be_visible()
    expect(page.get_by_label("Workspace archive JSON")).not_to_have_value(saved)
    newer = _archive(page)
    assert newer != saved, "Actual second parameter transaction did not update the source archive"
    transport.release(deliver=True)
    _navigate(page, "Run")
    expect(experiment.get_by_role("status", name="Experiment lifecycle")).to_contain_text("stale")
    expect(experiment.get_by_label(_RESULT, exact=True)).to_have_count(0)
    expect(experiment.get_by_role("button", name="Save experiment attempt")).to_be_disabled()
    diagnostics = experiment.get_by_label("Original attempt diagnostics", exact=True)
    return {
        "outcome": "test_local_experiment_journey_04",
        "original_revision": revision,
        "diagnostics": diagnostics.inner_text(),
    }, newer


def _fail_and_tamper(
    page: Page, experiment: Locator, transport: _Transport, newer: str
) -> dict[str, object]:
    """Fail the worker entry, then refuse a tampered raw member and reload the save.

    Parameters
    ----------
    page : Page
        Owned page on the Run route.
    experiment : Locator
        The local experiment region.
    transport : _Transport
        Route state of this boundary.
    newer : str
        Newer saved archive, which must survive both refusals.

    Returns
    -------
    dict[str, object]
        The observation.

    """
    from playwright.sync_api import expect

    experiment.get_by_role("button", name="Prepare numerical plan").click()
    expect(experiment.get_by_role("button", name=_RUN, exact=True)).to_be_enabled()
    transport.fail_next = True
    experiment.get_by_role("button", name=_RUN, exact=True).click()
    lifecycle = experiment.get_by_role("status", name="Experiment lifecycle")
    expect(lifecycle).to_contain_text("failed")
    wait_for_worker_disposal(page)
    expect(experiment.get_by_label(_RESULT, exact=True)).to_have_count(0)
    _navigate(page, "Workspace")
    assert _archive(page) == newer
    damaged = json.loads(newer)
    member = next(row for row in damaged["members"] if row["kind"] == "raw")
    flipped = "01" if member["content"][:2] != "01" else "00"
    member["content"] = flipped + member["content"][2:]
    page.get_by_label("Workspace archive JSON").fill(json.dumps(damaged))
    page.get_by_role("button", name="Preview archive", exact=True).click()
    save = page.get_by_role("button", name="Save draft and revision references")
    expect(save).to_be_disabled()
    page.get_by_role("button", name="Reload saved workspace").click()
    expect(page.get_by_label("Workspace archive JSON")).to_have_value(newer)
    return {"outcome": "native-worker-failure-and-tamper-refusal-retain-saved-source"}


def _original(
    page: Page,
    experiment: Locator,
    transport: _Transport,
    carried: _Carried,
    observations: list[dict[str, object]],
) -> None:
    """Edit, run, save, export, refuse, cancel, invalidate and fail on the original host.

    Parameters
    ----------
    page : Page
        Owned page on the Experiments route.
    experiment : Locator
        The local experiment region.
    transport : _Transport
        Route state of this boundary.
    carried : _Carried
        Receives the export, plan and result for the fresh replays.
    observations : list[dict[str, object]]
        Caller-owned list that records each outcome once it holds.

    """
    edited = _edit_sample(page, experiment)
    _run_and_export(page, experiment, edited, carried, observations)
    saved = carried.exported = _save_and_export(page, experiment)
    observations.append(_refuse_budget(page, experiment, saved))
    observations.append(_cancel_held(page, experiment, transport, saved))
    stale, newer = _stale_after_edit(page, experiment, transport, saved, carried.plan["Revision"])
    observations.append(stale)
    observations.append(_fail_and_tamper(page, experiment, transport, newer))


def _walk(
    page: Page,
    boundary: str,
    transport: _Transport,
    carried: _Carried,
    receipt: dict[str, object],
) -> None:
    """Run one boundary's part of the journey on its freshly loaded Experiments route.

    Parameters
    ----------
    page : Page
        Owned page on the Experiments route, with no worker started.
    boundary : str
        Boundary name.
    transport : _Transport
        Route state of this boundary.
    carried : _Carried
        State the original boundary hands to the fresh replays.
    receipt : dict[str, object]
        Caller-owned evidence holding the observations list.

    """
    observations = cast(list[dict[str, object]], receipt["observations"])
    experiment = page.get_by_label("Local experiment", exact=True)
    if boundary == "source-unmounted-sample":
        observations.append(_unmounted_sample(page, experiment, transport, receipt))
    elif boundary == "source-invalid-sample":
        _refusal_without_draft(page, experiment, "original committed sample is unavailable")
        assert len(transport.damaged) == 1
        observations.append(
            {
                "outcome": "invalid-original-sample-refuses-without-draft-or-worker",
                "actual_interrupted_source": transport.damaged[0],
            }
        )
    elif boundary == "missing-kernel":
        refused = "Local experiment operation refused; current draft and saved archive retained"
        _refusal_without_draft(page, experiment, refused)
        observations.append({"outcome": "missing-native-kernel-refuses-without-draft-or-worker"})
    elif boundary.endswith("fresh-replay"):
        observations.append(_fresh_replay(page, experiment, carried))
    else:
        _original(page, experiment, transport, carried, observations)


def _finish_boundary(
    page: Page,
    failures: tuple[list[str], list[str]],
    profiler: CDPSession | None,
    receipt: dict[str, object],
) -> None:
    """Refuse page errors and outside requests, require disposal, take any coverage.

    Parameters
    ----------
    page : Page
        Owned page of the finished boundary.
    failures : tuple[list[str], list[str]]
        The page errors and the refused outside requests; both must be empty.
    profiler : CDPSession | None
        Native profiler of a source boundary, if any.
    receipt : dict[str, object]
        Caller-owned evidence holding the native coverage records.

    """
    from tools.studio_workspace_browser_coverage import take_native_coverage

    for failure in failures:
        assert not failure, failure
    wait_for_worker_disposal(page)
    trace = page.evaluate("window.__ownedKernel")
    assert trace["started"] == trace["disposed"]
    if profiler is not None:
        records = cast(list[dict[str, object]], receipt["native_v8_coverage"])
        take_native_coverage(profiler, records, include_experiments=True)


def _boundary_on(
    browser: Browser,
    boundary: str,
    target: str,
    carried: _Carried,
    receipt: dict[str, object],
) -> None:
    """Run one boundary in its own context and check its custody and worker disposal.

    Parameters
    ----------
    browser : Browser
        Launched browser.
    boundary : str
        Boundary name; ``source-`` boundaries record native coverage.
    target : str
        Owned host URL of this boundary.
    carried : _Carried
        State the original boundary hands to the fresh replays.
    receipt : dict[str, object]
        Caller-owned evidence holding the observations and coverage records.

    """
    from playwright.sync_api import expect

    from tools.studio_workspace_browser_coverage import start_native_coverage

    context = browser.new_context(service_workers="block")
    context.set_default_timeout(15_000)
    context.add_init_script(_OBSERVE_WORKERS)
    transport = _Transport(
        urlsplit(target).netloc,
        unavailable=boundary == "missing-kernel",
        invalid_sample=boundary == "source-invalid-sample",
    )
    errors: list[str] = []
    try:
        context.route("**/*", transport)
        page = context.new_page()
        profiler = start_native_coverage(page) if boundary.startswith("source-") else None
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(target + "#/experiments", wait_until="networkidle")
        experiment = page.get_by_label("Local experiment", exact=True)
        expect(experiment.get_by_role("heading", name="Local experiment")).to_be_visible()
        assert page.evaluate(_STARTED) == 0
        _walk(page, boundary, transport, carried, receipt)
        _finish_boundary(page, (errors, transport.refused), profiler, receipt)
    finally:
        transport.release(deliver=False)
        context.close()


def _owned_hosts(base_url: str, source_url: str | None) -> tuple[str, str | None]:
    """Validate the built host and the optional distinct root source host.

    Parameters
    ----------
    base_url : str
        Owned literal loopback preview of the current production build.
    source_url : str | None
        Optional distinct owned root Vite host for original-source native counters.

    Returns
    -------
    tuple[str, str | None]
        The built host and the source host, or None without one.

    Raises
    ------
    ValueError
        The supplied preview is not an owned plain HTTP loopback URL.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    source = None if source_url is None else loopback_url(source_url)
    if source is not None and (
        urlsplit(source).path != "/" or urlsplit(source).netloc == urlsplit(url).netloc
    ):
        raise ValueError("Experiment source coverage requires a distinct owned root source host")
    return url, source


def _boundaries(source: str | None) -> tuple[str, ...]:
    """List the boundaries in order, with the source boundaries when a source host exists.

    Parameters
    ----------
    source : str | None
        Source host, if any.

    Returns
    -------
    tuple[str, ...]
        Boundary names.

    """
    built = ("original", "fresh-replay", "missing-kernel")
    if source is None:
        return built
    return (
        *built,
        "source-original",
        "source-fresh-replay",
        "source-invalid-sample",
        "source-unmounted-sample",
    )


def run_local_experiment_journey(
    base_url: str, *, source_url: str | None = None, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Exercise sample, edit, plan, genuine execution, save, import and replay.

    Parameters
    ----------
    base_url : str
        Owned literal loopback preview of the current production build.
    source_url : str | None
        Optional distinct owned root Vite host for original-source native counters.
    evidence : dict[str, object] | None
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
    url, source = _owned_hosts(base_url, source_url)
    from playwright.sync_api import sync_playwright

    receipt = evidence if evidence is not None else {}
    receipt.update(scenario="local_experiment_journey", observations=[])
    if source is not None:
        receipt.update(
            source_url=source, native_v8_coverage=[], coverage_percentage="not_calculated"
        )
    carried = _Carried()
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        receipt.update(playwright=version("playwright"), browser=browser.version)
        try:
            for boundary in _boundaries(source):
                target = source if source is not None and boundary.startswith("source-") else url
                _boundary_on(browser, boundary, target, carried, receipt)
            return receipt
        finally:
            browser.close()
