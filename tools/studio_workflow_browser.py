# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original workflow browser acceptance
"""Exercise native workflow grids, saved history and cancellation in actual Chromium."""

from __future__ import annotations

import os
from contextlib import suppress
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json
from tools.studio_owned_worker_journey import _OBSERVE_WORKERS, wait_for_worker_disposal

if TYPE_CHECKING:
    from playwright.sync_api import Page, Worker


def _navigate(page: Page, view: str) -> None:
    """Activate an original workbench view through its visible navigation link.

    Parameters
    ----------
    page : Page
        Actual owned workbench page.
    view : str
        Exact original view label.

    """
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name=view, exact=True
    ).click()


def _archive(page: Page) -> dict[str, object]:
    """Read the current original archive without changing its editor.

    Parameters
    ----------
    page : Page
        Actual owned page displaying the Workspace editor.

    Returns
    -------
    dict[str, object]
        Original losslessly decoded archive.

    """
    return cast(
        dict[str, object], read_json(page.get_by_label("Workspace archive JSON").input_value())
    )


def _selected_journal(archive: dict[str, object]) -> dict[str, object]:
    """Read the journal body of the archive's explicitly selected workflow.

    Parameters
    ----------
    archive : dict[str, object]
        Original archive containing its admitted workflow extension.

    Returns
    -------
    dict[str, object]
        Original selected journal body.

    """
    manifest = cast(dict[str, object], archive["manifest"])
    extension = cast(
        dict[str, object], cast(dict[str, object], manifest["extensions"])["experiment_workflows"]
    )
    items = cast(list[dict[str, object]], extension["items"])
    selected = next(item for item in items if item["hash"] == extension["selected"])
    return cast(dict[str, object], cast(dict[str, object], selected["journal"])["body"])


def _graph(page: Page, definition: dict[str, object], *, save: bool = True) -> None:
    """Preview an original graph and optionally commit it through its public UI.

    Parameters
    ----------
    page : Page
        Actual workbench page with workflow controls.
    definition : dict[str, object]
        Exact graph definition supplied to the original editor.
    save : bool
        Require original admission and explicit save when true.

    """
    from playwright.sync_api import expect

    panel = page.get_by_role("region", name="Reproducible workflows", exact=True)
    panel.get_by_role("textbox", name="Workflow JSON", exact=True).fill(write_json(definition))
    panel.get_by_role("button", name="Preview workflow graph", exact=True).click()
    editor = panel.get_by_role("region", name="Workflow graph editor", exact=True)
    if save:
        expect(editor.get_by_role("status")).to_contain_text("Original graph admitted")
        editor.get_by_role("button", name="Save workflow graph", exact=True).click()
        expect(editor.get_by_role("status")).to_contain_text("graph saved")


def _wait_for_entries(page: Page, statuses: list[str], state: str) -> None:
    """Await each original stage's progress and inspect its journal entry.

    Parameters
    ----------
    page : Page
        Actual owned workflow page with the original progress and journal controls.
    statuses : list[str]
        Ordered statuses required for every original stage attempt.
    state : str
        Required terminal journal state after all listed attempts settle.

    Raises
    ------
    AssertionError
        An original entry, journal state or final run control fails its assertion.
    TimeoutError
        A real stage does not settle before its existing individual deadline.

    """
    from playwright.sync_api import expect

    panel = page.get_by_role("region", name="Reproducible workflows", exact=True)
    history = panel.get_by_role("region", name="Original workflow attempt history", exact=True)
    entries = history.get_by_role("listitem")
    for index, status in enumerate(statuses):
        page.wait_for_function(
            """([index, status]) => {
              const panel = document.querySelector('[aria-label="Reproducible workflows"]');
              const history = panel?.querySelector('[aria-label="Original workflow attempt history"]');
              return Number(panel?.getAttribute('data-workflow-settled')) >= index + 1 &&
                history?.querySelectorAll('li')[index]?.textContent?.includes(' · ' + status);
            }""",
            arg=[index, status],
        )
        expect(entries.nth(index)).to_contain_text(" · " + status)
    expect(entries).to_have_count(len(statuses))
    expect(history).to_contain_text("Journal: " + state)
    expect(panel.get_by_role("button", name="Run or resume workflow", exact=True)).to_be_enabled()


def run_workflow_journey(
    base_url: str, *, source_url: str | None = None, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Run the original workflow acceptance on actual owned production/source hosts.

    Parameters
    ----------
    base_url : str
        Owned literal loopback root serving the current built original frontend.
    source_url : str | None
        Optional distinct owned loopback root serving the current original source.
    evidence : dict[str, object] | None
        Original caller-owned observation destination, retaining failures.

    Returns
    -------
    dict[str, object]
        Actual runtime versions, native worker counts and original saved journals.
        Source metadata and recorded results never attest a physical device.

    Raises
    ------
    ValueError
        A target is external, ambiguous, shared or a nested source path.
    AssertionError
        The actual graph, source custody, worker lifecycle or journal invariant fails.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    source = None if source_url is None else loopback_url(source_url)
    if source is not None and (
        urlsplit(source).path != "/" or urlsplit(source).netloc == urlsplit(url).netloc
    ):
        raise ValueError("Workflow source requires a distinct owned root source host")
    from playwright.sync_api import Error, Request, Route, expect, sync_playwright

    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    receipt = evidence if evidence is not None else {}
    observations: list[dict[str, object]] = []
    receipt.update(scenario="experiment_workflow_runner", observations=observations)
    records: list[dict[str, object]] = []
    if source is not None:
        receipt.update(
            source_url=source, native_v8_coverage=records, coverage_percentage="not_calculated"
        )
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        receipt.update(playwright=version("playwright"), browser=browser.version)
        try:
            for target in (url,) if source is None else (url, source):
                context = browser.new_context(service_workers="block")
                context.set_default_timeout(15_000)
                context.add_init_script(_OBSERVE_WORKERS)
                errors: list[str] = []
                refused: list[str] = []
                held: list[Route] = []
                held_entries: list[str] = []
                authority = urlsplit(target).netloc
                profiler = None

                def bound_request(
                    route: Route,
                    _sdk_request: Request,
                    *,
                    current_authority: str = authority,
                    rejected: list[str] = refused,
                ) -> None:
                    """Bound actual transport to this context's captured loopback origin.

                    Parameters
                    ----------
                    route : Route
                        Actual intercepted SDK route.
                    _sdk_request : Request
                        Original second SDK callback argument.
                    current_authority : str
                        Exact captured loopback authority.
                    rejected : list[str]
                        Destination for refused request URLs.

                    """
                    request = urlsplit(route.request.url)
                    if request.scheme != "http" or request.netloc != current_authority:
                        rejected.append(route.request.url)
                        route.abort()
                    else:
                        route.continue_()

                def page_error(error: Error, *, observed: list[str] = errors) -> None:
                    """Retain actual browser failures in this context's evidence.

                    Parameters
                    ----------
                    error : Error
                        Original browser failure.
                    observed : list[str]
                        Context-owned failure destination.

                    """
                    observed.append(str(error))

                try:
                    context.route("**/*", bound_request)
                    page = context.new_page()
                    if target == source:
                        profiler = start_native_coverage(page)
                    page.on("pageerror", page_error)
                    page.goto(target + "#/experiments", wait_until="networkidle")
                    page.get_by_role("region", name="Local experiment", exact=True).get_by_role(
                        "button", name="Open Kuramoto sample", exact=True
                    ).click()
                    _navigate(page, "Workspace")
                    page.get_by_role("button", name="Preview archive", exact=True).click()
                    save = page.get_by_role(
                        "button", name="Save draft and revision references", exact=True
                    )
                    expect(save).to_be_enabled()
                    save.click()
                    expect(
                        page.get_by_role("region", name="Local workspace", exact=True).get_by_role(
                            "status"
                        )
                    ).to_contain_text("transaction committed")
                    initial = _archive(page)
                    receipt["initial_native_workers"] = page.evaluate("window.__ownedKernel")
                    _navigate(page, "Run")
                    panel = page.get_by_role("region", name="Reproducible workflows", exact=True)
                    panel.get_by_role("button", name="Compose local workflow", exact=True).click()
                    expect(
                        panel.get_by_role(
                            "list", name="Original workflow dependency graph", exact=True
                        )
                    ).to_be_visible()
                    graph = cast(
                        dict[str, object],
                        read_json(
                            panel.get_by_role(
                                "textbox", name="Workflow JSON", exact=True
                            ).input_value()
                        ),
                    )
                    for fault in ("cycle", "port"):
                        wait_for_worker_disposal(page)
                        before_workers = page.evaluate("window.__ownedKernel.started")
                        invalid = cast(dict[str, object], read_json(write_json(graph)))
                        invalid_body = cast(dict[str, object], invalid["body"])
                        invalid_stages = cast(list[dict[str, object]], invalid_body["stages"])
                        if fault == "cycle":
                            invalid_stages[0]["depends_on"] = ["analyse"]
                        else:
                            binding = cast(list[dict[str, object]], invalid_stages[2]["inputs"])[0]
                            cast(dict[str, object], binding["type"])["unit"] = "rad"
                        _graph(page, invalid, save=False)
                        receipt["current_fault"] = fault
                        receipt["current_native_workers"] = page.evaluate("window.__ownedKernel")
                        editor = panel.get_by_role(
                            "region", name="Workflow graph editor", exact=True
                        )
                        expect(
                            editor.get_by_role("button", name="Save workflow graph", exact=True)
                        ).to_be_disabled()
                        assert page.evaluate("window.__ownedKernel.started") == before_workers
                        _navigate(page, "Workspace")
                        assert _archive(page) == initial
                        _navigate(page, "Run")
                        observations.append(
                            {
                                "host": target,
                                "case": fault,
                                "native_workers": 0,
                                "original_source_unchanged": True,
                            }
                        )
                    body = cast(dict[str, object], graph["body"])
                    stages = cast(list[dict[str, object]], body["stages"])
                    for stage in stages[:2]:
                        stage["parameters"] = {"steps": 4}
                    sweep = cast(dict[str, object], body["sweep"])
                    sweep["axes"] = [
                        {"stage_id": "simulate", "parameter": "coupling", "values": [1.2, 1.4]},
                        {"stage_id": "simulate", "parameter": "dt", "values": [0.01, 0.02, 0.04]},
                    ]
                    sweep["evaluation_budget"] = 18
                    _graph(page, graph)
                    with page.expect_download() as transfer:
                        panel.get_by_role(
                            "button", name="Export workflow JSON", exact=True
                        ).click()
                    with TemporaryDirectory(prefix="studio-workflow-export-") as directory:
                        exported = Path(directory) / "experiment-workflow.json"
                        transfer.value.save_as(exported)
                        assert read_json(exported.read_text(encoding="utf-8")) == graph
                    before_failure = page.evaluate("window.__ownedKernel.started")
                    refused_kernel_requests: list[str] = []

                    def refuse_kernel(
                        route: Route,
                        request: Request,
                        *,
                        rejected: list[str] = refused_kernel_requests,
                    ) -> None:
                        """Refuse the actual kernel request and retain its intercepted URL.

                        Parameters
                        ----------
                        route : Route
                            Actual kernel transport route.
                        request : Request
                            Actual request, including any built asset hash.
                        rejected : list[str]
                            Context-owned record of refused kernel URLs.

                        """
                        rejected.append(request.url)
                        route.fulfill(status=503, body="Original owned kernel transport refused")

                    kernel_route = "**/scpn_quantum_studio_wasm_kernel*.wasm*"
                    context.route(kernel_route, refuse_kernel)
                    try:
                        panel.get_by_role(
                            "button", name="Run or resume workflow", exact=True
                        ).click()
                        expect(
                            panel.get_by_text(
                                "Original workflow operation refused; prior evidence retained",
                                exact=True,
                            )
                        ).to_be_visible()
                        assert refused_kernel_requests
                        assert page.evaluate("window.__ownedKernel.started") == before_failure
                    finally:
                        context.unroute(kernel_route, refuse_kernel)
                    wait_for_worker_disposal(page)
                    before_run = page.evaluate("window.__ownedKernel.started")
                    previous = panel.get_attribute("data-workflow-traversal")
                    panel.get_by_role("button", name="Run or resume workflow", exact=True).click()
                    page.wait_for_function(
                        """previous => document.querySelector('[aria-label="Reproducible workflows"]')
                          ?.getAttribute('data-workflow-traversal') !== previous""",
                        arg=previous,
                        polling=100,
                    )
                    _wait_for_entries(page, ["complete"] * 18, "complete")
                    wait_for_worker_disposal(page)
                    assert page.evaluate("window.__ownedKernel.started") - before_run == 6
                    _navigate(page, "Workspace")
                    completed = _archive(page)
                    journal = _selected_journal(completed)
                    entries = cast(list[dict[str, object]], journal["entries"])
                    assert journal["evaluations"] == 18 and len(entries) == 18
                    assert len({entry["cell_id"] for entry in entries}) == 6
                    initial_members = cast(list[dict[str, object]], initial["members"])
                    final_members = cast(list[dict[str, object]], completed["members"])
                    assert all(member in final_members for member in initial_members)
                    observations.append(
                        {
                            "host": target,
                            "case": "six_cells",
                            "journal": journal,
                            "native_workers": 6,
                            "input_shape": {"oscillators": 12, "steps": 4},
                            "exported_graph_matches": True,
                            "kernel_transport_refusal_allocated_no_worker": True,
                            "kernel_transport_refused_requests": refused_kernel_requests,
                        }
                    )
                    if profiler is not None:
                        session = profiler
                        profiler = None
                        take_native_coverage(session, records, include_workflows=True)
                        profiler = start_native_coverage(page, session=session)
                    page.reload(wait_until="networkidle")
                    _navigate(page, "Run")
                    panel = page.get_by_role("region", name="Reproducible workflows", exact=True)
                    expect(
                        panel.get_by_role("button", name="Run or resume workflow", exact=True)
                    ).to_be_enabled()
                    wait_for_worker_disposal(page)
                    before_resume = page.evaluate("window.__ownedKernel.started")
                    previous = panel.get_attribute("data-workflow-traversal")
                    receipt["before_resume_traversal"] = previous
                    receipt["before_resume_source"] = panel.evaluate(
                        "element => ({traversal:element.getAttribute('data-workflow-traversal'),settled:element.getAttribute('data-workflow-settled')})"
                    )
                    panel.get_by_role("button", name="Run or resume workflow", exact=True).click()
                    page.wait_for_function(
                        """previous => document.querySelector('[aria-label="Reproducible workflows"]')
                          ?.getAttribute('data-workflow-traversal') !== previous""",
                        arg=previous,
                        polling=100,
                    )
                    _wait_for_entries(page, ["complete"] * 18, "complete")
                    assert page.evaluate("window.__ownedKernel.started") == before_resume
                    _navigate(page, "Workspace")
                    resumed = _archive(page)
                    assert _selected_journal(resumed) == journal
                    assert resumed["members"] == completed["members"]
                    observations.append({"host": target, "case": "restart", "native_workers": 0})
                    _navigate(page, "Run")
                    graph = cast(dict[str, object], read_json(write_json(graph)))
                    body = cast(dict[str, object], graph["body"])
                    body["workflow_id"] = "failed-native-parent"
                    stages = cast(list[dict[str, object]], body["stages"])
                    stages[0]["parameters"] = {"steps": 0}
                    _graph(page, graph)
                    wait_for_worker_disposal(page)
                    before_failed = page.evaluate("window.__ownedKernel.started")
                    previous = panel.get_attribute("data-workflow-traversal")
                    panel.get_by_role("button", name="Run or resume workflow", exact=True).click()
                    page.wait_for_function(
                        """previous => document.querySelector('[aria-label="Reproducible workflows"]')
                          ?.getAttribute('data-workflow-traversal') !== previous""",
                        arg=previous,
                        polling=100,
                    )
                    _wait_for_entries(page, ["failed", "blocked", "blocked"] * 6, "partial")
                    assert page.evaluate("window.__ownedKernel.started") == before_failed
                    _navigate(page, "Workspace")
                    failed = _selected_journal(_archive(page))
                    assert [
                        entry["status"]
                        for entry in cast(list[dict[str, object]], failed["entries"])
                    ] == ["failed", "blocked", "blocked"] * 6
                    observations.append(
                        {"host": target, "case": "failed_parent", "journal": failed}
                    )
                    _navigate(page, "Run")
                    cancellation = cast(dict[str, object], read_json(write_json(graph)))
                    cancel_body = cast(dict[str, object], cancellation["body"])
                    cancel_body["workflow_id"] = "cancel-original-workflow"
                    cast(list[dict[str, object]], cancel_body["stages"])[0]["parameters"] = {}
                    cast(dict[str, object], cancel_body["sweep"]).update(
                        axes=[], evaluation_budget=3
                    )
                    _graph(page, cancellation)

                    def hold_original_worker(
                        route: Route,
                        _sdk_request: Request,
                        *,
                        current_authority: str = authority,
                        rejected: list[str] = refused,
                        entries: list[str] = held_entries,
                    ) -> None:
                        """Hold the actual worker entry's startup fetch, preserving its dependencies."""
                        request = urlsplit(route.request.url)
                        if request.scheme != "http" or request.netloc != current_authority:
                            rejected.append(route.request.url)
                            route.abort()
                        elif Path(request.path).name != "kernelWorker.ts" and not Path(
                            request.path
                        ).name.startswith("kernelWorker-"):
                            route.continue_()
                        else:
                            entries.append(route.request.url)
                            response = route.fetch()
                            route.fulfill(
                                response=response,
                                body="await fetch('/workflow-worker-startup-barrier');\n"
                                + response.text(),
                            )

                    def hold_startup(
                        route: Route, _sdk_request: Request, *, retained: list[Route] = held
                    ) -> None:
                        """Retain an actual worker fetch until its owning worker is disposed."""
                        retained.append(route)

                    context.route("**/workflow-worker-startup-barrier", hold_startup)
                    page.route("**/*kernelWorker*", hold_original_worker)
                    before_cancel = page.evaluate("window.__ownedKernel.started")
                    closed: list[bool] = []

                    def record_closed(_worker: Worker, *, observed: list[bool] = closed) -> None:
                        """Record Chromium's actual worker-close event before any cancel assertion."""
                        observed.append(True)

                    with context.expect_event(
                        "request",
                        predicate=lambda request: (
                            urlsplit(request.url).path == "/workflow-worker-startup-barrier"
                        ),
                    ) as startup:
                        with page.expect_worker() as created:
                            panel.get_by_role(
                                "button", name="Run or resume workflow", exact=True
                            ).click()
                        created.value.once("close", record_closed)
                    panel.get_by_role("button", name="Cancel workflow", exact=True).click()
                    _wait_for_entries(page, ["complete", "cancelled"], "cancelled")
                    wait_for_worker_disposal(page)
                    assert closed == [True]
                    assert len(held_entries) == 1
                    assert len(held) == 1
                    assert startup.value.url == held[0].request.url
                    assert page.evaluate("window.__ownedKernel.started") - before_cancel == 1
                    page.unroute("**/*kernelWorker*", hold_original_worker)
                    context.unroute("**/workflow-worker-startup-barrier", hold_startup)
                    for route in held:
                        # Chromium may already have closed the request with its worker.
                        with suppress(Error):
                            route.abort()
                    _navigate(page, "Workspace")
                    cancelled = _selected_journal(_archive(page))
                    assert cancelled["state"] == "cancelled"
                    cancel_entries = cast(list[dict[str, object]], cancelled["entries"])
                    assert [entry["status"] for entry in cancel_entries] == [
                        "complete",
                        "cancelled",
                    ]
                    assert (
                        cast(dict[str, object], cancel_entries[-1]["output"])["disposed"] is True
                    )
                    _navigate(page, "Run")
                    observations.append(
                        {
                            "host": target,
                            "case": "cancelled",
                            "journal": cancelled,
                            "native_workers": 1,
                            "native_close_observed": closed == [True],
                            "worker_fault": "original script startup fetch held",
                            "worker_url": created.value.url,
                            "held_worker_entries": held_entries,
                            "held_startup_request": startup.value.url,
                        }
                    )
                    assert not errors and not refused
                    if target == source:
                        view_page = context.new_page()
                        view_page.on("pageerror", page_error)
                        view_profiler = start_native_coverage(view_page)
                        try:
                            view_page.goto(target + "#/workspace", wait_until="networkidle")
                            expect(
                                view_page.locator(".qsp-play")
                                .get_by_role("status")
                                .filter(has_text="verified against the committed ground truth")
                            ).to_be_visible()
                            wait_for_worker_disposal(view_page)
                            receipt["native_view_refusals"] = view_page.evaluate(
                                "async () => (await import('/browser-tests/workflowRunner.native.tsx')).runNativeWorkflowViewRefusals()"
                            )
                        finally:
                            take_native_coverage(view_profiler, records, include_workflows=True)
                            view_page.close()
                    assert not errors and not refused
                    if profiler is not None:
                        session, profiler = profiler, None
                        take_native_coverage(session, records, include_workflows=True)
                except Exception as cause:
                    receipt["original_error"] = f"{type(cause).__name__}: {cause}"
                    from traceback import format_exception

                    receipt["original_traceback"] = "".join(format_exception(cause))
                    receipt["failed_context"] = target
                    receipt["page_errors"] = errors
                    receipt["refused_requests"] = refused
                    receipt["held_worker_entries"] = held_entries
                    receipt["held_startup_requests"] = [route.request.url for route in held]
                    if profiler is not None:
                        session, profiler = profiler, None
                        try:
                            take_native_coverage(session, records, include_workflows=True)
                        except Exception as coverage_error:
                            receipt["native_coverage_error"] = (
                                f"{type(coverage_error).__name__}: {coverage_error}"
                            )
                    try:
                        receipt["failed_page_text"] = page.locator("body").inner_text()
                        receipt["resume_predicate_probe"] = page.evaluate(
                            "previous => document.querySelector('[aria-label=\"Reproducible workflows\"]')?.getAttribute('data-workflow-traversal') !== previous",
                            receipt.get("before_resume_traversal"),
                        )
                        receipt["failed_workflow_state"] = page.get_by_role(
                            "region", name="Reproducible workflows", exact=True
                        ).evaluate(
                            "element => ({traversal: element.getAttribute('data-workflow-traversal'), settled: element.getAttribute('data-workflow-settled'), progress: element.querySelector('[aria-label=\"Workflow stage progress\"]')?.textContent, runDisabled: Array.from(element.querySelectorAll('button')).find(button => button.textContent === 'Run or resume workflow')?.disabled})"
                        )
                        receipt["failed_native_worker_counts"] = {
                            name: page.evaluate("window.__ownedKernel." + name)
                            for name in ("started", "disposed", "active")
                        }
                    except Exception as capture_error:
                        receipt["diagnostic_capture_error"] = (
                            f"{type(capture_error).__name__}: {capture_error}"
                        )
                    raise
                finally:
                    context.close()
        finally:
            browser.close()
    receipt.update(
        status="passed",
        claim_boundary="Actual original classical WASM workflow and recorded source metadata; no quantum equivalence or physical device qualification",
    )
    return receipt
