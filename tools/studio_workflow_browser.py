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
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json
from tools.studio_owned_worker_journey import _OBSERVE_WORKERS, wait_for_worker_disposal

if TYPE_CHECKING:
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        CDPSession,
        Locator,
        Page,
        Request,
        Route,
    )

_PANEL = "Reproducible workflows"
_RUN = "Run or resume workflow"
_STARTED = "window.__ownedKernel.started"
_TRAVERSAL_CHANGED = """previous => document.querySelector('[aria-label="Reproducible workflows"]')
  ?.getAttribute('data-workflow-traversal') !== previous"""
_KERNEL_ROUTE = "**/scpn_quantum_studio_wasm_kernel*.wasm*"
_BARRIER_ROUTE = "**/workflow-worker-startup-barrier"
_WORKER_ROUTE = "**/*kernelWorker*"
_CLAIM_BOUNDARY = (
    "Actual original classical WASM workflow and recorded source metadata; "
    "no quantum equivalence or physical device qualification"
)


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


@dataclass
class _Host:
    """One owned host's browser context, page, custody records and profiler."""

    context: BrowserContext
    target: str
    receipt: dict[str, object]
    observations: list[dict[str, object]]
    records: list[dict[str, object]]
    page: Page | None = None
    profiler: CDPSession | None = None
    errors: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    held: list[Route] = field(default_factory=list)
    held_entries: list[str] = field(default_factory=list)

    @property
    def owned(self) -> Page:
        """Return the opened page.

        Returns
        -------
        Page
            The owned page of this host.

        """
        assert self.page is not None, "the owned page is not open"
        return self.page

    def bound_request(self, route: Route) -> None:
        """Bound actual transport to this context's loopback origin.

        Parameters
        ----------
        route : Route
            Actual intercepted SDK route.

        """
        request = urlsplit(route.request.url)
        if request.scheme != "http" or request.netloc != urlsplit(self.target).netloc:
            self.refused.append(route.request.url)
            route.abort()
        else:
            route.continue_()

    def open(self, source: str | None) -> None:
        """Route the context, open the page and start its profiler on the source host.

        Parameters
        ----------
        source : str | None
            Source host whose native coverage is recorded, if any.

        """
        from tools.studio_workspace_browser_coverage import start_native_coverage

        self.context.route("**/*", self.bound_request)
        self.page = self.context.new_page()
        if self.target == source:
            self.profiler = start_native_coverage(self.page)
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))

    def take_coverage(self) -> CDPSession | None:
        """Take the running profiler's coverage and detach it from this host.

        Returns
        -------
        CDPSession | None
            The profiler session, or None when none was running.

        """
        from tools.studio_workspace_browser_coverage import take_native_coverage

        session, self.profiler = self.profiler, None
        if session is not None:
            take_native_coverage(session, self.records, include_workflows=True)
        return session


def _start_and_wait(page: Page, panel: Locator, previous: str | None) -> None:
    """Run or resume the workflow and wait until its traversal changes.

    Parameters
    ----------
    page : Page
        Owned workflow page.
    panel : Locator
        The workflow panel.
    previous : str | None
        Traversal identity before the run.

    """
    panel.get_by_role("button", name=_RUN, exact=True).click()
    page.wait_for_function(_TRAVERSAL_CHANGED, arg=previous, polling=100)


def _save_initial(host: _Host) -> dict[str, object]:
    """Open the Kuramoto sample and save its draft and revision references.

    Parameters
    ----------
    host : _Host
        The owned host.

    Returns
    -------
    dict[str, object]
        The initial saved archive.

    """
    from playwright.sync_api import expect

    page = host.owned
    page.goto(host.target + "#/experiments", wait_until="networkidle")
    page.get_by_role("region", name="Local experiment", exact=True).get_by_role(
        "button", name="Open Kuramoto sample", exact=True
    ).click()
    _navigate(page, "Workspace")
    page.get_by_role("button", name="Preview archive", exact=True).click()
    save = page.get_by_role("button", name="Save draft and revision references", exact=True)
    expect(save).to_be_enabled()
    save.click()
    workspace = page.get_by_role("region", name="Local workspace", exact=True)
    expect(workspace.get_by_role("status")).to_contain_text("transaction committed")
    initial = _archive(page)
    host.receipt["initial_native_workers"] = page.evaluate("window.__ownedKernel")
    return initial


def _compose(host: _Host) -> tuple[Locator, dict[str, object]]:
    """Compose the local workflow on the Run route and read its graph.

    Parameters
    ----------
    host : _Host
        The owned host.

    Returns
    -------
    tuple[Locator, dict[str, object]]
        The workflow panel and the composed graph.

    """
    from playwright.sync_api import expect

    page = host.owned
    _navigate(page, "Run")
    panel = page.get_by_role("region", name=_PANEL, exact=True)
    panel.get_by_role("button", name="Compose local workflow", exact=True).click()
    dependencies = panel.get_by_role("list", name="Original workflow dependency graph", exact=True)
    expect(dependencies).to_be_visible()
    text = panel.get_by_role("textbox", name="Workflow JSON", exact=True).input_value()
    return panel, cast(dict[str, object], read_json(text))


def _invalid_graph(graph: dict[str, object], fault: str) -> dict[str, object]:
    """Copy the graph with a dependency cycle or an incompatible port unit.

    Parameters
    ----------
    graph : dict[str, object]
        Composed graph, left unchanged.
    fault : str
        ``cycle`` or ``port``.

    Returns
    -------
    dict[str, object]
        The invalid copy.

    """
    invalid = cast(dict[str, object], read_json(write_json(graph)))
    stages = cast(list[dict[str, object]], cast(dict[str, object], invalid["body"])["stages"])
    if fault == "cycle":
        stages[0]["depends_on"] = ["analyse"]
    else:
        binding = cast(list[dict[str, object]], stages[2]["inputs"])[0]
        cast(dict[str, object], binding["type"])["unit"] = "rad"
    return invalid


def _refuse_invalid_graphs(
    host: _Host, panel: Locator, graph: dict[str, object], initial: dict[str, object]
) -> None:
    """Refuse a cycle and a port-unit mismatch without a worker or a saved change.

    Parameters
    ----------
    host : _Host
        The owned host.
    panel : Locator
        The workflow panel.
    graph : dict[str, object]
        Composed graph.
    initial : dict[str, object]
        Initial saved archive, which must stay unchanged.

    """
    from playwright.sync_api import expect

    page = host.owned
    for fault in ("cycle", "port"):
        wait_for_worker_disposal(page)
        before_workers = page.evaluate(_STARTED)
        _graph(page, _invalid_graph(graph, fault), save=False)
        host.receipt["current_fault"] = fault
        host.receipt["current_native_workers"] = page.evaluate("window.__ownedKernel")
        editor = panel.get_by_role("region", name="Workflow graph editor", exact=True)
        save = editor.get_by_role("button", name="Save workflow graph", exact=True)
        expect(save).to_be_disabled()
        assert page.evaluate(_STARTED) == before_workers
        _navigate(page, "Workspace")
        assert _archive(page) == initial
        _navigate(page, "Run")
        host.observations.append(
            {
                "host": host.target,
                "case": fault,
                "native_workers": 0,
                "original_source_unchanged": True,
            }
        )


def _sweep_six_cells(graph: dict[str, object]) -> None:
    """Shorten the first two stages and sweep two couplings by three time steps.

    Parameters
    ----------
    graph : dict[str, object]
        Composed graph, changed in place.

    """
    body = cast(dict[str, object], graph["body"])
    for stage in cast(list[dict[str, object]], body["stages"])[:2]:
        stage["parameters"] = {"steps": 4}
    sweep = cast(dict[str, object], body["sweep"])
    sweep["axes"] = [
        {"stage_id": "simulate", "parameter": "coupling", "values": [1.2, 1.4]},
        {"stage_id": "simulate", "parameter": "dt", "values": [0.01, 0.02, 0.04]},
    ]
    sweep["evaluation_budget"] = 18


def _check_export(page: Page, panel: Locator, graph: dict[str, object]) -> None:
    """Save the graph and find its exported JSON identical.

    Parameters
    ----------
    page : Page
        Owned workflow page.
    panel : Locator
        The workflow panel.
    graph : dict[str, object]
        Graph to save and export.

    """
    _graph(page, graph)
    with page.expect_download() as transfer:
        panel.get_by_role("button", name="Export workflow JSON", exact=True).click()
    with TemporaryDirectory(prefix="studio-workflow-export-") as directory:
        exported = Path(directory) / "experiment-workflow.json"
        transfer.value.save_as(exported)
        assert read_json(exported.read_text(encoding="utf-8")) == graph


def _refuse_kernel_transport(host: _Host, panel: Locator) -> list[str]:
    """Answer the kernel request with 503 and find the run refused without a worker.

    Parameters
    ----------
    host : _Host
        The owned host.
    panel : Locator
        The workflow panel.

    Returns
    -------
    list[str]
        The refused kernel request URLs.

    """
    from playwright.sync_api import expect

    page = host.owned
    before_failure = page.evaluate(_STARTED)
    refused: list[str] = []

    def refuse_kernel(route: Route) -> None:
        refused.append(route.request.url)
        route.fulfill(status=503, body="Original owned kernel transport refused")

    host.context.route(_KERNEL_ROUTE, refuse_kernel)
    try:
        panel.get_by_role("button", name=_RUN, exact=True).click()
        message = "Original workflow operation refused; prior evidence retained"
        expect(panel.get_by_text(message, exact=True)).to_be_visible()
        assert refused
        assert page.evaluate(_STARTED) == before_failure
    finally:
        host.context.unroute(_KERNEL_ROUTE, refuse_kernel)
    return refused


def _run_six_cells(
    host: _Host, panel: Locator, initial: dict[str, object], refused_kernel: list[str]
) -> tuple[dict[str, object], dict[str, object]]:
    """Run the eighteen evaluations over six cells and check the saved journal.

    Parameters
    ----------
    host : _Host
        The owned host.
    panel : Locator
        The workflow panel.
    initial : dict[str, object]
        Initial saved archive, whose members must all be kept.
    refused_kernel : list[str]
        Kernel requests refused before this run.

    Returns
    -------
    tuple[dict[str, object], dict[str, object]]
        The completed journal and the completed archive.

    """
    page = host.owned
    wait_for_worker_disposal(page)
    before_run = page.evaluate(_STARTED)
    _start_and_wait(page, panel, panel.get_attribute("data-workflow-traversal"))
    _wait_for_entries(page, ["complete"] * 18, "complete")
    wait_for_worker_disposal(page)
    assert page.evaluate(_STARTED) - before_run == 6
    journal, completed = _completed_journal(page, initial)
    host.observations.append(
        {
            "host": host.target,
            "case": "six_cells",
            "journal": journal,
            "native_workers": 6,
            "input_shape": {"oscillators": 12, "steps": 4},
            "exported_graph_matches": True,
            "kernel_transport_refusal_allocated_no_worker": True,
            "kernel_transport_refused_requests": refused_kernel,
        }
    )
    return journal, completed


def _completed_journal(
    page: Page, initial: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Read the saved six-cell journal and keep every initial archive member.

    Parameters
    ----------
    page : Page
        Owned workflow page.
    initial : dict[str, object]
        Initial saved archive.

    Returns
    -------
    tuple[dict[str, object], dict[str, object]]
        The completed journal and the completed archive.

    """
    _navigate(page, "Workspace")
    completed = _archive(page)
    journal = _selected_journal(completed)
    entries = cast(list[dict[str, object]], journal["entries"])
    assert journal["evaluations"] == 18 and len(entries) == 18
    assert len({entry["cell_id"] for entry in entries}) == 6
    initial_members = cast(list[dict[str, object]], initial["members"])
    final_members = cast(list[dict[str, object]], completed["members"])
    assert all(member in final_members for member in initial_members)
    return journal, completed


def _resume_after_reload(
    host: _Host, journal: dict[str, object], completed: dict[str, object]
) -> Locator:
    """Reload, resume the completed workflow and find no new worker and the same journal.

    Parameters
    ----------
    host : _Host
        The owned host.
    journal : dict[str, object]
        Completed journal.
    completed : dict[str, object]
        Completed archive.

    Returns
    -------
    Locator
        The workflow panel after the reload.

    """
    from playwright.sync_api import expect

    page = host.owned
    page.reload(wait_until="networkidle")
    _navigate(page, "Run")
    panel = page.get_by_role("region", name=_PANEL, exact=True)
    expect(panel.get_by_role("button", name=_RUN, exact=True)).to_be_enabled()
    wait_for_worker_disposal(page)
    before_resume = page.evaluate(_STARTED)
    previous = panel.get_attribute("data-workflow-traversal")
    host.receipt["before_resume_traversal"] = previous
    host.receipt["before_resume_source"] = panel.evaluate(
        "element => ({traversal:element.getAttribute('data-workflow-traversal'),"
        "settled:element.getAttribute('data-workflow-settled')})"
    )
    _start_and_wait(page, panel, previous)
    _wait_for_entries(page, ["complete"] * 18, "complete")
    assert page.evaluate(_STARTED) == before_resume
    _navigate(page, "Workspace")
    resumed = _archive(page)
    assert _selected_journal(resumed) == journal
    assert resumed["members"] == completed["members"]
    host.observations.append({"host": host.target, "case": "restart", "native_workers": 0})
    return panel


def _fail_parent(host: _Host, panel: Locator, graph: dict[str, object]) -> dict[str, object]:
    """Run a graph whose first stage fails and find every child blocked, with no worker.

    Parameters
    ----------
    host : _Host
        The owned host.
    panel : Locator
        The workflow panel.
    graph : dict[str, object]
        Six-cell graph, left unchanged.

    Returns
    -------
    dict[str, object]
        The failing-parent graph.

    """
    page = host.owned
    _navigate(page, "Run")
    failing = cast(dict[str, object], read_json(write_json(graph)))
    body = cast(dict[str, object], failing["body"])
    body["workflow_id"] = "failed-native-parent"
    cast(list[dict[str, object]], body["stages"])[0]["parameters"] = {"steps": 0}
    _graph(page, failing)
    wait_for_worker_disposal(page)
    before_failed = page.evaluate(_STARTED)
    _start_and_wait(page, panel, panel.get_attribute("data-workflow-traversal"))
    _wait_for_entries(page, ["failed", "blocked", "blocked"] * 6, "partial")
    assert page.evaluate(_STARTED) == before_failed
    _navigate(page, "Workspace")
    failed = _selected_journal(_archive(page))
    statuses = [entry["status"] for entry in cast(list[dict[str, object]], failed["entries"])]
    assert statuses == ["failed", "blocked", "blocked"] * 6
    host.observations.append({"host": host.target, "case": "failed_parent", "journal": failed})
    return failing


def _hold_worker_entry(host: _Host, route: Route) -> None:
    """Delay the worker entry behind a startup barrier fetch; pass its dependencies.

    Parameters
    ----------
    host : _Host
        The owned host.
    route : Route
        Intercepted worker-related request.

    """
    request = urlsplit(route.request.url)
    if request.scheme != "http" or request.netloc != urlsplit(host.target).netloc:
        host.refused.append(route.request.url)
        route.abort()
    elif Path(request.path).name != "kernelWorker.ts" and not Path(request.path).name.startswith(
        "kernelWorker-"
    ):
        route.continue_()
    else:
        host.held_entries.append(route.request.url)
        response = route.fetch()
        route.fulfill(
            response=response,
            body="await fetch('/workflow-worker-startup-barrier');\n" + response.text(),
        )


def _cancellation_graph(failing: dict[str, object]) -> dict[str, object]:
    """Copy the failing graph as a three-evaluation workflow with no sweep axis.

    Parameters
    ----------
    failing : dict[str, object]
        Failing-parent graph, left unchanged.

    Returns
    -------
    dict[str, object]
        The cancellation graph.

    """
    cancellation = cast(dict[str, object], read_json(write_json(failing)))
    body = cast(dict[str, object], cancellation["body"])
    body["workflow_id"] = "cancel-original-workflow"
    cast(list[dict[str, object]], body["stages"])[0]["parameters"] = {}
    cast(dict[str, object], body["sweep"]).update(axes=[], evaluation_budget=3)
    return cancellation


def _start_held_run(host: _Host, panel: Locator) -> tuple[str, str, list[bool]]:
    """Start a run whose worker waits on a held startup fetch, then cancel it.

    Parameters
    ----------
    host : _Host
        The owned host.
    panel : Locator
        The workflow panel.

    Returns
    -------
    tuple[str, str, list[bool]]
        The worker URL, the held startup request URL and the worker-close record.

    """
    page = host.owned
    closed: list[bool] = []

    def hold_entry(route: Route) -> None:
        _hold_worker_entry(host, route)

    def hold_startup(route: Route) -> None:
        host.held.append(route)

    host.context.route(_BARRIER_ROUTE, hold_startup)
    page.route(_WORKER_ROUTE, hold_entry)
    before_cancel = page.evaluate(_STARTED)

    def is_barrier(request: Request) -> bool:
        return urlsplit(request.url).path == "/workflow-worker-startup-barrier"

    with host.context.expect_event("request", predicate=is_barrier) as startup:
        with page.expect_worker() as created:
            panel.get_by_role("button", name=_RUN, exact=True).click()
        created.value.once("close", lambda _worker: closed.append(True))
    panel.get_by_role("button", name="Cancel workflow", exact=True).click()
    _wait_for_entries(page, ["complete", "cancelled"], "cancelled")
    wait_for_worker_disposal(page)
    assert closed == [True]
    assert len(host.held_entries) == 1
    assert len(host.held) == 1
    assert startup.value.url == host.held[0].request.url
    assert page.evaluate(_STARTED) - before_cancel == 1
    page.unroute(_WORKER_ROUTE, hold_entry)
    host.context.unroute(_BARRIER_ROUTE, hold_startup)
    return created.value.url, startup.value.url, closed


def _cancel_held_run(host: _Host, panel: Locator, failing: dict[str, object]) -> None:
    """Cancel a run during worker startup and find one disposed, cancelled attempt saved.

    Parameters
    ----------
    host : _Host
        The owned host.
    panel : Locator
        The workflow panel.
    failing : dict[str, object]
        Failing-parent graph the cancellation graph derives from.

    """
    from playwright.sync_api import Error

    page = host.owned
    _navigate(page, "Run")
    _graph(page, _cancellation_graph(failing))
    worker_url, startup_url, closed = _start_held_run(host, panel)
    for route in host.held:
        # Chromium may already have closed the request with its worker.
        with suppress(Error):
            route.abort()
    _navigate(page, "Workspace")
    cancelled = _selected_journal(_archive(page))
    assert cancelled["state"] == "cancelled"
    entries = cast(list[dict[str, object]], cancelled["entries"])
    assert [entry["status"] for entry in entries] == ["complete", "cancelled"]
    assert cast(dict[str, object], entries[-1]["output"])["disposed"] is True
    _navigate(page, "Run")
    host.observations.append(
        {
            "host": host.target,
            "case": "cancelled",
            "journal": cancelled,
            "native_workers": 1,
            "native_close_observed": closed == [True],
            "worker_fault": "original script startup fetch held",
            "worker_url": worker_url,
            "held_worker_entries": host.held_entries,
            "held_startup_request": startup_url,
        }
    )


def _view_refusals(host: _Host) -> None:
    """Record the source workspace view's native workflow refusals in a second page.

    Parameters
    ----------
    host : _Host
        The owned source host.

    """
    from playwright.sync_api import expect

    from tools.studio_workspace_browser_coverage import start_native_coverage, take_native_coverage

    view_page = host.context.new_page()
    view_page.on("pageerror", lambda error: host.errors.append(str(error)))
    view_profiler = start_native_coverage(view_page)
    try:
        view_page.goto(host.target + "#/workspace", wait_until="networkidle")
        verified = view_page.locator(".qsp-play").get_by_role("status")
        expect(
            verified.filter(has_text="verified against the committed ground truth")
        ).to_be_visible()
        wait_for_worker_disposal(view_page)
        host.receipt["native_view_refusals"] = view_page.evaluate(
            "async () => (await import('/browser-tests/workflowRunner.native.tsx'))"
            ".runNativeWorkflowViewRefusals()"
        )
    finally:
        take_native_coverage(view_profiler, host.records, include_workflows=True)
        view_page.close()


def _walk(host: _Host, source: str | None) -> None:
    """Refuse invalid graphs, run, resume, fail a parent and cancel on one host.

    Parameters
    ----------
    host : _Host
        The owned host with its page open.
    source : str | None
        Source host whose native coverage is recorded, if any.

    """
    from tools.studio_workspace_browser_coverage import start_native_coverage

    initial = _save_initial(host)
    panel, graph = _compose(host)
    _refuse_invalid_graphs(host, panel, graph, initial)
    _sweep_six_cells(graph)
    _check_export(host.owned, panel, graph)
    refused_kernel = _refuse_kernel_transport(host, panel)
    journal, completed = _run_six_cells(host, panel, initial, refused_kernel)
    session = host.take_coverage()
    if session is not None:
        host.profiler = start_native_coverage(host.owned, session=session)
    panel = _resume_after_reload(host, journal, completed)
    failing = _fail_parent(host, panel, graph)
    _cancel_held_run(host, panel, failing)
    assert not host.errors and not host.refused
    if host.target == source:
        _view_refusals(host)
    assert not host.errors and not host.refused
    host.take_coverage()


def _record_failure(host: _Host, cause: BaseException) -> None:
    """Keep the failure, the custody records, any coverage and what the page shows.

    Parameters
    ----------
    host : _Host
        The failed host.
    cause : BaseException
        The failure.

    """
    from traceback import format_exception

    receipt = host.receipt
    receipt["original_error"] = f"{type(cause).__name__}: {cause}"
    receipt["original_traceback"] = "".join(format_exception(cause))
    receipt["failed_context"] = host.target
    receipt["page_errors"] = host.errors
    receipt["refused_requests"] = host.refused
    receipt["held_worker_entries"] = host.held_entries
    receipt["held_startup_requests"] = [route.request.url for route in host.held]
    try:
        host.take_coverage()
    except Exception as coverage_error:
        receipt["native_coverage_error"] = f"{type(coverage_error).__name__}: {coverage_error}"
    try:
        _record_page_state(host.owned, receipt)
    except Exception as capture_error:
        receipt["diagnostic_capture_error"] = f"{type(capture_error).__name__}: {capture_error}"


def _record_page_state(page: Page, receipt: dict[str, object]) -> None:
    """Record the failed page's text, workflow state and worker counts.

    Parameters
    ----------
    page : Page
        The failed host's page.
    receipt : dict[str, object]
        Caller-owned evidence receiving the diagnostics.

    """
    receipt["failed_page_text"] = page.locator("body").inner_text()
    receipt["resume_predicate_probe"] = page.evaluate(
        "previous => document.querySelector('[aria-label=\"Reproducible workflows\"]')"
        "?.getAttribute('data-workflow-traversal') !== previous",
        receipt.get("before_resume_traversal"),
    )
    receipt["failed_workflow_state"] = page.get_by_role(
        "region", name=_PANEL, exact=True
    ).evaluate(
        "element => ({traversal: element.getAttribute('data-workflow-traversal'), "
        "settled: element.getAttribute('data-workflow-settled'), progress: "
        "element.querySelector('[aria-label=\"Workflow stage progress\"]')?.textContent, "
        "runDisabled: Array.from(element.querySelectorAll('button'))"
        ".find(button => button.textContent === 'Run or resume workflow')?.disabled})"
    )
    receipt["failed_native_worker_counts"] = {
        name: page.evaluate("window.__ownedKernel." + name)
        for name in ("started", "disposed", "active")
    }


def _workflow_on(
    browser: Browser,
    target: str,
    source: str | None,
    receipt: dict[str, object],
    collected: tuple[list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """Run the whole workflow acceptance on one owned host in its own context.

    Parameters
    ----------
    browser : Browser
        Launched browser.
    target : str
        Owned host URL.
    source : str | None
        Source host whose native coverage is recorded, if any.
    receipt : dict[str, object]
        Caller-owned evidence.
    collected : tuple
        The observations list and the native coverage records.

    """
    context = browser.new_context(service_workers="block")
    context.set_default_timeout(15_000)
    context.add_init_script(_OBSERVE_WORKERS)
    host = _Host(context, target, receipt, *collected)
    try:
        host.open(source)
        _walk(host, source)
    except Exception as cause:
        _record_failure(host, cause)
        raise
    finally:
        context.close()


def _owned_hosts(base_url: str, source_url: str | None) -> tuple[str, ...]:
    """Validate the built host and the optional distinct root source host.

    Parameters
    ----------
    base_url : str
        Owned literal loopback root serving the current built original frontend.
    source_url : str | None
        Optional distinct owned loopback root serving the current original source.

    Returns
    -------
    tuple[str, ...]
        The built host, then the source host when one is supplied.

    Raises
    ------
    ValueError
        A target is external, ambiguous, shared or a nested source path.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    source = None if source_url is None else loopback_url(source_url)
    if source is None:
        return (url,)
    if urlsplit(source).path != "/" or urlsplit(source).netloc == urlsplit(url).netloc:
        raise ValueError("Workflow source requires a distinct owned root source host")
    return (url, source)


def _start_receipt(
    receipt: dict[str, object], source: str | None
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Record the scenario and, with a source host, its native coverage destination.

    Parameters
    ----------
    receipt : dict[str, object]
        Caller-owned evidence.
    source : str | None
        Source host whose native coverage is recorded, if any.

    Returns
    -------
    tuple
        The observations list and the native coverage records.

    """
    observations: list[dict[str, object]] = []
    receipt.update(scenario="experiment_workflow_runner", observations=observations)
    records: list[dict[str, object]] = []
    if source is not None:
        receipt.update(
            source_url=source, native_v8_coverage=records, coverage_percentage="not_calculated"
        )
    return observations, records


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
                _workflow_on(browser, target, source, receipt, collected)
        finally:
            browser.close()
    receipt.update(status="passed", claim_boundary=_CLAIM_BOUNDARY)
    return receipt
