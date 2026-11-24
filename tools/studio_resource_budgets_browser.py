# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio resource budgets measured in a real browser
"""Measure the declared Studio resource budgets on the built preview and record every verdict.

The journey measures, in the browser that runs it:

- the cold load (largest contentful paint, cumulative layout shift and total
  blocking time);
- the compressed initial JavaScript the served entry page loads;
- the visible response to a graph selection;
- each graph tier's admission, the bounded neighbourhood query at the largest
  tier and the cancellation of its admission;
- the workers and JavaScript heap left after repeated mounting and unmounting;
- a 100 000-point trajectory opened as a producer file in the Results view:
  the time to admit and draw it, the live JavaScript heap it adds and whether
  the chart keeps every spike when it decimates.

The record judges each measurement against its declared budget
(:func:`tools.studio_resource_budgets.judge_budgets`). A missed budget is
recorded as a failure and never retried under a relaxed threshold; the journey
itself fails only when a measurement cannot be made.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from tools.studio_owned_worker_journey import wait_for_worker_disposal
from tools.studio_preview_custody import run_custodied_journey
from tools.studio_resource_budgets import (
    FROZEN_TIER_DIGESTS,
    TIERS,
    initial_javascript_size,
    judge_budgets,
    load_budgets,
    tier_pin_text,
)

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

#: Each tier's pin text, built once per process.
_pin_text = functools.cache(tier_pin_text)

#: Timed repetitions behind each percentile.
SAMPLES: Final[int] = 20
#: Mount and unmount cycles of the graph.
CYCLES: Final[int] = 5
#: Points of the trajectory the result inspector must admit and decimate.
TRAJECTORY_POINTS: Final[int] = 100_000
#: Samples set far outside the series, which decimation must keep as markers.
TRAJECTORY_SPIKES: Final[dict[int, float]] = {1: 50.0, 40_001: -50.0}
#: Title of the synthetic trajectory panel, which names its chart.
TRAJECTORY_TITLE: Final[str] = "Synthetic budget trajectory"
#: Markers the result inspector draws at most per panel (``resultLimits.displayPoints``).
DISPLAY_MARKERS: Final[int] = 1000
#: Window after navigation start in which long tasks count towards blocking time.
TBT_WINDOW_MS: Final[float] = 5000.0
#: Where each measurement that is not taken in the page comes from.
PROVENANCE: Final[dict[str, str]] = {
    "initial_javascript": "gzip level 9 of the entry script and module preloads the preview serves",
    "cold_load_tbt": "long tasks over 50 ms after first contentful paint, within 5 s of navigation",
}

_OBSERVE: Final[str] = """(() => {
  window.__budget = { lcp: null, cls: 0, longTasks: [] };
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) window.__budget.lcp = entry.startTime;
  }).observe({ type: 'largest-contentful-paint', buffered: true });
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) if (!entry.hadRecentInput) window.__budget.cls += entry.value;
  }).observe({ type: 'layout-shift', buffered: true });
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) window.__budget.longTasks.push([entry.startTime, entry.duration]);
  }).observe({ type: 'longtask', buffered: true });
})();"""
_FRAME: Final[
    str
] = """const frame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));
  const until = async (holds, what) => {
    const deadline = performance.now() + 120000;
    while (!holds()) {
      if (performance.now() > deadline) {
        const state = [...document.querySelectorAll('[aria-label="Atlas summary"], [aria-label="Atlas progress"], [role="alert"]')]
          .map((n) => n.textContent).join(' | ');
        throw new Error(`timed out waiting for ${what}; the page shows: ${state}`);
      }
      await frame();
    }
  };"""
_SELECT_TIMED: Final[str] = f"""async (graph, index) => {{
  {_FRAME}
  const buttons = [...graph.querySelectorAll('button')];
  const target = buttons[index % buttons.length];
  const started = performance.now();
  target.click();
  await until(() => target.getAttribute('aria-pressed') === 'true', 'the selection');
  await frame();
  return performance.now() - started;
}}"""
_NEIGHBOURHOOD_TIMED: Final[str] = f"""async (atlas, row) => {{
  {_FRAME}
  const buttons = () => [...atlas.querySelectorAll('button')];
  const leave = () => buttons().find((b) => b.textContent.startsWith('Leave the neighbourhood of'));
  atlas.querySelectorAll('table[aria-label="Atlas nodes"] tbody tr button')[row].click();
  const show = () => buttons().find((b) => b.textContent === 'Show its neighbourhood');
  await until(show, 'the neighbourhood button');
  const summary = atlas.querySelector('[aria-label="Atlas summary"]');
  const before = summary.textContent;
  const started = performance.now();
  show().click();
  await until(() => leave() && summary.textContent !== before, 'the neighbourhood');
  const elapsed = performance.now() - started;
  leave().click();
  await until(() => summary.textContent === before, 'the full view');
  return elapsed;
}}"""
_CANCEL_TIMED: Final[str] = f"""async (atlas) => {{
  {_FRAME}
  const cancel = () => [...atlas.querySelectorAll('[aria-label="Atlas progress"] button')]
    .find((b) => b.textContent === 'Cancel');
  await until(cancel, 'the cancel button');
  const started = performance.now();
  cancel().click();
  await until(() => (atlas.querySelector('[role="alert"]')?.textContent ?? '').includes('was cancelled'),
    'the cancellation');
  return performance.now() - started;
}}"""


def percentile95(samples: list[float]) -> float:
    """Return the nearest-rank 95th percentile.

    Parameters
    ----------
    samples
        At least one measurement.

    Returns
    -------
    float
        The value at rank ``ceil(0.95 n)``.

    """
    ordered = sorted(samples)
    return ordered[-(-95 * len(ordered) // 100) - 1]


class _Evaluates(Protocol):
    """Anything that runs a script against one page element, as a locator does."""

    def evaluate(self, expression: str, arg: object = None, /) -> object:
        """Run ``expression`` with the element and ``arg``."""


def _timed(target: _Evaluates, script: str, argument: int | None = None) -> float:
    """Run one in-page timing and report a wait that never ends as a failed measurement.

    Parameters
    ----------
    target
        The element the script receives.
    script
        An async function returning milliseconds.
    argument
        Its second argument, if any.

    Returns
    -------
    float
        The measured milliseconds.

    Raises
    ------
    AssertionError
        The page never reached the awaited state, with what it showed instead.

    """
    from playwright.sync_api import Error

    try:
        measured = target.evaluate(script, argument)
    except Error as error:
        raise AssertionError(f"a measurement could not be made: {error.message}") from error
    assert isinstance(measured, int | float), measured
    return float(measured)


def host_environment() -> dict[str, object]:
    """Return the CPU, core count, memory and load of the host running the browser.

    Returns
    -------
    dict[str, object]
        Values read from ``/proc`` and :mod:`os`; ``None`` where the host does not say.

    """

    def first(path: str, pattern: str) -> str | None:
        try:
            match = re.search(pattern, Path(path).read_text(encoding="utf-8"), re.MULTILINE)
        except OSError:
            return None
        return None if match is None else match.group(1).strip()

    return {
        "cpu": first("/proc/cpuinfo", r"^model name\s*:\s*(.+)$"),
        "cores": os.cpu_count(),
        "memory_kib": first("/proc/meminfo", r"^MemTotal:\s*(\d+)"),
        "load_average": list(os.getloadavg()),
    }


def _served(page: Page, url: str, relative: str) -> bytes | None:
    """Read one file the owned preview serves through the page's own request context.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    relative
        Bundle-relative path.

    Returns
    -------
    bytes | None
        The served bytes, or ``None`` when the preview serves none.

    """
    response = page.request.get(url + relative)
    return response.body() if response.ok else None


def _cold_load(page: Page, url: str) -> dict[str, object]:
    """Load the workbench cold and read its paint, shift and blocking time.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.

    Returns
    -------
    dict[str, object]
        ``cold_load_lcp`` and ``cold_load_tbt`` in milliseconds and ``cold_load_cls``.

    """
    page.add_init_script(_OBSERVE)
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(1000)
    paint = page.evaluate(
        "() => (performance.getEntriesByName('first-contentful-paint')[0] || {startTime: 0}).startTime"
    )
    observed = page.evaluate("() => window.__budget")
    blocking = sum(
        max(0.0, float(duration) - 50.0)
        for start, duration in observed["longTasks"]
        if float(paint) <= float(start) < TBT_WINDOW_MS
    )
    assert observed["lcp"] is not None, "no largest contentful paint was reported"
    return {
        "cold_load_lcp": float(observed["lcp"]),
        "cold_load_cls": float(observed["cls"]),
        "cold_load_tbt": blocking,
    }


def _open_atlas(page: Page) -> Locator:
    """Open the Atlas stage and wait for its committed snapshot.

    Parameters
    ----------
    page
        Owned page bound to the preview.

    Returns
    -------
    Locator
        The Atlas stage.

    """
    from playwright.sync_api import expect

    page.get_by_role("navigation", name="Workbench context destinations").get_by_role(
        "link", name="Atlas", exact=True
    ).click()
    atlas = page.get_by_role("article", name="Atlas")
    expect(atlas.get_by_role("status", name="Atlas summary")).to_contain_text(
        "Showing", timeout=60000
    )
    return atlas


def _open_tier(atlas: Locator, nodes: int, relations: int) -> float:
    """Open one generated tier and return its admission time in milliseconds.

    Parameters
    ----------
    atlas
        The Atlas stage.
    nodes
        Node count of the tier.
    relations
        Relation count of the tier.

    Returns
    -------
    float
        Wall time from choosing the file to the tier's summary.

    """
    from playwright.sync_api import expect

    with tempfile.TemporaryDirectory() as scratch:
        pin = Path(scratch) / f"tier_{nodes}_{relations}.json"
        pin.write_text(_pin_text(nodes, relations), encoding="utf-8")
        started = time.perf_counter()
        atlas.get_by_label("Open a snapshot pin").set_input_files(str(pin))
        digest = FROZEN_TIER_DIGESTS[TIERS.index((nodes, relations))]
        # Two tiers have the same node count, so the tier is known by its snapshot digest.
        expect(atlas.get_by_text(re.compile(rf"^Snapshot {digest[:12]} · "))).to_be_visible(
            timeout=180000
        )
        expect(atlas.get_by_role("status", name="Atlas summary")).to_have_text(
            re.compile(rf"^Showing \d+ of {nodes} nodes"), timeout=180000
        )
    return (time.perf_counter() - started) * 1000.0


def _largest_tier(page: Page, atlas: Locator) -> tuple[dict[str, object], dict[str, object]]:
    """Measure the largest tier: cancellation, admission, drawn bounds and the neighbourhood query.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    atlas
        The Atlas stage.

    Returns
    -------
    tuple[dict[str, object], dict[str, object]]
        The budget measurements and their samples.

    """
    nodes, relations = TIERS[-1]
    with tempfile.TemporaryDirectory() as scratch:
        pin = Path(scratch) / "largest_tier.json"
        pin.write_text(_pin_text(nodes, relations), encoding="utf-8")
        atlas.get_by_label("Open a snapshot pin").set_input_files(str(pin))
        cancel_ms = _timed(atlas, _CANCEL_TIMED)
    opened_at = float(page.evaluate("performance.now()"))
    admission_ms = _open_tier(atlas, nodes, relations)
    summary = atlas.get_by_role("status", name="Atlas summary").inner_text()
    shown = re.match(r"^Showing (\d+) of", summary)
    assert shown is not None, summary
    drawn_edges = int(atlas.evaluate("(root) => root.querySelectorAll('line[data-edge]').length"))
    samples = [_timed(atlas, _NEIGHBOURHOOD_TIMED, row) for row in range(SAMPLES)]
    long_tasks = [
        float(duration)
        for start, duration in page.evaluate("() => window.__budget.longTasks.slice()")
        if float(start) >= opened_at
    ]
    measured: dict[str, object] = {
        "worker_cancel_ack": cancel_ms,
        "graph_tiers": float(nodes),
        "visible_subgraph_nodes": float(shown.group(1)),
        "visible_subgraph_edges": float(drawn_edges),
        "graph_query_p95": percentile95(samples),
    }
    detail: dict[str, object] = {
        "largest_admission_ms": admission_ms,
        "neighbourhood_samples_ms": samples,
        "long_tasks_ms": long_tasks,
    }
    return measured, detail


def _cycles(page: Page) -> dict[str, object]:
    """Mount and unmount the graph repeatedly and read the workers and heap left.

    Parameters
    ----------
    page
        Owned page bound to the preview.

    Returns
    -------
    dict[str, object]
        ``worker_leak``, and ``retained_memory`` or ``None`` where the browser reports no heap.

    """
    heap = "() => performance.memory ? performance.memory.usedJSHeapSize : null"
    views = page.get_by_role("navigation", name="Workbench views")
    before = page.evaluate(heap)
    for _cycle in range(CYCLES):
        views.get_by_role("link", name="Workspace", exact=True).click()
        wait_for_worker_disposal(page, observe_termination=False)
        _open_atlas(page)
    views.get_by_role("link", name="Workspace", exact=True).click()
    wait_for_worker_disposal(page, observe_termination=False)
    after = page.evaluate(heap)
    # A browser without a heap reading leaves retained memory unmeasured, never zero.
    retained = None if before is None or after is None else float(after) - float(before)
    return {"worker_leak": float(len(page.workers)), "retained_memory": retained}


def trajectory_export(points: int = TRAJECTORY_POINTS) -> str:
    """Return a real CLI analyse export whose inspection is one synthetic trajectory.

    The request, plan and claim boundary are those of an actual ``analyse`` run; only the
    inspection is replaced, by a deterministic float64 series ``sin(i / 500)`` with the
    :data:`TRAJECTORY_SPIKES` set far outside it. It is written as the CLI writes, with
    sorted keys and two-space indentation.

    Parameters
    ----------
    points
        Samples of the series; a spike beyond the last sample is left out.

    Returns
    -------
    str
        The export text.

    """
    from tools.studio_result_browser import analyse_export

    record = json.loads(analyse_export())
    inspection = record["result"]["outputs"]["inspection"]
    template = inspection["panels"][0]
    values = [math.sin(index / 500) for index in range(points)]
    for index, value in TRAJECTORY_SPIKES.items():
        if index < points:
            values[index] = value
    inspection["title"] = TRAJECTORY_TITLE
    inspection["caption"] = (
        "Deterministic synthetic series of the Studio resource budget, not a measurement or a "
        "model output. Two spikes test that decimation keeps extrema."
    )
    inspection["panels"] = [
        {
            **template,
            "id": "trajectory",
            "title": TRAJECTORY_TITLE,
            "valueLabel": "Value",
            "valueDtype": "float64",
            "samples": [
                {
                    "coordinate": index * 0.01,
                    "objectId": "trajectory",
                    "columnId": None,
                    "x": index * 0.01,
                    "y": None,
                    "value": value,
                    "status": "finite",
                    "interval": None,
                }
                for index, value in enumerate(values)
            ],
        }
    ]
    return json.dumps(record, sort_keys=True, indent=2)


def _heap(page: Page) -> float:
    """Collect garbage and return the page's used JavaScript heap, from the DevTools protocol.

    Parameters
    ----------
    page
        Owned Chromium page.

    Returns
    -------
    float
        Used heap bytes after a full collection, so only live objects count.

    """
    session = page.context.new_cdp_session(page)
    try:
        session.send("HeapProfiler.collectGarbage")
        return float(session.send("Runtime.getHeapUsage")["usedSize"])
    finally:
        session.detach()


def measure_trajectory(page: Page) -> tuple[dict[str, object], dict[str, object]]:
    """Open the synthetic trajectory in the Results view and check its decimated chart.

    The export is opened as a file, as a user would at this size: pasted, the text area alone
    takes minutes to lay out.

    Parameters
    ----------
    page
        Owned page bound to the preview.

    Returns
    -------
    tuple[dict[str, object], dict[str, object]]
        ``trajectory_points`` (the points admitted and drawn with every spike kept, else 0),
        and the export's bytes, the time to admit and draw it, the live heap it adds and the
        drawn markers.

    """
    from playwright.sync_api import expect

    text = trajectory_export()
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name="Analyse", exact=True
    ).click()
    loader = page.get_by_role("region", name="Source result inspector", exact=True)
    status = loader.get_by_role("status", name="Result import status")
    before = _heap(page)
    with tempfile.TemporaryDirectory() as directory:
        export = Path(directory) / "trajectory.json"
        export.write_text(text, encoding="utf-8")
        started = time.perf_counter()
        loader.get_by_label("Open a producer JSON file").set_input_files(export, timeout=600000)
        expect(status).not_to_have_text("", timeout=600000)
    detail: dict[str, object] = {"bytes": len(text.encode("utf-8")), "status": status.inner_text()}
    drawn: list[int] = []
    if detail["status"] == "Original producer metadata admitted; saved workspace unchanged.":
        chart = loader.get_by_role("img", name=f"{TRAJECTORY_TITLE} chart")
        expect(chart.locator("[data-index]").first).to_be_attached(timeout=600000)
        drawn = chart.locator("[data-index]").evaluate_all(
            "nodes => nodes.map(node => Number(node.dataset.index))"
        )
    detail["admission_ms"] = (time.perf_counter() - started) * 1000.0
    # Decimated means within the marker budget, with both ends and every spike still drawn.
    kept = len(drawn) <= DISPLAY_MARKERS and {0, TRAJECTORY_POINTS - 1, *TRAJECTORY_SPIKES} <= set(
        drawn
    )
    detail |= {"markers": len(drawn), "extrema_kept": kept, "heap_bytes": _heap(page) - before}
    points = float(TRAJECTORY_POINTS) if kept else 0.0
    return {"trajectory_points": points}, {"trajectory": detail}


def _walk(
    page: Page, url: str, observations: list[str], *, evidence: dict[str, object]
) -> dict[str, object]:
    """Measure every budget the browser can measure and record the verdicts.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.
    evidence
        Journey evidence; receives the budget record as soon as it exists.

    Returns
    -------
    dict
        The budget record, with its details, provenance and host environment.

    """
    measured = _cold_load(page, url)
    observations.append("cold-load-measured")
    html = _served(page, url, "index.html")
    assert html is not None, "the preview serves no entry page"
    measured["initial_javascript"] = float(
        initial_javascript_size(html.decode("utf-8"), partial(_served, page, url))
    )
    atlas = _open_atlas(page)
    graph = atlas.get_by_role("region", name="Atlas graph")
    selections = [_timed(graph, _SELECT_TIMED, index) for index in range(SAMPLES)]
    measured["interaction_p95"] = percentile95(selections)
    observations.append("interaction-measured")
    admissions = {f"{n}/{r}": _open_tier(atlas, n, r) for n, r in TIERS[:-1]}
    largest, detail = _largest_tier(page, atlas)
    measured.update(largest)
    observations.append("graph-tiers-measured")
    measured.update(_cycles(page))
    observations.append("repeated-navigation-measured")
    trajectory, trajectory_detail = measure_trajectory(page)
    measured.update(trajectory)
    observations.append("trajectory-measured")
    record = judge_budgets(load_budgets(), measured)
    record["detail"] = {
        "tier_admission_ms": admissions,
        "interaction_samples_ms": selections,
        **detail,
        **trajectory_detail,
    }
    record["provenance"] = dict(PROVENANCE)
    record["environment"] = host_environment()
    evidence["budget_record"] = record
    return {"budget_record": record}


def run_resource_budgets_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Measure the Studio resource budgets through the built workbench.

    Parameters
    ----------
    base_url
        Owned literal HTTP loopback preview containing genuine current Studio and WASM.
    evidence
        Caller-owned observations retained after an actual runtime refusal.

    Returns
    -------
    dict[str, object]
        Browser identity, the budget record with every verdict and missed budget,
        and the custody record.

    Raises
    ------
    ValueError
        If the shared preview ownership contract refuses the URL.
    AssertionError
        If a measurement cannot be made or a runtime contract fails.

    """
    observed: dict[str, object] = {} if evidence is None else evidence
    walk = partial(_walk, evidence=observed)
    return run_custodied_journey(base_url, observed, "workbench_resource_budgets", walk)
