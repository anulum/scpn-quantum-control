# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Atlas graph navigation through the built Studio
"""Browse the Atlas in a real browser: the committed snapshot, then a large pin in a worker."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

from tools.studio_owned_worker_journey import wait_for_worker_disposal
from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

AXE_SOURCE: Final[Path] = (
    Path(__file__).resolve().parents[1] / "studio-web/node_modules/axe-core/axe.min.js"
)
#: Nodes of the generated large pin; the query owner's node ceiling.
LARGE_NODES: Final[int] = 10_000
#: Relations per node of the generated large pin, so its admission runs long enough to cancel.
LARGE_FAN_OUT: Final[int] = 2
#: Longest main-thread task, in milliseconds, admitted while the large pin is open.
LONG_TASK_BOUND_MS: Final[float] = 250.0
_LARGE_SUMMARY: Final = re.compile(r"^Showing 500 of 10000 nodes; 9500 more are not shown")
_CONTROLS: Final[str] = "button:visible, a:visible, input:visible, select:visible"
_OBSERVE_LONG_TASKS: Final[str] = """(() => {
  window.__atlasLongTasks = [];
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries())
      window.__atlasLongTasks.push([entry.startTime, entry.duration]);
  }).observe({ type: 'longtask', buffered: true });
})();"""


def large_pin(directory: Path, nodes: int = LARGE_NODES, fan_out: int = LARGE_FAN_OUT) -> Path:
    """Write a contract-valid snapshot pin of definitions that require their successors.

    Parameters
    ----------
    directory
        Caller-owned directory that receives ``large_atlas_pin.json``.
    nodes
        Number of definition nodes.
    fan_out
        Relations from each node: node ``i`` requires nodes ``i + 1`` to ``i + fan_out``,
        modulo ``nodes``.

    Returns
    -------
    Path
        The written pin, digest computed by the Python contract owner.

    """
    from scpn_quantum_control.atlas import (
        atlas_snapshot_digest,
        parse_atlas_snapshot,
        snapshot_to_wire,
    )
    from tools.studio_atlas_snapshot import PIN_SCHEMA

    unknown = {"source_digest": "generated", "validity_domain": "generated", "as_of": "generated"}
    name = "definition:journey/n{:05d}".format
    wire = {
        "schema": "quantum_atlas_snapshot.v1",
        "extractor_version": "browser-journey",
        "source_commit": "0" * 40,
        "evidence_set_hash": "a" * 64,
        "access_scope": "public",
        "nodes": [
            {
                "id": name(index),
                "kind": "definition",
                "label": f"Node n{index:05d}",
                "source_ref": f"journey/n{index:05d}",
                "source_digest": None,
                "validity_domain": None,
                "as_of": None,
                "access": "public",
                "status": "current",
                "evidence_refs": [],
                "null_reasons": unknown,
            }
            for index in range(nodes)
        ],
        "edges": [
            {
                "source": name(index),
                "kind": "requires",
                "target": name((index + step) % nodes),
                "relation_version": "atlas_relation.v1",
                "derivation_method": "declared",
                "source_ref": "journey/edges",
                "evidence_digest": None,
                "assumptions": [],
                "mapping": None,
                "as_of": None,
                "access": "public",
            }
            for index in range(nodes)
            for step in range(1, fan_out + 1)
        ],
        "tombstones": [],
        "exclusions": [],
    }
    snapshot = parse_atlas_snapshot(wire)
    pin = {
        "schema": PIN_SCHEMA,
        "snapshot_digest": atlas_snapshot_digest(snapshot),
        "snapshot": snapshot_to_wire(snapshot),
    }
    path = directory / "large_atlas_pin.json"
    path.write_text(json.dumps(pin), encoding="utf-8")
    return path


def _selection(atlas: Locator, observations: list[str]) -> None:
    """Select a graph node, see it in the table and the inspector, then move by key.

    Parameters
    ----------
    atlas
        The Atlas stage.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    graph = atlas.get_by_role("region", name="Atlas graph")
    first = graph.get_by_role("button").first
    label = first.inner_text()
    first.click()
    expect(first).to_have_attribute("aria-pressed", "true")
    selected_row = atlas.get_by_role("table", name="Atlas nodes").get_by_role("row", selected=True)
    expect(selected_row).to_contain_text(label)
    expect(atlas.get_by_role("region", name="Atlas inspector")).to_contain_text("Content digest")
    observations.append("graph-table-and-inspector-share-the-selection")
    first.press("ArrowDown")
    expect(graph.locator('button[aria-pressed="true"]')).not_to_have_text(label)
    observations.append("keyboard-moves-the-selection")


def _deep_link(page: Page, url: str, atlas: Locator, observations: list[str]) -> None:
    """Narrow the view, follow its link after a reload and find the same view.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    atlas
        The Atlas stage.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    summary = atlas.get_by_role("status", name="Atlas summary")
    whole = summary.inner_text()
    atlas.get_by_label("Search").fill("readme")
    expect(summary).not_to_have_text(whole)
    narrowed = summary.inner_text()
    href = atlas.get_by_role("link", name="Link to this view").get_attribute("href")
    assert href is not None and "&filter=q%3Dreadme" in href, href
    page.goto(url + href, wait_until="networkidle")
    page.reload(wait_until="networkidle")
    expect(atlas.get_by_label("Search")).to_have_value("readme")
    expect(summary).to_have_text(narrowed)
    observations.append("deep-link-restores-snapshot-and-filter")
    atlas.get_by_label("Search").fill("")
    expect(summary).to_have_text(whole)


def _zoom(atlas: Locator, observations: list[str]) -> None:
    """Zoom from kind groups to node details and back.

    Parameters
    ----------
    atlas
        The Atlas stage.
    observations
        Caller-owned list that records each contract once it holds.

    """
    from playwright.sync_api import expect

    zoom = atlas.get_by_role("group", name="Atlas zoom")
    zoom.get_by_role("button", name="Zoom out").click()
    expect(zoom).to_contain_text("Detail: kinds")
    expect(atlas.get_by_role("list", name="Atlas relations by kind")).to_contain_text("documents")
    zoom.get_by_role("button", name="Zoom in").click()
    zoom.get_by_role("button", name="Zoom in").click()
    expect(zoom).to_contain_text("Detail: details")
    expect(atlas.locator("[data-node] text").first).to_contain_text(" · ")
    zoom.get_by_role("button", name="Zoom out").click()
    expect(zoom).to_contain_text("Detail: nodes")
    observations.append("semantic-zoom-changes-the-representation")


def _controls_fit_when_zoomed(page: Page, observations: list[str]) -> None:
    """At 200 % zoom, keep every visible Atlas control inside the 1280 px viewport.

    Parameters
    ----------
    page
        Playwright page showing the Atlas at 1280 px width.
    observations
        Caller-owned list that records each contract once it holds.

    """
    page.evaluate("document.documentElement.style.zoom = '200%'")
    controls = page.locator('[aria-label="Atlas"]').locator(_CONTROLS).all()
    for control in controls:
        control.scroll_into_view_if_needed()
        box = control.bounding_box()
        assert box is not None and box["width"] > 0 and box["height"] > 0, box
        assert box["x"] >= -1 and box["x"] + box["width"] <= 1281, box
    page.evaluate("document.documentElement.style.zoom = ''")
    assert controls
    observations.append("controls-fit-at-200-percent-zoom")


def _accessibility(page: Page) -> list[dict[str, object]]:
    """Run the pinned axe rules on the Atlas at desktop and mobile width.

    Parameters
    ----------
    page
        Playwright page showing the Atlas.

    Returns
    -------
    list of dict
        One empty violation list per checked viewport width.

    """
    page.add_script_tag(path=str(AXE_SOURCE))
    accessibility: list[dict[str, object]] = []
    for width in (1280, 390):
        page.set_viewport_size({"width": width, "height": 900})
        violations = page.evaluate("""async () => {
            const root = document.querySelector('[aria-label="Atlas"]');
            const result = await axe.run(root);
            return result.violations.map(item => ({id: item.id, nodes: item.nodes.length}));
        }""")
        assert not violations, violations
        accessibility.append({"width": width, "violations": violations})
    page.set_viewport_size({"width": 1280, "height": 900})
    return accessibility


def _large_tier(page: Page, atlas: Locator, observations: list[str]) -> dict[str, object]:
    """Cancel a large pin's admission in its worker, open it again and query it there.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    atlas
        The Atlas stage.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    dict
        The large pin's size and the long main-thread tasks observed while it was open.

    """
    from playwright.sync_api import expect

    summary = atlas.get_by_role("status", name="Atlas summary")
    committed = summary.inner_text()
    chooser = atlas.get_by_label("Open a snapshot pin")
    with tempfile.TemporaryDirectory() as scratch:
        pin = large_pin(Path(scratch))
        with page.expect_worker() as started:
            chooser.set_input_files(str(pin))
        progress = atlas.get_by_role("status", name="Atlas progress")
        expect(progress).to_contain_text("Verifying large_atlas_pin.json in a worker")
        with started.value.expect_event("close"):
            progress.get_by_role("button", name="Cancel").click()
        expect(atlas.get_by_role("alert")).to_have_text(
            "Opening large_atlas_pin.json was cancelled; the open snapshot stays."
        )
        expect(summary).to_have_text(committed)
        observations.append("large-pin-admission-cancelled-and-its-worker-closed")
        # Observer entries arrive late, so tasks are selected by start time, not by arrival.
        opened_at = float(page.evaluate("performance.now()"))
        with page.expect_worker():
            chooser.set_input_files(str(pin))
        expect(summary).to_have_text(_LARGE_SUMMARY, timeout=60000)
        size = pin.stat().st_size
    expect(atlas.get_by_text(re.compile("queried in a worker"))).to_be_visible()
    observations.append("large-pin-admitted-and-queried-in-a-worker")
    atlas.get_by_label("Search").fill("n09999")
    expect(summary).to_have_text("Showing 1 of 1 nodes.", timeout=60000)
    observations.append("worker-answers-the-next-query")
    long_tasks = [
        float(duration)
        for start, duration in page.evaluate("window.__atlasLongTasks.slice()")
        if float(start) >= opened_at
    ]
    assert max(long_tasks, default=0.0) < LONG_TASK_BOUND_MS, long_tasks
    observations.append("main-thread-stays-responsive-with-the-large-pin")
    return {"large_pin_nodes": LARGE_NODES, "large_pin_bytes": size, "long_tasks_ms": long_tasks}


def _walk(page: Page, url: str, observations: list[str]) -> dict[str, object]:
    """Open the Atlas, check every navigation contract, then the large tier, then leave.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.

    Returns
    -------
    dict
        Accessibility results and the large-tier evidence.

    """
    from playwright.sync_api import expect

    page.add_init_script(_OBSERVE_LONG_TASKS)
    page.goto(url, wait_until="networkidle")
    page.get_by_role("navigation", name="Workbench context destinations").get_by_role(
        "link", name="Atlas", exact=True
    ).click()
    atlas = page.get_by_role("article", name="Atlas")
    expect(atlas.get_by_role("status", name="Atlas summary")).to_contain_text(
        "Showing", timeout=30000
    )
    expect(atlas.get_by_text(re.compile("queried on the main thread"))).to_be_visible()
    observations.append("committed-snapshot-admitted-on-the-main-thread")
    _selection(atlas, observations)
    _deep_link(page, url, atlas, observations)
    _zoom(atlas, observations)
    accessibility = _accessibility(page)
    observations.append("accessible-at-desktop-and-mobile")
    _controls_fit_when_zoomed(page, observations)
    large = _large_tier(page, atlas, observations)
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name="Workspace", exact=True
    ).click()
    wait_for_worker_disposal(page, observe_termination=False)
    observations.append("worker-disposed-on-leaving-the-atlas")
    return {"accessibility": accessibility, **large}


def run_atlas_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Navigate the Atlas graph through the production Atlas stage.

    Parameters
    ----------
    base_url
        Owned literal-loopback preview carrying the actual built Studio.
    evidence
        Caller-owned partial evidence retained on a later runtime refusal.

    Returns
    -------
    dict
        Browser identity, every observed contract and the absence of retained workers,
        submissions, outside requests and page errors.

    Raises
    ------
    ValueError
        The preview violates the shared loopback ownership contract.
    AssertionError
        A real UI, network or runtime contract fails.

    """
    return run_custodied_journey(base_url, evidence, "atlas_graph_navigation", _walk)
