# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real owned kernel browser lifecycle evidence
"""Exercise the deployed kernel worker through the original Studio controls."""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Locator, Page, Route, Worker

_OBSERVE_WORKERS = """(() => {
  window.__ownedKernel = { started: 0, disposed: 0, active: 0, commands: [], events: [] };
  const NativeWorker = window.Worker;
  window.Worker = class extends NativeWorker {
    constructor(...args) {
      super(...args);
      this.observedClosed = false;
      window.__ownedKernel.started += 1;
      window.__ownedKernel.active += 1;
      this.addEventListener('message', ({ data }) => {
        window.__ownedKernel.events.push({
          ...data, payload: { ...data.payload,
            orderParameter: data.payload.orderParameter ? Array.from(data.payload.orderParameter) : null,
            thetaFinal: data.payload.thetaFinal ? Array.from(data.payload.thetaFinal) : null
          }
        });
      });
    }
    postMessage(message, transfers) {
      const copy = message.payload?.input;
      const binary = message.payload?.wasm;
      const row = { run_id: message.run_id, revision_hash: message.revision_hash,
        plan_hash: message.plan_hash, version: message.version, command: message.command,
        bytes: binary?.byteLength ?? 0, n: copy?.omega.length ?? null,
        steps: copy?.steps ?? null, transferred: transfers?.length ?? 0 };
      super.postMessage(message, transfers);
      row.owned_copies_detached = binary ? binary.byteLength === 0 && copy.omega.byteLength === 0 : null;
      window.__ownedKernel.commands.push(row);
    }
    terminate() {
      const result = super.terminate();
      if (!this.observedClosed) {
        this.observedClosed = true;
        window.__ownedKernel.disposed += 1;
        window.__ownedKernel.active -= 1;
      }
      return result;
    }
  };
})();"""


_VERIFIED = "verified against the committed ground truth"
_BOUNDARIES = ("cancel", "replacement", "route", "project", "timeout")
_NAVIGATE_AWAY = (
    "kind => { window.location.hash = kind === 'route' ? '#/build' "
    ": '#/workspace?project=owned-lifecycle'; }"
)
_PREVIOUS_RESULT = "window.__ownedKernel.events.filter(row => row.kind === 'result').slice(-2)[0]"


@dataclass
class _ClosureWatch:
    """Track the workers of one page until Chromium closes each of them."""

    pending: set[Worker] = field(default_factory=set)

    def record_closure(self, worker: Worker) -> None:
        """Remove a worker after Chromium emits its native close event.

        Parameters
        ----------
        worker
            Natively closed Studio worker.

        """
        self.pending.discard(worker)

    def observe_worker(self, worker: Worker) -> None:
        """Track native closure without creating a synchronous wait in a callback.

        Parameters
        ----------
        worker
            Existing or newly created native Studio worker.

        """
        self.pending.add(worker)
        worker.once("close", self.record_closure)


def wait_for_worker_disposal(page: Page, *, observe_termination: bool = True) -> None:
    """Require Chromium closure of workers observed while awaiting disposal.

    Parameters
    ----------
    page
        Actual Studio page whose workers are being disposed. Workers created
        during the wait also require a native close event.
    observe_termination
        Also await the installed observer's termination-request counter. A zero
        counter alone does not prove native closure.

    Raises
    ------
    AssertionError
        A native worker does not close before the page's existing timeout,
        the observer does not confirm termination, or a worker remains.

    """
    from playwright.sync_api import TimeoutError

    watch = _ClosureWatch()
    page.on("worker", watch.observe_worker)
    try:
        for worker in page.workers:
            watch.observe_worker(worker)
        if observe_termination:
            page.wait_for_function("window.__ownedKernel.active === 0")
        while watch.pending:
            with ExitStack() as closed:
                for worker in tuple(watch.pending):
                    closed.enter_context(worker.expect_event("close"))
    except TimeoutError as error:
        raise AssertionError("Studio native worker disposal was not confirmed") from error
    finally:
        page.remove_listener("worker", watch.observe_worker)
        for worker in tuple(watch.pending):
            worker.remove_listener("close", watch.record_closure)
    assert not page.workers, "Studio native worker disposal left an owned worker"


@dataclass
class _Transport:
    """Keep one context on its preview, optionally without kernels, holding one worker."""

    origin: str
    unavailable: bool
    refused: list[str] = field(default_factory=list)
    held: list[Route] = field(default_factory=list)
    hold_next: bool = False

    def __call__(self, route: Route) -> None:
        """Keep requests on the owned preview and hold one real worker entry.

        Parameters
        ----------
        route
            Intercepted request.

        """
        target = urlsplit(route.request.url)
        if target.scheme != "http" or target.netloc != self.origin:
            self.refused.append(route.request.url)
            route.abort()
        elif self.unavailable and target.path.endswith(".wasm"):
            route.abort()
        elif self.hold_next and Path(target.path).name.startswith("kernelWorker-"):
            self.hold_next = False
            self.held.append(route)
        else:
            route.continue_()

    def release(self) -> None:
        """Stop holding and abort every held worker entry."""
        self.hold_next = False
        for held_route in self.held:
            held_route.abort()
        self.held.clear()


def _expect_verified(play: Locator) -> None:
    """Require the chart and its verification against the committed reference.

    Parameters
    ----------
    play
        The playground region.

    """
    from playwright.sync_api import expect

    expect(play.get_by_role("img", name="order parameter over time")).to_be_visible()
    expect(play.get_by_text(_VERIFIED, exact=False)).to_be_visible()


def _configure_two_nodes(play: Locator) -> Locator:
    """Select two oscillators, no spread, coupling 1.4 and eight steps by keyboard.

    Parameters
    ----------
    play
        The playground region.

    Returns
    -------
    Locator
        The oscillator-count slider.

    """
    oscillators = play.get_by_label("Oscillators N:", exact=False)
    oscillators.press("Home")
    oscillators.press("ArrowRight")
    play.get_by_label("Frequency spread:", exact=False).press("Home")
    coupling = play.get_by_label("Coupling K:", exact=False)
    coupling.press("Home")
    for _ in range(14):
        coupling.press("ArrowRight")
    steps = play.get_by_label("Steps:", exact=False)
    steps.press("Home")
    for _ in range(7):
        steps.press("ArrowRight")
    return oscillators


def _two_node_result(trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the two-node result against its closed form and every envelope's custody.

    Parameters
    ----------
    trace
        The owned kernel observer's record.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The two-node result and the validate commands by run identity.

    """
    commands = {row["run_id"]: row for row in trace["commands"] if row["command"] == "validate"}
    results = [
        row
        for row in trace["events"]
        if row["kind"] == "result"
        and commands[row["run_id"]]["n"] == 2
        and commands[row["run_id"]]["steps"] == 8
    ]
    actual = results[-1]
    delta = 2 * math.atan(math.tan(1.5) * math.exp(-1.4 * 0.4))
    theta = actual["payload"]["thetaFinal"]
    assert abs(theta[1] - theta[0] - delta) < 1e-6
    assert abs(actual["payload"]["orderParameter"][-1] - math.cos(delta / 2)) < 1e-6
    assert all(
        row["owned_copies_detached"] is True and row["transferred"] == 4
        for row in commands.values()
    )
    sequences: dict[str, int] = {}
    for event in trace["events"]:
        run_id = event["run_id"]
        command = commands[run_id]
        assert event["version"] == 1 and event["sequence"] > sequences.get(run_id, 0)
        assert event["payload"]["revision_hash"] == command["revision_hash"]
        assert event["payload"]["plan_hash"] == command["plan_hash"]
        sequences[run_id] = event["sequence"]
    assert trace["started"] == trace["disposed"]
    return actual, commands


def _check_two_nodes(page: Page, play: Locator) -> tuple[Locator, dict[str, object]]:
    """Compute two nodes in the real worker and match the closed-form reference.

    Parameters
    ----------
    page
        Owned page.
    play
        The playground region.

    Returns
    -------
    tuple[Locator, dict[str, object]]
        The oscillator-count slider and the observation.

    """
    _expect_verified(play)
    wait_for_worker_disposal(page)
    oscillators = _configure_two_nodes(play)
    _expect_verified(play)
    page.wait_for_function("window.__ownedKernel.active === 0")
    actual, commands = _two_node_result(page.evaluate("window.__ownedKernel"))
    return oscillators, {
        "outcome": "real-two-node-original-WASM-and-committed-reference",
        "actual": actual,
        "bounded_transfer_commands": list(commands.values()),
    }


def _start_boundary(page: Page, play: Locator, boundary: str) -> None:
    """Start a run whose worker entry is held, or arm the 1 ms timeout.

    Parameters
    ----------
    page
        Owned page.
    play
        The playground region.
    boundary
        Lifecycle boundary under test.

    """
    if boundary == "timeout":
        play.get_by_label("Simulation timeout (ms)").fill("1")
    else:
        play.get_by_role("button", name="Run simulation", exact=True).click()
        page.wait_for_function("window.__ownedKernel.active === 1")


def _cross_boundary(page: Page, play: Locator, oscillators: Locator, boundary: str) -> None:
    """Cancel, replace, navigate away or time out the held run and check what remains.

    Parameters
    ----------
    page
        Owned page.
    play
        The playground region.
    oscillators
        The oscillator-count slider.
    boundary
        Lifecycle boundary under test.

    """
    from playwright.sync_api import expect

    chart = play.get_by_role("img", name="order parameter over time")
    if boundary == "cancel":
        play.get_by_role("button", name="Cancel simulation", exact=True).click()
        expect(play.get_by_role("alert")).to_contain_text("cancelled after worker disposal")
        expect(chart).to_have_count(0)
    elif boundary == "replacement":
        oscillators.press("ArrowRight")
        _expect_verified(play)
        latest = page.evaluate(_PREVIOUS_RESULT)
        assert len(latest["payload"]["thetaFinal"]) == 3
    elif boundary in ("route", "project"):
        page.evaluate(_NAVIGATE_AWAY, boundary)
        if boundary == "route":
            expect(play).to_have_count(0)
    else:
        expect(play.get_by_role("alert")).to_contain_text("operational deadline reached")
        expect(chart).to_have_count(0)


def _recover(page: Page, play: Locator, boundary: str) -> None:
    """Return to the workspace or restore the timeout and find a verified fresh run.

    Parameters
    ----------
    page
        Owned page.
    play
        The playground region.
    boundary
        Lifecycle boundary just crossed.

    """
    if boundary in ("route", "project"):
        page.evaluate("window.location.hash = '#/workspace'")
    if boundary == "timeout":
        play.get_by_label("Simulation timeout (ms)").fill("5000")
    else:
        play.get_by_role("button", name="Run simulation", exact=True).click()
    _expect_verified(play)
    wait_for_worker_disposal(page)


def _check_boundaries(
    page: Page, play: Locator, oscillators: Locator, transport: _Transport
) -> list[dict[str, object]]:
    """Dispose the held worker at every lifecycle boundary with no stale result.

    Parameters
    ----------
    page
        Owned page.
    play
        The playground region.
    oscillators
        The oscillator-count slider.
    transport
        Route state that holds the next worker entry.

    Returns
    -------
    list[dict[str, object]]
        One observation per boundary.

    """
    observations: list[dict[str, object]] = []
    for boundary in _BOUNDARIES:
        transport.hold_next = True
        _start_boundary(page, play, boundary)
        _cross_boundary(page, play, oscillators, boundary)
        wait_for_worker_disposal(page)
        transport.release()
        observations.append({"outcome": boundary + "-disposed-no-stale-worker"})
        _recover(page, play, boundary)
    assert not page.workers
    return observations


def _worker_case(
    browser: Browser, url: str, missing: bool, observations: list[dict[str, object]]
) -> None:
    """Run one worker case in its own context: missing kernel, or compute and dispose.

    Parameters
    ----------
    browser
        Launched browser.
    url
        Owned preview.
    missing
        Whether every kernel request is refused.
    observations
        Caller-owned list that records each outcome once it holds.

    """
    from playwright.sync_api import expect

    context = browser.new_context(service_workers="block")
    context.set_default_timeout(15_000)
    context.add_init_script(_OBSERVE_WORKERS)
    transport = _Transport(urlsplit(url).netloc, missing)
    errors: list[str] = []
    try:
        context.route("**/*", transport)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="networkidle")
        play = page.locator(".qsp-play")
        if missing:
            expect(play.get_by_role("alert")).to_contain_text("Failed to fetch")
            expect(play.get_by_role("img", name="order parameter over time")).to_have_count(0)
            assert page.evaluate("window.__ownedKernel.started") == 0
            actual = play.get_by_role("alert").inner_text()
            observations.append(
                {"outcome": "missing-original-WASM-visible-refusal", "actual": actual}
            )
        else:
            oscillators, computed = _check_two_nodes(page, play)
            observations.append(computed)
            observations.extend(_check_boundaries(page, play, oscillators, transport))
        assert not errors, errors
        assert not transport.refused, transport.refused
    finally:
        transport.release()
        context.close()


def run_owned_worker_journey(
    base_url: str, *, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Observe native worker computation, refusal, cancellation and route disposal.

    Parameters
    ----------
    base_url
        Owned literal loopback preview containing the built original WASM.
    evidence
        Optional shared dispatcher receipt updated after each observed boundary.

    Returns
    -------
    dict[str, object]
        Actual browser versions, bounded native envelopes and lifecycle outcomes.

    Raises
    ------
    ValueError
        The preview is not an explicit plain HTTP loopback address.
    AssertionError
        Source parity, cancellation, stale-result custody or disposal fails.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import sync_playwright

    receipt = evidence if evidence is not None else {}
    observations: list[dict[str, object]] = []
    receipt["observations"] = observations
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            for missing in (False, True, False):
                _worker_case(browser, url, missing, observations)
            return {
                "scenario": "owned_kernel_worker",
                "playwright": version("playwright"),
                "browser": browser.version,
                "observations": observations,
            }
        finally:
            browser.close()
