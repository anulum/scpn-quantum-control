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
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

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
    from playwright.sync_api import Error, Route, expect, sync_playwright

    receipt = evidence if evidence is not None else {}
    observations: list[dict[str, object]] = []
    receipt["observations"] = observations
    origin = urlsplit(url).netloc
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            for missing in (False, True, False):
                context = browser.new_context(service_workers="block")
                context.set_default_timeout(15_000)
                context.add_init_script(_OBSERVE_WORKERS)
                held: list[Route] = []
                hold_next = False
                errors: list[str] = []
                rejected: list[str] = []

                def route_request(
                    route: Route,
                    *,
                    unavailable: bool = missing,
                    refused: list[str] = rejected,
                    waiting: list[Route] = held,
                ) -> None:
                    """Keep requests on the owned preview and hold one real worker entry."""
                    nonlocal hold_next
                    target = urlsplit(route.request.url)
                    if target.scheme != "http" or target.netloc != origin:
                        refused.append(route.request.url)
                        route.abort()
                    elif unavailable and target.path.endswith(".wasm"):
                        route.abort()
                    elif hold_next and Path(target.path).name.startswith("kernelWorker-"):
                        hold_next = False
                        waiting.append(route)
                    else:
                        route.continue_()

                try:
                    context.route("**/*", route_request)
                    page = context.new_page()

                    def record_error(error: Error, *, observed: list[str] = errors) -> None:
                        """Retain actual page failures inside this context's receipt."""
                        observed.append(str(error))

                    page.on("pageerror", record_error)
                    page.goto(url, wait_until="networkidle")
                    play = page.locator(".qsp-play")
                    chart = play.get_by_role("img", name="order parameter over time")
                    if missing:
                        expect(play.get_by_role("alert")).to_contain_text("Failed to fetch")
                        expect(chart).to_have_count(0)
                        assert page.evaluate("window.__ownedKernel.started") == 0
                        observations.append(
                            {
                                "outcome": "missing-original-WASM-visible-refusal",
                                "actual": play.get_by_role("alert").inner_text(),
                            }
                        )
                    else:
                        expect(chart).to_be_visible()
                        expect(
                            play.get_by_text(
                                "verified against the committed ground truth", exact=False
                            )
                        ).to_be_visible()
                        page.wait_for_function("window.__ownedKernel.active === 0")
                        assert not page.workers
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
                        expect(chart).to_be_visible()
                        expect(
                            play.get_by_text(
                                "verified against the committed ground truth", exact=False
                            )
                        ).to_be_visible()
                        page.wait_for_function("window.__ownedKernel.active === 0")
                        trace = page.evaluate("window.__ownedKernel")
                        commands = {
                            row["run_id"]: row
                            for row in trace["commands"]
                            if row["command"] == "validate"
                        }
                        results = [
                            row
                            for row in trace["events"]
                            if row["kind"] == "result"
                            and commands[row["run_id"]]["n"] == 2
                            and commands[row["run_id"]]["steps"] == 8
                        ]
                        actual = results[-1]
                        delta = 2 * math.atan(math.tan(1.5) * math.exp(-1.4 * 0.4))
                        assert (
                            abs(
                                actual["payload"]["thetaFinal"][1]
                                - actual["payload"]["thetaFinal"][0]
                                - delta
                            )
                            < 1e-6
                        )
                        assert (
                            abs(actual["payload"]["orderParameter"][-1] - math.cos(delta / 2))
                            < 1e-6
                        )
                        assert all(
                            row["owned_copies_detached"] is True and row["transferred"] == 4
                            for row in commands.values()
                        )
                        sequences: dict[str, int] = {}
                        for event in trace["events"]:
                            run_id = event["run_id"]
                            command = commands[run_id]
                            assert event["version"] == 1 and event["sequence"] > sequences.get(
                                run_id, 0
                            )
                            assert event["payload"]["revision_hash"] == command["revision_hash"]
                            assert event["payload"]["plan_hash"] == command["plan_hash"]
                            sequences[run_id] = event["sequence"]
                        assert trace["started"] == trace["disposed"]
                        observations.append(
                            {
                                "outcome": "real-two-node-original-WASM-and-committed-reference",
                                "actual": actual,
                                "bounded_transfer_commands": list(commands.values()),
                            }
                        )

                        for boundary in ("cancel", "replacement", "route", "project", "timeout"):
                            hold_next = True
                            if boundary == "timeout":
                                play.get_by_label("Simulation timeout (ms)").fill("1")
                            else:
                                play.get_by_role(
                                    "button", name="Run simulation", exact=True
                                ).click()
                                page.wait_for_function("window.__ownedKernel.active === 1")
                            if boundary == "cancel":
                                play.get_by_role(
                                    "button", name="Cancel simulation", exact=True
                                ).click()
                                expect(play.get_by_role("alert")).to_contain_text(
                                    "cancelled after worker disposal"
                                )
                                expect(chart).to_have_count(0)
                            elif boundary == "replacement":
                                oscillators.press("ArrowRight")
                                expect(chart).to_be_visible()
                                expect(
                                    play.get_by_text(
                                        "verified against the committed ground truth", exact=False
                                    )
                                ).to_be_visible()
                                latest = page.evaluate(
                                    "window.__ownedKernel.events.filter(row => row.kind === 'result').slice(-2)[0]"
                                )
                                assert len(latest["payload"]["thetaFinal"]) == 3
                            elif boundary in ("route", "project"):
                                page.evaluate(
                                    "kind => { window.location.hash = kind === 'route' ? '#/build' : '#/workspace?project=owned-lifecycle'; }",
                                    boundary,
                                )
                                if boundary == "route":
                                    expect(play).to_have_count(0)
                            else:
                                expect(play.get_by_role("alert")).to_contain_text(
                                    "operational deadline reached"
                                )
                                expect(chart).to_have_count(0)
                            page.wait_for_function("window.__ownedKernel.active === 0")
                            assert not page.workers
                            hold_next = False
                            for held_route in held:
                                held_route.abort()
                            held.clear()
                            observations.append(
                                {"outcome": boundary + "-disposed-no-stale-worker"}
                            )
                            if boundary in ("route", "project"):
                                page.evaluate("window.location.hash = '#/workspace'")
                            if boundary == "timeout":
                                play.get_by_label("Simulation timeout (ms)").fill("5000")
                            else:
                                play.get_by_role(
                                    "button", name="Run simulation", exact=True
                                ).click()
                            expect(chart).to_be_visible()
                            expect(
                                play.get_by_text(
                                    "verified against the committed ground truth", exact=False
                                )
                            ).to_be_visible()
                            page.wait_for_function("window.__ownedKernel.active === 0")
                        assert not page.workers
                    assert not errors, errors
                    assert not rejected, rejected
                finally:
                    for held_route in held:
                        held_route.abort()
                    context.close()
            return {
                "scenario": "owned_kernel_worker",
                "playwright": version("playwright"),
                "browser": browser.version,
                "observations": observations,
            }
        finally:
            browser.close()
