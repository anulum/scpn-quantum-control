# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — studio browser journey
"""Observe Studio resource refusal and recovery with the real built WASM."""

from __future__ import annotations

from importlib.metadata import version
from urllib.parse import urlsplit


def run_resource_journey(base_url: str) -> dict[str, object]:
    """Exercise public resource controls before native allocation and after recovery.

    Parameters
    ----------
    base_url
        Owned loopback preview of the built Studio bundle.

    Returns
    -------
    dict[str, object]
        Actual browser versions, allocator observations and policy outcomes.

    Raises
    ------
    ValueError
        The preview is not an explicit loopback URL.
    AssertionError
        Refusal, numerical recovery, recalculation or disposal fails.

    """
    from tools.studio_browser_journey import loopback_url

    url = loopback_url(base_url)
    from playwright.sync_api import Route, expect, sync_playwright

    origin = urlsplit(url).netloc
    errors: list[str] = []
    rejected: list[str] = []
    observations: list[dict[str, object]] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        context = browser.new_context(service_workers="block")
        context.set_default_timeout(15_000)
        try:
            context.add_init_script(
                """(() => {
                  window.__studioResourceAllocations = 0;
                  window.__studioResourceWorkers = 0;
                  const NativeWorker = window.Worker;
                  window.Worker = class extends NativeWorker {
                    constructor(...args) {
                      super(...args);
                      window.__studioResourceWorkers += 1;
                    }
                  };
                  const nativeInstantiate = WebAssembly.instantiate.bind(WebAssembly);
                  WebAssembly.instantiate = async (...args) => {
                    const native = await nativeInstantiate(...args);
                    const instance = native instanceof WebAssembly.Instance ? native : native.instance;
                    const exports = instance.exports;
                    if (typeof exports.scpn_kuramoto_simulate !== 'function') return native;
                    const observed = { ...exports, scpn_alloc: (...allocation) => {
                      window.__studioResourceAllocations += 1;
                      return exports.scpn_alloc(...allocation);
                    }};
                    const bound = { exports: observed };
                    return native instanceof WebAssembly.Instance ? bound : { ...native, instance: bound };
                  };
                })();"""
            )

            def bound_request(route: Route) -> None:
                parsed = urlsplit(route.request.url)
                if parsed.scheme == "http" and parsed.netloc == origin:
                    route.continue_()
                else:
                    rejected.append(route.request.url)
                    route.abort()

            context.route("**/*", bound_request)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url, wait_until="networkidle")
            play = page.locator(".qsp-play")
            expect(play.get_by_role("img", name="order parameter over time")).to_be_visible()
            expect(
                page.get_by_role("img", name="phase-space cylinder: oscillator phases over time")
            ).to_be_visible()
            plan = play.get_by_role("region", name="Resource plan")
            expect(plan).to_contain_text("shipped-kuramoto-wasm-float64")
            expect(plan).to_contain_text("allocator/object overhead excluded")
            oscillators = play.get_by_label("Oscillators N:", exact=False)
            oscillators.press("Home")
            for _ in range(15):
                oscillators.press("ArrowRight")
            expect(oscillators).to_have_value("16")
            play.get_by_label("Steps:", exact=False).press("End")
            play.get_by_label("Topology").select_option("networked")
            expect(plan).to_contain_text("networked")
            expect(play.get_by_role("img", name="order parameter over time")).to_be_visible()
            allocations_before = page.evaluate("window.__studioResourceAllocations")
            assert isinstance(allocations_before, int) and allocations_before > 0
            workers_before = page.evaluate("window.__studioResourceWorkers")
            play.get_by_label("Memory ceiling (KiB)").fill("0")
            expect(plan.get_by_role("alert")).to_contain_text("declared_storage_exceeds_budget")
            expect(play.get_by_role("img", name="order parameter over time")).to_have_count(0)
            allocations_after = page.evaluate("window.__studioResourceAllocations")
            assert allocations_after == allocations_before
            assert page.evaluate("window.__studioResourceWorkers") == workers_before
            observations.append(
                {
                    "outcome": "refused-before-native-allocation",
                    "before": allocations_before,
                    "after": allocations_after,
                    "worker_constructors": workers_before,
                }
            )
            manifest = page.request.get(url + "deploy-manifest.json").json()
            binary_bytes = next(
                row["bytes"]
                for row in manifest["artifacts"]
                if row["path"] == "wasm/scpn_quantum_studio_wasm_kernel.wasm"
            )
            play.get_by_label("Memory ceiling (KiB)").fill(
                str((2 * binary_bytes + 2048 + 1023) // 1024)
            )
            expect(play.get_by_role("img", name="order parameter over time")).to_have_count(0)
            play.get_by_role(
                "button", name="Apply smaller supported configuration", exact=False
            ).click()
            expect(play.get_by_role("img", name="order parameter over time")).to_be_visible()
            expect(plan.get_by_role("alert")).to_have_count(0)
            expect(play.get_by_label("Topology")).to_have_value("networked")
            expect(plan).to_contain_text("float64")
            allocations_recovered = page.evaluate("window.__studioResourceAllocations")
            workers_recovered = page.evaluate("window.__studioResourceWorkers")
            assert isinstance(workers_recovered, int) and workers_recovered > workers_before
            observations.append(
                {
                    "outcome": "same-method-smaller-recovered",
                    "main_allocations": allocations_recovered,
                    "worker_constructors": workers_recovered,
                }
            )
            play.get_by_label("Memory ceiling (KiB)").fill("4096")
            play.get_by_label("Topology").select_option("mean-field")
            expect(plan).to_contain_text("mean-field")
            expect(play.get_by_role("img", name="order parameter over time")).to_be_visible()
            before_deadline = page.evaluate("window.__studioResourceAllocations")
            workers_before_deadline = page.evaluate("window.__studioResourceWorkers")
            play.get_by_label("Wall-clock ceiling (ms; optional)").fill("1")
            expect(plan.get_by_role("alert")).to_contain_text("wall_clock_admission_unavailable")
            expect(play.get_by_role("img", name="order parameter over time")).to_have_count(0)
            assert page.evaluate("window.__studioResourceAllocations") == before_deadline
            assert page.evaluate("window.__studioResourceWorkers") == workers_before_deadline
            observations.append(
                {"outcome": "unsupported-wall-clock-refused", "allocations": before_deadline}
            )
            play.get_by_label("Wall-clock ceiling (ms; optional)").fill("")
            expect(play.get_by_role("img", name="order parameter over time")).to_be_visible()
            assert not errors, errors
            assert not rejected, rejected
            assert not page.workers, "Resource preview must not leave an owned worker"
            return {
                "scenario": "resource_plan_projection",
                "playwright": version("playwright"),
                "browser": browser.version,
                "observations": observations,
            }
        finally:
            context.close()
            browser.close()
