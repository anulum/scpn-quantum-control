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
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Locator, Page, Route

_ALLOCATION_PROBE = """(() => {
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
_ORDER = "order parameter over time"
_ALLOCATIONS = "window.__studioResourceAllocations"
_WORKERS = "window.__studioResourceWorkers"


def _configure_large_networked(play: Locator, plan: Locator) -> None:
    """Select sixteen oscillators, the most steps and the networked topology.

    Parameters
    ----------
    play
        The playground region.
    plan
        Its resource plan region.

    """
    from playwright.sync_api import expect

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
    expect(play.get_by_role("img", name=_ORDER)).to_be_visible()


def _refuse_before_allocation(page: Page, play: Locator, plan: Locator) -> dict[str, object]:
    """Refuse a zero memory ceiling before any native allocation or worker.

    Parameters
    ----------
    page
        Owned page carrying the allocation probe.
    play
        The playground region.
    plan
        Its resource plan region.

    Returns
    -------
    dict[str, object]
        Observation with the allocation and worker counts around the refusal.

    """
    from playwright.sync_api import expect

    allocations_before = page.evaluate(_ALLOCATIONS)
    assert isinstance(allocations_before, int) and allocations_before > 0
    workers_before = page.evaluate(_WORKERS)
    play.get_by_label("Memory ceiling (KiB)").fill("0")
    expect(plan.get_by_role("alert")).to_contain_text("declared_storage_exceeds_budget")
    expect(play.get_by_role("img", name=_ORDER)).to_have_count(0)
    allocations_after = page.evaluate(_ALLOCATIONS)
    assert allocations_after == allocations_before
    assert page.evaluate(_WORKERS) == workers_before
    return {
        "outcome": "refused-before-native-allocation",
        "before": allocations_before,
        "after": allocations_after,
        "worker_constructors": workers_before,
    }


def _recover_smaller(
    page: Page, url: str, play: Locator, plan: Locator, workers_before: int
) -> dict[str, object]:
    """Offer a ceiling just above the kernel and recover with the same method, smaller.

    Parameters
    ----------
    page
        Owned page carrying the allocation probe.
    url
        Owned literal-loopback preview.
    play
        The playground region.
    plan
        Its resource plan region.
    workers_before
        Worker constructions counted before the refusal.

    Returns
    -------
    dict[str, object]
        Observation with the recovered allocation and worker counts.

    """
    from playwright.sync_api import expect

    manifest = page.request.get(url + "deploy-manifest.json").json()
    binary_bytes = next(
        row["bytes"]
        for row in manifest["artifacts"]
        if row["path"] == "wasm/scpn_quantum_studio_wasm_kernel.wasm"
    )
    ceiling = play.get_by_label("Memory ceiling (KiB)")
    ceiling.fill(str((2 * binary_bytes + 2048 + 1023) // 1024))
    expect(play.get_by_role("img", name=_ORDER)).to_have_count(0)
    smaller = play.get_by_role("button", name="Apply smaller supported configuration", exact=False)
    smaller.click()
    expect(play.get_by_role("img", name=_ORDER)).to_be_visible()
    expect(plan.get_by_role("alert")).to_have_count(0)
    expect(play.get_by_label("Topology")).to_have_value("networked")
    expect(plan).to_contain_text("float64")
    allocations_recovered = page.evaluate(_ALLOCATIONS)
    workers_recovered = page.evaluate(_WORKERS)
    assert isinstance(workers_recovered, int) and workers_recovered > workers_before
    return {
        "outcome": "same-method-smaller-recovered",
        "main_allocations": allocations_recovered,
        "worker_constructors": workers_recovered,
    }


def _refuse_wall_clock(page: Page, play: Locator, plan: Locator) -> dict[str, object]:
    """Refuse an unsupported wall-clock ceiling before allocation, then clear it.

    Parameters
    ----------
    page
        Owned page carrying the allocation probe.
    play
        The playground region.
    plan
        Its resource plan region.

    Returns
    -------
    dict[str, object]
        Observation with the allocation count at the refusal.

    """
    from playwright.sync_api import expect

    play.get_by_label("Memory ceiling (KiB)").fill("4096")
    play.get_by_label("Topology").select_option("mean-field")
    expect(plan).to_contain_text("mean-field")
    expect(play.get_by_role("img", name=_ORDER)).to_be_visible()
    before_deadline = page.evaluate(_ALLOCATIONS)
    workers_before_deadline = page.evaluate(_WORKERS)
    wall_clock = play.get_by_label("Wall-clock ceiling (ms; optional)")
    wall_clock.fill("1")
    expect(plan.get_by_role("alert")).to_contain_text("wall_clock_admission_unavailable")
    expect(play.get_by_role("img", name=_ORDER)).to_have_count(0)
    assert page.evaluate(_ALLOCATIONS) == before_deadline
    assert page.evaluate(_WORKERS) == workers_before_deadline
    wall_clock.fill("")
    expect(play.get_by_role("img", name=_ORDER)).to_be_visible()
    return {"outcome": "unsupported-wall-clock-refused", "allocations": before_deadline}


def _walk(page: Page, url: str) -> list[dict[str, object]]:
    """Refuse, recover and refuse again in the playground's resource plan.

    Parameters
    ----------
    page
        Owned page carrying the allocation probe.
    url
        Owned literal-loopback preview.

    Returns
    -------
    list[dict[str, object]]
        The three observed outcomes in order.

    """
    from playwright.sync_api import expect

    page.goto(url, wait_until="networkidle")
    play = page.locator(".qsp-play")
    expect(play.get_by_role("img", name=_ORDER)).to_be_visible()
    cylinder = "phase-space cylinder: oscillator phases over time"
    expect(page.get_by_role("img", name=cylinder)).to_be_visible()
    plan = play.get_by_role("region", name="Resource plan")
    _configure_large_networked(play, plan)
    refused = _refuse_before_allocation(page, play, plan)
    workers_before = refused["worker_constructors"]
    assert isinstance(workers_before, int)
    recovered = _recover_smaller(page, url, play, plan, workers_before)
    return [refused, recovered, _refuse_wall_clock(page, play, plan)]


def _owned_journey(browser: Browser, url: str) -> list[dict[str, object]]:
    """Run the walk in a context bound to the owned preview and check its custody.

    Parameters
    ----------
    browser
        Launched browser.
    url
        Owned literal-loopback preview.

    Returns
    -------
    list[dict[str, object]]
        The observed outcomes.

    """
    origin = urlsplit(url).netloc
    errors: list[str] = []
    rejected: list[str] = []
    context = browser.new_context(service_workers="block")
    context.set_default_timeout(15_000)
    try:
        context.add_init_script(_ALLOCATION_PROBE)

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
        observations = _walk(page, url)
        assert not errors, errors
        assert not rejected, rejected
        assert not page.workers, "Resource preview must not leave an owned worker"
        return observations
    finally:
        context.close()


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
    from playwright.sync_api import sync_playwright

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            observations = _owned_journey(browser, url)
            return {
                "scenario": "resource_plan_projection",
                "playwright": version("playwright"),
                "browser": browser.version,
                "observations": observations,
            }
        finally:
            browser.close()
