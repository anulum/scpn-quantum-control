# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — studio browser journey
"""Run a real Studio catalogue journey against an owned loopback preview.

Install the hash-locked CI browser extra and Chromium on the runner. This
command neither starts a provider nor permits navigation away from the preview.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit


def loopback_url(value: str) -> str:
    """Validate a plain HTTP preview URL with a literal loopback address.

    Parameters
    ----------
    value
        Owned preview URL, including its port and optional deployment prefix.

    Returns
    -------
    str
        Validated URL with a trailing slash.

    Raises
    ------
    ValueError
        The URL has an external address, credentials, query, fragment or port
        outside the TCP range. Hostnames are refused to avoid DNS rebinding.

    """
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname is None
        or not ipaddress.ip_address(parsed.hostname).is_loopback
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.port is None
        or parsed.port == 0
    ):
        raise ValueError("Use an owned http://loopback-address:port/ preview")
    return value.rstrip("/") + "/"


def run_catalogue_journey(base_url: str) -> dict[str, object]:
    """Exercise catalogue filters, keyboard routing, real WASM and recovery.

    Parameters
    ----------
    base_url
        Owned loopback preview of the built Studio bundle.

    Returns
    -------
    dict[str, object]
        Browser/package versions, actual catalogue identity and observed
        positive, missing-kernel and recovery outcomes.

    Raises
    ------
    ValueError
        The preview URL is not a literal loopback address.
    AssertionError
        A rendered route, recomputation, refusal or disposal contract fails.

    """
    from playwright.sync_api import Error, Route, expect, sync_playwright

    url = loopback_url(base_url)
    origin = urlsplit(url).netloc
    observations: list[dict[str, object]] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            for missing_kernel in (False, True, False):
                context = browser.new_context(service_workers="block")
                context.set_default_timeout(15_000)
                rejected: list[str] = []
                errors: list[str] = []

                def bound_request(
                    route: Route,
                    *,
                    refused: list[str] = rejected,
                    unavailable: bool = missing_kernel,
                ) -> None:
                    target = urlsplit(route.request.url)
                    if target.scheme != "http" or target.netloc != origin:
                        refused.append("outside-preview")
                        route.abort()
                    elif unavailable and target.path.endswith(".wasm"):
                        route.abort()
                    else:
                        route.continue_()

                try:
                    context.route("**/*", bound_request)
                    page = context.new_page()

                    def record_error(error: Error, observed: list[str] = errors) -> None:
                        observed.append(str(error))

                    page.on("pageerror", record_error)
                    page.goto(url, wait_until="networkidle")
                    catalogue = page.get_by_role("region", name="Capability catalogue")
                    expect(
                        catalogue.get_by_test_id("capability-execute").get_by_role("link")
                    ).to_have_count(0)
                    catalogue.get_by_label("Capability task").fill("compile")
                    catalogue.get_by_label("Capability runtime").select_option("browser-wasm")
                    catalogue.get_by_label("Capability backend").select_option("rust")
                    expect(catalogue.get_by_test_id("capability-compile")).to_have_count(1)
                    expect(catalogue.locator("li")).to_have_count(1)
                    link = catalogue.get_by_role("link", name="Open XY compile recomputation")
                    if missing_kernel:
                        expect(link).to_have_count(0)
                        expect(
                            catalogue.get_by_test_id("capability-compile").locator(".qsp-boundary")
                        ).not_to_have_text("Backend availability unknown")
                        outcome = "missing-kernel-refused"
                    else:
                        expect(link).to_be_visible()
                        link.focus()
                        page.keyboard.press("Enter")
                        expect(page.locator('[id="/build/compile-recompute"]')).to_be_focused()
                        expect(page).to_have_url(url + "#/build/compile-recompute")
                        panel = page.locator('[id="/build/compile-recompute"]')
                        panel.get_by_role("button", name="Recompute in browser").click()
                        expect(panel.get_by_role("status")).to_have_text(
                            "recomputed digest matches the signed claim"
                        )
                        outcome = panel.get_by_role("status").inner_text()
                    identity = catalogue.locator(".qsp-digest code").inner_text()
                    catalogue.get_by_label("Capability backend").select_option("numpy")
                    expect(
                        catalogue.get_by_text("No capability matches these filters.")
                    ).to_be_visible()
                    catalogue.get_by_role("button", name="Clear filters").click()
                    expect(catalogue.locator("li")).to_have_count(9)
                    assert not rejected, rejected
                    assert not errors, errors
                    assert not page.workers, "Catalogue must not leave an owned worker"
                    observations.append(
                        {
                            "missing_kernel": missing_kernel,
                            "outcome": outcome,
                            "identity": identity,
                            "workers": len(page.workers),
                        }
                    )
                finally:
                    context.close()
            return {
                "scenario": "capability_catalogue",
                "playwright": version("playwright"),
                "browser": browser.version,
                "observations": observations,
            }
        finally:
            browser.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected journey and write bounded JSON evidence.

    Parameters
    ----------
    argv
        Command arguments, or the process arguments when omitted.

    Returns
    -------
    int
        Zero on observed acceptance; one on a failed journey or invalid URL.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("capability_catalogue",), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence: dict[str, object] = {
        "scenario": args.scenario,
        "base_url": "rejected",
        "command": "studio_browser_journey --scenario capability_catalogue",
    }
    try:
        url = loopback_url(args.base_url)
        evidence["base_url"] = url
        evidence.update(run_catalogue_journey(url))
        evidence["passed"] = True
        code = 0
    except Exception as error:
        evidence.update(passed=False, error=f"{type(error).__name__}: {error}")
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": code == 0, "evidence": str(args.output)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
