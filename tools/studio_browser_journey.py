# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — studio browser journey
"""Run real Studio catalogue and evidence journeys on an owned loopback preview.

Install the hash-locked CI browser extra and Chromium on the runner. This
command neither starts a provider nor permits navigation away from the preview.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
from collections.abc import Callable, Sequence
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import Page


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

    from tools.studio_owned_worker_journey import wait_for_worker_disposal

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
                    identity = catalogue.locator(".qsp-digest code").inner_text()
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
                        page.get_by_role("navigation", name="Workbench views").get_by_role(
                            "link", name="Workspace", exact=True
                        ).click()
                        catalogue.get_by_label("Capability task").fill("compile")
                        catalogue.get_by_label("Capability runtime").select_option("browser-wasm")
                    expect(catalogue.locator(".qsp-digest code")).to_have_text(identity)
                    catalogue.get_by_label("Capability backend").select_option("numpy")
                    expect(
                        catalogue.get_by_text("No capability matches these filters.")
                    ).to_be_visible()
                    catalogue.get_by_role("button", name="Clear filters").click()
                    expect(catalogue.locator("li")).to_have_count(9)
                    assert not rejected, rejected
                    assert not errors, errors
                    wait_for_worker_disposal(page, observe_termination=False)
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


def run_evidence_journey(
    base_url: str,
    *,
    audit: Callable[[Page, str], None] | None = None,
    color_scheme: Literal["light", "dark"] = "light",
) -> dict[str, object]:
    """Inspect original metadata and real WASM replay across snapshot changes.

    Parameters
    ----------
    base_url
        Owned loopback preview of the built Studio bundle.
    audit
        Optional actual-page audit at each observed evidence state.
    color_scheme
        Native browser theme used for the original evidence journey.

    Returns
    -------
    dict[str, object]
        Observed source, mismatch, custody and asynchronous revision boundaries.

    Raises
    ------
    ValueError
        The preview URL is not a literal loopback address.
    AssertionError
        A visible claim or verification crosses the wrong snapshot boundary.

    """
    from playwright.sync_api import Error, Route, expect, sync_playwright

    url = loopback_url(base_url)
    origin = urlsplit(url).netloc
    source = (
        Path(__file__).resolve().parents[1]
        / "data/studio/program_ad_replay_rational_20260714.json"
    )
    original_text = source.read_text(encoding="utf-8")
    original = json.loads(original_text)
    changed = json.loads(original_text)
    changed["expected"]["gradient"][1] = 99.0
    observations: list[str] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        context = browser.new_context(service_workers="block", color_scheme=color_scheme)
        context.set_default_timeout(15_000)
        held: list[Route] = []
        hold_next = False
        rejected: list[str] = []
        errors: list[str] = []

        def bound_request(route: Route) -> None:
            nonlocal hold_next
            target = urlsplit(route.request.url)
            if target.scheme != "http" or target.netloc != origin:
                rejected.append("outside-preview")
                route.abort()
            elif (
                hold_next
                and Path(target.path).name.startswith("scpn_quantum_studio_program_ad_wasm")
                and target.path.endswith(".wasm")
            ):
                hold_next = False
                held.append(route)
            else:
                route.continue_()

        try:
            context.route("**/*", bound_request)
            page = context.new_page()
            page.add_init_script("""
                window.__studioEvidenceDigests = 0;
                const digest = crypto.subtle.digest.bind(crypto.subtle);
                Object.defineProperty(crypto.subtle, "digest", {
                    value: (...args) => digest(...args).finally(() => {
                        window.__studioEvidenceDigests += 1;
                    })
                });
            """)

            def record_error(error: Error) -> None:
                errors.append(str(error))

            page.on("pageerror", record_error)

            def record(state: str) -> None:
                """Retain the original observation and its optional live-page audit."""
                observations.append(state)
                if audit is not None:
                    audit(page, state)

            page.goto(url, wait_until="networkidle")
            viewer = page.get_by_role("region", name="Inspect evidence JSON", exact=True)
            editor = viewer.get_by_label("Evidence JSON")
            inspect = viewer.get_by_role("button", name="Inspect snapshot")
            editor.fill(original_text)
            inspect.click()
            inspector = viewer.get_by_role("region", name="Evidence inspector", exact=True)
            inspector.get_by_role("button").click()
            expect(inspector.get_by_role("status")).to_have_attribute("data-verdict", "match")
            record("original-source-real-wasm-match")

            # Hold a real kernel response for A, finish B, then allow A to resolve last.
            hold_next = True
            inspector.get_by_role("button").click()
            expect(inspector.get_by_role("button")).to_have_text("Recomputing…")
            if audit is not None:
                audit(page, "actual-original-verification-pending")
            editor.fill(json.dumps(changed))
            inspect.click()
            expect(inspector.get_by_role("status")).to_have_count(0)
            inspector.get_by_role("button").click()
            expect(inspector.get_by_role("status")).to_have_attribute("data-verdict", "mismatch")
            assert len(held) == 1, "Expected exactly one held original kernel response"
            completed_digests = page.evaluate("window.__studioEvidenceDigests")
            with page.expect_request_finished(
                lambda request: (
                    Path(urlsplit(request.url).path).name.startswith(
                        "scpn_quantum_studio_program_ad_wasm"
                    )
                    and urlsplit(request.url).path.endswith(".wasm")
                )
            ):
                held.pop().continue_()
            page.wait_for_function(
                "previous => window.__studioEvidenceDigests > previous", arg=completed_digests
            )
            expect(inspector.get_by_role("status")).to_have_attribute("data-verdict", "mismatch")
            record("same-id-changed-claim-retains-B-after-delayed-A")

            tampered = json.loads(original_text)
            # Flip one original bit; this always changes the actual fixture bytes.
            changed_byte = int(original["input_hex"][-2:], 16) ^ 1
            tampered["input_hex"] = original["input_hex"][:-2] + f"{changed_byte:02x}"
            editor.fill(json.dumps(tampered))
            inspect.click()
            inspector.get_by_role("button").click()
            expect(inspector.get_by_role("status")).to_have_attribute(
                "data-verdict", "unverifiable"
            )
            expect(inspector.get_by_role("status")).to_contain_text("SHA-256 binding")
            record("altered-source-digest-refused")

            negative = {
                "schema": "studio.evidence-replay.v1",
                "prov": {
                    "entity": {
                        "id": "synthetic-negative-presentation",
                        "digest": "sha256:" + "a" * 64,
                    }
                },
                "evidence_kind": "falsified",
                "claim_boundary": {"status": "refuted", "admission": "rejected"},
                "freshness": "traceable-unchecked",
                "attestation": {"signature": "synthetic-unverified-metadata"},
            }
            editor.fill(json.dumps(negative))
            inspect.click()
            expect(inspector.get_by_text("falsified", exact=True)).to_be_visible()
            expect(inspector.get_by_text("refuted", exact=True)).to_be_visible()
            expect(
                inspector.get_by_text("Seal present — not verified", exact=True)
            ).to_be_visible()
            expect(inspector.get_by_role("button")).to_have_count(0)
            expect(inspector.get_by_role("status")).to_have_count(0)
            record("attested-falsification-retains-source-status")
            editor.fill("{}")
            inspect.click()
            for message in (
                "Missing schema",
                "Missing source",
                "Missing seal",
                "Partial or unsupported evidence",
            ):
                expect(inspector.get_by_text(message, exact=True)).to_be_visible()
            record("missing-source-schema-seal-visible")
            editor.fill("{")
            inspect.click()
            expect(viewer.get_by_role("alert")).to_contain_text("Cannot inspect evidence")
            expect(viewer.get_by_role("region", name="Evidence inspector")).to_have_count(0)
            if audit is not None:
                audit(page, "malformed-evidence-retains-raw-input")
            assert not errors, errors
            assert not rejected, rejected
            assert not page.workers, "Evidence inspection must not leak a worker"
            return {
                "scenario": "evidence_inspector",
                "playwright": version("playwright"),
                "browser": browser.version,
                "observations": observations,
            }
        finally:
            for route in held:
                route.abort()
            context.close()
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
    parser.add_argument(
        "--scenario",
        choices=(
            "capability_catalogue",
            "evidence_inspector",
            "resource_plan_projection",
            "workspace_recovery",
            "workspace_panel_refusal",
            "workbench_navigation",
            "parameter_graph_editor",
            "program_authoring",
            "compiler_trace_inspector",
            "operator_backend_profiles",
            "operator_policy_decisions",
            "operator_review_dossiers",
            "workbench_accessibility",
            "owned_kernel_worker",
            "local_experiment_journey",
            "result_value_inspector",
            "immutable_run_comparison",
            "experiment_workflow_runner",
        ),
        required=True,
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--workspace-source-url",
        help="Distinct owned loopback Vite server for native workspace/workbench cases",
    )
    parser.add_argument(
        "--axe-source", type=Path, help="Exact locked axe-core auditor for workbench_accessibility"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence: dict[str, object] = {
        "scenario": args.scenario,
        "base_url": "rejected",
        "command": f"studio_browser_journey --scenario {args.scenario}",
    }
    try:
        url = loopback_url(args.base_url)
        evidence["base_url"] = url
        if args.axe_source is not None and args.scenario != "workbench_accessibility":
            raise ValueError("--axe-source is valid only for workbench_accessibility")
        if args.scenario in (
            "workspace_recovery",
            "workbench_navigation",
            "parameter_graph_editor",
            "workbench_accessibility",
        ):
            if args.workspace_source_url is None:
                raise ValueError(f"{args.scenario} requires --workspace-source-url")
            if args.scenario == "workbench_accessibility":
                from tools.studio_accessibility_browser import run_accessibility_journey

                run_accessibility_journey(
                    url, args.workspace_source_url, args.axe_source, evidence
                )
            elif args.scenario == "parameter_graph_editor":
                from tools.studio_parameter_browser_journey import run_parameter_journey

                run_parameter_journey(url, args.workspace_source_url, evidence)
            elif args.scenario == "workbench_navigation":
                from tools.studio_workbench_browser_journey import run_workbench_journey

                run_workbench_journey(url, args.workspace_source_url, evidence)
            else:
                from tools.studio_workspace_browser_journey import run_workspace_journey

                run_workspace_journey(url, args.workspace_source_url, evidence)
        else:
            if args.workspace_source_url is not None and args.scenario not in (
                "local_experiment_journey",
                "result_value_inspector",
                "immutable_run_comparison",
                "experiment_workflow_runner",
            ):
                raise ValueError(
                    "--workspace-source-url is valid only for workspace_recovery/workbench_navigation/parameter_graph_editor/workbench_accessibility/local_experiment_journey/result_value_inspector/immutable_run_comparison/experiment_workflow_runner"
                )
            journey: Callable[[str], dict[str, object]]
            if args.scenario == "workspace_panel_refusal":
                from tools.studio_workspace_browser_journey import run_panel_refusal_journey

                journey = run_panel_refusal_journey
            elif args.scenario == "resource_plan_projection":
                from tools.studio_resource_browser_journey import run_resource_journey

                journey = run_resource_journey
            elif args.scenario == "owned_kernel_worker":
                from tools.studio_owned_worker_journey import run_owned_worker_journey

                journey = partial(run_owned_worker_journey, evidence=evidence)
            elif args.scenario == "result_value_inspector":
                from tools.studio_result_browser import run_result_journey

                journey = partial(
                    run_result_journey, source_url=args.workspace_source_url, evidence=evidence
                )
            elif args.scenario == "experiment_workflow_runner":
                from tools.studio_workflow_browser import run_workflow_journey

                journey = partial(
                    run_workflow_journey, source_url=args.workspace_source_url, evidence=evidence
                )
            elif args.scenario == "immutable_run_comparison":
                from tools.studio_comparison_browser import run_comparison_journey

                journey = partial(
                    run_comparison_journey,
                    source_url=args.workspace_source_url,
                    evidence=evidence,
                )
            elif args.scenario == "local_experiment_journey":
                from tools.studio_local_experiment_browser import run_local_experiment_journey

                journey = partial(
                    run_local_experiment_journey,
                    source_url=args.workspace_source_url,
                    evidence=evidence,
                )
            elif args.scenario == "capability_catalogue":
                journey = run_catalogue_journey
            elif args.scenario == "program_authoring":
                from tools.studio_program_authoring_browser import run_program_authoring_journey

                journey = partial(run_program_authoring_journey, evidence=evidence)
            elif args.scenario == "operator_backend_profiles":
                from tools.studio_backend_profiles_browser import run_backend_profiles_journey

                journey = partial(run_backend_profiles_journey, evidence=evidence)
            elif args.scenario == "operator_policy_decisions":
                from tools.studio_operator_policy_browser import run_operator_policy_journey

                journey = partial(run_operator_policy_journey, evidence=evidence)
            elif args.scenario == "operator_review_dossiers":
                from tools.studio_operator_dossier_browser import run_operator_dossier_journey

                journey = partial(run_operator_dossier_journey, evidence=evidence)
            elif args.scenario == "compiler_trace_inspector":
                from tools.studio_compiler_trace_browser import run_compiler_trace_journey

                journey = partial(run_compiler_trace_journey, evidence=evidence)
            else:
                journey = run_evidence_journey
            evidence.update(journey(url))
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
