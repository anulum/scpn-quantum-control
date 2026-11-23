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
The catalogue and evidence journeys honour the selected existing browser executable
and retain Chromium sandboxing.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Locator, Page, Route


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


_CATALOGUE_RECOMPUTED = "recomputed digest matches the signed claim"


@dataclass
class _CatalogueTransport:
    """Keep one catalogue context on its preview and optionally refuse every kernel."""

    origin: str
    unavailable: bool
    refused: list[str] = field(default_factory=list)

    def __call__(self, route: Route) -> None:
        """Refuse outside requests and, without kernels, every ``.wasm`` request.

        Parameters
        ----------
        route
            Intercepted request.

        """
        target = urlsplit(route.request.url)
        if target.scheme != "http" or target.netloc != self.origin:
            self.refused.append("outside-preview")
            route.abort()
        elif self.unavailable and target.path.endswith(".wasm"):
            route.abort()
        else:
            route.continue_()


def _filter_compile(catalogue: Locator) -> None:
    """Filter the catalogue to the compile task on the browser WASM runtime.

    Parameters
    ----------
    catalogue
        Capability catalogue region.

    """
    catalogue.get_by_label("Capability task").fill("compile")
    catalogue.get_by_label("Capability runtime").select_option("browser-wasm")


def _recompute_from_catalogue(page: Page, url: str, catalogue: Locator) -> str:
    """Follow the compile link by keyboard, recompute in the browser and return.

    Parameters
    ----------
    page
        Owned page.
    url
        Owned preview.
    catalogue
        Capability catalogue region.

    Returns
    -------
    str
        The recomputation status.

    """
    from playwright.sync_api import expect

    link = catalogue.get_by_role("link", name="Open XY compile recomputation")
    expect(link).to_be_visible()
    link.focus()
    page.keyboard.press("Enter")
    panel = page.locator('[id="/build/compile-recompute"]')
    expect(panel).to_be_focused()
    expect(page).to_have_url(url + "#/build/compile-recompute")
    panel.get_by_role("button", name="Recompute in browser").click()
    expect(panel.get_by_role("status")).to_have_text(_CATALOGUE_RECOMPUTED)
    outcome = panel.get_by_role("status").inner_text()
    page.get_by_role("navigation", name="Workbench views").get_by_role(
        "link", name="Workspace", exact=True
    ).click()
    _filter_compile(catalogue)
    return outcome


def _catalogue_case(page: Page, url: str, missing_kernel: bool) -> tuple[str, str]:
    """Filter, recompute or refuse, then find no match and clear the filters.

    Parameters
    ----------
    page
        Owned page.
    url
        Owned preview.
    missing_kernel
        Whether every kernel request is refused in this context.

    Returns
    -------
    tuple[str, str]
        The outcome and the catalogue identity.

    """
    from playwright.sync_api import expect

    page.goto(url, wait_until="networkidle")
    catalogue = page.get_by_role("region", name="Capability catalogue")
    identity = catalogue.locator(".qsp-digest code").inner_text()
    execute = catalogue.get_by_test_id("capability-execute").get_by_role("link")
    expect(execute).to_have_count(0)
    _filter_compile(catalogue)
    catalogue.get_by_label("Capability backend").select_option("rust")
    compile_row = catalogue.get_by_test_id("capability-compile")
    expect(compile_row).to_have_count(1)
    expect(catalogue.locator("li")).to_have_count(1)
    if missing_kernel:
        link = catalogue.get_by_role("link", name="Open XY compile recomputation")
        expect(link).to_have_count(0)
        boundary = compile_row.locator(".qsp-boundary")
        expect(boundary).not_to_have_text("Backend availability unknown")
        outcome = "missing-kernel-refused"
    else:
        outcome = _recompute_from_catalogue(page, url, catalogue)
    expect(catalogue.locator(".qsp-digest code")).to_have_text(identity)
    catalogue.get_by_label("Capability backend").select_option("numpy")
    expect(catalogue.get_by_text("No capability matches these filters.")).to_be_visible()
    catalogue.get_by_role("button", name="Clear filters").click()
    expect(catalogue.locator("li")).to_have_count(9)
    return outcome, identity


def _catalogue_context(browser: Browser, url: str, missing_kernel: bool) -> dict[str, object]:
    """Run one catalogue case in its own context and check its custody and disposal.

    Parameters
    ----------
    browser
        Launched browser.
    url
        Owned preview.
    missing_kernel
        Whether every kernel request is refused in this context.

    Returns
    -------
    dict[str, object]
        The case observation.

    """
    from tools.studio_owned_worker_journey import wait_for_worker_disposal

    context = browser.new_context(service_workers="block")
    context.set_default_timeout(15_000)
    transport = _CatalogueTransport(urlsplit(url).netloc, missing_kernel)
    errors: list[str] = []
    try:
        context.route("**/*", transport)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        outcome, identity = _catalogue_case(page, url, missing_kernel)
        assert not transport.refused, transport.refused
        assert not errors, errors
        wait_for_worker_disposal(page, observe_termination=False)
        return {
            "missing_kernel": missing_kernel,
            "outcome": outcome,
            "identity": identity,
            "workers": len(page.workers),
        }
    finally:
        context.close()


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
    from playwright.sync_api import sync_playwright

    url = loopback_url(base_url)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        try:
            observations = [
                _catalogue_context(browser, url, missing_kernel)
                for missing_kernel in (False, True, False)
            ]
            return {
                "scenario": "capability_catalogue",
                "playwright": version("playwright"),
                "browser": browser.version,
                "observations": observations,
            }
        finally:
            browser.close()


_EVIDENCE_SOURCE = (
    Path(__file__).resolve().parents[1] / "data/studio/program_ad_replay_rational_20260714.json"
)
_DIGEST_PROBE = """
    window.__studioEvidenceDigests = 0;
    const digest = crypto.subtle.digest.bind(crypto.subtle);
    Object.defineProperty(crypto.subtle, "digest", {
        value: (...args) => digest(...args).finally(() => {
            window.__studioEvidenceDigests += 1;
        })
    });
"""
_NEGATIVE_EVIDENCE = {
    "schema": "studio.evidence-replay.v1",
    "prov": {"entity": {"id": "synthetic-negative-presentation", "digest": "sha256:" + "a" * 64}},
    "evidence_kind": "falsified",
    "claim_boundary": {"status": "refuted", "admission": "rejected"},
    "freshness": "traceable-unchecked",
    "attestation": {"signature": "synthetic-unverified-metadata"},
}
_MISSING_EVIDENCE = (
    "Missing schema",
    "Missing source",
    "Missing seal",
    "Partial or unsupported evidence",
)


def _is_program_ad_kernel(url: str) -> bool:
    """Tell whether a request loads the program-AD WASM kernel.

    Parameters
    ----------
    url
        Request URL.

    Returns
    -------
    bool
        Whether the path names the program-AD kernel.

    """
    path = urlsplit(url).path
    return Path(path).name.startswith("scpn_quantum_studio_program_ad_wasm") and path.endswith(
        ".wasm"
    )


@dataclass
class _EvidenceTransport:
    """Keep the evidence context on its preview and hold the next program-AD kernel."""

    origin: str
    rejected: list[str] = field(default_factory=list)
    held: list[Route] = field(default_factory=list)
    hold_next: bool = False

    def __call__(self, route: Route) -> None:
        """Refuse outside requests, hold the armed kernel request, pass the rest.

        Parameters
        ----------
        route
            Intercepted request.

        """
        target = urlsplit(route.request.url)
        if target.scheme != "http" or target.netloc != self.origin:
            self.rejected.append("outside-preview")
            route.abort()
        elif self.hold_next and _is_program_ad_kernel(route.request.url):
            self.hold_next = False
            self.held.append(route)
        else:
            route.continue_()


@dataclass(frozen=True)
class _Evidence:
    """Evidence viewer controls and the observation recorder of one journey."""

    page: Page
    viewer: Locator
    observations: list[str]
    audit: Callable[[Page, str], None] | None

    @property
    def inspector(self) -> Locator:
        """Return the evidence inspector region.

        Returns
        -------
        Locator
            The inspector.

        """
        return self.viewer.get_by_role("region", name="Evidence inspector", exact=True)

    def inspect(self, text: str) -> None:
        """Paste evidence text and inspect it.

        Parameters
        ----------
        text
            Evidence JSON text.

        """
        self.viewer.get_by_label("Evidence JSON").fill(text)
        self.viewer.get_by_role("button", name="Inspect snapshot").click()

    def record(self, state: str) -> None:
        """Retain the observation and audit the live page when an auditor is set.

        Parameters
        ----------
        state
            Observed state.

        """
        self.observations.append(state)
        if self.audit is not None:
            self.audit(self.page, state)


def _check_delayed_verification(
    evidence: _Evidence, transport: _EvidenceTransport, changed: dict[str, Any]
) -> None:
    """Hold A's kernel, verify a changed claim B, then let A resolve last without effect.

    Parameters
    ----------
    evidence
        Evidence viewer controls.
    transport
        Route state that holds the next program-AD kernel.
    changed
        The original evidence with one expected gradient changed.

    """
    from playwright.sync_api import expect

    page, inspector = evidence.page, evidence.inspector
    # Hold a real kernel response for A, finish B, then allow A to resolve last.
    transport.hold_next = True
    inspector.get_by_role("button").click()
    expect(inspector.get_by_role("button")).to_have_text("Recomputing…")
    if evidence.audit is not None:
        evidence.audit(page, "actual-original-verification-pending")
    evidence.inspect(json.dumps(changed))
    expect(inspector.get_by_role("status")).to_have_count(0)
    inspector.get_by_role("button").click()
    expect(inspector.get_by_role("status")).to_have_attribute("data-verdict", "mismatch")
    assert len(transport.held) == 1, "Expected exactly one held original kernel response"
    completed_digests = page.evaluate("window.__studioEvidenceDigests")
    with page.expect_request_finished(lambda request: _is_program_ad_kernel(request.url)):
        transport.held.pop().continue_()
    page.wait_for_function(
        "previous => window.__studioEvidenceDigests > previous", arg=completed_digests
    )
    expect(inspector.get_by_role("status")).to_have_attribute("data-verdict", "mismatch")
    evidence.record("same-id-changed-claim-retains-B-after-delayed-A")


def _check_refusals(evidence: _Evidence, original_text: str) -> None:
    """Refuse an altered source, keep a falsified status and show every missing part.

    Parameters
    ----------
    evidence
        Evidence viewer controls.
    original_text
        Exact original evidence text.

    """
    from playwright.sync_api import expect

    inspector = evidence.inspector
    original = json.loads(original_text)
    tampered = json.loads(original_text)
    # Flip one original bit; this always changes the actual fixture bytes.
    changed_byte = int(original["input_hex"][-2:], 16) ^ 1
    tampered["input_hex"] = original["input_hex"][:-2] + f"{changed_byte:02x}"
    evidence.inspect(json.dumps(tampered))
    inspector.get_by_role("button").click()
    status = inspector.get_by_role("status")
    expect(status).to_have_attribute("data-verdict", "unverifiable")
    expect(status).to_contain_text("SHA-256 binding")
    evidence.record("altered-source-digest-refused")
    evidence.inspect(json.dumps(_NEGATIVE_EVIDENCE))
    for shown in ("falsified", "refuted", "Seal present — not verified"):
        expect(inspector.get_by_text(shown, exact=True)).to_be_visible()
    expect(inspector.get_by_role("button")).to_have_count(0)
    expect(inspector.get_by_role("status")).to_have_count(0)
    evidence.record("attested-falsification-retains-source-status")
    evidence.inspect("{}")
    for message in _MISSING_EVIDENCE:
        expect(inspector.get_by_text(message, exact=True)).to_be_visible()
    evidence.record("missing-source-schema-seal-visible")


def _walk_evidence(evidence: _Evidence, url: str, transport: _EvidenceTransport) -> None:
    """Match the original source, then cross every changed, refused and malformed state.

    Parameters
    ----------
    evidence
        Evidence viewer controls.
    url
        Owned preview.
    transport
        Route state that holds the next program-AD kernel.

    """
    from playwright.sync_api import expect

    original_text = _EVIDENCE_SOURCE.read_text(encoding="utf-8")
    changed = json.loads(original_text)
    changed["expected"]["gradient"][1] = 99.0
    evidence.page.goto(url, wait_until="networkidle")
    evidence.inspect(original_text)
    evidence.inspector.get_by_role("button").click()
    status = evidence.inspector.get_by_role("status")
    expect(status).to_have_attribute("data-verdict", "match")
    evidence.record("original-source-real-wasm-match")
    _check_delayed_verification(evidence, transport, changed)
    _check_refusals(evidence, original_text)
    evidence.inspect("{")
    viewer = evidence.viewer
    expect(viewer.get_by_role("alert")).to_contain_text("Cannot inspect evidence")
    expect(viewer.get_by_role("region", name="Evidence inspector")).to_have_count(0)
    if evidence.audit is not None:
        evidence.audit(evidence.page, "malformed-evidence-retains-raw-input")


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
    from playwright.sync_api import sync_playwright

    url = loopback_url(base_url)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=os.environ.get("STUDIO_NATIVE_BROWSER_EXECUTABLE"),
            chromium_sandbox=True,
        )
        try:
            observations = _evidence_context(browser, url, color_scheme, audit)
            return {
                "scenario": "evidence_inspector",
                "playwright": version("playwright"),
                "browser": browser.version,
                "observations": observations,
            }
        finally:
            browser.close()


def _evidence_context(
    browser: Browser,
    url: str,
    color_scheme: Literal["light", "dark"],
    audit: Callable[[Page, str], None] | None,
) -> list[str]:
    """Run the evidence walk in its own themed context and check its custody.

    Parameters
    ----------
    browser
        Launched browser.
    url
        Owned preview.
    color_scheme
        Native browser theme.
    audit
        Optional actual-page audit at each observed evidence state.

    Returns
    -------
    list[str]
        The recorded observations.

    """
    context = browser.new_context(service_workers="block", color_scheme=color_scheme)
    context.set_default_timeout(15_000)
    transport = _EvidenceTransport(urlsplit(url).netloc)
    errors: list[str] = []
    try:
        context.route("**/*", transport)
        page = context.new_page()
        page.add_init_script(_DIGEST_PROBE)
        page.on("pageerror", lambda error: errors.append(str(error)))
        viewer = page.get_by_role("region", name="Inspect evidence JSON", exact=True)
        evidence = _Evidence(page, viewer, [], audit)
        _walk_evidence(evidence, url, transport)
        assert not errors, errors
        assert not transport.rejected, transport.rejected
        assert not page.workers, "Evidence inspection must not leak a worker"
        return evidence.observations
    finally:
        for route in transport.held:
            route.abort()
        context.close()


_SOURCE_REQUIRED: dict[str, tuple[str, str]] = {
    "workspace_recovery": ("tools.studio_workspace_browser_journey", "run_workspace_journey"),
    "workbench_navigation": ("tools.studio_workbench_browser_journey", "run_workbench_journey"),
    "parameter_graph_editor": ("tools.studio_parameter_browser_journey", "run_parameter_journey"),
    "workbench_accessibility": ("tools.studio_accessibility_browser", "run_accessibility_journey"),
}
_SOURCE_OPTIONAL: dict[str, tuple[str, str]] = {
    "local_experiment_journey": (
        "tools.studio_local_experiment_browser",
        "run_local_experiment_journey",
    ),
    "result_value_inspector": ("tools.studio_result_browser", "run_result_journey"),
    "immutable_run_comparison": ("tools.studio_comparison_browser", "run_comparison_journey"),
    "method_comparison_instrument": (
        "tools.studio_method_comparison_browser",
        "run_method_comparison_journey",
    ),
    "experiment_workflow_runner": ("tools.studio_workflow_browser", "run_workflow_journey"),
}
_WITH_EVIDENCE: dict[str, tuple[str, str]] = {
    "owned_kernel_worker": ("tools.studio_owned_worker_journey", "run_owned_worker_journey"),
    "program_authoring": (
        "tools.studio_program_authoring_browser",
        "run_program_authoring_journey",
    ),
    "operator_backend_profiles": (
        "tools.studio_backend_profiles_browser",
        "run_backend_profiles_journey",
    ),
    "operator_policy_decisions": (
        "tools.studio_operator_policy_browser",
        "run_operator_policy_journey",
    ),
    "operator_review_dossiers": (
        "tools.studio_operator_dossier_browser",
        "run_operator_dossier_journey",
    ),
    "provider_observation_journal": (
        "tools.studio_provider_observation_browser",
        "run_provider_observation_journey",
    ),
    "gradient_debugger": (
        "tools.studio_gradient_debugger_browser",
        "run_gradient_debugger_journey",
    ),
    "qec_resource_instrument": ("tools.studio_qec_resource_browser", "run_qec_resource_journey"),
    "mitigation_inspector": ("tools.studio_mitigation_browser", "run_mitigation_journey"),
    "local_job_service": ("tools.studio_local_jobs_browser", "run_local_jobs_journey"),
    "sensing_entropy_instrument": ("tools.studio_sensing_browser", "run_sensing_journey"),
    "control_proposal_instrument": ("tools.studio_control_browser", "run_control_journey"),
    "learning_domain_instrument": ("tools.studio_learning_browser", "run_learning_journey"),
    "topology_symmetry_instrument": ("tools.studio_topology_browser", "run_topology_journey"),
    "attestation_verification": ("tools.studio_attestation_browser", "run_attestation_journey"),
    "benchmark_databank": ("tools.studio_benchmark_browser", "run_benchmark_journey"),
    "atlas_graph_navigation": ("tools.studio_atlas_browser", "run_atlas_journey"),
    "atlas_evidence_lineage": (
        "tools.studio_atlas_lineage_browser",
        "run_atlas_lineage_journey",
    ),
    "atlas_studio_selection": (
        "tools.studio_atlas_selection_browser",
        "run_atlas_selection_journey",
    ),
    "capability_resolver": ("tools.studio_capability_browser", "run_capability_journey"),
    "calibration_snapshot_diff": (
        "tools.studio_calibration_browser",
        "run_calibration_journey",
    ),
    "pulse_feasibility": ("tools.studio_pulse_browser", "run_pulse_journey"),
    "compiler_trace_inspector": (
        "tools.studio_compiler_trace_browser",
        "run_compiler_trace_journey",
    ),
}
_PLAIN: dict[str, tuple[str, str]] = {
    "workspace_panel_refusal": (
        "tools.studio_workspace_browser_journey",
        "run_panel_refusal_journey",
    ),
    "resource_plan_projection": ("tools.studio_resource_browser_journey", "run_resource_journey"),
}
_SCENARIOS = (
    "capability_catalogue",
    "evidence_inspector",
    "resource_plan_projection",
    "workspace_recovery",
    "workspace_panel_refusal",
    "workbench_navigation",
    "parameter_graph_editor",
    "program_authoring",
    "compiler_trace_inspector",
    "gradient_debugger",
    "qec_resource_instrument",
    "mitigation_inspector",
    "local_job_service",
    "pulse_feasibility",
    "sensing_entropy_instrument",
    "control_proposal_instrument",
    "learning_domain_instrument",
    "topology_symmetry_instrument",
    "attestation_verification",
    "benchmark_databank",
    "atlas_graph_navigation",
    "atlas_evidence_lineage",
    "atlas_studio_selection",
    "capability_resolver",
    "calibration_snapshot_diff",
    "operator_backend_profiles",
    "operator_policy_decisions",
    "operator_review_dossiers",
    "provider_observation_journal",
    "workbench_accessibility",
    "owned_kernel_worker",
    "local_experiment_journey",
    "result_value_inspector",
    "immutable_run_comparison",
    "method_comparison_instrument",
    "experiment_workflow_runner",
)
_MISAPPLIED_SOURCE = (
    "--workspace-source-url is valid only for workspace_recovery/workbench_navigation/"
    "parameter_graph_editor/workbench_accessibility/local_experiment_journey/"
    "result_value_inspector/immutable_run_comparison/method_comparison_instrument/"
    "experiment_workflow_runner"
)


def _resolve(owner: tuple[str, str]) -> Callable[..., dict[str, object]]:
    """Import a journey runner only when its scenario is selected.

    Parameters
    ----------
    owner
        Module path and function name of the runner.

    Returns
    -------
    Callable
        The runner.

    """
    module, name = owner
    return cast("Callable[..., dict[str, object]]", getattr(import_module(module), name))


def _run_with_source(args: argparse.Namespace, url: str, evidence: dict[str, object]) -> None:
    """Run a journey that needs the owned source server; it fills the evidence itself.

    Parameters
    ----------
    args
        Parsed command arguments.
    url
        Validated owned preview.
    evidence
        Evidence the journey fills in place.

    Raises
    ------
    ValueError
        The source server is missing.

    """
    if args.workspace_source_url is None:
        raise ValueError(f"{args.scenario} requires --workspace-source-url")
    journey = _resolve(_SOURCE_REQUIRED[args.scenario])
    if args.scenario == "workbench_accessibility":
        journey(url, args.workspace_source_url, args.axe_source, evidence)
    else:
        journey(url, args.workspace_source_url, evidence)


def _journey(
    args: argparse.Namespace, evidence: dict[str, object]
) -> Callable[[str], dict[str, object]]:
    """Select the runner of a scenario that takes only the preview.

    Parameters
    ----------
    args
        Parsed command arguments.
    evidence
        Evidence passed to runners that keep partial evidence.

    Returns
    -------
    Callable[[str], dict[str, object]]
        The runner, bound to its other arguments.

    Raises
    ------
    ValueError
        A source server is supplied to a scenario that does not take one.

    """
    scenario = args.scenario
    if args.workspace_source_url is not None and scenario not in _SOURCE_OPTIONAL:
        raise ValueError(_MISAPPLIED_SOURCE)
    if scenario in _SOURCE_OPTIONAL:
        source = args.workspace_source_url
        return partial(_resolve(_SOURCE_OPTIONAL[scenario]), source_url=source, evidence=evidence)
    if scenario in _WITH_EVIDENCE:
        return partial(_resolve(_WITH_EVIDENCE[scenario]), evidence=evidence)
    if scenario in _PLAIN:
        return _resolve(_PLAIN[scenario])
    return run_catalogue_journey if scenario == "capability_catalogue" else run_evidence_journey


def _dispatch(args: argparse.Namespace, evidence: dict[str, object]) -> None:
    """Validate the preview and the options, then run the selected journey.

    Parameters
    ----------
    args
        Parsed command arguments.
    evidence
        Evidence the journey fills.

    Raises
    ------
    ValueError
        The preview, the auditor option or the source option is invalid.

    """
    url = loopback_url(args.base_url)
    evidence["base_url"] = url
    if args.axe_source is not None and args.scenario != "workbench_accessibility":
        raise ValueError("--axe-source is valid only for workbench_accessibility")
    if args.scenario in _SOURCE_REQUIRED:
        _run_with_source(args, url, evidence)
    else:
        evidence.update(_journey(args, evidence)(url))


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
    parser.add_argument("--scenario", choices=_SCENARIOS, required=True)
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
        _dispatch(args, evidence)
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
