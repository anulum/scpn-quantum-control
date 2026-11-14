# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — gradient debugger through the built Studio and genuine WASM
"""Exercise the gradient debugger in a real browser against the shipped WASM kernel."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from tools.studio_preview_custody import run_custodied_journey

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

RY_GRAD = "ry::pauli_expectation::statevector::grad::native"
HESSIAN = "ry::pauli_expectation::qasm_simulator::hessian::native"
JAX = "ry::pauli_expectation::statevector::value_and_grad::jax"


def _check_debugger(debugger: Locator, observations: list[str], kernels: list[str]) -> None:
    """Refuse, prepare and recompute cells in the debugger.

    Parameters
    ----------
    debugger
        The rendered gradient debugger region.
    observations
        Caller-owned list that records each contract once it holds.
    kernels
        Record of loaded ``.wasm`` paths.

    """
    from playwright.sync_api import expect

    selector = debugger.get_by_label("Objective, transform, backend and framework")
    selector.select_option(HESSIAN)
    expect(debugger.get_by_role("region", name="Refused before execution")).to_contain_text(
        "hessian support is limited to deterministic local backends"
    )
    expect(debugger.get_by_role("button", name="Recompute in browser")).to_have_count(0)
    observations.append("higher-order-refused-before-execution")
    before = len(kernels)
    selector.select_option(JAX)
    expect(debugger.get_by_role("note")).to_have_text(
        "No jax runtime runs in this browser; it is not emulated."
    )
    expect(debugger.get_by_role("region", name="Local runner request")).to_contain_text(
        "scpn-studio-run differentiate --action-id gradient-debugger-ry-z-expectation"
    )
    expect(debugger.get_by_role("table", name="Gradient comparison")).to_have_count(0)
    assert len(kernels) == before, "An unavailable runtime must not load a kernel"
    observations.append("jax-unavailable-local-runner-prepared")
    selector.select_option("two-angle-mix")
    active = debugger.get_by_role("status", name="Active inputs")
    expect(active).to_have_text("Active inputs: a, b")
    debugger.get_by_label("b trainable").uncheck()
    expect(active).to_have_text("Active inputs: a")
    observations.append("trainable-mask-changes-active-inputs")
    selector.select_option(RY_GRAD)
    debugger.get_by_role("button", name="Recompute in browser").click()
    expect(debugger.get_by_role("status", name="Browser recompute")).to_have_text(
        "The browser kernel agrees with the source within the float64 budget."
    )
    assert any("program_ad" in path for path in kernels), kernels
    observations.append("ry-z-gradient-recomputed-by-shipped-wasm")


def _walk(
    page: Page, url: str, observations: list[str], *, kernels: list[str]
) -> dict[str, object]:
    """Debug the declared cells, follow the deep link and return to the workspace.

    Parameters
    ----------
    page
        Owned page bound to the preview.
    url
        Owned literal-loopback preview.
    observations
        Caller-owned list that records each contract once it holds.
    kernels
        Record of loaded ``.wasm`` paths.

    Returns
    -------
    dict
        No extra evidence fields.

    """
    from playwright.sync_api import expect

    page.goto(url, wait_until="networkidle")
    page.get_by_label("Workspace archive JSON").fill('{"unsaved":"exact retained α draft"}')
    views = page.get_by_role("navigation", name="Workbench views")
    views.get_by_role("link", name="Analyse", exact=True).click()
    debugger = page.get_by_role("region", name="Gradient debugger", exact=True)
    expect(debugger).to_be_visible()
    selector = debugger.get_by_label("Objective, transform, backend and framework")
    expect(selector).to_have_value(RY_GRAD)
    observations.append("declared-cell-selected-by-default")
    _check_debugger(debugger, observations, kernels)
    page.goto(url + "#/results/gradient-debugger", wait_until="networkidle")
    expect(page.locator('[id="/results/gradient-debugger"]')).to_be_focused()
    observations.append("deep-link-focuses-the-debugger")
    views.get_by_role("link", name="Workspace", exact=True).click()
    expect(page.get_by_label("Workspace archive JSON")).to_be_visible()
    observations.append("workspace-reachable-after-debugging")
    return {}


def run_gradient_debugger_journey(
    base_url: str, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Select, refuse, prepare and recompute through the production Analyse stage.

    Parameters
    ----------
    base_url
        Owned literal-loopback preview carrying the actual built Studio and WASM.
    evidence
        Caller-owned partial evidence retained on a later runtime refusal.

    Returns
    -------
    dict
        Browser/runtime identity, every observed contract and the absence of
        workers, submissions, outside requests and page errors.

    Raises
    ------
    ValueError
        The preview violates the shared loopback ownership contract.
    AssertionError
        A real UI, kernel, network or runtime contract fails.

    """
    kernels: list[str] = []
    return run_custodied_journey(
        base_url,
        evidence,
        "gradient_debugger",
        partial(_walk, kernels=kernels),
        kernels=kernels,
    )
