# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original accessibility runtime acceptance
"""Check locked engine custody and its actual findings on the production page."""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from tools.studio_accessibility_checks import audit_page, keyboard_route, read_auditor
from tools.tests.test_studio_accessibility_browser_runtime import axe_source as axe_source
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host
from tools.tests.test_studio_program_authoring_browser import program_bundle as program_bundle


def test_locked_auditor_refuses_substituted_engine_before_execution(tmp_path: Path) -> None:
    """Refuse a real file that would otherwise fabricate an empty findings report.

    Parameters
    ----------
    tmp_path
        Owned source file with deliberately changed auditor bytes.

    """
    source = tmp_path / "axe.min.js"
    source.write_text("window.axe={version:'4.13.0',run:async()=>({violations:[]})};")
    with pytest.raises(ValueError, match="locked axe-core source"):
        read_auditor(source)


def test_actual_engine_retains_default_rules_and_rejects_an_unnamed_action(
    program_bundle: Path, axe_source: Path
) -> None:
    """Keep complete real reports before and after an actual accessibility defect.

    Parameters
    ----------
    program_bundle
        Original built production page and unchanged WASM artifacts.
    axe_source
        Existing exact package source admitted by its content digest.

    """
    reports: list[dict[str, object]] = []
    with owned_fault_host(program_bundle) as preview, sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            with browser.new_context(service_workers="block") as context:
                page = context.new_page()
                page.goto(preview, wait_until="networkidle")
                auditor = read_auditor(axe_source)
                audit_page(page, auditor, "actual-original-page", reports)
                page.evaluate("""() => {
                    const action = document.createElement('button');
                    action.type = 'button'; action.style.cssText = 'width:50px;height:30px';
                    document.body.append(action);
                }""")
                with pytest.raises(AssertionError, match="button-name"):
                    audit_page(page, auditor, "actual-unnamed-action", reports)
        finally:
            browser.close()
    assert len(reports) == 2
    for record in reports:
        report = record["report"]
        assert isinstance(report, dict)
        assert "passes" in report and "incomplete" in report
    last = reports[-1]["report"]
    assert isinstance(last, dict) and last["violations"]


def test_native_keyboard_refuses_a_real_focus_trap(program_bundle: Path) -> None:
    """Refuse a visible enabled control made unreachable by a real focus handler.

    Parameters
    ----------
    program_bundle
        Original production page, retained unchanged by the owned HTTP host.

    """
    with owned_fault_host(program_bundle) as preview, sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            with browser.new_context(service_workers="block") as context:
                page = context.new_page()
                page.goto(preview, wait_until="networkidle")
                page.evaluate("""() => {
                    const origin = document.querySelector('.qsp-skip');
                    const blocked = [...document.querySelectorAll('button')].find(button => button.textContent === 'Keyboard help');
                    blocked.addEventListener('focus', () => origin.focus());
                }""")
                with pytest.raises(AssertionError, match="expected"):
                    keyboard_route(page)
        finally:
            browser.close()
