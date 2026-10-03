# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original accessibility runtime acceptance
"""Exercise original route audits, native values and actual transport refusals."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from tools.studio_accessibility_browser import run_accessibility_journey
from tools.studio_accessibility_checks import AXE_SHA256, AXE_VERSION, read_auditor
from tools.tests.test_studio_browser_runtime_refusal import owned_fault_host
from tools.tests.test_studio_parameter_browser_journey import parameter_source as parameter_source
from tools.tests.test_studio_program_authoring_browser import program_bundle as program_bundle


@pytest.fixture
def axe_source() -> Path:
    """Require the locked auditor already present in the declared environment.

    Returns
    -------
    Path
        Existing exact package asset; this fixture never installs dependencies.

    """
    proposed = os.environ.get("STUDIO_AXE_SOURCE")
    source = (
        Path(proposed)
        if proposed
        else Path(__file__).resolve().parents[2] / "studio-web/node_modules/axe-core/axe.min.js"
    )
    read_auditor(source)
    return source


def test_workbench_accessibility_real_routes_values_and_refusal_states(
    program_bundle: Path, parameter_source: str, axe_source: Path
) -> None:
    """Qualify all delivered routes and original stale/refused evidence in both themes.

    Parameters
    ----------
    program_bundle
        Genuine newly built Studio deployment with its original Rust WASM.
    parameter_source
        Distinct owned Vite server containing verbatim production sources.
    axe_source
        Existing admitted auditor with its locked version and content digest.

    """
    with owned_fault_host(program_bundle) as preview:
        observed = run_accessibility_journey(preview, parameter_source, axe_source)
    assert observed["axe_version"] == AXE_VERSION
    assert observed["axe_sha256"] == AXE_SHA256
    assert (
        observed["page_errors"] == observed["external_requests"] == observed["submissions"] == []
    )
    assert (
        observed["workers"]
        == observed["public_screenshots"]
        == observed["user_workspace_exports"]
        == 0
    )
    audits = observed["audits"]
    assert isinstance(audits, list)
    states = {audit["state"] for audit in audits}
    for theme in ("light", "dark"):
        for route in ("build", "workspace", "results", "operations", "experiments", "atlas"):
            assert f"{theme}:{route}:normal" in states
            assert f"{theme}:{route}:css-zoom-200" in states
        for state in (
            "actual-kernel-loading",
            "malformed-route",
            "malformed-evidence",
            "partial-evidence",
            "original-parameter-graph",
            "actual-offline-kernel-refusal",
            "native-compiled-tables:css-zoom-200",
            "native-located-refusal:css-zoom-200",
            "empty-result-filter",
            "original-source-real-wasm-match",
            "actual-original-verification-pending",
            "same-id-changed-claim-retains-B-after-delayed-A",
            "altered-source-digest-refused",
            "attested-falsification-retains-source-status",
            "missing-source-schema-seal-visible",
            "malformed-evidence-retains-raw-input",
        ):
            assert f"{theme}:{state}" in states
        chart = observed[f"{theme}_chart_table"]
        assert isinstance(chart, dict)
        assert chart["samples"] == 301 and chart["phase_columns"] == 12
        assert chart["play_lab_raw_values_exact"] is True
        assert observed[f"{theme}_reduced_motion"] is True
        graph = observed[f"{theme}_graph_table"]
        assert isinstance(graph, dict) and graph["workspace_writes"] == 0
        assert graph["original_archive_unchanged"] is True
    for audit in audits:
        assert "passes" in audit["report"] and "incomplete" in audit["report"]
        assert not any(
            row["impact"] in ("serious", "critical") for row in audit["report"]["violations"]
        )
    keyboard = observed["keyboard"]
    assert isinstance(keyboard, list) and len(keyboard) == 12
    assert all(
        row["enabled_controls"] == row["tab_reached"] and row["native_modal_trap_and_origin"]
        for row in keyboard
    )


@pytest.mark.parametrize("fault", ["page", "external"])
def test_accessibility_cannot_pass_actual_page_or_network_failure(
    program_bundle: Path, parameter_source: str, axe_source: Path, fault: str
) -> None:
    """Keep the real failure while refusing otherwise valid original route data.

    Parameters
    ----------
    program_bundle
        Original production build retained unchanged by the fault host.
    parameter_source
        Separately owned original source origin.
    axe_source
        Original content-verified audit engine.
    fault
        Actual uncaught exception or attempted request outside owned origins.

    """
    evidence: dict[str, object] = {}
    with (
        owned_fault_host(
            program_bundle, page_error=fault == "page", external_request=fault == "external"
        ) as preview,
        pytest.raises(AssertionError),
    ):
        run_accessibility_journey(preview, parameter_source, axe_source, evidence)
    assert evidence["page_errors"] if fault == "page" else evidence["external_requests"]
    assert evidence["audits"], "Retain the actual page audit before refusing the runtime"


@pytest.mark.parametrize("fault", ["unnamed-control", "submission"])
def test_accessibility_refuses_actual_unnamed_control_or_submission(
    program_bundle: Path, parameter_source: str, axe_source: Path, tmp_path: Path, fault: str
) -> None:
    """Refuse a serious engine finding or an actual POST attempted by the loaded page.

    Parameters
    ----------
    program_bundle
        Genuine production artifact retained unchanged.
    parameter_source
        Distinct original source host.
    axe_source
        Locked actual engine with every default audit rule enabled.
    tmp_path
        Owned copy that adds the explicit runtime fault to the original HTML.
    fault
        Real visible unnamed button or attempted POST, never a replacement UI.

    """
    built = shutil.copytree(program_bundle, tmp_path / "built")
    script = (
        "const button=document.createElement('button');button.type='button';button.style.cssText='width:50px;height:30px';document.body.append(button);"
        if fault == "unnamed-control"
        else "fetch('/',{method:'POST',body:'owned-refused-submission'}).catch(()=>{});"
    )
    index = built / "index.html"
    index.write_text(
        index.read_text()
        + "<script>addEventListener('DOMContentLoaded',()=>{"
        + script
        + "});</script>"
    )
    evidence: dict[str, object] = {}
    with owned_fault_host(built) as preview, pytest.raises(AssertionError):
        run_accessibility_journey(preview, parameter_source, axe_source, evidence)
    if fault == "submission":
        assert evidence["submissions"] == ["POST"]
    else:
        audits = evidence["audits"]
        assert isinstance(audits, list) and audits
        violations = audits[-1]["report"]["violations"]
        assert any(
            row["id"] == "button-name" and row["impact"] in ("serious", "critical")
            for row in violations
        )
        assert "passes" in audits[-1]["report"] and "incomplete" in audits[-1]["report"]
    json.dumps(evidence, allow_nan=False)
