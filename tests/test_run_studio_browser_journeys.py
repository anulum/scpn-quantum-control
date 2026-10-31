# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio browser pipeline ownership
"""Guard actual browser admission, isolated workers and exact owner reports."""

from __future__ import annotations

from pathlib import Path

import yaml

from tools.ci_workflow_inventory import workflow_path_for_job


def test_studio_executes_the_original_browser_scenarios_and_isolated_refusals() -> None:
    """Keep real scientific/UI journeys wired through the executing shell owner."""
    workflow = yaml.safe_load(workflow_path_for_job("studio-web").read_text())
    steps = workflow["jobs"]["studio-web"]["steps"]
    step = next(
        item for item in steps if item.get("name") == "Run the real catalogue browser journey"
    )
    assert step["run"].strip() == "bash tools/run_studio_browser_journeys.sh"
    source = Path("tools/run_studio_browser_journeys.sh").read_text()
    for scenario in (
        "capability_catalogue",
        "evidence_inspector",
        "resource_plan_projection",
        "owned_kernel_worker",
        "workspace_recovery",
        "workbench_navigation",
        "parameter_graph_editor",
        "result_value_inspector",
        "immutable_run_comparison",
        "experiment_workflow_runner",
        "local_experiment_journey",
        "workbench_accessibility",
        "program_authoring",
        "gradient_debugger",
        "compiler_trace_inspector",
        "operator_backend_profiles",
        "operator_policy_decisions",
        "operator_review_dossiers",
        "qec_resource_instrument",
        "mitigation_inspector",
    ):
        assert "--scenario " + scenario in source
    assert "bash tools/run_ci_test_shards.sh studio-browser 2" in source
    for owner in (
        "studio_browser_runtime_refusal",
        "studio_owned_worker_browser_runtime",
        "studio_workbench_browser_journey",
        "studio_parameter_browser_journey",
        "studio_local_experiment_browser_runtime",
        "studio_result_browser_runtime",
        "studio_comparison_browser_runtime",
        "studio_workflow_browser_runtime",
    ):
        assert f"tools/tests/test_{owner}.py" in source
    assert "--concurrency=greenlet" in source
    for journey in ("gradient_debugger", "qec_resource", "mitigation"):
        assert f"tools/tests/test_studio_{journey}_browser.py" in source
    # One exact coverage report per separately measured journey owner.
    assert source.count("--fail-under=100") == 8
    assert "trap " in source and 'wait "$preview_pid"' in source and 'wait "$source_pid"' in source
    evidence = next(
        item for item in steps if item.get("name") == "Preserve browser acceptance evidence"
    )
    assert "${{ runner.temp }}/studio-browser-shards" in evidence["with"]["path"]
