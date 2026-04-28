# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Frontier interface guard tests
"""Tests for fail-fast frontier interfaces."""

from __future__ import annotations

import asyncio
import importlib.util
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from scpn_quantum_control.analysis import RLPulseOptimizer, dla_truncated_tn


def _load_frontier_orchestrator(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = repo_root / "scripts/frontier_campaign_2026"
    monkeypatch.syspath_prepend(str(script_dir))
    for module_name, function_name in {
        "test_quantum_advantage_scaling": "run_advantage_scaling",
        "test_live_scneurocore_loop": "run_live_scneurocore",
        "test_sync_distillation": "run_distillation",
        "test_multi_backend_distributed": "run_multi_backend",
        "test_dla_tensor_network": "run_dla_tn_mapping",
        "test_rl_pulse_optimization": "run_rl_pulse_opt",
        "test_pt_symmetric_kuramoto": "run_pt_symmetric",
        "test_logical_sync_protection": "run_logical_protection",
    }.items():
        module = types.ModuleType(module_name)
        setattr(module, function_name, lambda: None)
        monkeypatch.setitem(sys.modules, module_name, module)

    spec = importlib.util.spec_from_file_location(
        "frontier_campaign_orchestrator_for_test",
        script_dir / "run_frontier_campaign.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_credible_runner(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = repo_root / "scripts/frontier_campaign_2026"
    monkeypatch.syspath_prepend(str(script_dir))
    for module_name, function_name in {
        "test_quantum_advantage_scaling": "run_advantage_scaling",
        "test_multi_backend_distributed": "run_multi_backend",
        "test_pt_symmetric_kuramoto": "run_pt_symmetric",
    }.items():
        module = types.ModuleType(module_name)
        setattr(module, function_name, lambda: None)
        monkeypatch.setitem(sys.modules, module_name, module)

    spec = importlib.util.spec_from_file_location(
        "frontier_credible_runner_for_test",
        script_dir / "run_credible_tests.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dla_truncated_tensor_network_fails_until_implemented():
    K_nm = np.zeros((4, 4), dtype=np.float64)

    with pytest.raises(NotImplementedError, match="not implemented"):
        dla_truncated_tn(K_nm)


def test_rl_pulse_optimizer_fails_until_implemented():
    optimiser = RLPulseOptimizer(runner=object(), target_sync_order=0.5, episodes=1)

    with pytest.raises(NotImplementedError, match="not implemented"):
        asyncio.run(optimiser.optimize_pulses())

    with pytest.raises(NotImplementedError, match="No RL pulse"):
        optimiser.save_results("unused.json")


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/frontier_campaign_2026/mock_injector.py",
        "scripts/hardware_campaign_2026/mock_injector.py",
        "scripts/primary_campaign_2026/mock_injector.py",
        "scripts/sophisticated_campaign_2026/mock_injector.py",
    ],
)
def test_retired_campaign_injectors_fail_fast(relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(RuntimeError, match="Local campaign injectors are retired"):
        runpy.run_path(str(repo_root / relative_path))


def test_frontier_orchestrator_classifies_implementation_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    orchestrator = _load_frontier_orchestrator(monkeypatch)

    def success():
        return None

    def gated():
        raise NotImplementedError("future implementation")

    summary = asyncio.run(
        orchestrator.run_frontier_campaign(
            tests=[("success_case", success), ("gated_case", gated)],
            campaign_dir=tmp_path,
        )
    )

    assert summary["status"] == "completed_with_gates"
    assert summary["counts"] == {
        "success": 1,
        "implementation_gated": 1,
        "failed": 0,
    }
    assert summary["tests"]["gated_case"]["status"] == "implementation_gated"
    assert Path(summary["summary_path"]).exists()


def test_frontier_shell_launcher_delegates_to_python_orchestrator():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts/frontier_campaign_2026/run_frontier_campaign.sh"
    text = launcher.read_text(encoding="utf-8")

    assert "python3 run_frontier_campaign.py" in text
    assert "test_dla_tensor_network.py" not in text
    assert "test_rl_pulse_optimization.py" not in text


def test_credible_runner_summary_reflects_failed_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("SCPN_IBM_TOKEN", "test-token")
    runner = _load_credible_runner(monkeypatch)

    def success():
        return None

    def failure():
        raise RuntimeError("hardware unavailable")

    summary = asyncio.run(
        runner.run_credible_tests(
            tests=[("success_case", success), ("failure_case", failure)],
            results_dir=tmp_path,
        )
    )

    assert summary["status"] == "completed_with_failures"
    assert summary["counts"] == {"success": 1, "failed": 1}
    assert summary["tests"]["failure_case"]["status"] == "failed"
    assert Path(summary["summary_path"]).exists()


def test_credible_runner_default_results_are_campaign_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("SCPN_IBM_TOKEN", "test-token")
    runner = _load_credible_runner(monkeypatch)
    monkeypatch.setattr(runner, "campaign_path", lambda *parts: tmp_path.joinpath(*parts))

    def success():
        return None

    summary = asyncio.run(runner.run_credible_tests(tests=[("success_case", success)]))

    summary_path = Path(summary["summary_path"])
    assert summary_path.exists()
    assert summary_path.parent == tmp_path / "results"


def test_retrieve_all_jobs_discovers_campaign_local_results(tmp_path: Path):
    repo_root = tmp_path
    root_result = repo_root / "results" / "root.json"
    campaign_result = (
        repo_root / "scripts" / "frontier_campaign_2026" / "results" / "frontier.json"
    )
    campaign_summary = (
        repo_root
        / "scripts"
        / "frontier_campaign_2026"
        / "results"
        / "frontier_campaign"
        / "summary.json"
    )
    for path in (root_result, campaign_result, campaign_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    repo_script = Path(__file__).resolve().parents[1] / "scripts" / "retrieve_all_jobs.py"
    spec = importlib.util.spec_from_file_location("retrieve_all_jobs_for_test", repo_script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    discovered = {
        path.relative_to(repo_root) for path in module._discover_result_json_files(repo_root)
    }

    assert discovered == {
        Path("results/root.json"),
        Path("scripts/frontier_campaign_2026/results/frontier.json"),
        Path("scripts/frontier_campaign_2026/results/frontier_campaign/summary.json"),
    }
