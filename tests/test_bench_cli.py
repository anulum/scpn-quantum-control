# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — scpn-bench CLI tests
"""Tests for the benchmark reproducibility command line interface."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scpn_quantum_control import bench_cli


def test_methods_selection_excludes_gpu_by_default() -> None:
    harnesses = bench_cli._selected_harnesses("methods", include_gpu=False)

    labels = [harness.label for harness in harnesses]
    assert "methods-rust-core" in labels
    assert "methods-gpu" not in labels
    assert "methods-ansatz-scaling-tn" not in labels
    assert all("fim" not in harness.groups for harness in harnesses)


def test_methods_selection_can_include_gpu() -> None:
    harnesses = bench_cli._selected_harnesses("methods", include_gpu=True)

    labels = [harness.label for harness in harnesses]
    assert "methods-gpu" in labels


def test_methods_selection_can_include_scaling() -> None:
    harnesses = bench_cli._selected_harnesses(
        "methods",
        include_gpu=False,
        include_scaling=True,
    )

    labels = [harness.label for harness in harnesses]
    assert "methods-ansatz-scaling-tn" in labels


def test_all_selection_includes_methods_and_fim() -> None:
    harnesses = bench_cli._selected_harnesses("all", include_gpu=False)

    labels = {harness.label for harness in harnesses}
    assert "methods-rust-core" in labels
    assert "fim-spectrum" in labels
    assert "methods-gpu" not in labels
    assert "methods-ansatz-scaling-tn" not in labels
    assert "fim-readout-matrix-mitigation" not in labels


def test_fim_selection_can_include_readout_matrix_mitigation() -> None:
    harnesses = bench_cli._selected_harnesses(
        "fim",
        include_gpu=False,
        include_readout=True,
    )

    labels = [harness.label for harness in harnesses]
    assert "fim-readout-matrix-mitigation" in labels


def test_s1_feedback_selection_is_no_qpu_latency_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s1", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s1-feedback-loop",
            "scripts/benchmark_s1_feedback_loop.py",
            frozenset({"s1"}),
        )
    ]


def test_s1_feedback_ready_selection_is_bundle_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s1-ready", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s1-feedback-readiness",
            "scripts/reproduce_s1_feedback_readiness.py",
            frozenset({"s1-ready"}),
        )
    ]


def test_realtime_control_e2e_selection_is_no_qpu_e2e_harness() -> None:
    harnesses = bench_cli._selected_harnesses("realtime-e2e", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "realtime-control-e2e",
            "scripts/benchmark_realtime_control_e2e.py",
            frozenset({"realtime-e2e"}),
        )
    ]


def test_s2_scaling_lite_selection_includes_protocol_and_lite_rows() -> None:
    harnesses = bench_cli._selected_harnesses("s2", include_gpu=False)

    labels = [harness.label for harness in harnesses]

    assert labels == ["s2-scaling-protocol", "s2-scaling-lite", "s2-claim-boundary"]


def test_s2_tn_mps_design_selection_is_design_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s2-tn-design", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s2-tn-mps-baseline-design",
            "scripts/export_tn_mps_baseline_design.py",
            frozenset({"s2-tn-design"}),
        )
    ]


def test_s2_tn_crossover_stage1_selection_is_design_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s2-tn-stage1", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s2-tn-crossover-stage1",
            "scripts/export_tn_mps_crossover_stage1.py",
            frozenset({"s2-tn-stage1"}),
        )
    ]


def test_knm_josephson_magnitude_study_selection_is_design_harness() -> None:
    harnesses = bench_cli._selected_harnesses("knm-josephson-study", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "knm-josephson-magnitude-study",
            "scripts/export_josephson_knm_magnitude_study.py",
            frozenset({"knm-josephson-study"}),
        )
    ]


def test_ground_state_optimizer_convergence_selection_is_harness() -> None:
    harnesses = bench_cli._selected_harnesses("diff-optimizer-convergence", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "ground-state-optimizer-convergence",
            "scripts/export_ground_state_optimizer_convergence.py",
            frozenset({"diff-optimizer-convergence"}),
        )
    ]


def test_p_h1_open_guard_selection_is_claim_guard_harness() -> None:
    harnesses = bench_cli._selected_harnesses("p-h1-open-guard", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "p-h1-open-guard",
            "scripts/check_p_h1_open_claim_guard.py",
            frozenset({"p-h1-open-guard"}),
        )
    ]


def test_s3_design_ready_selection_is_readiness_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s3", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s3-design-readiness",
            "scripts/export_s3_design_readiness.py",
            frozenset({"s3"}),
        )
    ]


def test_s3_design_surrogate_selection_is_surrogate_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s3-surrogate", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s3-design-surrogate",
            "scripts/train_s3_design_surrogate.py",
            frozenset({"s3-surrogate"}),
        )
    ]


def test_s3_ansatz_observables_selection_is_observable_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s3-observables", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s3-ansatz-observables",
            "scripts/validate_s3_ansatz_observables.py",
            frozenset({"s3-observables"}),
        )
    ]


def test_s3_pulse_feasibility_selection_is_probe_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s3-pulse", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s3-pulse-feasibility",
            "scripts/probe_s3_pulse_feasibility.py",
            frozenset({"s3-pulse"}),
        )
    ]


def test_s3_hardware_dossiers_selection_is_dossier_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s3-dossiers", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s3-hardware-dossiers",
            "scripts/export_s3_hardware_dossiers.py",
            frozenset({"s3-dossiers"}),
        )
    ]


def test_s4_multi_hardware_selection_is_readiness_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s4", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s4-multi-hardware-readiness",
            "scripts/export_s4_multi_hardware_readiness.py",
            frozenset({"s4"}),
        )
    ]


def test_s4_provider_preregistration_selection_is_dossier_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s4-provider", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s4-provider-preregistration",
            "scripts/export_s4_provider_preregistration.py",
            frozenset({"s4-provider"}),
        )
    ]


def test_s4_neutral_atom_preregistration_selection_is_dossier_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s4-neutral", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s4-neutral-atom-preregistration",
            "scripts/export_s4_neutral_atom_preregistration.py",
            frozenset({"s4-neutral"}),
        )
    ]


def test_s5_benchmark_suite_selection_is_open_data_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s5", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s5-benchmark-suite",
            "scripts/run_benchmark_suite.py",
            frozenset({"s5"}),
        )
    ]


def test_s5_benchmark_registry_selection_is_registry_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s5-registry", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s5-benchmark-registry",
            "scripts/export_benchmark_registry.py",
            frozenset({"s5-registry"}),
        )
    ]


def test_stable_core_capability_matrix_selection_is_export_harness() -> None:
    harnesses = bench_cli._selected_harnesses("stable-core", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "stable-core-capability-matrix",
            "scripts/export_stable_core_capability_matrix.py",
            frozenset({"stable-core"}),
        )
    ]


def test_stable_core_capability_gate_selection_is_harness() -> None:
    harnesses = bench_cli._selected_harnesses("stable-core-gate", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "stable-core-capability-gate",
            "scripts/run_stable_core_capability_gate.py",
            frozenset({"stable-core-gate"}),
        )
    ]


def test_stable_core_contract_gate_selection_is_harness() -> None:
    harnesses = bench_cli._selected_harnesses("stable-core-contract-gate", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "stable-core-contract-gate",
            "scripts/run_stable_core_contract_gate.py",
            frozenset({"stable-core-contract-gate"}),
        )
    ]


def test_stable_core_preflight_gate_selection_is_harness() -> None:
    harnesses = bench_cli._selected_harnesses("stable-core-preflight-gate", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "stable-core-preflight-gate",
            "scripts/run_stable_core_preflight_gate.py",
            frozenset({"stable-core-preflight-gate"}),
        )
    ]


def test_stable_core_release_gate_selection_is_harness() -> None:
    harnesses = bench_cli._selected_harnesses("stable-core-release-gate", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "stable-core-release-gate",
            "scripts/run_stable_core_release_gate.py",
            frozenset({"stable-core-release-gate"}),
        )
    ]


def test_capability_manifest_check_selection_is_harness() -> None:
    harnesses = bench_cli._selected_harnesses("capability-manifest-check", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "capability-manifest-check",
            "scripts/run_capability_manifest_gate.py",
            frozenset({"capability-manifest-check"}),
        )
    ]


def test_s6_split_audit_selection_is_boundary_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s6", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s6-split-audit",
            "scripts/audit_quantum_kuramoto_split.py",
            frozenset({"s6"}),
        )
    ]


def test_s6_boundary_review_selection_is_review_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s6-review", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s6-boundary-review",
            "scripts/export_quantum_kuramoto_boundary_review.py",
            frozenset({"s6-review"}),
        )
    ]


def test_s6_api_contract_selection_is_contract_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s6-contract", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s6-api-contract",
            "scripts/export_quantum_kuramoto_api_contract.py",
            frozenset({"s6-contract"}),
        )
    ]


def test_dry_run_prints_selected_harnesses(capsys: pytest.CaptureFixture[str]) -> None:
    rc = bench_cli.run(["fim-all", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "selected harnesses" in captured.out
    assert "fim-spectrum" in captured.out


def test_stable_core_capability_matrix_dry_run_selects_matrix_harness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bench_cli.run(["stable-core-capability-matrix", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "selected harnesses" in captured.out
    assert "stable-core-capability-matrix" in captured.out
    assert "scripts/export_stable_core_capability_matrix.py" in captured.out


def test_s2_tn_mps_design_dry_run_selects_design_harness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bench_cli.run(["s2-tn-mps-baseline-design", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "s2-tn-mps-baseline-design" in captured.out
    assert "scripts/export_tn_mps_baseline_design.py" in captured.out


def test_s2_tn_crossover_stage1_dry_run_selects_stage1_harness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bench_cli.run(["s2-tn-crossover-stage1", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "s2-tn-crossover-stage1" in captured.out
    assert "scripts/export_tn_mps_crossover_stage1.py" in captured.out


def test_ground_state_optimizer_convergence_dry_run_selects_harness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bench_cli.run(["ground-state-optimizer-convergence", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "ground-state-optimizer-convergence" in captured.out
    assert "scripts/export_ground_state_optimizer_convergence.py" in captured.out


def test_realtime_control_e2e_dry_run_selects_e2e_harness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bench_cli.run(["realtime-control-e2e", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "selected harnesses" in captured.out
    assert "realtime-control-e2e" in captured.out
    assert "scripts/benchmark_realtime_control_e2e.py" in captured.out


def test_stable_core_capability_gate_dry_run_selects_gate_harness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bench_cli.run(["stable-core-capability-gate", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "selected harnesses" in captured.out
    assert "stable-core-capability-gate" in captured.out
    assert "scripts/run_stable_core_capability_gate.py" in captured.out


def test_stable_core_contract_gate_dry_run_selects_gate_harness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bench_cli.run(["stable-core-contract-gate", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "selected harnesses" in captured.out
    assert "stable-core-contract-gate" in captured.out
    assert "scripts/run_stable_core_contract_gate.py" in captured.out


def test_stable_core_preflight_gate_dry_run_selects_gate_harness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bench_cli.run(["stable-core-preflight-gate", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "selected harnesses" in captured.out
    assert "stable-core-preflight-gate" in captured.out
    assert "scripts/run_stable_core_preflight_gate.py" in captured.out


def test_stable_core_release_gate_dry_run_selects_gate_harness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bench_cli.run(["stable-core-release-gate", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "selected harnesses" in captured.out
    assert "stable-core-release-gate" in captured.out
    assert "scripts/run_stable_core_release_gate.py" in captured.out


def test_capability_manifest_check_dry_run_selects_gate_harness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bench_cli.run(["capability-manifest-check", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "selected harnesses" in captured.out
    assert "capability-manifest-check" in captured.out
    assert "scripts/run_capability_manifest_gate.py" in captured.out


def test_diff_summary_returns_two_when_artifacts_changed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "data" / "rust_vqe_methods" / "example.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"value": 1}\n', encoding="utf-8")
    _run_git(tmp_path, "init", "-q")
    _run_git(tmp_path, "config", "user.email", "test@example.test")
    _run_git(tmp_path, "config", "user.name", "Benchmark Test")
    _run_git(tmp_path, "add", "data/rust_vqe_methods/example.json")
    _run_git(tmp_path, "commit", "-q", "-m", "baseline")
    artifact.write_text('{"value": 2}\n', encoding="utf-8")

    monkeypatch.setattr(bench_cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bench_cli, "ARTEFACT_PATHS", ("data/rust_vqe_methods/example.json",))

    rc = bench_cli._print_diff_summary()

    captured = capsys.readouterr()
    assert rc == 2
    assert "regenerated artefacts differ" in captured.out
    assert "data/rust_vqe_methods/example.json" in captured.out


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True)


def test_diff_summary_propagates_missing_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.setattr(bench_cli, "REPO_ROOT", tmp_path)

    assert bench_cli._print_diff_summary() == 127


def test_run_harness_dispatches_repo_script_with_current_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "scripts" / "benchmark_rust_core_methods.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                "Path('data/rust_vqe_methods').mkdir(parents=True, exist_ok=True)",
                "Path('data/rust_vqe_methods/ran.txt').write_text('ok', encoding='utf-8')",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bench_cli, "REPO_ROOT", tmp_path)

    rc = bench_cli._run_harness(
        bench_cli.Harness(
            "unit",
            "scripts/benchmark_rust_core_methods.py",
            frozenset({"unit"}),
        )
    )

    assert rc == 0
    assert (tmp_path / "data" / "rust_vqe_methods" / "ran.txt").read_text(encoding="utf-8") == "ok"


def test_run_harness_preserves_symlinked_python_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Virtualenv Python symlinks must not be canonicalized to the base interpreter."""
    script = tmp_path / "scripts" / "benchmark_rust_core_methods.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    real_python = tmp_path / "base-python"
    venv_python = tmp_path / "venv-python"
    seen_path = tmp_path / "seen.txt"
    real_python.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                f"printf '%s\\n' \"$0\" > {seen_path}",
                "exit 0",
            )
        ),
        encoding="utf-8",
    )
    real_python.chmod(0o755)
    venv_python.symlink_to(real_python)
    monkeypatch.setattr(bench_cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bench_cli, "PYTHON", str(venv_python))

    rc = bench_cli._run_harness(
        bench_cli.Harness(
            "unit",
            "scripts/benchmark_rust_core_methods.py",
            frozenset({"unit"}),
        )
    )

    assert rc == 0
    assert seen_path.read_text(encoding="utf-8").strip() == str(venv_python)


def test_run_harness_rejects_non_executable_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "scripts" / "benchmark_rust_core_methods.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o644)
    monkeypatch.setattr(bench_cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bench_cli, "PYTHON", str(python))

    rc = bench_cli._run_harness(
        bench_cli.Harness(
            "unit",
            "scripts/benchmark_rust_core_methods.py",
            frozenset({"unit"}),
        )
    )

    assert rc == 127


def test_run_returns_two_when_dispatch_selects_no_harnesses(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(bench_cli, "_selected_harnesses", lambda *args, **kwargs: [])

    rc = bench_cli.run(["reproduce-methods"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "no harnesses selected" in captured.err


def test_run_propagates_harness_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = [bench_cli.Harness("broken", "scripts/missing.py", frozenset({"methods"}))]
    monkeypatch.setattr(bench_cli, "_selected_harnesses", lambda *args, **kwargs: selected)
    monkeypatch.setattr(bench_cli, "_run_harness", lambda harness: 9)
    monkeypatch.setattr(bench_cli, "_print_diff_summary", lambda: 0)

    assert bench_cli.run(["reproduce-methods"]) == 1


def test_run_returns_diff_status_when_harnesses_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = [bench_cli.Harness("ok", "scripts/ok.py", frozenset({"methods"}))]
    monkeypatch.setattr(bench_cli, "_selected_harnesses", lambda *args, **kwargs: selected)
    monkeypatch.setattr(bench_cli, "_run_harness", lambda harness: 0)
    monkeypatch.setattr(bench_cli, "_print_diff_summary", lambda: 2)

    assert bench_cli.run(["reproduce-methods"]) == 2


def test_run_stops_after_first_failure_without_keep_going(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = [
        bench_cli.Harness("broken", "scripts/broken.py", frozenset({"methods"})),
        bench_cli.Harness("skipped", "scripts/skipped.py", frozenset({"methods"})),
    ]
    calls: list[str] = []

    def fake_run_harness(harness: bench_cli.Harness) -> int:
        calls.append(harness.label)
        return 5

    monkeypatch.setattr(bench_cli, "_selected_harnesses", lambda *args, **kwargs: selected)
    monkeypatch.setattr(bench_cli, "_run_harness", fake_run_harness)
    monkeypatch.setattr(bench_cli, "_print_diff_summary", lambda: 0)

    assert bench_cli.run(["reproduce-methods"]) == 1
    assert calls == ["broken"]


def test_run_keep_going_executes_later_harnesses_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [
        bench_cli.Harness("broken", "scripts/broken.py", frozenset({"methods"})),
        bench_cli.Harness("later", "scripts/later.py", frozenset({"methods"})),
    ]
    calls: list[str] = []

    def fake_run_harness(harness: bench_cli.Harness) -> int:
        calls.append(harness.label)
        return 7 if harness.label == "broken" else 0

    monkeypatch.setattr(bench_cli, "_selected_harnesses", lambda *args, **kwargs: selected)
    monkeypatch.setattr(bench_cli, "_run_harness", fake_run_harness)
    monkeypatch.setattr(bench_cli, "_print_diff_summary", lambda: 0)

    assert bench_cli.run(["reproduce-methods", "--keep-going"]) == 1
    assert calls == ["broken", "later"]


def test_run_no_diff_skips_diff_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = [bench_cli.Harness("ok", "scripts/ok.py", frozenset({"methods"}))]
    diff_called = False

    def fail_if_called() -> int:
        nonlocal diff_called
        diff_called = True
        return 2

    monkeypatch.setattr(bench_cli, "_selected_harnesses", lambda *args, **kwargs: selected)
    monkeypatch.setattr(bench_cli, "_run_harness", lambda harness: 0)
    monkeypatch.setattr(bench_cli, "_print_diff_summary", fail_if_called)

    assert bench_cli.run(["reproduce-methods", "--no-diff"]) == 0
    assert diff_called is False


def test_sync_benchmark_registry_harness_is_registered() -> None:
    """The synchronisation benchmark registry is exposed through scpn-bench."""

    from scpn_quantum_control.bench_cli import _selected_harnesses

    harnesses = _selected_harnesses("sync-registry", include_gpu=False)

    assert [harness.label for harness in harnesses] == ["sync-benchmark-registry"]
    assert harnesses[0].script == "scripts/export_synchronisation_benchmark_registry.py"


def test_sync_benchmark_run_harness_is_registered() -> None:
    """The first synchronisation benchmark runner is exposed through scpn-bench."""

    from scpn_quantum_control.bench_cli import _selected_harnesses

    harnesses = _selected_harnesses("sync-run", include_gpu=False)

    assert [harness.label for harness in harnesses] == ["sync-benchmark-run"]
    assert harnesses[0].script == "scripts/run_synchronisation_benchmark.py"


def test_sync_benchmark_compare_harness_is_registered() -> None:
    """The synchronisation benchmark comparator is exposed through scpn-bench."""

    from scpn_quantum_control.bench_cli import _selected_harnesses

    harnesses = _selected_harnesses("sync-compare", include_gpu=False)

    assert [harness.label for harness in harnesses] == ["sync-benchmark-compare"]
    assert harnesses[0].script == "scripts/compare_synchronisation_benchmark.py"


def test_sync_benchmark_gate_harness_is_registered() -> None:
    """The full synchronisation benchmark gate is exposed through scpn-bench."""

    from scpn_quantum_control.bench_cli import _selected_harnesses

    harnesses = _selected_harnesses("sync-gate", include_gpu=False)

    assert [harness.label for harness in harnesses] == ["sync-benchmark-gate"]
    assert harnesses[0].script == "scripts/run_synchronisation_benchmark_gate.py"


def test_symmetry_sector_mitigation_gate_harness_is_registered() -> None:
    """The symmetry-sector fixture gate is exposed through scpn-bench."""

    from scpn_quantum_control.bench_cli import _selected_harnesses

    harnesses = _selected_harnesses("symmetry-sector-gate", include_gpu=False)

    assert [harness.label for harness in harnesses] == ["symmetry-sector-mitigation-gate"]
    assert harnesses[0].script == "scripts/run_symmetry_sector_mitigation_gate.py"
