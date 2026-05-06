# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- scpn-bench CLI tests
"""Tests for the benchmark reproducibility command line interface."""

from __future__ import annotations

import subprocess

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


def test_s2_scaling_lite_selection_includes_protocol_and_lite_rows() -> None:
    harnesses = bench_cli._selected_harnesses("s2", include_gpu=False)

    labels = [harness.label for harness in harnesses]

    assert labels == ["s2-scaling-protocol", "s2-scaling-lite", "s2-claim-boundary"]


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


def test_s6_split_audit_selection_is_boundary_harness() -> None:
    harnesses = bench_cli._selected_harnesses("s6", include_gpu=False)

    assert harnesses == [
        bench_cli.Harness(
            "s6-split-audit",
            "scripts/audit_quantum_kuramoto_split.py",
            frozenset({"s6"}),
        )
    ]


def test_dry_run_prints_selected_harnesses(capsys: pytest.CaptureFixture[str]) -> None:
    rc = bench_cli.run(["fim-all", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "selected harnesses" in captured.out
    assert "fim-spectrum" in captured.out


def test_diff_summary_returns_two_when_artifacts_changed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if "--stat" in command:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="data/rust_vqe_methods/example.json\n",
            stderr="",
        )

    monkeypatch.setattr(bench_cli.subprocess, "run", fake_run)

    rc = bench_cli._print_diff_summary()

    captured = capsys.readouterr()
    assert rc == 2
    assert "regenerated artefacts differ" in captured.out
    assert "data/rust_vqe_methods/example.json" in captured.out


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


def test_run_stops_after_first_failure_without_keep_going(monkeypatch: pytest.MonkeyPatch):
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


def test_run_keep_going_executes_later_harnesses_after_failure(monkeypatch: pytest.MonkeyPatch):
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


def test_run_no_diff_skips_diff_summary(monkeypatch: pytest.MonkeyPatch):
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
