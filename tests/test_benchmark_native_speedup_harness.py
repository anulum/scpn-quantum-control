# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the native speedup benchmark harness
"""Tests for the dense-Hamiltonian benchmark harness and its gate round-trip."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    """Load a script/tool module from its file path (neither is a package)."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load("native_speedup_harness", "scripts/benchmark_native_speedup.py")
gate_mod = _load("native_speedup_gate_rt", "tools/benchmark_native_speedup_gate.py")


def test_stats_reports_ordered_percentiles_and_throughput() -> None:
    stats = harness._stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["p50_us"] <= stats["p95_us"] <= stats["p99_us"]
    assert stats["throughput_ops_s"] > 0.0


def test_measure_discards_warmup_and_collects_repeats() -> None:
    calls: list[int] = []
    harness._measure(lambda: calls.append(1), warmup=3, repeats=5)
    assert len(calls) == 8


@pytest.mark.parametrize(
    ("size", "expected_repeats"),
    [(4, 200), (8, 80), (10, 25), (12, 8)],
)
def test_repeats_scale_with_system_size(size: int, expected_repeats: int) -> None:
    _, repeats = harness._repeats_for(size)
    assert repeats == expected_repeats


def test_digests_are_deterministic_and_exclude_existing_field() -> None:
    payload = {"a": 1, "payload_sha256": "stale"}
    assert harness.payload_digest(payload) == harness.payload_digest({"a": 1})
    benches = {"x": {"languages": {"rust_pyo3": {"p50_us": 1.0}}}}
    assert harness.canonical_metrics_digest(benches) == harness.canonical_metrics_digest(benches)


def test_provenance_helpers_return_sane_values() -> None:
    assert isinstance(harness._cpu_model(), str)
    assert isinstance(harness._rust_release_profile(), dict)
    assert harness._peak_rss_mb() >= 0.0
    assert isinstance(harness._git_commit(), str)


def test_build_report_round_trips_through_the_gate() -> None:
    """A fresh L=4 report and its baseline pass the gate on the same machine."""
    report = harness.build_report((4,), evidence_class="local_regression", generated_utc="t")

    assert report["production_claim_allowed"] is False
    assert report["payload_sha256"] == harness.payload_digest(report)
    bench = report["benchmarks"]["dense_xy_hamiltonian_L4"]
    assert "qiskit_sparsepauliop" in bench["languages"]

    baseline = harness.report_to_baseline(report)
    assert baseline["production_claim_allowed"] is False
    assert baseline["baseline_sha256"] == harness.canonical_metrics_digest(baseline["benchmarks"])

    thresholds = {
        "default": {"p50_us": 5.0, "p95_us": 5.0, "p99_us": 5.0, "throughput_ops_s": 0.2}
    }
    verdict = gate_mod.gate(report, baseline, thresholds, generated_utc="t")
    assert verdict["passed"] is True


def test_build_report_falls_back_when_rust_engine_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the Rust engine unavailable, only the Qiskit backend is measured."""
    monkeypatch.setattr(harness, "optional_rust_engine", lambda: None)
    report = harness.build_report((4,), evidence_class="local_regression", generated_utc="t")

    bench = report["benchmarks"]["dense_xy_hamiltonian_L4"]
    assert bench["rust_available"] is False
    assert bench["cross_language_parity"] is None
    assert "rust_pyo3" not in bench["languages"]
    assert report["provenance"]["rust_backend"] == "absent"


def test_main_writes_report_and_baseline(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    baseline_path = tmp_path / "baseline.json"
    code = harness.main(
        [
            "--sizes",
            "4",
            "--json-out",
            str(report_path),
            "--write-baseline",
            str(baseline_path),
        ]
    )
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "scpn-quantum-control.native-speedup.v1"
    assert baseline["schema_version"] == "scpn-quantum-control.native-speedup-baseline.v1"
    assert gate_mod.validate_report(report) == []
    assert gate_mod.verify_baseline_integrity(baseline) == []
