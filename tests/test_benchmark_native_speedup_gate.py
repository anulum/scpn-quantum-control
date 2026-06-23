# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the native speedup benchmark regression gate
"""Tests for the fail-closed native dense-Hamiltonian benchmark regression gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE_PATH = _REPO_ROOT / "tools" / "benchmark_native_speedup_gate.py"


def _load_gate() -> ModuleType:
    """Load the gate module from its file path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("native_speedup_gate", _GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate_mod = _load_gate()

_METRICS = {"p50_us": 10.0, "p95_us": 12.0, "p99_us": 14.0, "throughput_ops_s": 100_000.0}


def _benchmarks() -> dict[str, Any]:
    return {
        "dense_xy_hamiltonian_L4": {
            "languages": {
                "rust_pyo3": dict(_METRICS),
                "qiskit_sparsepauliop": {k: v * 50 for k, v in _METRICS.items()},
            },
        }
    }


def _make_report(benchmarks: dict[str, Any], *, cpu: str = "TestCPU") -> dict[str, Any]:
    report = {
        "schema_version": "scpn-quantum-control.native-speedup.v1",
        "benchmarks": benchmarks,
        "provenance": {"cpu_model": cpu, "commit": "abc123"},
    }
    report["payload_sha256"] = gate_mod._payload_digest(report)
    return report


def _make_baseline(benchmarks: dict[str, Any], *, cpu: str = "TestCPU") -> dict[str, Any]:
    baseline = {
        "schema_version": "scpn-quantum-control.native-speedup-baseline.v1",
        "baseline_commit": "abc123",
        "benchmarks": benchmarks,
        "provenance": {"cpu_model": cpu},
    }
    baseline["baseline_sha256"] = gate_mod.canonical_metrics_digest(benchmarks)
    return baseline


def test_metric_direction_splits_latency_and_throughput() -> None:
    assert gate_mod.metric_direction("p50_us") == "upper"
    assert gate_mod.metric_direction("p99_us") == "upper"
    assert gate_mod.metric_direction("throughput_ops_s") == "lower"


def test_digests_are_deterministic_and_exclude_existing_field() -> None:
    payload = {"a": 1, "b": 2, "payload_sha256": "stale"}
    assert gate_mod._payload_digest(payload) == gate_mod._payload_digest({"a": 1, "b": 2})
    benches = _benchmarks()
    assert gate_mod.canonical_metrics_digest(benches) == gate_mod.canonical_metrics_digest(benches)


def test_parse_thresholds_accepts_a_valid_policy() -> None:
    policy = gate_mod.parse_thresholds({"default": {"p50_us": 2.0}, "bench": {"p95_us": 1.5}})
    assert policy["default"]["p50_us"] == 2.0
    assert policy["bench"]["p95_us"] == 1.5


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({}, "non-empty .default. table"),
        ({"default": {}}, "non-empty .default. table"),
        ({"default": {"p50_us": 2.0}, "bench": 5}, "must be a table"),
        ({"default": {"p50_us": True}}, "must be a number"),
        ({"default": {"p50_us": "x"}}, "must be a number"),
        ({"default": {"p50_us": 0.0}}, "positive and finite"),
        ({"default": {"p50_us": float("inf")}}, "positive and finite"),
    ],
)
def test_parse_thresholds_rejects_bad_policy(raw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        gate_mod.parse_thresholds(raw)


def test_resolve_threshold_prefers_per_benchmark_then_default() -> None:
    thresholds = {"default": {"p50_us": 2.0}, "bench": {"p50_us": 1.3}}
    assert gate_mod.resolve_threshold(thresholds, "bench", "p50_us") == 1.3
    assert gate_mod.resolve_threshold(thresholds, "other", "p50_us") == 2.0
    assert gate_mod.resolve_threshold(thresholds, "other", "absent") is None


def test_validate_report_accepts_a_well_formed_report() -> None:
    assert gate_mod.validate_report(_make_report(_benchmarks())) == []


def test_validate_report_flags_missing_fields_and_tamper() -> None:
    assert any(
        "schema_version" in e
        for e in gate_mod.validate_report({"benchmarks": {}, "payload_sha256": "x"})
    )
    assert any("benchmarks" in e for e in gate_mod.validate_report({"schema_version": "v"}))
    report = _make_report(_benchmarks())
    report["benchmarks"]["tampered"] = {"languages": {}}
    assert any("tampered" in e for e in gate_mod.validate_report(report))
    no_digest = {"schema_version": "v", "benchmarks": {}}
    assert any("payload_sha256" in e for e in gate_mod.validate_report(no_digest))


def test_verify_baseline_integrity_accepts_and_rejects() -> None:
    assert gate_mod.verify_baseline_integrity(_make_baseline(_benchmarks())) == []
    assert any("missing benchmarks" in e for e in gate_mod.verify_baseline_integrity({}))
    assert any(
        "missing baseline_sha256" in e
        for e in gate_mod.verify_baseline_integrity({"benchmarks": {}})
    )
    bad = _make_baseline(_benchmarks())
    bad["benchmarks"]["x"] = {"languages": {}}
    assert any("tampered" in e for e in gate_mod.verify_baseline_integrity(bad))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.5, 1.5), (3, 3.0), (True, None), ("x", None), (float("nan"), None)],
)
def test_coerce_float(value: Any, expected: float | None) -> None:
    assert gate_mod._coerce_float(value) == expected or (
        expected is None and gate_mod._coerce_float(value) is None
    )


def test_hardware_mismatch_only_when_both_cpus_present_and_differ() -> None:
    report = _make_report(_benchmarks(), cpu="CpuA")
    assert gate_mod.hardware_mismatch(report, _make_baseline(_benchmarks(), cpu="CpuA")) is None
    finding = gate_mod.hardware_mismatch(report, _make_baseline(_benchmarks(), cpu="CpuB"))
    assert finding is not None and finding.kind == "hardware_mismatch"
    no_cpu: dict[str, Any] = {"provenance": {}}
    assert gate_mod.hardware_mismatch(no_cpu, _make_baseline(_benchmarks())) is None


def test_compare_passes_when_within_policy() -> None:
    thresholds = {
        "default": {"p50_us": 2.0, "p95_us": 2.0, "p99_us": 2.0, "throughput_ops_s": 0.5}
    }
    assert (
        gate_mod.compare(_make_report(_benchmarks()), _make_baseline(_benchmarks()), thresholds)
        == []
    )


def test_compare_flags_latency_regression() -> None:
    thresholds = {
        "default": {"p50_us": 1.2, "p95_us": 2.0, "p99_us": 2.0, "throughput_ops_s": 0.5}
    }
    report_benches = _benchmarks()
    report_benches["dense_xy_hamiltonian_L4"]["languages"]["rust_pyo3"]["p50_us"] = 50.0
    findings = gate_mod.compare(
        _make_report(report_benches), _make_baseline(_benchmarks()), thresholds
    )
    assert any(f.kind == "regression" and f.metric == "p50_us" for f in findings)


def test_compare_flags_throughput_regression() -> None:
    thresholds = {
        "default": {"p50_us": 5.0, "p95_us": 5.0, "p99_us": 5.0, "throughput_ops_s": 0.9}
    }
    report_benches = _benchmarks()
    report_benches["dense_xy_hamiltonian_L4"]["languages"]["rust_pyo3"]["throughput_ops_s"] = 10.0
    findings = gate_mod.compare(
        _make_report(report_benches), _make_baseline(_benchmarks()), thresholds
    )
    assert any(f.kind == "regression" and f.metric == "throughput_ops_s" for f in findings)


def test_compare_flags_policy_gap_and_missing_metric() -> None:
    gap_thresholds = {"default": {"p95_us": 2.0, "p99_us": 2.0, "throughput_ops_s": 0.5}}
    gap = gate_mod.compare(
        _make_report(_benchmarks()), _make_baseline(_benchmarks()), gap_thresholds
    )
    assert any(f.kind == "policy_gap" and f.metric == "p50_us" for f in gap)

    thresholds = {
        "default": {"p50_us": 2.0, "p95_us": 2.0, "p99_us": 2.0, "throughput_ops_s": 0.5}
    }
    report_benches = _benchmarks()
    del report_benches["dense_xy_hamiltonian_L4"]["languages"]["rust_pyo3"]["p50_us"]
    missing = gate_mod.compare(
        _make_report(report_benches), _make_baseline(_benchmarks()), thresholds
    )
    assert any(f.kind == "missing_metric" and f.metric == "p50_us" for f in missing)


def test_gate_passes_clean_and_short_circuits_on_structural_failure() -> None:
    thresholds = {
        "default": {"p50_us": 2.0, "p95_us": 2.0, "p99_us": 2.0, "throughput_ops_s": 0.5}
    }
    verdict = gate_mod.gate(
        _make_report(_benchmarks()), _make_baseline(_benchmarks()), thresholds, generated_utc="t"
    )
    assert verdict["passed"] is True
    assert verdict["findings"] == []

    invalid_report = {"benchmarks": {}, "payload_sha256": "wrong"}
    bad_verdict = gate_mod.gate(
        invalid_report, _make_baseline(_benchmarks()), thresholds, generated_utc="t"
    )
    assert bad_verdict["passed"] is False
    assert all(
        f["kind"] in {"report_invalid", "baseline_invalid"} for f in bad_verdict["findings"]
    )


def _write_thresholds(path: Path) -> Path:
    path.write_text(
        "[default]\np50_us = 2.0\np95_us = 2.0\np99_us = 2.0\nthroughput_ops_s = 0.5\n",
        encoding="utf-8",
    )
    return path


def test_load_thresholds_file_reads_toml(tmp_path: Path) -> None:
    policy = gate_mod.load_thresholds_file(_write_thresholds(tmp_path / "t.toml"))
    assert policy["default"]["p50_us"] == 2.0


def _write_run(tmp_path: Path, *, report_cpu: str = "TestCPU") -> tuple[Path, Path, Path]:
    report_path = tmp_path / "report.json"
    baseline_path = tmp_path / "baseline.json"
    thresholds_path = _write_thresholds(tmp_path / "t.toml")
    report_path.write_text(
        json.dumps(_make_report(_benchmarks(), cpu=report_cpu)), encoding="utf-8"
    )
    baseline_path.write_text(json.dumps(_make_baseline(_benchmarks())), encoding="utf-8")
    return report_path, baseline_path, thresholds_path


def test_main_passes_on_matching_run(tmp_path: Path) -> None:
    report, baseline, thresholds = _write_run(tmp_path)
    out = tmp_path / "verdict.json"
    code = gate_mod.main(
        [
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--thresholds",
            str(thresholds),
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True


def test_main_fails_on_hardware_mismatch(tmp_path: Path) -> None:
    report, baseline, thresholds = _write_run(tmp_path, report_cpu="DifferentCPU")
    code = gate_mod.main(
        ["--report", str(report), "--baseline", str(baseline), "--thresholds", str(thresholds)]
    )
    assert code == 1


def test_main_evidence_only_never_fails(tmp_path: Path) -> None:
    report, baseline, thresholds = _write_run(tmp_path, report_cpu="DifferentCPU")
    code = gate_mod.main(
        [
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--thresholds",
            str(thresholds),
            "--evidence-only",
        ]
    )
    assert code == 0


def test_main_fails_closed_on_missing_files(tmp_path: Path) -> None:
    _, baseline, thresholds = _write_run(tmp_path)
    code = gate_mod.main(
        [
            "--report",
            str(tmp_path / "absent.json"),
            "--baseline",
            str(baseline),
            "--thresholds",
            str(thresholds),
        ]
    )
    assert code == 1
