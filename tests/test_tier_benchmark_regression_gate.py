# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the tier benchmark regression alarm
"""Tests for the fail-closed tier benchmark regression alarm."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE_PATH = _REPO_ROOT / "tools" / "tier_benchmark_regression_gate.py"


def _load_gate() -> ModuleType:
    """Load the gate module from its file path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("tier_benchmark_regression_gate", _GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate_mod = _load_gate()


def _stats(p50_us: float) -> dict[str, float | int]:
    return {
        "p50_us": p50_us,
        "p95_us": p50_us * 1.2,
        "p99_us": p50_us * 1.4,
        "mean_us": p50_us * 1.1,
        "min_us": p50_us * 0.9,
        "max_us": p50_us * 1.5,
        "throughput_ops_s": 1e6 / p50_us if p50_us else 0.0,
        "samples": 5,
    }


def _measured(backend: str, p50_us: float) -> dict[str, Any]:
    return {"backend": backend, "status": "measured", "stats": _stats(p50_us), "reason": None}


def _unavailable(backend: str, reason: str) -> dict[str, Any]:
    return {"backend": backend, "status": "unavailable", "stats": None, "reason": reason}


def _result(operation: str, size: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "operation": operation,
        "size": size,
        "fastest_backend": rows[0]["backend"],
        "parity_max_abs_diff": 0.0,
        "rows": rows,
        "extra": {},
    }


def _artifact(
    results: list[dict[str, Any]],
    *,
    environment: str = "ci",
    cpu: str = "TestCPU",
    commit: str = "abc123",
) -> dict[str, Any]:
    payload = {
        "schema_version": gate_mod.ARTIFACT_SCHEMA,
        "environment": environment,
        "generated_utc": "2026-07-07T00:00:00+00:00",
        "production_claim_allowed": False,
        "provenance": {"cpu_model": cpu, "commit": commit},
        "parameters": {"repeats": 5},
        "results": results,
    }
    payload["payload_sha256"] = gate_mod._payload_digest(payload)
    return payload


_THRESHOLDS = {"default": {"rust": 2.5, "julia": 3.0, "python": 2.5}}


def _single_cell(p50_us: float) -> list[dict[str, Any]]:
    return [_result("order_parameter", 8, [_measured("rust", p50_us)])]


def test_digest_is_deterministic_and_excludes_existing_field() -> None:
    payload = {"a": 1, "payload_sha256": "stale"}
    assert gate_mod._payload_digest(payload) == gate_mod._payload_digest({"a": 1})


def test_parse_thresholds_accepts_a_valid_policy() -> None:
    policy = gate_mod.parse_thresholds(
        {"default": {"rust": 2.5}, "order_parameter": {"julia": 4.0}}
    )
    assert policy["default"]["rust"] == 2.5
    assert policy["order_parameter"]["julia"] == 4.0


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({}, "non-empty .default. table"),
        ({"default": {}}, "non-empty .default. table"),
        ({"default": {"rust": 2.0}, "op": 5}, "must be a table"),
        ({"default": {"rust": True}}, "must be a number"),
        ({"default": {"rust": "x"}}, "must be a number"),
        ({"default": {"rust": 0.0}}, "positive and finite"),
        ({"default": {"rust": -1.0}}, "positive and finite"),
        ({"default": {"rust": float("inf")}}, "positive and finite"),
        ({"default": {"rust": float("nan")}}, "positive and finite"),
    ],
)
def test_parse_thresholds_rejects_invalid_policies(raw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        gate_mod.parse_thresholds(raw)


def test_resolve_threshold_prefers_operation_override() -> None:
    policy = {"default": {"rust": 2.5}, "order_parameter": {"rust": 1.5}}
    assert gate_mod.resolve_threshold(policy, "order_parameter", "rust") == 1.5
    assert gate_mod.resolve_threshold(policy, "mean_phase", "rust") == 2.5
    assert gate_mod.resolve_threshold(policy, "mean_phase", "fortran") is None


def test_validate_artifact_accepts_a_stamped_artifact() -> None:
    assert gate_mod.validate_artifact(_artifact(_single_cell(10.0)), "report") == []


def test_validate_artifact_rejects_schema_results_and_digest_defects() -> None:
    wrong_schema = _artifact(_single_cell(10.0))
    wrong_schema["schema_version"] = "other.v1"
    assert any("schema_version" in e for e in gate_mod.validate_artifact(wrong_schema, "report"))

    no_results = _artifact(_single_cell(10.0))
    no_results["results"] = []
    assert any("results" in e for e in gate_mod.validate_artifact(no_results, "report"))

    unstamped = _artifact(_single_cell(10.0))
    del unstamped["payload_sha256"]
    assert any("payload_sha256" in e for e in gate_mod.validate_artifact(unstamped, "report"))

    tampered = _artifact(_single_cell(10.0))
    tampered["results"][0]["rows"][0]["stats"]["p50_us"] = 1.0
    assert any("tampered" in e for e in gate_mod.validate_artifact(tampered, "report"))


def test_coerce_float_rejects_bools_strings_and_nan() -> None:
    assert gate_mod._coerce_float(1.5) == 1.5
    assert gate_mod._coerce_float(3) == 3.0
    assert gate_mod._coerce_float(True) is None
    assert gate_mod._coerce_float("fast") is None
    assert gate_mod._coerce_float(float("nan")) is None


def test_environment_mismatch_flags_ci_vs_local() -> None:
    report = _artifact(_single_cell(10.0), environment="local")
    baseline = _artifact(_single_cell(10.0))
    finding = gate_mod.environment_mismatch(report, baseline)
    assert finding is not None and finding.kind == "environment_mismatch"
    assert gate_mod.environment_mismatch(baseline, baseline) is None


def test_hardware_mismatch_flags_cpu_drift_and_tolerates_absence() -> None:
    report = _artifact(_single_cell(10.0), cpu="AMD EPYC 9B14")
    baseline = _artifact(_single_cell(10.0), cpu="AMD EPYC 7763 64-Core Processor")
    finding = gate_mod.hardware_mismatch(report, baseline)
    assert finding is not None and finding.kind == "hardware_mismatch"
    assert gate_mod.hardware_mismatch(baseline, baseline) is None
    anonymous = _artifact(_single_cell(10.0), cpu="")
    assert gate_mod.hardware_mismatch(anonymous, baseline) is None


def test_compare_passes_within_threshold_and_counts_cells() -> None:
    baseline = _artifact(_single_cell(10.0))
    report = _artifact(_single_cell(20.0))
    findings, summary = gate_mod.compare(report, baseline, _THRESHOLDS)
    assert findings == []
    assert summary == {"compared_cells": 1, "skipped_cells": 0}


def test_compare_flags_a_regression_beyond_threshold() -> None:
    baseline = _artifact(_single_cell(10.0))
    report = _artifact(_single_cell(30.0))
    findings, _ = gate_mod.compare(report, baseline, _THRESHOLDS)
    assert [f.kind for f in findings] == ["regression"]
    assert findings[0].ratio == pytest.approx(3.0)
    assert findings[0].threshold == 2.5
    assert findings[0].operation == "order_parameter"
    assert findings[0].size == 8
    assert findings[0].backend == "rust"


def test_compare_treats_zero_baseline_p50_as_infinite_ratio() -> None:
    baseline = _artifact(_single_cell(0.0))
    report = _artifact(_single_cell(10.0))
    findings, _ = gate_mod.compare(report, baseline, _THRESHOLDS)
    assert [f.kind for f in findings] == ["regression"]
    assert findings[0].ratio == float("inf")


def test_compare_skips_deliberate_tier_exclusion_without_alarming() -> None:
    baseline = _artifact(
        [_result("order_parameter", 8, [_measured("rust", 10.0), _measured("julia", 20.0)])]
    )
    report = _artifact(
        [
            _result(
                "order_parameter",
                8,
                [_measured("rust", 10.0), _unavailable("julia", "tier excluded by --tiers")],
            )
        ]
    )
    findings, summary = gate_mod.compare(report, baseline, _THRESHOLDS)
    assert findings == []
    assert summary == {"compared_cells": 1, "skipped_cells": 1}


def test_compare_alarms_when_a_tier_silently_vanishes() -> None:
    baseline = _artifact(
        [_result("order_parameter", 8, [_measured("rust", 10.0), _measured("julia", 20.0)])]
    )
    report = _artifact(
        [
            _result(
                "order_parameter",
                8,
                [_measured("rust", 10.0), _unavailable("julia", "juliacall import failed")],
            )
        ]
    )
    findings, _ = gate_mod.compare(report, baseline, _THRESHOLDS)
    assert [f.kind for f in findings] == ["missing_backend"]
    assert "juliacall import failed" in findings[0].detail


def test_compare_alarms_when_the_whole_result_row_is_gone() -> None:
    baseline = _artifact(
        [
            _result("order_parameter", 8, [_measured("rust", 10.0)]),
            _result("mean_phase", 8, [_measured("rust", 10.0)]),
        ]
    )
    report = _artifact([_result("order_parameter", 8, [_measured("rust", 10.0)])])
    findings, _ = gate_mod.compare(report, baseline, _THRESHOLDS)
    assert [f.kind for f in findings] == ["missing_backend"]
    assert findings[0].operation == "mean_phase"
    assert "no row" in findings[0].detail


def test_compare_flags_policy_gap_for_an_unknown_backend() -> None:
    baseline = _artifact([_result("order_parameter", 8, [_measured("fortran", 10.0)])])
    report = _artifact([_result("order_parameter", 8, [_measured("fortran", 10.0)])])
    findings, _ = gate_mod.compare(report, baseline, _THRESHOLDS)
    kinds = [f.kind for f in findings]
    assert "policy_gap" in kinds
    assert "no_overlap" in kinds


def test_compare_flags_measured_row_with_non_finite_p50() -> None:
    baseline = _artifact(_single_cell(10.0))
    report = _artifact(_single_cell(10.0))
    report["results"][0]["rows"][0]["stats"]["p50_us"] = float("nan")
    report["payload_sha256"] = gate_mod._payload_digest(report)
    findings, _ = gate_mod.compare(report, baseline, _THRESHOLDS)
    kinds = [f.kind for f in findings]
    assert "missing_metric" in kinds
    assert "no_overlap" in kinds


def test_compare_handles_measured_rows_with_malformed_stats() -> None:
    baseline = _artifact(_single_cell(10.0))
    report = _artifact(_single_cell(10.0))
    report["results"][0]["rows"][0]["stats"] = None
    report["payload_sha256"] = gate_mod._payload_digest(report)
    findings, _ = gate_mod.compare(report, baseline, _THRESHOLDS)
    assert [f.kind for f in findings] == ["missing_metric", "no_overlap"]

    malformed_baseline = _artifact(_single_cell(10.0))
    malformed_baseline["results"][0]["rows"][0]["stats"] = None
    findings, summary = gate_mod.compare(report, malformed_baseline, _THRESHOLDS)
    assert [f.kind for f in findings] == ["no_overlap"]
    assert summary["compared_cells"] == 0


def test_compare_ignores_unmeasured_baseline_rows() -> None:
    baseline = _artifact(
        [
            _result(
                "order_parameter",
                8,
                [_measured("rust", 10.0), _unavailable("julia", "tier excluded by --tiers")],
            )
        ]
    )
    report = _artifact(_single_cell(10.0))
    findings, summary = gate_mod.compare(report, baseline, _THRESHOLDS)
    assert findings == []
    assert summary["compared_cells"] == 1


def test_gate_passes_and_stamps_a_verdict() -> None:
    baseline = _artifact(_single_cell(10.0))
    report = _artifact(_single_cell(12.0), commit="def456")
    verdict = gate_mod.gate(report, baseline, _THRESHOLDS, generated_utc="2026-07-07T00:00:00Z")
    assert verdict["passed"] is True
    assert verdict["schema_version"] == gate_mod.VERDICT_SCHEMA
    assert verdict["baseline_commit"] == "abc123"
    assert verdict["report_commit"] == "def456"
    assert verdict["summary"] == {"compared_cells": 1, "skipped_cells": 0}
    assert verdict["payload_sha256"] == gate_mod._payload_digest(verdict)


def test_gate_stops_at_validation_findings_without_comparing() -> None:
    baseline = _artifact(_single_cell(10.0))
    tampered = _artifact(_single_cell(10.0))
    tampered["results"][0]["rows"][0]["stats"]["p50_us"] = 1.0
    verdict = gate_mod.gate(tampered, baseline, _THRESHOLDS, generated_utc="")
    assert verdict["passed"] is False
    assert [f["kind"] for f in verdict["findings"]] == ["report_invalid"]
    assert verdict["summary"] == {"compared_cells": 0, "skipped_cells": 0}


def test_gate_reports_baseline_invalid_separately() -> None:
    baseline = _artifact(_single_cell(10.0))
    del baseline["payload_sha256"]
    report = _artifact(_single_cell(10.0))
    verdict = gate_mod.gate(report, baseline, _THRESHOLDS, generated_utc="")
    assert [f["kind"] for f in verdict["findings"]] == ["baseline_invalid"]


def test_gate_combines_mismatch_and_cell_findings() -> None:
    baseline = _artifact(_single_cell(10.0), environment="ci", cpu="CPU-A")
    report = _artifact(_single_cell(30.0), environment="local", cpu="CPU-B")
    verdict = gate_mod.gate(report, baseline, _THRESHOLDS, generated_utc="")
    kinds = [f["kind"] for f in verdict["findings"]]
    assert kinds == ["environment_mismatch", "hardware_mismatch", "regression"]


def test_annotation_lines_cap_and_level() -> None:
    findings = [
        {
            "kind": "regression",
            "operation": f"op{i}",
            "size": 8,
            "backend": "rust",
            "detail": "d",
        }
        for i in range(gate_mod.MAX_ANNOTATIONS + 5)
    ]
    warn_lines = gate_mod._annotation_lines(findings, evidence_only=True)
    assert len(warn_lines) == gate_mod.MAX_ANNOTATIONS + 1
    assert warn_lines[0].startswith("::warning ")
    assert "5 further findings truncated" in warn_lines[-1]
    error_lines = gate_mod._annotation_lines(findings[:1], evidence_only=False)
    assert error_lines == [
        "::error title=Tier benchmark regression gate::[regression] op0/8/rust: d"
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_thresholds(path: Path) -> Path:
    path.write_text(
        "[default]\nrust = 2.5\njulia = 3.0\npython = 2.5\n",
        encoding="utf-8",
    )
    return path


def test_main_passes_writes_verdict_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _write_json(tmp_path / "report.json", _artifact(_single_cell(12.0)))
    baseline = _write_json(tmp_path / "baseline.json", _artifact(_single_cell(10.0)))
    thresholds = _write_thresholds(tmp_path / "thresholds.toml")
    verdict_path = tmp_path / "out" / "verdict.json"
    code = gate_mod.main(
        [
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--thresholds",
            str(thresholds),
            "--json-out",
            str(verdict_path),
            "--generated-utc",
            "2026-07-07T00:00:00Z",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "tier benchmark gate PASSED" in captured.out
    assert "compared 1 cells" in captured.out
    stored = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert stored["passed"] is True
    assert stored["generated_utc"] == "2026-07-07T00:00:00Z"


def test_main_fails_on_regression_without_evidence_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _write_json(tmp_path / "report.json", _artifact(_single_cell(30.0)))
    baseline = _write_json(tmp_path / "baseline.json", _artifact(_single_cell(10.0)))
    thresholds = _write_thresholds(tmp_path / "thresholds.toml")
    code = gate_mod.main(
        ["--report", str(report), "--baseline", str(baseline), "--thresholds", str(thresholds)]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "tier benchmark gate FAILED" in captured.err
    assert "[regression]" in captured.err
    assert "::error " in captured.out


def test_main_evidence_only_reports_but_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _write_json(tmp_path / "report.json", _artifact(_single_cell(30.0)))
    baseline = _write_json(tmp_path / "baseline.json", _artifact(_single_cell(10.0)))
    thresholds = _write_thresholds(tmp_path / "thresholds.toml")
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
    captured = capsys.readouterr()
    assert "evidence-only: not failing" in captured.err
    assert "::warning " in captured.out


def test_main_fails_closed_when_report_or_baseline_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_json(tmp_path / "baseline.json", _artifact(_single_cell(10.0)))
    code = gate_mod.main(["--report", str(tmp_path / "absent.json"), "--baseline", str(baseline)])
    assert code == 1
    assert "report not found" in capsys.readouterr().err

    report = _write_json(tmp_path / "report.json", _artifact(_single_cell(10.0)))
    code = gate_mod.main(["--report", str(report), "--baseline", str(tmp_path / "absent.json")])
    assert code == 1
    assert "baseline not found" in capsys.readouterr().err


def test_committed_baseline_and_thresholds_satisfy_the_gate_contract() -> None:
    """The committed CI artefact and threshold policy must stay gate-ready."""
    baseline = json.loads(gate_mod.DEFAULT_BASELINE.read_text(encoding="utf-8"))
    assert gate_mod.validate_artifact(baseline, "baseline") == []
    thresholds = gate_mod.load_thresholds_file(gate_mod.DEFAULT_THRESHOLDS)
    backends = {
        str(row["backend"])
        for result in baseline["results"]
        for row in result["rows"]
        if row["status"] == "measured"
    }
    for result in baseline["results"]:
        for backend in backends:
            assert (
                gate_mod.resolve_threshold(thresholds, str(result["operation"]), backend)
                is not None
            ), f"no threshold policy for {result['operation']}.{backend}"


def test_committed_baseline_self_comparison_passes() -> None:
    """The committed CI artefact compared against itself must pass cleanly."""
    baseline = json.loads(gate_mod.DEFAULT_BASELINE.read_text(encoding="utf-8"))
    thresholds = gate_mod.load_thresholds_file(gate_mod.DEFAULT_THRESHOLDS)
    verdict = gate_mod.gate(baseline, baseline, thresholds, generated_utc="")
    assert verdict["passed"] is True
    assert verdict["summary"]["compared_cells"] > 0
