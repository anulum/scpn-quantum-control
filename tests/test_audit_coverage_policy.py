# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — line gate and branch telemetry policy audit tests
"""Tests for the branch-enabled coverage policy audit."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools" / "audit_coverage_policy.py"
    spec = importlib.util.spec_from_file_location("audit_coverage_policy_for_tests", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load coverage policy audit from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_audit = _load_tool()


def _baseline() -> dict[str, object]:
    return {
        "origin_commit": "a" * 40,
        "remote_ci_run": 123,
        "source": "coverage artifact",
        "covered_lines": 95,
        "measured_lines": 100,
        "line_percent": 95.0,
        "branch_percent": None,
    }


def _branch(
    *,
    mode: str = "observe",
    require_data: object = True,
    minimum_percent: object = None,
) -> dict[str, object]:
    return {
        "mode": mode,
        "require_data": require_data,
        "minimum_percent": minimum_percent,
    }


def _payload(
    *,
    schema_version: object = 1,
    coverage_xml: object = "coverage.xml",
    line_minimum_percent: object = 90.0,
    branch: object | None = None,
    baseline: object | None = None,
    promotion: object | None = None,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "coverage_xml": coverage_xml,
        "line_minimum_percent": line_minimum_percent,
        "branch": _branch() if branch is None else branch,
        "baseline": _baseline() if baseline is None else baseline,
        "branch_promotion": {
            "minimum_successful_remote_runs": 3,
            "rule": "promote after stable remote evidence",
        }
        if promotion is None
        else promotion,
    }


def _write_xml(
    path: Path,
    *,
    root: str = "coverage",
    attributes: dict[str, str] | None = None,
) -> None:
    values = {
        "lines-covered": "95",
        "lines-valid": "100",
        "line-rate": "0.95",
        "branches-covered": "18",
        "branches-valid": "20",
        "branch-rate": "0.9",
    }
    if attributes is not None:
        values = attributes
    rendered = " ".join(f'{key}="{value}"' for key, value in values.items())
    path.write_text(f"<{root} {rendered}></{root}>\n", encoding="utf-8")


def test_parse_policy_exposes_observation_contract() -> None:
    policy = _audit.parse_policy(_payload())

    assert policy.schema_version == 1
    assert policy.coverage_xml == "coverage.xml"
    assert policy.line_minimum_percent == 90.0
    assert policy.branch.mode == "observe"
    assert policy.branch.require_data is True
    assert policy.branch.minimum_percent is None
    assert policy.baseline.line_percent == 95.0
    assert policy.minimum_successful_remote_runs == 3


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "policy must be an object"),
        (_payload(schema_version=True), "schema_version must be an integer"),
        (_payload(coverage_xml=""), "coverage_xml must be a non-empty string"),
        (_payload(line_minimum_percent="90"), "line_minimum_percent must be a number"),
        (_payload(line_minimum_percent=math.inf), "line_minimum_percent must be finite"),
        (_payload(branch=[]), "policy.branch must be an object"),
        (_payload(branch=_branch(require_data=1)), "require_data must be a boolean"),
        (_payload(promotion=[]), "policy.branch_promotion must be an object"),
        (
            _payload(
                promotion={"minimum_successful_remote_runs": 3, "rule": ""},
            ),
            "rule must be a non-empty string",
        ),
    ],
)
def test_parse_policy_rejects_structural_type_errors(payload: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(payload)


@pytest.mark.parametrize(
    ("coverage_xml", "message"),
    [
        ("/tmp/coverage.xml", "must be repository-relative"),
        ("reports/../coverage.xml", "must be repository-relative"),
        ("coverage.json", "must name an XML file"),
    ],
)
def test_parse_policy_rejects_invalid_coverage_paths(coverage_xml: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(_payload(coverage_xml=coverage_xml))


@pytest.mark.parametrize("value", [-0.1, 100.1])
def test_parse_policy_rejects_line_percent_outside_bounds(value: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        _audit.parse_policy(_payload(line_minimum_percent=value))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("measured_lines", 0, "measured_lines must be positive"),
        ("covered_lines", -1, "covered_lines must be within"),
        ("covered_lines", 101, "covered_lines must be within"),
        ("line_percent", 94.0, "line_percent does not match"),
        ("branch_percent", 101.0, "branch_percent must be between"),
        ("remote_ci_run", True, "remote_ci_run must be an integer"),
        ("remote_ci_run", 0, "remote_ci_run must be positive"),
    ],
)
def test_parse_policy_rejects_invalid_baseline(field: str, value: object, message: str) -> None:
    baseline = _baseline()
    baseline[field] = value

    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(_payload(baseline=baseline))


@pytest.mark.parametrize(
    ("branch", "message"),
    [
        (_branch(mode="gate"), "mode is unsupported"),
        (_branch(minimum_percent=80.0), "must be null in observe mode"),
        (_branch(mode="enforce"), "is required in enforce mode"),
        (_branch(mode="enforce", minimum_percent=101.0), "must be between 0 and 100"),
        (_branch(require_data=False), "require_data must remain true"),
    ],
)
def test_parse_policy_rejects_invalid_branch_contract(
    branch: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(_payload(branch=branch))


def test_parse_policy_accepts_enforced_branch_threshold() -> None:
    policy = _audit.parse_policy(_payload(branch=_branch(mode="enforce", minimum_percent=80.0)))

    assert policy.branch.mode == "enforce"
    assert policy.branch.minimum_percent == 80.0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload(schema_version=2), "schema_version must be 1"),
        (
            _payload(
                promotion={"minimum_successful_remote_runs": 0, "rule": "rule"},
            ),
            "minimum_successful_remote_runs must be positive",
        ),
    ],
)
def test_parse_policy_rejects_invalid_top_level_contract(payload: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(payload)


def test_load_policy_reads_utf8_json(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    assert _audit.load_policy(path).baseline.source == "coverage artifact"


def test_parse_coverage_xml_returns_aggregate_metrics(tmp_path: Path) -> None:
    path = tmp_path / "coverage.xml"
    _write_xml(path)

    metrics = _audit.parse_coverage_xml(path)

    assert metrics.lines_covered == 95
    assert metrics.lines_valid == 100
    assert metrics.line_rate == 0.95
    assert metrics.line_percent == 95.0
    assert metrics.branches_covered == 18
    assert metrics.branches_valid == 20
    assert metrics.branch_rate == 0.9
    assert metrics.branch_percent == 90.0


@pytest.mark.parametrize(
    ("root", "attributes", "message"),
    [
        ("report", {}, "root must be <coverage>"),
        ("coverage", {}, "misses lines-covered"),
        (
            "coverage",
            {
                "lines-covered": "x",
                "lines-valid": "100",
                "line-rate": "0.95",
                "branches-covered": "18",
                "branches-valid": "20",
                "branch-rate": "0.9",
            },
            "lines-covered must be an integer",
        ),
        (
            "coverage",
            {
                "lines-covered": "-1",
                "lines-valid": "100",
                "line-rate": "0.95",
                "branches-covered": "18",
                "branches-valid": "20",
                "branch-rate": "0.9",
            },
            "lines-covered must be non-negative",
        ),
        (
            "coverage",
            {
                "lines-covered": "95",
                "lines-valid": "100",
                "branches-covered": "18",
                "branches-valid": "20",
                "branch-rate": "0.9",
            },
            "misses line-rate",
        ),
        (
            "coverage",
            {
                "lines-covered": "95",
                "lines-valid": "100",
                "line-rate": "bad",
                "branches-covered": "18",
                "branches-valid": "20",
                "branch-rate": "0.9",
            },
            "line-rate must be numeric",
        ),
        (
            "coverage",
            {
                "lines-covered": "95",
                "lines-valid": "100",
                "line-rate": "nan",
                "branches-covered": "18",
                "branches-valid": "20",
                "branch-rate": "0.9",
            },
            "line-rate must be between 0 and 1",
        ),
        (
            "coverage",
            {
                "lines-covered": "95",
                "lines-valid": "100",
                "line-rate": "1.1",
                "branches-covered": "18",
                "branches-valid": "20",
                "branch-rate": "0.9",
            },
            "line-rate must be between 0 and 1",
        ),
        (
            "coverage",
            {
                "lines-covered": "101",
                "lines-valid": "100",
                "line-rate": "1.0",
                "branches-covered": "18",
                "branches-valid": "20",
                "branch-rate": "0.9",
            },
            "line covered count exceeds valid count",
        ),
        (
            "coverage",
            {
                "lines-covered": "95",
                "lines-valid": "100",
                "line-rate": "0.9",
                "branches-covered": "18",
                "branches-valid": "20",
                "branch-rate": "0.9",
            },
            "line rate does not match",
        ),
        (
            "coverage",
            {
                "lines-covered": "1",
                "lines-valid": "0",
                "line-rate": "0",
                "branches-covered": "0",
                "branches-valid": "0",
                "branch-rate": "0",
            },
            "line zero-valid counts are inconsistent",
        ),
        (
            "coverage",
            {
                "lines-covered": "0",
                "lines-valid": "0",
                "line-rate": "0",
                "branches-covered": "0",
                "branches-valid": "0",
                "branch-rate": "0",
            },
            "must contain measured lines",
        ),
        (
            "coverage",
            {
                "lines-covered": "95",
                "lines-valid": "100",
                "line-rate": "0.95",
                "branches-covered": "21",
                "branches-valid": "20",
                "branch-rate": "1",
            },
            "branch covered count exceeds valid count",
        ),
        (
            "coverage",
            {
                "lines-covered": "95",
                "lines-valid": "100",
                "line-rate": "0.95",
                "branches-covered": "18",
                "branches-valid": "20",
                "branch-rate": "0.8",
            },
            "branch rate does not match",
        ),
    ],
)
def test_parse_coverage_xml_rejects_invalid_reports(
    tmp_path: Path,
    root: str,
    attributes: dict[str, str],
    message: str,
) -> None:
    path = tmp_path / "coverage.xml"
    _write_xml(path, root=root, attributes=attributes)

    with pytest.raises(ValueError, match=message):
        _audit.parse_coverage_xml(path)


def test_audit_metrics_observes_branch_and_enforces_line() -> None:
    policy = _audit.parse_policy(_payload())
    passing = _audit.CoverageMetrics(95, 100, 0.95, 18, 20, 0.9)
    low_line = _audit.CoverageMetrics(89, 100, 0.89, 18, 20, 0.9)
    no_branches = _audit.CoverageMetrics(95, 100, 0.95, 0, 0, 0.0)

    assert _audit.audit_metrics(policy, passing).errors == ()
    assert _audit.audit_metrics(policy, low_line).errors == (
        "line coverage 89.00% is below 90.00%",
    )
    assert _audit.audit_metrics(policy, no_branches).errors == (
        "branch coverage data is required but no branch opportunities were measured",
    )


def test_audit_metrics_enforces_promoted_branch_threshold() -> None:
    policy = _audit.parse_policy(_payload(branch=_branch(mode="enforce", minimum_percent=80.0)))
    metrics = _audit.CoverageMetrics(95, 100, 0.95, 15, 20, 0.75)

    assert _audit.audit_metrics(policy, metrics).errors == (
        "branch coverage 75.00% is below 80.00%",
    )


def test_result_formatters_report_observation_and_blockers() -> None:
    policy = _audit.parse_policy(_payload())
    metrics = _audit.CoverageMetrics(89, 100, 0.89, 18, 20, 0.9)
    result = _audit.audit_metrics(policy, metrics)

    assert result.passed is False
    assert "branch: 90.00% (observe; 20 opportunities)" in _audit.format_result(result)
    assert "blocker: line coverage" in _audit.format_result(result)
    assert "status: fail" in _audit.format_result(result)
    payload = json.loads(_audit._json_result(result))
    assert payload["status"] == "fail"
    assert payload["branch"]["minimum_percent"] is None

    assert "line minimum 90.00%" in _audit.format_policy(policy)
    policy_payload = json.loads(_audit._json_policy(policy))
    assert policy_payload["status"] == "pass"
    assert policy_payload["branch_mode"] == "observe"


def test_main_supports_default_xml_explicit_xml_json_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_payload()), encoding="utf-8")
    xml_path = tmp_path / "coverage.xml"
    _write_xml(xml_path)
    monkeypatch.setattr(_audit, "ROOT", tmp_path)

    assert _audit.main(["--policy", str(policy_path), "--validate-policy"]) == 0
    assert "coverage policy current" in capsys.readouterr().out
    assert _audit.main(["--policy", str(policy_path), "--validate-policy", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"

    assert _audit.main(["--policy", str(policy_path)]) == 0
    assert "status: pass" in capsys.readouterr().out

    assert (
        _audit.main(["--policy", str(policy_path), "--coverage-xml", str(xml_path), "--json"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "pass"

    xml_path.write_text("<coverage>", encoding="utf-8")
    assert _audit.main(["--policy", str(policy_path)]) == 2
    assert "coverage policy audit failed" in capsys.readouterr().out
    assert _audit.main(["--policy", str(policy_path), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_live_policy_records_latest_remote_line_baseline() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    policy = _audit.load_policy(repo_root / "tools" / "coverage_policy.json")

    assert policy.line_minimum_percent == 90.0
    assert policy.branch.mode == "observe"
    assert policy.branch.require_data is True
    assert policy.baseline.origin_commit == "4c3a4fee4935ca4eb8c75bb7f33b3262038faeaa"
    assert policy.baseline.remote_ci_run == 29180328986
    assert policy.baseline.covered_lines == 75731
    assert policy.baseline.measured_lines == 81858
