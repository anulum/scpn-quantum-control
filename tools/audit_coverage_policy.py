# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — line gate and branch telemetry policy audit
"""Enforce the line gate while validating branch-enabled Cobertura telemetry.

Coverage.py reports a combined percentage when branch collection is enabled,
so reusing ``--cov-fail-under=90`` would silently change the existing 90%
line contract. This audit parses the generated XML, preserves the line gate,
requires real branch opportunity data, and reports branch coverage separately.
A branch threshold remains optional until remote CI establishes a baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET  # noqa: S405
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "tools" / "coverage_policy.json"

BranchMode = Literal["observe", "enforce"]
_BRANCH_MODES = frozenset({"observe", "enforce"})
_RATE_TOLERANCE = 0.00011


@dataclass(frozen=True)
class CoverageBaseline:
    """Latest line-only remote CI evidence before branch collection."""

    origin_commit: str
    remote_ci_run: int
    source: str
    covered_lines: int
    measured_lines: int
    line_percent: float
    branch_percent: float | None


@dataclass(frozen=True)
class BranchPolicy:
    """Branch-data collection and optional enforcement policy."""

    mode: BranchMode
    require_data: bool
    minimum_percent: float | None


@dataclass(frozen=True)
class CoveragePolicy:
    """Parsed line and branch coverage policy."""

    schema_version: int
    coverage_xml: str
    line_minimum_percent: float
    branch: BranchPolicy
    baseline: CoverageBaseline
    minimum_successful_remote_runs: int
    promotion_rule: str


@dataclass(frozen=True)
class CoverageMetrics:
    """Aggregate Cobertura counts and rates emitted by coverage.py."""

    lines_covered: int
    lines_valid: int
    line_rate: float
    branches_covered: int
    branches_valid: int
    branch_rate: float

    @property
    def line_percent(self) -> float:
        """Return aggregate line coverage as a percentage."""
        return self.line_rate * 100.0

    @property
    def branch_percent(self) -> float:
        """Return aggregate branch coverage as a percentage."""
        return self.branch_rate * 100.0


@dataclass(frozen=True)
class AuditResult:
    """Coverage metrics compared with the tracked policy."""

    policy: CoveragePolicy
    metrics: CoverageMetrics
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether all active coverage requirements pass."""
        return not self.errors


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _text(mapping: dict[str, object], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _integer(mapping: dict[str, object], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be an integer")
    return value


def _number(mapping: dict[str, object], key: str, context: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}.{key} must be finite")
    return result


def _optional_number(mapping: dict[str, object], key: str, context: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _number(mapping, key, context)


def _boolean(mapping: dict[str, object], key: str, context: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a boolean")
    return value


def _percent(value: float, context: str) -> float:
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{context} must be between 0 and 100")
    return value


def _relative_path(value: str, context: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must be repository-relative")
    if path.suffix != ".xml":
        raise ValueError(f"{context} must name an XML file")
    return path.as_posix()


def _parse_baseline(value: object) -> CoverageBaseline:
    context = "policy.baseline"
    row = _mapping(value, context)
    covered_lines = _integer(row, "covered_lines", context)
    measured_lines = _integer(row, "measured_lines", context)
    if measured_lines <= 0:
        raise ValueError(f"{context}.measured_lines must be positive")
    if not 0 <= covered_lines <= measured_lines:
        raise ValueError(f"{context}.covered_lines must be within measured_lines")
    line_percent = _percent(_number(row, "line_percent", context), f"{context}.line_percent")
    expected_percent = covered_lines * 100.0 / measured_lines
    if not math.isclose(line_percent, expected_percent, abs_tol=0.0001):
        raise ValueError(f"{context}.line_percent does not match its line counts")
    branch_percent = _optional_number(row, "branch_percent", context)
    if branch_percent is not None:
        branch_percent = _percent(branch_percent, f"{context}.branch_percent")
    remote_ci_run = _integer(row, "remote_ci_run", context)
    if remote_ci_run <= 0:
        raise ValueError(f"{context}.remote_ci_run must be positive")
    return CoverageBaseline(
        origin_commit=_text(row, "origin_commit", context),
        remote_ci_run=remote_ci_run,
        source=_text(row, "source", context),
        covered_lines=covered_lines,
        measured_lines=measured_lines,
        line_percent=line_percent,
        branch_percent=branch_percent,
    )


def _parse_branch_policy(value: object) -> BranchPolicy:
    context = "policy.branch"
    row = _mapping(value, context)
    mode_value = _text(row, "mode", context)
    if mode_value not in _BRANCH_MODES:
        raise ValueError(f"{context}.mode is unsupported: {mode_value}")
    mode = cast(BranchMode, mode_value)
    minimum_percent = _optional_number(row, "minimum_percent", context)
    if minimum_percent is not None:
        minimum_percent = _percent(minimum_percent, f"{context}.minimum_percent")
    if mode == "observe" and minimum_percent is not None:
        raise ValueError(f"{context}.minimum_percent must be null in observe mode")
    if mode == "enforce" and minimum_percent is None:
        raise ValueError(f"{context}.minimum_percent is required in enforce mode")
    require_data = _boolean(row, "require_data", context)
    if not require_data:
        raise ValueError(f"{context}.require_data must remain true")
    return BranchPolicy(
        mode=mode,
        require_data=require_data,
        minimum_percent=minimum_percent,
    )


def parse_policy(payload: object) -> CoveragePolicy:
    """Parse and validate a decoded coverage policy document."""
    root = _mapping(payload, "policy")
    schema_version = _integer(root, "schema_version", "policy")
    if schema_version != 1:
        raise ValueError("policy.schema_version must be 1")
    promotion = _mapping(root.get("branch_promotion"), "policy.branch_promotion")
    minimum_runs = _integer(
        promotion,
        "minimum_successful_remote_runs",
        "policy.branch_promotion",
    )
    if minimum_runs <= 0:
        raise ValueError("policy.branch_promotion.minimum_successful_remote_runs must be positive")
    return CoveragePolicy(
        schema_version=schema_version,
        coverage_xml=_relative_path(_text(root, "coverage_xml", "policy"), "policy.coverage_xml"),
        line_minimum_percent=_percent(
            _number(root, "line_minimum_percent", "policy"),
            "policy.line_minimum_percent",
        ),
        branch=_parse_branch_policy(root.get("branch")),
        baseline=_parse_baseline(root.get("baseline")),
        minimum_successful_remote_runs=minimum_runs,
        promotion_rule=_text(promotion, "rule", "policy.branch_promotion"),
    )


def load_policy(path: Path) -> CoveragePolicy:
    """Load a UTF-8 JSON coverage policy."""
    return parse_policy(cast(object, json.loads(path.read_text(encoding="utf-8"))))


def _xml_integer(attributes: dict[str, str], key: str) -> int:
    value = attributes.get(key)
    if value is None:
        raise ValueError(f"coverage XML misses {key}")
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"coverage XML {key} must be an integer") from exc
    if result < 0:
        raise ValueError(f"coverage XML {key} must be non-negative")
    return result


def _xml_rate(attributes: dict[str, str], key: str) -> float:
    value = attributes.get(key)
    if value is None:
        raise ValueError(f"coverage XML misses {key}")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"coverage XML {key} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"coverage XML {key} must be between 0 and 1")
    return result


def _validate_rate(rate: float, covered: int, valid: int, context: str) -> None:
    if valid == 0:
        if covered != 0 or rate != 0.0:
            raise ValueError(f"coverage XML {context} zero-valid counts are inconsistent")
        return
    if covered > valid:
        raise ValueError(f"coverage XML {context} covered count exceeds valid count")
    expected = covered / valid
    if not math.isclose(rate, expected, abs_tol=_RATE_TOLERANCE):
        raise ValueError(f"coverage XML {context} rate does not match its counts")


def parse_coverage_xml(path: Path) -> CoverageMetrics:
    """Parse aggregate coverage.py Cobertura metrics from an XML report."""
    root = ET.parse(path).getroot()  # noqa: S314
    if root.tag != "coverage":
        raise ValueError("coverage XML root must be <coverage>")
    attributes = dict(root.attrib)
    lines_covered = _xml_integer(attributes, "lines-covered")
    lines_valid = _xml_integer(attributes, "lines-valid")
    line_rate = _xml_rate(attributes, "line-rate")
    branches_covered = _xml_integer(attributes, "branches-covered")
    branches_valid = _xml_integer(attributes, "branches-valid")
    branch_rate = _xml_rate(attributes, "branch-rate")
    _validate_rate(line_rate, lines_covered, lines_valid, "line")
    _validate_rate(branch_rate, branches_covered, branches_valid, "branch")
    if lines_valid == 0:
        raise ValueError("coverage XML must contain measured lines")
    return CoverageMetrics(
        lines_covered=lines_covered,
        lines_valid=lines_valid,
        line_rate=line_rate,
        branches_covered=branches_covered,
        branches_valid=branches_valid,
        branch_rate=branch_rate,
    )


def audit_metrics(policy: CoveragePolicy, metrics: CoverageMetrics) -> AuditResult:
    """Compare aggregate XML metrics with the active line and branch policy."""
    errors: list[str] = []
    if metrics.line_percent + 1e-9 < policy.line_minimum_percent:
        errors.append(
            f"line coverage {metrics.line_percent:.2f}% is below "
            f"{policy.line_minimum_percent:.2f}%"
        )
    if policy.branch.require_data and metrics.branches_valid == 0:
        errors.append("branch coverage data is required but no branch opportunities were measured")
    branch_minimum = policy.branch.minimum_percent
    if (
        policy.branch.mode == "enforce"
        and branch_minimum is not None
        and metrics.branch_percent + 1e-9 < branch_minimum
    ):
        errors.append(
            f"branch coverage {metrics.branch_percent:.2f}% is below {branch_minimum:.2f}%"
        )
    return AuditResult(policy=policy, metrics=metrics, errors=tuple(errors))


def format_result(result: AuditResult) -> str:
    """Render a deterministic coverage policy summary."""
    lines = [
        "coverage policy audit:",
        f"- line: {result.metrics.line_percent:.2f}% "
        f"(minimum {result.policy.line_minimum_percent:.2f}%)",
        f"- branch: {result.metrics.branch_percent:.2f}% "
        f"({result.policy.branch.mode}; {result.metrics.branches_valid} opportunities)",
    ]
    lines.extend(f"- blocker: {error}" for error in result.errors)
    lines.append(f"status: {'pass' if result.passed else 'fail'}")
    return "\n".join(lines)


def _json_result(result: AuditResult) -> str:
    return json.dumps(
        {
            "branch": {
                "covered": result.metrics.branches_covered,
                "minimum_percent": result.policy.branch.minimum_percent,
                "mode": result.policy.branch.mode,
                "percent": round(result.metrics.branch_percent, 6),
                "valid": result.metrics.branches_valid,
            },
            "errors": list(result.errors),
            "line": {
                "covered": result.metrics.lines_covered,
                "minimum_percent": result.policy.line_minimum_percent,
                "percent": round(result.metrics.line_percent, 6),
                "valid": result.metrics.lines_valid,
            },
            "status": "pass" if result.passed else "fail",
        },
        indent=2,
        sort_keys=True,
    )


def format_policy(policy: CoveragePolicy) -> str:
    """Render the active thresholds without requiring a coverage report."""
    return (
        "coverage policy current: "
        f"line minimum {policy.line_minimum_percent:.2f}%; "
        f"branch mode {policy.branch.mode}; "
        f"branch data required {policy.branch.require_data}"
    )


def _json_policy(policy: CoveragePolicy) -> str:
    return json.dumps(
        {
            "branch_data_required": policy.branch.require_data,
            "branch_minimum_percent": policy.branch.minimum_percent,
            "branch_mode": policy.branch.mode,
            "line_minimum_percent": policy.line_minimum_percent,
            "status": "pass",
        },
        indent=2,
        sort_keys=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--coverage-xml", type=Path)
    parser.add_argument(
        "--validate-policy",
        action="store_true",
        help="validate and report policy without reading coverage XML",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the line-gate and branch-telemetry coverage audit."""
    args = _parser().parse_args(argv)
    try:
        policy = load_policy(args.policy.resolve())
        if args.validate_policy:
            print(_json_policy(policy) if args.json else format_policy(policy))
            return 0
        coverage_xml = (
            args.coverage_xml.resolve()
            if args.coverage_xml is not None
            else (ROOT / policy.coverage_xml).resolve()
        )
        result = audit_metrics(policy, parse_coverage_xml(coverage_xml))
    except (OSError, ValueError, ET.ParseError) as exc:
        if args.json:
            print(json.dumps({"errors": [str(exc)], "status": "error"}, indent=2, sort_keys=True))
        else:
            print(f"coverage policy audit failed:\n- {exc}")
        return 2
    print(_json_result(result) if args.json else format_result(result))
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
