#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native speedup benchmark regression gate
"""Fail-closed regression gate for the native dense-Hamiltonian benchmark.

Validates a fresh report against a committed baseline under an explicit
threshold policy, mirroring the SCPN-CONTROL apparatus. It fails closed: a
missing report or baseline, a tampered report payload or baseline digest, a
baseline metric absent from the report, or a metric with no threshold policy are
all failures. Absolute-latency comparison is only valid on the same CPU as the
baseline, so a CPU-model mismatch is flagged; ``--evidence-only`` collects the
verdict without ever failing, for generic CI runners whose CPU differs from the
declared-hardware baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - retained for downstream legacy interpreters
    import tomli as tomllib

VERDICT_SCHEMA = "scpn-quantum-control.native-speedup-verdict.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = REPO_ROOT / "benchmarks" / "native_speedup_thresholds.toml"

#: Latency metrics are upper-bounded; throughput is lower-bounded.
_LOWER_BOUNDED = frozenset({"throughput_ops_s"})


@dataclass(frozen=True)
class Finding:
    """A single gate failure: a regression, missing evidence, or policy gap."""

    kind: str
    benchmark: str
    language: str
    metric: str
    baseline: float | None
    current: float | None
    ratio: float | None
    threshold: float | None
    direction: str
    detail: str


def metric_direction(metric: str) -> str:
    """Return ``lower`` for throughput-like metrics, ``upper`` otherwise."""
    return "lower" if metric in _LOWER_BOUNDED else "upper"


def _payload_digest(payload: dict[str, Any]) -> str:
    """Return the SHA-256 of a payload, excluding any existing digest field."""
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    serialised = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


def canonical_metrics_digest(benchmarks: dict[str, Any]) -> str:
    """Return the SHA-256 of the benchmark metrics for baseline tamper detection."""
    serialised = json.dumps(benchmarks, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


def parse_thresholds(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Validate and normalise a threshold policy mapping.

    A ``[default]`` table is mandatory; a non-numeric or non-positive ratio is a
    policy error (raised), not a silent default.
    """
    if "default" not in raw or not raw["default"]:
        raise ValueError("threshold policy must define a non-empty [default] table")
    policy: dict[str, dict[str, float]] = {}
    for section, table in raw.items():
        if not isinstance(table, dict):
            raise ValueError(f"threshold section '{section}' must be a table")
        normalised: dict[str, float] = {}
        for metric, ratio in table.items():
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise ValueError(f"threshold '{section}.{metric}' must be a number")
            value = float(ratio)
            if value <= 0.0 or value != value or value == float("inf"):
                raise ValueError(f"threshold '{section}.{metric}' must be positive and finite")
            normalised[metric] = value
        policy[section] = normalised
    return policy


def resolve_threshold(
    thresholds: dict[str, dict[str, float]], benchmark: str, metric: str
) -> float | None:
    """Return the per-benchmark threshold, falling back to ``[default]``."""
    bench_table = thresholds.get(benchmark, {})
    if metric in bench_table:
        return bench_table[metric]
    return thresholds.get("default", {}).get(metric)


def validate_report(report: dict[str, Any]) -> list[str]:
    """Reject a report missing required fields or with a tampered digest."""
    errors: list[str] = []
    if report.get("schema_version") is None:
        errors.append("report is missing schema_version")
    if "benchmarks" not in report:
        errors.append("report is missing benchmarks")
    stamped = report.get("payload_sha256")
    if stamped is None:
        errors.append("report is missing payload_sha256")
    elif stamped != _payload_digest(report):
        errors.append("payload_sha256 does not match the report body (tampered)")
    return errors


def verify_baseline_integrity(baseline: dict[str, Any]) -> list[str]:
    """Reject a baseline whose recorded digest does not match its metrics."""
    errors: list[str] = []
    if "benchmarks" not in baseline:
        errors.append("baseline is missing benchmarks")
        return errors
    stamped = baseline.get("baseline_sha256")
    if stamped is None:
        errors.append("baseline is missing baseline_sha256")
    elif stamped != canonical_metrics_digest(baseline["benchmarks"]):
        errors.append("baseline_sha256 does not match the baseline metrics (tampered)")
    return errors


def _coerce_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for bools/non-numbers/NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if out == out else None


def hardware_mismatch(report: dict[str, Any], baseline: dict[str, Any]) -> Finding | None:
    """Flag a latency comparison across different CPUs as invalid."""
    report_cpu = report.get("provenance", {}).get("cpu_model")
    baseline_cpu = baseline.get("provenance", {}).get("cpu_model")
    if report_cpu and baseline_cpu and report_cpu != baseline_cpu:
        return Finding(
            "hardware_mismatch",
            "",
            "",
            "",
            None,
            None,
            None,
            None,
            "",
            f"report CPU {report_cpu!r} != baseline CPU {baseline_cpu!r}; "
            "latency comparison is invalid off declared hardware",
        )
    return None


def compare(
    report: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
) -> list[Finding]:
    """Compare every baseline metric against the report, fail-closed on gaps."""
    findings: list[Finding] = []
    report_benches = report.get("benchmarks", {})
    for bench_name, base_bench in baseline.get("benchmarks", {}).items():
        report_bench = report_benches.get(bench_name)
        for language, base_metrics in base_bench.get("languages", {}).items():
            report_metrics = None
            if isinstance(report_bench, dict):
                report_metrics = report_bench.get("languages", {}).get(language)
            for metric, base_raw in base_metrics.items():
                base_value = _coerce_float(base_raw)
                if base_value is None:
                    continue
                direction = metric_direction(metric)
                threshold = resolve_threshold(thresholds, bench_name, metric)
                if threshold is None:
                    findings.append(
                        Finding(
                            "policy_gap",
                            bench_name,
                            language,
                            metric,
                            base_value,
                            None,
                            None,
                            None,
                            direction,
                            "no threshold policy for this metric",
                        )
                    )
                    continue
                current_value = None
                if isinstance(report_metrics, dict):
                    current_value = _coerce_float(report_metrics.get(metric))
                if current_value is None:
                    findings.append(
                        Finding(
                            "missing_metric",
                            bench_name,
                            language,
                            metric,
                            base_value,
                            None,
                            None,
                            threshold,
                            direction,
                            "metric present in baseline but absent from report",
                        )
                    )
                    continue
                ratio = current_value / base_value if base_value != 0.0 else float("inf")
                regressed = ratio > threshold if direction == "upper" else ratio < threshold
                if regressed:
                    bound = "<=" if direction == "upper" else ">="
                    findings.append(
                        Finding(
                            "regression",
                            bench_name,
                            language,
                            metric,
                            base_value,
                            current_value,
                            ratio,
                            threshold,
                            direction,
                            f"ratio {ratio:.3f} violates {bound} {threshold:.3f}",
                        )
                    )
    return findings


def gate(
    report: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
    *,
    generated_utc: str,
) -> dict[str, Any]:
    """Run validation + comparison and assemble a fail-closed verdict report."""
    findings: list[Finding] = []
    for error in validate_report(report):
        findings.append(Finding("report_invalid", "", "", "", None, None, None, None, "", error))
    for error in verify_baseline_integrity(baseline):
        findings.append(Finding("baseline_invalid", "", "", "", None, None, None, None, "", error))
    if not findings:
        mismatch = hardware_mismatch(report, baseline)
        if mismatch is not None:
            findings.append(mismatch)
        findings.extend(compare(report, baseline, thresholds))
    verdict = {
        "schema_version": VERDICT_SCHEMA,
        "generated_utc": generated_utc,
        "passed": not findings,
        "baseline_commit": baseline.get("baseline_commit"),
        "report_commit": report.get("provenance", {}).get("commit"),
        "findings": [asdict(finding) for finding in findings],
    }
    verdict["payload_sha256"] = _payload_digest(verdict)
    return verdict


def _load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_thresholds_file(path: Path) -> dict[str, dict[str, float]]:
    """Load and validate a threshold policy from a TOML file."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return parse_thresholds(raw)


def main(argv: list[str] | None = None) -> int:
    """Load the report, baseline, and thresholds, then run the gate."""
    parser = argparse.ArgumentParser(description="Native speedup benchmark regression gate.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--generated-utc", default="")
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Collect and report the verdict but never fail (for generic CI "
        "runners whose CPU differs from the declared-hardware baseline).",
    )
    args = parser.parse_args(argv)

    for label, path in (("report", args.report), ("baseline", args.baseline)):
        if not path.is_file():
            print(f"benchmark gate FAILED: {label} not found at {path}", file=sys.stderr)
            return 1

    report = _load_json(args.report)
    baseline = _load_json(args.baseline)
    thresholds = load_thresholds_file(args.thresholds)
    verdict = gate(report, baseline, thresholds, generated_utc=args.generated_utc)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if verdict["passed"]:
        print("benchmark gate PASSED")
        return 0
    for finding in verdict["findings"]:
        print(
            f"  [{finding['kind']}] {finding['benchmark']} {finding['language']} "
            f"{finding['metric']}: {finding['detail']}",
            file=sys.stderr,
        )
    if args.evidence_only:
        print("benchmark gate findings present but evidence-only: not failing", file=sys.stderr)
        return 0
    print("benchmark gate FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
