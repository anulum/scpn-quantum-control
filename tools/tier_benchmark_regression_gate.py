#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tier benchmark regression alarm
"""Fail-closed regression alarm for ``scpn-quantum-control.tier-benchmark.v1`` artefacts.

Compares a freshly generated tier-benchmark artefact against the committed
same-environment artefact under an explicit per-backend threshold policy,
mirroring ``tools/benchmark_native_speedup_gate.py``. It fails closed: a
missing or tampered artefact on either side, a measured baseline cell whose
tier silently vanished from the report, a backend with no threshold policy,
and a comparison with zero overlapping cells are all findings. A tier that the
report deliberately skipped (``--tiers`` subset runs) is counted as skipped,
not failed. P50 latency is the only compared statistic: at CI sample counts
the higher percentiles are dominated by hosted-runner scheduling noise.

Hosted-runner CPU models drift across runner generations, so a CPU-model
mismatch against the committed baseline is flagged rather than compared
silently; ``--evidence-only`` collects the verdict without ever failing, which
is the warn-first stage of the staged promotion documented in
``.github/workflows/benchmark-kuramoto-tiers.yml``.
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

ARTIFACT_SCHEMA = "scpn-quantum-control.tier-benchmark.v1"
VERDICT_SCHEMA = "scpn-quantum-control.tier-benchmark-verdict.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = REPO_ROOT / "benchmarks" / "tier_regression_thresholds.toml"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "benchmarks" / "tiers" / "kuramoto_tiers.ci.json"

#: An unavailable report row whose reason carries this marker was excluded on
#: purpose (a ``--tiers`` subset run), not lost to a packaging or dispatch bug.
EXCLUDED_REASON_MARKER = "excluded by --tiers"

#: Cap on per-finding GitHub annotation lines so a systemic slowdown (every
#: cell regressing at once) cannot flood the Actions annotation budget.
MAX_ANNOTATIONS = 20


@dataclass(frozen=True)
class Finding:
    """A single alarm: a regression, missing evidence, or policy gap."""

    kind: str
    operation: str
    size: int | None
    backend: str
    baseline_p50_us: float | None
    current_p50_us: float | None
    ratio: float | None
    threshold: float | None
    detail: str


def _payload_digest(payload: dict[str, Any]) -> str:
    """Return the SHA-256 of a payload, excluding any existing digest field."""
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    serialised = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


def parse_thresholds(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Validate and normalise a per-backend threshold policy mapping.

    A ``[default]`` table is mandatory; a non-numeric or non-positive ratio is
    a policy error (raised), not a silent default. Section names are operation
    names (``[default]`` aside); keys are backend names; values are the largest
    acceptable current/baseline P50 ratio.
    """
    if "default" not in raw or not raw["default"]:
        raise ValueError("threshold policy must define a non-empty [default] table")
    policy: dict[str, dict[str, float]] = {}
    for section, table in raw.items():
        if not isinstance(table, dict):
            raise ValueError(f"threshold section '{section}' must be a table")
        normalised: dict[str, float] = {}
        for backend, ratio in table.items():
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise ValueError(f"threshold '{section}.{backend}' must be a number")
            value = float(ratio)
            if value <= 0.0 or value != value or value == float("inf"):
                raise ValueError(f"threshold '{section}.{backend}' must be positive and finite")
            normalised[backend] = value
        policy[section] = normalised
    return policy


def resolve_threshold(
    thresholds: dict[str, dict[str, float]], operation: str, backend: str
) -> float | None:
    """Return the per-operation threshold, falling back to ``[default]``."""
    operation_table = thresholds.get(operation, {})
    if backend in operation_table:
        return operation_table[backend]
    return thresholds.get("default", {}).get(backend)


def validate_artifact(payload: dict[str, Any], label: str) -> list[str]:
    """Reject an artefact with the wrong schema, no results, or a bad digest."""
    errors: list[str] = []
    if payload.get("schema_version") != ARTIFACT_SCHEMA:
        errors.append(f"{label} schema_version must be {ARTIFACT_SCHEMA}")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        errors.append(f"{label} is missing a non-empty results list")
    stamped = payload.get("payload_sha256")
    if stamped is None:
        errors.append(f"{label} is missing payload_sha256")
    elif stamped != _payload_digest(payload):
        errors.append(f"{label} payload_sha256 does not match its body (tampered)")
    return errors


def _coerce_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for bools/non-numbers/NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if out == out else None


def _blank_finding(kind: str, detail: str) -> Finding:
    """Return an artefact-level finding with no per-cell coordinates."""
    return Finding(kind, "", None, "", None, None, None, None, detail)


def environment_mismatch(report: dict[str, Any], baseline: dict[str, Any]) -> Finding | None:
    """Flag a comparison across environments (``ci`` vs ``local``) as invalid."""
    report_env = report.get("environment")
    baseline_env = baseline.get("environment")
    if report_env != baseline_env:
        return _blank_finding(
            "environment_mismatch",
            f"report environment {report_env!r} != baseline environment "
            f"{baseline_env!r}; tier latencies are only comparable within one "
            "environment class",
        )
    return None


def hardware_mismatch(report: dict[str, Any], baseline: dict[str, Any]) -> Finding | None:
    """Flag a latency comparison across different CPU models as invalid."""
    report_cpu = report.get("provenance", {}).get("cpu_model")
    baseline_cpu = baseline.get("provenance", {}).get("cpu_model")
    if report_cpu and baseline_cpu and report_cpu != baseline_cpu:
        return _blank_finding(
            "hardware_mismatch",
            f"report CPU {report_cpu!r} != baseline CPU {baseline_cpu!r}; "
            "absolute-latency comparison is invalid across CPU models",
        )
    return None


def _measured_p50(row: dict[str, Any]) -> float | None:
    """Return the row's P50 latency when it is a measured row, else ``None``."""
    if row.get("status") != "measured":
        return None
    stats = row.get("stats")
    if not isinstance(stats, dict):
        return None
    return _coerce_float(stats.get("p50_us"))


def _index_report_rows(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    """Index report rows as ``(operation, size) -> backend -> row``."""
    index: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for result in report.get("results", []):
        key = (str(result.get("operation")), int(result.get("size", -1)))
        index[key] = {str(row.get("backend")): row for row in result.get("rows", [])}
    return index


def compare(
    report: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
) -> tuple[list[Finding], dict[str, int]]:
    """Compare every measured baseline cell against the report, fail-closed.

    Returns the findings plus a summary of compared and deliberately skipped
    cells. Zero compared cells is itself a finding: a vacuous comparison must
    not read as a pass.
    """
    findings: list[Finding] = []
    compared = 0
    skipped = 0
    report_index = _index_report_rows(report)
    for result in baseline.get("results", []):
        operation = str(result.get("operation"))
        size = int(result.get("size", -1))
        report_rows = report_index.get((operation, size), {})
        for row in result.get("rows", []):
            backend = str(row.get("backend"))
            baseline_p50 = _measured_p50(row)
            if baseline_p50 is None:
                continue
            threshold = resolve_threshold(thresholds, operation, backend)
            if threshold is None:
                findings.append(
                    Finding(
                        "policy_gap",
                        operation,
                        size,
                        backend,
                        baseline_p50,
                        None,
                        None,
                        None,
                        "no threshold policy for this backend",
                    )
                )
                continue
            report_row = report_rows.get(backend)
            if report_row is None or report_row.get("status") != "measured":
                reason = (report_row or {}).get("reason") or ""
                if EXCLUDED_REASON_MARKER in reason:
                    skipped += 1
                    continue
                findings.append(
                    Finding(
                        "missing_backend",
                        operation,
                        size,
                        backend,
                        baseline_p50,
                        None,
                        None,
                        threshold,
                        "tier measured in the baseline but absent from the report "
                        f"(reason: {reason or 'no row'})",
                    )
                )
                continue
            current_p50 = _measured_p50(report_row)
            if current_p50 is None:
                findings.append(
                    Finding(
                        "missing_metric",
                        operation,
                        size,
                        backend,
                        baseline_p50,
                        None,
                        None,
                        threshold,
                        "measured report row carries no finite p50_us",
                    )
                )
                continue
            compared += 1
            ratio = current_p50 / baseline_p50 if baseline_p50 != 0.0 else float("inf")
            if ratio > threshold:
                findings.append(
                    Finding(
                        "regression",
                        operation,
                        size,
                        backend,
                        baseline_p50,
                        current_p50,
                        ratio,
                        threshold,
                        f"p50 ratio {ratio:.3f} violates <= {threshold:.3f}",
                    )
                )
    if compared == 0:
        findings.append(
            _blank_finding(
                "no_overlap",
                "no baseline cell could be compared against the report; a "
                "vacuous comparison must not pass",
            )
        )
    return findings, {"compared_cells": compared, "skipped_cells": skipped}


def gate(
    report: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
    *,
    generated_utc: str,
) -> dict[str, Any]:
    """Run validation + comparison and assemble a fail-closed verdict report."""
    findings: list[Finding] = []
    for error in validate_artifact(report, "report"):
        findings.append(_blank_finding("report_invalid", error))
    for error in validate_artifact(baseline, "baseline"):
        findings.append(_blank_finding("baseline_invalid", error))
    summary = {"compared_cells": 0, "skipped_cells": 0}
    if not findings:
        for mismatch in (
            environment_mismatch(report, baseline),
            hardware_mismatch(report, baseline),
        ):
            if mismatch is not None:
                findings.append(mismatch)
        cell_findings, summary = compare(report, baseline, thresholds)
        findings.extend(cell_findings)
    verdict = {
        "schema_version": VERDICT_SCHEMA,
        "generated_utc": generated_utc,
        "passed": not findings,
        "baseline_commit": baseline.get("provenance", {}).get("commit"),
        "report_commit": report.get("provenance", {}).get("commit"),
        "summary": summary,
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


def _annotation_lines(findings: list[dict[str, Any]], *, evidence_only: bool) -> list[str]:
    """Render capped GitHub Actions annotation lines for the findings."""
    level = "warning" if evidence_only else "error"
    lines = [
        f"::{level} title=Tier benchmark regression gate::"
        f"[{finding['kind']}] {finding['operation']}/{finding['size']}/"
        f"{finding['backend']}: {finding['detail']}"
        for finding in findings[:MAX_ANNOTATIONS]
    ]
    if len(findings) > MAX_ANNOTATIONS:
        lines.append(
            f"::{level} title=Tier benchmark regression gate::"
            f"{len(findings) - MAX_ANNOTATIONS} further findings truncated; "
            "see the verdict artefact"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Load the report, baseline, and thresholds, then run the alarm."""
    parser = argparse.ArgumentParser(description="Tier benchmark regression alarm.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--generated-utc", default="")
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Collect and report the verdict but never fail (the warn-first "
        "stage of the staged promotion to a blocking gate).",
    )
    args = parser.parse_args(argv)

    for label, path in (("report", args.report), ("baseline", args.baseline)):
        if not path.is_file():
            print(f"tier benchmark gate FAILED: {label} not found at {path}", file=sys.stderr)
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

    summary = verdict["summary"]
    print(
        f"tier benchmark gate: compared {summary['compared_cells']} cells, "
        f"skipped {summary['skipped_cells']} deliberately excluded cells"
    )
    if verdict["passed"]:
        print("tier benchmark gate PASSED")
        return 0
    for finding in verdict["findings"]:
        print(
            f"  [{finding['kind']}] {finding['operation']}/{finding['size']}/"
            f"{finding['backend']}: {finding['detail']}",
            file=sys.stderr,
        )
    for line in _annotation_lines(verdict["findings"], evidence_only=args.evidence_only):
        print(line)
    if args.evidence_only:
        print(
            "tier benchmark gate findings present but evidence-only: not failing", file=sys.stderr
        )
        return 0
    print("tier benchmark gate FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
