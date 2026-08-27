# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — convergence-example convergence evidence writer
"""Digest-bound JSON and Markdown evidence for the convergence-example convergence suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .contracts import (
    ML_CONVERGENCE_CLAIM_BOUNDARY,
    ML_CONVERGENCE_SCHEMA,
    ConvergenceSuiteEvidence,
    ModelFamily,
)
from .suite import run_ml_convergence_suite


def canonical_json(value: object) -> str:
    """Return canonical JSON text used by the evidence digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def evidence_payload(suite: ConvergenceSuiteEvidence) -> dict[str, object]:
    """Return suite payload plus a SHA-256 integrity digest."""
    payload = suite.to_payload()
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload | {"content_digest": digest}


def render_evidence_markdown(payload: Mapping[str, object]) -> str:
    """Render a compact human-readable view of validated suite evidence."""
    certificates = cast(list[dict[str, object]], payload["certificates"])
    rows = cast(list[dict[str, object]], payload["framework_rows"])
    lines = [
        "# QNN/QGNN/QSNN convergence evidence",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Passed: `{str(payload['passed']).lower()}`",
        f"- Content digest: `{payload['content_digest']}`",
        "- Execution: local synthetic simulator only; no provider or hardware execution.",
        "",
        "## Convergence certificates",
        "",
        "| Family | Example | Initial loss | Best loss | Target | Loss drop | Replay | Passed |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for certificate in certificates:
        spec = cast(dict[str, object], certificate["spec"])
        lines.append(
            "| {family} | {example} | {initial:.9g} | {best:.9g} | {target:.9g} | "
            "{drop:.9g} | {replay} | {passed} |".format(
                family=spec["family"],
                example=spec["example_id"],
                initial=_as_float(certificate["initial_loss"], "initial_loss"),
                best=_as_float(certificate["best_loss"], "best_loss"),
                target=_as_float(spec["target_loss"], "target_loss"),
                drop=_as_float(certificate["loss_drop"], "loss_drop"),
                replay=certificate["deterministic_replay"],
                passed=certificate["passed"],
            )
        )
    lines.extend(
        [
            "",
            "## Framework matrix",
            "",
            "| Family | Framework | Status | Required | Executed | Passed | Reason |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        reason = str(row["reason"]).replace("|", "\\|")
        lines.append(
            f"| {row['family']} | {row['framework']} | {row['status']} | "
            f"{row['required']} | {row['executed']} | {row['passed']} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(payload["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_ml_convergence_evidence(
    json_path: Path,
    markdown_path: Path,
    *,
    required_qnn_frameworks: tuple[str, ...] = (),
) -> dict[str, object]:
    """Run the suite and atomically replace its JSON and Markdown evidence."""
    suite = run_ml_convergence_suite(required_qnn_frameworks=required_qnn_frameworks)
    payload = evidence_payload(suite)
    findings = validate_ml_convergence_evidence(payload)
    if findings:
        raise RuntimeError("invalid ML convergence evidence: " + "; ".join(findings))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_path, render_evidence_markdown(payload))
    return payload


def validate_ml_convergence_evidence(payload: object) -> tuple[str, ...]:
    """Return fail-closed findings for one committed ML convergence payload."""
    if not isinstance(payload, dict):
        return ("payload must be a JSON object",)
    data = cast(dict[str, object], payload)
    findings: list[str] = []
    expected = {
        "schema": ML_CONVERGENCE_SCHEMA,
        "claim_boundary": ML_CONVERGENCE_CLAIM_BOUNDARY,
        "passed": True,
        "provider_execution": False,
        "hardware_execution": False,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            findings.append(f"{key} must equal {value!r}")
    certificates = data.get("certificates")
    if not isinstance(certificates, list):
        findings.append("certificates must be an array")
    else:
        families = {
            cast(dict[str, object], certificate.get("spec", {})).get("family")
            for certificate in certificates
            if isinstance(certificate, dict)
        }
        if families != {family.value for family in ModelFamily} or len(certificates) != 3:
            findings.append("certificates must cover qnn, qgnn, and qsnn exactly once")
        if any(
            not isinstance(certificate, dict) or certificate.get("passed") is not True
            for certificate in certificates
        ):
            findings.append("every convergence certificate must pass")
    framework_rows = data.get("framework_rows")
    if not isinstance(framework_rows, list) or not framework_rows:
        findings.append("framework_rows must be a non-empty array")
    elif any(
        not isinstance(row, dict)
        or not row.get("family")
        or not row.get("framework")
        or not row.get("status")
        or not row.get("reason")
        or row.get("gate_passed") is not True
        for row in framework_rows
    ):
        findings.append("framework rows must be complete and pass their declared gates")
    digest = data.get("content_digest")
    unsigned = {key: value for key, value in data.items() if key != "content_digest"}
    expected_digest = hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()
    if digest != expected_digest:
        findings.append("content_digest does not match canonical payload bytes")
    return tuple(findings)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    """Run the unified evidence CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/ml_convergence_examples/convergence_evidence.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("data/ml_convergence_examples/convergence_evidence.md"),
    )
    parser.add_argument(
        "--require-qnn-framework",
        action="append",
        default=[],
        choices=("jax", "pytorch", "tensorflow"),
    )
    args = parser.parse_args(argv)
    payload = write_ml_convergence_evidence(
        args.json_output,
        args.markdown_output,
        required_qnn_frameworks=tuple(args.require_qnn_framework),
    )
    print(args.json_output)
    print(args.markdown_output)
    print(f"passed={str(payload['passed']).lower()}")
    print("No provider or QPU execution")
    return 0


__all__ = [
    "canonical_json",
    "evidence_payload",
    "main",
    "render_evidence_markdown",
    "validate_ml_convergence_evidence",
    "write_ml_convergence_evidence",
]
