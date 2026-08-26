# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded-director L16 evidence writer
"""Digest-bound JSON and Markdown evidence for the bounded L16 director."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .director_contracts import L16_DIRECTOR_CLAIM_BOUNDARY, L16_DIRECTOR_SCHEMA
from .director_product import run_l16_director_suite


def canonical_l16_json(value: object) -> str:
    """Return canonical JSON text used by the L16 director integrity digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def l16_evidence_payload() -> dict[str, object]:
    """Run the real suite and return its payload with a SHA-256 digest."""
    payload = run_l16_director_suite().to_payload()
    digest = hashlib.sha256(canonical_l16_json(payload).encode("utf-8")).hexdigest()
    return payload | {"content_digest": digest}


def render_l16_evidence_markdown(payload: Mapping[str, object]) -> str:
    """Render a compact human-readable view of validated L16 director evidence."""
    certificates = cast(list[dict[str, object]], payload["certificates"])
    routes = cast(list[dict[str, object]], payload["routes"])
    blockers = cast(list[str], payload["promotion_blockers"])
    lines = [
        "# L16 director functional evidence",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Functional passed: `{str(payload['functional_passed']).lower()}`",
        f"- Promotion ready: `{str(payload['promotion_ready']).lower()}`",
        f"- Action diversity: `{str(payload['action_diversity']).lower()}`",
        f"- Content digest: `{payload['content_digest']}`",
        "- Execution: bounded local exact simulation; no provider, QPU, or hardware actuation.",
        "",
        "## Indicator certificates",
        "",
        "| Scenario | Echo | Variance | Susceptibility | R | Score | Heuristic | Safety action | Informative |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for certificate in certificates:
        scenario = cast(dict[str, object], certificate["scenario"])
        informative = cast(list[str], certificate["informative_indicators"])
        lines.append(
            "| {scenario} | {echo:.9g} | {variance:.9g} | {susceptibility:.9g} | "
            "{order:.9g} | {score:.9g} | {heuristic} | {codesign} | {informative} |".format(
                scenario=scenario["scenario_id"],
                echo=_as_float(certificate["loschmidt_echo"], "loschmidt_echo"),
                variance=_as_float(certificate["energy_variance"], "energy_variance"),
                susceptibility=_as_float(
                    certificate["fidelity_susceptibility"],
                    "fidelity_susceptibility",
                ),
                order=_as_float(certificate["order_parameter"], "order_parameter"),
                score=_as_float(certificate["heuristic_score"], "heuristic_score"),
                heuristic=certificate["heuristic_action"],
                codesign=certificate["codesign_action"],
                informative=", ".join(informative) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Governed routes",
            "",
            "| Route | Status | Boundary |",
            "|---|---|---|",
        ]
    )
    for route in routes:
        reason = str(route["closure_reason"]).replace("|", "\\|") or "bounded local only"
        lines.append(f"| {route['route_id']} | {route['closure_status']} | {reason} |")
    lines.extend(["", "## Promotion blockers", ""])
    lines.extend(f"- {blocker}" for blocker in blockers)
    lines.extend(["", "## Claim boundary", "", str(payload["claim_boundary"]), ""])
    return "\n".join(lines)


def validate_l16_evidence(payload: object) -> tuple[str, ...]:
    """Return fail-closed findings for one L16 director evidence payload."""
    if not isinstance(payload, dict):
        return ("payload must be a JSON object",)
    data = cast(dict[str, object], payload)
    findings: list[str] = []
    expected = {
        "schema": L16_DIRECTOR_SCHEMA,
        "claim_boundary": L16_DIRECTOR_CLAIM_BOUNDARY,
        "functional_passed": True,
        "promotion_ready": False,
        "provider_execution": False,
        "hardware_execution": False,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            findings.append(f"{key} must equal {value!r}")
    certificates = data.get("certificates")
    expected_scenarios = {
        "paper27_baseline",
        "susceptibility_probe",
        "weak_coupling_probe",
    }
    if not isinstance(certificates, list):
        findings.append("certificates must be an array")
    else:
        scenario_ids = {
            cast(dict[str, object], certificate.get("scenario", {})).get("scenario_id")
            for certificate in certificates
            if isinstance(certificate, dict)
        }
        if len(certificates) != 3 or scenario_ids != expected_scenarios:
            findings.append("certificates must cover the three frozen scenarios exactly once")
        if any(
            not isinstance(certificate, dict) or certificate.get("passed") is not True
            for certificate in certificates
        ):
            findings.append("every L16 certificate must pass its functional gate")
    routes = data.get("routes")
    if not isinstance(routes, list) or len(routes) != 2:
        findings.append("routes must contain the two governed L16 rows")
    elif {
        (route.get("route_id"), route.get("closure_status"))
        for route in routes
        if isinstance(route, dict)
    } != {
        ("adapter:l16.local_indicator", "supported"),
        ("adapter:l16.autonomous_hardware_control", "permanent_boundary"),
    }:
        findings.append("routes must retain supported-local and blocked-hardware statuses")
    blockers = data.get("promotion_blockers")
    if (
        not isinstance(blockers, list)
        or not blockers
        or any(not isinstance(item, str) or not item.strip() for item in blockers)
    ):
        findings.append("promotion_blockers must be a non-empty string array")
    digest = data.get("content_digest")
    unsigned = {key: value for key, value in data.items() if key != "content_digest"}
    try:
        expected_digest = hashlib.sha256(canonical_l16_json(unsigned).encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        findings.append("payload contains a non-canonical JSON value")
        expected_digest = None
    if expected_digest is not None and digest != expected_digest:
        findings.append("content_digest does not match canonical payload bytes")
    return tuple(findings)


def write_l16_evidence(
    json_path: Path,
    markdown_path: Path,
    *,
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate and atomically write real or independently supplied L16 evidence."""
    selected_payload = dict(payload) if payload is not None else l16_evidence_payload()
    findings = validate_l16_evidence(selected_payload)
    if findings:
        raise RuntimeError("invalid L16 director evidence: " + "; ".join(findings))
    json_text = json.dumps(selected_payload, indent=2, sort_keys=True) + "\n"
    markdown_text = render_l16_evidence_markdown(selected_payload)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(json_path, json_text)
    _atomic_write(markdown_path, markdown_text)
    return selected_payload


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    """Run the bounded L16 director evidence CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/l16_director_product/bounded_director_evidence.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("data/l16_director_product/bounded_director_evidence.md"),
    )
    args = parser.parse_args(argv)
    payload = write_l16_evidence(args.json_output, args.markdown_output)
    print(args.json_output)
    print(args.markdown_output)
    print(f"functional_passed={str(payload['functional_passed']).lower()}")
    print("promotion_ready=false; no provider, QPU, or hardware actuation")
    return 0


__all__ = [
    "canonical_l16_json",
    "l16_evidence_payload",
    "main",
    "render_l16_evidence_markdown",
    "validate_l16_evidence",
    "write_l16_evidence",
]
