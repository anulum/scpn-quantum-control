#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 ethical-gauge fixture runner
"""Run Paper 0 EQ0123-EQ0128 ethical-gauge simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.ethical_gauge_validation import (
    CausalEntropicForceValidationResult,
    EthicalConnectionBoundaryValidationResult,
    EthicalYangMillsActionValidationResult,
    validate_causal_entropic_force_fixture,
    validate_ethical_connection_boundary_fixture,
    validate_ethical_yang_mills_action_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)


@dataclass(frozen=True, slots=True)
class EthicalGaugeFixtureBundle:
    """Combined EQ0123-EQ0128 ethical-gauge fixture result."""

    action: EthicalYangMillsActionValidationResult
    boundary: EthicalConnectionBoundaryValidationResult
    cef: CausalEntropicForceValidationResult


def run_default_fixture() -> EthicalGaugeFixtureBundle:
    """Run the default Paper 0 ethical-gauge validation fixtures."""
    return EthicalGaugeFixtureBundle(
        action=validate_ethical_yang_mills_action_fixture(),
        boundary=validate_ethical_connection_boundary_fixture(),
        cef=validate_causal_entropic_force_fixture(),
    )


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the ethical-gauge fixtures."""
    lines = [
        "# Paper 0 Ethical-Gauge Fixture",
        "",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Runtime: `{payload['runtime_ms']}` ms",
        "",
    ]
    for title, key in (
        ("Yang-Mills Action", "action"),
        ("Connection Boundary", "boundary"),
        ("Causal Entropic Force", "cef"),
    ):
        item = payload[key]
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Spec key: `{item['spec_key']}`",
                f"- Protocol: `{item['validation_protocol']}`",
                f"- Source equations: `{', '.join(item['source_equation_ids'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "No provider submission is represented. These are simulator-only "
            "finite mathematical fixtures and do not promote empirical ethics, "
            "teleology, or force-coupling claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: EthicalGaugeFixtureBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the ethical-gauge fixtures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_ethical_gauge_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_ethical_gauge_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(result: EthicalGaugeFixtureBundle, *, runtime_ms: float) -> dict[str, Any]:
    return {
        "hardware_status": "simulator_only_no_provider_submission",
        "action": _result_payload(result.action),
        "boundary": _result_payload(result.boundary),
        "cef": _result_payload(result.cef),
        "runtime_ms": runtime_ms,
    }


def _result_payload(result: Any) -> dict[str, Any]:
    payload = {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "null_controls": dict(result.null_controls),
        "problem_metadata": dict(result.problem_metadata),
    }
    for name in (
        "action_value",
        "gauge_invariance_error",
        "stationary_residual",
        "euler_lagrange_residual",
        "orientation_reversal_error",
        "complexity_flux_margin",
        "force_norm",
        "gradient_residual",
        "entropy_ascent_delta",
    ):
        if hasattr(result, name):
            payload[name] = getattr(result, name)
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    result = run_default_fixture()
    runtime_ms = (time.perf_counter() - start) * 1000.0
    paths = write_outputs(
        result, output_dir=args.output_dir, date_tag=args.date_tag, runtime_ms=runtime_ms
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(_json_ready_payload(result, runtime_ms=runtime_ms), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
