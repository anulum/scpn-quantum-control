#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 computational-threshold fixture runner
"""Run Paper 0 EQ0119-EQ0122 computational-threshold simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.computational_threshold_validation import (
    CoherenceCurrentValidationResult,
    IITThresholdValidationResult,
    InformationEnergyTransductionValidationResult,
    validate_coherence_noether_current_fixture,
    validate_iit_or_threshold_fixture,
    validate_information_energy_transduction_fixture,
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
class ComputationalThresholdFixtureBundle:
    """Combined EQ0119-EQ0122 computational-threshold fixture result."""

    iit_or_threshold: IITThresholdValidationResult
    coherence_noether_current: CoherenceCurrentValidationResult
    information_energy_transduction: InformationEnergyTransductionValidationResult


def run_default_fixture() -> ComputationalThresholdFixtureBundle:
    """Run the default Paper 0 computational-threshold validation fixtures."""
    return ComputationalThresholdFixtureBundle(
        iit_or_threshold=validate_iit_or_threshold_fixture(),
        coherence_noether_current=validate_coherence_noether_current_fixture(),
        information_energy_transduction=validate_information_energy_transduction_fixture(),
    )


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the computational-threshold fixtures."""
    threshold = payload["iit_or_threshold"]
    noether = payload["coherence_noether_current"]
    iet = payload["information_energy_transduction"]
    lines = [
        "# Paper 0 Computational-Threshold Fixture",
        "",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Runtime: `{payload['runtime_ms']}` ms",
        "",
        "## IIT-OR Threshold Boundary",
        "",
        f"- Spec key: `{threshold['spec_key']}`",
        f"- Protocol: `{threshold['validation_protocol']}`",
        f"- Source equations: `{', '.join(threshold['source_equation_ids'])}`",
        f"- Proportionality residual: `{threshold['proportionality_residual']}`",
        f"- Threshold labels: `{threshold['threshold_labels']}`",
        "",
        "## Noether Coherence Current",
        "",
        f"- Spec key: `{noether['spec_key']}`",
        f"- Protocol: `{noether['validation_protocol']}`",
        f"- Source equations: `{', '.join(noether['source_equation_ids'])}`",
        f"- Global phase invariance error: `{noether['global_phase_invariance_error']}`",
        f"- Divergence residual: `{noether['divergence_residual']}`",
        "",
        "## Information-Energy Transduction",
        "",
        f"- Spec key: `{iet['spec_key']}`",
        f"- Protocol: `{iet['validation_protocol']}`",
        f"- Source equations: `{', '.join(iet['source_equation_ids'])}`",
        f"- Constant-density max abs: `{iet['constant_density_max_abs']}`",
        f"- Gaussian residual RMS: `{iet['gaussian_residual_rms']}`",
        "",
        "## Null Controls",
        "",
        "### IIT-OR Threshold Boundary",
        "",
    ]
    for key, value in sorted(threshold["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "### Noether Coherence Current", ""])
    for key, value in sorted(noether["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "### Information-Energy Transduction", ""])
    for key, value in sorted(iet["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "No provider submission is represented. These are simulator-only "
            "finite mathematical fixtures for source-anchored Paper 0 "
            "computational-threshold equations. They do not promote empirical "
            "collapse, consciousness, or downward-causation claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: ComputationalThresholdFixtureBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the computational-threshold fixtures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_computational_threshold_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_computational_threshold_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: ComputationalThresholdFixtureBundle,
    *,
    runtime_ms: float,
) -> dict[str, Any]:
    return {
        "hardware_status": "simulator_only_no_provider_submission",
        "iit_or_threshold": _threshold_payload(result.iit_or_threshold),
        "coherence_noether_current": _noether_payload(result.coherence_noether_current),
        "information_energy_transduction": _iet_payload(result.information_energy_transduction),
        "runtime_ms": runtime_ms,
    }


def _threshold_payload(result: IITThresholdValidationResult) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "threshold_labels": list(result.threshold_labels),
        "proportionality_residual": result.proportionality_residual,
        "null_controls": dict(result.null_controls),
        "problem_metadata": dict(result.problem_metadata),
    }


def _noether_payload(result: CoherenceCurrentValidationResult) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "global_phase_invariance_error": result.global_phase_invariance_error,
        "divergence_residual": result.divergence_residual,
        "null_controls": dict(result.null_controls),
        "problem_metadata": dict(result.problem_metadata),
    }


def _iet_payload(result: InformationEnergyTransductionValidationResult) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "constant_density_max_abs": result.constant_density_max_abs,
        "gaussian_residual_rms": result.gaussian_residual_rms,
        "null_controls": dict(result.null_controls),
        "problem_metadata": dict(result.problem_metadata),
    }


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
