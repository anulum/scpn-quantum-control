#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 computational-unifier fixture runner
"""Run Paper 0 EQ0115-EQ0118 computational-unifier simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.computational_unifier_validation import (
    ABLBoundaryValidationResult,
    CyclicOperatorValidationResult,
    InformationThermodynamicsValidationResult,
    validate_cyclic_operator_fixture,
    validate_information_thermodynamics_fixture,
    validate_tsvf_abl_fixture,
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
class ComputationalUnifierFixtureBundle:
    """Combined EQ0115-EQ0118 computational-unifier fixture result."""

    cyclic_operator: CyclicOperatorValidationResult
    tsvf_abl: ABLBoundaryValidationResult
    information_thermodynamics: InformationThermodynamicsValidationResult


def run_default_fixture() -> ComputationalUnifierFixtureBundle:
    """Run the default Paper 0 computational-unifier validation fixtures."""
    return ComputationalUnifierFixtureBundle(
        cyclic_operator=validate_cyclic_operator_fixture(),
        tsvf_abl=validate_tsvf_abl_fixture(),
        information_thermodynamics=validate_information_thermodynamics_fixture(),
    )


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the computational-unifier fixtures."""
    cyclic = payload["cyclic_operator"]
    tsvf = payload["tsvf_abl"]
    thermo = payload["information_thermodynamics"]
    lines = [
        "# Paper 0 Computational-Unifier Fixture",
        "",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Runtime: `{payload['runtime_ms']}` ms",
        "",
        "## Cyclic Operator Boundary",
        "",
        f"- Spec key: `{cyclic['spec_key']}`",
        f"- Protocol: `{cyclic['validation_protocol']}`",
        f"- Source equations: `{', '.join(cyclic['source_equation_ids'])}`",
        f"- Unitarity error: `{cyclic['unitarity_error']}`",
        f"- Cycle-closure residual: `{cyclic['cycle_closure_residual']}`",
        "",
        "## TSVF/ABL Boundary Probability",
        "",
        f"- Spec key: `{tsvf['spec_key']}`",
        f"- Protocol: `{tsvf['validation_protocol']}`",
        f"- Source equations: `{', '.join(tsvf['source_equation_ids'])}`",
        f"- Probability normalisation error: `{tsvf['probability_normalisation_error']}`",
        "",
        "## Information Thermodynamics",
        "",
        f"- Spec key: `{thermo['spec_key']}`",
        f"- Protocol: `{thermo['validation_protocol']}`",
        f"- Source equations: `{', '.join(thermo['source_equation_ids'])}`",
        f"- GSL margin: `{thermo['gsl_margin']}`",
        f"- MI-negentropy error: `{thermo['mutual_information_negentropy_error']}`",
        "",
        "## Null Controls",
        "",
        "### Cyclic Operator Boundary",
        "",
    ]
    for key, value in sorted(cyclic["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "### TSVF/ABL Boundary Probability", ""])
    for key, value in sorted(tsvf["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "### Information Thermodynamics", ""])
    for key, value in sorted(thermo["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "No provider submission is represented. These are simulator-only "
            "boundary fixtures for source-anchored Paper 0 computational-unifier "
            "equations. They validate finite mathematical consistency checks, "
            "not empirical confirmation of the broader mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: ComputationalUnifierFixtureBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the computational-unifier fixtures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_computational_unifier_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_computational_unifier_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: ComputationalUnifierFixtureBundle,
    *,
    runtime_ms: float,
) -> dict[str, Any]:
    return {
        "hardware_status": "simulator_only_no_provider_submission",
        "cyclic_operator": _cyclic_payload(result.cyclic_operator),
        "tsvf_abl": _abl_payload(result.tsvf_abl),
        "information_thermodynamics": _thermo_payload(result.information_thermodynamics),
        "runtime_ms": runtime_ms,
    }


def _cyclic_payload(result: CyclicOperatorValidationResult) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "unitarity_error": result.unitarity_error,
        "cycle_closure_residual": result.cycle_closure_residual,
        "null_controls": dict(result.null_controls),
        "problem_metadata": dict(result.problem_metadata),
    }


def _abl_payload(result: ABLBoundaryValidationResult) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "probabilities": list(result.probabilities),
        "probability_normalisation_error": result.probability_normalisation_error,
        "null_controls": dict(result.null_controls),
        "problem_metadata": dict(result.problem_metadata),
    }


def _thermo_payload(result: InformationThermodynamicsValidationResult) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "negentropy_rate": result.negentropy_rate,
        "information_entropy_rate": result.information_entropy_rate,
        "total_entropy_rate": result.total_entropy_rate,
        "gsl_margin": result.gsl_margin,
        "mutual_information_negentropy_error": result.mutual_information_negentropy_error,
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
