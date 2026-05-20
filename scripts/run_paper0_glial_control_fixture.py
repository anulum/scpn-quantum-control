#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 glial-control fixture runner
"""Run the Paper 0 EQ0105-EQ0112 glial-control simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.glial_control_validation import (
    GlialSigmaValidationResult,
    QuantumImmuneValidationResult,
    validate_glial_sigma_control_fixture,
    validate_quantum_immune_interface_fixture,
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
class GlialControlFixtureBundle:
    """Combined EQ0105 and EQ0106-EQ0112 fixture result."""

    quantum_immune: QuantumImmuneValidationResult
    glial_sigma: GlialSigmaValidationResult


def run_default_fixture() -> GlialControlFixtureBundle:
    """Run the default Paper 0 glial-control validation fixtures."""
    return GlialControlFixtureBundle(
        quantum_immune=validate_quantum_immune_interface_fixture(),
        glial_sigma=validate_glial_sigma_control_fixture(),
    )


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the glial-control fixtures."""
    immune = payload["quantum_immune"]
    glial = payload["glial_sigma"]
    lines = [
        "# Paper 0 Glial-Control Fixture",
        "",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Runtime: `{payload['runtime_ms']}` ms",
        "",
        "## Quantum-Immune Interface",
        "",
        f"- Spec key: `{immune['spec_key']}`",
        f"- Protocol: `{immune['validation_protocol']}`",
        f"- Source equations: `{', '.join(immune['source_equation_ids'])}`",
        f"- Lambda value: `{immune['lambda_value']}`",
        f"- Cytokine spectral shift: `{immune['cytokine_spectral_shift']}`",
        f"- Hermiticity error: `{immune['hermiticity_error']}`",
        "",
        "## Glial Sigma Control",
        "",
        f"- Spec key: `{glial['spec_key']}`",
        f"- Protocol: `{glial['validation_protocol']}`",
        f"- Source equations: `{', '.join(glial['source_equation_ids'])}`",
        f"- Final sigma: `{glial['final_sigma']}`",
        f"- Final G: `{glial['final_G']}`",
        f"- Integrated calcium drive: `{glial['integrated_calcium_drive']}`",
        "",
        "## Null Controls",
        "",
        "### Quantum-Immune Interface",
        "",
    ]
    for key, value in sorted(immune["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "### Glial Sigma Control", ""])
    for key, value in sorted(glial["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "No provider submission is represented. These are simulator-only "
            "fixtures for source-anchored Paper 0 immune-interface and "
            "glial-control equations. Biological and quantum-biological claims "
            "remain mechanism hypotheses until empirical boundary labels pass.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: GlialControlFixtureBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the glial-control fixtures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_glial_control_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_glial_control_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: GlialControlFixtureBundle,
    *,
    runtime_ms: float,
) -> dict[str, Any]:
    return {
        "hardware_status": "simulator_only_no_provider_submission",
        "quantum_immune": _immune_payload(result.quantum_immune),
        "glial_sigma": _glial_payload(result.glial_sigma),
        "runtime_ms": runtime_ms,
    }


def _immune_payload(result: QuantumImmuneValidationResult) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "lambda_value": result.lambda_value,
        "high_cytokine_lambda_value": result.high_cytokine_lambda_value,
        "operator_norm": result.operator_norm,
        "cytokine_spectral_shift": result.cytokine_spectral_shift,
        "hermiticity_error": result.hermiticity_error,
        "null_controls": dict(result.null_controls),
        "problem_metadata": dict(result.problem_metadata),
    }


def _glial_payload(result: GlialSigmaValidationResult) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "final_sigma": result.final_sigma,
        "final_G": result.final_G,
        "final_sigma_shift": result.final_sigma_shift,
        "integrated_calcium_drive": result.integrated_calcium_drive,
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
