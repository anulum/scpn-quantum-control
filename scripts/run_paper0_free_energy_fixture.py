#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 free-energy fixture runner
"""Run Paper 0 EQ0130-EQ0131 free-energy simulator fixture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.free_energy_validation import (
    FreeEnergyValidationResult,
    validate_variational_free_energy_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)


def run_default_fixture() -> FreeEnergyValidationResult:
    """Run the default Paper 0 free-energy validation fixture."""
    return validate_variational_free_energy_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the free-energy fixture."""
    lines = [
        "# Paper 0 Free-Energy Fixture",
        "",
        f"- Spec key: `{payload['spec_key']}`",
        f"- Protocol: `{payload['validation_protocol']}`",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Source equations: `{', '.join(payload['source_equation_ids'])}`",
        f"- Complexity KL: `{payload['complexity_kl']}`",
        f"- Accuracy loss: `{payload['accuracy_loss']}`",
        f"- Total free energy: `{payload['total_free_energy']}`",
        f"- Surprise upper-bound margin: `{payload['surprise_upper_bound_margin']}`",
        f"- Runtime: `{payload['runtime_ms']}` ms",
        "",
        "## Null Controls",
        "",
    ]
    for key, value in sorted(payload["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "No provider submission is represented. This is a simulator-only "
            "finite probability-simplex fixture and does not promote empirical "
            "neurobiological or consciousness claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: FreeEnergyValidationResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the free-energy fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_free_energy_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_free_energy_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: FreeEnergyValidationResult, *, runtime_ms: float
) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "complexity_kl": result.complexity_kl,
        "accuracy_loss": result.accuracy_loss,
        "total_free_energy": result.total_free_energy,
        "surprise": result.surprise,
        "decomposition_residual": result.decomposition_residual,
        "surprise_upper_bound_margin": result.surprise_upper_bound_margin,
        "null_controls": dict(result.null_controls),
        "problem_metadata": dict(result.problem_metadata),
        "runtime_ms": runtime_ms,
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
