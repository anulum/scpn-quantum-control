#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 HPC/UPDE bridge fixture runner
"""Run Paper 0 HPC/UPDE bridge simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.hpc_upde_bridge_validation import (
    HpcUpdeBridgeFixtureResult,
    validate_hpc_upde_bridge_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)


def run_default_fixture() -> HpcUpdeBridgeFixtureResult:
    """Run the default Paper 0 HPC/UPDE bridge validation fixtures."""
    return validate_hpc_upde_bridge_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the HPC/UPDE bridge fixture."""
    lines = [
        "# Paper 0 HPC/UPDE Bridge Fixture",
        "",
        f"- Spec keys: `{', '.join(payload['spec_keys'])}`",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- HPC prediction-error norm: `{payload['hpc_prediction_error_norm']}`",
        f"- Phase squared-error delta: `{payload['phase_squared_error_delta']}`",
        f"- Initial XY potential: `{payload['initial_potential']}`",
        f"- Aligned XY potential: `{payload['aligned_potential']}`",
        f"- Max gradient residual: `{payload['max_gradient_residual']}`",
        f"- Max drift residual: `{payload['max_drift_residual']}`",
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
            "finite mathematical fixture for predictive-coding flow, phase-error "
            "coupling, and the XY-potential gradient identity. It does not promote "
            "biological, cosmological, or consciousness claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: HpcUpdeBridgeFixtureResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the HPC/UPDE bridge fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_hpc_upde_bridge_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_hpc_upde_bridge_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: HpcUpdeBridgeFixtureResult, *, runtime_ms: float
) -> dict[str, Any]:
    return {
        "spec_keys": [result.hpc.spec_key, result.phase.spec_key, result.gradient.spec_key],
        "hardware_status": result.gradient.hardware_status,
        "hpc_prediction_error_norm": result.hpc.prediction_error_norm,
        "hpc_upward_error_norm": result.hpc.upward_error_norm,
        "phase_weighted_residual_norm": result.phase.weighted_residual_norm,
        "phase_squared_error_delta": result.phase.squared_error_delta,
        "initial_potential": result.gradient.initial_potential,
        "aligned_potential": result.gradient.aligned_potential,
        "max_gradient_residual": result.gradient.max_gradient_residual,
        "max_drift_residual": result.gradient.max_drift_residual,
        "null_controls": {
            **dict(result.hpc.null_controls),
            **dict(result.phase.null_controls),
            **dict(result.gradient.null_controls),
        },
        "problem_metadata": {
            "hpc": dict(result.hpc.problem_metadata),
            "phase": dict(result.phase.problem_metadata),
            "gradient": dict(result.gradient.problem_metadata),
        },
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
