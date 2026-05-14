#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Stuart-Landau precision fixture runner
"""Run Paper 0 Stuart-Landau precision simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.stuart_landau_precision_validation import (
    StuartLandauPrecisionFixtureResult,
    validate_stuart_landau_precision_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"


def run_default_fixture() -> StuartLandauPrecisionFixtureResult:
    """Run the default Paper 0 Stuart-Landau precision validation fixtures."""
    return validate_stuart_landau_precision_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the Stuart-Landau precision fixture."""
    lines = [
        "# Paper 0 Stuart-Landau Precision Fixture",
        "",
        f"- Spec keys: `{', '.join(payload['spec_keys'])}`",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Max complex/polar residual: `{payload['max_complex_polar_residual']}`",
        f"- Max phase-ratio residual: `{payload['max_phase_ratio_residual']}`",
        f"- Amplitude-ratio deviation: `{payload['max_amplitude_ratio_deviation']}`",
        f"- Rho-gain radius-dot delta: `{payload['rho_gain_radius_dot_delta']}`",
        f"- Incoming/prior phase-drive ratio: `{payload['high_incoming_over_prior_phase_drive_ratio']}`",
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
            "finite Stuart-Landau fixture for complex/polar consistency, "
            "amplitude-ratio precision weighting, and radial gain controls. "
            "It does not promote biological or active-inference claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: StuartLandauPrecisionFixtureResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the Stuart-Landau precision fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_stuart_landau_precision_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_stuart_landau_precision_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: StuartLandauPrecisionFixtureResult, *, runtime_ms: float
) -> dict[str, Any]:
    return {
        "spec_keys": [
            result.upgrade.spec_key,
            result.dynamics.spec_key,
            result.salience.spec_key,
        ],
        "hardware_status": result.upgrade.hardware_status,
        "max_complex_polar_residual": result.upgrade.max_complex_polar_residual,
        "phase_only_missing_amplitude_norm": result.upgrade.phase_only_missing_amplitude_norm,
        "max_phase_ratio_residual": result.dynamics.max_phase_ratio_residual,
        "max_amplitude_ratio_deviation": result.dynamics.max_amplitude_ratio_deviation,
        "rho_gain_radius_dot_delta": result.salience.rho_gain_radius_dot_delta,
        "high_incoming_over_prior_phase_drive_ratio": (
            result.salience.high_incoming_over_prior_phase_drive_ratio
        ),
        "null_controls": {
            **dict(result.upgrade.null_controls),
            **dict(result.dynamics.null_controls),
            **dict(result.salience.null_controls),
        },
        "problem_metadata": {
            "upgrade": dict(result.upgrade.problem_metadata),
            "dynamics": dict(result.dynamics.problem_metadata),
            "salience": dict(result.salience.problem_metadata),
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
