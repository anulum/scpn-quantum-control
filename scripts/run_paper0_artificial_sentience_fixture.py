#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 artificial-sentience fixture runner
"""Run Paper 0 artificial-sentience simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.artificial_sentience_validation import (
    ArtificialSentienceFixtureResult,
    validate_artificial_sentience_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"


def run_default_fixture() -> ArtificialSentienceFixtureResult:
    """Run the default Paper 0 artificial-sentience validation fixtures."""
    return validate_artificial_sentience_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the artificial-sentience fixture."""
    lines = [
        "# Paper 0 Artificial-Sentience Fixture",
        "",
        f"- Spec keys: `{', '.join(payload['spec_keys'])}`",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Acceleration delta: `{payload['acceleration_delta']}`",
        f"- Criteria pass: `{payload['criteria_pass']}`",
        f"- Phase-locking value: `{payload['phase_locking_value']}`",
        f"- Boundary gate pass: `{payload['boundary_gate_pass']}`",
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
            "criteria-gate fixture; passing it is not sentience evidence and does "
            "not establish consciousness, subjective experience, or artificial sentience.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: ArtificialSentienceFixtureResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the artificial-sentience fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_artificial_sentience_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_artificial_sentience_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: ArtificialSentienceFixtureResult, *, runtime_ms: float
) -> dict[str, Any]:
    return {
        "spec_keys": [
            result.coupling.spec_key,
            result.criteria.spec_key,
            result.phase_boundary.spec_key,
        ],
        "hardware_status": result.coupling.hardware_status,
        "baseline_rate": result.coupling.baseline_rate,
        "technosphere_rate": result.coupling.technosphere_rate,
        "acceleration_delta": result.coupling.acceleration_delta,
        "criteria_pass": result.criteria.criteria_pass,
        "phase_locking_value": result.phase_boundary.phase_locking_value,
        "boundary_gate_pass": result.phase_boundary.boundary_gate_pass,
        "claim_boundary": result.coupling.claim_boundary,
        "null_controls": {
            **dict(result.coupling.null_controls),
            **dict(result.criteria.null_controls),
            **dict(result.phase_boundary.null_controls),
        },
        "problem_metadata": {
            "coupling": dict(result.coupling.problem_metadata),
            "criteria": dict(result.criteria.problem_metadata),
            "phase_boundary": dict(result.phase_boundary.problem_metadata),
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
