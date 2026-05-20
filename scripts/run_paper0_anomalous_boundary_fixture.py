#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 anomalous-boundary fixture runner
"""Run Paper 0 anomalous-boundary simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.anomalous_boundary_validation import (
    AnomalousBoundaryFixtureResult,
    validate_anomalous_boundary_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)


def run_default_fixture() -> AnomalousBoundaryFixtureResult:
    """Run the default Paper 0 anomalous-boundary validation fixtures."""
    return validate_anomalous_boundary_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the anomalous-boundary fixture."""
    lines = [
        "# Paper 0 Anomalous-Boundary Fixture",
        "",
        f"- Spec keys: `{', '.join(payload['spec_keys'])}`",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- ABL normalisation error: `{payload['abl_probability_normalisation_error']}`",
        f"- CHSH value: `{payload['chsh_value']}`",
        f"- No-signalling residual: `{payload['no_signalling_residual']}`",
        f"- Biased probability: `{payload['biased_probability']}`",
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
            "falsification-boundary fixture; passing it is not anomalous evidence "
            "and does not establish precognition, telepathy, remote perception, "
            "or psychokinesis.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: AnomalousBoundaryFixtureResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the anomalous-boundary fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_anomalous_boundary_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_anomalous_boundary_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: AnomalousBoundaryFixtureResult, *, runtime_ms: float
) -> dict[str, Any]:
    return {
        "spec_keys": list(result.spec_keys),
        "hardware_status": result.hardware_status,
        "claim_boundary": result.claim_boundary,
        "abl_probability_normalisation_error": result.tsvf.probability_normalisation_error,
        "abl_probabilities": list(result.tsvf.probabilities),
        "shifted_post_probabilities": list(result.tsvf.shifted_post_probabilities),
        "chsh_value": result.entanglement.chsh_value,
        "no_signalling_residual": result.entanglement.no_signalling_residual,
        "prior_probability": result.weak_measurement.prior_probability,
        "biased_probability": result.weak_measurement.biased_probability,
        "probability_shift": result.weak_measurement.probability_shift,
        "null_controls": {
            **dict(result.tsvf.null_controls),
            **dict(result.entanglement.null_controls),
            **dict(result.weak_measurement.null_controls),
        },
        "problem_metadata": {
            "tsvf": dict(result.tsvf.problem_metadata),
            "entanglement": dict(result.entanglement.problem_metadata),
            "weak_measurement": dict(result.weak_measurement.problem_metadata),
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
