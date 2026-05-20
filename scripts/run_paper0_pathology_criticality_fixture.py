#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 pathology/criticality fixture runner
"""Run Paper 0 pathology/criticality simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.pathology_criticality_validation import (
    PathologyCriticalityFixtureResult,
    validate_pathology_criticality_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)


def run_default_fixture() -> PathologyCriticalityFixtureResult:
    """Run the default Paper 0 pathology/criticality validation fixtures."""
    return validate_pathology_criticality_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the pathology/criticality fixture."""
    lines = [
        "# Paper 0 Pathology/Criticality Fixture",
        "",
        f"- Spec keys: `{', '.join(payload['spec_keys'])}`",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Pathology index: `{payload['pathology_index']}`",
        f"- Baseline index: `{payload['baseline_index']}`",
        f"- Index delta vs baseline: `{payload['index_delta_vs_baseline']}`",
        f"- Sigma label: `{payload['sigma_label']}`",
        f"- Restoration index delta: `{payload['restoration_index_delta']}`",
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
            "No clinical or provider submission is represented. This is a "
            "simulator-only finite systems-metric fixture and does not represent "
            "diagnosis, treatment guidance, medical advice, or empirical clinical evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: PathologyCriticalityFixtureResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the pathology/criticality fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_pathology_criticality_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_pathology_criticality_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: PathologyCriticalityFixtureResult, *, runtime_ms: float
) -> dict[str, Any]:
    return {
        "spec_keys": [
            result.coherence.spec_key,
            result.criticality.spec_key,
            result.restoration.spec_key,
        ],
        "hardware_status": result.coherence.hardware_status,
        "pathology_index": result.coherence.pathology_index,
        "baseline_index": result.coherence.baseline_index,
        "index_delta_vs_baseline": result.coherence.index_delta_vs_baseline,
        "sigma": result.criticality.sigma,
        "sigma_label": result.criticality.sigma_label,
        "criticality_distance": result.criticality.criticality_distance,
        "initial_index": result.restoration.initial_index,
        "restored_index": result.restoration.restored_index,
        "restoration_index_delta": result.restoration.restoration_index_delta,
        "claim_boundary": result.coherence.claim_boundary,
        "null_controls": {
            **dict(result.coherence.null_controls),
            **dict(result.criticality.null_controls),
            **dict(result.restoration.null_controls),
        },
        "problem_metadata": {
            "coherence": dict(result.coherence.problem_metadata),
            "criticality": dict(result.criticality.problem_metadata),
            "restoration": dict(result.restoration.problem_metadata),
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
