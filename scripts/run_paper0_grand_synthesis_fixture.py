#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Grand Synthesis fixture runner
"""Run Paper 0 Grand Synthesis and NTHS phase-test fixtures."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.grand_synthesis_validation import (
    GrandSynthesisFixtureResult,
    validate_grand_synthesis_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)


def run_default_fixture() -> GrandSynthesisFixtureResult:
    """Run the default Paper 0 Grand Synthesis fixture."""
    return validate_grand_synthesis_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the Grand Synthesis fixture."""
    phase = payload["nths_phase"]
    lines = [
        "# Paper 0 Grand Synthesis Fixture",
        "",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- SEC delta: `{phase['sec_delta']}`",
        f"- Frustration delta: `{phase['frustration_delta']}`",
        f"- Engagement spin-glass label: `{phase['engagement_spin_glass_label']}`",
        f"- Coherence consensus label: `{phase['coherence_consensus_label']}`",
        f"- Runtime: `{payload['runtime_ms']}` ms",
        "",
        "## Spec Keys",
        "",
    ]
    for key in payload["spec_keys"]:
        lines.append(f"- `{key}`")
    lines.extend(["", "## Null Controls", ""])
    for key, value in sorted(phase["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "No provider submission is represented. This is a source-bounded "
            "simulator contract; passing it is not empirical evidence and does "
            "not establish that any societal phase transition has occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: GrandSynthesisFixtureResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the Grand Synthesis fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_grand_synthesis_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_grand_synthesis_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: GrandSynthesisFixtureResult, *, runtime_ms: float
) -> dict[str, Any]:
    nths_phase = result.nths_phase
    return {
        "spec_keys": list(result.spec_keys),
        "hardware_status": result.hardware_status,
        "claim_boundary": result.claim_boundary,
        "problem_metadata": dict(result.problem_metadata),
        "nths_phase": {
            "spec_key": nths_phase.spec_key,
            "validation_protocol": nths_phase.validation_protocol,
            "hardware_status": nths_phase.hardware_status,
            "source_ledger_ids": list(nths_phase.source_ledger_ids),
            "coherent_metrics": asdict(nths_phase.coherent_metrics),
            "engagement_metrics": asdict(nths_phase.engagement_metrics),
            "sec_delta": nths_phase.sec_delta,
            "frustration_delta": nths_phase.frustration_delta,
            "engagement_spin_glass_label": nths_phase.engagement_spin_glass_label,
            "coherence_consensus_label": nths_phase.coherence_consensus_label,
            "claim_boundary": nths_phase.claim_boundary,
            "null_controls": dict(nths_phase.null_controls),
            "problem_metadata": dict(nths_phase.problem_metadata),
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
