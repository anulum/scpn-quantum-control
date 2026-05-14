#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 A-CEF alignment fixture runner
"""Run Paper 0 A-CEF ethical-alignment fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.acef_alignment_validation import (
    ACEFAlignmentFixtureResult,
    validate_acef_alignment_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"


def run_default_fixture() -> ACEFAlignmentFixtureResult:
    """Run the default Paper 0 A-CEF alignment fixture."""
    return validate_acef_alignment_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the A-CEF alignment fixture."""
    acef = payload["acef"]
    lines = [
        "# Paper 0 A-CEF Alignment Fixture",
        "",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Force norm: `{acef['force_norm']}`",
        f"- SEC objective delta: `{acef['sec_objective_delta']}`",
        f"- Consequence phase-steering label: `{acef['consequence_phase_steering_label']}`",
        f"- Runtime: `{payload['runtime_ms']}` ms",
        "",
        "## Spec Keys",
        "",
    ]
    for key in payload["spec_keys"]:
        lines.append(f"- `{key}`")
    lines.extend(["", "## Null Controls", ""])
    for key, value in sorted(acef["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "No provider submission is represented. This is a source-bounded "
            "A-CEF simulator contract; passing it is not empirical evidence and "
            "does not establish that any governance or alignment deployment is safe.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: ACEFAlignmentFixtureResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the A-CEF alignment fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_acef_alignment_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_acef_alignment_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: ACEFAlignmentFixtureResult, *, runtime_ms: float
) -> dict[str, Any]:
    acef = result.acef
    return {
        "spec_keys": list(result.spec_keys),
        "hardware_status": result.hardware_status,
        "claim_boundary": result.claim_boundary,
        "problem_metadata": dict(result.problem_metadata),
        "acef": {
            "spec_key": acef.spec_key,
            "validation_protocol": acef.validation_protocol,
            "hardware_status": acef.hardware_status,
            "source_ledger_ids": list(acef.source_ledger_ids),
            "source_equation_ids": list(acef.source_equation_ids),
            "formal_statement": acef.formal_statement,
            "evaluation_state": list(acef.evaluation_state),
            "force": list(acef.force),
            "force_norm": acef.force_norm,
            "sec_objective_delta": acef.sec_objective_delta,
            "consequence_phase_steering_label": acef.consequence_phase_steering_label,
            "claim_boundary": acef.claim_boundary,
            "null_controls": dict(acef.null_controls),
            "problem_metadata": dict(acef.problem_metadata),
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
