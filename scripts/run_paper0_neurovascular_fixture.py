#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 neurovascular fixture runner
"""Run the Paper 0 neurovascular phase-coupling simulator fixture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.neurovascular_validation import (
    NeurovascularValidationResult,
    validate_neurovascular_phase_coupling_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)


def run_default_fixture() -> NeurovascularValidationResult:
    """Run the default Paper 0 neurovascular phase-coupling validation fixture."""
    return validate_neurovascular_phase_coupling_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the neurovascular fixture."""
    lines = [
        "# Paper 0 Neurovascular Phase-Coupling Fixture",
        "",
        f"- Spec key: `{payload['spec_key']}`",
        f"- Protocol: `{payload['validation_protocol']}`",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Source equations: `{', '.join(payload['source_equation_ids'])}`",
        f"- Source ledgers: `{', '.join(payload['source_ledger_ids'])}`",
        f"- Phase-locking value: `{payload['phase_locking_value']}`",
        f"- Mean frequency slip: `{payload['mean_frequency_slip']}`",
        f"- Final phase difference: `{payload['final_phase_difference']}`",
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
            "two-oscillator fixture for the source-anchored Paper 0 "
            "neurovascular phase-coupling equation. Biomedical pathology "
            "channels are labelled controls, not claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: NeurovascularValidationResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the neurovascular fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_neurovascular_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_neurovascular_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: NeurovascularValidationResult,
    *,
    runtime_ms: float,
) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "phase_locking_value": result.phase_locking_value,
        "mean_frequency_slip": result.mean_frequency_slip,
        "final_phase_difference": result.final_phase_difference,
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
