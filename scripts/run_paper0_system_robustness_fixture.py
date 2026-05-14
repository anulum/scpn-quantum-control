#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 system-robustness fixture runner
"""Run Paper 0 system-robustness simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.system_robustness_validation import (
    SystemRobustnessFixtureResult,
    validate_system_robustness_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"


def run_default_fixture() -> SystemRobustnessFixtureResult:
    """Run the default Paper 0 system-robustness validation fixtures."""
    return validate_system_robustness_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the system-robustness fixture."""
    lines = [
        "# Paper 0 System-Robustness Fixture",
        "",
        f"- Spec keys: `{', '.join(payload['spec_keys'])}`",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Largest component loss: `{payload['largest_component_loss']}`",
        f"- Recovery-time ratio: `{payload['recovery_time_ratio']}`",
        f"- Failure probability: `{payload['failure_probability']}`",
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
            "robustness fixture; passing it is not operational security evidence "
            "and does not establish real-world attack resistance, safety, or resilience.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: SystemRobustnessFixtureResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the system-robustness fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_system_robustness_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_system_robustness_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: SystemRobustnessFixtureResult, *, runtime_ms: float
) -> dict[str, Any]:
    return {
        "spec_keys": list(result.spec_keys),
        "hardware_status": result.hardware_status,
        "claim_boundary": result.claim_boundary,
        "intact_largest_component_fraction": result.cascade.intact_largest_component_fraction,
        "attacked_largest_component_fraction": result.cascade.attacked_largest_component_fraction,
        "largest_component_loss": result.cascade.largest_component_loss,
        "near_critical_recovery_time": result.critical_slowing.near_critical_recovery_time,
        "reference_recovery_time": result.critical_slowing.reference_recovery_time,
        "recovery_time_ratio": result.critical_slowing.recovery_time_ratio,
        "ms_qec_success_probability": result.decoherence.ms_qec_success_probability,
        "failure_probability": result.decoherence.failure_probability,
        "unprotected_failure_probability": result.decoherence.unprotected_failure_probability,
        "null_controls": {
            **dict(result.cascade.null_controls),
            **dict(result.critical_slowing.null_controls),
            **dict(result.decoherence.null_controls),
        },
        "problem_metadata": {
            "cascade": dict(result.cascade.problem_metadata),
            "critical_slowing": dict(result.critical_slowing.problem_metadata),
            "decoherence": dict(result.decoherence.problem_metadata),
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
