#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 validation-strategy fixture runner
"""Run Paper 0 validation-strategy roadmap fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.validation_strategy import (
    ValidationStrategyFixtureResult,
    validate_validation_strategy_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"


def run_default_fixture() -> ValidationStrategyFixtureResult:
    """Run the default Paper 0 validation-strategy fixture."""
    return validate_validation_strategy_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the validation-strategy fixture."""
    lines = [
        "# Paper 0 Validation Strategy Fixture",
        "",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Stage count: `{payload['stage_count']}`",
        f"- Domain count: `{payload['domain_count']}`",
        f"- Stage order valid: `{payload['stage_order_valid']}`",
        f"- Runtime: `{payload['runtime_ms']}` ms",
        "",
        "## Domains",
        "",
    ]
    for domain in payload["domains"]:
        lines.append(f"- `{domain}`")
    lines.extend(["", "## Null Controls", ""])
    for key, value in sorted(payload["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "No provider submission is represented. This is a validation-roadmap "
            "contract; passing it is not empirical evidence and does not establish "
            "that any listed target has been experimentally validated.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: ValidationStrategyFixtureResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the validation-strategy fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_validation_strategy_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_validation_strategy_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: ValidationStrategyFixtureResult, *, runtime_ms: float
) -> dict[str, Any]:
    return {
        "spec_keys": list(result.spec_keys),
        "validation_protocols": list(result.validation_protocols),
        "hardware_status": result.hardware_status,
        "domains": list(result.domains),
        "stages": list(result.stages),
        "domain_count": result.domain_count,
        "stage_count": result.stage_count,
        "stage_order_valid": result.stage_order_valid,
        "claim_boundary": result.claim_boundary,
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
