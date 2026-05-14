#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Gaian safety fixture runner
"""Run Paper 0 Gaian ethic and societal-safety fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.gaian_safety_validation import (
    GaianSafetyFixtureResult,
    validate_gaian_safety_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"


def run_default_fixture() -> GaianSafetyFixtureResult:
    """Run the default Paper 0 Gaian safety fixture."""
    return validate_gaian_safety_fixture()


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the Gaian safety fixture."""
    governance = payload["governance"]
    lines = [
        "# Paper 0 Gaian Safety Fixture",
        "",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Gaian stability delta: `{payload['gaian_stability_delta']}`",
        f"- Risk score: `{governance['risk_score']}`",
        f"- Safeguard score: `{governance['safeguard_score']}`",
        f"- Runtime: `{payload['runtime_ms']}` ms",
        "",
        "## Phase Categories",
        "",
    ]
    for category in payload["phase_categories"]:
        lines.append(f"- `{category}`")
    lines.extend(["", "## Null Controls", ""])
    for key, value in sorted(governance["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "No provider submission is represented. This is a source-bounded "
            "Gaian safety simulator contract; passing it is not empirical evidence "
            "and does not establish that any ecological, societal, or governance "
            "claim is validated.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: GaianSafetyFixtureResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the Gaian safety fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_gaian_safety_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_gaian_safety_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(result: GaianSafetyFixtureResult, *, runtime_ms: float) -> dict[str, Any]:
    governance = result.governance
    return {
        "spec_keys": list(result.spec_keys),
        "hardware_status": result.hardware_status,
        "protected_stability": result.protected_stability,
        "degraded_stability": result.degraded_stability,
        "gaian_stability_delta": result.gaian_stability_delta,
        "phase_categories": list(result.phase_categories),
        "claim_boundary": result.claim_boundary,
        "problem_metadata": dict(result.problem_metadata),
        "governance": {
            "spec_key": governance.spec_key,
            "validation_protocol": governance.validation_protocol,
            "hardware_status": governance.hardware_status,
            "source_ledger_ids": list(governance.source_ledger_ids),
            "risk_score": governance.risk_score,
            "safeguard_score": governance.safeguard_score,
            "claim_boundary": governance.claim_boundary,
            "null_controls": dict(governance.null_controls),
            "problem_metadata": dict(governance.problem_metadata),
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
