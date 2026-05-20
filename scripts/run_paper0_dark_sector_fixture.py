#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 dark-sector fixture runner
"""Run Paper 0 dark-sector simulator fixtures."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scpn_quantum_control.paper0.dark_sector_validation import (
    DarkSectorFixtureResult,
    validate_dark_sector_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)


def run_default_fixture() -> DarkSectorFixtureResult:
    """Run default simulator-only dark-sector fixture."""
    return validate_dark_sector_fixture()


def _json_ready_payload(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {str(k): _json_ready_payload(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_ready_payload(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_json_ready_payload(item) for item in value]
    if isinstance(value, list):
        return [_json_ready_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_ready_payload(v) for k, v in value.items()}
    return value


def render_report(result: DarkSectorFixtureResult, runtime_ms: float) -> str:
    """Render a Markdown report for the dark-sector fixture."""
    return "\n".join(
        [
            "# Paper 0 Dark Sector Fixture",
            "",
            f"- Hardware status: `{result.hardware_status}`",
            f"- Source span: `{', '.join(result.source_ledger_span)}`",
            f"- MMC score: `{result.mmc_score:.6f}`",
            f"- Dark-energy context score: `{result.dark_energy_score:.6f}`",
            f"- Psi-DM candidate label: `{result.psi_dm_candidate}`",
            f"- Interaction score: `{result.interaction.interaction_score:.6f}`",
            f"- Reservoir score: `{result.reservoir_score:.6f}`",
            f"- Runtime ms: `{runtime_ms:.6f}`",
            f"- Claim boundary: `{result.claim_boundary}`",
            "",
            "Passing this fixture is not empirical evidence and does not validate dark energy, "
            "dark matter, psi-DM, halo coherence, L8, or L12 claims.",
            "",
        ]
    )


def write_outputs(
    result: DarkSectorFixtureResult,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown fixture outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload = _json_ready_payload(result)
    payload["runtime_ms"] = (time.perf_counter() - started) * 1000.0
    json_path = output_dir / f"paper0_dark_sector_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_dark_sector_fixture_report_{date_tag}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(result, float(payload["runtime_ms"])), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    result = run_default_fixture()
    paths = write_outputs(result, output_dir=args.output_dir, date_tag=args.date_tag)
    payload = _json_ready_payload(result)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
