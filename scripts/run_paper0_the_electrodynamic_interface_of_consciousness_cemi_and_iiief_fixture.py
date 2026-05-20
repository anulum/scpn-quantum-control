#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Electrodynamic Interface of Consciousness (CEMI and IIIEF) fixture runner
"""Run the Paper 0 The Electrodynamic Interface of Consciousness (CEMI and IIIEF) source-accounting fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.the_electrodynamic_interface_of_consciousness_cemi_and_iiief_validation import (
    validate_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIR
    / "paper0_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_fixture_result_2026-05-17.json"
)
DEFAULT_REPORT_PATH = (
    DEFAULT_OUTPUT_DIR
    / "paper0_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_fixture_report_2026-05-17.md"
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(payload: dict[str, Any]) -> str:
    """Render a compact Markdown report for the fixture result."""
    lines = [
        "# Paper 0 "
        + "The Electrodynamic Interface of Consciousness (CEMI and IIIEF)"
        + " Fixture",
        "",
        f"- Source span: {payload['source_ledger_span'][0]} - {payload['source_ledger_span'][1]}",
        f"- Hardware status: {payload['hardware_status']}",
        f"- Claim boundary: {payload['claim_boundary']}",
        f"- Source records: {payload['source_record_count']}",
        f"- Components: {payload['component_count']}",
        f"- Next source boundary: {payload['next_source_boundary']}",
        f"- Protocol state: {payload['problem_metadata']['protocol_state']}",
        "",
        "## Components",
    ]
    for key, role in payload["components"].items():
        lines.append(f"- `{key}`: `{role}`")
    lines.extend(["", "## Null Controls"])
    for key, value in payload["null_controls"].items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def write_outputs(
    *, output_path: Path = DEFAULT_OUTPUT_PATH, report_path: Path = DEFAULT_REPORT_PATH
) -> dict[str, Path]:
    """Write the fixture JSON and report."""
    result = validate_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_fixture()
    payload = _json_ready(result.as_dict())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": output_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    outputs = write_outputs(output_path=args.output, report_path=args.report)
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
