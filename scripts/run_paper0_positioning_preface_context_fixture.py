#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Positioning Preface context runner
"""Run the Paper 0 Positioning Preface context validation fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from scpn_quantum_control.paper0.positioning_preface_context_validation import (
    validate_positioning_preface_context_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def render_report(payload: dict[str, Any]) -> str:
    """Render a compact Markdown report for the Positioning Preface fixture."""
    return "\n".join(
        [
            "# Paper 0 Positioning Preface Context Fixture",
            "",
            f"- Source span: {payload['source_ledger_span'][0]} - "
            f"{payload['source_ledger_span'][1]}",
            f"- Hardware status: {payload['hardware_status']}",
            f"- Claim boundary: {payload['claim_boundary']}",
            f"- Blank separators: {payload['blank_separator_count']}",
            f"- Image markers: {payload['image_marker_count']}",
            f"- Part I boundary: {payload['part_i_boundary']}",
            f"- Context classification: {payload['context_classification']}",
            "",
            "## Null Controls",
            *(f"- `{key}`: {value}" for key, value in sorted(payload["null_controls"].items())),
            "",
        ]
    )


def write_outputs(
    *,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Run the fixture and write JSON plus Markdown outputs."""
    result = validate_positioning_preface_context_fixture()
    payload = cast(dict[str, Any], _json_ready(result.as_dict()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    """Run the Positioning Preface context fixture and write artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
        / "paper0_positioning_preface_context_fixture_result_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
        / "paper0_positioning_preface_context_fixture_report_2026-05-13.md",
    )
    args = parser.parse_args()
    payload = write_outputs(output_path=args.output, report_path=args.report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
