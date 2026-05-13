#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Paper 0 topology schema artefact builder
"""Build the Paper 0 topology source-boundary artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.topology_schema import (
    build_paper0_topology_schema,
    schema_to_s19_source_boundary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"


def write_topology_schema_outputs(
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write the Paper 0 topology source-boundary JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary = schema_to_s19_source_boundary(build_paper0_topology_schema())
    json_path = output_dir / f"paper0_topology_source_boundary_{date_tag}.json"
    report_path = output_dir / f"paper0_topology_source_boundary_{date_tag}.md"
    json_path.write_text(json.dumps(boundary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_topology_schema_report(boundary), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_topology_schema_report(boundary: dict[str, Any]) -> str:
    """Render a human-readable topology source-boundary report."""
    lines = [
        "# Paper 0 Topology Source Boundary",
        "",
        f"- Schema key: `{boundary['schema_key']}`",
        f"- Layer count: `{boundary['layer_count']}`",
        f"- Meta-layer: `L{boundary['meta_layer']}`",
        f"- Provider ready: `{boundary['provider_ready']}`",
        f"- Hardware status: `{boundary['hardware_status']}`",
        f"- Source equations: `{len(boundary['source_equation_ids'])}`",
        f"- Source ledger records: `{len(boundary['source_ledger_ids'])}`",
        "",
        "## Policy",
        "",
        "No numeric coupling matrix is exported by this boundary. Paper 0 currently "
        "supports source-anchored layer, coupling-channel, field-port, and adaptive "
        "parameter provenance only. Synthetic chain, ring, and complete graphs are "
        "validation controls, not Paper 0 topology claims.",
        "",
        "## Coupling Channels",
        "",
    ]
    channels = sorted({edge["channel"] for edge in boundary["inter_layer_edges"]})
    for channel in channels:
        count = sum(1 for edge in boundary["inter_layer_edges"] if edge["channel"] == channel)
        lines.append(f"- `{channel}`: `{count}` directed edges")
    lines.extend(["", "## Synthetic Controls", ""])
    for key in boundary["synthetic_controls"]:
        lines.append(f"- `{key}`")
    lines.extend(["", "## S19 Boundary", ""])
    lines.append(
        "S19 scans may consume this file as a provenance boundary only. They must "
        "still supply an explicitly labelled experimental or synthetic numeric "
        "topology before any simulation, provider transpilation, or hardware run."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    paths = write_topology_schema_outputs(output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    summary = json.loads(paths["json"].read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "schema_key": summary["schema_key"],
                "provider_ready": summary["provider_ready"],
                "hardware_status": summary["hardware_status"],
                "layer_count": summary["layer_count"],
                "synthetic_controls": summary["synthetic_controls"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
