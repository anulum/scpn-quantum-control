#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — export synchronisation benchmark registry script
# scpn-quantum-control -- synchronisation benchmark registry exporter
"""Export the canonical synchronisation benchmark registry artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scpn_quantum_control.benchmark_harness.synchronisation import (
    synchronisation_benchmark_registry_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "synchronisation_benchmarks"
DOC_PATH = REPO_ROOT / "docs" / "synchronisation_benchmark_suite.md"


def write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write deterministic JSON and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def markdown(payload: dict[str, Any]) -> str:
    """Render a compact public registry document."""

    lines = [
        "# Standardised Synchronisation Benchmark Suite",
        "",
        "This suite defines canonical coupled-oscillator benchmark instances and a stable result schema. It is a contract for future backends, not a new hardware claim.",
        "",
        "## Result schema",
        "",
        "Required result fields:",
        "",
    ]
    for field in payload["result_schema"]["required_fields"]:
        lines.append(f"- `{field}`")
    lines.extend(
        [
            "",
            "Observable row fields:",
            "",
        ]
    )
    for field in payload["result_schema"]["observable_row_fields"]:
        lines.append(f"- `{field}`")
    lines.extend(
        [
            "",
            "## Canonical instances",
            "",
            "| Benchmark | Family | N | Evidence class | Required backends | Claim boundary |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for row in payload["instances"]:
        lines.append(
            "| `{benchmark_id}` | {family} | {n_oscillators} | {evidence_class} | {backends} | {boundary} |".format(
                benchmark_id=row["benchmark_id"],
                family=row["family"],
                n_oscillators=row["n_oscillators"],
                evidence_class=row["evidence_class"],
                backends=", ".join(row["required_backends"]),
                boundary=row["claim_boundary"],
            )
        )
    lines.extend(
        [
            "",
            "## Hardware rule",
            "",
            str(payload["result_schema"]["hardware_rule"]),
            "",
            "## Claim boundary",
            "",
            str(payload["claim_boundary"]),
        ]
    )
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> str:
    """Write text and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    """Export synchronisation benchmark registry artefacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    parser.add_argument("--exclude-planned", action="store_true")
    args = parser.parse_args()

    payload = synchronisation_benchmark_registry_payload(include_planned=not args.exclude_planned)
    json_path = args.out_dir / "synchronisation_benchmark_registry.json"
    json_digest = write_json(json_path, payload)
    doc_digest = write_text(args.doc_path, markdown(payload))
    print(f"wrote {json_path.relative_to(REPO_ROOT)} sha256={json_digest}")
    print(f"wrote {args.doc_path.relative_to(REPO_ROOT)} sha256={doc_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
