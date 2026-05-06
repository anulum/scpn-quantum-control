#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Layer-selective layout readiness audit
"""Generate a no-QPU layer-selective qubit-assignment readiness audit.

The preregistered layer-selective protocol requires default, SABRE, and
layer-selective comparator layouts before hardware execution can be promoted.
This script consumes committed Phase 3 state/layout artefacts and records what
can be concluded from them without fabricating missing comparator rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "data"
    / "phase3_state_layout_dla"
    / "phase3_state_layout_ibm_marrakesh_2026-05-06T224531Z.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase3_layer_layout"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
TODAY = date(2026, 5, 7).isoformat()


@dataclass(frozen=True)
class LayoutResource:
    """Compiled resource summary for one saved physical layout."""

    layout_id: str
    physical_qubits: tuple[int, ...]
    n_rows: int
    max_depth: int
    mean_depth: float
    max_total_gates: int
    mean_total_gates: float
    max_ecr_gates: int
    mean_ecr_gates: float
    readout_error_mean: float | None
    two_qubit_error_mean: float | None
    high_priority_cost: float

    def to_row(self) -> dict[str, object]:
        """Return a CSV/JSON-compatible row."""
        return {
            "layout_id": self.layout_id,
            "physical_qubits": " ".join(str(item) for item in self.physical_qubits),
            "n_rows": self.n_rows,
            "max_depth": self.max_depth,
            "mean_depth": self.mean_depth,
            "max_total_gates": self.max_total_gates,
            "mean_total_gates": self.mean_total_gates,
            "max_ecr_gates": self.max_ecr_gates,
            "mean_ecr_gates": self.mean_ecr_gates,
            "readout_error_mean": self.readout_error_mean,
            "two_qubit_error_mean": self.two_qubit_error_mean,
            "high_priority_cost": self.high_priority_cost,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_edges(n_qubits: int = 4) -> tuple[tuple[int, int, float], ...]:
    """Return preregistered logical Kuramoto-XY edge priorities."""
    edges: list[tuple[int, int, float]] = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            weight = 0.45 * math.exp(-0.3 * abs(i - j))
            edges.append((i, j, weight))
    return tuple(sorted(edges, key=lambda item: (-item[2], item[0], item[1])))


def _layout_cost(layout: Mapping[str, Any]) -> float:
    """Approximate preregistered physical-pair cost for saved layout metadata."""
    readout = layout.get("readout_error_mean")
    twoq = layout.get("two_qubit_error_mean")
    readout_cost = float(readout) if readout is not None else 0.02
    twoq_cost = float(twoq) if twoq is not None else 0.02
    path_cost = 1.0
    t1_cost = 0.0
    return 0.55 * twoq_cost + 0.20 * readout_cost + 0.15 * t1_cost + 0.10 * path_cost


def _mean_int(values: Sequence[int]) -> float:
    return float(mean(values)) if values else 0.0


def _rows_for_block(payload: Mapping[str, Any], block: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for item in payload.get("circuits", []):
        meta = item.get("meta", {})
        if meta.get("block") == block:
            rows.append(item)
    return rows


def summarise_saved_layouts(payload: Mapping[str, Any]) -> tuple[LayoutResource, ...]:
    """Summarise saved transpiled rows by physical layout."""
    layouts_by_id = {str(item["layout_id"]): item for item in payload.get("layouts", [])}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in _rows_for_block(payload, "main"):
        grouped[str(item.get("meta", {}).get("layout_id"))].append(item)

    summaries: list[LayoutResource] = []
    for layout_id, rows in sorted(grouped.items()):
        layout = layouts_by_id.get(layout_id, {})
        depths = [int(item.get("metadata", {}).get("depth", 0)) for item in rows]
        total_gates = [int(item.get("metadata", {}).get("total_gates", 0)) for item in rows]
        ecr_gates = [int(item.get("metadata", {}).get("ecr_gates", 0)) for item in rows]
        physical = tuple(int(q) for q in layout.get("physical_qubits", ()))
        summaries.append(
            LayoutResource(
                layout_id=layout_id,
                physical_qubits=physical,
                n_rows=len(rows),
                max_depth=max(depths),
                mean_depth=_mean_int(depths),
                max_total_gates=max(total_gates),
                mean_total_gates=_mean_int(total_gates),
                max_ecr_gates=max(ecr_gates),
                mean_ecr_gates=_mean_int(ecr_gates),
                readout_error_mean=layout.get("readout_error_mean"),
                two_qubit_error_mean=layout.get("two_qubit_error_mean"),
                high_priority_cost=_layout_cost(layout),
            )
        )
    return tuple(summaries)


def build_readiness(payload: Mapping[str, Any], *, input_path: Path) -> dict[str, Any]:
    """Build the preregistered no-QPU readiness payload."""
    layout_resources = summarise_saved_layouts(payload)
    comparator_methods_present = ["saved_connected_low_readout_windows"]
    required_methods = ["default", "sabre", "layer_selective"]
    missing_methods = [item for item in required_methods if item not in comparator_methods_present]
    depths = sorted(
        {item.get("meta", {}).get("depth") for item in _rows_for_block(payload, "main")}
    )
    states = sorted(
        {item.get("meta", {}).get("initial") for item in _rows_for_block(payload, "main")}
    )
    seeds_present: list[int] = []
    ready = not missing_methods and bool(layout_resources)
    return {
        "schema": "scpn_phase3_layer_selective_readiness_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_artifact": str(input_path.relative_to(REPO_ROOT)),
        "source_sha256": _sha256(input_path),
        "backend": payload.get("backend"),
        "hardware_submission": False,
        "qpu_minutes_spent": 0.0,
        "required_methods": required_methods,
        "comparator_methods_present": comparator_methods_present,
        "missing_comparator_methods": missing_methods,
        "ready_for_hardware_comparison": ready,
        "readiness_decision": "blocked_missing_comparators" if not ready else "ready",
        "depths": depths,
        "states": states,
        "transpiler_seeds_present": seeds_present,
        "canonical_logical_edges": [
            {"source": i, "target": j, "priority": weight} for i, j, weight in _canonical_edges()
        ],
        "layout_resource_rows": [item.to_row() for item in layout_resources],
        "claim_boundary": {
            "supported": [
                "saved connected-layout resource summary",
                "missing-comparator blocker for layer-selective hardware promotion",
            ],
            "blocked": [
                "layer-selective improvement claim",
                "hardware leakage reduction claim",
                "default-versus-SABRE-versus-layer-selective comparison",
                "QPU submission authorisation",
            ],
        },
        "next_required_step": (
            "Generate default, SABRE, and true layer-selective transpilation rows from a "
            "fresh backend snapshot before considering the 152-circuit hardware follow-up."
        ),
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    """Return a stable path for manifests inside or outside the repository."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _manifest_text(summary: Mapping[str, Any], *, json_path: Path, csv_path: Path) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- SCPN Quantum Control — Layer-Selective Readiness Manifest -->",
            "",
            "# Phase 3 Layer-Selective Readiness Manifest",
            "",
            f"Date: {TODAY}",
            "",
            "## Decision",
            "",
            f"- Ready for hardware comparison: `{summary['ready_for_hardware_comparison']}`",
            f"- Readiness decision: `{summary['readiness_decision']}`",
            f"- Backend: `{summary['backend']}`",
            "- Hardware submission: `False`",
            "- QPU minutes spent: `0.0`",
            "",
            "## Artefacts",
            "",
            f"- JSON summary: `{_display_path(json_path)}`",
            f"- Resource rows: `{_display_path(csv_path)}`",
            f"- Source artefact: `{summary['source_artifact']}`",
            f"- Source SHA256: `{summary['source_sha256']}`",
            "",
            "## Blocker",
            "",
            str(summary["next_required_step"]),
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-linux/bin/python scripts/analyse_layer_selective_readiness.py",
            "```",
            "",
        ]
    )


def write_outputs(
    summary: Mapping[str, Any], *, output_dir: Path, docs_dir: Path
) -> tuple[Path, Path, Path]:
    """Write JSON, CSV, and manifest outputs."""
    backend = str(summary.get("backend", "unknown")).replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"layer_selective_readiness_{backend}_{TODAY}.json"
    csv_path = output_dir / f"layer_selective_transpile_rows_{TODAY}.csv"
    md_path = docs_dir / f"phase3_layer_layout_readiness_{TODAY}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, summary["layout_resource_rows"])
    md_path.write_text(
        _manifest_text(summary, json_path=json_path, csv_path=csv_path), encoding="utf-8"
    )
    return json_path, csv_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = build_readiness(payload, input_path=args.input)
    json_path, csv_path, md_path = write_outputs(
        summary,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"readiness_decision={summary['readiness_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
