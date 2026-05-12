#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 3 state/layout DLA analysis
"""Generate preregistered analysis artefacts for Phase 3 state/layout DLA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scipy.stats import combine_pvalues, ttest_ind

REPO_ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-05-07"
DEFAULT_INPUT = (
    REPO_ROOT
    / "data"
    / "phase3_state_layout_dla"
    / "phase3_state_layout_ibm_marrakesh_2026-05-06T224531Z.json"
)
OUT_DIR = REPO_ROOT / "data" / "phase3_state_layout_dla"
DOC_PATH = REPO_ROOT / "docs" / f"phase3_state_layout_dla_manifest_{DATE}.md"

COMPARISONS = {
    "original_E0_minus_O0": ("E0", "O0"),
    "within_even_E0_minus_E1": ("E0", "E1"),
    "within_odd_O0_minus_O1": ("O0", "O1"),
    "excitation_inversion_E0_minus_O3": ("E0", "O3"),
}


@dataclass(frozen=True)
class StateDepthLayoutRow:
    layout_id: str
    physical_qubits: str
    depth: int
    state_label: str
    initial: str
    sector: str
    popcount: int
    n_reps: int
    mean_parity_leakage: float
    sem_parity_leakage: float
    mean_initial_state_retention: float
    sem_initial_state_retention: float


@dataclass(frozen=True)
class ComparisonRow:
    comparison: str
    layout_id: str
    depth: int
    left_label: str
    right_label: str
    left_mean_leakage: float
    right_mean_leakage: float
    difference: float
    relative_to_right: float
    welch_t: float
    welch_p: float
    n_left: int
    n_right: int


@dataclass(frozen=True)
class LayoutSummary:
    layout_id: str
    physical_qubits: str
    readout_error_mean: float | None
    two_qubit_error_mean: float | None
    mean_parity_leakage: float
    mean_initial_state_retention: float
    original_contrast_mean: float
    layout_spread_reference: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def _safe_relative(left: float, right: float) -> float:
    if abs(right) < 1e-12:
        return math.inf
    return (left - right) / right


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return _sha256(path)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("raw-count artefact must be a JSON object")
    if payload.get("schema") != "scpn_phase3_state_layout_dla_v1":
        raise ValueError(f"unexpected schema: {payload.get('schema')}")
    if payload.get("status") != "completed":
        raise ValueError(f"input is not a completed hardware artefact: {payload.get('status')}")
    if payload.get("job_ids") != ["ibm-run-aabcf620230b1438", "ibm-run-eea172711aa52b78"]:
        raise ValueError("job IDs do not match the committed state/layout run")
    if payload.get("n_circuits") != 495:
        raise ValueError(f"unexpected circuit count: {payload.get('n_circuits')}")
    return dict(payload)


def _bucket_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    buckets: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row["meta"]
        buckets[(str(meta["layout_id"]), int(meta["depth"]), str(meta["state_label"]))].append(row)
    return buckets


def _state_rows(main_rows: list[dict[str, Any]]) -> list[StateDepthLayoutRow]:
    buckets = _bucket_rows(main_rows)
    output: list[StateDepthLayoutRow] = []
    for key in sorted(buckets):
        rows = buckets[key]
        meta = rows[0]["meta"]
        leakages = [float(row["stats"]["parity_leakage"]) for row in rows]
        retentions = [float(row["stats"]["initial_state_retention"]) for row in rows]
        output.append(
            StateDepthLayoutRow(
                layout_id=str(meta["layout_id"]),
                physical_qubits=",".join(str(q) for q in meta["physical_qubits"]),
                depth=int(meta["depth"]),
                state_label=str(meta["state_label"]),
                initial=str(meta["initial"]),
                sector=str(meta["sector"]),
                popcount=int(meta["popcount"]),
                n_reps=len(rows),
                mean_parity_leakage=float(mean(leakages)),
                sem_parity_leakage=float(_sem(leakages)),
                mean_initial_state_retention=float(mean(retentions)),
                sem_initial_state_retention=float(_sem(retentions)),
            )
        )
    return output


def _comparison_rows(main_rows: list[dict[str, Any]]) -> list[ComparisonRow]:
    buckets = _bucket_rows(main_rows)
    layouts = sorted({key[0] for key in buckets})
    depths = sorted({key[1] for key in buckets})
    output: list[ComparisonRow] = []
    for layout in layouts:
        for depth in depths:
            for name, (left, right) in COMPARISONS.items():
                left_values = [
                    float(row["stats"]["parity_leakage"]) for row in buckets[(layout, depth, left)]
                ]
                right_values = [
                    float(row["stats"]["parity_leakage"])
                    for row in buckets[(layout, depth, right)]
                ]
                left_mean = float(mean(left_values))
                right_mean = float(mean(right_values))
                welch = ttest_ind(left_values, right_values, equal_var=False)
                output.append(
                    ComparisonRow(
                        comparison=name,
                        layout_id=layout,
                        depth=depth,
                        left_label=left,
                        right_label=right,
                        left_mean_leakage=left_mean,
                        right_mean_leakage=right_mean,
                        difference=float(left_mean - right_mean),
                        relative_to_right=float(_safe_relative(left_mean, right_mean)),
                        welch_t=float(welch.statistic),
                        welch_p=float(welch.pvalue),
                        n_left=len(left_values),
                        n_right=len(right_values),
                    )
                )
    return output


def _layout_summaries(
    payload: dict[str, Any],
    state_rows: list[StateDepthLayoutRow],
    comparison_rows: list[ComparisonRow],
) -> list[LayoutSummary]:
    layout_meta = {layout["layout_id"]: layout for layout in payload["layouts"]}
    original_by_layout: dict[str, list[float]] = {
        row.layout_id: [] for row in comparison_rows if row.comparison == "original_E0_minus_O0"
    }
    for row in comparison_rows:
        if row.comparison == "original_E0_minus_O0":
            original_by_layout[row.layout_id].append(row.difference)
    all_original = [value for values in original_by_layout.values() for value in values]
    global_spread = max(all_original) - min(all_original) if all_original else 0.0
    output: list[LayoutSummary] = []
    for layout_id in sorted({row.layout_id for row in state_rows}):
        rows = [row for row in state_rows if row.layout_id == layout_id]
        meta = layout_meta[layout_id]
        output.append(
            LayoutSummary(
                layout_id=layout_id,
                physical_qubits=",".join(str(q) for q in meta["physical_qubits"]),
                readout_error_mean=meta.get("readout_error_mean"),
                two_qubit_error_mean=meta.get("two_qubit_error_mean"),
                mean_parity_leakage=float(mean(row.mean_parity_leakage for row in rows)),
                mean_initial_state_retention=float(
                    mean(row.mean_initial_state_retention for row in rows)
                ),
                original_contrast_mean=float(mean(original_by_layout[layout_id])),
                layout_spread_reference=float(global_spread),
            )
        )
    return output


def _readout_rows(readout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in readout_rows:
        meta = row["meta"]
        rows.append(
            {
                "layout_id": meta["layout_id"],
                "physical_qubits": ",".join(str(q) for q in meta["physical_qubits"]),
                "state_label": meta["state_label"],
                "initial": meta["initial"],
                "total_shots": row["stats"]["total_shots"],
                "initial_state_retention": row["stats"]["initial_state_retention"],
                "parity_leakage": row["stats"]["parity_leakage"],
            }
        )
    return sorted(rows, key=lambda item: (item["layout_id"], item["state_label"]))


def _fisher_by_comparison(comparison_rows: list[ComparisonRow]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name in sorted({row.comparison for row in comparison_rows}):
        rows = [row for row in comparison_rows if row.comparison == name]
        chi2, p_value = combine_pvalues([row.welch_p for row in rows], method="fisher")
        output[name] = {
            "chi2": float(chi2),
            "p": float(p_value),
            "n_tests": len(rows),
            "n_tests_significant_at_0_05": sum(row.welch_p < 0.05 for row in rows),
            "mean_difference": float(mean(row.difference for row in rows)),
            "min_difference": float(min(row.difference for row in rows)),
            "max_difference": float(max(row.difference for row in rows)),
            "signs": {
                "positive": sum(row.difference > 0 for row in rows),
                "negative": sum(row.difference < 0 for row in rows),
                "zero": sum(row.difference == 0 for row in rows),
            },
        }
    return output


def build_analysis(payload: dict[str, Any], *, input_sha256: str) -> dict[str, Any]:
    circuits = payload["circuits"]
    main_rows = [row for row in circuits if row["meta"]["block"] == "main"]
    readout = [row for row in circuits if row["meta"]["block"] == "readout"]
    if len(main_rows) != 480 or len(readout) != 15:
        raise ValueError(f"unexpected split: main={len(main_rows)}, readout={len(readout)}")
    state_rows = _state_rows(main_rows)
    comparison_rows = _comparison_rows(main_rows)
    layout_summaries = _layout_summaries(payload, state_rows, comparison_rows)
    readout_summary = _readout_rows(readout)
    fisher = _fisher_by_comparison(comparison_rows)
    max_layout_spread = max(row.layout_spread_reference for row in layout_summaries)
    return {
        "schema": "scpn_phase3_state_layout_dla_analysis_v1",
        "date": DATE,
        "source_raw_counts": str(DEFAULT_INPUT.relative_to(REPO_ROOT)),
        "source_sha256": input_sha256,
        "backend": payload["backend"],
        "job_ids": payload["job_ids"],
        "n_circuits": payload["n_circuits"],
        "readiness": payload["readiness"],
        "state_depth_layout_rows": [asdict(row) for row in state_rows],
        "comparison_rows": [asdict(row) for row in comparison_rows],
        "layout_summaries": [asdict(row) for row in layout_summaries],
        "readout_rows": readout_summary,
        "fisher_by_comparison": fisher,
        "claim_boundary": {
            "supported": [
                "mechanism-separation evidence for this backend and calibration window",
                "state-level, depth-level, and layout-level parity-leakage summaries",
                "whether layout/state spread is comparable to the original contrast",
            ],
            "blocked": [
                "DLA-parity-only causality",
                "backend-universal protection",
                "monotone scaling",
                "quantum advantage",
                "full 16-state confusion-matrix mitigation from the five-state readout block",
            ],
        },
        "decision_flags": {
            "layout_spread_exceeds_mean_original_contrast": bool(
                max_layout_spread > abs(fisher["original_E0_minus_O0"]["mean_difference"])
            ),
            "original_contrast_mixed_sign": bool(
                fisher["original_E0_minus_O0"]["signs"]["positive"] > 0
                and fisher["original_E0_minus_O0"]["signs"]["negative"] > 0
            ),
            "within_sector_controls_significant": bool(
                fisher["within_even_E0_minus_E1"]["n_tests_significant_at_0_05"] > 0
                or fisher["within_odd_O0_minus_O1"]["n_tests_significant_at_0_05"] > 0
            ),
        },
    }


def _markdown(summary: dict[str, Any], artefact_hashes: dict[str, str]) -> str:
    lines = [
        "# Phase 3 State/Layout DLA Manifest",
        "",
        f"Date: {DATE}",
        f"Backend: `{summary['backend']}`",
        f"Jobs: `{summary['job_ids'][0]}`, `{summary['job_ids'][1]}`",
        f"Raw-count SHA256: `{summary['source_sha256']}`",
        "",
        "## Scope",
        "",
        "This analysis uses the committed Phase 3 state/layout raw-count artefact only. It separates state, excitation-count, and physical-layout effects for the `n=4` DLA parity programme.",
        "",
        "## Readiness",
        f"- Circuits: `{summary['n_circuits']}`",
        f"- Max transpiled depth: `{summary['readiness']['depth_summary']['max']}`",
        f"- Max total gates: `{summary['readiness']['total_gate_summary']['max']}`",
        "",
        "## Comparison Summary",
    ]
    for name, stats in summary["fisher_by_comparison"].items():
        lines.append(
            f"- `{name}`: mean difference `{stats['mean_difference']:.6f}`, "
            f"signs +/−/0 = `{stats['signs']['positive']}/{stats['signs']['negative']}/{stats['signs']['zero']}`, "
            f"Fisher p `{stats['p']:.6e}`"
        )
    lines.extend(
        [
            "",
            "## Decision Flags",
        ]
    )
    lines.extend(f"- `{key}`: `{value}`" for key, value in summary["decision_flags"].items())
    lines.extend(["", "## Artefacts"])
    lines.extend(f"- `{path}` SHA256 `{digest}`" for path, digest in artefact_hashes.items())
    lines.extend(["", "## Claim Boundary", "", "Supported:"])
    lines.extend(f"- {item}" for item in summary["claim_boundary"]["supported"])
    lines.extend(["", "Blocked:"])
    lines.extend(f"- {item}" for item in summary["claim_boundary"]["blocked"])
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "python scripts/analyse_phase3_state_layout_dla.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(summary: dict[str, Any]) -> dict[str, str]:
    summary_path = OUT_DIR / f"phase3_state_layout_summary_{DATE}.json"
    rows_path = OUT_DIR / f"phase3_state_layout_row_metrics_{DATE}.csv"
    layout_path = OUT_DIR / f"phase3_state_layout_layout_metrics_{DATE}.csv"
    readout_path = OUT_DIR / f"phase3_state_layout_readout_metrics_{DATE}.csv"
    hashes = {
        str(summary_path.relative_to(REPO_ROOT)): _write_json(summary_path, summary),
        str(rows_path.relative_to(REPO_ROOT)): _write_csv(rows_path, summary["comparison_rows"]),
        str(layout_path.relative_to(REPO_ROOT)): _write_csv(
            layout_path, summary["layout_summaries"]
        ),
        str(readout_path.relative_to(REPO_ROOT)): _write_csv(
            readout_path, summary["readout_rows"]
        ),
    }
    hashes[str(DOC_PATH.relative_to(REPO_ROOT))] = _write_text(
        DOC_PATH, _markdown(summary, hashes)
    )
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = _load_payload(args.input)
    summary = build_analysis(payload, input_sha256=_sha256(args.input))
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    hashes = write_outputs(summary)
    print("Phase 3 state/layout DLA analysis")
    print(f"  backend: {summary['backend']}")
    print(f"  jobs:    {', '.join(summary['job_ids'])}")
    print(f"  rows:    {len(summary['comparison_rows'])} comparison rows")
    for path, digest in hashes.items():
        print(f"  wrote:   {path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
