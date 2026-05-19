# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — EEG PLV Cohort Comparison
"""Compare two derived EEG PLV measured-coupling cohort artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPEN = REPO_ROOT / "data" / "knm_physical_validation" / "measured_couplings.json"
DEFAULT_CLOSED = (
    REPO_ROOT / "data" / "knm_physical_validation" / "measured_couplings_baseline_closed.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "knm_physical_validation" / "baseline_open_closed_comparison.json"
)
SCHEMA_VERSION = "scpn-quantum-control.eeg-plv-cohort-comparison.v1"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_payload(path: Path) -> dict[str, Any]:
    """Load and validate one measured EEG PLV coupling payload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "scpn-quantum-control.measured-couplings.v1":
        raise ValueError(f"{path} is not a measured-couplings v1 artifact")
    if payload.get("unit") != "phase_locking_value":
        raise ValueError(f"{path} does not use phase_locking_value units")
    if "couplings" not in payload or "source_dataset" not in payload:
        raise ValueError(f"{path} is missing required measured-coupling fields")
    return payload


def coupling_map(payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """Index payload couplings by one-based edge endpoint pairs."""
    mapped = {}
    for edge in payload["couplings"]:
        key = (int(edge["i"]), int(edge["j"]))
        if key in mapped:
            raise ValueError(f"Duplicate coupling edge {key}")
        mapped[key] = edge
    return mapped


def _pearson_or_none(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def compare_payloads(
    *,
    open_payload: dict[str, Any],
    closed_payload: dict[str, Any],
    open_path: Path,
    closed_path: Path,
    command: list[str],
) -> dict[str, Any]:
    """Compare baseline-open and baseline-closed PLV payloads edge by edge."""
    open_edges = coupling_map(open_payload)
    closed_edges = coupling_map(closed_payload)
    if set(open_edges) != set(closed_edges):
        missing_closed = sorted(set(open_edges) - set(closed_edges))
        missing_open = sorted(set(closed_edges) - set(open_edges))
        raise ValueError(
            "Cohort artifacts must contain the same edge set: "
            f"missing_closed={missing_closed}, missing_open={missing_open}"
        )

    edge_rows = []
    for key in sorted(open_edges):
        open_edge = open_edges[key]
        closed_edge = closed_edges[key]
        open_value = float(open_edge["value"])
        closed_value = float(closed_edge["value"])
        delta = closed_value - open_value
        edge_rows.append(
            {
                "i": key[0],
                "j": key[1],
                "open_value": open_value,
                "closed_value": closed_value,
                "delta_closed_minus_open": delta,
                "absolute_delta": abs(delta),
                "open_uncertainty": float(open_edge["uncertainty"]),
                "closed_uncertainty": float(closed_edge["uncertainty"]),
                "open_q25": float(open_edge.get("q25", open_value)),
                "open_q75": float(open_edge.get("q75", open_value)),
                "closed_q25": float(closed_edge.get("q25", closed_value)),
                "closed_q75": float(closed_edge.get("q75", closed_value)),
            }
        )

    open_values = np.asarray([row["open_value"] for row in edge_rows], dtype=np.float64)
    closed_values = np.asarray([row["closed_value"] for row in edge_rows], dtype=np.float64)
    deltas = closed_values - open_values
    max_row = max(edge_rows, key=lambda row: float(row["absolute_delta"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "comparison": "PhysioNet EEGMMIDB baseline eyes closed minus baseline eyes open",
        "unit": "phase_locking_value",
        "normalisation": open_payload["normalisation"],
        "normalisation_locked": bool(
            open_payload.get("normalisation_locked")
            and closed_payload.get("normalisation_locked")
            and open_payload.get("normalisation") == closed_payload.get("normalisation")
        ),
        "claim_boundary": (
            "Descriptive condition comparison for the same 8-channel alpha-band PLV "
            "pipeline; this is not a physical-unit K_nm magnitude validation."
        ),
        "cohorts": {
            "baseline_eyes_open": {
                "artifact": str(open_path),
                "condition": open_payload["source_dataset"].get("condition"),
                "n_records": int(open_payload["source_dataset"].get("n_records", 0)),
            },
            "baseline_eyes_closed": {
                "artifact": str(closed_path),
                "condition": closed_payload["source_dataset"].get("condition"),
                "n_records": int(closed_payload["source_dataset"].get("n_records", 0)),
            },
        },
        "summary": {
            "edge_count": len(edge_rows),
            "open_mean": float(np.mean(open_values)),
            "closed_mean": float(np.mean(closed_values)),
            "mean_delta_closed_minus_open": float(np.mean(deltas)),
            "median_delta_closed_minus_open": float(np.median(deltas)),
            "mean_absolute_delta": float(np.mean(np.abs(deltas))),
            "max_absolute_delta_edge": {
                "i": int(max_row["i"]),
                "j": int(max_row["j"]),
                "absolute_delta": float(max_row["absolute_delta"]),
                "delta_closed_minus_open": float(max_row["delta_closed_minus_open"]),
            },
            "pearson_r_across_edge_medians": _pearson_or_none(open_values, closed_values),
        },
        "edges": edge_rows,
        "provenance": {
            "repo_root": str(REPO_ROOT),
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "command": command,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the EEG PLV cohort comparison CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", type=Path, default=DEFAULT_OPEN, dest="open_path")
    parser.add_argument("--closed", type=Path, default=DEFAULT_CLOSED, dest="closed_path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the baseline-open versus baseline-closed PLV comparison artefact."""
    args = parse_args(argv)
    command = [Path(sys.executable).name, *sys.argv]
    payload = compare_payloads(
        open_payload=load_payload(args.open_path),
        closed_payload=load_payload(args.closed_path),
        open_path=args.open_path,
        closed_path=args.closed_path,
        command=command,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote EEG PLV cohort comparison: {args.output}")
    print(f"Edges: {payload['summary']['edge_count']}")
    print(f"Mean delta closed-open: {payload['summary']['mean_delta_closed_minus_open']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
