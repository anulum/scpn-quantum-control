#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — train s3 design surrogate script
# scpn-quantum-control -- S3 design surrogate rehearsal
"""Train a deterministic no-QPU surrogate on S3 design rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.benchmarks.s3_design_protocol import (
    grid_s3_design_protocol,
    score_s3_candidates,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s3_pulse_ansatz_design"
DATE = "2026-05-06"
RIDGE_ALPHA = 1.0e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="3,4,5", help="Comma-separated qubit sizes.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_for_sizes(sizes: list[int]) -> list[dict[str, Any]]:
    protocol = grid_s3_design_protocol()
    rows: list[dict[str, Any]] = []
    for n_qubits in sizes:
        if n_qubits < 2 or n_qubits > len(OMEGA_N_16):
            raise ValueError("sizes must be between 2 and len(OMEGA_N_16)")
        scored = score_s3_candidates(
            protocol,
            build_knm_paper27(n_qubits),
            np.asarray(OMEGA_N_16[:n_qubits], dtype=np.float64),
        )
        for row in scored:
            data = row.to_dict()
            data["n_qubits"] = n_qubits
            rows.append(data)
    return rows


def _feature_vector(row: dict[str, Any]) -> list[float]:
    metrics = row["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a mapping")
    family = row["family"]
    return [
        1.0 if family == "ansatz" else 0.0,
        1.0 if family == "pulse" else 0.0,
        float(row["n_qubits"]),
        float(metrics.get("depth", 0.0)),
        float(metrics.get("two_qubit_gates", 0.0)),
        float(metrics.get("pulse_count", 0.0)),
        float(metrics.get("infidelity_bound", 0.0)),
        float(metrics.get("max_points_per_pulse", 0.0)),
    ]


def _fit_ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xtx = x.T @ x
    regulariser = RIDGE_ALPHA * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(xtx + regulariser, x.T @ y)


def _evaluate(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denom if denom > 0.0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _train_holdout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([_feature_vector(row) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    holdout = np.asarray([index % 5 == 0 for index in range(len(rows))], dtype=bool)
    train = ~holdout
    coefficients = _fit_ridge(x[train], y[train])
    train_metrics = _evaluate(x[train] @ coefficients, y[train])
    holdout_metrics = _evaluate(x[holdout] @ coefficients, y[holdout])
    families = sorted({str(row["family"]) for row in rows})
    family_metrics: dict[str, dict[str, float]] = {}
    for family in families:
        mask = np.asarray([row["family"] == family for row in rows], dtype=bool)
        family_metrics[family] = _evaluate(x[mask] @ coefficients, y[mask])
    return {
        "model": "closed_form_ridge_linear_surrogate",
        "ridge_alpha": RIDGE_ALPHA,
        "feature_names": [
            "is_ansatz",
            "is_pulse",
            "n_qubits",
            "depth",
            "two_qubit_gates",
            "pulse_count",
            "infidelity_bound",
            "max_points_per_pulse",
        ],
        "coefficients": [float(value) for value in coefficients],
        "train_rows": int(np.sum(train)),
        "holdout_rows": int(np.sum(holdout)),
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "family_metrics": family_metrics,
    }


def _markdown(summary: dict[str, Any]) -> str:
    surrogate = summary["surrogate"]
    if not isinstance(surrogate, dict):
        raise TypeError("surrogate must be a mapping")
    lines = [
        "# S3 Design Surrogate Rehearsal",
        "",
        f"Protocol ID: `{summary['protocol_id']}`",
        "",
        "Submission state: no hardware submission; deterministic no-QPU surrogate rehearsal.",
        "",
        "## Dataset",
        f"- rows: {summary['row_count']}",
        f"- sizes: {summary['sizes']}",
        "",
        "## Surrogate",
        f"- model: {surrogate['model']}",
        f"- train rows: {surrogate['train_rows']}",
        f"- holdout rows: {surrogate['holdout_rows']}",
        f"- holdout metrics: {json.dumps(surrogate['holdout_metrics'], sort_keys=True)}",
        "",
        "## Claim Boundary",
        summary["claim_boundary"],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Train and persist the S3 design-surrogate model artefact."""
    args = _parse_args()
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    rows = _rows_for_sizes(sizes)
    surrogate = _train_holdout(rows)
    summary = {
        "date": DATE,
        "protocol_id": grid_s3_design_protocol().protocol_id,
        "script": "scripts/train_s3_design_surrogate.py",
        "hardware_submission": False,
        "ml_training_performed": True,
        "sizes": sizes,
        "row_count": len(rows),
        "rows": rows,
        "surrogate": surrogate,
        "claim_boundary": (
            "This is a deterministic no-QPU surrogate rehearsal over proxy scores. "
            "It does not demonstrate pulse-level hardware improvement, VQE improvement, "
            "or quantum advantage."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"s3_design_surrogate_summary_{DATE}.json"
    md_path = args.out_dir / f"s3_design_surrogate_summary_{DATE}.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_markdown={_sha256(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
