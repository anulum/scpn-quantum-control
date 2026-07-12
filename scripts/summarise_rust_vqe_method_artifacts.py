#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — summarise rust VQE method artifacts script
# scpn-quantum-control -- methods paper artefact summariser
"""Combine generated methods-paper benchmark artefacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rust_vqe_methods"
DATE = "2026-05-05"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _remote_machine_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_DIR.glob("remote_knm_benchmark_*_2026-05-05.json")):
        data = _load_json(path)
        for row in data["rows"]:
            if row.get("status") != "ok":
                continue
            rows.append(
                {
                    "source": "remote_knm",
                    "label": data["label"],
                    "hostname": data["machine"]["hostname"],
                    "load_average_1m": data["machine"]["load_average"][0]
                    if data["machine"].get("load_average")
                    else None,
                    "backend": row["language"],
                    "n_qubits": row["n"],
                    "batch_size": None,
                    "median_ms": row["median_ms"],
                    "mean_ms": row["mean_ms"],
                    "status": row["status"],
                    "artefact": path.name,
                }
            )
    return rows


def _gpu_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_DIR.glob("gpu_benchmark_summary_*_2026-05-05.json")):
        data = _load_json(path)
        for row in data["rows"]:
            rows.append(
                {
                    "source": "gpu_expectation",
                    "label": data["label"],
                    "hostname": data["machine"]["hostname"],
                    "load_average_1m": data["machine"]["load_average"][0]
                    if data["machine"].get("load_average")
                    else None,
                    "backend": row["backend"],
                    "n_qubits": row["n_qubits"],
                    "batch_size": row["batch_size"],
                    "median_ms": row.get("median_ms"),
                    "mean_ms": row.get("mean_ms"),
                    "status": row["status"],
                    "artefact": path.name,
                }
            )
    return rows


def main() -> int:
    """Summarize Rust/VQE method artefacts into the configured report."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _remote_machine_rows() + _gpu_rows()
    summary = {
        "date": DATE,
        "schema": "scpn_rust_vqe_methods_combined_summary_v1",
        "timing_caveat": (
            "Combined generated artefact summary. All timings are opportunistic "
            "unless the source artefact states otherwise; CPU/GPU load, affinity, "
            "clock governor, thermal state, and competing workloads were not "
            "controlled."
        ),
        "rows": rows,
    }
    json_path = OUT_DIR / f"combined_methods_benchmark_summary_{DATE}.json"
    csv_path = OUT_DIR / f"combined_methods_benchmark_summary_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary["artefacts"] = {
        "json": str(json_path),
        "json_sha256": _sha256(json_path),
        "csv": str(csv_path),
        "csv_sha256": _sha256(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
