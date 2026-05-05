#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Ansatz methods benchmark harness
"""Generate ansatz construction and transpilation tables for the methods paper."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

from qiskit import transpile
from qiskit.circuit.library import efficient_su2, n_local

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27, knm_to_ansatz

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rust_vqe_methods"
DATE = "2026-05-05"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _time_ms(fn, repeats: int) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "build_mean_ms": float(statistics.mean(values)),
        "build_median_ms": float(statistics.median(values)),
        "build_min_ms": float(min(values)),
        "build_max_ms": float(max(values)),
        "repeats": repeats,
    }


def _make_ansatz(name: str, n: int, reps: int):
    k = build_knm_paper27(n)
    if name == "knm_informed":
        return knm_to_ansatz(k, reps=reps)
    if name == "two_local":
        return n_local(n, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=reps)
    if name == "efficient_su2":
        return efficient_su2(n, reps=reps)
    raise ValueError(name)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in [3, 4, 6, 8]:
        for reps in [1, 2]:
            for name in ["knm_informed", "two_local", "efficient_su2"]:
                stats = _time_ms(lambda name=name, n=n, reps=reps: _make_ansatz(name, n, reps), 30)
                circuit = _make_ansatz(name, n, reps)
                tqc = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
                ops = circuit.count_ops()
                tops = tqc.count_ops()
                rows.append(
                    {
                        "ansatz": name,
                        "n_qubits": n,
                        "reps": reps,
                        "parameters": circuit.num_parameters,
                        "raw_depth": circuit.depth(),
                        "raw_two_qubit_gates": int(
                            sum(v for k, v in ops.items() if k in {"cx", "cz", "rzz", "ecr"})
                        ),
                        "transpiled_depth": tqc.depth(),
                        "transpiled_two_qubit_gates": int(
                            sum(v for k, v in tops.items() if k in {"cx", "cz", "rzz", "ecr"})
                        ),
                        **stats,
                    }
                )
    summary = {
        "date": DATE,
        "command": "PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/benchmark_ansatz_methods.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "timing_caveat": (
            "Opportunistic local timing on a shared workstation. CPU load from other "
            "processes was not pinned or isolated; publication-grade numbers should be "
            "rerun on an isolated benchmark host with governor/load metadata."
        ),
        "rows": rows,
    }
    json_path = OUT_DIR / f"ansatz_benchmark_summary_{DATE}.json"
    csv_path = OUT_DIR / f"ansatz_benchmark_summary_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
