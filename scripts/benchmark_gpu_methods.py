#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- GPU methods benchmark harness
"""Generate GPU-assisted classical-validation benchmark tables.

The benchmark targets batched state-vector expectation evaluation, which is a
real workload in VQE and classical validation.  It deliberately does not use GPU
timings for tiny scalar K_nm construction, where transfer overhead dominates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scpn_quantum_control.bridge.knm_hamiltonian import (
        OMEGA_N_16,
        build_knm_paper27,
        knm_to_dense_matrix,
    )
except ImportError:
    OMEGA_N_16 = np.array(
        [
            0.37,
            -0.21,
            0.13,
            -0.08,
            0.31,
            -0.17,
            0.23,
            -0.29,
            0.11,
            -0.19,
            0.07,
            -0.05,
            0.27,
            -0.33,
            0.15,
            -0.25,
        ],
        dtype=np.float64,
    )

    def build_knm_paper27(L: int = 16, K_base: float = 0.45, K_alpha: float = 0.3) -> np.ndarray:
        """Build the local fallback Paper-27 K_nm coupling matrix."""

        n = L
        idx = np.arange(n)
        matrix = K_base * np.exp(-K_alpha * np.abs(idx[:, None] - idx[None, :]))
        np.fill_diagonal(matrix, 0.0)
        anchors = {(0, 1): 0.302, (1, 2): 0.201, (2, 3): 0.252, (3, 4): 0.154}
        for (row, column), value in anchors.items():
            if row < n and column < n:
                matrix[row, column] = matrix[column, row] = value
        if n > 15:
            matrix[0, 15] = matrix[15, 0] = max(matrix[0, 15], 0.05)
        if n > 6:
            matrix[4, 6] = matrix[6, 4] = max(matrix[4, 6], 0.15)
        return matrix

    def knm_to_dense_matrix(
        K: np.ndarray, omega: np.ndarray, delta: float = 0.0, *, max_dense_gib: float | None = None
    ) -> np.ndarray:
        """Build the local fallback dense Hamiltonian matrix."""

        del delta, max_dense_gib
        k_matrix = K
        n = int(k_matrix.shape[0])
        dim = 2**n
        hamiltonian = np.zeros((dim, dim), dtype=np.complex64)
        for basis in range(dim):
            diagonal = 0.0
            for qubit in range(n):
                diagonal += omega[qubit] * (1.0 if ((basis >> qubit) & 1) == 0 else -1.0)
            hamiltonian[basis, basis] = diagonal
            for row in range(n):
                for column in range(row + 1, n):
                    bit_row = (basis >> row) & 1
                    bit_column = (basis >> column) & 1
                    if bit_row != bit_column:
                        target = basis ^ (1 << row) ^ (1 << column)
                        hamiltonian[target, basis] += k_matrix[row, column]
        return hamiltonian


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rust_vqe_methods"
DATE = "2026-05-05"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _samples_summary(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
    return {
        "mean_ms": float(statistics.mean(samples_ms)),
        "median_ms": float(statistics.median(samples_ms)),
        "p95_ms": float(p95),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
    }


def _normalised_states(rng: np.random.Generator, batch: int, dim: int) -> np.ndarray:
    real = rng.normal(size=(batch, dim)).astype(np.float32)
    imag = rng.normal(size=(batch, dim)).astype(np.float32)
    states = real + 1j * imag
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    return np.asarray((states / norms).astype(np.complex64))


def _numpy_expectation(states: np.ndarray, hamiltonian: np.ndarray) -> np.ndarray:
    h_states = states @ hamiltonian.T
    return np.asarray(np.einsum("bi,bi->b", states.conj(), h_states).real)


def _time_numpy(
    states: np.ndarray, hamiltonian: np.ndarray, repeats: int
) -> tuple[dict[str, float], float]:
    samples = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        values = _numpy_expectation(states, hamiltonian)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
        checksum = float(np.sum(values))
    return _samples_summary(samples), checksum


def _torch_metadata() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }


def _time_torch(
    states: np.ndarray,
    hamiltonian: np.ndarray,
    repeats: int,
    device_name: str,
) -> tuple[dict[str, float], float]:
    import torch

    device = torch.device(device_name)
    torch_states = torch.as_tensor(states, dtype=torch.complex64, device=device)
    torch_hamiltonian = torch.as_tensor(
        hamiltonian.astype(np.complex64), dtype=torch.complex64, device=device
    )

    def evaluate() -> Any:
        h_states = torch_states @ torch_hamiltonian.T
        return (torch.conj(torch_states) * h_states).sum(dim=1).real

    for _ in range(5):
        evaluate()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    samples = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        values = evaluate()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
        checksum = float(values.sum().detach().cpu().item())
    return _samples_summary(samples), checksum


def _machine_metadata() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }


def _benchmark_case(
    n: int, batch: int, repeats: int, include_torch_cpu: bool
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260505 + n + batch)
    k = build_knm_paper27(n)
    hamiltonian = knm_to_dense_matrix(k, OMEGA_N_16[:n]).astype(np.complex64)
    states = _normalised_states(rng, batch, 2**n)
    rows: list[dict[str, Any]] = []

    numpy_stats, numpy_checksum = _time_numpy(states, hamiltonian, repeats)
    rows.append(
        {
            "backend": "numpy_cpu",
            "status": "ok",
            "n_qubits": n,
            "hilbert_dim": 2**n,
            "batch_size": batch,
            "repeats": repeats,
            "checksum": numpy_checksum,
            **numpy_stats,
        }
    )

    torch_info = _torch_metadata()
    if not torch_info["available"]:
        rows.append(
            {
                "backend": "torch_cpu",
                "status": "unavailable",
                "reason": torch_info["reason"],
                "n_qubits": n,
                "hilbert_dim": 2**n,
                "batch_size": batch,
                "repeats": repeats,
            }
        )
        rows.append(
            {
                "backend": "torch_cuda",
                "status": "unavailable",
                "reason": torch_info["reason"],
                "n_qubits": n,
                "hilbert_dim": 2**n,
                "batch_size": batch,
                "repeats": repeats,
            }
        )
        return rows

    if include_torch_cpu:
        torch_cpu_stats, torch_cpu_checksum = _time_torch(states, hamiltonian, repeats, "cpu")
        rows.append(
            {
                "backend": "torch_cpu",
                "status": "ok",
                "n_qubits": n,
                "hilbert_dim": 2**n,
                "batch_size": batch,
                "repeats": repeats,
                "checksum": torch_cpu_checksum,
                **torch_cpu_stats,
            }
        )

    if torch_info["cuda_available"]:
        torch_cuda_stats, torch_cuda_checksum = _time_torch(states, hamiltonian, repeats, "cuda")
        rows.append(
            {
                "backend": "torch_cuda",
                "status": "ok",
                "n_qubits": n,
                "hilbert_dim": 2**n,
                "batch_size": batch,
                "repeats": repeats,
                "checksum": torch_cuda_checksum,
                **torch_cuda_stats,
            }
        )
    else:
        rows.append(
            {
                "backend": "torch_cuda",
                "status": "unavailable",
                "reason": "CUDA backend unavailable in this Python environment",
                "n_qubits": n,
                "hilbert_dim": 2**n,
                "batch_size": batch,
                "repeats": repeats,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--label", default=socket.gethostname().replace(".", "_"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--include-torch-cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the GPU-methods benchmark CLI."""

    ns = _parse_args()
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for n in ns.sizes:
        rows.extend(_benchmark_case(n, ns.batch_size, ns.repeats, ns.include_torch_cpu))

    summary = {
        "date": DATE,
        "schema": "scpn_gpu_methods_benchmark_v1",
        "label": ns.label,
        "command": (
            "PYTHONDONTWRITEBYTECODE=1 .venv-linux/bin/python "
            "scripts/benchmark_gpu_methods.py --include-torch-cpu"
        ),
        "machine": _machine_metadata(),
        "torch": _torch_metadata(),
        "timing_caveat": (
            "Opportunistic shared-machine timing. GPU clocks, CPU affinity, thermal "
            "state, driver version, and competing workloads were not pinned. The "
            "workload is batched state-vector expectation evaluation for classical "
            "validation and VQE scoring, not K_nm construction."
        ),
        "rows": rows,
    }

    json_path = ns.output_dir / f"gpu_benchmark_summary_{ns.label}_{DATE}.json"
    csv_path = ns.output_dir / f"gpu_benchmark_summary_{ns.label}_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row} | {"label"})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"label": ns.label, **row})

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
