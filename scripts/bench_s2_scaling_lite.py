#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S2 lite scaling harness
"""Generate lightweight protocol-compliant S2 scaling rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import tracemalloc
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import eigsh

from scpn_quantum_control.benchmarks.advantage_protocol import (
    default_s2_scaling_protocol,
    validate_scaling_rows,
)
from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_dense_matrix,
    knm_to_sparse_matrix,
)
from scpn_quantum_control.hardware.classical import classical_exact_evolution
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s2_advantage_scaling"
DATE = "2026-05-06"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="4,6", help="Comma-separated lite qubit sizes.")
    parser.add_argument("--max-dense-qubits", type=int, default=6)
    parser.add_argument("--max-sparse-qubits", type=int, default=6)
    parser.add_argument("--max-tn-qubits", type=int, default=6)
    parser.add_argument("--max-statevector-qubits", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    git_path = REPO_ROOT / ".git"
    if git_path.is_file():
        prefix = "gitdir: "
        text = git_path.read_text(encoding="utf-8").strip()
        if not text.startswith(prefix):
            return "unknown"
        git_path = (REPO_ROOT / text[len(prefix) :]).resolve()
    head_path = git_path / "HEAD"
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    ref_prefix = "ref: "
    if not head.startswith(ref_prefix):
        return head
    ref_path = git_path / head[len(ref_prefix) :]
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()
    return "unknown"


def _dependencies() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _base_row(n_qubits: int, baseline: str, status: str, notes: list[str]) -> dict[str, Any]:
    protocol = default_s2_scaling_protocol()
    return {
        "protocol_id": protocol.protocol_id,
        "n_qubits": n_qubits,
        "baseline": baseline,
        "status": status,
        "wall_time_ms": None,
        "memory_bytes": None,
        "metric_payload": {},
        "command": "PYTHONDONTWRITEBYTECODE=1 python scripts/bench_s2_scaling_lite.py",
        "machine": platform.platform(),
        "dependencies": _dependencies(),
        "git_commit": _git_commit(),
        "notes": notes,
    }


def _timed(fn: Callable[[], dict[str, Any]]) -> tuple[float, int, dict[str, Any]]:
    tracemalloc.start()
    started = time.perf_counter_ns()
    payload = fn()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed_ms, int(peak_bytes), payload


def _problem(n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    k_matrix = build_knm_paper27(n_qubits)
    omega = (
        OMEGA_N_16[:n_qubits] if n_qubits <= len(OMEGA_N_16) else np.linspace(0.1, 1.0, n_qubits)
    )
    return k_matrix, np.asarray(omega, dtype=np.float64)


def _classical_ode_row(n_qubits: int) -> dict[str, Any]:
    row = _base_row(n_qubits, "classical_ode", "ok", [])
    k_matrix, omega = _problem(n_qubits)
    elapsed_ms, peak_bytes, payload = _timed(
        lambda: {
            "final_R": float(
                classical_exact_evolution(n_qubits, 0.1, 0.1, K=k_matrix, omega=omega)["R"][-1]
            )
        }
    )
    row["wall_time_ms"] = elapsed_ms
    row["memory_bytes"] = peak_bytes
    row["metric_payload"] = payload
    row["metric_payload"]["input_bytes"] = int(k_matrix.nbytes + omega.nbytes)
    row["metric_payload"]["peak_tracemalloc_bytes"] = peak_bytes
    return row


def _dense_eigh_row(n_qubits: int, max_dense_qubits: int) -> dict[str, Any]:
    if n_qubits > max_dense_qubits:
        return _base_row(n_qubits, "dense_eigh", "skipped", ["size gate"])
    row = _base_row(n_qubits, "dense_eigh", "ok", [])
    k_matrix, omega = _problem(n_qubits)

    def run() -> dict[str, Any]:
        hamiltonian = knm_to_dense_matrix(k_matrix, omega)
        values = np.linalg.eigvalsh(hamiltonian)
        return {
            "ground_energy": float(values[0]),
            "hilbert_dim": int(2**n_qubits),
        }

    elapsed_ms, peak_bytes, payload = _timed(run)
    row["wall_time_ms"] = elapsed_ms
    row["memory_bytes"] = peak_bytes
    row["metric_payload"] = payload
    row["metric_payload"]["estimated_dense_matrix_bytes"] = int(
        (2**n_qubits) ** 2 * np.dtype(np.complex128).itemsize
    )
    row["metric_payload"]["peak_tracemalloc_bytes"] = peak_bytes
    return row


def _schmidt_values(state: np.ndarray, n_qubits: int, cut: int) -> np.ndarray:
    left_dim = 2**cut
    right_dim = 2 ** (n_qubits - cut)
    return np.linalg.svd(state.reshape(left_dim, right_dim), compute_uv=False)


def _discarded_weight(singular_values: np.ndarray, max_bond: int) -> float:
    if max_bond >= singular_values.size:
        return 0.0
    tail = singular_values[max_bond:]
    return float(np.sum(np.square(np.abs(tail))))


def _entropy_bits(singular_values: np.ndarray) -> float:
    probabilities = np.square(np.abs(singular_values))
    nonzero = probabilities[probabilities > 1.0e-15]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _sparse_eigsh_row(n_qubits: int, max_sparse_qubits: int) -> dict[str, Any]:
    if n_qubits > max_sparse_qubits:
        return _base_row(n_qubits, "sparse_eigsh", "skipped", ["size gate"])
    row = _base_row(n_qubits, "sparse_eigsh", "ok", [])
    k_matrix, omega = _problem(n_qubits)

    def run() -> dict[str, Any]:
        hamiltonian = knm_to_sparse_matrix(k_matrix, omega)
        values, vectors = eigsh(hamiltonian, k=1, which="SA", tol=1.0e-9)
        ground_energy = float(values[0])
        vector = np.asarray(vectors[:, 0], dtype=np.complex128)
        residual = hamiltonian @ vector - ground_energy * vector
        return {
            "ground_energy": ground_energy,
            "residual_norm": float(np.linalg.norm(residual)),
            "hilbert_dim": int(2**n_qubits),
        }

    elapsed_ms, peak_bytes, payload = _timed(run)
    row["wall_time_ms"] = elapsed_ms
    row["memory_bytes"] = peak_bytes
    row["metric_payload"] = payload
    row["metric_payload"]["estimated_statevector_bytes"] = int(
        (2**n_qubits) * np.dtype(np.complex128).itemsize
    )
    row["metric_payload"]["peak_tracemalloc_bytes"] = peak_bytes
    return row


def _mps_tensor_network_row(n_qubits: int, max_tn_qubits: int, max_bond: int) -> dict[str, Any]:
    if n_qubits > max_tn_qubits:
        return _base_row(n_qubits, "mps_tensor_network", "skipped", ["size gate"])
    row = _base_row(n_qubits, "mps_tensor_network", "ok", [])
    k_matrix, omega = _problem(n_qubits)

    def run() -> dict[str, Any]:
        hamiltonian = knm_to_dense_matrix(k_matrix, omega)
        values, vectors = np.linalg.eigh(hamiltonian)
        state = np.asarray(vectors[:, 0], dtype=np.complex128)
        spectra = [_schmidt_values(state, n_qubits, cut) for cut in range(1, n_qubits)]
        return {
            "ground_energy": float(values[0]),
            "max_bond": max_bond,
            "worst_cut_discarded_weight": float(
                max(_discarded_weight(spectrum, max_bond) for spectrum in spectra)
            ),
            "max_midchain_entropy_bits": float(
                max(_entropy_bits(spectrum) for spectrum in spectra)
            ),
            "hilbert_dim": int(2**n_qubits),
        }

    elapsed_ms, peak_bytes, payload = _timed(run)
    row["wall_time_ms"] = elapsed_ms
    row["memory_bytes"] = peak_bytes
    row["metric_payload"] = payload
    row["metric_payload"]["estimated_dense_matrix_bytes"] = int(
        (2**n_qubits) ** 2 * np.dtype(np.complex128).itemsize
    )
    row["metric_payload"]["peak_tracemalloc_bytes"] = peak_bytes
    return row


def _aer_statevector_row(n_qubits: int, max_statevector_qubits: int) -> dict[str, Any]:
    if n_qubits > max_statevector_qubits:
        return _base_row(n_qubits, "aer_statevector", "skipped", ["size gate"])
    row = _base_row(n_qubits, "aer_statevector", "ok", [])
    k_matrix, omega = _problem(n_qubits)

    def run() -> dict[str, Any]:
        solver = QuantumKuramotoSolver(n_qubits, k_matrix, omega)
        circuit = solver.evolve(0.1, trotter_steps=1)
        state = Statevector.from_label("0" * n_qubits).evolve(circuit)
        r_value, psi_value = solver.measure_order_parameter(state)
        return {
            "trotter_steps": 1,
            "circuit_depth": int(circuit.depth()),
            "final_R": float(r_value),
            "final_psi": float(psi_value),
            "hilbert_dim": int(2**n_qubits),
        }

    elapsed_ms, peak_bytes, payload = _timed(run)
    row["wall_time_ms"] = elapsed_ms
    row["memory_bytes"] = peak_bytes
    row["metric_payload"] = payload
    row["metric_payload"]["estimated_statevector_bytes"] = int(
        (2**n_qubits) * np.dtype(np.complex128).itemsize
    )
    row["metric_payload"]["peak_tracemalloc_bytes"] = peak_bytes
    return row


def _required_skip_row(n_qubits: int, baseline: str, reason: str) -> dict[str, Any]:
    return _base_row(n_qubits, baseline, "skipped", [reason])


def build_rows(
    sizes: Sequence[int],
    *,
    max_dense_qubits: int = 6,
    max_sparse_qubits: int = 6,
    max_tn_qubits: int = 6,
    max_statevector_qubits: int = 6,
) -> list[dict[str, Any]]:
    """Build lite S2 rows for the selected sizes."""
    rows: list[dict[str, Any]] = []
    for n_qubits in sizes:
        rows.append(_classical_ode_row(n_qubits))
        rows.append(_dense_eigh_row(n_qubits, max_dense_qubits=max_dense_qubits))
        rows.append(_sparse_eigsh_row(n_qubits, max_sparse_qubits=max_sparse_qubits))
        rows.append(_mps_tensor_network_row(n_qubits, max_tn_qubits=max_tn_qubits, max_bond=8))
        rows.append(_aer_statevector_row(n_qubits, max_statevector_qubits=max_statevector_qubits))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the lite S2 scaling benchmark row generator CLI."""

    args = _parse_args(argv)
    sizes = tuple(int(item.strip()) for item in args.sizes.split(",") if item.strip())
    rows = build_rows(
        sizes,
        max_dense_qubits=args.max_dense_qubits,
        max_sparse_qubits=args.max_sparse_qubits,
        max_tn_qubits=args.max_tn_qubits,
        max_statevector_qubits=args.max_statevector_qubits,
    )
    validation = validate_scaling_rows(default_s2_scaling_protocol(), rows)
    if not validation.valid:
        raise RuntimeError(f"S2 lite rows failed validation: {validation.to_dict()}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"s2_scaling_lite_rows_{DATE}.json"
    payload = {
        "date": DATE,
        "script": "scripts/bench_s2_scaling_lite.py",
        "hardware_submission": False,
        "advantage_claim": False,
        "size_gates": {
            "max_dense_qubits": args.max_dense_qubits,
            "max_sparse_qubits": args.max_sparse_qubits,
            "max_tn_qubits": args.max_tn_qubits,
            "max_statevector_qubits": args.max_statevector_qubits,
        },
        "rows": rows,
        "validation": validation.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote_json={out_path}")
    print(f"sha256_json={_sha256(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
