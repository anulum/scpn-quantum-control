# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum-advantage guardrail matrix for classical / Rust / GPU baselines

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_dense_matrix,
    knm_to_hamiltonian,
)
from scpn_quantum_control.hardware import classical_exact_evolution
from scpn_quantum_control.hardware.gpu_accel import eigh as gpu_eigh
from scpn_quantum_control.hardware.gpu_accel import gpu_device_name, is_gpu_available

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "classical_rust_gpu_matrix_2026-05-03.json"


@dataclass(frozen=True)
class BenchmarkRow:
    n_qubits: int
    dim: int
    classical_ode_ms: float | None
    exact_diag_cpu_ms: float | None
    exact_diag_sparse_ms: float | None
    exact_diag_gpu_ms: float | None
    rust_ode_ms: float | None
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "dim": self.dim,
            "classical_ode_ms": self.classical_ode_ms,
            "exact_diag_cpu_ms": self.exact_diag_cpu_ms,
            "exact_diag_sparse_ms": self.exact_diag_sparse_ms,
            "exact_diag_gpu_ms": self.exact_diag_gpu_ms,
            "rust_ode_ms": self.rust_ode_ms,
            "notes": self.notes,
        }


def _git_commit() -> str:
    git_dir = REPO_ROOT / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return "unknown"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref_path = git_dir / head.removeprefix("ref: ").strip()
        return ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _dependency_versions() -> dict[str, str]:
    try:
        import qiskit

        qiskit_version = str(qiskit.__version__)
    except Exception:
        qiskit_version = "not installed"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "qiskit": qiskit_version,
        "gpu": gpu_device_name(),
    }


def _timed_call(func: Callable[[], Any], *, repeats: int = 1) -> tuple[float | None, str]:
    timings: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        try:
            func()
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)
    return float(np.median(timings)), ""


def _bench_rust_ode_ms(
    theta0: np.ndarray, omega: np.ndarray, K: np.ndarray, dt: float, steps: int
) -> tuple[float | None, str]:
    try:
        engine = importlib.import_module("scpn_quantum_engine")
    except Exception as exc:
        return None, f"scpn_quantum_engine unavailable: {type(exc).__name__}"
    kuramoto_euler = getattr(engine, "kuramoto_euler", None)
    if not callable(kuramoto_euler):
        return None, "scpn_quantum_engine.kuramoto_euler unavailable"

    def _run() -> None:
        _ = kuramoto_euler(theta0, omega, K, dt, steps)

    return _timed_call(_run)


def _bench_exact_diag_cpu_ms(K: np.ndarray, omega: np.ndarray) -> tuple[float | None, str]:
    def _run() -> None:
        H = knm_to_dense_matrix(K, omega)
        _ = np.linalg.eigh(H)

    return _timed_call(_run)


def _bench_exact_diag_sparse_ms(
    K: np.ndarray, omega: np.ndarray, maxiter: int
) -> tuple[float | None, str]:
    H_op = knm_to_hamiltonian(K, omega)
    raw = H_op.to_matrix(sparse=True)
    H_sparse = raw if hasattr(raw, "tocsc") else csc_matrix(raw)
    H = H_sparse.tocsc()

    dim = int(H.shape[0])
    k = min(6, max(1, dim - 2))

    def _run() -> None:
        _ = eigsh(H, k=k, which="SA", tol=1.0e-3, maxiter=maxiter)

    return _timed_call(_run)


def _bench_exact_diag_gpu_ms(K: np.ndarray, omega: np.ndarray) -> tuple[float | None, str]:
    if not is_gpu_available():
        return None, "gpu_disabled_or_unavailable"
    if len(K) >= 14:
        return None, "exact_diag_gpu_ms:skipped_by_size_gate"

    def _run() -> None:
        H = knm_to_dense_matrix(K, omega)
        _ = gpu_eigh(H)

    return _timed_call(_run)


def _bench_row(
    n: int,
    t_max: float,
    dt: float,
    repeats: int,
    max_dense_dim: int,
    max_sparse_dim: int,
    max_sparse_iter: int,
) -> BenchmarkRow:
    if n <= len(OMEGA_N_16):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n].copy()
    else:
        K = build_knm_paper27(L=n)
        omega = np.linspace(0.1, 1.0, n)

    theta0 = np.asarray(omega % (2 * np.pi), dtype=float)
    dim = int(2**n)
    notes: list[str] = []
    n_steps = max(1, round(t_max / dt))

    ode_ms, msg = _timed_call(
        lambda: classical_exact_evolution(n, t_max, dt, K=K, omega=omega),
        repeats=repeats,
    )
    if msg:
        notes.append(f"classical_ode_ms:{msg}")
        ode_ms = None

    cpu_ms = None
    if dim <= max_dense_dim:
        cpu_ms, msg = _bench_exact_diag_cpu_ms(K, omega)
        if msg:
            notes.append(f"exact_diag_cpu_ms:{msg}")
            cpu_ms = None
    else:
        notes.append("exact_diag_cpu_ms:skipped_by_size_gate")

    sparse_ms = None
    if dim <= max_sparse_dim:
        sparse_ms, msg = _bench_exact_diag_sparse_ms(K, omega, max_sparse_iter)
        if msg:
            notes.append(f"exact_diag_sparse_ms:{msg}")
            sparse_ms = None
    else:
        notes.append("exact_diag_sparse_ms:skipped_by_size_gate")

    gpu_ms, msg = _bench_exact_diag_gpu_ms(K, omega)
    if msg:
        notes.append(msg if msg.startswith("exact_diag_gpu_ms:") else f"exact_diag_gpu_ms:{msg}")
        gpu_ms = None

    rust_ms, msg = _bench_rust_ode_ms(theta0, omega, K, dt, n_steps)
    if msg:
        notes.append(f"rust_ode_ms:{msg}")
        rust_ms = None

    if n > 12 and cpu_ms is None:
        notes.append("exact_diag_cpu_ms:expected_unavailable_for_large_dim")

    return BenchmarkRow(
        n_qubits=n,
        dim=dim,
        classical_ode_ms=ode_ms,
        exact_diag_cpu_ms=cpu_ms,
        exact_diag_sparse_ms=sparse_ms,
        exact_diag_gpu_ms=gpu_ms,
        rust_ode_ms=rust_ms,
        notes=notes,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="4,6,8,10,12,14,16",
        help="Comma-separated qubit counts.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=0.2,
        help="Total integration time.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Integration step.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat count per method.",
    )
    parser.add_argument(
        "--max-dense-dim",
        type=int,
        default=2048,
        help="Skip dense CPU eigensolve above this Hilbert dimension.",
    )
    parser.add_argument(
        "--max-sparse-dim",
        type=int,
        default=2048,
        help="Skip sparse eigensolve above this Hilbert dimension.",
    )
    parser.add_argument(
        "--max-sparse-iter",
        type=int,
        default=40,
        help="ARPACK maxiter for sparse eigensolve.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON artifact path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sizes = [int(token.strip()) for token in args.sizes.split(",") if token.strip()]
    if any(n < 2 for n in sizes):
        raise ValueError("sizes must be >=2")
    if args.dt <= 0:
        raise ValueError("dt must be > 0")
    if args.t_max < 0:
        raise ValueError("t-max must be >=0")

    rows = [
        _bench_row(
            size,
            t_max=args.t_max,
            dt=args.dt,
            repeats=args.repeats,
            max_dense_dim=args.max_dense_dim,
            max_sparse_dim=args.max_sparse_dim,
            max_sparse_iter=args.max_sparse_iter,
        )
        for size in sizes
    ]

    payload = {
        "schema_version": 1,
        "matrix": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": [Path(sys.argv[0]).name] + sys.argv[1:],
            "git_commit": _git_commit(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": _dependency_versions(),
            "sizes": sizes,
            "settings": {
                "t_max": args.t_max,
                "dt": args.dt,
                "repeats": args.repeats,
                "max_dense_dim": args.max_dense_dim,
                "max_sparse_dim": args.max_sparse_dim,
                "max_sparse_iter": args.max_sparse_iter,
            },
            "rows": [row.as_dict() for row in rows],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote quantum advantage classical/Rust/GPU matrix: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
