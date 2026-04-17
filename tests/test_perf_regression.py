# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Perf regression gate
"""Performance regression gate.

Runs the five paired Python↔Rust benchmarks published in
`docs/pipeline_performance.md §21` and fails the test if on the
runner it executes on, the Rust path is **less than 2× faster than
the Python path** on the same runner in the same run.

Why a relative-to-Python-on-the-same-runner gate and not an
absolute-to-§21 gate. The §21 numbers were measured on an ML350
Gen8 with 24 Xeon E5-2640 cores; GitHub Actions runners and a
developer laptop exhibit very different absolute speedups for the
same code. An absolute gate would flake on every CI class change.
A `Rust / Python` ratio on the same runner is runner-independent;
if the ratio ever falls below 2×, something in the Rust path has
genuinely regressed.

This is the companion to audit item **C1** in
the internal gap audit and the
falsifier for claim **C4** in `docs/falsification.md`.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

try:
    import scpn_quantum_engine as eng

    _RUST_OK = True
except ImportError:
    _RUST_OK = False

pytestmark = pytest.mark.skipif(not _RUST_OK, reason="scpn-quantum-engine not available")

# Minimum acceptable speedup for every paired benchmark. Generous so
# a slow CI runner does not flake; strict enough that a Rust path
# becoming slower than Python (regression) is caught.
MIN_SPEEDUP = 2.0


def _timed(fn, *args, **kwargs) -> tuple[object, float]:
    t0 = time.perf_counter()
    r = fn(*args, **kwargs)
    return r, (time.perf_counter() - t0) * 1000.0


def _timed_mean(fn, *args, repeats: int = 5, **kwargs) -> float:
    """Take the median of `repeats` runs in milliseconds.

    Median of 5 is robust to a single scheduler hiccup and stays
    under a second even on slow GitHub runners.
    """
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def _speedup(label: str, dt_rust_ms: float, dt_py_ms: float) -> float:
    ratio = dt_py_ms / max(dt_rust_ms, 1e-6)
    print(
        f"\n  perf-gate {label}: Rust {dt_rust_ms:.3f} ms, "
        f"Python {dt_py_ms:.3f} ms -> {ratio:.2f}x speedup "
        f"(floor {MIN_SPEEDUP:.1f}x)"
    )
    return ratio


def _assert_above_floor(label: str, observed: float) -> None:
    assert observed >= MIN_SPEEDUP, (
        f"{label} regressed: Rust is {observed:.2f}x Python, floor is {MIN_SPEEDUP:.1f}x. "
        f"Either the Rust path lost optimisation or the Python baseline found a shortcut; "
        f"investigate before merging."
    )


class TestBuildKnmSpeedup:
    def test_build_knm_16(self) -> None:
        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        n = 16
        # Warm-up each path once to exclude import / JIT cost.
        eng.build_knm(n, 0.45, 0.3)
        build_knm_paper27(L=n)

        dt_r = _timed_mean(eng.build_knm, n, 0.45, 0.3)
        dt_p = _timed_mean(build_knm_paper27, L=n)
        _assert_above_floor("build_knm", _speedup("build_knm", dt_r, dt_p))


class TestKuramotoEulerSpeedup:
    def test_kuramoto_euler_8_1000(self) -> None:
        from scpn_quantum_control.hardware.classical import (
            classical_kuramoto_reference,
        )

        n = 8
        theta0 = np.zeros(n)
        omega = np.linspace(0.9, 1.1, n)
        K = np.ones((n, n)) * 0.1
        np.fill_diagonal(K, 0.0)

        eng.kuramoto_euler(theta0, omega, K, 0.01, 100)
        classical_kuramoto_reference(n_osc=n, t_max=1.0, dt=0.01)

        dt_r = _timed_mean(eng.kuramoto_euler, theta0, omega, K, 0.01, 1000)
        dt_p = _timed_mean(classical_kuramoto_reference, n_osc=n, t_max=10.0, dt=0.01)
        _assert_above_floor("kuramoto_euler", _speedup("kuramoto_euler", dt_r, dt_p))


class TestCorrelationMatrixXYSpeedup:
    def test_correlation_matrix_xy_n4(self) -> None:
        """Use n=4 (not n=3) so Python overhead does not dominate.

        At n=3 both paths finish in < 2 ms and the Rust win is
        hidden behind per-call cost on the Python side. n=4
        (dim=16) puts Rust firmly ahead while still running in
        a few milliseconds.
        """
        from qiskit.quantum_info import SparsePauliOp, Statevector

        n = 4
        dim = 2**n
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)
        psi_re = np.ascontiguousarray(psi.real)
        psi_im = np.ascontiguousarray(psi.imag)

        def py_correlation() -> np.ndarray:
            sv = Statevector(psi)
            C = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    x_str = ["I"] * n
                    x_str[i] = "X"
                    x_str[j] = "X"
                    y_str = ["I"] * n
                    y_str[i] = "Y"
                    y_str[j] = "Y"
                    xx = sv.expectation_value(SparsePauliOp("".join(reversed(x_str)))).real
                    yy = sv.expectation_value(SparsePauliOp("".join(reversed(y_str)))).real
                    C[i, j] = xx + yy
                    C[j, i] = xx + yy
            return C

        eng.correlation_matrix_xy(psi_re, psi_im, n)
        py_correlation()

        dt_r = _timed_mean(eng.correlation_matrix_xy, psi_re, psi_im, n)
        dt_p = _timed_mean(py_correlation)
        _assert_above_floor(
            "correlation_matrix_xy",
            _speedup("correlation_matrix_xy", dt_r, dt_p),
        )


class TestLindbladJumpOpsCOOSpeedup:
    def test_lindblad_jump_ops_coo_n4(self) -> None:
        """Use n=4 so the inner loop is long enough to measure."""
        n = 4
        dim = 1 << n
        rng = np.random.default_rng(42)
        K = rng.random((n, n)) * 0.3
        K = (K + K.T) / 2
        np.fill_diagonal(K, 0.0)

        def py_jump_ops() -> list[tuple[list[int], list[int]]]:
            ops: list[tuple[list[int], list[int]]] = []
            for i in range(n):
                for j in range(n):
                    if i != j and abs(K[i, j]) > 1e-5:
                        r_py, c_py = [], []
                        for idx in range(dim):
                            if ((idx >> i) & 1) == 1 and ((idx >> j) & 1) == 0:
                                r_py.append(idx ^ ((1 << i) | (1 << j)))
                                c_py.append(idx)
                        ops.append((sorted(r_py), sorted(c_py)))
            return ops

        eng.lindblad_jump_ops_coo(K.ravel(), n, 1e-5)
        py_jump_ops()

        dt_r = _timed_mean(eng.lindblad_jump_ops_coo, K.ravel(), n, 1e-5)
        dt_p = _timed_mean(py_jump_ops)
        _assert_above_floor(
            "lindblad_jump_ops_coo",
            _speedup("lindblad_jump_ops_coo", dt_r, dt_p),
        )


class TestLindbladAntiHermitianDiagSpeedup:
    def test_lindblad_anti_hermitian_diag_n4(self) -> None:
        """Use n=4 so the inner loop dominates."""
        n = 4
        dim = 1 << n
        rng = np.random.default_rng(42)
        K = rng.random((n, n)) * 0.5
        K = (K + K.T) / 2
        np.fill_diagonal(K, 0.0)

        def py_anti_hermitian_diag() -> np.ndarray:
            diag = np.zeros(dim)
            for i in range(n):
                for j in range(n):
                    if i != j and abs(K[i, j]) > 1e-5:
                        for idx in range(dim):
                            if ((idx >> i) & 1) == 1 and ((idx >> j) & 1) == 0:
                                diag[idx] += 1.0
            return diag

        eng.lindblad_anti_hermitian_diag(K.ravel(), n, 1e-5)
        py_anti_hermitian_diag()

        dt_r = _timed_mean(eng.lindblad_anti_hermitian_diag, K.ravel(), n, 1e-5)
        dt_p = _timed_mean(py_anti_hermitian_diag)
        _assert_above_floor(
            "lindblad_anti_hermitian_diag",
            _speedup("lindblad_anti_hermitian_diag", dt_r, dt_p),
        )
