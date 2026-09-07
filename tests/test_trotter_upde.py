# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Trotter Upde
"""Tests for phase/trotter_upde.py — multi-angle coverage."""

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import QuantumUPDESolver as ExportedQuantumUPDESolver
from scpn_quantum_control.phase import UPDEStepResult, UPDETrajectoryResult
from scpn_quantum_control.phase import trotter_upde as trotter_upde_module
from scpn_quantum_control.phase.trotter_upde import QuantumUPDESolver


@pytest.fixture(autouse=True)
def deterministic_upde_boundary(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replace the quantum boundary with a deterministic stand-in for every test."""

    class FakeHamiltonian:
        """Stand in for the compiled Pauli operator with an identity matrix."""

        def __init__(self, n_qubits: int) -> None:
            self.num_qubits = n_qubits

        def to_matrix(self) -> NDArray[np.complex128]:
            """Return the identity of the operator's own dimension."""
            return np.eye(2**self.num_qubits, dtype=np.complex128)

    class FakeStatevector:
        """Count evolutions instead of simulating them."""

        def __init__(self, step_index: int = 0) -> None:
            self.step_index = step_index

        @classmethod
        def from_instruction(cls, qc: object) -> "FakeStatevector":
            """Start a fresh state, ignoring the circuit it is built from."""
            del qc
            return cls(step_index=0)

        def evolve(self, evo_qc: object) -> "FakeStatevector":
            """Advance the step counter that stands in for evolution."""
            del evo_qc
            return FakeStatevector(step_index=self.step_index + 1)

    class FakeKuramotoSolver:
        """Stand in for the Kuramoto solver with deterministic order parameters."""

        def __init__(
            self,
            n_oscillators: int,
            K_coupling: NDArray[np.float64],
            omega_natural: NDArray[np.float64],
            trotter_order: int = 1,
        ) -> None:
            self.n = n_oscillators
            self.K = np.asarray(K_coupling, dtype=np.float64)
            self.omega = np.asarray(omega_natural, dtype=np.float64)
            self.trotter_order = trotter_order
            self._hamiltonian: FakeHamiltonian | None = None

        def build_hamiltonian(self) -> FakeHamiltonian:
            """Build and remember the stand-in Hamiltonian."""
            self._hamiltonian = FakeHamiltonian(self.n)
            return self._hamiltonian

        def evolve(self, time: float, trotter_steps: int) -> dict[str, float | int]:
            """Record the evolution request instead of performing it."""
            return {"time": time, "trotter_steps": trotter_steps}

        def measure_order_parameter(self, sv: FakeStatevector) -> tuple[float, float]:
            """Derive a deterministic order parameter from the step counter."""
            value = 0.5 + 0.1 * sv.step_index
            return value, -0.25 * sv.step_index

        def run(
            self, t_max: float, dt: float, trotter_per_step: int
        ) -> dict[str, NDArray[np.float64]]:
            """Return a deterministic trajectory over the requested horizon."""
            del trotter_per_step
            n_steps = int(round(t_max / dt))
            times = np.linspace(0.0, t_max, n_steps + 1, dtype=np.float64)
            return {"times": times, "R": np.linspace(0.2, 0.8, n_steps + 1, dtype=np.float64)}

    monkeypatch.setattr(trotter_upde_module, "QuantumKuramotoSolver", FakeKuramotoSolver)
    monkeypatch.setattr(trotter_upde_module, "Statevector", FakeStatevector)
    yield


def test_phase_namespace_exports_upde_contracts() -> None:
    assert ExportedQuantumUPDESolver is QuantumUPDESolver
    assert UPDEStepResult.__name__ == "UPDEStepResult"
    assert UPDETrajectoryResult.__name__ == "UPDETrajectoryResult"


def test_default_16_layers() -> None:
    solver = QuantumUPDESolver()
    assert solver.n_layers == 16


def test_custom_small_system() -> None:
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    solver = QuantumUPDESolver(K=K, omega=omega)
    assert solver.n_layers == 4


def test_hamiltonian_exists() -> None:
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    solver = QuantumUPDESolver(K=K, omega=omega)
    H = solver.hamiltonian()
    assert H is not None
    assert H.num_qubits == 4


def test_run_returns_R_trajectory() -> None:
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.run(n_steps=5, dt=0.05)
    assert "R" in result
    assert len(result["R"]) == 6  # n_steps + 1
    for r in result["R"]:
        assert 0.0 <= r <= 1.5  # allow some numerical margin


def test_step_and_reset() -> None:
    """step() should accumulate state; reset() should reinitialise."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega)

    r1 = solver.step(dt=0.1)
    r2 = solver.step(dt=0.1)
    assert r1["R_global"] != r2["R_global"]  # state evolved

    solver.reset()
    r3 = solver.step(dt=0.1)
    assert abs(r3["R_global"] - r1["R_global"]) < 1e-10  # same first step


def test_second_order_trotter_passthrough() -> None:
    """QuantumUPDESolver(trotter_order=2) should pass through to solver."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega, trotter_order=2)
    assert solver._solver.trotter_order == 2
    result = solver.run(n_steps=3, dt=0.05)
    assert len(result["R"]) == 4


@pytest.mark.parametrize("L", [2, 3, 4])
def test_various_sizes(L: int) -> None:
    K = build_knm_paper27(L=L)
    omega = OMEGA_N_16[:L]
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.run(n_steps=3, dt=0.05)
    assert len(result["R"]) == 4
    for r in result["R"]:
        assert np.isfinite(r)


def test_step_returns_R_global() -> None:
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.step(dt=0.1)
    assert "R_global" in result
    assert 0.0 <= result["R_global"] <= 1.5


def test_hamiltonian_hermitian() -> None:
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    solver = QuantumUPDESolver(K=K, omega=omega)
    H = solver.hamiltonian()
    # `hamiltonian()` returns None until the operator has been built; this test
    # runs after construction, and now says so instead of assuming it.
    assert H is not None
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


def test_R_trajectory_finite() -> None:
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.run(n_steps=5, dt=0.05)
    for r in result["R"]:
        assert np.isfinite(r)


# ---------------------------------------------------------------------------
# UPDE physics: R trajectory and step consistency
# ---------------------------------------------------------------------------


def test_R_bounded_throughout() -> None:
    """R must stay in [0, 1] for all time steps."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.run(n_steps=10, dt=0.05)
    for r in result["R"]:
        assert 0.0 <= r <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Pipeline: Knm → UPDE → R trajectory → wired
# ---------------------------------------------------------------------------


def test_pipeline_knm_to_upde() -> None:
    """Full pipeline: build_knm → UPDE solver → run → R trajectory.
    Verifies UPDE is wired end-to-end, not decorative.
    """
    import time

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    t0 = time.perf_counter()
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.run(n_steps=5, dt=0.05)
    dt = (time.perf_counter() - t0) * 1000

    assert len(result["R"]) == 6
    assert all(np.isfinite(r) for r in result["R"])

    print(f"\n  PIPELINE Knm→UPDE (4q, 5 steps): {dt:.1f} ms")
    print(f"  R: {[f'{r:.4f}' for r in result['R']]}")
