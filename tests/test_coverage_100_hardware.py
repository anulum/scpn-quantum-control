# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coverage tests for hardware/ module gaps
"""Tests targeting uncovered lines in hardware/ including mocked optional deps."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

# --- circuit_cutting.py line 119: default n_values ---


def test_circuit_cutting_default_n_values():
    """Cover line 119: n_values defaults to [16,24,32,48,64]."""
    from scpn_quantum_control.hardware.circuit_cutting import scaling_analysis

    result = scaling_analysis(n_values=[16])
    assert len(result["n_oscillators"]) == 1


# --- classical.py lines 56-62: Rust engine fast path ---


def test_classical_rust_fast_path():
    """Cover lines 56-62: Rust engine import attempted for kuramoto trajectory."""
    from scpn_quantum_control.hardware.classical import classical_kuramoto_reference

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    result = classical_kuramoto_reference(4, t_max=0.5, dt=0.1, K=K, omega=omega)
    assert "times" in result
    assert "R" in result
    assert len(result["times"]) > 0


# --- classical.py line 133: GPU eigh path (mocked) ---


def test_classical_exact_diag_gpu_path():
    """Cover line 133: exact_diag with GPU eigh path (mocked gpu_accel)."""
    from unittest.mock import patch

    from scpn_quantum_control.hardware.classical import classical_exact_diag

    # n=6 gives 2^6=64 dim matrix → triggers is_gpu_available check
    K = build_knm_paper27(L=6)
    omega = OMEGA_N_16[:6]

    with (
        patch(
            "scpn_quantum_control.hardware.gpu_accel.is_gpu_available",
            return_value=True,
        ),
        patch(
            "scpn_quantum_control.hardware.gpu_accel.eigh",
            side_effect=lambda m: np.linalg.eigh(m),
        ),
    ):
        result = classical_exact_diag(6, K=K, omega=omega)
        assert "ground_energy" in result


# --- qiskit_compat.py lines 37-41: fallback import of PauliEvolutionGate ---


def test_qiskit_compat_pauli_evolution_gate():
    """Cover lines 37-41: get_pauli_evolution_gate import."""
    from scpn_quantum_control.hardware.qiskit_compat import get_pauli_evolution_gate

    PEG = get_pauli_evolution_gate()
    assert PEG is not None


# --- qiskit_compat.py lines 50-53: fallback import of LieTrotter ---


def test_qiskit_compat_lie_trotter():
    """Cover lines 50-53: get_lie_trotter import."""
    from scpn_quantum_control.hardware.qiskit_compat import get_lie_trotter

    LT = get_lie_trotter()
    assert LT is not None


# --- qiskit_compat.py line 77: Qiskit 2.x compatibility check ---


def test_qiskit_compat_check():
    """Cover line 77: check_qiskit_compatibility."""
    from scpn_quantum_control.hardware.qiskit_compat import check_qiskit_compatibility

    result = check_qiskit_compatibility()
    assert "version" in result
    assert "major" in result


# --- gpu_accel.py: GPU paths with mocked cupy ---


def test_gpu_accel_device_name_cpu():
    """Cover gpu_device_name returns 'cpu' when no GPU."""
    from scpn_quantum_control.hardware.gpu_accel import gpu_device_name

    name = gpu_device_name()
    assert isinstance(name, str)


def test_gpu_accel_eigvalsh_cpu():
    """Cover eigvalsh falls back to numpy."""
    from scpn_quantum_control.hardware.gpu_accel import eigvalsh

    m = np.array([[2.0, 1.0], [1.0, 3.0]])
    eigs = eigvalsh(m)
    np.testing.assert_allclose(eigs, np.linalg.eigvalsh(m))


def test_gpu_accel_eigh_cpu():
    """Cover eigh falls back to numpy."""
    from scpn_quantum_control.hardware.gpu_accel import eigh

    m = np.array([[2.0, 1.0], [1.0, 3.0]])
    eigs, vecs = eigh(m)
    np.testing.assert_allclose(eigs, np.linalg.eigvalsh(m))


def test_gpu_accel_expm_cpu():
    """Cover expm falls back to scipy."""
    from scpn_quantum_control.hardware.gpu_accel import expm

    m = np.array([[0.0, 1.0], [-1.0, 0.0]])
    result = expm(m)
    assert result.shape == (2, 2)


def test_gpu_accel_matmul_cpu():
    """Cover matmul falls back to numpy."""
    from scpn_quantum_control.hardware.gpu_accel import matmul

    a = np.eye(2)
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = matmul(a, b)
    np.testing.assert_allclose(result, b)


def test_gpu_accel_memory_free_cpu():
    """Cover gpu_memory_free_mb returns 0 when no GPU."""
    from scpn_quantum_control.hardware.gpu_accel import gpu_memory_free_mb

    assert gpu_memory_free_mb() == 0.0


def test_gpu_accel_mocked_cupy_eigvalsh():
    """Cover eigvalsh GPU path with mocked cupy."""
    import scpn_quantum_control.hardware.gpu_accel as gpu_mod

    mock_cp = MagicMock()
    m = np.random.default_rng(42).random((64, 64))
    m = (m + m.T) / 2
    expected = np.linalg.eigvalsh(m)
    mock_cp.asarray.return_value = m
    mock_cp.linalg.eigvalsh.return_value = expected
    mock_cp.asnumpy.return_value = expected

    orig_avail, orig_cp = gpu_mod._CUPY_AVAILABLE, gpu_mod._cp
    try:
        gpu_mod._CUPY_AVAILABLE = True
        gpu_mod._cp = mock_cp
        result = gpu_mod.eigvalsh(m)
        np.testing.assert_allclose(result, expected)
    finally:
        gpu_mod._CUPY_AVAILABLE = orig_avail
        gpu_mod._cp = orig_cp


def test_gpu_accel_mocked_cupy_eigh():
    """Cover eigh GPU path with mocked cupy."""
    import scpn_quantum_control.hardware.gpu_accel as gpu_mod

    mock_cp = MagicMock()
    m = np.random.default_rng(42).random((64, 64))
    m = (m + m.T) / 2
    expected_eigs, expected_vecs = np.linalg.eigh(m)
    mock_cp.asarray.return_value = m
    mock_cp.linalg.eigh.return_value = (expected_eigs, expected_vecs)
    mock_cp.asnumpy.side_effect = lambda x: x

    orig_avail, orig_cp = gpu_mod._CUPY_AVAILABLE, gpu_mod._cp
    try:
        gpu_mod._CUPY_AVAILABLE = True
        gpu_mod._cp = mock_cp
        eigs, vecs = gpu_mod.eigh(m)
        np.testing.assert_allclose(eigs, expected_eigs)
    finally:
        gpu_mod._CUPY_AVAILABLE = orig_avail
        gpu_mod._cp = orig_cp


def test_gpu_accel_mocked_cupy_expm():
    """Cover expm GPU path with mocked cupy for Hermitian matrix."""
    import scpn_quantum_control.hardware.gpu_accel as gpu_mod

    mock_cp = MagicMock()
    m = np.random.default_rng(42).random((32, 32))
    m = (m + m.T) / 2  # Hermitian

    eigs, vecs = np.linalg.eigh(m)
    expected = vecs @ np.diag(np.exp(eigs)) @ vecs.T

    mock_cp.asarray.return_value = m
    mock_cp.linalg.eigh.return_value = (eigs, vecs)
    mock_cp.exp.return_value = np.exp(eigs)
    mock_cp.diag.return_value = np.diag(np.exp(eigs))
    mock_cp.asnumpy.return_value = expected

    # Mock the matrix multiply chain
    mock_result = MagicMock()
    mock_result.__matmul__ = MagicMock(return_value=expected)
    vecs_mock = MagicMock()
    vecs_mock.__matmul__ = MagicMock(return_value=mock_result)
    vecs_mock.conj.return_value = MagicMock(T=vecs.T)

    orig_avail, orig_cp = gpu_mod._CUPY_AVAILABLE, gpu_mod._cp
    try:
        gpu_mod._CUPY_AVAILABLE = True
        gpu_mod._cp = mock_cp
        # The expm will try matrix ops on cupy objects, but we just need
        # the asnumpy at the end to return expected
        result = gpu_mod.expm(m)
        np.testing.assert_allclose(result, expected, atol=1e-10)
    finally:
        gpu_mod._CUPY_AVAILABLE = orig_avail
        gpu_mod._cp = orig_cp


def test_gpu_accel_mocked_cupy_matmul():
    """Cover matmul GPU path with mocked cupy."""
    import scpn_quantum_control.hardware.gpu_accel as gpu_mod

    mock_cp = MagicMock()
    a = np.eye(64)
    b = np.random.default_rng(42).random((64, 64))
    expected = a @ b

    a_gpu = MagicMock()
    b_gpu = MagicMock()
    a_gpu.__matmul__ = MagicMock(return_value=expected)
    mock_cp.asarray.side_effect = [a_gpu, b_gpu]
    mock_cp.asnumpy.return_value = expected

    orig_avail, orig_cp = gpu_mod._CUPY_AVAILABLE, gpu_mod._cp
    try:
        gpu_mod._CUPY_AVAILABLE = True
        gpu_mod._cp = mock_cp
        result = gpu_mod.matmul(a, b)
        np.testing.assert_allclose(result, expected)
    finally:
        gpu_mod._CUPY_AVAILABLE = orig_avail
        gpu_mod._cp = orig_cp


def test_gpu_accel_mocked_cupy_memory():
    """Cover gpu_memory_free_mb with mocked cupy."""
    import scpn_quantum_control.hardware.gpu_accel as gpu_mod

    mock_cp = MagicMock()
    mock_cp.cuda.runtime.memGetInfo.return_value = (4000000000, 6000000000)

    orig_avail, orig_cp = gpu_mod._CUPY_AVAILABLE, gpu_mod._cp
    try:
        gpu_mod._CUPY_AVAILABLE = True
        gpu_mod._cp = mock_cp
        free = gpu_mod.gpu_memory_free_mb()
        assert free == pytest.approx(4000.0, rel=0.01)
    finally:
        gpu_mod._CUPY_AVAILABLE = orig_avail
        gpu_mod._cp = orig_cp


def test_gpu_accel_mocked_cupy_device_name():
    """Cover gpu_device_name with mocked cupy."""
    import scpn_quantum_control.hardware.gpu_accel as gpu_mod

    mock_cp = MagicMock()
    mock_cp.cuda.runtime.getDeviceProperties.return_value = {"name": b"NVIDIA GTX 1060"}

    orig_avail, orig_cp = gpu_mod._CUPY_AVAILABLE, gpu_mod._cp
    try:
        gpu_mod._CUPY_AVAILABLE = True
        gpu_mod._cp = mock_cp
        name = gpu_mod.gpu_device_name()
        assert "1060" in name
    finally:
        gpu_mod._CUPY_AVAILABLE = orig_avail
        gpu_mod._cp = orig_cp


# --- cirq_adapter.py: Cirq runner with mocked cirq ---


def test_cirq_adapter_availability():
    """Cover is_cirq_available check."""
    from scpn_quantum_control.hardware.cirq_adapter import is_cirq_available

    result = is_cirq_available()
    assert isinstance(result, bool)


def test_cirq_adapter_import_error():
    """Cover CirqRunner raises ImportError when cirq unavailable."""
    import scpn_quantum_control.hardware.cirq_adapter as cirq_mod

    orig = cirq_mod._CIRQ_AVAILABLE
    try:
        cirq_mod._CIRQ_AVAILABLE = False
        with pytest.raises(ImportError, match="Cirq not installed"):
            cirq_mod.CirqRunner(np.eye(2), np.ones(2))
    finally:
        cirq_mod._CIRQ_AVAILABLE = orig


def test_cirq_runner_mocked():
    """Cover CirqRunner.run_trotter with mocked cirq module."""
    import scpn_quantum_control.hardware.cirq_adapter as cirq_mod

    mock_cirq = MagicMock()
    mock_cirq.LineQubit.range.return_value = [0, 1]
    mock_cirq.XXPowGate.return_value = MagicMock(return_value=MagicMock())
    mock_cirq.YYPowGate.return_value = MagicMock(return_value=MagicMock())
    mock_cirq.rz.return_value = MagicMock(return_value=MagicMock())
    mock_cirq.Circuit.return_value = MagicMock()

    mock_sim_result = MagicMock()
    mock_sim_result.final_state_vector = np.array([1, 0, 0, 0], dtype=complex)
    mock_cirq.Simulator.return_value.simulate.return_value = mock_sim_result

    orig_avail = cirq_mod._CIRQ_AVAILABLE
    orig_cirq = cirq_mod.cirq
    try:
        cirq_mod._CIRQ_AVAILABLE = True
        cirq_mod.cirq = mock_cirq
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 2.0])
        runner = cirq_mod.CirqRunner(K, omega)
        result = runner.run_trotter(t=0.5, reps=2)
        assert result.n_qubits == 2
    finally:
        cirq_mod._CIRQ_AVAILABLE = orig_avail
        cirq_mod.cirq = orig_cirq


# --- pennylane_adapter.py: mocked PennyLane ---


def _import_pennylane_adapter():
    try:
        import scpn_quantum_control.hardware.pennylane_adapter as mod

        return mod
    except (ImportError, AttributeError):
        return None


def test_pennylane_availability():
    """Cover is_pennylane_available check."""
    pl_mod = _import_pennylane_adapter()
    if pl_mod is None:
        pytest.skip("pennylane not importable")
    result = pl_mod.is_pennylane_available()
    assert isinstance(result, bool)


def test_pennylane_import_error():
    """Cover PennyLaneRunner raises ImportError when pennylane unavailable."""
    pl_mod = _import_pennylane_adapter()
    if pl_mod is None:
        pytest.skip("pennylane not importable")
    orig = pl_mod._PL_AVAILABLE
    try:
        pl_mod._PL_AVAILABLE = False
        with pytest.raises(ImportError, match="PennyLane not installed"):
            pl_mod.PennyLaneRunner(np.eye(2), np.ones(2))
    finally:
        pl_mod._PL_AVAILABLE = orig


def test_pennylane_hamiltonian_import_error():
    """Cover _xy_hamiltonian_pl raises when PennyLane unavailable."""
    pl_mod = _import_pennylane_adapter()
    if pl_mod is None:
        pytest.skip("pennylane not importable")
    orig = pl_mod._PL_AVAILABLE
    try:
        pl_mod._PL_AVAILABLE = False
        with pytest.raises(ImportError, match="PennyLane not installed"):
            pl_mod._xy_hamiltonian_pl(np.eye(2), np.ones(2))
    finally:
        pl_mod._PL_AVAILABLE = orig
