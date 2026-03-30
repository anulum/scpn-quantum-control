# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for New Module Properties
"""Property-based (hypothesis) tests for v1.0 modules."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# --- PEC properties ---


@given(p=st.floats(min_value=0.001, max_value=0.49))
@settings(max_examples=30)
def test_pec_coefficients_sum_to_one(p: float) -> None:
    """Quasi-probability coefficients sum to 1 (trace preservation)."""
    from scpn_quantum_control.mitigation.pec import pauli_twirl_decompose

    coeffs = pauli_twirl_decompose(p)
    assert abs(float(np.sum(coeffs)) - 1.0) < 1e-10


@given(p=st.floats(min_value=0.001, max_value=0.49))
@settings(max_examples=30)
def test_pec_overhead_increases_with_error(p: float) -> None:
    """Higher error rate -> higher sampling overhead."""
    from scpn_quantum_control.mitigation.pec import pauli_twirl_decompose

    coeffs = pauli_twirl_decompose(p)
    gamma = float(np.sum(np.abs(coeffs)))
    assert gamma >= 1.0  # overhead >= 1 always
    if p > 0.01:
        coeffs_low = pauli_twirl_decompose(p / 2)
        gamma_low = float(np.sum(np.abs(coeffs_low)))
        assert gamma > gamma_low


# --- Trapped-ion properties ---


@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=10, deadline=30000)
def test_trapped_ion_transpile_preserves_qubit_count(n: int) -> None:
    """Transpilation preserves qubit count."""
    from qiskit import QuantumCircuit

    from scpn_quantum_control.hardware.trapped_ion import transpile_for_trapped_ion

    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(0.5, i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    result = transpile_for_trapped_ion(qc)
    assert result.num_qubits == n


# --- ITER normalization properties ---


@given(
    raw=st.lists(
        st.floats(min_value=-10, max_value=500, allow_nan=False, allow_infinity=False),
        min_size=11,
        max_size=11,
    )
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_iter_normalized_in_unit_interval(raw: list[float]) -> None:
    """Normalized ITER features always in [0, 1]."""
    from scpn_quantum_control.control.q_disruption_iter import normalize_iter_features

    normed = normalize_iter_features(np.array(raw))
    assert np.all(normed >= 0.0)
    assert np.all(normed <= 1.0)


# --- Quantum advantage properties ---


@given(n=st.sampled_from([4, 6, 8]))
@settings(max_examples=6, deadline=60000)
def test_advantage_quantum_always_finite(n: int) -> None:
    """Quantum benchmark always returns finite time."""
    from scpn_quantum_control.benchmarks.quantum_advantage import quantum_benchmark

    result = quantum_benchmark(n, t_max=0.2, dt=0.1)
    assert np.isfinite(result["t_total_ms"])
    assert result["t_total_ms"] > 0


# --- Bridge roundtrip properties ---


@given(
    theta=st.lists(
        st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False),
        min_size=3,
        max_size=3,
    )
)
@settings(max_examples=20, deadline=10000)
def test_ssgf_phase_roundtrip(theta: list[float]) -> None:
    """SSGF encode -> statevector -> decode recovers phases (mod 2pi)."""
    from qiskit.quantum_info import Statevector

    from scpn_quantum_control.bridge.ssgf_adapter import (
        quantum_to_ssgf_state,
        ssgf_state_to_quantum,
    )

    qc = ssgf_state_to_quantum({"theta": np.array(theta)})
    sv = Statevector.from_instruction(qc)
    recovered = quantum_to_ssgf_state(sv, 3)
    diff = np.angle(np.exp(1j * (recovered["theta"] - np.array(theta))))
    np.testing.assert_allclose(diff, 0.0, atol=1e-6)


@given(window=st.integers(min_value=1, max_value=50))
@settings(max_examples=20)
def test_snn_rotations_in_valid_range(window: int) -> None:
    """Spike-to-rotation angles always in [0, pi]."""
    from scpn_quantum_control.bridge.snn_adapter import spike_train_to_rotations

    rng = np.random.default_rng(42)
    spikes = (rng.random((max(window, 5), 4)) > 0.5).astype(float)
    angles = spike_train_to_rotations(spikes, window=window)
    assert np.all(angles >= 0.0)
    assert np.all(angles <= np.pi + 1e-10)


# --- Surface code properties ---


@given(d=st.sampled_from([3, 5, 7]))
@settings(max_examples=3)
def test_surface_code_qubit_formula(d: int) -> None:
    """Physical qubits = n_osc * (2d² - 1)."""
    from scpn_quantum_control.qec.surface_code_upde import SurfaceCodeUPDE

    sc = SurfaceCodeUPDE(n_osc=2, code_distance=d)
    assert sc.total_qubits == 2 * (2 * d * d - 1)


# --- Orchestrator mapping properties ---


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=20)
def test_orchestrator_roundtrip_modular(seed: int) -> None:
    """Phase mapping roundtrip preserves phases mod 2pi."""
    from scpn_quantum_control.identity.binding_spec import (
        orchestrator_to_quantum_phases,
        quantum_to_orchestrator_phases,
    )

    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, 18)
    orch = quantum_to_orchestrator_phases(theta)
    back = orchestrator_to_quantum_phases(orch)
    diff = np.angle(np.exp(1j * (back - theta)))
    np.testing.assert_allclose(diff, 0.0, atol=1e-10)
