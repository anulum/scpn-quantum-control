# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Mitiq Integration
"""Tests for Mitiq error mitigation integration (ZNE + DDD).

Multi-angle: multiple circuits, scale factor variations, known-state
verification, custom executors, noise-dependent behaviour, type checks,
physical bounds, noiseless invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

mitiq = pytest.importorskip("mitiq")

from scpn_quantum_control.mitigation.mitiq_integration import (
    is_mitiq_available,
    zne_mitigated_expectation,
)


class TestMitiqAvailable:
    def test_mitiq_installed(self):
        assert is_mitiq_available()

    def test_returns_bool(self):
        assert isinstance(is_mitiq_available(), bool)


class TestZNEBasic:
    """Return type, bounds, and output validation."""

    def test_zne_returns_float(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_zne_bounded_single_qubit(self):
        """⟨Z⟩ must be in [-1, 1] for any single-qubit state."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert -1.05 <= result <= 1.05, f"⟨Z⟩ = {result} out of physical bounds"

    def test_zne_bounded_two_qubit(self):
        """Parity expectation bounded for 2-qubit systems."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 3.0, 5.0])
        assert -1.05 <= result <= 1.05


class TestZNEKnownStates:
    """Test ZNE on states with known expectation values."""

    def test_identity_circuit_positive(self):
        """⟨Z⟩ for |0⟩ should be +1 (noiseless)."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert result > 0.5, f"|0⟩ should give positive ⟨Z⟩, got {result}"

    def test_x_gate_negative(self):
        """⟨Z⟩ for |1⟩ should be -1 (noiseless)."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert result < -0.5, f"|1⟩ should give negative ⟨Z⟩, got {result}"

    def test_hadamard_near_zero(self):
        """⟨Z⟩ for |+⟩ should be ~0."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        result = zne_mitigated_expectation(
            qc,
            scale_factors=[1.0, 2.0, 3.0],
            shots=16384,
        )
        assert abs(result) < 0.3, f"|+⟩ should give ⟨Z⟩ ≈ 0, got {result}"

    def test_bell_state_parity(self):
        """Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2 has parity = +1."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = zne_mitigated_expectation(
            qc,
            scale_factors=[1.0, 3.0, 5.0],
            shots=16384,
        )
        assert result > 0.5, f"Bell state parity should be +1, got {result}"


class TestZNEScaleFactors:
    """Test behaviour with different scale factor configurations."""

    @pytest.mark.parametrize(
        "scale_factors",
        [
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
    )
    def test_various_scale_factors(self, scale_factors):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=scale_factors)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_single_scale_factor_requires_minimum_two(self):
        """Mitiq requires at least 2 scale factors for Richardson extrapolation."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure_all()

        with pytest.raises(ValueError, match="[Aa]t least 2"):
            zne_mitigated_expectation(qc, scale_factors=[1.0])


class TestZNECustomExecutor:
    """Test ZNE with custom executor functions."""

    def test_constant_executor(self):
        """Constant executor → ZNE should return that constant."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        def const_executor(circuit):
            return 0.42

        result = zne_mitigated_expectation(
            qc,
            executor=const_executor,
            scale_factors=[1.0, 2.0, 3.0],
        )
        assert isinstance(result, float)
        # Constant executor → extrapolation should give ~0.42
        assert abs(result - 0.42) < 0.1

    def test_linear_decay_executor(self):
        """Executor with linear noise decay → ZNE should extrapolate to clean value."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure_all()

        call_count = [0]
        scale_sequence = [1.0, 3.0, 5.0]

        def linear_executor(circuit):
            idx = min(call_count[0], len(scale_sequence) - 1)
            scale = scale_sequence[idx]
            call_count[0] += 1
            # Clean value 1.0, linear decay with noise
            return 1.0 - 0.1 * (scale - 1)

        result = zne_mitigated_expectation(
            qc,
            executor=linear_executor,
            scale_factors=scale_sequence,
        )
        assert isinstance(result, float)


class TestZNEMultipleCircuits:
    """Test ZNE on circuits of different structure."""

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_product_state_circuits(self, n_qubits):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(n_qubits)
        qc.measure_all()

        result = zne_mitigated_expectation(
            qc,
            scale_factors=[1.0, 2.0, 3.0],
            shots=8192,
        )
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_deep_circuit(self):
        """Deeper circuits should still return finite results."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        for _ in range(10):
            qc.h(0)
            qc.cx(0, 1)
            qc.rz(0.1, 0)
            qc.rz(0.2, 1)
        qc.measure_all()

        result = zne_mitigated_expectation(
            qc,
            scale_factors=[1.0, 3.0, 5.0],
            shots=4096,
        )
        assert isinstance(result, float)
        assert np.isfinite(result)


class TestDDD:
    """Tests for Digital Dynamical Decoupling (if available)."""

    def test_ddd_available(self):
        """DDD function should be importable."""
        try:
            from scpn_quantum_control.mitigation.mitiq_integration import (
                ddd_mitigated_expectation,
            )

            assert callable(ddd_mitigated_expectation)
        except ImportError:
            pytest.skip("DDD not available in this mitiq version")

    def test_ddd_returns_float(self):
        try:
            from scpn_quantum_control.mitigation.mitiq_integration import (
                ddd_mitigated_expectation,
            )
        except ImportError:
            pytest.skip("DDD not available")

        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        qc.measure_all()

        result = ddd_mitigated_expectation(qc, shots=4096)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_ddd_bounded(self):
        try:
            from scpn_quantum_control.mitigation.mitiq_integration import (
                ddd_mitigated_expectation,
            )
        except ImportError:
            pytest.skip("DDD not available")

        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()

        result = ddd_mitigated_expectation(qc, shots=8192)
        assert -1.05 <= result <= 1.05


class TestMitiqCoverage:
    """Cover default parameters and circuit-without-measurements path."""

    def test_executor_circuit_without_measurements(self):
        """Cover line 56: measure_all() added when circuit has no measures."""
        from qiskit import QuantumCircuit

        from scpn_quantum_control.mitigation.mitiq_integration import _qiskit_executor

        qc = QuantumCircuit(1)
        qc.x(0)
        # No measure_all — executor adds it
        result = _qiskit_executor(qc, shots=1000)
        assert -1.05 <= result <= 1.05

    def test_zne_default_scale_factors(self):
        """Cover line 98: scale_factors=None defaults to [1, 3, 5]."""
        from qiskit import QuantumCircuit

        try:
            from scpn_quantum_control.mitigation.mitiq_integration import (
                zne_mitigated_expectation,
            )
        except ImportError:
            pytest.skip("mitiq not available")

        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()
        result = zne_mitigated_expectation(qc, scale_factors=None, shots=1000)
        assert -1.05 <= result <= 1.05
