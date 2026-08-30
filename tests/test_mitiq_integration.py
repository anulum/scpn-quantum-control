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

import builtins
import importlib
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit

import scpn_quantum_control.mitigation.mitiq_integration as mitiq_module
from scpn_quantum_control.mitigation.mitiq_integration import (
    _qiskit_executor,
    _qiskit_executor_default,
    ddd_mitigated_expectation,
    is_mitiq_available,
    zne_mitigated_expectation,
)

MITIQ_AVAILABLE = is_mitiq_available()
requires_mitiq = pytest.mark.skipif(not MITIQ_AVAILABLE, reason="mitiq is not installed")


class TestMitiqAvailable:
    """Verify the optional-dependency capability signal."""

    def test_capability_matches_module_state(self) -> None:
        """Expose the import outcome through the public capability helper."""
        assert is_mitiq_available() is mitiq_module._MITIQ_AVAILABLE

    def test_returns_bool(self) -> None:
        """Return a strict boolean capability value."""
        assert isinstance(is_mitiq_available(), bool)


@requires_mitiq
class TestZNEBasic:
    """Return type, bounds, and output validation."""

    def test_zne_returns_float(self) -> None:
        """Return a finite float for a measured Bell circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_zne_bounded_single_qubit(self) -> None:
        """⟨Z⟩ must be in [-1, 1] for any single-qubit state."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert -1.05 <= result <= 1.05, f"⟨Z⟩ = {result} out of physical bounds"

    def test_zne_bounded_two_qubit(self) -> None:
        """Parity expectation bounded for 2-qubit systems."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 3.0, 5.0])
        assert -1.05 <= result <= 1.05


@requires_mitiq
class TestZNEKnownStates:
    """Test ZNE on states with known expectation values."""

    def test_identity_circuit_positive(self) -> None:
        """⟨Z⟩ for |0⟩ should be +1 (noiseless)."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert result > 0.5, f"|0⟩ should give positive ⟨Z⟩, got {result}"

    def test_x_gate_negative(self) -> None:
        """⟨Z⟩ for |1⟩ should be -1 (noiseless)."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert result < -0.5, f"|1⟩ should give negative ⟨Z⟩, got {result}"

    def test_hadamard_near_zero(self) -> None:
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

    def test_bell_state_parity(self) -> None:
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


@requires_mitiq
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
    def test_various_scale_factors(self, scale_factors: list[float]) -> None:
        """Accept each supported Richardson scale-factor cohort."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=scale_factors)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_single_scale_factor_requires_minimum_two(self) -> None:
        """Mitiq requires at least 2 scale factors for Richardson extrapolation."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure_all()

        with pytest.raises(ValueError, match="[Aa]t least 2"):
            zne_mitigated_expectation(qc, scale_factors=[1.0])


@requires_mitiq
class TestZNECustomExecutor:
    """Test ZNE with custom executor functions."""

    def test_constant_executor(self) -> None:
        """Constant executor → ZNE should return that constant."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        def const_executor(circuit: QuantumCircuit) -> float:
            return 0.42

        result = zne_mitigated_expectation(
            qc,
            executor=const_executor,
            scale_factors=[1.0, 2.0, 3.0],
        )
        assert isinstance(result, float)
        # Constant executor → extrapolation should give ~0.42
        assert abs(result - 0.42) < 0.1

    def test_linear_decay_executor(self) -> None:
        """Executor with linear noise decay → ZNE should extrapolate to clean value."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure_all()

        call_count = [0]
        scale_sequence = [1.0, 3.0, 5.0]

        def linear_executor(circuit: QuantumCircuit) -> float:
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


@requires_mitiq
class TestZNEMultipleCircuits:
    """Test ZNE on circuits of different structure."""

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_product_state_circuits(self, n_qubits: int) -> None:
        """Return finite values across several product-state widths."""
        qc = QuantumCircuit(n_qubits)
        qc.measure_all()

        result = zne_mitigated_expectation(
            qc,
            scale_factors=[1.0, 2.0, 3.0],
            shots=8192,
        )
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_deep_circuit(self) -> None:
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


@requires_mitiq
class TestDDD:
    """Tests for Digital Dynamical Decoupling (if available)."""

    def test_ddd_available(self) -> None:
        """DDD function should be importable."""
        try:
            from scpn_quantum_control.mitigation.mitiq_integration import (
                ddd_mitigated_expectation,
            )

            assert callable(ddd_mitigated_expectation)
        except ImportError:
            pytest.skip("DDD not available in this mitiq version")

    def test_ddd_returns_float(self) -> None:
        """Return a finite float for a decoupled Bell circuit."""
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

    def test_ddd_bounded(self) -> None:
        """Keep a decoupled one-qubit result within physical bounds."""
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
    """Cover deterministic adapters and optional-dependency boundaries."""

    def test_executor_circuit_without_measurements(self) -> None:
        """Add measurements before executing a circuit that has none."""
        qc = QuantumCircuit(1)
        qc.x(0)
        result = _qiskit_executor(qc, shots=1000)
        assert -1.05 <= result <= 1.05

    def test_executor_preserves_existing_measurements(self) -> None:
        """Execute a circuit with an existing measurement unchanged."""
        qc = QuantumCircuit(1)
        qc.measure_all()
        result = _qiskit_executor(qc, shots=1000)
        assert result == pytest.approx(1.0)

    def test_default_executor_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Forward a circuit through the single-argument Mitiq wrapper."""
        monkeypatch.setattr(mitiq_module, "_qiskit_executor", lambda circuit: 0.75)
        assert _qiskit_executor_default(QuantumCircuit(1)) == pytest.approx(0.75)

    @requires_mitiq
    def test_zne_default_scale_factors(self) -> None:
        """Use the documented Richardson scale factors by default."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()
        result = zne_mitigated_expectation(qc, scale_factors=None, shots=1000)
        assert -1.05 <= result <= 1.05

    def test_unavailable_mitiq_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reject ZNE and DDD calls when the optional package is absent."""
        monkeypatch.setattr(mitiq_module, "_MITIQ_AVAILABLE", False)
        with pytest.raises(ImportError, match="mitiq not installed"):
            zne_mitigated_expectation(QuantumCircuit(1))
        with pytest.raises(ImportError, match="mitiq not installed"):
            ddd_mitigated_expectation(QuantumCircuit(1))

    def test_unavailable_qiskit_executor_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject default execution when Qiskit Aer is unavailable."""
        monkeypatch.setattr(mitiq_module, "_QISKIT_AVAILABLE", False)
        with pytest.raises(ImportError, match="qiskit-aer required"):
            _qiskit_executor(QuantumCircuit(1))

    def test_default_zne_executor_forwards_requested_shots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forward requested shots through the default ZNE executor."""
        calls: list[int] = []

        def fake_executor(circuit: QuantumCircuit, shots: int = 8192) -> float:
            calls.append(shots)
            return 0.25

        def fake_execute_with_zne(
            circuit: QuantumCircuit,
            executor: Callable[[QuantumCircuit], float],
            factory: Any,
        ) -> float:
            return executor(circuit)

        fake_zne = SimpleNamespace(
            inference=SimpleNamespace(RichardsonFactory=lambda scale_factors: scale_factors),
            execute_with_zne=fake_execute_with_zne,
        )
        monkeypatch.setattr(mitiq_module, "_MITIQ_AVAILABLE", True)
        monkeypatch.setattr(mitiq_module, "_qiskit_executor", fake_executor)
        monkeypatch.setattr(mitiq_module, "zne", fake_zne, raising=False)

        result = zne_mitigated_expectation(
            QuantumCircuit(1),
            scale_factors=[1.0, 3.0],
            shots=1234,
        )

        assert result == pytest.approx(0.25)
        assert calls == [1234]

    def test_custom_zne_executor_and_default_scale_factors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pass a custom executor and construct default scale factors."""
        observed_scales: list[list[float]] = []

        def build_factory(scale_factors: list[float]) -> list[float]:
            observed_scales.append(scale_factors)
            return scale_factors

        fake_zne = SimpleNamespace(
            inference=SimpleNamespace(RichardsonFactory=build_factory),
            execute_with_zne=lambda circuit, executor, factory: executor(circuit),
        )
        monkeypatch.setattr(mitiq_module, "_MITIQ_AVAILABLE", True)
        monkeypatch.setattr(mitiq_module, "zne", fake_zne, raising=False)

        result = zne_mitigated_expectation(
            QuantumCircuit(1),
            executor=lambda circuit: -0.25,
            scale_factors=None,
        )

        assert result == pytest.approx(-0.25)
        assert observed_scales == [[1.0, 3.0, 5.0]]

    def test_default_ddd_executor_forwards_requested_shots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forward requested shots through the default DDD executor."""
        calls: list[int] = []

        def fake_executor(circuit: QuantumCircuit, shots: int = 8192) -> float:
            calls.append(shots)
            return -0.5

        def fake_execute_with_ddd(
            circuit: QuantumCircuit,
            executor: Callable[[QuantumCircuit], float],
            rule: object,
        ) -> float:
            assert rule == "xx-rule"
            return executor(circuit)

        fake_ddd = SimpleNamespace(
            rules=SimpleNamespace(xx="xx-rule"),
            execute_with_ddd=fake_execute_with_ddd,
        )
        monkeypatch.setattr(mitiq_module, "_MITIQ_AVAILABLE", True)
        monkeypatch.setattr(mitiq_module, "_qiskit_executor", fake_executor)
        monkeypatch.setattr(mitiq_module, "ddd", fake_ddd, raising=False)

        result = ddd_mitigated_expectation(QuantumCircuit(1), shots=4321)

        assert result == pytest.approx(-0.5)
        assert calls == [4321]

    def test_custom_ddd_executor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pass a custom executor and the Mitiq XX decoupling rule."""
        observed_rules: list[object] = []

        def fake_execute_with_ddd(
            circuit: QuantumCircuit,
            executor: Callable[[QuantumCircuit], float],
            rule: object,
        ) -> float:
            observed_rules.append(rule)
            return executor(circuit)

        fake_ddd = SimpleNamespace(
            rules=SimpleNamespace(xx="xx-rule"),
            execute_with_ddd=fake_execute_with_ddd,
        )
        monkeypatch.setattr(mitiq_module, "_MITIQ_AVAILABLE", True)
        monkeypatch.setattr(mitiq_module, "ddd", fake_ddd, raising=False)

        result = ddd_mitigated_expectation(QuantumCircuit(1), executor=lambda circuit: 0.125)

        assert result == pytest.approx(0.125)
        assert observed_rules == ["xx-rule"]

    def test_module_import_capability_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercise successful and failed optional Mitiq imports."""
        fake_mitiq = ModuleType("mitiq")
        fake_mitiq.__dict__["ddd"] = SimpleNamespace()
        fake_mitiq.__dict__["zne"] = SimpleNamespace()
        monkeypatch.setitem(sys.modules, "mitiq", fake_mitiq)
        assert importlib.reload(mitiq_module)._MITIQ_AVAILABLE is True

        monkeypatch.delitem(sys.modules, "mitiq")
        assert importlib.reload(mitiq_module)._MITIQ_AVAILABLE is False

    def test_module_import_without_qiskit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercise the Qiskit optional-import failure path."""
        real_import = builtins.__import__

        def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "qiskit" or name.startswith("qiskit_aer"):
                raise ImportError("blocked for capability test")
            return real_import(name, *args, **kwargs)

        with monkeypatch.context() as import_patch:
            import_patch.setattr(builtins, "__import__", blocked_import)
            assert importlib.reload(mitiq_module)._QISKIT_AVAILABLE is False

        assert importlib.reload(mitiq_module)._QISKIT_AVAILABLE is True
