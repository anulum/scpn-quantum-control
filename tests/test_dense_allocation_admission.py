# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Dense Allocation Admission Tests
"""Public exponential allocations refuse before they enter the allocator.

Four public entry points simulated or built a ``2**n`` object with no admission
check: ``QAOA_MPC.optimize``, ``QSNNTrainer``, ``RepetitionCodeUPDE.step_with_qec``
and ``build_readout_confusion_matrix``. A caller-controlled horizon, layer width,
code distance or qubit count therefore reached the allocator directly.

Every refusal here is checked with an allocation spy: the allocator each path
would call is replaced with a function that fails if it is reached at all. A
test that only asserted the exception could not tell a check that refuses early
from one that allocates first and raises afterwards, which is the whole point of
the contract. No test provokes a real large allocation; a tiny configured budget
does the work instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.control import qaoa_mpc as qaoa_mpc_module
from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.dense_budget import GIB, DenseAllocationError
from scpn_quantum_control.mitigation import readout_matrix as readout_module
from scpn_quantum_control.mitigation.readout_matrix import build_readout_confusion_matrix
from scpn_quantum_control.qec import fault_tolerant as fault_tolerant_module
from scpn_quantum_control.qec.fault_tolerant import RepetitionCodeUPDE
from scpn_quantum_control.qsnn import training as qsnn_training_module
from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer
from scpn_quantum_control.qsnn.training import QSNNTrainer

TINY_BUDGET_GIB = 1e-12
"""Budget smaller than any admissible object, used instead of a real big one."""


class _AllocationSpy:
    """Stand-in allocator that fails if the guarded path ever reaches it."""

    def __init__(self, name: str) -> None:
        """Record which allocator this spy replaces.

        Parameters
        ----------
        name
            Human-readable name of the replaced allocator.

        """
        self.name = name
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Fail the test, recording that the allocator was entered.

        Parameters
        ----------
        *args, **kwargs
            Ignored; the call itself is the failure.

        Returns
        -------
        Any
            Never returns.

        """
        self.calls += 1
        raise AssertionError(f"{self.name} was entered despite an inadmissible budget")


def _calibration_counts(n_qubits: int) -> dict[str, dict[str, int]]:
    """Return a perfect-readout calibration set for ``n_qubits``.

    Parameters
    ----------
    n_qubits
        Number of measured qubits.

    Returns
    -------
    dict
        Prepared-state label to observed counts, one prepared state per basis
        label with all shots on the diagonal.

    """
    return {
        format(index, f"0{n_qubits}b"): {format(index, f"0{n_qubits}b"): 100}
        for index in range(2**n_qubits)
    }


class TestQaoaMpcAdmission:
    """The confirmed direct bypass: a user-controlled horizon sets 2**horizon."""

    def test_tiny_budget_refuses(self) -> None:
        """The public call names its own allocation in the refusal."""
        controller = QAOA_MPC(np.eye(2), np.array([1.0, 0.0]), horizon=3)

        with pytest.raises(DenseAllocationError, match="QAOA-MPC statevector"):
            controller.optimize(seed=1, max_dense_gib=TINY_BUDGET_GIB)

    def test_refusal_precedes_the_allocator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """COBYLA calls the cost function up to two hundred times.

        The check therefore has to run once before the optimiser starts, not
        inside the cost function.

        Parameters
        ----------
        monkeypatch
            Used to install the allocation spy.

        """
        spy = _AllocationSpy("Statevector.from_instruction")
        monkeypatch.setattr(qaoa_mpc_module, "Statevector", spy)
        controller = QAOA_MPC(np.eye(2), np.array([1.0, 0.0]), horizon=3)

        with pytest.raises(DenseAllocationError):
            controller.optimize(seed=1, max_dense_gib=TINY_BUDGET_GIB)

        assert spy.calls == 0

    def test_accounting_covers_two_simultaneous_vectors(self) -> None:
        """A budget that holds one statevector but not two must still refuse.

        The state is live while the expectation value, and later
        ``probabilities()``, build a second array of the same dimension. A
        budget sized for a single vector is therefore not enough, and this
        distinguishes the recorded accounting from ``object_count=1``.
        """
        one_vector_bytes = (2**3) * np.dtype(np.complex128).itemsize
        between = (one_vector_bytes + 20) / GIB
        controller = QAOA_MPC(np.eye(2), np.array([1.0, 0.0]), horizon=3)

        with pytest.raises(DenseAllocationError, match="2 objects"):
            controller.optimize(seed=1, max_dense_gib=between)

    def test_admissible_horizon_still_optimises(self) -> None:
        """A small admissible problem is unchanged by the guard."""
        controller = QAOA_MPC(np.eye(2), np.array([1.0, 0.0]), horizon=3)

        actions = controller.optimize(seed=7)

        assert actions.shape == (3,)
        assert set(np.unique(actions)).issubset({0, 1})


class TestQsnnAdmission:
    """Every trainer entry point goes through the constructor."""

    def test_runtime_budget_reduction_refuses_before_circuit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An earlier construction permit does not override the current process budget."""
        trainer = QSNNTrainer(QuantumDenseLayer(1, 1, seed=3))
        spy = _AllocationSpy("QuantumCircuit")
        monkeypatch.setattr(qsnn_training_module, "QuantumCircuit", spy)
        monkeypatch.setenv("SCPN_MAX_DENSE_GIB", str(TINY_BUDGET_GIB))
        with pytest.raises(DenseAllocationError):
            trainer.train_epoch(np.array([[0.5]]), np.array([[1.0]]))
        assert spy.calls == 0

    def test_replaced_layer_is_rechecked_against_explicit_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A larger replacement cannot reuse admission for the original layer."""
        trainer = QSNNTrainer(QuantumDenseLayer(1, 1), max_dense_gib=256 / GIB)
        trainer.layer = QuantumDenseLayer(2, 2)
        spy = _AllocationSpy("QuantumCircuit")
        monkeypatch.setattr(qsnn_training_module, "QuantumCircuit", spy)
        with pytest.raises(DenseAllocationError):
            trainer.train_epoch(np.array([[0.5, 0.2]]), np.array([[1.0, 0.0]]))
        assert spy.calls == 0

    def test_probability_workspace_is_included(self) -> None:
        """State plus absolute-value and square buffers exceed a single-vector budget."""
        with pytest.raises(DenseAllocationError, match="2 objects"):
            QSNNTrainer(QuantumDenseLayer(1, 1), max_dense_gib=80 / GIB)

    def test_direct_layer_forward_refuses_tiny_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The direct NumPy layer path also admits its exponential state before allocation."""
        layer = QuantumDenseLayer(1, 1)
        inputs = np.array([0.5])
        monkeypatch.setenv("SCPN_MAX_DENSE_GIB", str(TINY_BUDGET_GIB))
        spy = _AllocationSpy("numpy.zeros")
        monkeypatch.setattr(np, "zeros", spy)
        with pytest.raises(DenseAllocationError):
            layer.forward(inputs)
        assert spy.calls == 0

    def test_explicit_forward_budget_preserves_the_real_numpy_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An admitted zero-synapse layer emits no spike even with encoded input one."""
        layer = QuantumDenseLayer(1, 1, weights=np.zeros((1, 1)))
        monkeypatch.setenv("SCPN_MAX_DENSE_GIB", str(TINY_BUDGET_GIB))
        spikes = layer.forward(np.ones(1), max_dense_gib=64 / GIB)
        np.testing.assert_array_equal(spikes, [0])

    def test_tiny_budget_refuses_at_construction(self) -> None:
        """An inadmissible layer is refused before a run starts."""
        layer = QuantumDenseLayer(n_inputs=2, n_neurons=2)

        with pytest.raises(DenseAllocationError, match="QSNN forward-pass statevector"):
            QSNNTrainer(layer, max_dense_gib=TINY_BUDGET_GIB)

    def test_refusal_precedes_the_allocator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The forward pass is never reached.

        Parameters
        ----------
        monkeypatch
            Used to install the allocation spy.

        """
        spy = _AllocationSpy("Statevector.from_instruction")
        monkeypatch.setattr(qsnn_training_module, "Statevector", spy)
        layer = QuantumDenseLayer(n_inputs=2, n_neurons=2)

        with pytest.raises(DenseAllocationError):
            QSNNTrainer(layer, max_dense_gib=TINY_BUDGET_GIB)

        assert spy.calls == 0

    def test_admissible_layer_still_trains(self) -> None:
        """A small admissible layer trains as before."""
        layer = QuantumDenseLayer(n_inputs=2, n_neurons=2)
        trainer = QSNNTrainer(layer, lr=0.05)
        features = np.array([[0.0, 1.0], [1.0, 0.0]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])

        loss = trainer.train_epoch(features, targets)

        assert np.isfinite(loss)


class TestQecAdmission:
    """The protected circuit carries data and ancilla qubits together."""

    def test_probability_workspace_is_included(self) -> None:
        """The retained state and probability temporaries require more than one vector."""
        code = RepetitionCodeUPDE(n_osc=2, code_distance=3)
        one_vector = 16 * (1 << code.physical_qubit_count())
        with pytest.raises(DenseAllocationError, match="2 objects"):
            code.step_with_qec(max_dense_gib=(one_vector + 32) / GIB)

    def test_tiny_budget_refuses(self) -> None:
        """The refusal names the physical-qubit statevector."""
        code = RepetitionCodeUPDE(n_osc=2, code_distance=3)

        with pytest.raises(DenseAllocationError, match="QEC protected-step statevector"):
            code.step_with_qec(max_dense_gib=TINY_BUDGET_GIB)

    def test_refusal_precedes_circuit_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Neither the circuit nor the statevector is built.

        Parameters
        ----------
        monkeypatch
            Used to install the allocation spy.

        """
        spy = _AllocationSpy("Statevector.from_instruction")
        monkeypatch.setattr(fault_tolerant_module, "Statevector", spy)
        code = RepetitionCodeUPDE(n_osc=2, code_distance=3)

        with pytest.raises(DenseAllocationError):
            code.step_with_qec(max_dense_gib=TINY_BUDGET_GIB)

        assert spy.calls == 0

    def test_the_guard_uses_the_physical_qubit_count(self) -> None:
        """Data plus ancilla qubits, not the oscillator count, set the size."""
        code = RepetitionCodeUPDE(n_osc=2, code_distance=3)

        with pytest.raises(DenseAllocationError, match=f"n={code.physical_qubit_count()}"):
            code.step_with_qec(max_dense_gib=TINY_BUDGET_GIB)

    def test_admissible_code_still_steps(self) -> None:
        """A small admissible code produces syndromes as before."""
        code = RepetitionCodeUPDE(n_osc=2, code_distance=3)

        result = code.step_with_qec()

        assert len(result["syndromes"]) == 2
        assert result["errors_detected"] >= 0


class TestReadoutMatrixAdmission:
    """The full-basis matrix is 2**n square, and its labels are 2**n long."""

    def test_tiny_budget_refuses(self) -> None:
        """The refusal names the confusion matrix."""
        with pytest.raises(DenseAllocationError, match="readout confusion matrix"):
            build_readout_confusion_matrix(
                _calibration_counts(2), 2, max_dense_gib=TINY_BUDGET_GIB
            )

    def test_refusal_precedes_label_enumeration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The label tuple is itself exponential, so it must not be built.

        Parameters
        ----------
        monkeypatch
            Used to install the allocation spy.

        """
        spy = _AllocationSpy("computational_basis_labels")
        monkeypatch.setattr(readout_module, "computational_basis_labels", spy)

        with pytest.raises(DenseAllocationError):
            build_readout_confusion_matrix(
                _calibration_counts(2), 2, max_dense_gib=TINY_BUDGET_GIB
            )

        assert spy.calls == 0

    def test_the_guard_accounts_for_a_square_matrix(self) -> None:
        """A budget that holds one row but not the matrix must refuse."""
        dimension = 2**3
        one_row_bytes = dimension * np.dtype(np.float64).itemsize
        between = (one_row_bytes + 20) / GIB

        with pytest.raises(DenseAllocationError, match=r"shape \(8, 8\)"):
            build_readout_confusion_matrix(_calibration_counts(3), 3, max_dense_gib=between)

    def test_admissible_matrix_is_unchanged(self) -> None:
        """A perfect calibration still yields the identity."""
        result = build_readout_confusion_matrix(_calibration_counts(2), 2)

        np.testing.assert_allclose(result.matrix, np.eye(4), atol=1e-12)
        assert result.labels == ("00", "01", "10", "11")
