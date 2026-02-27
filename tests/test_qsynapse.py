"""Tests for qsnn/qsynapse.py."""
import numpy as np
import pytest

from qiskit import QuantumCircuit
from scpn_quantum_control.qsnn.qsynapse import QuantumSynapse


def test_theta_range():
    syn = QuantumSynapse(0.0)
    assert syn.theta == pytest.approx(0.0)
    syn = QuantumSynapse(1.0)
    assert syn.theta == pytest.approx(np.pi)


def test_effective_weight_matches_probability():
    syn = QuantumSynapse(0.5)
    theta = syn.theta
    expected = np.sin(theta / 2.0) ** 2
    assert syn.effective_weight() == pytest.approx(expected)


def test_apply_adds_cry_gate():
    syn = QuantumSynapse(0.7)
    qc = QuantumCircuit(2)
    syn.apply(qc, 0, 1)
    ops = [inst.operation.name for inst in qc.data]
    assert "cry" in ops


def test_weight_clipping():
    syn = QuantumSynapse(2.0, w_min=0.0, w_max=1.0)
    assert syn.weight == 1.0
    syn.update_weight(-0.5)
    assert syn.weight == 0.0


def test_zero_weight_no_rotation():
    syn = QuantumSynapse(0.0)
    assert syn.theta == pytest.approx(0.0)
    assert syn.effective_weight() == pytest.approx(0.0)
