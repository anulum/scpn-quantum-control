# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Qsynapse
"""Tests for qsnn/qsynapse.py — elite multi-angle coverage."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control.qsnn.qsynapse import QuantumSynapse

# ---------------------------------------------------------------------------
# Constructor and theta mapping
# ---------------------------------------------------------------------------


class TestQuantumSynapseInit:
    def test_theta_at_zero_weight(self):
        syn = QuantumSynapse(0.0)
        assert syn.theta == pytest.approx(0.0)

    def test_theta_at_max_weight(self):
        syn = QuantumSynapse(1.0)
        assert syn.theta == pytest.approx(np.pi)

    def test_theta_at_midpoint(self):
        syn = QuantumSynapse(0.5)
        assert syn.theta == pytest.approx(np.pi / 2)

    def test_custom_range(self):
        syn = QuantumSynapse(5.0, w_min=0.0, w_max=10.0)
        assert syn.theta == pytest.approx(np.pi / 2)

    def test_weight_clamped_above_max(self):
        syn = QuantumSynapse(2.0, w_min=0.0, w_max=1.0)
        assert syn.weight == 1.0

    def test_weight_clamped_below_min(self):
        syn = QuantumSynapse(-1.0, w_min=0.0, w_max=1.0)
        assert syn.weight == 0.0

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="must exceed"):
            QuantumSynapse(0.5, w_min=1.0, w_max=0.5)


# ---------------------------------------------------------------------------
# effective_weight
# ---------------------------------------------------------------------------


class TestEffectiveWeight:
    def test_matches_sin_squared(self):
        syn = QuantumSynapse(0.5)
        expected = np.sin(syn.theta / 2.0) ** 2
        assert syn.effective_weight() == pytest.approx(expected)

    def test_zero_weight_gives_zero(self):
        syn = QuantumSynapse(0.0)
        assert syn.effective_weight() == pytest.approx(0.0)

    def test_max_weight_gives_one(self):
        syn = QuantumSynapse(1.0)
        assert syn.effective_weight() == pytest.approx(1.0)

    @pytest.mark.parametrize("w", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_monotonically_increasing(self, w):
        """Higher weight → higher effective weight."""
        syn_lo = QuantumSynapse(w - 0.05)
        syn_hi = QuantumSynapse(w + 0.05)
        assert syn_hi.effective_weight() > syn_lo.effective_weight()

    def test_bounded_zero_one(self):
        for w in np.linspace(0, 1, 20):
            ew = QuantumSynapse(w).effective_weight()
            assert 0.0 <= ew <= 1.0


# ---------------------------------------------------------------------------
# apply (CRy gate)
# ---------------------------------------------------------------------------


class TestApply:
    def test_adds_cry_gate(self):
        syn = QuantumSynapse(0.7)
        qc = QuantumCircuit(2)
        syn.apply(qc, 0, 1)
        ops = [inst.operation.name for inst in qc.data]
        assert "cry" in ops

    def test_gate_angle_matches_theta(self):
        syn = QuantumSynapse(0.6)
        qc = QuantumCircuit(2)
        syn.apply(qc, 0, 1)
        for inst in qc.data:
            if inst.operation.name == "cry":
                assert inst.operation.params[0] == pytest.approx(syn.theta)

    def test_pre_controls_post(self):
        """When pre=|1>, post should rotate. When pre=|0>, post stays |0>."""
        syn = QuantumSynapse(1.0)  # theta=pi → full rotation → |1>

        # pre=|1>: post should rotate to |1>
        qc_on = QuantumCircuit(2)
        qc_on.x(0)  # pre = |1>
        syn.apply(qc_on, 0, 1)
        sv = Statevector.from_instruction(qc_on)
        p11 = float(sv.probabilities_dict().get("11", 0))
        assert p11 > 0.99

        # pre=|0>: post stays |0>
        qc_off = QuantumCircuit(2)
        syn.apply(qc_off, 0, 1)
        sv_off = Statevector.from_instruction(qc_off)
        p00 = float(sv_off.probabilities_dict().get("00", 0))
        assert p00 > 0.99


# ---------------------------------------------------------------------------
# update_weight
# ---------------------------------------------------------------------------


class TestUpdateWeight:
    def test_update_within_range(self):
        syn = QuantumSynapse(0.5)
        syn.update_weight(0.8)
        assert syn.weight == pytest.approx(0.8)

    def test_update_clamped_above(self):
        syn = QuantumSynapse(0.5)
        syn.update_weight(1.5)
        assert syn.weight == 1.0

    def test_update_clamped_below(self):
        syn = QuantumSynapse(0.5)
        syn.update_weight(-0.5)
        assert syn.weight == 0.0

    def test_theta_updates_after_weight_change(self):
        syn = QuantumSynapse(0.0)
        assert syn.theta == pytest.approx(0.0)
        syn.update_weight(1.0)
        assert syn.theta == pytest.approx(np.pi)
