# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Coverage 100 Control
"""Multi-angle tests for control/ subpackage: q_disruption_iter, qpetri, vqls_gs.

Covers: edge cases, shape validation, physical bounds, parametrised inputs,
error conditions, output types, reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest


# =====================================================================
# q_disruption_iter — Synthetic ITER Disruption Data
# =====================================================================
class TestQDisruptionIter:
    def test_default_rng(self):
        from scpn_quantum_control.control.q_disruption_iter import (
            generate_synthetic_iter_data,
        )

        X, y = generate_synthetic_iter_data(n_samples=20, rng=None)
        assert X.shape[0] == 20
        assert y.shape[0] == 20

    @pytest.mark.parametrize("n_samples", [1, 10, 50, 100])
    def test_shape_varies_with_n_samples(self, n_samples):
        from scpn_quantum_control.control.q_disruption_iter import (
            generate_synthetic_iter_data,
        )

        X, y = generate_synthetic_iter_data(
            n_samples=n_samples,
            rng=np.random.default_rng(42),
        )
        assert X.shape[0] == n_samples
        assert y.shape[0] == n_samples

    def test_labels_binary(self):
        from scpn_quantum_control.control.q_disruption_iter import (
            generate_synthetic_iter_data,
        )

        _, y = generate_synthetic_iter_data(n_samples=100, rng=np.random.default_rng(42))
        assert set(np.unique(y)).issubset({0, 1})

    def test_features_finite(self):
        from scpn_quantum_control.control.q_disruption_iter import (
            generate_synthetic_iter_data,
        )

        X, _ = generate_synthetic_iter_data(n_samples=50, rng=np.random.default_rng(42))
        assert np.all(np.isfinite(X))

    def test_reproducible_with_rng(self):
        from scpn_quantum_control.control.q_disruption_iter import (
            generate_synthetic_iter_data,
        )

        X1, y1 = generate_synthetic_iter_data(n_samples=20, rng=np.random.default_rng(42))
        X2, y2 = generate_synthetic_iter_data(n_samples=20, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# =====================================================================
# QuantumPetriNet — Shape Validation + Construction
# =====================================================================
class TestQPetri:
    def test_w_out_shape_mismatch_raises(self):
        from scpn_quantum_control.control.qpetri import QuantumPetriNet

        W_in = np.array([[1.0, 0.0], [0.0, 1.0]])
        W_out = np.array([[1.0, 0.0]])  # wrong shape
        thresholds = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="W_out shape"):
            QuantumPetriNet(
                n_places=2,
                n_transitions=2,
                W_in=W_in,
                W_out=W_out,
                thresholds=thresholds,
            )

    def test_valid_construction(self):
        from scpn_quantum_control.control.qpetri import QuantumPetriNet

        n_p, n_t = 2, 2
        W_in = np.eye(n_p)
        W_out = np.eye(n_p)
        thresholds = np.ones(n_t) * 0.5
        net = QuantumPetriNet(
            n_places=n_p,
            n_transitions=n_t,
            W_in=W_in,
            W_out=W_out,
            thresholds=thresholds,
        )
        assert net.n_places == n_p
        assert net.n_transitions == n_t

    def test_encode_marking(self):
        from scpn_quantum_control.control.qpetri import QuantumPetriNet

        W_in = np.eye(2)
        W_out = np.eye(2)
        thresholds = np.array([0.5, 0.5])
        net = QuantumPetriNet(
            n_places=2,
            n_transitions=2,
            W_in=W_in,
            W_out=W_out,
            thresholds=thresholds,
        )
        qc = net.encode_marking(np.array([1.0, 0.0]))
        assert qc.num_qubits == 2

    def test_step_returns_marking(self):
        from scpn_quantum_control.control.qpetri import QuantumPetriNet

        W_in = np.eye(2)
        W_out = np.eye(2)
        thresholds = np.array([0.5, 0.5])
        net = QuantumPetriNet(
            n_places=2,
            n_transitions=2,
            W_in=W_in,
            W_out=W_out,
            thresholds=thresholds,
        )
        marking = net.step(np.array([0.8, 0.3]))
        assert marking.shape == (2,)
        assert all(np.isfinite(marking))


# =====================================================================
# VQLS_GradShafranov — Variational Quantum Linear System
# =====================================================================
class TestVQLSGradShafranov:
    def test_denominator_near_zero_returns_array(self):
        from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

        solver = VQLS_GradShafranov(n_qubits=2)
        result = solver.solve(maxiter=1, seed=42)
        assert isinstance(result, np.ndarray)

    def test_solve_returns_correct_shape(self):
        from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

        solver = VQLS_GradShafranov(n_qubits=2)
        result = solver.solve(maxiter=5, seed=42)
        assert result.shape == (4,)  # 2^n_qubits

    def test_solve_output_finite(self):
        from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

        solver = VQLS_GradShafranov(n_qubits=2)
        result = solver.solve(maxiter=5, seed=42)
        assert np.all(np.isfinite(result))

    def test_imaginary_tolerance_zero_raises(self):
        from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

        solver = VQLS_GradShafranov(n_qubits=2, imag_tol=0.0)
        with pytest.raises(ValueError, match="imaginary norm"):
            solver.solve(reps=1, maxiter=1, seed=0)

    def test_reproducible_with_seed(self):
        from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

        s1 = VQLS_GradShafranov(n_qubits=2)
        s2 = VQLS_GradShafranov(n_qubits=2)
        r1 = s1.solve(maxiter=3, seed=42)
        r2 = s2.solve(maxiter=3, seed=42)
        np.testing.assert_array_equal(r1, r2)
