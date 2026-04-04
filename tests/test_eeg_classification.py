# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for EEG structured VQE classification."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.applications.eeg_classification import (
    EEGVQEResult,
    eeg_plv_to_vqe,
    eeg_quantum_kernel,
)


class TestEEGClassification:
    def test_eeg_plv_to_vqe_basic(self):
        """Verify structured VQE runs on a synthetic EEG PLV matrix."""
        n = 3
        # Synthetic PLV matrix (strong coupling 0-1, weak 1-2, moderate 0-2)
        plv = np.array([[0.0, 0.8, 0.3], [0.8, 0.0, 0.1], [0.3, 0.1, 0.0]])
        omega = np.array([10.0, 10.0, 10.0])  # alpha band peaks

        # Threshold 0.2 means only (0,1) and (0,2) get entangled.
        res = eeg_plv_to_vqe(plv, omega, reps=1, threshold=0.2)

        assert isinstance(res, EEGVQEResult)
        assert res.n_channels == n
        assert res.success
        # 6 single-qubit ops + 2 CZs = 8 params?
        # Actually 6 single-qubit ops => 6 parameters per rep. reps=1 => 6 params.
        assert res.n_params == 6

    def test_eeg_quantum_kernel(self):
        """Verify quantum kernel evaluates correctly."""
        state_a = np.array([1.0, 0.0])
        state_b = np.array([1.0, 0.0])
        state_c = np.array([0.0, 1.0])

        k_ab = eeg_quantum_kernel(state_a, state_b)
        k_ac = eeg_quantum_kernel(state_a, state_c)

        assert abs(k_ab - 1.0) < 1e-10
        assert abs(k_ac - 0.0) < 1e-10
