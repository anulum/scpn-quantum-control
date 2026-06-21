# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Engine-fallback test for witness candidate features
"""Native-engine fallback test for the witness candidate-feature evaluator."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scpn_quantum_control.analysis import witness_discovery


def test_candidate_features_falls_back_when_engine_export_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native engine without the candidate-feature export uses the NumPy path.

    A stand-in ``scpn_quantum_engine`` module that lacks
    ``kuramoto_witness_candidate_features`` forces the ``AttributeError`` arm of
    the Rust-preferred branch, exercising the deterministic NumPy fallback while
    keeping ``prefer_rust=True``.
    """
    stub = types.ModuleType("scpn_quantum_engine")
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    theta0 = np.zeros(2, dtype=np.float64)
    omega = np.array([0.3, -0.2], dtype=np.float64)
    k_nm = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    candidates = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)

    final_r, mean_corr, final_theta, backend = witness_discovery._candidate_features(
        theta0,
        omega,
        k_nm,
        candidates,
        dt=0.01,
        n_steps=3,
        prefer_rust=True,
    )

    assert backend == "numpy:kuramoto_witness_candidate_features"
    assert final_r.shape == (1,)
    assert mean_corr.shape == (1,)
    assert final_theta.shape == (1, 2)
    assert np.all(np.isfinite(final_r))
    assert np.all(np.isfinite(final_theta))
