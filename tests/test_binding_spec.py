# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Binding Spec
"""Tests for identity/binding_spec.py."""

import numpy as np
import pytest

from scpn_quantum_control.identity.binding_spec import (
    ARCANE_SAPIENCE_SPEC,
    ORCHESTRATOR_MAPPING,
    _build_knm_from_spec,
    build_identity_attractor,
    solve_identity,
)


def test_arcane_spec_structure():
    assert "layers" in ARCANE_SAPIENCE_SPEC
    assert len(ARCANE_SAPIENCE_SPEC["layers"]) == 6
    total_osc = sum(len(lay["oscillator_ids"]) for lay in ARCANE_SAPIENCE_SPEC["layers"])
    assert total_osc == 18


def test_build_knm_symmetric():
    K, omega = _build_knm_from_spec(ARCANE_SAPIENCE_SPEC)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.all(np.diag(K) == 0)


def test_build_knm_shape():
    K, omega = _build_knm_from_spec(ARCANE_SAPIENCE_SPEC)
    assert K.shape == (18, 18)
    assert omega.shape == (18,)


def test_build_knm_positive():
    K, omega = _build_knm_from_spec(ARCANE_SAPIENCE_SPEC)
    assert np.all(K >= 0)
    assert np.all(omega > 0)


def test_small_spec_attractor():
    spec = {
        "layers": [
            {"name": "a", "oscillator_ids": ["a0", "a1"], "natural_frequency": 1.0},
            {"name": "b", "oscillator_ids": ["b0"], "natural_frequency": 2.0},
        ],
        "coupling": {"base_strength": 0.5, "decay_alpha": 0.2},
    }
    attractor = build_identity_attractor(spec, ansatz_reps=1)
    result = attractor.solve(maxiter=20, seed=0)
    assert "robustness_gap" in result
    assert result["robustness_gap"] >= 0.0


@pytest.mark.slow
def test_build_attractor_default_spec():
    attractor = build_identity_attractor(ansatz_reps=1)
    assert attractor.K.shape == (18, 18)


def test_solve_identity_small():
    spec = {
        "layers": [
            {"name": "x", "oscillator_ids": ["x0"], "natural_frequency": 1.5},
            {"name": "y", "oscillator_ids": ["y0"], "natural_frequency": 0.8},
        ],
        "coupling": {"base_strength": 0.3, "decay_alpha": 0.1},
    }
    result = solve_identity(spec, maxiter=20, seed=0)
    assert "ground_energy" in result
    assert np.isfinite(result["ground_energy"])


def test_arcane_sapience_spec_structure():
    assert "layers" in ARCANE_SAPIENCE_SPEC
    assert "coupling" in ARCANE_SAPIENCE_SPEC
    assert len(ARCANE_SAPIENCE_SPEC["layers"]) > 0


def test_orchestrator_mapping_nonempty():
    assert len(ORCHESTRATOR_MAPPING) > 0


def test_solve_identity_returns_dict():
    result = solve_identity(maxiter=5, seed=0)
    assert isinstance(result, dict)
    assert "ground_energy" in result


def test_build_attractor_custom_reps():
    attractor = build_identity_attractor(ansatz_reps=2)
    assert attractor is not None
