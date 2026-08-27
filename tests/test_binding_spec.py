# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Binding Spec
"""Tests for identity/binding_spec.py."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.identity.binding_spec import (
    ARCANE_SAPIENCE_SPEC,
    ORCHESTRATOR_MAPPING,
    _build_knm_from_spec,
    build_identity_attractor,
    orchestrator_to_quantum_phases,
    quantum_to_orchestrator_phases,
    solve_identity,
)


def test_arcane_spec_structure() -> None:
    """The canonical spec retains six layers and 18 reduced oscillators."""
    assert "layers" in ARCANE_SAPIENCE_SPEC
    assert len(ARCANE_SAPIENCE_SPEC["layers"]) == 6
    total_osc = sum(len(lay["oscillator_ids"]) for lay in ARCANE_SAPIENCE_SPEC["layers"])
    assert total_osc == 18


def test_build_knm_symmetric() -> None:
    """The compiled coupling matrix is symmetric with a zero diagonal."""
    K, omega = _build_knm_from_spec(ARCANE_SAPIENCE_SPEC)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.all(np.diag(K) == 0)


def test_build_knm_shape() -> None:
    """The canonical matrix and frequency vector retain their exact shapes."""
    K, omega = _build_knm_from_spec(ARCANE_SAPIENCE_SPEC)
    assert K.shape == (18, 18)
    assert omega.shape == (18,)


def test_build_knm_positive() -> None:
    """Canonical coupling and frequency values remain non-negative."""
    K, omega = _build_knm_from_spec(ARCANE_SAPIENCE_SPEC)
    assert np.all(K >= 0)
    assert np.all(omega > 0)


def test_small_spec_attractor() -> None:
    """A small public custom spec builds and solves through the attractor API."""
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
def test_build_attractor_default_spec() -> None:
    """The default public attractor retains its 18-oscillator shape."""
    attractor = build_identity_attractor(ansatz_reps=1)
    assert attractor.K.shape == (18, 18)


def test_solve_identity_small() -> None:
    """The one-call public solver returns a finite small-system result."""
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


def test_arcane_sapience_spec_structure() -> None:
    """The canonical spec exposes both layers and coupling metadata."""
    assert "layers" in ARCANE_SAPIENCE_SPEC
    assert "coupling" in ARCANE_SAPIENCE_SPEC
    assert len(ARCANE_SAPIENCE_SPEC["layers"]) > 0


def test_orchestrator_mapping_nonempty() -> None:
    """The canonical reduced-to-orchestrator mapping is populated."""
    assert len(ORCHESTRATOR_MAPPING) > 0


@pytest.mark.slow
def test_solve_identity_returns_dict() -> None:
    """The default public solver returns its documented mapping result."""
    result = solve_identity(maxiter=5, seed=0)
    assert isinstance(result, dict)
    assert "ground_energy" in result


@pytest.mark.slow
def test_build_attractor_custom_reps() -> None:
    """The default public builder accepts a custom ansatz repetition count."""
    attractor = build_identity_attractor(ansatz_reps=2)
    assert attractor is not None


def test_descriptive_working_style_mapping_matches_domainpack() -> None:
    """Reduced working-style IDs map to the unchanged canonical leaf groups."""
    working_style_ids = ARCANE_SAPIENCE_SPEC["layers"][0]["oscillator_ids"]
    assert working_style_ids == [
        "working_style_action_verification",
        "working_style_delivery_preflight",
        "working_style_single_task",
    ]
    assert [ORCHESTRATOR_MAPPING[oscillator_id] for oscillator_id in working_style_ids] == [
        ["ws_action_first", "ws_verify_before_claim"],
        ["ws_commit_incremental", "ws_preflight_push"],
        ["ws_one_at_a_time"],
    ]


def test_stale_working_style_contract_is_rejected_by_public_builder() -> None:
    """The public builder rejects the exact obsolete abbreviated ID family."""
    stale_spec = copy.deepcopy(ARCANE_SAPIENCE_SPEC)
    stale_spec["layers"][0]["oscillator_ids"] = ["ws_0", "ws_1", "ws_2"]

    with pytest.raises(ValueError, match="stale abbreviated working-style"):
        build_identity_attractor(stale_spec)


def test_phase_mapping_roundtrip_preserves_canonical_order() -> None:
    """The public forward and reverse maps preserve all canonical phases."""
    quantum_phases: NDArray[np.float64] = np.linspace(
        -np.pi,
        np.pi,
        18,
        endpoint=False,
        dtype=np.float64,
    )

    orchestrator_phases = quantum_to_orchestrator_phases(quantum_phases)
    recovered = orchestrator_to_quantum_phases(orchestrator_phases)

    assert len(orchestrator_phases) == 35
    np.testing.assert_allclose(
        np.angle(np.exp(1j * (recovered - quantum_phases))),
        0.0,
        atol=1e-12,
    )


def test_phase_mapping_supports_descriptive_custom_spec() -> None:
    """A descriptive custom ID passes through both public phase maps."""
    spec = {
        "layers": [
            {
                "name": "custom",
                "oscillator_ids": ["custom_phase"],
                "natural_frequency": 1.0,
            }
        ],
        "coupling": {"base_strength": 0.0, "decay_alpha": 0.0},
    }

    orchestrator_phases = quantum_to_orchestrator_phases(np.array([0.75]), spec)
    recovered = orchestrator_to_quantum_phases(orchestrator_phases, spec)

    assert orchestrator_phases == {"custom_phase": 0.75}
    np.testing.assert_allclose(recovered, [0.75])


def test_phase_mapping_rejects_wrong_reduced_vector_length() -> None:
    """The public forward map rejects a phase vector with the wrong length."""
    with pytest.raises(ValueError, match="expected 18 quantum phases"):
        quantum_to_orchestrator_phases(np.zeros(17, dtype=np.float64))
