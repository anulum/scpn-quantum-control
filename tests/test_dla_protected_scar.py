# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for DLA-Protected Scar Memory
"""Tests for DLA-protected scar-memory revivals."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.qec import (
    DLAProtectedScarPrototype,
    DLAProtectedScarSimulationResult,
    DLAProtectedScarSpec,
    DLAProtectedSubspaceSpec,
    build_dla_protected_scar_prototype,
    evaluate_dla_protected_scar_counts,
    simulate_dla_protected_scar_memory,
)


def _small_spec() -> DLAProtectedScarSpec:
    return DLAProtectedScarSpec(
        memory_spec=DLAProtectedSubspaceSpec(
            n_logical=2,
            code_distance=3,
            target_parity=0,
        ),
        revival_period=1.0,
        n_time_steps=8,
    )


def test_prototype_builds_protected_logical_cat_memory():
    prototype = build_dla_protected_scar_prototype(_small_spec())

    assert isinstance(prototype, DLAProtectedScarPrototype)
    assert prototype.certificate.is_provable
    assert prototype.scar_logical_words == ((0, 0), (1, 1))
    assert prototype.scar_basis_indices == (0, 63)
    assert prototype.preparation_circuit.num_qubits == 6
    assert prototype.preparation_circuit.depth() > 0
    np.testing.assert_allclose(np.linalg.norm(prototype.initial_state), 1.0)


def test_scar_revival_has_unit_final_survival_and_low_midcycle_survival():
    result = simulate_dla_protected_scar_memory(spec=_small_spec())

    assert isinstance(result, DLAProtectedScarSimulationResult)
    assert result.passes
    assert result.backend in {
        "rust:dla_protected_trajectory_metrics",
        "python:evaluate_dla_protected_memory",
    }
    assert result.final_revival_fidelity == pytest.approx(1.0, abs=1e-12)
    assert result.midcycle_survival == pytest.approx(0.0, abs=1e-12)
    assert result.min_protected_weight == pytest.approx(1.0, abs=1e-12)
    assert result.max_parity_leakage == pytest.approx(0.0, abs=1e-12)


def test_survival_matches_cosine_revival_law_for_two_level_scar():
    spec = _small_spec()
    result = simulate_dla_protected_scar_memory(spec=spec)
    expected = np.cos(np.pi * result.times / spec.revival_period) ** 2

    np.testing.assert_allclose(result.survival_probability, expected, atol=1e-12)


def test_scar_support_stays_inside_selected_protected_basis_states():
    result = simulate_dla_protected_scar_memory(spec=_small_spec())

    np.testing.assert_allclose(result.scar_support, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.code_weight, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.target_parity_weight, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.total_weight, 1.0, atol=1e-12)


def test_counts_path_detects_protected_scar_support():
    prototype = build_dla_protected_scar_prototype(_small_spec())
    counts = [
        {"000000": 512, "111111": 512},
        {"000000": 256, "111111": 768},
        {"000000": 512, "111111": 512},
    ]

    result = evaluate_dla_protected_scar_counts(counts, prototype=prototype)

    assert result.passes
    assert result.backend.endswith(":counts")
    np.testing.assert_allclose(result.protected_weight, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.scar_support, 1.0, atol=1e-12)


def test_counts_path_fails_on_code_leakage_outside_scar_support():
    prototype = build_dla_protected_scar_prototype(_small_spec())
    counts = [
        {"000000": 512, "111111": 512},
        {"000000": 512, "010110": 512},
        {"000000": 512, "111111": 512},
    ]

    result = evaluate_dla_protected_scar_counts(counts, prototype=prototype)

    assert not result.passes
    assert "scar_support_below_threshold" in result.failure_reasons
    assert "parity_leakage_above_threshold" in result.failure_reasons


def test_custom_scar_words_must_live_in_protected_sector():
    spec = _small_spec()

    with pytest.raises(ValueError, match="protected DLA sector"):
        build_dla_protected_scar_prototype(
            spec,
            scar_logical_words=((0, 0), (1, 0)),
        )


def test_scar_spec_validates_thresholds_and_time_grid():
    with pytest.raises(ValueError, match="revival_period"):
        DLAProtectedScarSpec(revival_period=0.0)
    with pytest.raises(ValueError, match="n_time_steps"):
        DLAProtectedScarSpec(n_time_steps=1)
    with pytest.raises(ValueError, match="min_revival_fidelity"):
        DLAProtectedScarSpec(min_revival_fidelity=1.5)


def test_result_serialisation_contains_certificate_and_metrics():
    result = simulate_dla_protected_scar_memory(spec=_small_spec())
    payload = result.to_dict()

    assert payload["passes"] is True
    assert payload["prototype"]["certificate"]["is_provable"] is True
    assert payload["prototype"]["n_scar_states"] == 2
    assert payload["final_revival_fidelity"] == pytest.approx(1.0, abs=1e-12)


def test_default_four_logical_spec_runs_without_dense_overflow():
    result = simulate_dla_protected_scar_memory()

    assert result.prototype.spec.memory_spec.n_physical == 12
    assert result.final_revival_fidelity == pytest.approx(1.0, abs=1e-12)
    assert result.passes
