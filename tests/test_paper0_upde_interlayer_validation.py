# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE inter-layer validation tests
"""Executable simulator fixtures for the Paper 0 UPDE inter-layer equation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.upde_validation import (
    InterlayerCouplingConfig,
    circular_mean_phase,
    interlayer_coupling_terms,
    validate_upde_interlayer_fixture,
)


def _layer_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.array([0.12, 0.18, 0.27, 0.31], dtype=np.float64)
    current = np.array([0.42, 0.61, 0.83, 1.05], dtype=np.float64)
    upper = np.array([1.28, 1.34, 1.41, 1.49], dtype=np.float64)
    return lower, current, upper


def test_interlayer_terms_separate_downward_and_upward_channels() -> None:
    lower, current, upper = _layer_fixture()
    config = InterlayerCouplingConfig(epsilon_lower=0.7, epsilon_upper=0.25)

    terms = interlayer_coupling_terms(lower, current, upper, config=config)
    lower_shifted = interlayer_coupling_terms(lower + 0.2, current, upper, config=config)
    upper_shifted = interlayer_coupling_terms(lower, current, upper - 0.2, config=config)

    assert terms.downward.shape == current.shape
    assert terms.upward.shape == current.shape
    np.testing.assert_allclose(terms.total, terms.downward + terms.upward)
    assert np.linalg.norm(lower_shifted.downward - terms.downward) > 0.05
    np.testing.assert_allclose(lower_shifted.upward, terms.upward)
    assert np.linalg.norm(upper_shifted.upward - terms.upward) > 0.05
    np.testing.assert_allclose(upper_shifted.downward, terms.downward)


def test_interlayer_fixture_consumes_spec_and_records_disconnected_null() -> None:
    lower, current, upper = _layer_fixture()

    result = validate_upde_interlayer_fixture(lower, current, upper)

    assert result.spec_key == "upde.interlayer_coupling"
    assert result.validation_protocol == "paper0.upde.interlayer.directional_coupling"
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_equation_ids == ("EQ0033", "EQ0040")
    assert result.null_controls["disconnected_layer_linf"] == pytest.approx(0.0)
    assert result.directional_sensitivity["lower_to_downward_l2"] > 0.0
    assert result.directional_sensitivity["upper_to_upward_l2"] > 0.0
    assert result.predictive_error_norm >= 0.0


def test_interlayer_fixture_exports_phase_artifact_payload() -> None:
    lower, current, upper = _layer_fixture()

    result = validate_upde_interlayer_fixture(lower, current, upper)
    payload = result.phase_artifact_payload

    assert payload["regime_id"] == "paper0_upde_interlayer_fixture"
    assert payload["metadata"]["paper0_spec_key"] == "upde.interlayer_coupling"
    assert payload["metadata"]["hardware_status"] == "simulator_only_no_provider_submission"
    assert len(payload["layers"]) == 3
    assert np.asarray(payload["cross_layer_alignment"]).shape == (3, 3)


def test_interlayer_fixture_rejects_invalid_phase_inputs() -> None:
    lower, current, upper = _layer_fixture()

    with pytest.raises(ValueError, match="current_theta must be a non-empty 1-D array"):
        validate_upde_interlayer_fixture(lower, np.zeros((2, 2)), upper)

    bad_upper = upper.copy()
    bad_upper[0] = np.inf
    with pytest.raises(ValueError, match="upper_theta must contain only finite values"):
        validate_upde_interlayer_fixture(lower, current, bad_upper)

    with pytest.raises(ValueError, match="epsilon_lower must be finite and non-negative"):
        validate_upde_interlayer_fixture(
            lower,
            current,
            upper,
            config=InterlayerCouplingConfig(epsilon_lower=-0.1),
        )


def test_circular_mean_handles_phase_wraparound() -> None:
    phases = np.array([np.pi - 0.05, -np.pi + 0.05], dtype=np.float64)

    mean = circular_mean_phase(phases)

    assert abs(abs(mean) - np.pi) < 1e-12
