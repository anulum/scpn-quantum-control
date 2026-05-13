# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 macro-transition RG validation tests
"""Executable simulator fixture tests for the Paper 0 effective-coupling RG anchor."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.macro_transition_rg_validation import (
    RGFlowValidationConfig,
    classify_fixed_point_stability,
    constant_beta_flow,
    integrate_rg_flow,
    validate_macro_transition_rg_fixture,
    zero_beta_flow,
)


def test_constant_beta_flow_matches_log_scale_analytic_solution() -> None:
    scales = np.array([1.0, 1.5, 2.0, 4.0, 8.0], dtype=np.float64)
    beta_value = 0.23

    result = constant_beta_flow(0.4, scales, beta_value)
    expected = 0.4 + beta_value * np.log(scales / scales[0])

    np.testing.assert_allclose(result, expected, atol=2e-12, rtol=2e-12)


def test_zero_beta_flow_is_invariant_on_scale_grid() -> None:
    scales = np.geomspace(0.5, 8.0, 9)

    result = zero_beta_flow(0.73, scales)

    np.testing.assert_allclose(result, np.full_like(scales, 0.73), atol=0.0, rtol=0.0)


def test_integrated_flow_converges_to_stable_fixed_point() -> None:
    scales = np.geomspace(1.0, 16.0, 33)
    K_star = 1.25

    trajectory = integrate_rg_flow(
        initial_K_eff=0.22,
        scale_grid=scales,
        beta_function=lambda value, _scale: 0.9 * (K_star - value),
    )

    assert trajectory[0] == pytest.approx(0.22)
    assert trajectory[-1] > trajectory[0]
    assert abs(trajectory[-1] - K_star) < 0.1
    assert classify_fixed_point_stability(lambda value: 0.9 * (K_star - value), K_star) == (
        "stable"
    )


def test_rg_fixture_consumes_macro_spec_and_records_controls() -> None:
    result = validate_macro_transition_rg_fixture()

    assert result.spec_key == "macro_transition.effective_coupling_rg"
    assert result.validation_protocol == "paper0.macro_transition.effective_coupling_rg_flow"
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_equation_ids == ("EQ0114",)
    assert "P0R05639" in result.source_ledger_ids
    assert result.fixed_point_candidate == pytest.approx(1.25)
    assert result.fixed_point_stability == "stable"
    assert result.final_K_eff > result.initial_K_eff
    assert result.null_controls["zero_beta_invariance_linf"] < 1e-12
    assert result.null_controls["constant_beta_analytic_error_linf"] < 1e-10
    assert result.null_controls["reverse_beta_final_delta"] > 0.1
    assert result.problem_metadata["scale_count"] == 33


def test_rg_fixture_rejects_invalid_scale_and_beta_inputs() -> None:
    with pytest.raises(ValueError, match="scale_grid must contain only positive values"):
        integrate_rg_flow(
            initial_K_eff=0.2,
            scale_grid=np.array([1.0, 0.0, 2.0]),
            beta_function=lambda value, _scale: value,
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        integrate_rg_flow(
            initial_K_eff=0.2,
            scale_grid=np.array([1.0, 2.0, 1.5]),
            beta_function=lambda value, _scale: value,
        )

    with pytest.raises(ValueError, match="initial_K_eff must be finite"):
        validate_macro_transition_rg_fixture(
            config=RGFlowValidationConfig(initial_K_eff=float("nan")),
        )
