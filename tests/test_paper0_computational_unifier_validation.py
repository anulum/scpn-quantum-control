# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 computational-unifier validation tests
"""Executable fixture tests for Paper 0 EQ0115-EQ0118 anchors."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.computational_unifier_validation import (
    ABLBoundaryConfig,
    CyclicOperatorConfig,
    InformationThermodynamicsConfig,
    abl_probabilities,
    entropy_budget_rates,
    validate_cyclic_operator_fixture,
    validate_information_thermodynamics_fixture,
    validate_tsvf_abl_fixture,
)


def test_cyclic_operator_fixture_verifies_unitarity_and_cycle_closure() -> None:
    result = validate_cyclic_operator_fixture(CyclicOperatorConfig(dimension=4, period=7))

    assert result.spec_key == "computational.cyclic_operator_boundary"
    assert result.source_equation_ids == ("EQ0115",)
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.unitarity_error < 1.0e-12
    assert result.cycle_closure_residual < 1.0e-12
    assert result.null_controls["non_unitary_rejection_label"] == pytest.approx(1.0)
    assert result.null_controls["wrong_period_residual"] > 0.1


def test_abl_probabilities_are_normalised_and_reduce_to_born_rule() -> None:
    config = ABLBoundaryConfig()
    probabilities = abl_probabilities(config.pre_state, config.post_state, config.projectors)

    assert np.isclose(sum(probabilities), 1.0)
    assert all(value >= 0.0 for value in probabilities)

    result = validate_tsvf_abl_fixture(config)
    assert result.spec_key == "computational.tsvf_abl_boundary"
    assert result.source_equation_ids == ("EQ0116",)
    assert result.probability_normalisation_error < 1.0e-12
    assert result.null_controls["born_rule_reduction_l1"] < 1.0e-12
    assert result.null_controls["zero_denominator_rejection_label"] == pytest.approx(1.0)


def test_information_thermodynamics_fixture_checks_gsl_and_landauer_budget() -> None:
    config = InformationThermodynamicsConfig(
        thermodynamic_entropy_rate=-0.12,
        mutual_information_rate=0.4,
        landauer_cost_per_nat=0.5,
        proportionality=0.3,
    )

    rates = entropy_budget_rates(config)
    result = validate_information_thermodynamics_fixture(config)

    assert rates.negentropy_rate == pytest.approx(0.12)
    assert rates.information_entropy_rate == pytest.approx(0.2)
    assert rates.total_entropy_rate == pytest.approx(0.08)
    assert result.spec_key == "computational.info_thermodynamics"
    assert result.source_equation_ids == ("EQ0117", "EQ0118")
    assert result.gsl_margin == pytest.approx(0.08)
    assert result.mutual_information_negentropy_error < 1.0e-12
    assert result.null_controls["independent_channel_negentropy_abs"] == pytest.approx(0.0)
    assert result.null_controls["landauer_violation_label"] == pytest.approx(1.0)


def test_computational_unifier_fixtures_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="dimension must be at least two"):
        validate_cyclic_operator_fixture(CyclicOperatorConfig(dimension=1))

    with pytest.raises(ValueError, match="post_state denominator"):
        abl_probabilities(
            np.array([1.0, 0.0], dtype=np.complex128),
            np.array([0.0, 1.0], dtype=np.complex128),
            (np.diag([1.0, 0.0]),),
        )

    with pytest.raises(ValueError, match="landauer_cost_per_nat"):
        InformationThermodynamicsConfig(landauer_cost_per_nat=-1.0)
