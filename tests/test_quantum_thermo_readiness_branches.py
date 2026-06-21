# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for quantum-thermodynamics readiness
"""Branch tests for the S9 quantum-thermodynamics readiness helpers.

Covers the four result-dataclass ``to_dict`` serialisers, the K-sweep config
validation guards, the work-identity and heat-dissipation input guards, and the
finite / non-negative numeric guard helpers.
"""

from __future__ import annotations

import pytest

from scpn_quantum_control.thermodynamics.readiness import (
    ThermodynamicSweepConfig,
    calibrated_work_identity,
    entropy_production_rate,
    heat_dissipation_rate,
    irreversibility_residual,
)


def test_result_dataclasses_serialise_to_dict() -> None:
    """Each readiness result dataclass produces JSON-compatible row data."""
    entropy = entropy_production_rate(
        heat_current_joule_per_s=1.0,
        bath_beta_per_joule=0.5,
        system_entropy_rate_nat_per_s=0.1,
        information_entropy_rate_nat_per_s=0.1,
    )
    assert entropy.to_dict()["non_negative"] is True

    identity = calibrated_work_identity(
        work_samples_joule=(0.1, 0.2, 0.15),
        beta_per_joule=0.5,
        delta_free_energy_joule=0.1,
    )
    assert identity.to_dict()["n_work_samples"] == 3

    residual = irreversibility_residual(identity)
    assert "jarzynski_residual_abs_joule" in residual.to_dict()

    heat = heat_dissipation_rate(jump_counts=(3, 4, 5), jump_energy_joule=0.02, duration_s=1.0)
    assert heat.to_dict()["duration_s"] == 1.0


def test_sweep_config_requires_three_k_values() -> None:
    """A K grid shorter than three points is rejected."""
    with pytest.raises(ValueError, match="at least three values"):
        ThermodynamicSweepConfig(k_values=(0.4, 0.6))


def test_sweep_config_rejects_non_finite_k_values() -> None:
    """A non-finite K value is rejected."""
    with pytest.raises(ValueError, match="k_values must be finite"):
        ThermodynamicSweepConfig(k_values=(0.4, 0.6, float("inf")))


def test_sweep_config_requires_transition_k_on_grid() -> None:
    """The transition K must lie on the readiness grid."""
    with pytest.raises(ValueError, match="transition_k must be one of"):
        ThermodynamicSweepConfig(k_values=(0.4, 0.6, 0.8), transition_k=9.9)


def test_work_identity_rejects_empty_samples() -> None:
    """An empty work-sample set is rejected."""
    with pytest.raises(ValueError, match="must not be empty"):
        calibrated_work_identity(
            work_samples_joule=(), beta_per_joule=0.5, delta_free_energy_joule=0.1
        )


def test_work_identity_rejects_non_finite_samples() -> None:
    """A non-finite work sample is rejected."""
    with pytest.raises(ValueError, match="must contain finite values"):
        calibrated_work_identity(
            work_samples_joule=(1.0, float("inf")),
            beta_per_joule=0.5,
            delta_free_energy_joule=0.1,
        )


def test_heat_dissipation_rejects_empty_jump_counts() -> None:
    """An empty jump-count set is rejected."""
    with pytest.raises(ValueError, match="must not be empty"):
        heat_dissipation_rate(jump_counts=(), jump_energy_joule=0.02, duration_s=1.0)


def test_heat_dissipation_rejects_negative_jump_counts() -> None:
    """A negative jump count is rejected."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        heat_dissipation_rate(jump_counts=(-1, 2), jump_energy_joule=0.02, duration_s=1.0)


def test_entropy_production_requires_finite_heat_current() -> None:
    """A non-finite heat current is rejected by the finite guard."""
    with pytest.raises(ValueError, match="heat_current_joule_per_s must be finite"):
        entropy_production_rate(
            heat_current_joule_per_s=float("inf"),
            bath_beta_per_joule=0.5,
            system_entropy_rate_nat_per_s=0.1,
            information_entropy_rate_nat_per_s=0.1,
        )


def test_entropy_production_requires_non_negative_information_rate() -> None:
    """A negative information-entropy rate is rejected by the non-negative guard."""
    with pytest.raises(ValueError, match="information_entropy_rate_nat_per_s.*non-negative"):
        entropy_production_rate(
            heat_current_joule_per_s=1.0,
            bath_beta_per_joule=0.5,
            system_entropy_rate_nat_per_s=0.1,
            information_entropy_rate_nat_per_s=-1.0,
        )
