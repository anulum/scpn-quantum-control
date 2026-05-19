# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S9 quantum thermodynamics tests
"""Tests for the S9 quantum-thermodynamics readiness layer."""

from __future__ import annotations

import pytest

from scpn_quantum_control.thermodynamics import (
    QUANTUM_THERMO_SCHEMA,
    ThermodynamicSweepConfig,
    calibrated_work_identity,
    entropy_production_rate,
    heat_dissipation_rate,
    irreversibility_residual,
    quantum_thermo_markdown,
    quantum_thermo_payload,
    run_k_sweep_protocol,
)


def test_entropy_production_rate_combines_heat_and_information_channels() -> None:
    rate = entropy_production_rate(
        heat_current_joule_per_s=2.0,
        bath_beta_per_joule=0.5,
        system_entropy_rate_nat_per_s=0.25,
        information_entropy_rate_nat_per_s=0.1,
    )

    assert rate.total_entropy_production_nat_per_s == pytest.approx(1.35)
    assert rate.non_negative is True
    assert rate.claim_boundary == "thermodynamic protocol estimate only; not hardware evidence"


def test_entropy_production_rejects_negative_total_budget() -> None:
    with pytest.raises(ValueError, match="negative entropy-production"):
        entropy_production_rate(
            heat_current_joule_per_s=-5.0,
            bath_beta_per_joule=1.0,
            system_entropy_rate_nat_per_s=0.1,
            information_entropy_rate_nat_per_s=0.1,
        )


def test_irreversibility_residual_uses_calibrated_work_identity() -> None:
    identity = calibrated_work_identity(
        work_samples_joule=(0.90, 1.00, 1.10),
        beta_per_joule=1.0,
        delta_free_energy_joule=0.95,
    )
    residual = irreversibility_residual(identity)

    assert identity.n_work_samples == 3
    assert identity.jarzynski_delta_free_energy_joule > 0.0
    assert residual.dissipated_work_joule == pytest.approx(0.05)
    assert residual.jarzynski_residual_abs_joule < 0.1


def test_heat_dissipation_requires_finite_jump_statistics() -> None:
    rate = heat_dissipation_rate(
        jump_counts=(2, 3, 4),
        jump_energy_joule=0.25,
        duration_s=3.0,
    )

    assert rate.mean_jump_count == pytest.approx(3.0)
    assert rate.heat_current_joule_per_s == pytest.approx(0.25)
    with pytest.raises(ValueError, match="duration_s"):
        heat_dissipation_rate(jump_counts=(1,), jump_energy_joule=0.1, duration_s=0.0)


def test_k_sweep_protocol_marks_peak_without_hardware_promotion() -> None:
    config = ThermodynamicSweepConfig(k_values=(0.4, 0.8, 1.2), transition_k=0.8)
    result = run_k_sweep_protocol(config)

    assert result.schema == "s9_quantum_thermo_k_sweep_v1"
    assert result.peak_k == pytest.approx(0.8)
    assert result.hardware_submission_allowed is False
    assert result.hardware_claim_allowed is False
    assert all(row.entropy_production_nat_per_s >= 0.0 for row in result.rows)


def test_quantum_thermo_payload_keeps_claims_blocked() -> None:
    payload = quantum_thermo_payload()

    assert payload["schema"] == QUANTUM_THERMO_SCHEMA
    assert payload["no_qpu_submission"] is True
    assert payload["hardware_submission_allowed"] is False
    assert payload["thermodynamic_peak_claim_allowed"] is False
    assert payload["k_sweep"]["peak_k"] in payload["k_sweep"]["k_values"]
    assert "win-rate" not in payload["falsifier"]


def test_quantum_thermo_markdown_records_gate_and_falsifier() -> None:
    markdown = quantum_thermo_markdown(quantum_thermo_payload())

    assert "scpn-bench s9-quantum-thermo-readiness" in markdown
    assert "no statistically significant peak" in markdown
    assert "no hardware submission" in markdown


def test_sweep_config_fails_closed_on_unsorted_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ThermodynamicSweepConfig(k_values=(0.8, 0.4, 1.2))
