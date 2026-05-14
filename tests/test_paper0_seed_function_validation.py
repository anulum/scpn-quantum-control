# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 seed-function fixture tests
"""Tests for Paper 0 teleological seed function fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.seed_function_validation import (
    SeedFunctionConfig,
    compute_teleological_seed,
    mu_squared_seed,
    validate_seed_function_fixture,
)


def test_mu_squared_seed_matches_source_formula() -> None:
    result = mu_squared_seed(prev_cycle_sec=0.81, coupling_constant_g=0.36)

    assert result == pytest.approx(1.5)


def test_compute_teleological_seed_returns_source_payload_contract() -> None:
    payload = compute_teleological_seed(prev_cycle_sec=0.81, coupling_constant_g=0.36)

    assert payload["ssb_bias_magnitude"] == pytest.approx(1.5)
    assert payload["is_random_reset"] is False
    assert payload["conformal_continuity"] is True


def test_compute_teleological_seed_rejects_invalid_source_inputs() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        compute_teleological_seed(prev_cycle_sec=-0.1, coupling_constant_g=0.36)
    with pytest.raises(ValueError, match="finite and positive"):
        compute_teleological_seed(prev_cycle_sec=0.81, coupling_constant_g=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        SeedFunctionConfig(coupling_threshold=0.0)


def test_seed_function_fixture_preserves_boundaries_and_null_controls() -> None:
    result = validate_seed_function_fixture()

    assert result.spec_keys == (
        "seed_function.python_format_source_boundary",
        "seed_function.mu_squared_seed_formula",
        "seed_function.return_payload_contract",
        "seed_function.conformal_continuity_boundary",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06363", "P0R06377")
    assert result.payload["ssb_bias_magnitude"] > result.config_thresholds["bias_threshold"]
    assert result.payload["is_random_reset"] is False
    assert result.payload["conformal_continuity"] is True
    assert result.null_controls["negative_sec_rejection_label"] == 1.0
    assert result.null_controls["zero_coupling_rejection_label"] == 1.0
    assert result.null_controls["unsupported_seed_evidence_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
