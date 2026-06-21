# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the aggregator/provider route matrix
"""Validation and resolution branch tests for the aggregator/provider matrix.

Covers the empty-IR-format route guard, the token validator, the route-id and
ambiguous-route resolution branches, and the route/profile IR-consistency guard.
"""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.aggregators import (
    AggregatorProviderRoute,
    _validate_routes_against_profiles,
    _validate_token,
    resolve_aggregator_provider_route,
)
from scpn_quantum_control.hardware.hal import built_in_backend_profiles


def _route(**overrides: object) -> AggregatorProviderRoute:
    """Build a minimal valid aggregator/provider route with field overrides."""
    fields: dict[str, object] = {
        "route_id": "agg/prov",
        "aggregator": "agg",
        "provider": "prov",
        "backend_id": "backend",
        "adapter_module": "scpn_quantum_control.hardware.adapters.example",
        "sdk_package": "example-sdk",
        "ir_formats": ("openqasm3",),
        "submit_requires_approval": False,
        "target_family": "superconducting",
    }
    fields.update(overrides)
    return AggregatorProviderRoute(**fields)  # type: ignore[arg-type]


def test_route_rejects_empty_ir_formats() -> None:
    """A route advertising no IR formats is rejected at construction."""
    with pytest.raises(ValueError, match="ir_formats must not be empty"):
        _route(ir_formats=())


def test_validate_token_rejects_empty_value() -> None:
    """The token validator rejects an empty identifier."""
    with pytest.raises(ValueError, match="must be a non-empty identifier token"):
        _validate_token("", "route_id")


def test_resolve_with_explicit_route_id() -> None:
    """An explicit route id filters the candidate routes to one match."""
    resolved = resolve_aggregator_provider_route(
        aggregator="aws_braket",
        provider="amazon",
        route_id="aws_braket/amazon_simulators",
    )
    assert resolved.route.route_id == "aws_braket/amazon_simulators"


def test_resolve_ambiguous_route_is_rejected() -> None:
    """An aggregator/provider pair with multiple routes and no filter is ambiguous."""
    with pytest.raises(LookupError, match="ambiguous aggregator/provider route"):
        resolve_aggregator_provider_route(aggregator="strangeworks", provider="ibm_quantum")


def test_route_profile_consistency_guard_rejects_unsupported_ir() -> None:
    """A route advertising an IR format outside its HAL profile is rejected."""
    profiles = {profile.backend_id: profile for profile in built_in_backend_profiles()}
    backend_id, profile = next(iter(profiles.items()))
    bad_route = _route(
        route_id="agg/bad",
        backend_id=backend_id,
        ir_formats=(*profile.ir_formats, "bogus_ir_format"),
    )
    with pytest.raises(ValueError, match="advertises IR formats not supported"):
        _validate_routes_against_profiles((bad_route,), profiles)
