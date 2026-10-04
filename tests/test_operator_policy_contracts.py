# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded operator input contracts
"""Qualify exact monetary, integer, Unicode, date and immutable plan inputs."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from scpn_quantum_control.hardware.operator_policy_contracts import (
    MAX_OPERATOR_INTEGER,
    OperatorPolicy,
    OperatorRequest,
    PricingEstimate,
    utc_second,
)


def sample_policy() -> OperatorPolicy:
    """Construct a literal source policy without relying on the policy algorithm."""
    return OperatorPolicy(
        "contract-policy",
        "route",
        ("target",),
        ("eu",),
        1024,
        2,
        60000,
        "12.50",
        "USD",
        "2026-01-01T00:00:00Z",
        "2027-01-01T00:00:00Z",
    )


def sample_request() -> OperatorRequest:
    """Supply exact literal shape and explicit automation intent."""
    return OperatorRequest("a" * 64, "route", "target", "eu", 1024, 2, 60000, True)


def test_detached_exact_source_inputs() -> None:
    """Original choice sequences and exported dictionaries cannot mutate contracts."""
    policy = sample_policy()
    assert OperatorPolicy.from_dict(policy.to_dict()) == policy
    exported = policy.to_dict()
    exported["targets"] = ["substitute"]
    assert policy.targets == ("target",)
    request = sample_request()
    assert OperatorRequest.from_dict(request.to_dict()) == request
    changed = request.to_dict()
    changed["shots"] = 7
    assert request.shots == 1024
    assert replace(request, shots=1025).sha256 != request.sha256
    price = PricingEstimate(
        request.sha256,
        "0.000000001",
        "USD",
        "synthetic",
        "2026-01-01T00:00:00Z",
        "2027-01-01T00:00:00Z",
    )
    assert PricingEstimate.from_dict(price.to_dict()) == price
    assert PricingEstimate.from_dict(replace(price, amount=None).to_dict()).amount is None
    with pytest.raises(FrozenInstanceError):
        request.__setattr__("shots", 1)
    assert sample_policy() == policy


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_shots", True),
        ("max_shots", 0),
        ("max_concurrency", -1),
        ("max_time_limit_ms", MAX_OPERATOR_INTEGER + 1),
        ("max_shots", 1.0),
        ("max_cost", 12.50),
        ("max_cost", "01"),
        ("max_cost", "-0"),
        ("max_cost", "1e1"),
        ("max_cost", "0.0000000001"),
        ("max_cost", "9" * 19),
        ("currency", "usd"),
        ("currency", "EU"),
        ("currency", 1),
        ("reference", ""),
        ("reference", "x" * 257),
        ("reference", "control\n"),
        ("reference", "\ud800"),
        ("reference", 1),
        ("backend_id", "\x7f"),
        ("targets", ()),
        ("targets", "target"),
        ("targets", ["target", "target"]),
        ("regions", ["r"] * 257),
        ("regions", [None]),
        ("valid_from", "0000-01-01T00:00:00Z"),
        ("expires_at", "2026-01-01T00:00:00Z"),
    ],
)
def test_invalid_policy_input_refusal(field: str, value: object) -> None:
    """Malformed ceilings, text, precision and choice lists cannot become authority."""
    candidate = sample_policy().to_dict()
    candidate[field] = value
    with pytest.raises(ValueError):
        OperatorPolicy.from_dict(candidate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("workload_sha256", "A" * 64),
        ("workload_sha256", 0),
        ("backend_id", ""),
        ("target", "\x80"),
        ("region", "\udfff"),
        ("shots", False),
        ("shots", 0),
        ("shots", 1.0),
        ("unattended", 1),
        ("unattended", "true"),
    ],
)
def test_invalid_request_input_refusal(field: str, value: object) -> None:
    """Binding identities and permission shapes reject before any assessment."""
    candidate = sample_request().to_dict()
    candidate[field] = value
    with pytest.raises(ValueError):
        OperatorRequest.from_dict(candidate)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-02-30T00:00:00Z",
        "2026-01-01",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00.1Z",
        "２０２６-01-01T00:00:00Z",
    ],
)
def test_exact_utc_spelling(timestamp: str) -> None:
    """Invalid calendar values, fractional seconds and local offsets remain refusals."""
    with pytest.raises(ValueError):
        utc_second(timestamp)


def test_unknown_regions_and_boundary_integer_remain_exact() -> None:
    """Unknown fields remain None and the uint63 transport edge is literal."""
    value = replace(
        sample_request(), region=None, target=None, shots=MAX_OPERATOR_INTEGER, unattended=False
    )
    assert OperatorRequest.from_dict(value.to_dict()) == value
    assert value.to_dict()["shots"] == 9223372036854775807


@pytest.mark.parametrize("missing", [False, True])
def test_missing_or_extra_contract_fields(missing: bool) -> None:
    """No silently supplied defaults or unknown future fields enter a contract."""
    value = sample_request().to_dict()
    if missing:
        del value["unattended"]
    else:
        value["future"] = 1
    with pytest.raises(ValueError, match="fields"):
        OperatorRequest.from_dict(value)


def test_empty_price_validity_interval_refuses() -> None:
    """A reversed or zero-duration estimate cannot be treated as dated evidence."""
    with pytest.raises(ValueError, match="interval"):
        PricingEstimate(
            "a" * 64, None, "USD", "unknown", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
