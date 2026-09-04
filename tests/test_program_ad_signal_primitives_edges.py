# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD Signal Primitive Edge Tests
"""Exercise registered signal contracts and public runtime corruption guards."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.program_ad_signal_primitives as signal_module
from scpn_quantum_control.differentiable import (
    PrimitiveContract,
    PrimitiveIdentity,
    primitive_contract_for,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_registry import DEFAULT_CUSTOM_DERIVATIVE_REGISTRY


class TraceADArray:
    """Provide a trace-shaped probe with deliberately non-static shape metadata."""

    context = object()
    shape: object = [2]


def _signal_contract(name: str = "convolve") -> PrimitiveContract:
    return primitive_contract_for(f"scpn.program_ad.signal:{name}")


def _run_public_convolve() -> None:
    values = np.array([0.2, -0.4], dtype=np.float64)
    whole_program_value_and_grad(
        lambda parameters: np.sum(np.convolve(parameters, np.array([1.0]))),
        values,
    )


def _substitute_signal_contract(
    monkeypatch: pytest.MonkeyPatch,
    contract: PrimitiveContract,
) -> None:
    registry = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY
    original = registry.require_contract

    def require_contract(identity: PrimitiveIdentity) -> PrimitiveContract:
        if identity == contract.identity:
            return contract
        return original(identity)

    monkeypatch.setattr(registry, "require_contract", require_contract)


def _non_tuple_static_arguments(
    _args: tuple[object, ...],
) -> tuple[object, ...]:
    return cast(tuple[object, ...], cast(object, ["not-a-tuple"]))


def _negative_shape(_args: tuple[object, ...]) -> tuple[int, ...]:
    return (-1,)


def _empty_dtype(_args: tuple[object, ...]) -> str:
    return ""


def test_registered_signal_contract_rejects_trace_shape_and_complex_dtype() -> None:
    """Reject dynamic trace shapes and non-real dtype evidence through contract rules."""
    contract = _signal_contract()
    assert contract.shape_rule is not None
    assert contract.dtype_rule is not None

    with pytest.raises(ValueError, match="shape must be static"):
        contract.shape_rule((TraceADArray(), np.array([1.0]), "full"))
    with pytest.raises(ValueError, match="requires real numeric arrays"):
        contract.dtype_rule(
            (
                np.array([1.0 + 2.0j], dtype=np.complex128),
                np.array([1.0]),
                "full",
            )
        )


def test_registered_trace_derivative_rule_fails_closed_when_called_directly() -> None:
    """Reject direct value and tangent calls on operator-intercepted contracts."""
    rule = _signal_contract().derivative_rule
    values = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        rule.value_fn(values)
    assert rule.jvp_rule is not None
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        rule.jvp_rule(values, values)


def test_registered_shape_dtype_and_static_rules_reject_missing_arguments() -> None:
    """Reject incomplete convolve/correlate argument tuples at contract boundaries."""
    convolve = _signal_contract("convolve")
    correlate = _signal_contract("correlate")
    left = np.array([1.0, 2.0], dtype=np.float64)
    right = np.array([1.0], dtype=np.float64)
    assert convolve.shape_rule is not None
    assert convolve.dtype_rule is not None
    assert correlate.shape_rule is not None

    with pytest.raises(ValueError, match="requires left, right, and mode"):
        convolve.shape_rule((left, right))
    with pytest.raises(ValueError, match="dtype rule requires"):
        convolve.dtype_rule((left, right))
    with pytest.raises(ValueError, match="requires left, right, and mode"):
        correlate.shape_rule((left, right))


def test_registered_batching_rule_validates_arity_and_unmapped_input() -> None:
    """Reject malformed batching metadata and preserve an unbatched result."""
    contract = _signal_contract()
    assert contract.batching_rule is not None

    def convolve(left: object, right: object, mode: object) -> object:
        return np.convolve(cast(Any, left), cast(Any, right), mode=cast(Any, mode))

    left = np.array([1.0, 2.0], dtype=np.float64)
    right = np.array([0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="requires left, right, and mode"):
        contract.batching_rule(convolve, (left, right, "full"), (0, None), 0)

    result = contract.batching_rule(
        convolve,
        (left, right, "full"),
        (None, None, None),
        0,
    )
    np.testing.assert_allclose(cast(Any, result), np.convolve(left, right, mode="full"))


def test_signal_registration_is_idempotent_and_unknown_identity_fails_closed() -> None:
    """Keep repeated registration inert and reject an unknown signal identity."""
    contract = _signal_contract()

    signal_module._register_program_ad_signal_primitive_contracts()

    assert signal_module._require_program_ad_signal_contract("convolve") == contract
    with pytest.raises(ValueError, match="no program AD signal primitive identity"):
        signal_module._require_program_ad_signal_contract("unknown")


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"nondifferentiable_policy": "wrong-policy"}, "invalid.*policy"),
        ({"effect": "stateful"}, "invalid.*effect"),
        (
            {
                "batching_rule": None,
                "lowering_metadata": {},
                "shape_rule": None,
                "dtype_rule": None,
                "static_argument_rule": None,
            },
            "incomplete.*batching_rule.*lowering_metadata.*mlir_op.*"
            "nondifferentiable_boundary.*nondifferentiable_boundary_policy.*"
            "shape_rule.*dtype_rule.*static_argument_rule",
        ),
    ),
)
def test_public_runtime_rejects_incomplete_or_altered_signal_contracts(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    message: str,
) -> None:
    """Fail closed through whole-program execution on altered registry contracts."""
    contract = cast(PrimitiveContract, cast(Any, replace)(_signal_contract(), **changes))
    _substitute_signal_contract(monkeypatch, contract)

    with pytest.raises(ValueError, match=message):
        _run_public_convolve()


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"static_argument_rule": _non_tuple_static_arguments}, "static rule must return a tuple"),
        ({"shape_rule": _negative_shape}, "shape rule must return non-negative"),
        ({"dtype_rule": _empty_dtype}, "dtype rule must return a dtype name"),
    ),
)
def test_public_runtime_rejects_malformed_signal_dispatch_results(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    message: str,
) -> None:
    """Fail closed when registry dispatch helpers return malformed evidence."""
    contract = cast(PrimitiveContract, cast(Any, replace)(_signal_contract(), **changes))
    _substitute_signal_contract(monkeypatch, contract)

    with pytest.raises(ValueError, match=message):
        _run_public_convolve()
