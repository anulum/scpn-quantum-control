# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for Program AD primitive registry contracts
"""Tests for extracted Program AD primitive registry contracts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from scpn_quantum_control import differentiable as differentiable_facade
from scpn_quantum_control.program_ad_registry import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    ProgramADRegistryDispatchCoverageReport,
    ProgramADRegistryDispatchCoverageRow,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_registry_dispatch_coverage_report,
    register_custom_derivative_rule,
    register_primitive_batching_rule,
    register_primitive_lowering_rule,
)


def _rule(name: str = "registry_rule") -> CustomDerivativeRule:
    """Return a small exact derivative rule for registry tests."""

    return CustomDerivativeRule(
        name=name,
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0]], dtype=np.float64),
    )


def _batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    """Return a deterministic batching result for registry tests."""

    del function, axes
    return np.moveaxis(np.asarray(args[0], dtype=np.float64), 0, out_axes)


def _lowering_rule(rule: CustomDerivativeRule) -> object:
    """Return deterministic lowering metadata for registry tests."""

    return rule.name


def _shape_rule(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the fixed scalar-vector output shape for registry tests."""

    del args
    return (1,)


def _dtype_rule(args: tuple[object, ...]) -> str:
    """Return the fixed dtype for registry tests."""

    del args
    return "float64"


def _static_argument_rule(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return no static arguments for registry tests."""

    del args
    return ()


def test_program_ad_registry_module_and_facade_share_public_objects() -> None:
    """The extracted module and differentiable facade should expose identical objects."""

    import scpn_quantum_control as scpn

    assert differentiable_facade.CustomDerivativeRule is CustomDerivativeRule
    assert differentiable_facade.PrimitiveIdentity is PrimitiveIdentity
    assert differentiable_facade.PrimitiveTransformRule is PrimitiveTransformRule
    assert differentiable_facade.PrimitiveContract is PrimitiveContract
    assert differentiable_facade.CustomDerivativeRegistry is CustomDerivativeRegistry
    assert differentiable_facade.DEFAULT_CUSTOM_DERIVATIVE_REGISTRY is (
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY
    )
    assert scpn.PrimitiveIdentity is PrimitiveIdentity
    assert scpn.program_ad_registry_dispatch_coverage_report is (
        program_ad_registry_dispatch_coverage_report
    )


def test_primitive_contract_round_trip_from_extracted_registry() -> None:
    """A complete transform should round-trip through contract helpers."""

    identity = PrimitiveIdentity("scpn.test", "contract", "1")
    rule = _rule()
    registry = CustomDerivativeRegistry()
    transform = PrimitiveTransformRule(
        identity=identity,
        derivative_rule=rule,
        batching_rule=_batching_rule,
        lowering_rule=_lowering_rule,
        lowering_metadata={
            "mlir_op": "scpn_diff.test_contract",
            "nondifferentiable_boundary": "test_boundary",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=_shape_rule,
        dtype_rule=_dtype_rule,
        static_argument_rule=_static_argument_rule,
        nondifferentiable_policy="fail_closed_at_boundaries",
        effect="pure",
    )

    assert registry.register_transform(transform) is transform
    contract = primitive_contract_for(identity, registry=registry)
    complete = primitive_complete_contract_for(identity, registry=registry)

    assert contract == complete
    assert contract.identity is identity
    assert contract.derivative_rule is rule
    assert contract.batching_rule is _batching_rule
    assert contract.lowering_rule is _lowering_rule
    assert contract.shape_rule is _shape_rule
    assert contract.dtype_rule is _dtype_rule
    assert contract.static_argument_rule is _static_argument_rule
    assert contract.nondifferentiable_policy == "fail_closed_at_boundaries"


def test_program_ad_registry_dispatch_report_uses_extracted_default_registry() -> None:
    """Registry-dispatch coverage should resolve all default Program AD contracts."""

    report = program_ad_registry_dispatch_coverage_report()
    facade_report = differentiable_facade.program_ad_registry_dispatch_coverage_report()

    assert isinstance(report, ProgramADRegistryDispatchCoverageReport)
    assert report.supported is True
    assert report.total_primitives == 118
    assert report.covered_primitives == report.total_primitives
    assert report.blocked_identities == ()
    assert report.family_counts == facade_report.family_counts
    assert all(isinstance(row, ProgramADRegistryDispatchCoverageRow) for row in report.rows)
    assert all(row.complete for row in report.rows)
    assert any(row.identity == "scpn.program_ad.linalg:pinv@1" for row in report.rows)
    assert "not executable Rust, LLVM, JIT" in report.claim_boundary


def test_program_ad_registry_dispatch_report_fails_closed_for_missing_contracts() -> None:
    """Registry-dispatch coverage should report missing registry contracts as blockers."""

    report = program_ad_registry_dispatch_coverage_report(registry=CustomDerivativeRegistry())

    assert report.supported is False
    assert report.covered_primitives == 0
    assert len(report.blocked_identities) == report.total_primitives
    assert all(row.derivative_rule is None for row in report.rows)
    assert all(
        row.blocked_reasons == ("missing primitive registry contract",) for row in report.rows
    )


def test_program_ad_registry_dispatch_report_detects_incomplete_boundary_metadata() -> None:
    """Boundary metadata drift should remain visible in registry-dispatch coverage."""

    registry = CustomDerivativeRegistry()
    for transform in DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.transform_snapshot().values():
        registry.register_transform(transform)
    original = primitive_contract_for("scpn.program_ad.elementwise:sin")
    registry.register_transform(
        PrimitiveTransformRule(
            identity=original.identity,
            derivative_rule=original.derivative_rule,
            batching_rule=original.batching_rule,
            lowering_rule=original.lowering_rule,
            lowering_metadata={
                key: value
                for key, value in original.lowering_metadata.items()
                if key
                not in {
                    "nondifferentiable_boundary",
                    "nondifferentiable_boundary_policy",
                }
            },
            shape_rule=original.shape_rule,
            dtype_rule=original.dtype_rule,
            static_argument_rule=original.static_argument_rule,
            nondifferentiable_policy=original.nondifferentiable_policy,
            effect=original.effect,
        ),
        overwrite=True,
    )

    report = program_ad_registry_dispatch_coverage_report(registry=registry)
    sin_row = next(row for row in report.rows if row.identity == original.identity.key)

    assert report.supported is False
    assert original.identity.key in report.blocked_identities
    assert sin_row.complete is False
    assert sin_row.blocked_reasons == (
        "incomplete registry-dispatch contract: missing "
        "nondifferentiable_boundary, nondifferentiable_boundary_policy",
    )


def test_default_registry_helper_paths_register_and_unregister() -> None:
    """Default helper functions should still mutate the shared default registry."""

    identity = PrimitiveIdentity("scpn.test", "default_helper", "1")
    rule = _rule("default_helper_rule")

    assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.lookup(identity) is None
    register_custom_derivative_rule(identity, rule)
    try:
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require(identity) is rule
        assert register_primitive_batching_rule(identity, _batching_rule) is _batching_rule
        assert register_primitive_lowering_rule(identity, _lowering_rule) is _lowering_rule
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_batching_rule(identity) is (
            _batching_rule
        )
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_lowering_rule(identity) is (
            _lowering_rule
        )
    finally:
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.lookup(identity) is not None:
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(identity)

    with pytest.raises(ValueError, match="no custom derivative rule"):
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require(identity)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "", "value_fn": lambda values: values, "jvp_rule": lambda v, t: t}, "name"),
        ({"name": "bad", "value_fn": object(), "jvp_rule": lambda v, t: t}, "value_fn"),
        ({"name": "bad", "value_fn": lambda values: values}, "requires a JVP or VJP"),
        (
            {"name": "bad", "value_fn": lambda values: values, "jvp_rule": object()},
            "jvp_rule",
        ),
        (
            {
                "name": "bad",
                "value_fn": lambda values: values,
                "vjp_rule": object(),
            },
            "vjp_rule",
        ),
        (
            {
                "name": "bad",
                "value_fn": lambda values: values,
                "jvp_rule": lambda v, t: t,
                "parameter_names": ("",),
            },
            "parameter_names",
        ),
        (
            {
                "name": "bad",
                "value_fn": lambda values: values,
                "jvp_rule": lambda v, t: t,
                "trainable": (1,),
            },
            "trainable mask",
        ),
        (
            {
                "name": "bad",
                "value_fn": lambda values: values,
                "jvp_rule": lambda v, t: t,
                "parameter_names": ("x",),
                "trainable": (True, False),
            },
            "lengths must match",
        ),
    ],
)
def test_custom_derivative_rule_validation(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Custom derivative rules should reject malformed contracts."""

    with pytest.raises(ValueError, match=match):
        CustomDerivativeRule(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: PrimitiveIdentity("", "x"), "non-empty"),
        (lambda: PrimitiveIdentity("bad namespace", "x"), "whitespace"),
        (lambda: PrimitiveIdentity("bad:namespace", "x"), "':' or '@'"),
        (lambda: PrimitiveIdentity.parse("missing_namespace"), "namespace:name"),
        (lambda: PrimitiveIdentity.parse(""), "non-empty"),
    ],
)
def test_primitive_identity_validation(
    factory: Callable[[], object],
    match: str,
) -> None:
    """Primitive identities should reject ambiguous registry keys."""

    with pytest.raises(ValueError, match=match):
        factory()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"identity": "bad", "derivative_rule": _rule()}, "PrimitiveIdentity"),
        (
            {"identity": PrimitiveIdentity("scpn.test", "bad"), "derivative_rule": object()},
            "CustomDerivativeRule",
        ),
        (
            {
                "identity": PrimitiveIdentity("scpn.test", "bad"),
                "derivative_rule": _rule(),
                "batching_rule": object(),
            },
            "batching_rule",
        ),
        (
            {
                "identity": PrimitiveIdentity("scpn.test", "bad"),
                "derivative_rule": _rule(),
                "lowering_rule": object(),
            },
            "lowering_rule",
        ),
        (
            {
                "identity": PrimitiveIdentity("scpn.test", "bad"),
                "derivative_rule": _rule(),
                "shape_rule": object(),
            },
            "shape_rule",
        ),
        (
            {
                "identity": PrimitiveIdentity("scpn.test", "bad"),
                "derivative_rule": _rule(),
                "dtype_rule": object(),
            },
            "dtype_rule",
        ),
        (
            {
                "identity": PrimitiveIdentity("scpn.test", "bad"),
                "derivative_rule": _rule(),
                "static_argument_rule": object(),
            },
            "static_argument_rule",
        ),
        (
            {
                "identity": PrimitiveIdentity("scpn.test", "bad"),
                "derivative_rule": _rule(),
                "nondifferentiable_policy": "",
            },
            "nondifferentiable_policy",
        ),
        (
            {
                "identity": PrimitiveIdentity("scpn.test", "bad"),
                "derivative_rule": _rule(),
                "effect": "",
            },
            "effect",
        ),
        (
            {
                "identity": PrimitiveIdentity("scpn.test", "bad"),
                "derivative_rule": _rule(),
                "lowering_metadata": {"": "value"},
            },
            "metadata keys",
        ),
        (
            {
                "identity": PrimitiveIdentity("scpn.test", "bad"),
                "derivative_rule": _rule(),
                "lowering_metadata": {"key": ""},
            },
            "metadata values",
        ),
    ],
)
def test_primitive_transform_validation(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Primitive transform bindings should reject incomplete callable metadata."""

    with pytest.raises(ValueError, match=match):
        PrimitiveTransformRule(**kwargs)  # type: ignore[arg-type]


def test_registry_conflict_and_decorator_paths() -> None:
    """Registry registration should require explicit overwrite for conflicts."""

    identity = PrimitiveIdentity("scpn.test", "conflict", "1")
    first = _rule("first")
    second = _rule("second")
    registry = CustomDerivativeRegistry()

    assert registry.decorator(identity)(first) is first
    with pytest.raises(ValueError, match="already registered"):
        registry.register(identity, second)
    assert registry.register(identity, second, overwrite=True) is second
    with pytest.raises(ValueError, match="transform must be"):
        registry.register_transform(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no batching rule"):
        registry.require_batching_rule(identity)
    with pytest.raises(ValueError, match="no lowering rule"):
        registry.require_lowering_rule(identity)
    with pytest.raises(ValueError, match="no shape rule"):
        registry.require_shape_rule(identity)
    with pytest.raises(ValueError, match="no dtype rule"):
        registry.require_dtype_rule(identity)
    with pytest.raises(ValueError, match="no static argument rule"):
        registry.require_static_argument_rule(identity)
    with pytest.raises(ValueError, match="no nondifferentiable policy"):
        registry.require_nondifferentiable_policy(identity)


def test_registry_dispatch_row_and_report_validation() -> None:
    """Registry-dispatch coverage containers should reject inconsistent rows."""

    valid_row = ProgramADRegistryDispatchCoverageRow(
        family="family",
        primitive="primitive",
        identity="namespace:primitive@1",
        derivative_rule="rule",
        has_batching_rule=True,
        has_lowering_rule=False,
        has_lowering_metadata=True,
        has_shape_rule=True,
        has_dtype_rule=True,
        has_static_argument_rule=True,
        nondifferentiable_policy="fail_closed",
        effect="pure",
        lowering_metadata_keys=("boundary",),
        complete=True,
        blocked_reasons=(),
    )
    assert valid_row.to_dict()["identity"] == "namespace:primitive@1"

    with pytest.raises(ValueError, match="family"):
        ProgramADRegistryDispatchCoverageRow(
            **{**valid_row.to_dict(), "family": ""}  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="complete"):
        ProgramADRegistryDispatchCoverageRow(
            family="family",
            primitive="primitive",
            identity="namespace:primitive@1",
            derivative_rule="rule",
            has_batching_rule=True,
            has_lowering_rule=False,
            has_lowering_metadata=True,
            has_shape_rule=True,
            has_dtype_rule=True,
            has_static_argument_rule=True,
            nondifferentiable_policy="fail_closed",
            effect="pure",
            lowering_metadata_keys=("boundary",),
            complete=True,
            blocked_reasons=("blocked",),
        )
    with pytest.raises(ValueError, match="requires rows"):
        ProgramADRegistryDispatchCoverageReport(
            rows=(),
            family_counts={},
            covered_primitives=0,
            total_primitives=0,
        )
    report = ProgramADRegistryDispatchCoverageReport(
        rows=(valid_row,),
        family_counts={"family": 1},
        covered_primitives=1,
        total_primitives=1,
    )
    assert report.supported is True
    assert report.to_dict()["covered_primitives"] == 1
