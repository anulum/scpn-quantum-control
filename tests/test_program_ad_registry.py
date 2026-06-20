# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for Program AD primitive registry contracts
"""Tests for extracted Program AD primitive registry contracts."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from scpn_quantum_control import differentiable as differentiable_facade
from scpn_quantum_control.differentiable import (
    registered_custom_jacobian,
    registered_custom_jvp,
    registered_custom_vjp,
)
from scpn_quantum_control.program_ad_registry import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    ProgramADRegistryDispatchCoverageReport,
    ProgramADRegistryDispatchCoverageRow,
    custom_derivative_rule_for,
    primitive_complete_contract_for,
    primitive_contract_for,
    primitive_dtype_rule_for,
    primitive_effect_for,
    primitive_nondifferentiable_policy_for,
    primitive_shape_rule_for,
    primitive_static_argument_rule_for,
    program_ad_registry_dispatch_coverage_report,
    register_custom_derivative_rule,
    register_primitive_batching_rule,
    register_primitive_lowering_rule,
    register_primitive_transform_rule,
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
    assert scpn.custom_derivative_rule_for is custom_derivative_rule_for
    assert scpn.registered_custom_jvp is registered_custom_jvp
    assert scpn.registered_custom_vjp is registered_custom_vjp
    assert scpn.registered_custom_jacobian is registered_custom_jacobian


def test_custom_derivative_registry_binds_rule_by_primitive_identity() -> None:
    """Registered primitive identities should resolve exact custom rules automatically."""

    identity = PrimitiveIdentity("scpn.quantum", "rx_expectation", "1")
    rule = CustomDerivativeRule(
        name="rx_expectation_rule",
        value_fn=lambda values: np.array([np.cos(values[0]), values[1] ** 2], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [-np.sin(values[0]) * tangent[0], 2.0 * values[1] * tangent[1]],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [-np.sin(values[0]) * cotangent[0], 2.0 * values[1] * cotangent[1]],
            dtype=np.float64,
        ),
        parameter_names=("theta", "gain"),
        trainable=(True, False),
    )
    registry = CustomDerivativeRegistry()

    assert registry.register(identity, rule) is rule
    assert registry.lookup("scpn.quantum:rx_expectation@1") is rule
    assert custom_derivative_rule_for(identity, registry=registry) is rule
    np.testing.assert_allclose(
        registered_custom_jvp(
            "scpn.quantum:rx_expectation@1",
            np.array([0.25, 3.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            registry=registry,
        ),
        [-math.sin(0.25), 0.0],
        atol=1.0e-12,
    )
    vjp = registered_custom_vjp(
        identity,
        np.array([0.25, 3.0], dtype=np.float64),
        np.array([2.0, -1.0], dtype=np.float64),
        registry=registry,
    )
    np.testing.assert_allclose(vjp.vjp, [-2.0 * math.sin(0.25), 0.0], atol=1.0e-12)
    jacobian_result = registered_custom_jacobian(
        identity,
        np.array([0.25, 3.0], dtype=np.float64),
        registry=registry,
    )
    np.testing.assert_allclose(
        jacobian_result.jacobian,
        [[-math.sin(0.25), 0.0], [0.0, 0.0]],
        atol=1.0e-12,
    )


def test_custom_derivative_registry_rejects_ambiguous_identity_and_conflicts() -> None:
    """Registry bindings should fail closed on malformed keys and rule conflicts."""

    rule = CustomDerivativeRule(
        name="linear_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0]], dtype=np.float64),
    )
    other = CustomDerivativeRule(
        name="other_linear_rule",
        value_fn=lambda values: np.array([2.0 * values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([2.0 * tangent[0]], dtype=np.float64),
    )
    registry = CustomDerivativeRegistry()
    registry.register("scpn.test:linear@1", rule)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("scpn.test:linear@1", other)
    assert registry.register("scpn.test:linear@1", other, overwrite=True) is other
    with pytest.raises(ValueError, match="namespace:name"):
        PrimitiveIdentity.parse("bad-key")
    with pytest.raises(ValueError, match="no custom derivative rule"):
        custom_derivative_rule_for("scpn.test:missing@1", registry=registry)
    removed = registry.unregister("scpn.test:linear@1")
    assert removed is other
    assert registry.snapshot() == {}


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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"identity": "bad"}, "PrimitiveIdentity"),
        ({"derivative_rule": object()}, "CustomDerivativeRule"),
        ({"batching_rule": object()}, "batching_rule"),
        ({"lowering_rule": object()}, "lowering_rule"),
        ({"shape_rule": object()}, "shape_rule"),
        ({"dtype_rule": object()}, "dtype_rule"),
        ({"static_argument_rule": object()}, "static_argument_rule"),
        ({"nondifferentiable_policy": ""}, "nondifferentiable_policy"),
        ({"effect": ""}, "effect"),
        ({"lowering_metadata": {"": "value"}}, "metadata keys"),
        ({"lowering_metadata": {"key": ""}}, "metadata values"),
    ],
)
def test_primitive_contract_validation_paths(kwargs: dict[str, object], match: str) -> None:
    """Primitive contract views should reject malformed callable and metadata facets."""

    identity = PrimitiveIdentity("scpn.test", "contract_validation", "1")
    valid: dict[str, object] = {
        "identity": identity,
        "derivative_rule": _rule(),
        "batching_rule": _batching_rule,
        "lowering_rule": _lowering_rule,
        "lowering_metadata": {"mlir_op": "scpn_diff.contract_validation"},
        "shape_rule": _shape_rule,
        "dtype_rule": _dtype_rule,
        "static_argument_rule": _static_argument_rule,
        "nondifferentiable_policy": "fail_closed",
        "effect": "pure",
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=match):
        PrimitiveContract(**valid)  # type: ignore[arg-type]


def test_registry_registration_updates_preserve_existing_transform_metadata() -> None:
    """Rule replacement should preserve transform facets when overwrite is explicit."""

    identity = PrimitiveIdentity("scpn.test", "replace_preserves_metadata", "1")
    first_rule = _rule("replace_first")
    second_rule = _rule("replace_second")
    registry = CustomDerivativeRegistry({identity: first_rule})
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=first_rule,
            batching_rule=_batching_rule,
            lowering_rule=_lowering_rule,
            lowering_metadata={
                "mlir_op": "scpn_diff.replace_preserves_metadata",
                "nondifferentiable_boundary": "test_boundary",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=_shape_rule,
            dtype_rule=_dtype_rule,
            static_argument_rule=_static_argument_rule,
            nondifferentiable_policy="fail_closed",
            effect="pure",
        ),
        overwrite=True,
    )

    registry.register(identity, second_rule, overwrite=True)
    contract = registry.require_complete_contract(identity)

    assert contract.derivative_rule is second_rule
    assert contract.batching_rule is _batching_rule
    assert contract.lowering_rule is _lowering_rule
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.replace_preserves_metadata"
    assert contract.shape_rule is _shape_rule
    assert contract.dtype_rule is _dtype_rule
    assert contract.static_argument_rule is _static_argument_rule
    assert contract.nondifferentiable_policy == "fail_closed"
    assert contract.effect == "pure"


def test_registry_rejects_invalid_rule_and_reports_bare_contract_gaps() -> None:
    """Bare derivative registrations should expose every missing transform facet."""

    identity = PrimitiveIdentity("scpn.program_ad.elementwise", "sin", "1")
    registry = CustomDerivativeRegistry()

    with pytest.raises(ValueError, match="CustomDerivativeRule"):
        registry.register(identity, object())  # type: ignore[arg-type]

    registry.register(identity, _rule())
    with pytest.raises(ValueError) as exc_info:
        registry.require_complete_contract(identity)

    message = str(exc_info.value)
    for missing in (
        "batching_rule",
        "lowering_rule",
        "lowering_metadata",
        "nondifferentiable_boundary",
        "nondifferentiable_boundary_policy",
        "shape_rule",
        "dtype_rule",
        "static_argument_rule",
        "nondifferentiable_policy",
    ):
        assert missing in message

    report = program_ad_registry_dispatch_coverage_report(registry=registry)
    row = next(row for row in report.rows if row.identity == identity.key)
    assert row.complete is False
    for missing in (
        "batching_rule",
        "lowering_metadata",
        "shape_rule",
        "dtype_rule",
        "static_argument_rule",
        "nondifferentiable_policy",
    ):
        assert missing in row.blocked_reasons[0]


def test_transform_and_helper_registration_fail_closed_and_overwrite_paths() -> None:
    """Transform, batching, and lowering helpers should reject invalid or duplicate bindings."""

    identity = PrimitiveIdentity("scpn.test", "helper_overwrite", "1")
    first_rule = _rule("helper_first")
    second_rule = _rule("helper_second")
    registry = CustomDerivativeRegistry()
    first_transform = PrimitiveTransformRule(identity=identity, derivative_rule=first_rule)
    second_transform = PrimitiveTransformRule(identity=identity, derivative_rule=second_rule)

    assert register_primitive_transform_rule(first_transform, registry=registry) is first_transform
    with pytest.raises(ValueError, match="primitive transform already registered"):
        register_primitive_transform_rule(second_transform, registry=registry)
    assert (
        register_primitive_transform_rule(second_transform, overwrite=True, registry=registry)
        is second_transform
    )

    with pytest.raises(ValueError, match="batching_rule"):
        registry.register_batching_rule(identity, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lowering_rule"):
        registry.register_lowering_rule(identity, object())  # type: ignore[arg-type]

    registry.register_batching_rule(identity, _batching_rule)
    registry.register_lowering_rule(identity, _lowering_rule)
    with pytest.raises(ValueError, match="batching rule already registered"):
        registry.register_batching_rule(identity, _batching_rule)
    with pytest.raises(ValueError, match="lowering rule already registered"):
        registry.register_lowering_rule(identity, _lowering_rule)

    assert (
        registry.register_batching_rule(identity, _batching_rule, overwrite=True) is _batching_rule
    )
    assert (
        registry.register_lowering_rule(identity, _lowering_rule, overwrite=True) is _lowering_rule
    )


def test_primitive_contract_accessor_wrappers_and_missing_paths() -> None:
    """Top-level primitive accessors should route through the selected registry."""

    identity = PrimitiveIdentity("scpn.test", "accessor_wrappers", "1")
    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=_rule(),
            batching_rule=_batching_rule,
            lowering_rule=_lowering_rule,
            lowering_metadata={
                "mlir_op": "scpn_diff.accessor_wrappers",
                "nondifferentiable_boundary": "test_boundary",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=_shape_rule,
            dtype_rule=_dtype_rule,
            static_argument_rule=_static_argument_rule,
            nondifferentiable_policy="fail_closed",
            effect="pure",
        )
    )

    assert primitive_shape_rule_for(identity, registry=registry) is _shape_rule
    assert primitive_dtype_rule_for(identity, registry=registry) is _dtype_rule
    assert primitive_static_argument_rule_for(identity, registry=registry) is (
        _static_argument_rule
    )
    assert primitive_nondifferentiable_policy_for(identity, registry=registry) == "fail_closed"
    assert primitive_effect_for(identity, registry=registry) == "pure"

    missing = PrimitiveIdentity("scpn.test", "missing_accessors", "1")
    assert registry.shape_rule_for(missing) is None
    assert registry.dtype_rule_for(missing) is None
    assert registry.static_argument_rule_for(missing) is None
    assert registry.nondifferentiable_policy_for(missing) is None
    assert registry.effect_for(missing) is None
    assert registry.contract_for(missing) is None
    with pytest.raises(ValueError, match="no primitive contract"):
        registry.require_contract(missing)
    with pytest.raises(ValueError, match="no effect"):
        registry.require_effect(missing)
    with pytest.raises(ValueError, match="no custom derivative rule"):
        registry.unregister(missing)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"primitive": ""}, "primitive"),
        ({"identity": ""}, "identity"),
        ({"derivative_rule": ""}, "derivative_rule"),
        ({"lowering_metadata_keys": ("",)}, "metadata keys"),
        ({"blocked_reasons": ("",)}, "blocked reasons"),
        ({"complete": "yes"}, "complete"),
        ({"claim_boundary": ""}, "claim_boundary"),
    ],
)
def test_registry_dispatch_row_additional_validation_paths(
    kwargs: dict[str, object], match: str
) -> None:
    """Registry-dispatch rows should reject every malformed field independently."""

    valid: dict[str, object] = {
        "family": "family",
        "primitive": "primitive",
        "identity": "namespace:primitive@1",
        "derivative_rule": "rule",
        "has_batching_rule": True,
        "has_lowering_rule": False,
        "has_lowering_metadata": True,
        "has_shape_rule": True,
        "has_dtype_rule": True,
        "has_static_argument_rule": True,
        "nondifferentiable_policy": "fail_closed",
        "effect": "pure",
        "lowering_metadata_keys": ("boundary",),
        "complete": "blocked_reasons" not in kwargs,
        "blocked_reasons": (),
        "claim_boundary": "bounded",
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=match):
        ProgramADRegistryDispatchCoverageRow(**valid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("family_counts", "covered", "total", "claim_boundary", "match"),
    [
        ({"": 1}, 1, 1, "bounded", "families"),
        ({"family": 0}, 1, 1, "bounded", "positive"),
        ({"family": 1}, 1, 2, "bounded", "total"),
        ({"family": 1}, 0, 1, "bounded", "covered count"),
        ({"family": 2}, 1, 1, "bounded", "sum to total"),
        ({"family": 1}, 1, 1, "", "claim_boundary"),
    ],
)
def test_registry_dispatch_report_additional_validation_paths(
    family_counts: dict[str, int],
    covered: int,
    total: int,
    claim_boundary: str,
    match: str,
) -> None:
    """Registry-dispatch reports should reject inconsistent aggregate metadata."""

    valid_row = ProgramADRegistryDispatchCoverageRow(
        family="family",
        primitive="primitive",
        identity="namespace:primitive@1",
        derivative_rule="rule",
        has_batching_rule=True,
        has_lowering_rule=True,
        has_lowering_metadata=True,
        has_shape_rule=True,
        has_dtype_rule=True,
        has_static_argument_rule=True,
        nondifferentiable_policy="fail_closed",
        effect="pure",
        lowering_metadata_keys=("boundary",),
        complete=True,
        blocked_reasons=(),
        claim_boundary="bounded",
    )

    with pytest.raises(ValueError, match=match):
        ProgramADRegistryDispatchCoverageReport(
            rows=(valid_row,),
            family_counts=family_counts,
            covered_primitives=covered,
            total_primitives=total,
            claim_boundary=claim_boundary,
        )


def test_registry_dispatch_report_rejects_non_row_entries() -> None:
    """Registry-dispatch reports should reject non-row payload entries."""

    with pytest.raises(ValueError, match="coverage row"):
        ProgramADRegistryDispatchCoverageReport(
            rows=(object(),),  # type: ignore[arg-type]
            family_counts={"family": 1},
            covered_primitives=0,
            total_primitives=1,
            claim_boundary="bounded",
        )


def test_registry_dispatch_report_exposes_blocked_payload() -> None:
    """Registry-dispatch reports should expose blocked rows in JSON payloads."""

    blocked = ProgramADRegistryDispatchCoverageRow(
        family="family",
        primitive="primitive",
        identity="namespace:primitive@1",
        derivative_rule=None,
        has_batching_rule=False,
        has_lowering_rule=False,
        has_lowering_metadata=False,
        has_shape_rule=False,
        has_dtype_rule=False,
        has_static_argument_rule=False,
        nondifferentiable_policy=None,
        effect=None,
        lowering_metadata_keys=(),
        complete=False,
        blocked_reasons=("blocked",),
    )
    report = ProgramADRegistryDispatchCoverageReport(
        rows=(blocked,),
        family_counts={"family": 1},
        covered_primitives=0,
        total_primitives=1,
    )

    assert report.supported is False
    assert report.blocked_identities == ("namespace:primitive@1",)
    payload = report.to_dict()
    assert payload["supported"] is False
    assert payload["blocked_identities"] == ["namespace:primitive@1"]
