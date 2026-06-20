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
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import differentiable as differentiable_facade
from scpn_quantum_control.differentiable import (
    registered_custom_jacobian,
    registered_custom_jvp,
    registered_custom_vjp,
    vmap,
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


def test_vmap_uses_registered_primitive_batching_rule() -> None:
    """vmap should dispatch to primitive-specific batching rules when requested."""

    identity = PrimitiveIdentity("scpn.quantum", "batched_affine", "1")
    rule = CustomDerivativeRule(
        name="batched_affine_rule",
        value_fn=lambda values: np.array([values[0] + 2.0 * values[1]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [tangent[0] + 2.0 * tangent[1]], dtype=np.float64
        ),
        parameter_names=("offset", "phase"),
        trainable=(True, True),
    )
    registry = CustomDerivativeRegistry()
    registry.register(identity, rule)
    calls: list[str] = []

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        calls.append("batching_rule")
        assert axes == (0, None)
        del function
        batch = np.asarray(args[0], dtype=np.float64)
        scale = float(cast(float, args[1]))
        return np.stack([batch[:, 0] + scale * batch[:, 1]], axis=out_axes)

    registry.register_batching_rule(identity, batching_rule)
    batched = vmap(
        lambda row, scale: row[0] + scale * row[1],
        in_axes=(0, None),
        out_axes=1,
        primitive_identity=identity,
        registry=registry,
    )

    result = cast(
        NDArray[np.float64],
        batched(np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64), 2.0),
    )

    assert calls == ["batching_rule"]
    np.testing.assert_allclose(result, [[5.0], [1.0]], atol=1.0e-12)


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


def test_primitive_transform_registry_holds_complete_metadata() -> None:
    """Registry transform bindings should keep derivative and compiler facets together."""

    identity = PrimitiveIdentity("scpn.quantum", "lowered_batch", "1")
    rule = _rule("lowered_batch_rule")

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function, axes
        return np.asarray(args[0], dtype=np.float64).sum(axis=1 + out_axes * 0)

    registry = CustomDerivativeRegistry()
    transform = PrimitiveTransformRule(
        identity=identity,
        derivative_rule=rule,
        batching_rule=batching_rule,
        lowering_rule=_lowering_rule,
        lowering_metadata={"mlir_op": "scpn_diff.lowered_batch", "rust": "blocked"},
        shape_rule=_shape_rule,
        dtype_rule=_dtype_rule,
        static_argument_rule=_static_argument_rule,
        nondifferentiable_policy="fail_closed_at_boundaries",
        effect="pure",
    )

    assert registry.register_transform(transform) is transform
    assert registry.require(identity) is rule
    assert registry.require_batching_rule(identity) is batching_rule
    assert registry.require_lowering_rule(identity) is _lowering_rule
    assert registry.require_shape_rule(identity) is _shape_rule
    assert registry.require_dtype_rule(identity) is _dtype_rule
    assert registry.require_static_argument_rule(identity) is _static_argument_rule
    assert registry.require_nondifferentiable_policy(identity) == "fail_closed_at_boundaries"
    assert registry.require_effect(identity) == "pure"
    assert primitive_shape_rule_for(identity, registry=registry) is _shape_rule
    assert primitive_dtype_rule_for(identity, registry=registry) is _dtype_rule
    assert (
        primitive_nondifferentiable_policy_for(identity, registry=registry)
        == "fail_closed_at_boundaries"
    )
    assert primitive_effect_for(identity, registry=registry) == "pure"
    snapshot = registry.transform_snapshot()
    assert snapshot[identity].lowering_rule is _lowering_rule
    lowering_metadata = snapshot[identity].lowering_metadata
    assert lowering_metadata is not None
    assert lowering_metadata["mlir_op"] == "scpn_diff.lowered_batch"
    assert snapshot[identity].shape_rule is _shape_rule
    assert snapshot[identity].dtype_rule is _dtype_rule
    assert snapshot[identity].static_argument_rule is _static_argument_rule
    assert snapshot[identity].nondifferentiable_policy == "fail_closed_at_boundaries"
    assert snapshot[identity].effect == "pure"
    with pytest.raises(ValueError, match="batching rule already registered"):
        registry.register_batching_rule(identity, batching_rule)
    with pytest.raises(ValueError, match="lowering rule already registered"):
        registry.register_lowering_rule(identity, _lowering_rule)


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


def test_program_ad_registry_dispatch_report_is_complete() -> None:
    """Program AD primitive coverage should resolve through complete registry contracts."""

    report = program_ad_registry_dispatch_coverage_report()

    assert isinstance(report, ProgramADRegistryDispatchCoverageReport)
    assert report.supported is True
    assert report.blocked_identities == ()
    assert report.covered_primitives == report.total_primitives
    assert report.total_primitives > 90
    assert set(report.family_counts) == {
        "array",
        "shape",
        "reduction",
        "stencil",
        "interpolation",
        "assembly",
        "signal",
        "elementwise",
        "selection",
        "product",
        "cumulative",
        "linalg",
    }
    assert report.family_counts["elementwise"] >= 20
    assert all(isinstance(row, ProgramADRegistryDispatchCoverageRow) for row in report.rows)
    assert all(row.complete for row in report.rows)
    assert all(row.blocked_reasons == () for row in report.rows)
    assert all(row.has_batching_rule for row in report.rows)
    assert all(row.has_lowering_metadata for row in report.rows)
    assert all(row.has_shape_rule for row in report.rows)
    assert all(row.has_dtype_rule for row in report.rows)
    assert all(row.has_static_argument_rule for row in report.rows)
    assert all(
        row.nondifferentiable_policy == "program_ad_trace_exact_fail_closed" for row in report.rows
    )
    assert all(row.effect == "pure" for row in report.rows)
    assert any(row.identity == "scpn.program_ad.product:einsum@1" for row in report.rows)
    payload = report.to_dict()
    assert payload["supported"] is True
    assert payload["blocked_identities"] == []
    assert payload["covered_primitives"] == report.total_primitives
    assert "not executable Rust, LLVM, JIT" in str(payload["claim_boundary"])


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


def test_default_registry_helper_unregister_paths_and_missing_vmap_batching() -> None:
    """Default helper bindings should unregister cleanly and vmap should fail closed."""

    batch_identity = PrimitiveIdentity("scpn.quantum", "default_helper_batch", "1")
    batch_rule = CustomDerivativeRule(
        name="default_helper_rule",
        value_fn=lambda values: np.array([values[0] + values[1]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0] + tangent[1]], dtype=np.float64),
    )

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function
        assert axes == (0,)
        batch = np.asarray(args[0], dtype=np.float64)
        return np.moveaxis(batch[:, 0] + batch[:, 1], 0, out_axes)

    with pytest.raises(ValueError, match="no custom derivative rule"):
        register_primitive_batching_rule(batch_identity, batching_rule)
    register_custom_derivative_rule(batch_identity, batch_rule)
    try:
        assert register_primitive_batching_rule(batch_identity, batching_rule) is batching_rule
        batched = vmap(lambda row: row[0] + row[1], primitive_identity=batch_identity)
        batch_result = cast(
            NDArray[np.float64],
            batched(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)),
        )
        np.testing.assert_allclose(batch_result, [3.0, 7.0], atol=1.0e-12)
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(batch_identity) is batch_rule
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.batching_rule_for(batch_identity) is None
        with pytest.raises(ValueError, match="no custom derivative rule"):
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(batch_identity)
    finally:
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.lookup(batch_identity) is not None:
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(batch_identity)

    lowering_identity = PrimitiveIdentity("scpn.quantum", "default_helper_lower", "1")
    lowering_rule_binding = _rule("default_helper_lower_rule")
    with pytest.raises(ValueError, match="no custom derivative rule"):
        register_primitive_lowering_rule(lowering_identity, _lowering_rule)
    register_custom_derivative_rule(lowering_identity, lowering_rule_binding)
    try:
        assert register_primitive_lowering_rule(lowering_identity, _lowering_rule) is (
            _lowering_rule
        )
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_lowering_rule(lowering_identity) is (
            _lowering_rule
        )
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(lowering_identity) is (
            lowering_rule_binding
        )
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.lowering_rule_for(lowering_identity) is None
    finally:
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.lookup(lowering_identity) is not None:
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(lowering_identity)

    missing_identity = PrimitiveIdentity("scpn.quantum", "missing_batch", "1")
    missing_registry = CustomDerivativeRegistry({missing_identity: _rule("missing_batch_rule")})
    with pytest.raises(ValueError, match="no batching rule"):
        vmap(lambda row: row[0], primitive_identity=missing_identity, registry=missing_registry)


def test_primitive_registry_root_exports_match_facade() -> None:
    """Primitive registry and Program AD derivative exports should be stable root APIs."""

    import scpn_quantum_control as scpn

    names = (
        "PrimitiveBatchingRule",
        "PrimitiveContract",
        "PrimitiveDTypeRule",
        "PrimitiveLoweringRule",
        "PrimitiveShapeRule",
        "PrimitiveStaticArgumentRule",
        "PrimitiveTransformRule",
        "ProgramADRegistryDispatchCoverageReport",
        "ProgramADRegistryDispatchCoverageRow",
        "primitive_complete_contract_for",
        "primitive_dtype_rule_for",
        "primitive_effect_for",
        "primitive_contract_for",
        "primitive_nondifferentiable_policy_for",
        "primitive_shape_rule_for",
        "primitive_static_argument_rule_for",
        "program_ad_registry_dispatch_coverage_report",
        "program_ad_stencil_gradient_derivative_rule",
        "program_ad_interpolation_interp_derivative_rule",
        "program_ad_assembly_concatenate_derivative_rule",
        "program_ad_assembly_broadcast_to_derivative_rule",
        "program_ad_assembly_broadcast_arrays_derivative_rule",
        "program_ad_assembly_tril_derivative_rule",
        "program_ad_assembly_triu_derivative_rule",
        "program_ad_assembly_diagonal_derivative_rule",
        "program_ad_assembly_append_derivative_rule",
        "program_ad_assembly_block_derivative_rule",
        "program_ad_assembly_split_derivative_rule",
        "program_ad_assembly_stack_derivative_rule",
        "program_ad_assembly_hstack_derivative_rule",
        "program_ad_assembly_vstack_derivative_rule",
        "program_ad_assembly_column_stack_derivative_rule",
        "program_ad_assembly_dstack_derivative_rule",
        "program_ad_signal_convolve_derivative_rule",
        "program_ad_signal_correlate_derivative_rule",
        "program_ad_array_take_along_axis_derivative_rule",
        "program_ad_array_delete_derivative_rule",
        "program_ad_array_pad_derivative_rule",
        "program_ad_array_insert_derivative_rule",
        "program_ad_linalg_diag_derivative_rule",
        "program_ad_linalg_diagflat_derivative_rule",
        "program_ad_product_inner_derivative_rule",
        "program_ad_product_einsum_derivative_rule",
        "program_ad_product_outer_derivative_rule",
        "program_ad_product_tensordot_derivative_rule",
        "program_ad_selection_clip_derivative_rule",
        "program_ad_selection_where_derivative_rule",
        "program_ad_linalg_matrix_power_derivative_rule",
        "program_ad_linalg_multi_dot_derivative_rule",
        "program_ad_linalg_trace_derivative_rule",
        "register_primitive_batching_rule",
        "register_primitive_lowering_rule",
        "register_primitive_transform_rule",
    )

    for name in names:
        assert getattr(scpn, name) is getattr(differentiable_facade, name)


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
