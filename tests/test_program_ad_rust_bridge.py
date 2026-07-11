# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for Program AD Rust bridge wrappers
"""Facade, registry, and native baseline tests for Program AD Rust bridge wrappers."""

from __future__ import annotations

import json
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from _program_ad_rust_bridge_test_fixtures import _IR, _install_fake_engine

from scpn_quantum_control import differentiable as differentiable_facade
from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad
from scpn_quantum_control.program_ad_rust_bridge import (
    RustProgramADInterpreterResult,
    RustProgramADRegistryMetadataMirrorResult,
    RustProgramADValueAndGradientResult,
    interpret_program_ad_effect_ir_with_rust,
    mirror_program_ad_registry_metadata_with_rust,
    value_and_grad_program_ad_effect_ir_with_rust,
)


def test_value_and_gradient_bridge_is_shared_by_module_and_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct and facade imports should call the same extracted Rust bridge."""
    calls: list[tuple[str, list[float]]] = []
    fake_engine = ModuleType("scpn_quantum_engine")

    def replay(serialization: str, inputs: list[float]) -> str:
        calls.append((serialization, inputs))
        return json.dumps(
            {
                "supported": True,
                "value": 3.5,
                "gradient": [1.0, -2.0],
                "parameter_targets": ["%0", "%1"],
                "effect_count": 4,
                "supported_effect_count": 4,
                "blocked_reasons": [],
                "claim_boundary": "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit",
            }
        )

    engine_exports = cast(Any, fake_engine)
    engine_exports.program_ad_effect_ir_interpret_value_and_gradient = replay
    _install_fake_engine(monkeypatch, fake_engine)
    program_ir = _IR(serialization="program-ir-json")

    direct = value_and_grad_program_ad_effect_ir_with_rust(
        program_ir,
        np.array([0.25, -0.5], dtype=np.float64),
    )
    facade = differentiable_facade.value_and_grad_program_ad_effect_ir_with_rust(
        program_ir,
        [0.25, -0.5],
    )

    assert isinstance(direct, RustProgramADValueAndGradientResult)
    assert isinstance(facade, RustProgramADValueAndGradientResult)
    assert calls == [
        ("program-ir-json", [0.25, -0.5]),
        ("program-ir-json", [0.25, -0.5]),
    ]
    assert direct.supported == facade.supported
    assert direct.value == facade.value
    assert direct.parameter_targets == facade.parameter_targets
    assert direct.effect_count == facade.effect_count
    assert direct.supported_effect_count == facade.supported_effect_count
    assert direct.blocked_reasons == facade.blocked_reasons
    assert direct.claim_boundary == facade.claim_boundary
    np.testing.assert_allclose(direct.gradient, np.array([1.0, -2.0], dtype=np.float64))
    np.testing.assert_allclose(direct.gradient, facade.gradient)


def test_forward_interpreter_bridge_normalises_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward interpreter wrapper should parse Rust JSON into a typed result."""
    fake_engine = ModuleType("scpn_quantum_engine")

    def replay(serialization: str, inputs: list[float]) -> str:
        assert serialization == "program-ir-json"
        assert inputs == [2.0]
        return json.dumps(
            {
                "supported": True,
                "value": 9.0,
                "effect_count": 3,
                "supported_effect_count": 3,
                "blocked_reasons": [],
                "claim_boundary": "bounded_rust_program_ad_ir_scalar_static_signal_static_interpolation_static_stencil_static_cumulative_and_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit",
            }
        )

    engine_exports = cast(Any, fake_engine)
    engine_exports.program_ad_effect_ir_interpret_forward = replay
    _install_fake_engine(monkeypatch, fake_engine)

    result = interpret_program_ad_effect_ir_with_rust("program-ir-json", [2])

    assert isinstance(result, RustProgramADInterpreterResult)
    assert result.supported is True
    assert result.value == pytest.approx(9.0)
    assert result.supported_effect_count == 3
    assert "no_llvm_jit" in result.claim_boundary


def test_registry_metadata_mirror_is_shared_by_module_and_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct and facade imports should share the Rust registry metadata mirror."""
    calls: list[dict[str, object]] = []
    fake_engine = ModuleType("scpn_quantum_engine")

    def mirror(snapshot: str) -> str:
        payload = json.loads(snapshot)
        assert isinstance(payload, dict)
        assert payload["supported"] is True
        assert payload["covered_primitives"] == payload["total_primitives"] == 118
        assert payload["family_counts"] == {
            "array": 6,
            "shape": 17,
            "reduction": 11,
            "stencil": 1,
            "interpolation": 1,
            "assembly": 21,
            "signal": 2,
            "elementwise": 24,
            "selection": 11,
            "product": 7,
            "cumulative": 3,
            "linalg": 14,
        }
        assert len(cast(list[object], payload["rows"])) == 118
        calls.append(payload)
        return json.dumps(
            {
                "supported": True,
                "primitive_count": 118,
                "covered_primitives": 118,
                "family_counts": payload["family_counts"],
                "facet_counts": {
                    "derivative_rule": 118,
                    "batching_rule": 118,
                    "lowering_metadata": 118,
                    "shape_rule": 118,
                    "dtype_rule": 118,
                    "static_argument_rule": 118,
                    "nondifferentiable_policy": 118,
                    "effect": 118,
                },
                "executable_operation_count": 4,
                "executable_operations": ["det", "sin", "sqrt", "tanh"],
                "blocked_reasons": [],
                "claim_boundary": "rust_program_ad_registry_metadata_mirror_only_no_execution_promotion",
            }
        )

    engine_exports = cast(Any, fake_engine)
    engine_exports.program_ad_registry_metadata_mirror = mirror
    _install_fake_engine(monkeypatch, fake_engine)

    direct = mirror_program_ad_registry_metadata_with_rust()
    facade = differentiable_facade.mirror_program_ad_registry_metadata_with_rust()

    assert isinstance(direct, RustProgramADRegistryMetadataMirrorResult)
    assert isinstance(facade, RustProgramADRegistryMetadataMirrorResult)
    assert len(calls) == 2
    assert direct.supported is True
    assert direct.primitive_count == 118
    assert direct.family_counts["elementwise"] == 24
    assert direct.facet_counts["lowering_metadata"] == 118
    assert direct.executable_operations == ("det", "sin", "sqrt", "tanh")
    assert direct.as_dict() == facade.as_dict()


def test_rust_program_ad_value_and_gradient_replay_matches_python_trace() -> None:
    """Rust Program AD scalar replay should match the emitted Python trace."""
    pytest.importorskip("scpn_quantum_engine")
    values = np.array([0.4, -0.2], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y = trace_values
        return x * x + 2.0 * y + np.sin(x)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y")),
    )
    assert result.program_ir is not None

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert isinstance(rust_result, RustProgramADValueAndGradientResult)
    assert rust_result.supported is True
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    assert len(rust_result.parameter_targets) == values.size
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_registry_metadata_mirror_validates_python_registry() -> None:
    """Rust registry metadata mirror should validate Python registry coverage."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_registry_metadata_mirror", None))

    mirror = mirror_program_ad_registry_metadata_with_rust()

    assert mirror.supported is True, mirror.blocked_reasons
    assert mirror.primitive_count == 118
    assert mirror.covered_primitives == 118
    assert mirror.family_counts["elementwise"] == 24
    assert mirror.family_counts["linalg"] == 14
    assert mirror.facet_counts["derivative_rule"] == 118
    assert mirror.facet_counts["lowering_metadata"] == 118
    assert {
        "sin",
        "sqrt",
        "tanh",
        "det",
        "diag",
        "diagflat",
        "matrix_power",
        "multi_dot",
        "pinv",
        "solve",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "gradient",
        "svd",
    } <= set(mirror.executable_operations)
    assert mirror.executable_operation_count == len(mirror.executable_operations)
    assert (
        mirror.claim_boundary
        == "rust_program_ad_registry_metadata_mirror_only_no_execution_promotion"
    )
