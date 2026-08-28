# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR whole-program native quality-edge tests
"""Exercise fail-closed whole-program native compiler contract edges."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.compiler import mlir_whole_program_native as native_impl
from scpn_quantum_control.compiler.mlir import (
    DifferentiableMLIRCompileConfig,
    ExecutableWholeProgramADBatchResult,
    ExecutableWholeProgramADKernel,
    MLIRModule,
    NativeWholeProgramADKernel,
    WholeProgramADNativeLoweringReport,
    analyse_whole_program_ad_native_lowering,
    compile_whole_program_ad_trace_to_executable,
    compile_whole_program_ad_trace_to_mlir,
    compile_whole_program_ad_trace_to_native_llvm_jit,
)
from scpn_quantum_control.compiler.mlir_whole_program_emitter import (
    _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES,
    _WHOLE_PROGRAM_NATIVE_FACTORISATION_HELPER_SIZES,
)
from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad
from scpn_quantum_control.whole_program_ad_result import (
    WholeProgramADResult,
    WholeProgramIRNode,
)

FloatArray = NDArray[np.float64]


def _objective(values: FloatArray) -> object:
    """Return a branchless scalar suitable for replay and native compilation."""
    return values[0] ** 2 + values[0] * values[1] + np.sin(values[1])


@pytest.fixture(scope="module")
def scalar_result() -> WholeProgramADResult:
    """Capture one real two-parameter whole-program AD result."""
    return whole_program_value_and_grad(
        _objective,
        np.array([0.75, -0.25], dtype=np.float64),
        parameters=(Parameter("x"), Parameter("y")),
        trace=False,
    )


@pytest.fixture(scope="module")
def replay_kernel() -> ExecutableWholeProgramADKernel:
    """Compile one public executable replay kernel for constructor checks."""
    return compile_whole_program_ad_trace_to_executable(
        _objective,
        np.array([0.75, -0.25], dtype=np.float64),
        (Parameter("x"), Parameter("y")),
        trace=False,
    )


@pytest.fixture(scope="module")
def native_kernel() -> NativeWholeProgramADKernel:
    """Compile one verified public native kernel for boundary checks."""
    return compile_whole_program_ad_trace_to_native_llvm_jit(
        _objective,
        np.array([0.75, -0.25], dtype=np.float64),
        (Parameter("x"), Parameter("y")),
        trace=False,
    )


def _replace_raises(record: Any, match: str, **changes: Any) -> None:
    """Require a frozen public record replacement to fail with its contract error."""
    with pytest.raises(ValueError, match=match):
        replace(record, **changes)


def test_mlir_lowering_covers_optional_payload_and_metadata_paths(
    scalar_result: WholeProgramADResult,
) -> None:
    """Exercise public type refusal and payload-free MLIR emission."""
    with pytest.raises(ValueError, match="requires a WholeProgramADResult"):
        compile_whole_program_ad_trace_to_mlir(cast(Any, object()))

    minimal = replace(
        scalar_result,
        trace_events=(),
        bytecode_instructions=(),
        source_ir_features=(),
        semantics_report=None,
        program_ir=None,
        adjoint_result=None,
    )
    module = compile_whole_program_ad_trace_to_mlir(
        minimal,
        DifferentiableMLIRCompileConfig(
            include_numeric_payload=False,
            include_metadata=False,
        ),
    )
    assert "scpn_diff.value" not in module.text
    assert "scpn.metadata" not in module.text
    assert module.resource_counts["program_ad_effects"] == 0


def test_batch_result_rejects_every_inconsistent_public_field() -> None:
    """Cover every immutable batched-result validation boundary."""
    valid = ExecutableWholeProgramADBatchResult(
        values=np.array([1.0, 2.0], dtype=np.float64),
        gradients=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        parameter_names=("x", "y"),
        row_signatures=(("sig",), ("sig",)),
        mlir_sha256="digest",
    )
    cases: list[tuple[dict[str, Any], str]] = [
        ({"gradients": np.array([1.0, 2.0])}, "two-dimensional"),
        ({"gradients": np.ones((1, 2))}, "row count"),
        ({"gradients": np.ones((2, 1))}, "column count"),
        ({"gradients": np.array([[np.nan, 0.0], [0.0, 1.0]])}, "finite"),
        ({"row_signatures": (("sig",),)}, "row_signatures count"),
        ({"row_signatures": (("",), ("sig",))}, "non-empty strings"),
        ({"mlir_sha256": ""}, "mlir_sha256"),
        ({"backend": "python"}, "backend"),
        ({"claim_boundary": ""}, "claim_boundary"),
    ]
    for changes, match in cases:
        _replace_raises(valid, match, **changes)
    np.testing.assert_allclose(valid.values, [1.0, 2.0])
    np.testing.assert_allclose(valid.gradients, np.eye(2))


def test_lowering_report_rejects_inconsistent_operation_accounting() -> None:
    """Cover every public lowering-report validation boundary."""
    valid = WholeProgramADNativeLoweringReport(
        supported=True,
        lowerable_ops=("parameter",),
        unsupported_ops=(),
        control_flow_ops=(),
        effect_kinds=(),
        operation_count=1,
        lowerable_operation_count=1,
        unsupported_operation_count=0,
        fail_closed_reason="supported",
    )
    for field_name in ("lowerable_ops", "unsupported_ops", "control_flow_ops", "effect_kinds"):
        _replace_raises(valid, "non-empty strings", **{field_name: ("",)})
    cases: list[tuple[dict[str, Any], str]] = [
        ({"operation_count": 0}, "positive"),
        ({"lowerable_operation_count": -1}, "lowerable_operation_count"),
        ({"unsupported_operation_count": -1}, "unsupported_operation_count"),
        ({"operation_count": 2}, "partition"),
        ({"supported": False}, "supported must match"),
        ({"fail_closed_reason": ""}, "fail_closed_reason"),
    ]
    for changes, match in cases:
        _replace_raises(valid, match, **changes)
    assert valid.as_metadata()["operation_count"] == 1


def test_executable_kernel_rejects_invalid_constructor_contracts(
    replay_kernel: ExecutableWholeProgramADKernel,
) -> None:
    """Exercise all public replay-kernel constructor refusals."""
    bad_source = replace(replay_kernel.source_result)
    object.__setattr__(bad_source, "gradient", np.array([1.0], dtype=np.float64))
    cases: list[tuple[dict[str, Any], str]] = [
        ({"objective": cast(Callable[[Any], object], 0)}, "objective"),
        ({"source_result": cast(WholeProgramADResult, object())}, "source_result"),
        ({"mlir_module": cast(MLIRModule, object())}, "mlir_module"),
        ({"parameters": ()}, "parameters"),
        ({"parameters": (cast(Parameter, object()),)}, "parameters"),
        ({"parameter_names": ("other", "y")}, "match parameters"),
        (
            {
                "parameters": (Parameter("other"), Parameter("y")),
                "parameter_names": ("other", "y"),
            },
            "match source_result",
        ),
        ({"parameter_shape": (1,)}, "parameter_shape"),
        ({"source_result": bad_source}, "gradient shape"),
        ({"branch_signature": ("",)}, "non-empty strings"),
        ({"branch_signature": ("different",)}, "match source_result"),
        ({"backend": "native_llvm_jit"}, "backend"),
        ({"claim_boundary": ""}, "claim_boundary"),
    ]
    for changes, match in cases:
        _replace_raises(replay_kernel, match, **changes)
    with pytest.raises(ValueError, match="parameter shape"):
        replay_kernel.value([0.75])


def test_native_kernel_rejects_invalid_constructor_contracts(
    native_kernel: NativeWholeProgramADKernel,
) -> None:
    """Exercise all verified native-kernel constructor refusals."""
    functions = dict(native_kernel.native_functions)
    missing_functions = dict(functions)
    del missing_functions["value"]
    noncallable_functions = dict(functions)
    noncallable_functions["value"] = 0
    failed_verification = replace(native_kernel.verification, value_close=False)
    unsupported_report = replace(
        native_kernel.lowering_report,
        supported=False,
        unsupported_ops=("unknown",),
        operation_count=native_kernel.lowering_report.operation_count + 1,
        unsupported_operation_count=1,
        fail_closed_reason="unsupported",
    )
    bad_source = replace(native_kernel.source_result)
    object.__setattr__(bad_source, "gradient", np.array([1.0], dtype=np.float64))
    cases: list[tuple[dict[str, Any], str]] = [
        ({"objective": cast(Callable[[Any], object], 0)}, "objective"),
        ({"source_result": cast(WholeProgramADResult, object())}, "source_result"),
        ({"mlir_module": cast(MLIRModule, object())}, "mlir_module"),
        ({"llvm_ir": ""}, "llvm_ir"),
        ({"native_functions": missing_functions}, "missing value"),
        ({"native_functions": noncallable_functions}, "value must be callable"),
        ({"verification": cast(Any, object())}, "verification"),
        ({"verification": failed_verification}, "verification failed"),
        ({"parameters": ()}, "parameters"),
        ({"parameters": (cast(Parameter, object()),)}, "parameters"),
        ({"parameter_names": ("other", "y")}, "match parameters"),
        (
            {
                "parameters": (Parameter("other"), Parameter("y")),
                "parameter_names": ("other", "y"),
            },
            "match source_result",
        ),
        ({"parameter_shape": (1,)}, "parameter_shape"),
        ({"source_result": bad_source}, "gradient shape"),
        ({"trace_signature": ("different",)}, "trace_signature"),
        ({"supported_ops": ("",)}, "supported_ops"),
        ({"lowering_report": cast(Any, object())}, "lowering_report"),
        ({"lowering_report": unsupported_report}, "supported native trace"),
        ({"supported_ops": ("parameter",)}, "supported_ops must match"),
        ({"cache_key": "short"}, "sha256"),
        ({"cache_hit": cast(bool, 1)}, "cache_hit"),
        ({"backend": "python"}, "backend"),
        ({"claim_boundary": ""}, "claim_boundary"),
    ]
    for changes, match in cases:
        _replace_raises(native_kernel, match, **changes)


def test_native_kernel_public_batch_inputs_fail_closed(
    native_kernel: NativeWholeProgramADKernel,
) -> None:
    """Exercise public native batch shape and finiteness boundaries."""
    valid = np.array([[0.75, -0.25], [0.5, 0.25]], dtype=np.float64)
    with pytest.raises(ValueError, match="parameter shape"):
        native_kernel.value([0.75])
    for values, match in (
        (np.array([0.75, -0.25]), "two-dimensional"),
        (np.empty((0, 2)), "at least one row"),
        (np.ones((2, 1)), "parameter shape"),
        (np.array([[0.75, np.nan]]), "finite"),
    ):
        with pytest.raises(ValueError, match=match):
            native_kernel.batch_value(values)
    for tangents, match in (
        (np.array([1.0, 1.0]), "two-dimensional"),
        (np.ones((1, 2)), "match batch values"),
        (np.array([[1.0, np.nan], [1.0, 1.0]]), "finite"),
    ):
        with pytest.raises(ValueError, match=match):
            native_kernel.batch_jvp(valid, tangents)
    column_cotangents = np.array([[1.0], [0.5]], dtype=np.float64)
    assert native_kernel.batch_vjp(valid, column_cotangents).shape == (2, 2)
    for cotangents, match in (
        (np.ones((2, 2)), "one-dimensional"),
        (np.ones(1), "row count"),
        (np.array([1.0, np.nan]), "finite"),
    ):
        with pytest.raises(ValueError, match=match):
            native_kernel.batch_vjp(valid, cotangents)


def test_public_compilers_and_lowering_analysis_reject_invalid_inputs(
    scalar_result: WholeProgramADResult,
) -> None:
    """Cover public callable, type, empty-IR, and control-flow refusals."""
    for compiler, match in (
        (compile_whole_program_ad_trace_to_executable, "executable AD objective"),
        (compile_whole_program_ad_trace_to_native_llvm_jit, "native AD objective"),
    ):
        with pytest.raises(ValueError, match=match):
            compiler(cast(Callable[[Any], object], 0), [1.0])
    with pytest.raises(ValueError, match="requires a WholeProgramADResult"):
        analyse_whole_program_ad_native_lowering(cast(Any, object()))
    empty_ir = replace(scalar_result, ir_nodes=(), adjoint_result=None, program_ir=None)
    with pytest.raises(ValueError, match="requires captured IR nodes"):
        analyse_whole_program_ad_native_lowering(empty_ir)
    loop_node = WholeProgramIRNode(
        index=0,
        op="loop:body",
        inputs=("x",),
        value=0.0,
        tangent=np.zeros(2, dtype=np.float64),
    )
    loop_result = replace(
        scalar_result,
        ir_nodes=(loop_node,),
        adjoint_result=None,
        program_ir=None,
    )
    assert native_impl._whole_program_has_unsupported_native_control_flow(loop_result)
    assert analyse_whole_program_ad_native_lowering(loop_result).effect_kinds == ()


def test_cache_payload_and_helper_boundaries_are_deterministic(
    scalar_result: WholeProgramADResult,
    native_kernel: NativeWholeProgramADKernel,
) -> None:
    """Exercise cache normalization, eviction, and determinant-helper refusals."""
    payload = native_impl._jsonable_cache_payload(
        {
            "sequence": (np.array([np.float64(1.25)]),),
            "float": 0.5,
            "other": object(),
        }
    )
    assert isinstance(payload, dict)
    assert payload["float"] == "5.00000000000000000e-01"
    assert isinstance(payload["other"], str)
    assert native_impl._jsonable_cache_payload(np.float64(1.25)) == ("1.25000000000000000e+00")

    entry = native_impl._NativeWholeProgramADCacheEntry(
        mlir_module=native_kernel.mlir_module,
        llvm_ir=native_kernel.llvm_ir,
        native_functions=native_kernel.native_functions,
        verification=native_kernel.verification,
        supported_ops=native_kernel.supported_ops,
        lowering_report=native_kernel.lowering_report,
    )
    native_impl.clear_native_whole_program_ad_compile_cache()
    try:
        native_impl._store_native_whole_program_ad_cache_entry("same", entry)
        native_impl._store_native_whole_program_ad_cache_entry("same", entry)
        for index in range(native_impl._NATIVE_WHOLE_PROGRAM_AD_CACHE_MAXSIZE + 1):
            native_impl._store_native_whole_program_ad_cache_entry(f"key-{index}", entry)
        stats = native_impl.native_whole_program_ad_compile_cache_stats()
        assert stats["entries"] == native_impl._NATIVE_WHOLE_PROGRAM_AD_CACHE_MAXSIZE
    finally:
        native_impl.clear_native_whole_program_ad_compile_cache()

    helper_nodes = (
        WholeProgramIRNode(
            index=0,
            op="linalg:inv:5x5:0:0",
            inputs=("matrix",),
            value=1.0,
            tangent=np.zeros(2, dtype=np.float64),
        ),
        WholeProgramIRNode(
            index=1,
            op="linalg:det:6x6",
            inputs=("matrix",),
            value=1.0,
            tangent=np.zeros(2, dtype=np.float64),
        ),
        WholeProgramIRNode(
            index=2,
            op="linalg:det:7x7",
            inputs=("matrix",),
            value=1.0,
            tangent=np.zeros(2, dtype=np.float64),
        ),
    )
    helper_result = replace(
        scalar_result,
        ir_nodes=helper_nodes,
        adjoint_result=None,
        program_ir=None,
    )
    helper_ir = native_impl._compile_whole_program_native_helper_definitions(helper_result)
    assert "" in helper_ir
    assert (
        _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES
        <= _WHOLE_PROGRAM_NATIVE_FACTORISATION_HELPER_SIZES
    )
    with pytest.raises(ValueError, match="unsupported size"):
        native_impl._compile_whole_program_native_det_loop_helper_llvm_ir(99)


def test_internal_native_ffi_boundaries_reject_invalid_and_nonfinite_outputs() -> None:
    """Exercise defensive FFI checks beneath already-covered public kernels."""
    values = np.array([1.0, 2.0], dtype=np.float64)

    def unary_nan(_values: Any, output: Any) -> None:
        output[0] = np.nan

    def binary_nan(_values: Any, _vector: Any, output: Any) -> None:
        output[0] = np.nan

    with pytest.raises(ValueError, match="output_size"):
        native_impl._call_native_whole_program_unary(unary_nan, values, 0)
    with pytest.raises(ValueError, match="output must be finite"):
        native_impl._call_native_whole_program_unary(unary_nan, values, 1)
    with pytest.raises(ValueError, match="output_size"):
        native_impl._call_native_whole_program_binary(binary_nan, values, values, 0)
    with pytest.raises(ValueError, match="output must be finite"):
        native_impl._call_native_whole_program_binary(binary_nan, values, values, 1)

    def batch_values_nan(_values: Any, _rows: int, out_values: Any, _gradients: Any) -> None:
        out_values[0] = np.nan

    def batch_vector_nan(_values: Any, _vectors: Any, _rows: int, output: Any) -> None:
        output[0] = np.nan

    batch = np.ones((1, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="two-dimensional"):
        native_impl._call_native_whole_program_batch_value_gradient(
            batch_values_nan,
            batch.reshape(-1),
            2,
        )
    with pytest.raises(ValueError, match="batch output must be finite"):
        native_impl._call_native_whole_program_batch_value_gradient(batch_values_nan, batch, 2)
    for bad_values, match in (
        (np.empty((0, 2)), "at least one row"),
        (np.ones((1, 1)), "parameter count mismatch"),
        (np.array([[1.0, np.nan]]), "values must be finite"),
    ):
        with pytest.raises(ValueError, match=match):
            native_impl._call_native_whole_program_batch_value_gradient(
                batch_values_nan,
                bad_values,
                2,
            )
    for bad_values, tangents, match in (
        (batch.reshape(-1), batch, "two-dimensional"),
        (np.empty((0, 2)), np.empty((0, 2)), "at least one row"),
        (batch, np.ones((1, 1)), "tangents must match"),
        (np.ones((1, 1)), np.ones((1, 1)), "parameter count mismatch"),
        (np.array([[1.0, np.nan]]), batch, "inputs must be finite"),
    ):
        with pytest.raises(ValueError, match=match):
            native_impl._call_native_whole_program_batch_jvp(
                batch_vector_nan,
                bad_values,
                tangents,
                2,
            )
    with pytest.raises(ValueError, match="batch JVP output must be finite"):
        native_impl._call_native_whole_program_batch_jvp(batch_vector_nan, batch, batch, 2)
    for bad_values, cotangents, match in (
        (batch.reshape(-1), np.ones(1), "two-dimensional"),
        (np.empty((0, 2)), np.empty(0), "at least one row"),
        (np.ones((1, 1)), np.ones(1), "parameter count mismatch"),
        (batch, np.ones(2), "cotangent row count mismatch"),
        (np.array([[1.0, np.nan]]), np.ones(1), "inputs must be finite"),
    ):
        with pytest.raises(ValueError, match=match):
            native_impl._call_native_whole_program_batch_vjp(
                batch_vector_nan,
                bad_values,
                cotangents,
                2,
            )
    with pytest.raises(ValueError, match="batch VJP output must be finite"):
        native_impl._call_native_whole_program_batch_vjp(
            batch_vector_nan,
            batch,
            np.ones(1, dtype=np.float64),
            2,
        )


def test_private_compile_and_annotation_defenses_fail_closed(
    scalar_result: WholeProgramADResult,
) -> None:
    """Exercise defensive native compiler and MLIR terminator checks."""
    empty = replace(
        scalar_result,
        gradient=np.array([], dtype=np.float64),
        parameter_names=(),
        trainable=(),
        adjoint_result=None,
    )
    with pytest.raises(ValueError, match="requires parameters"):
        native_impl._compile_whole_program_ad_native_llvm_ir(empty, "empty")
    unsupported_node = WholeProgramIRNode(
        index=0,
        op="unsupported:quality-edge",
        inputs=("x",),
        value=0.0,
        tangent=np.zeros(2, dtype=np.float64),
    )
    unsupported = replace(
        scalar_result,
        ir_nodes=(unsupported_node,),
        adjoint_result=None,
        program_ir=None,
    )
    with pytest.raises(ValueError, match="failed closed"):
        native_impl._compile_whole_program_ad_native_llvm_ir(unsupported, "unsupported")

    text = "module return\n"
    malformed = MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        dialect="scpn_diff",
        resource_counts={},
    )
    with pytest.raises(ValueError, match="module terminator"):
        native_impl._annotate_whole_program_native_mlir(malformed, "llvm", scalar_result)


def test_static_linalg_classification_covers_verified_special_cases() -> None:
    """Keep verified product-linalg special cases outside the blocked set."""
    assert not native_impl._whole_program_native_is_unverified_static_linalg_op(
        "linalg:matrix_power:2x2:power:2:0:0"
    )
    assert not native_impl._whole_program_native_is_unverified_static_linalg_op(
        "linalg:multi_dot:2x2__2x2:out:2x2:0:0"
    )
    assert native_impl._whole_program_native_is_unverified_static_linalg_op(
        "linalg:matrix_power:3x3:power:2:0:0"
    )
