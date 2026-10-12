# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — executed native and cross-IR contracts
"""Public interpreter, LLVM and Rust execution against independent finite oracles."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
from scpn_quantum_control.compiler import (
    ExecutableWholeProgramADKernel,
    NativeWholeProgramADExecutionCase,
    NativeWholeProgramADKernel,
    clear_native_whole_program_ad_compile_cache,
    compile_whole_program_ad_trace_to_executable,
    compile_whole_program_ad_trace_to_mlir,
    compile_whole_program_ad_trace_to_native_llvm_jit,
)
from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad

FloatArray = NDArray[np.float64]
Kernel = ExecutableWholeProgramADKernel | NativeWholeProgramADKernel
PARAMETERS: tuple[Parameter, ...] = (Parameter("x"), Parameter("y"))
UNIT_PATH = (
    Path(__file__).resolve().parents[1] / "data/studio/program_ad_replay_rational_20260714.json"
)


def _polynomial(values: FloatArray) -> object:
    """Evaluate the rational scalar objective captured by all three public routes."""
    return values[0] * values[0] + 2.0 * values[1]


def _matrix_product(values: FloatArray) -> object:
    """Evaluate the sum of a row-major two-by-two matrix product."""
    left = np.reshape(values[:4], (2, 2))
    right = np.reshape(values[4:], (2, 2))
    return np.sum(np.linalg.multi_dot((left, right)))


def _rational_ir() -> str:
    """Read the unchanged committed scalar effect IR shared with the WASM consumer."""
    unit = cast(dict[str, object], json.loads(UNIT_PATH.read_text(encoding="utf-8")))
    program = unit["program"]
    assert isinstance(program, dict)
    serialization = program["effect_ir"]
    assert isinstance(serialization, str)
    return serialization


@pytest.fixture
def polynomial_kernels() -> tuple[Kernel, Kernel]:
    """Compile actual interpreter and LLVM kernels for the same finite scalar input."""
    sample = np.array([3.0, 5.0], dtype=np.float64)
    return (
        compile_whole_program_ad_trace_to_executable(_polynomial, sample, PARAMETERS),
        compile_whole_program_ad_trace_to_native_llvm_jit(_polynomial, sample, PARAMETERS),
    )


@pytest.mark.parametrize("family", ["scalar", "matrix"])
def test_native_program_execution_01(family: str) -> None:
    """Executed interpreter and machine code agree with hand-derived scalar/matrix oracles."""
    if family == "scalar":
        objective = _polynomial
        sample = np.array([3.0, 5.0], dtype=np.float64)
        replay = np.array([-2.0, 1.0], dtype=np.float64)
        parameters = PARAMETERS
        expected_value = 6.0
        expected_gradient = [-4.0, 2.0]
    else:
        objective = _matrix_product
        sample = np.arange(1.0, 9.0, dtype=np.float64)
        replay = np.array([2.0, -1.0, 0.5, 3.0, -2.0, 4.0, 1.0, 0.25], dtype=np.float64)
        parameters = tuple(Parameter(f"entry_{index}") for index in range(8))
        expected_value = 7.5
        expected_gradient = [2.0, 1.25, 2.0, 1.25, 2.5, 2.5, 2.0, 2.0]
    for factory in (
        compile_whole_program_ad_trace_to_executable,
        compile_whole_program_ad_trace_to_native_llvm_jit,
    ):
        kernel = factory(objective, sample, parameters)
        value, gradient = kernel.value_and_grad(replay)
        assert value == pytest.approx(expected_value, abs=1e-12)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=0.0, atol=1e-12)
        assert kernel.parameter_names == tuple(parameter.name for parameter in parameters)
        assert gradient.dtype == np.float64
        if family == "matrix":
            assert kernel.value(sample) == 134.0
            np.testing.assert_array_equal(kernel.gradient(sample), [11, 15, 11, 15, 4, 4, 6, 6])


@pytest.mark.parametrize("values", [[], [1.0], [1.0, 2.0, 3.0], [[1.0, 2.0]]])
def test_native_program_execution_02(
    values: list[float] | list[list[float]], polynomial_kernels: tuple[Kernel, Kernel]
) -> None:
    """Bad parameter dimensions raise a typed error and leave both kernels usable."""
    bad = np.asarray(values, dtype=np.float64)
    before = bad.tobytes()
    for kernel in polynomial_kernels:
        with pytest.raises(ValueError):
            kernel.value_and_grad(bad)
        assert bad.tobytes() == before
        assert kernel.value([3.0, 5.0]) == 19.0


def test_native_program_execution_03(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent LLVM compiler explicitly refuses native execution without input writes."""
    sample = np.array([3.0, 5.0], dtype=np.float64)
    before = sample.copy()
    clear_native_whole_program_ad_compile_cache()
    monkeypatch.setitem(sys.modules, "llvmlite.binding", None)
    with pytest.raises(ValueError, match="requires llvmlite.binding"):
        compile_whole_program_ad_trace_to_native_llvm_jit(_polynomial, sample, PARAMETERS)
    np.testing.assert_array_equal(sample, before)
    assert (
        compile_whole_program_ad_trace_to_executable(_polynomial, sample, PARAMETERS).value(sample)
        == 19.0
    )


def test_native_program_execution_04(monkeypatch: pytest.MonkeyPatch) -> None:
    """Actual MLIR text emission cannot satisfy the executed native evidence contract."""
    sample = np.array([3.0, 5.0], dtype=np.float64)
    captured = whole_program_value_and_grad(_polynomial, sample, parameters=PARAMETERS)
    emitted = compile_whole_program_ad_trace_to_mlir(captured)
    assert emitted.text
    assert emitted.metadata["polyglot_targets"]["llvm"].startswith("blocked")
    with pytest.raises(ValueError, match="status"):
        NativeWholeProgramADExecutionCase(
            case_id="polynomial",
            operation_family="scalar",
            operand_dimension=2,
            status="emitted",
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            native_symbol=None,
            failure_class=None,
            claim_boundary="Compiler text alone carries no executed native evidence",
        )
    clear_native_whole_program_ad_compile_cache()
    monkeypatch.setitem(sys.modules, "llvmlite.binding", None)
    with pytest.raises(ValueError, match="requires llvmlite.binding"):
        compile_whole_program_ad_trace_to_native_llvm_jit(_polynomial, sample, PARAMETERS)


def test_native_parameter_layout_dtype_and_derivative_abi(
    polynomial_kernels: tuple[Kernel, Kernel],
) -> None:
    """Strided vectors and numeric lists preserve order, float64 outputs and native derivative ABI."""
    storage = np.array([0.5, 91.0, -4.0, 92.0], dtype=np.float64)
    strided = storage[::2]
    assert not strided.flags.c_contiguous
    before = storage.copy()
    for kernel in polynomial_kernels:
        for values in (strided, [float(value) for value in strided]):
            value, gradient = kernel.value_and_grad(values)
            assert value == -7.75
            np.testing.assert_array_equal(gradient, [1.0, 2.0])
            assert gradient.dtype == np.float64
            if isinstance(kernel, NativeWholeProgramADKernel):
                assert kernel.jvp(values, [0.25, -0.5]) == -0.75
                np.testing.assert_array_equal(kernel.vjp(values, [3.0]), [3.0, 6.0])
        batch = kernel.batch_value_and_grad([[3.0, 5.0], [-2.0, 1.0], [0.5, -4.0]])
        np.testing.assert_array_equal(batch.values, [19.0, 6.0, -7.75])
        np.testing.assert_array_equal(batch.gradients, [[6, 2], [-4, 2], [1, 2]])
    np.testing.assert_array_equal(storage, before)


def test_native_engine_lifetime_survives_cache_eviction() -> None:
    """A retained callable owns its real LLVM engine after cache clear and collection."""
    sample = np.array([3.0, 5.0], dtype=np.float64)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(_polynomial, sample, PARAMETERS)
    del sample
    assert clear_native_whole_program_ad_compile_cache() >= 1
    gc.collect()
    for values, expected in (([-2.0, 1.0], 6.0), ([0.5, -4.0], -7.75)):
        assert kernel.value(values) == expected
    np.testing.assert_array_equal(kernel.gradient([-2.0, 1.0]), [-4.0, 2.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e308])
def test_native_nonfinite_and_overflow_refusal(
    bad: float,
    polynomial_kernels: tuple[Kernel, Kernel],
) -> None:
    """Invalid input or overflowing arithmetic raises before returning a claimed result."""
    values = np.array([bad, 5.0], dtype=np.float64)
    before = values.tobytes()
    for kernel in polynomial_kernels:
        with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
            kernel.value_and_grad(values)
        assert values.tobytes() == before
        assert kernel.value([3.0, 5.0]) == 19.0


@pytest.mark.parametrize(
    "values, expected, gradient",
    [
        ([3.0, 5.0], 19.0, [6.0, 2.0]),
        ([-2.0, 1.0], 6.0, [-4.0, 2.0]),
        ([0.5, -4.0], -7.75, [1.0, 2.0]),
    ],
)
def test_real_rust_scalar_replay(
    values: list[float],
    expected: float,
    gradient: list[float],
) -> None:
    """The actual native extension executes the shared IR against independent rational answers."""
    inputs = np.array(values, dtype=np.float64)
    before = inputs.copy()
    result = scpn.value_and_grad_program_ad_effect_ir_with_rust(_rational_ir(), inputs)
    assert result.supported, result.blocked_reasons
    assert result.value == expected
    np.testing.assert_array_equal(result.gradient, gradient)
    assert result.parameter_targets == ("%0", "%1")
    np.testing.assert_array_equal(inputs, before)


@pytest.mark.parametrize("values", [[3.0], [3.0, 5.0, 7.0], [np.nan, 5.0], [np.inf, 5.0]])
def test_rust_malformed_input_preserves_recovery(values: list[float]) -> None:
    """Rust facade rejects wrong arity/nonfinite input without mutating caller state."""
    inputs = np.array(values, dtype=np.float64)
    before = inputs.tobytes()
    if np.all(np.isfinite(inputs)):
        result = scpn.value_and_grad_program_ad_effect_ir_with_rust(_rational_ir(), inputs)
        assert not result.supported
        assert result.value is None
    else:
        with pytest.raises(ValueError):
            scpn.value_and_grad_program_ad_effect_ir_with_rust(_rational_ir(), inputs)
    assert inputs.tobytes() == before
    recovered = scpn.value_and_grad_program_ad_effect_ir_with_rust(_rational_ir(), [3.0, 5.0])
    assert recovered.supported and recovered.value == 19.0


def test_rust_runtime_absence_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing native extension yields an explicit unavailable result rather than Python fallback."""
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)
    result = scpn.value_and_grad_program_ad_effect_ir_with_rust(_rational_ir(), [3.0, 5.0])
    assert not result.supported
    assert result.value is None and result.gradient.size == 0
    assert result.blocked_reasons == ("scpn_quantum_engine native extension is not built",)


def test_real_rust_matrix_replay_and_shape_refusal() -> None:
    """Native Rust executes the captured matrix IR and safely refuses wrong input arity."""
    sample = np.arange(1.0, 9.0, dtype=np.float64)
    parameters = tuple(Parameter(f"entry_{index}") for index in range(8))
    captured = whole_program_value_and_grad(_matrix_product, sample, parameters=parameters)
    assert captured.program_ir is not None
    result = scpn.value_and_grad_program_ad_effect_ir_with_rust(captured.program_ir, sample)
    assert result.supported, result.blocked_reasons
    assert result.value == 134.0
    np.testing.assert_array_equal(result.gradient, [11, 15, 11, 15, 4, 4, 6, 6])
    malformed = sample[:-1]
    before = malformed.copy()
    refused = scpn.value_and_grad_program_ad_effect_ir_with_rust(captured.program_ir, malformed)
    assert not refused.supported and refused.value is None
    np.testing.assert_array_equal(malformed, before)
    recovered = scpn.value_and_grad_program_ad_effect_ir_with_rust(captured.program_ir, sample)
    assert recovered.supported and recovered.value == 134.0


def test_rust_strided_input_and_overflow_recovery() -> None:
    """Rust receives ordered scalar values from a view and refuses overflow without fallback."""
    storage = np.array([0.5, 91.0, -4.0, 92.0], dtype=np.float64)
    before = storage.copy()
    result = scpn.value_and_grad_program_ad_effect_ir_with_rust(_rational_ir(), storage[::2])
    assert result.supported and result.value == -7.75
    np.testing.assert_array_equal(result.gradient, [1.0, 2.0])
    np.testing.assert_array_equal(storage, before)
    refused = scpn.value_and_grad_program_ad_effect_ir_with_rust(_rational_ir(), [1e308, 5.0])
    assert not refused.supported and refused.value is None
    recovered = scpn.value_and_grad_program_ad_effect_ir_with_rust(_rational_ir(), [3.0, 5.0])
    assert recovered.supported and recovered.value == 19.0


def test_native_derivative_dimension_errors_and_recovery() -> None:
    """Native derivative and batch ABIs reject incorrect shapes before writing a result."""
    sample = np.array([3.0, 5.0], dtype=np.float64)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(_polynomial, sample, PARAMETERS)
    before = sample.copy()
    with pytest.raises(ValueError):
        kernel.jvp(sample, [1.0])
    with pytest.raises(ValueError):
        kernel.vjp(sample, [1.0, 2.0])
    with pytest.raises(ValueError):
        kernel.batch_value_and_grad([[3.0], [5.0]])
    with pytest.raises(ValueError):
        kernel.batch_jvp([[3.0, 5.0]], [[1.0]])
    with pytest.raises(ValueError):
        kernel.batch_vjp([[3.0, 5.0]], [1.0, 2.0])
    np.testing.assert_array_equal(sample, before)
    assert kernel.jvp(sample, [1.0, 0.0]) == 6.0
    np.testing.assert_array_equal(kernel.vjp(sample, [1.0]), [6.0, 2.0])


def test_rust_non_view_matrix_aliases_remain_explicitly_unsupported() -> None:
    """Alias-bearing method-form IR cannot silently borrow the admitted NumPy route's claim."""

    def objective(values: FloatArray) -> object:
        """Capture matrix methods whose local aliases are outside the Rust replay contract."""
        left = values[:4].reshape(2, 2)
        right = values[4:].reshape(2, 2)
        return np.linalg.multi_dot((left, right)).sum()

    sample = np.arange(1.0, 9.0, dtype=np.float64)
    before = sample.copy()
    captured = whole_program_value_and_grad(objective, sample)
    assert captured.program_ir is not None
    result = scpn.value_and_grad_program_ad_effect_ir_with_rust(captured.program_ir, sample)
    assert not result.supported and result.value is None
    assert any("non-view alias-bearing" in reason for reason in result.blocked_reasons)
    np.testing.assert_array_equal(sample, before)
