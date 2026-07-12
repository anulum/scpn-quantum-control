# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR native primitives module
# scpn-quantum-control -- shared native lowering primitives for the MLIR surface
"""Low-level primitives shared by the matrix-JIT and whole-program native lowering paths.

These helpers carry no AD policy: they format LLVM/MLIR textual tokens, coerce and
validate float operand arrays, and drive the llvmlite MCJIT engine that turns emitted
LLVM IR into callable native kernels. Both the per-primitive matrix/vector compilers and
the whole-program native lowering depend on them, so they live in a dependency-free leaf
module to keep those two larger surfaces from importing each other.
"""

from __future__ import annotations

import ctypes
import importlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _copy_float_array(values: FloatArray) -> FloatArray:
    copied: object = values.copy()
    return cast(FloatArray, copied)


def _as_finite_vector(name: str, value: object) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return _copy_float_array(array)


def _max_abs_error(left: FloatArray, right: FloatArray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _fmt_float(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("MLIR numeric attributes must be finite")
    return format(value, ".17g")


def _fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def _escape_mlir_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _safe_llvm_symbol(value: str) -> str:
    symbol = "".join(
        character if character.isalnum() or character == "_" else "_" for character in value
    )
    if not symbol or symbol[0].isdigit():
        symbol = f"_{symbol}"
    return symbol


def _load_llvmlite_binding() -> Any:
    try:
        llvm = importlib.import_module("llvmlite.binding")
    except ModuleNotFoundError as exc:
        raise ValueError(
            "native_llvm_jit backend requires llvmlite.binding to be installed"
        ) from exc

    for initializer in (
        llvm.initialize_native_target,
        llvm.initialize_native_asmprinter,
    ):
        try:
            initializer()
        except RuntimeError as exc:
            if "already" not in str(exc).lower():
                raise
    return llvm


def _compile_native_llvm_jit_functions(
    llvm_ir: str,
    base_symbol: str,
) -> Mapping[str, Any]:
    llvm = _load_llvmlite_binding()
    module = llvm.parse_assembly(llvm_ir)
    module.verify()
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_module = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_module, target_machine)
    engine.add_module(module)
    engine.finalize_object()
    engine.run_static_constructors()

    double_pointer = ctypes.POINTER(ctypes.c_double)
    unary_function = ctypes.CFUNCTYPE(None, double_pointer, double_pointer)
    binary_function = ctypes.CFUNCTYPE(None, double_pointer, double_pointer, double_pointer)
    batch_value_gradient_function = ctypes.CFUNCTYPE(
        None,
        double_pointer,
        ctypes.c_int64,
        double_pointer,
        double_pointer,
    )
    batch_binary_function = ctypes.CFUNCTYPE(
        None,
        double_pointer,
        double_pointer,
        ctypes.c_int64,
        double_pointer,
    )
    functions: dict[str, Any] = {"engine": engine}
    for name, signature in (
        ("value", unary_function),
        ("gradient", unary_function),
        ("jvp", binary_function),
        ("vjp", binary_function),
    ):
        address = engine.get_function_address(f"{base_symbol}_{name}")
        if address == 0:
            raise ValueError(f"native_llvm_jit symbol {base_symbol}_{name} was not emitted")
        functions[name] = signature(address)
    batch_address = engine.get_function_address(f"{base_symbol}_batch_value_gradient")
    if batch_address != 0:
        functions["batch_value_gradient"] = batch_value_gradient_function(batch_address)
    for name in ("batch_jvp", "batch_vjp"):
        address = engine.get_function_address(f"{base_symbol}_{name}")
        if address != 0:
            functions[name] = batch_binary_function(address)
    return MappingProxyType(functions)
