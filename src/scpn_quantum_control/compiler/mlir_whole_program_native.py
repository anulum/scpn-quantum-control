# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- whole-program AD native lowering for the MLIR surface
"""Native and MLIR lowering of whole-program autodiff traces.

This module turns a recorded ``WholeProgramADResult`` into either deterministic
MLIR text or a callable native LLVM/JIT kernel that evaluates the program and its
value-and-gradient, JVP and VJP, including batched variants. It carries the whole
emission machinery: the lowerability analysis and op-set tables, the structural
and linalg operation emitters (determinant, inverse, solve, trace, diagonal,
where-predicate), the determinant Faddeev-LeVerrier and fixed-size helpers, the
native compile cache, and the executable/native kernel wrappers that verify each
emitted kernel against the interpreted reference before handing it back.

It depends only on the shared native lowering primitives, the MLIR record types
and the differentiable trace contracts, so it stays a leaf of the compiler
package and never imports the ``mlir`` facade back.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ..differentiable import (
    Parameter,
    WholeProgramADResult,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)
from .mlir_native_primitives import (
    _as_finite_vector,
    _compile_native_llvm_jit_functions,
    _copy_float_array,
    _escape_mlir_string,
    _fmt_bool,
    _fmt_float,
    _max_abs_error,
    _safe_llvm_symbol,
)
from .mlir_records import (
    CompilerADKernelVerification,
    DifferentiableMLIRCompileConfig,
    MLIRModule,
)

FloatArray: TypeAlias = NDArray[np.float64]


def compile_whole_program_ad_trace_to_mlir(
    result: WholeProgramADResult,
    config: DifferentiableMLIRCompileConfig | None = None,
) -> MLIRModule:
    """Lower a whole-program AD execution trace to MLIR-style interchange text.

    The emitted module is an audit artefact for Python whole-program gradients
    and polyglot compiler planning. It deliberately records Rust and LLVM/JIT
    executable differentiation as blocked unless a real backend is provided.
    """

    if not isinstance(result, WholeProgramADResult):
        raise ValueError("whole-program MLIR lowering requires a WholeProgramADResult")
    compile_config = DifferentiableMLIRCompileConfig() if config is None else config
    program_ir = result.program_ir
    program_ir_format = "program_ad_effect_ir.v1" if program_ir is not None else "none"
    lines = [
        f'module attributes {{scpn.module = "whole_program_ad", '
        f'scpn.dialect = "{compile_config.dialect}", '
        f'scpn.program_ir_format = "{program_ir_format}", '
        f"scpn.n_parameters = {result.gradient.size}, "
        f"scpn.trace_events = {len(result.trace_events)}, "
        f"scpn.control_flow = {_fmt_bool(result.control_flow_observed)}, "
        f"scpn.numpy = {_fmt_bool(result.numpy_observed)}}} {{",
        "  func.func @main() {",
    ]

    def optional_int_attribute(value: int | None) -> str:
        return "none" if value is None else str(value)

    def optional_str_attribute(value: str | None) -> str:
        return "none" if value is None else _escape_mlir_string(value)

    def joined_ints(values: Sequence[int]) -> str:
        return ",".join(str(value) for value in values)

    def joined_strings(values: Sequence[str]) -> str:
        return ",".join(_escape_mlir_string(value) for value in values)

    if compile_config.include_numeric_payload:
        lines.append(f"    scpn_diff.value %objective {{value = {_fmt_float(result.value)}}}")
        for index, (name, trainable, gradient) in enumerate(
            zip(result.parameter_names, result.trainable, result.gradient, strict=True)
        ):
            lines.append(
                "    scpn_diff.parameter "
                f'%p{index} {{name = "{_escape_mlir_string(name)}", '
                f"trainable = {_fmt_bool(trainable)}, "
                f"gradient = {_fmt_float(float(gradient))}}}"
            )
    for index, event in enumerate(result.trace_events):
        lines.append(
            "    scpn_diff.trace_event "
            f'{{index = {index}, file = "{_escape_mlir_string(event.filename)}", '
            f'line = {event.line_number}, function = "{_escape_mlir_string(event.function_name)}", '
            f'source = "{_escape_mlir_string(event.source)}"}}'
        )
    for index, instruction in enumerate(result.bytecode_instructions):
        lines.append(
            "    scpn_diff.bytecode "
            f"{{index = {index}, offset = {instruction.offset}, "
            f'op = "{_escape_mlir_string(instruction.opname)}", '
            f'arg = "{_escape_mlir_string(instruction.argrepr)}"}}'
        )
    for index, feature in enumerate(result.source_ir_features):
        lines.append(
            "    scpn_diff.source_semantics "
            f'{{index = {index}, kind = "{_escape_mlir_string(feature.kind)}", '
            f'detail = "{_escape_mlir_string(feature.detail)}", line = {feature.line_number}}}'
        )
    if program_ir is not None:
        for index, value in enumerate(program_ir.ssa_values):
            lines.append(
                "    scpn_diff.program_ad_ssa "
                f'%ssa{index} {{name = "{_escape_mlir_string(value.name)}", '
                f"producer = {optional_int_attribute(value.producer)}, "
                f"version = {value.version}, "
                f'shape = "{joined_ints(value.shape)}", '
                f'dtype = "{_escape_mlir_string(value.dtype)}", '
                f"effect = {optional_int_attribute(value.effect)}}}"
            )
        for effect in program_ir.effects:
            lines.append(
                "    scpn_diff.program_ad_effect "
                f'{{index = {effect.index}, kind = "{_escape_mlir_string(effect.kind)}", '
                f'target = "{_escape_mlir_string(effect.target)}", '
                f'inputs = "{joined_strings(effect.inputs)}", '
                f"version = {effect.version}, ordering = {effect.ordering}}}"
            )
        for edge in program_ir.alias_edges:
            lines.append(
                "    scpn_diff.program_ad_alias_edge "
                f'{{source = "{_escape_mlir_string(edge.source)}", '
                f'target = "{_escape_mlir_string(edge.target)}", '
                f'kind = "{_escape_mlir_string(edge.kind)}", version = {edge.version}}}'
            )
        for region in program_ir.control_regions:
            lines.append(
                "    scpn_diff.program_ad_control_region "
                f'{{index = {region.index}, kind = "{_escape_mlir_string(region.kind)}", '
                f'predicate = "{optional_str_attribute(region.predicate)}", '
                f"entered = {_fmt_bool(region.entered)}, "
                f"source_line = {optional_int_attribute(region.source_line)}}}"
            )
        for phi in program_ir.phi_nodes:
            lines.append(
                "    scpn_diff.program_ad_phi "
                f'{{index = {phi.index}, target = "{_escape_mlir_string(phi.target)}", '
                f'incoming = "{joined_strings(phi.incoming)}", '
                f"control_region = {optional_int_attribute(phi.control_region)}, "
                f'selected = "{optional_str_attribute(phi.selected)}", '
                f"source_line = {optional_int_attribute(phi.source_line)}}}"
            )
    lines.append(
        "    scpn_diff.whole_program_ad "
        f'{{method = "{_escape_mlir_string(result.method)}", '
        'execution = "python_whole_program_ad_interchange"}}'
    )
    lines.append("    return")
    lines.append("  }")
    if compile_config.include_metadata:
        metadata = {
            "claim_boundary": result.claim_boundary,
            "method": result.method,
            "polyglot_targets": result.polyglot_targets,
            "semantics_report": None
            if result.semantics_report is None
            else {
                "aliasing_observed": result.semantics_report.aliasing_observed,
                "bytecode_frontend": result.semantics_report.bytecode_frontend,
                "control_flow_observed": result.semantics_report.control_flow_observed,
                "differentiation_semantics": result.semantics_report.differentiation_semantics,
                "graph_capture": result.semantics_report.graph_capture,
                "loop_observed": result.semantics_report.loop_observed,
                "mutation_observed": result.semantics_report.mutation_observed,
                "numpy_observed": result.semantics_report.numpy_observed,
                "source_frontend": result.semantics_report.source_frontend,
            },
            "program_ad_ir": None
            if program_ir is None
            else {
                "alias_edges": len(program_ir.alias_edges),
                "claim_boundary": ("program_ad_ir_mlir_interchange_only_no_executable_lowering"),
                "control_regions": len(program_ir.control_regions),
                "effects": len(program_ir.effects),
                "format": "program_ad_effect_ir.v1",
                "phi_nodes": len(program_ir.phi_nodes),
                "serialization_sha256": hashlib.sha256(
                    program_ir.serialization.encode("utf-8")
                ).hexdigest(),
                "ssa_values": len(program_ir.ssa_values),
            },
            "target": compile_config.target,
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        lines.append(f'  scpn.metadata {{json = "{_escape_mlir_string(encoded)}"}}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=compile_config.dialect,
        resource_counts={
            "parameters": int(result.gradient.size),
            "bytecode_instructions": len(result.bytecode_instructions),
            "source_ir_features": len(result.source_ir_features),
            "ir_nodes": len(result.ir_nodes),
            "program_ad_ssa_values": 0 if program_ir is None else len(program_ir.ssa_values),
            "program_ad_effects": 0 if program_ir is None else len(program_ir.effects),
            "program_ad_alias_edges": 0 if program_ir is None else len(program_ir.alias_edges),
            "program_ad_control_regions": 0
            if program_ir is None
            else len(program_ir.control_regions),
            "program_ad_phi_nodes": 0 if program_ir is None else len(program_ir.phi_nodes),
            "trace_events": len(result.trace_events),
            "trainable_parameters": int(sum(result.trainable)),
            "gradient_nnz": int(np.count_nonzero(result.gradient)),
        },
        metadata={
            "claim_boundary": "whole-program AD trace interchange; no executable Rust, LLVM, or JIT lowering",
            "target": compile_config.target,
            "polyglot_targets": dict(result.polyglot_targets),
            "semantics_report": None
            if result.semantics_report is None
            else {
                "aliasing_observed": result.semantics_report.aliasing_observed,
                "bytecode_frontend": result.semantics_report.bytecode_frontend,
                "control_flow_observed": result.semantics_report.control_flow_observed,
                "differentiation_semantics": result.semantics_report.differentiation_semantics,
                "graph_capture": result.semantics_report.graph_capture,
                "loop_observed": result.semantics_report.loop_observed,
                "mutation_observed": result.semantics_report.mutation_observed,
                "numpy_observed": result.semantics_report.numpy_observed,
                "source_frontend": result.semantics_report.source_frontend,
            },
            "program_ad_ir": None
            if program_ir is None
            else {
                "alias_edges": len(program_ir.alias_edges),
                "claim_boundary": ("program_ad_ir_mlir_interchange_only_no_executable_lowering"),
                "control_regions": len(program_ir.control_regions),
                "effects": len(program_ir.effects),
                "format": "program_ad_effect_ir.v1",
                "phi_nodes": len(program_ir.phi_nodes),
                "serialization_sha256": hashlib.sha256(
                    program_ir.serialization.encode("utf-8")
                ).hexdigest(),
                "ssa_values": len(program_ir.ssa_values),
            },
            "sha256_source": "module.text",
        },
    )


@dataclass(frozen=True)
class ExecutableWholeProgramADBatchResult:
    """Batched replay result from an executable whole-program AD kernel."""

    values: NDArray[np.float64]
    gradients: NDArray[np.float64]
    parameter_names: tuple[str, ...]
    row_signatures: tuple[tuple[str, ...], ...]
    mlir_sha256: str
    backend: str = "program_ad_trace_replay"
    claim_boundary: str = (
        "batched executable replay of supported captured scalar program AD IR; "
        "each row must preserve the compiled branch/signature contract"
    )

    def __post_init__(self) -> None:
        values = _as_finite_vector("batch values", self.values)
        gradients = np.asarray(self.gradients, dtype=np.float64)
        if gradients.ndim != 2:
            raise ValueError("batch gradients must be two-dimensional")
        if gradients.shape[0] != values.size:
            raise ValueError("batch gradients row count must match batch values")
        if gradients.shape[1] != len(self.parameter_names):
            raise ValueError("batch gradients column count must match parameter_names")
        if not np.all(np.isfinite(gradients)):
            raise ValueError("batch gradients must contain only finite values")
        if len(self.row_signatures) != values.size:
            raise ValueError("row_signatures count must match batch values")
        for signature in self.row_signatures:
            if any(not isinstance(item, str) or not item for item in signature):
                raise ValueError("row_signatures entries must be non-empty strings")
        if not self.mlir_sha256:
            raise ValueError("mlir_sha256 must be non-empty")
        if self.backend not in {"program_ad_trace_replay", "native_llvm_jit"}:
            raise ValueError("backend must be 'program_ad_trace_replay' or 'native_llvm_jit'")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "gradients", gradients.copy())


@dataclass(frozen=True)
class _NativeWholeProgramADCacheEntry:
    mlir_module: MLIRModule
    llvm_ir: str
    native_functions: Mapping[str, Any]
    verification: CompilerADKernelVerification
    supported_ops: tuple[str, ...]
    lowering_report: WholeProgramADNativeLoweringReport


@dataclass(frozen=True)
class WholeProgramADNativeLoweringReport:
    """Fail-closed native lowering audit for one captured program AD trace."""

    supported: bool
    lowerable_ops: tuple[str, ...]
    unsupported_ops: tuple[str, ...]
    control_flow_ops: tuple[str, ...]
    effect_kinds: tuple[str, ...]
    operation_count: int
    lowerable_operation_count: int
    unsupported_operation_count: int
    fail_closed_reason: str

    def __post_init__(self) -> None:
        for field_name, values in (
            ("lowerable_ops", self.lowerable_ops),
            ("unsupported_ops", self.unsupported_ops),
            ("control_flow_ops", self.control_flow_ops),
            ("effect_kinds", self.effect_kinds),
        ):
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{field_name} entries must be non-empty strings")
        if self.operation_count < 1:
            raise ValueError("operation_count must be positive")
        if self.lowerable_operation_count < 0:
            raise ValueError("lowerable_operation_count must be non-negative")
        if self.unsupported_operation_count < 0:
            raise ValueError("unsupported_operation_count must be non-negative")
        if self.operation_count != (
            self.lowerable_operation_count + self.unsupported_operation_count
        ):
            raise ValueError("operation counts must partition the native lowering report")
        if self.supported != (self.unsupported_operation_count == 0):
            raise ValueError("supported must match unsupported_operation_count")
        if not self.fail_closed_reason:
            raise ValueError("fail_closed_reason must be non-empty")

    def as_metadata(self) -> Mapping[str, object]:
        """Return deterministic MLIR-serialisable native lowering metadata."""

        return MappingProxyType(
            {
                "supported": self.supported,
                "lowerable_ops": self.lowerable_ops,
                "unsupported_ops": self.unsupported_ops,
                "control_flow_ops": self.control_flow_ops,
                "effect_kinds": self.effect_kinds,
                "operation_count": self.operation_count,
                "lowerable_operation_count": self.lowerable_operation_count,
                "unsupported_operation_count": self.unsupported_operation_count,
                "fail_closed_reason": self.fail_closed_reason,
            }
        )


_NATIVE_WHOLE_PROGRAM_AD_CACHE_LOCK = threading.RLock()


_NATIVE_WHOLE_PROGRAM_AD_CACHE: dict[str, _NativeWholeProgramADCacheEntry] = {}


_NATIVE_WHOLE_PROGRAM_AD_CACHE_MAXSIZE = 32


@dataclass(frozen=True)
class NativeWholeProgramADKernel:
    """Native LLVM/JIT kernel for a supported scalar program AD trace."""

    objective: Callable[[Any], object]
    source_result: WholeProgramADResult
    parameters: tuple[Parameter, ...]
    mlir_module: MLIRModule
    llvm_ir: str
    native_functions: Mapping[str, Any]
    verification: CompilerADKernelVerification
    parameter_names: tuple[str, ...]
    parameter_shape: tuple[int, ...]
    trace_signature: tuple[str, ...]
    supported_ops: tuple[str, ...]
    lowering_report: WholeProgramADNativeLoweringReport
    cache_key: str
    cache_hit: bool
    backend: str = "native_llvm_jit"
    claim_boundary: str = (
        "native LLVM/JIT execution for supported scalar program AD traces with "
        "stable executed branch signatures and finite supported primitive domains; "
        "compiled batch value/gradient, JVP, and VJP execution for matching row "
        "signatures; "
        "unsupported control flow, mutation-dependent path changes, and unsupported "
        "operations fail closed"
    )

    def __post_init__(self) -> None:
        if not callable(self.objective):
            raise ValueError("objective must be callable")
        if not isinstance(self.source_result, WholeProgramADResult):
            raise ValueError("source_result must be a WholeProgramADResult")
        if not isinstance(self.mlir_module, MLIRModule):
            raise ValueError("mlir_module must be an MLIRModule")
        if not self.llvm_ir.strip():
            raise ValueError("llvm_ir must be non-empty")
        for name in (
            "value",
            "gradient",
            "jvp",
            "vjp",
            "batch_value_gradient",
            "batch_jvp",
            "batch_vjp",
            "engine",
        ):
            if name not in self.native_functions:
                raise ValueError(f"native_functions missing {name}")
        for name in (
            "value",
            "gradient",
            "jvp",
            "vjp",
            "batch_value_gradient",
            "batch_jvp",
            "batch_vjp",
        ):
            if not callable(self.native_functions[name]):
                raise ValueError(f"native function {name} must be callable")
        if not isinstance(self.verification, CompilerADKernelVerification):
            raise ValueError("verification must be CompilerADKernelVerification")
        if not self.verification.passed:
            raise ValueError("native whole-program AD kernel verification failed")
        if not self.parameters or any(
            not isinstance(parameter, Parameter) for parameter in self.parameters
        ):
            raise ValueError("parameters must be a non-empty tuple of Parameter objects")
        if self.parameter_names != tuple(parameter.name for parameter in self.parameters):
            raise ValueError("parameter_names must match parameters")
        if self.parameter_names != self.source_result.parameter_names:
            raise ValueError("parameter_names must match source_result")
        if self.parameter_shape != (len(self.parameters),):
            raise ValueError("parameter_shape must match parameter count")
        if self.source_result.gradient.shape != self.parameter_shape:
            raise ValueError("source_result gradient shape must match parameter_shape")
        if self.trace_signature != _whole_program_replay_signature(self.source_result):
            raise ValueError("trace_signature must match source_result")
        if any(not isinstance(item, str) or not item for item in self.supported_ops):
            raise ValueError("supported_ops entries must be non-empty strings")
        if not isinstance(self.lowering_report, WholeProgramADNativeLoweringReport):
            raise ValueError("lowering_report must be a WholeProgramADNativeLoweringReport")
        if not self.lowering_report.supported:
            raise ValueError("lowering_report must describe a supported native trace")
        if self.supported_ops != self.lowering_report.lowerable_ops:
            raise ValueError("supported_ops must match lowering_report lowerable_ops")
        if len(self.cache_key) != 64:
            raise ValueError("cache_key must be a sha256 hex digest")
        if not isinstance(self.cache_hit, bool):
            raise ValueError("cache_hit must be a bool")
        if self.backend != "native_llvm_jit":
            raise ValueError("backend must be 'native_llvm_jit'")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    def _checked_values(self, values: Sequence[float] | FloatArray) -> NDArray[np.float64]:
        checked = _as_finite_vector("values", values)
        if checked.shape != self.parameter_shape:
            raise ValueError(
                "values shape must match native whole-program AD parameter shape "
                f"{self.parameter_shape}"
            )
        return np.ascontiguousarray(checked, dtype=np.float64)

    def _checked_batch_values(
        self,
        values: Sequence[Sequence[float]] | FloatArray,
    ) -> NDArray[np.float64]:
        checked = np.asarray(values, dtype=np.float64)
        if checked.ndim != 2:
            raise ValueError("batch values must be two-dimensional")
        if checked.shape[0] < 1:
            raise ValueError("batch values must contain at least one row")
        if checked.shape[1:] != self.parameter_shape:
            raise ValueError(
                "batch values shape must be (batch, parameters) with parameter shape "
                f"{self.parameter_shape}"
            )
        if not np.all(np.isfinite(checked)):
            raise ValueError("batch values must contain only finite values")
        return np.ascontiguousarray(checked, dtype=np.float64)

    def _checked_batch_tangents(
        self,
        tangents: Sequence[Sequence[float]] | FloatArray,
        row_count: int,
    ) -> NDArray[np.float64]:
        checked = np.asarray(tangents, dtype=np.float64)
        if checked.ndim != 2:
            raise ValueError("batch tangents must be two-dimensional")
        if checked.shape != (row_count, self.parameter_shape[0]):
            raise ValueError("batch tangents shape must match batch values shape")
        if not np.all(np.isfinite(checked)):
            raise ValueError("batch tangents must contain only finite values")
        return np.ascontiguousarray(checked, dtype=np.float64)

    @staticmethod
    def _checked_batch_cotangents(
        cotangents: Sequence[float] | Sequence[Sequence[float]] | FloatArray,
        row_count: int,
    ) -> NDArray[np.float64]:
        checked = np.asarray(cotangents, dtype=np.float64)
        if checked.ndim == 2 and checked.shape[1:] == (1,):
            checked = checked.reshape(-1)
        if checked.ndim != 1:
            raise ValueError("batch cotangents must be one-dimensional")
        if checked.shape != (row_count,):
            raise ValueError("batch cotangents row count must match batch values")
        if not np.all(np.isfinite(checked)):
            raise ValueError("batch cotangents must contain only finite values")
        return np.ascontiguousarray(checked, dtype=np.float64)

    def _validate_trace_signature(self, values: NDArray[np.float64]) -> None:
        if not _whole_program_native_requires_runtime_recapture(self.source_result):
            return
        result = whole_program_value_and_grad(
            self.objective,
            values,
            parameters=self.parameters,
            trace=False,
        )
        signature = _whole_program_native_replay_signature(result)
        if signature != _whole_program_native_replay_signature(self.source_result):
            raise ValueError(
                "native whole-program AD kernel branch signature changed; "
                "recompile with representative sample values"
            )

    def value(self, values: Sequence[float] | FloatArray) -> float:
        """Execute the native scalar value kernel."""

        checked = self._checked_values(values)
        self._validate_trace_signature(checked)
        output = _call_native_whole_program_unary(
            self.native_functions["value"],
            checked,
            1,
        )
        return float(output[0])

    def gradient(self, values: Sequence[float] | FloatArray) -> NDArray[np.float64]:
        """Execute the native scalar-output gradient kernel."""

        checked = self._checked_values(values)
        self._validate_trace_signature(checked)
        return _call_native_whole_program_unary(
            self.native_functions["gradient"],
            checked,
            self.parameter_shape[0],
        )

    def value_and_grad(
        self,
        values: Sequence[float] | FloatArray,
    ) -> tuple[float, NDArray[np.float64]]:
        """Execute native value and gradient kernels."""

        checked = self._checked_values(values)
        self._validate_trace_signature(checked)
        value = _call_native_whole_program_unary(
            self.native_functions["value"],
            checked,
            1,
        )
        gradient = _call_native_whole_program_unary(
            self.native_functions["gradient"],
            checked,
            self.parameter_shape[0],
        )
        return float(value[0]), gradient

    def jvp(
        self,
        values: Sequence[float] | FloatArray,
        tangent: Sequence[float] | FloatArray,
    ) -> float:
        """Execute the native scalar JVP kernel."""

        checked_values = self._checked_values(values)
        checked_tangent = _as_finite_vector("tangent", tangent)
        if checked_tangent.shape != self.parameter_shape:
            raise ValueError("tangent shape must match parameter_shape")
        self._validate_trace_signature(checked_values)
        output = _call_native_whole_program_binary(
            self.native_functions["jvp"],
            checked_values,
            checked_tangent,
            1,
        )
        return float(output[0])

    def vjp(
        self,
        values: Sequence[float] | FloatArray,
        cotangent: Sequence[float] | FloatArray,
    ) -> NDArray[np.float64]:
        """Execute the native scalar VJP kernel."""

        checked_cotangent = _as_finite_vector("cotangent", cotangent)
        if checked_cotangent.shape != (1,):
            raise ValueError("cotangent must contain exactly one scalar")
        checked_values = self._checked_values(values)
        self._validate_trace_signature(checked_values)
        return _call_native_whole_program_binary(
            self.native_functions["vjp"],
            checked_values,
            checked_cotangent,
            self.parameter_shape[0],
        )

    def batch_value_and_grad(
        self,
        values: Sequence[Sequence[float]] | FloatArray,
    ) -> ExecutableWholeProgramADBatchResult:
        """Execute native value and gradient kernels over a two-dimensional batch."""

        batch = self._checked_batch_values(values)
        for row in batch:
            self._validate_trace_signature(row)
        row_values, row_gradients = _call_native_whole_program_batch_value_gradient(
            self.native_functions["batch_value_gradient"],
            batch,
            self.parameter_shape[0],
        )
        return ExecutableWholeProgramADBatchResult(
            values=row_values,
            gradients=row_gradients,
            parameter_names=self.parameter_names,
            row_signatures=(self.trace_signature,) * batch.shape[0],
            mlir_sha256=self.mlir_module.sha256,
            backend=self.backend,
            claim_boundary=(
                "compiled batched native LLVM/JIT value/gradient execution for supported "
                "scalar program AD traces preserving compiled branch signatures and finite "
                "primitive domains"
            ),
        )

    def batch_value(self, values: Sequence[Sequence[float]] | FloatArray) -> NDArray[np.float64]:
        """Execute native value kernels over a two-dimensional batch."""

        return self.batch_value_and_grad(values).values

    def batch_gradient(
        self,
        values: Sequence[Sequence[float]] | FloatArray,
    ) -> NDArray[np.float64]:
        """Execute native gradient kernels over a two-dimensional batch."""

        return self.batch_value_and_grad(values).gradients

    def batch_jvp(
        self,
        values: Sequence[Sequence[float]] | FloatArray,
        tangents: Sequence[Sequence[float]] | FloatArray,
    ) -> NDArray[np.float64]:
        """Execute the compiled native JVP kernel over a two-dimensional batch."""

        batch = self._checked_batch_values(values)
        checked_tangents = self._checked_batch_tangents(tangents, batch.shape[0])
        for row in batch:
            self._validate_trace_signature(row)
        return _call_native_whole_program_batch_jvp(
            self.native_functions["batch_jvp"],
            batch,
            checked_tangents,
            self.parameter_shape[0],
        )

    def batch_vjp(
        self,
        values: Sequence[Sequence[float]] | FloatArray,
        cotangents: Sequence[float] | Sequence[Sequence[float]] | FloatArray,
    ) -> NDArray[np.float64]:
        """Execute the compiled native VJP kernel over a two-dimensional batch."""

        batch = self._checked_batch_values(values)
        checked_cotangents = self._checked_batch_cotangents(cotangents, batch.shape[0])
        for row in batch:
            self._validate_trace_signature(row)
        return _call_native_whole_program_batch_vjp(
            self.native_functions["batch_vjp"],
            batch,
            checked_cotangents,
            self.parameter_shape[0],
        )


@dataclass(frozen=True)
class ExecutableWholeProgramADKernel:
    """Executable replay kernel for a supported captured program AD trace.

    The kernel is intentionally bounded: it replays the original Python
    objective through the supported operator-intercepted program AD IR, checks
    the one-dimensional parameter shape, checks the captured control/signature
    surface, and computes gradients through reverse-mode adjoint replay. It is
    executable and deterministic for the supported captured trace contract; it
    does not claim arbitrary source compilation or native LLVM/JIT lowering for
    arbitrary Python programs.
    """

    objective: Callable[[Any], object]
    source_result: WholeProgramADResult
    parameters: tuple[Parameter, ...]
    mlir_module: MLIRModule
    parameter_names: tuple[str, ...]
    parameter_shape: tuple[int, ...]
    branch_signature: tuple[str, ...]
    backend: str = "program_ad_trace_replay"
    claim_boundary: str = (
        "executable replay of supported captured scalar program AD IR with "
        "deterministic MLIR provenance; branch/signature changes fail closed; "
        "no arbitrary source compiler or native LLVM/JIT claim"
    )

    def __post_init__(self) -> None:
        if not callable(self.objective):
            raise ValueError("objective must be callable")
        if not isinstance(self.source_result, WholeProgramADResult):
            raise ValueError("source_result must be a WholeProgramADResult")
        if not isinstance(self.mlir_module, MLIRModule):
            raise ValueError("mlir_module must be an MLIRModule")
        if not self.parameters or any(
            not isinstance(parameter, Parameter) for parameter in self.parameters
        ):
            raise ValueError("parameters must be a non-empty tuple of Parameter objects")
        if self.parameter_names != tuple(parameter.name for parameter in self.parameters):
            raise ValueError("parameter_names must match parameters")
        if self.parameter_names != self.source_result.parameter_names:
            raise ValueError("parameter_names must match source_result")
        if self.parameter_shape != (len(self.parameters),):
            raise ValueError("parameter_shape must match parameter count")
        if self.source_result.gradient.shape != self.parameter_shape:
            raise ValueError("source_result gradient shape must match parameter_shape")
        if any(not isinstance(item, str) or not item for item in self.branch_signature):
            raise ValueError("branch_signature entries must be non-empty strings")
        if self.branch_signature != _whole_program_replay_signature(self.source_result):
            raise ValueError("branch_signature must match source_result")
        if self.backend != "program_ad_trace_replay":
            raise ValueError("backend must be 'program_ad_trace_replay'")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    def _checked_values(self, values: Sequence[float] | FloatArray) -> NDArray[np.float64]:
        checked = _as_finite_vector("values", values)
        if checked.shape != self.parameter_shape:
            raise ValueError(
                "values shape must match executable whole-program AD parameter shape "
                f"{self.parameter_shape}"
            )
        return checked

    def _checked_batch_values(
        self,
        values: Sequence[Sequence[float]] | FloatArray,
    ) -> NDArray[np.float64]:
        checked = np.asarray(values, dtype=np.float64)
        if checked.ndim != 2:
            raise ValueError("batch values must be two-dimensional")
        if checked.shape[0] < 1:
            raise ValueError("batch values must contain at least one row")
        if checked.shape[1:] != self.parameter_shape:
            raise ValueError(
                "batch values shape must be (batch, parameters) with parameter shape "
                f"{self.parameter_shape}"
            )
        if not np.all(np.isfinite(checked)):
            raise ValueError("batch values must contain only finite values")
        return _copy_float_array(checked)

    def _recapture(self, values: Sequence[float] | FloatArray) -> WholeProgramADResult:
        checked = self._checked_values(values)
        result = whole_program_value_and_grad(
            self.objective,
            checked,
            parameters=self.parameters,
            trace=False,
        )
        signature = _whole_program_replay_signature(result)
        if signature != self.branch_signature:
            raise ValueError(
                "whole-program executable AD kernel branch signature changed; "
                "recompile with representative sample values"
            )
        return result

    def value_and_grad(
        self,
        values: Sequence[float] | FloatArray,
    ) -> tuple[float, NDArray[np.float64]]:
        """Execute value replay and reverse-mode adjoint gradient replay."""

        result = self._recapture(values)
        return result.value, program_adjoint_gradient(result)

    def value(self, values: Sequence[float] | FloatArray) -> float:
        """Execute value replay for the captured program AD trace."""

        return self.value_and_grad(values)[0]

    def gradient(self, values: Sequence[float] | FloatArray) -> NDArray[np.float64]:
        """Execute reverse-mode adjoint replay for the captured program AD trace."""

        return self.value_and_grad(values)[1]

    def batch_value_and_grad(
        self,
        values: Sequence[Sequence[float]] | FloatArray,
    ) -> ExecutableWholeProgramADBatchResult:
        """Execute same-branch batched value and reverse-adjoint gradient replay."""

        batch = self._checked_batch_values(values)
        row_values: list[float] = []
        row_gradients: list[NDArray[np.float64]] = []
        row_signatures: list[tuple[str, ...]] = []
        for row_index, row in enumerate(batch):
            result = self._recapture(row)
            signature = _whole_program_replay_signature(result)
            if signature != self.branch_signature:
                raise ValueError(
                    f"whole-program executable AD batch row {row_index} branch signature changed"
                )
            row_values.append(result.value)
            row_gradients.append(program_adjoint_gradient(result))
            row_signatures.append(signature)
        return ExecutableWholeProgramADBatchResult(
            values=np.asarray(row_values, dtype=np.float64),
            gradients=np.vstack(row_gradients).astype(np.float64, copy=False),
            parameter_names=self.parameter_names,
            row_signatures=tuple(row_signatures),
            mlir_sha256=self.mlir_module.sha256,
        )

    def batch_value(self, values: Sequence[Sequence[float]] | FloatArray) -> NDArray[np.float64]:
        """Execute batched value replay for rows preserving the compiled branch path."""

        return self.batch_value_and_grad(values).values

    def batch_gradient(
        self,
        values: Sequence[Sequence[float]] | FloatArray,
    ) -> NDArray[np.float64]:
        """Execute batched reverse-adjoint replay for rows preserving the branch path."""

        return self.batch_value_and_grad(values).gradients


def compile_whole_program_ad_trace_to_executable(
    objective: Callable[[Any], object],
    sample_values: Sequence[float] | FloatArray,
    parameters: Sequence[Parameter] | None = None,
    config: DifferentiableMLIRCompileConfig | None = None,
    *,
    trace: bool = True,
) -> ExecutableWholeProgramADKernel:
    """Compile a supported captured program AD trace to an executable replay kernel.

    This is the executable compiler boundary for whole-program AD today: it
    captures the supported scalar program IR, verifies reverse adjoint replay is
    available, emits deterministic MLIR provenance, then returns a fail-closed
    replay kernel. Shape drift, non-finite inputs, and branch/signature drift
    raise errors instead of silently changing the differentiated program.
    """

    if not callable(objective):
        raise ValueError("whole-program executable AD objective must be callable")
    checked_sample = _as_finite_vector("sample_values", sample_values)
    source_result = whole_program_value_and_grad(
        objective,
        checked_sample,
        parameters=parameters,
        trace=trace,
    )
    program_adjoint_gradient(source_result)
    compile_config = DifferentiableMLIRCompileConfig() if config is None else config
    mlir_module = compile_whole_program_ad_trace_to_mlir(source_result, compile_config)
    replay_parameters = tuple(
        Parameter(name, trainable=trainable)
        for name, trainable in zip(
            source_result.parameter_names,
            source_result.trainable,
            strict=True,
        )
    )
    return ExecutableWholeProgramADKernel(
        objective=objective,
        source_result=source_result,
        parameters=replay_parameters,
        mlir_module=mlir_module,
        parameter_names=source_result.parameter_names,
        parameter_shape=source_result.gradient.shape,
        branch_signature=_whole_program_replay_signature(source_result),
    )


def compile_whole_program_ad_trace_to_native_llvm_jit(
    objective: Callable[[Any], object],
    sample_values: Sequence[float] | FloatArray,
    parameters: Sequence[Parameter] | None = None,
    config: DifferentiableMLIRCompileConfig | None = None,
    *,
    trace: bool = True,
) -> NativeWholeProgramADKernel:
    """Compile a supported scalar program AD trace to native LLVM/JIT kernels."""

    if not callable(objective):
        raise ValueError("whole-program native AD objective must be callable")
    checked_sample = _as_finite_vector("sample_values", sample_values)
    source_result = whole_program_value_and_grad(
        objective,
        checked_sample,
        parameters=parameters,
        trace=trace,
    )
    lowering_report = analyse_whole_program_ad_native_lowering(source_result)
    if not lowering_report.supported:
        raise ValueError(
            f"native whole-program AD lowering failed closed: {lowering_report.fail_closed_reason}"
        )
    program_adjoint_gradient(source_result)
    base_symbol = f"whole_program_ad_{source_result.gradient.size}_{source_result.evaluations}"
    base_symbol = f"{base_symbol}_{source_result.method}"
    base_symbol = _safe_llvm_symbol(base_symbol)
    llvm_ir = _compile_whole_program_ad_native_llvm_ir(source_result, base_symbol)
    compile_config = DifferentiableMLIRCompileConfig() if config is None else config
    cache_key = _native_whole_program_ad_cache_key(
        source_result,
        checked_sample,
        compile_config,
        llvm_ir,
    )
    cache_hit = False
    with _NATIVE_WHOLE_PROGRAM_AD_CACHE_LOCK:
        cached_entry = _NATIVE_WHOLE_PROGRAM_AD_CACHE.get(cache_key)
    if cached_entry is None:
        native_functions = _compile_native_llvm_jit_functions(llvm_ir, base_symbol)
        verification = _verify_native_whole_program_ad_kernel(
            source_result,
            native_functions,
            checked_sample,
        )
        mlir_module = _annotate_whole_program_native_mlir(
            compile_whole_program_ad_trace_to_mlir(source_result, compile_config),
            llvm_ir,
            source_result,
        )
        cached_entry = _NativeWholeProgramADCacheEntry(
            mlir_module=mlir_module,
            llvm_ir=llvm_ir,
            native_functions=native_functions,
            verification=verification,
            supported_ops=_whole_program_native_supported_ops(source_result),
            lowering_report=lowering_report,
        )
        _store_native_whole_program_ad_cache_entry(cache_key, cached_entry)
    else:
        cache_hit = True
    mlir_module = _with_native_whole_program_cache_metadata(
        cached_entry.mlir_module,
        cache_key=cache_key,
        cache_hit=cache_hit,
    )
    replay_parameters = tuple(
        Parameter(name, trainable=trainable)
        for name, trainable in zip(
            source_result.parameter_names,
            source_result.trainable,
            strict=True,
        )
    )
    return NativeWholeProgramADKernel(
        objective=objective,
        source_result=source_result,
        parameters=replay_parameters,
        mlir_module=mlir_module,
        llvm_ir=cached_entry.llvm_ir,
        native_functions=cached_entry.native_functions,
        verification=cached_entry.verification,
        parameter_names=source_result.parameter_names,
        parameter_shape=source_result.gradient.shape,
        trace_signature=_whole_program_replay_signature(source_result),
        supported_ops=cached_entry.supported_ops,
        lowering_report=cached_entry.lowering_report,
        cache_key=cache_key,
        cache_hit=cache_hit,
    )


def _whole_program_has_control_flow(result: WholeProgramADResult) -> bool:
    return any(node.op.startswith(("branch:", "loop:", "control:")) for node in result.ir_nodes)


def _whole_program_has_unsupported_native_control_flow(result: WholeProgramADResult) -> bool:
    return any(node.op.startswith(("loop:", "control:")) for node in result.ir_nodes)


_WHOLE_PROGRAM_NATIVE_STRUCTURAL_OPS = frozenset({"parameter", "constant"})


_WHOLE_PROGRAM_NATIVE_UNARY_OPS = frozenset(
    {
        "sin",
        "cos",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "tan",
        "tanh",
        "arcsin",
        "arccos",
        "reciprocal",
        "square",
        "abs",
        "neg",
        "negative",
    }
)


_WHOLE_PROGRAM_NATIVE_BINARY_OPS = frozenset(
    {
        "add",
        "sub",
        "subtract",
        "mul",
        "multiply",
        "div",
        "divide",
        "truediv",
        "pow",
        "power",
        "maximum",
        "minimum",
        "clip",
        "where",
    }
)


_WHOLE_PROGRAM_NATIVE_LINALG_OPS = frozenset(
    {
        "linalg:det:2x2",
        "linalg:det:3x3",
        "linalg:det:4x4",
        "linalg:det:5x5",
        "linalg:inv:2x2:0:0",
        "linalg:inv:2x2:0:1",
        "linalg:inv:2x2:1:0",
        "linalg:inv:2x2:1:1",
        "linalg:solve:2x2:rhs:2:0",
        "linalg:solve:2x2:rhs:2:1",
    }
)


_WHOLE_PROGRAM_NATIVE_WIDE_DET_SIZES = frozenset(range(6, 20))


_WHOLE_PROGRAM_NATIVE_LOOP_HELPER_DET_SIZES = frozenset(range(6, 20))


_WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES = frozenset(range(5, 7))


_WHOLE_PROGRAM_NATIVE_INVERSE_SIZES = frozenset(range(3, 7))


_WHOLE_PROGRAM_NATIVE_SOLVE_VECTOR_SIZES = frozenset(range(3, 7))


_WHOLE_PROGRAM_NATIVE_SOLVE_MATRIX_SIZES = frozenset(range(2, 7))


_WHOLE_PROGRAM_NATIVE_SOLVE_MATRIX_MAX_RHS_COLS = 4


@dataclass
class _WholeProgramNativeInverseHelper:
    """Shared determinant/adjugate values for one static matrix in a native kernel."""

    determinant: str
    inverse_entries: tuple[tuple[str, ...], ...]


@dataclass
class _WholeProgramNativeSolveHelper:
    """Shared solution values for one static matrix/RHS column in a native kernel."""

    solution_entries: tuple[str, ...]


@dataclass
class _WholeProgramNativeEmissionState:
    """Per-kernel native lowering state for reusable linalg helper emissions."""

    inverse_helpers: dict[
        tuple[int, tuple[str, ...]],
        _WholeProgramNativeInverseHelper,
    ] = field(default_factory=dict)
    solve_helpers: dict[
        tuple[int, tuple[str, ...], tuple[str, ...]],
        _WholeProgramNativeSolveHelper,
    ] = field(default_factory=dict)


def native_whole_program_ad_linalg_support() -> Mapping[str, object]:
    """Return the native whole-program AD linalg support contract."""

    expression_determinant_sizes = tuple(range(2, 6))
    helper_determinant_sizes = tuple(sorted(_WHOLE_PROGRAM_NATIVE_LOOP_HELPER_DET_SIZES))
    determinant_sizes = expression_determinant_sizes + helper_determinant_sizes
    return MappingProxyType(
        {
            "determinant_expression_sizes": expression_determinant_sizes,
            "determinant_helper_sizes": helper_determinant_sizes,
            "determinant_static_dense_sizes": determinant_sizes,
            "determinant_fail_closed_from": max(determinant_sizes) + 1,
            "determinant_layout": "row_major",
            "determinant_dtype": "float64",
            "determinant_derivative": "exact_forward_partials",
            "determinant_policy": "static_dense_native_or_fail_closed",
            "inverse_sizes": (2, *tuple(sorted(_WHOLE_PROGRAM_NATIVE_INVERSE_SIZES))),
            "inverse_fail_closed_from": max(_WHOLE_PROGRAM_NATIVE_INVERSE_SIZES) + 1,
            "solve_sizes": (2, *tuple(sorted(_WHOLE_PROGRAM_NATIVE_SOLVE_VECTOR_SIZES))),
            "solve_matrix_sizes": tuple(sorted(_WHOLE_PROGRAM_NATIVE_SOLVE_MATRIX_SIZES)),
            "solve_matrix_max_rhs_columns": _WHOLE_PROGRAM_NATIVE_SOLVE_MATRIX_MAX_RHS_COLS,
            "solve_rhs_policy": "static_vector_or_matrix_rhs",
            "solve_fail_closed_from": max(_WHOLE_PROGRAM_NATIVE_SOLVE_VECTOR_SIZES) + 1,
            "quotient_linalg_helper_sizes": tuple(
                sorted(_WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES)
            ),
            "quotient_linalg_reuse_policy": "shared_determinant_adjugate_per_static_matrix",
            "quotient_linalg_unsuitable_from": max(
                _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES
            )
            + 1,
            "quotient_linalg_unsuitable_reason": (
                "full_output_inverse_and_matrix_rhs_solve_require_native_factorisation_helper"
            ),
            "trace_policy": "static_square_or_rectangular_fixed_offset",
            "unsupported_policy": "fail_closed_report_before_compile",
        }
    )


def analyse_whole_program_ad_native_lowering(
    result: WholeProgramADResult,
) -> WholeProgramADNativeLoweringReport:
    """Return the fail-closed native LLVM/JIT lowering audit for a program AD trace."""

    if not isinstance(result, WholeProgramADResult):
        raise ValueError("native lowering analysis requires a WholeProgramADResult")
    if not result.ir_nodes:
        raise ValueError("native lowering analysis requires captured IR nodes")
    lowerable_count = 0
    unsupported: list[str] = []
    lowerable: list[str] = []
    control_flow: list[str] = []
    for node in result.ir_nodes:
        if node.op.startswith(("branch:", "loop:", "control:")):
            control_flow.append(node.op)
        if _whole_program_native_node_is_lowerable(node.op):
            lowerable_count += 1
            lowerable.append(node.op)
        else:
            unsupported.append(node.op)
    unsupported_ops = tuple(dict.fromkeys(unsupported))
    lowerable_ops = tuple(dict.fromkeys(lowerable))
    effect_kinds: tuple[str, ...]
    if result.program_ir is None:
        effect_kinds = ()
    else:
        effect_kinds = tuple(dict.fromkeys(effect.kind for effect in result.program_ir.effects))
    if unsupported_ops:
        reason = "unsupported native ops: " + ", ".join(unsupported_ops)
        if any(_whole_program_native_is_unverified_static_linalg_op(op) for op in unsupported_ops):
            reason = (
                f"{reason}; native LLVM/JIT and Rust static linalg lowering blocked "
                "until independently verified executable kernels exist"
            )
    else:
        reason = "supported native LLVM/JIT lowering surface"
    return WholeProgramADNativeLoweringReport(
        supported=not unsupported_ops,
        lowerable_ops=lowerable_ops,
        unsupported_ops=unsupported_ops,
        control_flow_ops=tuple(dict.fromkeys(control_flow)),
        effect_kinds=effect_kinds,
        operation_count=len(result.ir_nodes),
        lowerable_operation_count=lowerable_count,
        unsupported_operation_count=len(result.ir_nodes) - lowerable_count,
        fail_closed_reason=reason,
    )


def _whole_program_native_is_unverified_static_linalg_op(op: str) -> bool:
    if op.startswith("linalg:matrix_power:2x2:power:2:"):
        return False
    if op.startswith("linalg:multi_dot:2x2__2x2:out:2x2:"):
        return False
    return op.startswith(("linalg:matrix_power:", "linalg:multi_dot:"))


def _whole_program_native_node_is_lowerable(op: str) -> bool:
    if op in _WHOLE_PROGRAM_NATIVE_STRUCTURAL_OPS:
        return True
    if op in _WHOLE_PROGRAM_NATIVE_UNARY_OPS:
        return True
    if op in _WHOLE_PROGRAM_NATIVE_BINARY_OPS:
        return True
    if op in _WHOLE_PROGRAM_NATIVE_LINALG_OPS:
        return True
    if _whole_program_native_wide_det_size(op) is not None:
        return True
    if _whole_program_native_inverse_spec(op) is not None:
        return True
    if _whole_program_native_solve_vector_spec(op) is not None:
        return True
    if _whole_program_native_solve_matrix_spec(op) is not None:
        return True
    if _whole_program_native_trace_input_count(op) is not None:
        return True
    if _whole_program_native_diag_input_count(op) is not None:
        return True
    if op.startswith("linalg:matrix_power:2x2:power:2:"):
        return True
    if op.startswith("linalg:multi_dot:2x2__2x2:out:2x2:"):
        return True
    return bool(op.startswith("branch:"))


def _whole_program_native_requires_runtime_recapture(result: WholeProgramADResult) -> bool:
    return _whole_program_has_control_flow(result) or any(
        node.op in {"maximum", "minimum", "clip", "where"}
        or node.op.startswith(("linalg:inv:", "linalg:solve:"))
        for node in result.ir_nodes
    )


def _whole_program_native_replay_signature(result: WholeProgramADResult) -> tuple[str, ...]:
    where_branch_ops = {
        branch_op
        for node in result.ir_nodes
        if node.op == "where" and node.inputs
        for branch_op in (_whole_program_native_where_branch_op(node.inputs[0]),)
        if branch_op is not None
    }
    control_signature = tuple(
        f"{node.index}:{node.op}:{','.join(node.inputs)}"
        for node in result.ir_nodes
        if node.op.startswith(("branch:", "loop:", "control:")) and node.op not in where_branch_ops
    )
    if control_signature:
        return control_signature
    return tuple(
        f"{node.index}:{node.op}:{','.join(_whole_program_native_signature_inputs(node))}"
        for node in result.ir_nodes
        if not (node.op.startswith("branch:") and node.op in where_branch_ops)
    )


def _whole_program_native_signature_inputs(node: Any) -> tuple[str, ...]:
    if node.op != "where" or not node.inputs:
        return tuple(node.inputs)
    return (
        _whole_program_native_where_predicate_body(node.inputs[0]),
        *tuple(node.inputs[1:]),
    )


def _whole_program_native_where_branch_op(predicate: str) -> str | None:
    if ":truth:" not in predicate:
        return None
    label, truth = predicate.rsplit(":truth:", 1)
    if truth == "1":
        return f"branch:{label}:True"
    if truth == "0":
        return f"branch:{label}:False"
    return None


def _whole_program_native_supported_ops(result: WholeProgramADResult) -> tuple[str, ...]:
    return analyse_whole_program_ad_native_lowering(result).lowerable_ops


def _native_whole_program_ad_cache_key(
    result: WholeProgramADResult,
    sample_values: NDArray[np.float64],
    config: DifferentiableMLIRCompileConfig,
    llvm_ir: str,
) -> str:
    payload = {
        "format": "native_whole_program_ad_cache.v1",
        "parameter_names": result.parameter_names,
        "trainable": result.trainable,
        "trace_signature": _whole_program_replay_signature(result),
        "method": result.method,
        "evaluations": result.evaluations,
        "ir_nodes": [
            {
                "index": node.index,
                "op": node.op,
                "inputs": node.inputs,
                "value": _fmt_llvm_float(node.value),
            }
            for node in result.ir_nodes
        ],
        "sample_values": [_fmt_llvm_float(value) for value in sample_values],
        "config": _jsonable_cache_payload(config),
        "llvm_ir_sha256": hashlib.sha256(llvm_ir.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _jsonable_cache_payload(value: object) -> object:
    if is_dataclass(value):
        return _jsonable_cache_payload(vars(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable_cache_payload(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable_cache_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable_cache_payload(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable_cache_payload(value.item())
    if isinstance(value, float):
        return _fmt_llvm_float(value)
    if isinstance(value, str | int | bool) or value is None:
        return value
    return repr(value)


def _store_native_whole_program_ad_cache_entry(
    cache_key: str,
    entry: _NativeWholeProgramADCacheEntry,
) -> None:
    with _NATIVE_WHOLE_PROGRAM_AD_CACHE_LOCK:
        if cache_key in _NATIVE_WHOLE_PROGRAM_AD_CACHE:
            return
        if len(_NATIVE_WHOLE_PROGRAM_AD_CACHE) >= _NATIVE_WHOLE_PROGRAM_AD_CACHE_MAXSIZE:
            oldest_key = next(iter(_NATIVE_WHOLE_PROGRAM_AD_CACHE))
            del _NATIVE_WHOLE_PROGRAM_AD_CACHE[oldest_key]
        _NATIVE_WHOLE_PROGRAM_AD_CACHE[cache_key] = entry


def native_whole_program_ad_compile_cache_stats() -> Mapping[str, object]:
    """Return bounded process-local native whole-program AD compile-cache metadata."""

    with _NATIVE_WHOLE_PROGRAM_AD_CACHE_LOCK:
        return MappingProxyType(
            {
                "entries": len(_NATIVE_WHOLE_PROGRAM_AD_CACHE),
                "max_size": _NATIVE_WHOLE_PROGRAM_AD_CACHE_MAXSIZE,
                "keys": tuple(_NATIVE_WHOLE_PROGRAM_AD_CACHE.keys()),
            }
        )


def clear_native_whole_program_ad_compile_cache() -> int:
    """Clear verified native whole-program AD compile-cache entries and return count."""

    with _NATIVE_WHOLE_PROGRAM_AD_CACHE_LOCK:
        removed = len(_NATIVE_WHOLE_PROGRAM_AD_CACHE)
        _NATIVE_WHOLE_PROGRAM_AD_CACHE.clear()
        return removed


def _compile_whole_program_native_helper_definitions(
    result: WholeProgramADResult,
) -> list[str]:
    """Emit compact native helper functions required by the captured trace."""

    helper_sizes: set[int] = set()
    for node in result.ir_nodes:
        wide_det_size = _whole_program_native_wide_det_size(node.op)
        if wide_det_size in _WHOLE_PROGRAM_NATIVE_LOOP_HELPER_DET_SIZES:
            helper_sizes.add(wide_det_size)
        derivative_helper_size = _whole_program_native_det_derivative_helper_size(node.op)
        if derivative_helper_size in _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES:
            helper_sizes.add(derivative_helper_size)
    lines: list[str] = []
    for size in sorted(helper_sizes):
        if lines:
            lines.append("")
        lines.extend(_compile_whole_program_native_det_loop_helper_llvm_ir(size))
    return lines


def _compile_whole_program_native_det_loop_helper_llvm_ir(size: int) -> list[str]:
    """Emit a loop-based Faddeev-LeVerrier determinant/partial helper."""

    if size not in (
        _WHOLE_PROGRAM_NATIVE_LOOP_HELPER_DET_SIZES
        | _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES
    ):
        raise ValueError("native determinant loop helper requested for unsupported size")
    total = size * size
    symbol = _whole_program_native_det_loop_helper_symbol(size)
    det_value = "%det_value" if size % 2 == 0 else "%det_value_signed"
    partial_value = "%partial_value" if size % 2 == 1 else "%partial_value_signed"
    lines = [
        f"define internal void @{symbol}(double* %matrix, double* %out) {{",
        "entry:",
        f"  %b = alloca [{total} x double]",
        f"  %product = alloca [{total} x double]",
        "  br label %init_cond",
        "",
        "init_cond:",
        "  %init_i = phi i64 [0, %entry], [%init_next, %init_body]",
        f"  %init_more = icmp slt i64 %init_i, {total}",
        "  br i1 %init_more, label %init_body, label %s1_entry",
        "",
        "init_body:",
        f"  %init_row = sdiv i64 %init_i, {size}",
        f"  %init_col = srem i64 %init_i, {size}",
        "  %init_diag = icmp eq i64 %init_row, %init_col",
        "  %init_value = select i1 %init_diag, double 1.00000000000000000e+00, "
        "double 0.00000000000000000e+00",
        f"  %init_ptr = getelementptr [{total} x double], [{total} x double]* %b, "
        "i64 0, i64 %init_i",
        "  store double %init_value, double* %init_ptr",
        "  %init_next = add i64 %init_i, 1",
        "  br label %init_cond",
        "",
    ]
    for step in range(1, size + 1):
        entry_label = f"s{step}_entry"
        product_cond = f"s{step}_product_cond"
        product_body = f"s{step}_product_body"
        inner_cond = f"s{step}_inner_cond"
        inner_body = f"s{step}_inner_body"
        inner_exit = f"s{step}_inner_exit"
        trace_cond = f"s{step}_trace_cond"
        trace_body = f"s{step}_trace_body"
        trace_exit = f"s{step}_trace_exit"
        coefficient = f"%s{step}_coefficient"
        lines.extend(
            [
                f"{entry_label}:",
                f"  br label %{product_cond}",
                "",
                f"{product_cond}:",
                f"  %s{step}_product_i = phi i64 [0, %{entry_label}], "
                f"[%s{step}_product_next, %{inner_exit}]",
                f"  %s{step}_product_more = icmp slt i64 %s{step}_product_i, {total}",
                f"  br i1 %s{step}_product_more, label %{product_body}, label %{trace_cond}",
                "",
                f"{product_body}:",
                f"  %s{step}_row = sdiv i64 %s{step}_product_i, {size}",
                f"  %s{step}_col = srem i64 %s{step}_product_i, {size}",
                f"  br label %{inner_cond}",
                "",
                f"{inner_cond}:",
                f"  %s{step}_inner_i = phi i64 [0, %{product_body}], "
                f"[%s{step}_inner_next, %{inner_body}]",
                f"  %s{step}_inner_sum = phi double [0.00000000000000000e+00, "
                f"%{product_body}], [%s{step}_inner_sum_next, %{inner_body}]",
                f"  %s{step}_inner_more = icmp slt i64 %s{step}_inner_i, {size}",
                f"  br i1 %s{step}_inner_more, label %{inner_body}, label %{inner_exit}",
                "",
                f"{inner_body}:",
                f"  %s{step}_matrix_row_offset = mul i64 %s{step}_row, {size}",
                f"  %s{step}_matrix_index = add i64 %s{step}_matrix_row_offset, %s{step}_inner_i",
                f"  %s{step}_b_row_offset = mul i64 %s{step}_inner_i, {size}",
                f"  %s{step}_b_index = add i64 %s{step}_b_row_offset, %s{step}_col",
                f"  %s{step}_matrix_ptr = getelementptr double, double* %matrix, "
                f"i64 %s{step}_matrix_index",
                f"  %s{step}_matrix_value = load double, double* %s{step}_matrix_ptr",
                f"  %s{step}_b_ptr = getelementptr [{total} x double], "
                f"[{total} x double]* %b, i64 0, i64 %s{step}_b_index",
                f"  %s{step}_b_value = load double, double* %s{step}_b_ptr",
                f"  %s{step}_term = fmul double %s{step}_matrix_value, %s{step}_b_value",
                f"  %s{step}_inner_sum_next = fadd double %s{step}_inner_sum, %s{step}_term",
                f"  %s{step}_inner_next = add i64 %s{step}_inner_i, 1",
                f"  br label %{inner_cond}",
                "",
                f"{inner_exit}:",
                f"  %s{step}_product_ptr = getelementptr [{total} x double], "
                f"[{total} x double]* %product, i64 0, i64 %s{step}_product_i",
                f"  store double %s{step}_inner_sum, double* %s{step}_product_ptr",
                f"  %s{step}_product_next = add i64 %s{step}_product_i, 1",
                f"  br label %{product_cond}",
                "",
                f"{trace_cond}:",
                f"  %s{step}_trace_i = phi i64 [0, %{product_cond}], "
                f"[%s{step}_trace_next, %{trace_body}]",
                f"  %s{step}_trace_sum = phi double [0.00000000000000000e+00, "
                f"%{product_cond}], [%s{step}_trace_sum_next, %{trace_body}]",
                f"  %s{step}_trace_more = icmp slt i64 %s{step}_trace_i, {size}",
                f"  br i1 %s{step}_trace_more, label %{trace_body}, label %{trace_exit}",
                "",
                f"{trace_body}:",
                f"  %s{step}_trace_row_offset = mul i64 %s{step}_trace_i, {size}",
                f"  %s{step}_trace_index = add i64 %s{step}_trace_row_offset, %s{step}_trace_i",
                f"  %s{step}_trace_ptr = getelementptr [{total} x double], "
                f"[{total} x double]* %product, i64 0, i64 %s{step}_trace_index",
                f"  %s{step}_trace_value = load double, double* %s{step}_trace_ptr",
                f"  %s{step}_trace_sum_next = fadd double %s{step}_trace_sum, "
                f"%s{step}_trace_value",
                f"  %s{step}_trace_next = add i64 %s{step}_trace_i, 1",
                f"  br label %{trace_cond}",
                "",
                f"{trace_exit}:",
                f"  {coefficient} = fmul double {_fmt_llvm_float(-1.0 / step)}, "
                f"%s{step}_trace_sum",
            ]
        )
        if step == size:
            if size % 2 == 0:
                lines.append(f"  %det_value = fadd double {coefficient}, 0.00000000000000000e+00")
            else:
                lines.append(
                    f"  %det_value_signed = fsub double 0.00000000000000000e+00, {coefficient}"
                )
            lines.extend(
                [
                    "  %det_out_ptr = getelementptr double, double* %out, i64 0",
                    f"  store double {det_value}, double* %det_out_ptr",
                    "  br label %partial_cond",
                    "",
                    "partial_cond:",
                    f"  %partial_i = phi i64 [0, %s{step}_trace_exit], "
                    "[%partial_next, %partial_body]",
                    f"  %partial_more = icmp slt i64 %partial_i, {total}",
                    "  br i1 %partial_more, label %partial_body, label %partial_exit",
                    "",
                    "partial_body:",
                    f"  %partial_row = sdiv i64 %partial_i, {size}",
                    f"  %partial_col = srem i64 %partial_i, {size}",
                    f"  %partial_b_row_offset = mul i64 %partial_col, {size}",
                    "  %partial_b_index = add i64 %partial_b_row_offset, %partial_row",
                    f"  %partial_b_ptr = getelementptr [{total} x double], "
                    f"[{total} x double]* %b, i64 0, i64 %partial_b_index",
                    "  %partial_b_value = load double, double* %partial_b_ptr",
                ]
            )
            if size % 2 == 0:
                lines.append(
                    "  %partial_value_signed = fsub double "
                    "0.00000000000000000e+00, %partial_b_value"
                )
            else:
                lines.append(
                    "  %partial_value = fadd double %partial_b_value, 0.00000000000000000e+00"
                )
            lines.extend(
                [
                    "  %partial_out_index = add i64 %partial_i, 1",
                    "  %partial_out_ptr = getelementptr double, double* %out, "
                    "i64 %partial_out_index",
                    f"  store double {partial_value}, double* %partial_out_ptr",
                    "  %partial_next = add i64 %partial_i, 1",
                    "  br label %partial_cond",
                    "",
                    "partial_exit:",
                    "  ret void",
                    "}",
                ]
            )
            continue
        update_cond = f"s{step}_update_cond"
        update_body = f"s{step}_update_body"
        next_entry = f"s{step + 1}_entry"
        lines.extend(
            [
                f"  br label %{update_cond}",
                "",
                f"{update_cond}:",
                f"  %s{step}_update_i = phi i64 [0, %{trace_exit}], "
                f"[%s{step}_update_next, %{update_body}]",
                f"  %s{step}_update_more = icmp slt i64 %s{step}_update_i, {total}",
                f"  br i1 %s{step}_update_more, label %{update_body}, label %{next_entry}",
                "",
                f"{update_body}:",
                f"  %s{step}_update_row = sdiv i64 %s{step}_update_i, {size}",
                f"  %s{step}_update_col = srem i64 %s{step}_update_i, {size}",
                f"  %s{step}_update_diag = icmp eq i64 %s{step}_update_row, %s{step}_update_col",
                f"  %s{step}_update_coeff = select i1 %s{step}_update_diag, "
                f"double {coefficient}, double 0.00000000000000000e+00",
                f"  %s{step}_update_product_ptr = getelementptr [{total} x double], "
                f"[{total} x double]* %product, i64 0, i64 %s{step}_update_i",
                f"  %s{step}_update_product = load double, double* %s{step}_update_product_ptr",
                f"  %s{step}_update_value = fadd double %s{step}_update_product, "
                f"%s{step}_update_coeff",
                f"  %s{step}_update_b_ptr = getelementptr [{total} x double], "
                f"[{total} x double]* %b, i64 0, i64 %s{step}_update_i",
                f"  store double %s{step}_update_value, double* %s{step}_update_b_ptr",
                f"  %s{step}_update_next = add i64 %s{step}_update_i, 1",
                f"  br label %{update_cond}",
                "",
            ]
        )
    return lines


def _whole_program_native_det_loop_helper_symbol(size: int) -> str:
    return f"scpn_det{size}_fl_value_partials"


def _with_native_whole_program_cache_metadata(
    module: MLIRModule,
    *,
    cache_key: str,
    cache_hit: bool,
) -> MLIRModule:
    metadata = dict(module.metadata)
    metadata["native_compile_cache_key"] = cache_key
    metadata["native_compile_cache_hit"] = cache_hit
    resource_counts = dict(module.resource_counts)
    resource_counts["native_compile_cache_hit"] = int(cache_hit)
    return MLIRModule(
        text=module.text,
        sha256=module.sha256,
        dialect=module.dialect,
        resource_counts=resource_counts,
        metadata=metadata,
    )


def _compile_whole_program_ad_native_llvm_ir(
    result: WholeProgramADResult,
    base_symbol: str,
) -> str:
    if result.gradient.ndim != 1 or result.gradient.size < 1:
        raise ValueError("native whole-program AD lowering requires parameters")
    lowering_report = analyse_whole_program_ad_native_lowering(result)
    if not lowering_report.supported:
        raise ValueError(
            f"native whole-program AD lowering failed closed: {lowering_report.fail_closed_reason}"
        )
    computation_lines, final_value, final_derivatives = _emit_whole_program_native_computation(
        result,
        values_pointer="%values",
    )
    (
        batch_computation_lines,
        batch_final_value,
        batch_final_derivatives,
    ) = _emit_whole_program_native_computation(
        result,
        values_pointer="%row_values",
    )
    helper_lines = _compile_whole_program_native_helper_definitions(result)

    lines = [
        "declare double @llvm.sin.f64(double)",
        "declare double @llvm.cos.f64(double)",
        "declare double @llvm.exp.f64(double)",
        "declare double @llvm.log.f64(double)",
        "declare double @llvm.sqrt.f64(double)",
        "declare double @llvm.pow.f64(double, double)",
        "declare double @llvm.asin.f64(double)",
        "declare double @llvm.acos.f64(double)",
        "",
        *helper_lines,
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        *computation_lines,
        "  %value_out_ptr = getelementptr double, double* %out, i64 0",
        f"  store double {final_value}, double* %value_out_ptr",
        "  ret void",
        "}",
        "",
        f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
        *computation_lines,
    ]
    for index, derivative in enumerate(final_derivatives):
        lines.extend(
            [
                f"  %gradient_out_ptr_{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double {derivative}, double* %gradient_out_ptr_{index}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            *computation_lines,
        ]
    )
    jvp_accumulator = _fmt_llvm_float(0.0)
    for index, derivative in enumerate(final_derivatives):
        tangent_ptr = f"%jvp_tangent_ptr_{index}"
        tangent_value = f"%jvp_tangent_{index}"
        term = f"%jvp_term_{index}"
        accumulator = f"%jvp_acc_{index}"
        lines.extend(
            [
                f"  {tangent_ptr} = getelementptr double, double* %tangent, i64 {index}",
                f"  {tangent_value} = load double, double* {tangent_ptr}",
                f"  {term} = fmul double {derivative}, {tangent_value}",
                f"  {accumulator} = fadd double {jvp_accumulator}, {term}",
            ]
        )
        jvp_accumulator = accumulator
    lines.extend(
        [
            "  %jvp_out_ptr = getelementptr double, double* %out, i64 0",
            f"  store double {jvp_accumulator}, double* %jvp_out_ptr",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            *computation_lines,
            "  %vjp_cotangent_ptr = getelementptr double, double* %cotangent, i64 0",
            "  %vjp_cotangent = load double, double* %vjp_cotangent_ptr",
        ]
    )
    for index, derivative in enumerate(final_derivatives):
        vjp_value = f"%vjp_value_{index}"
        lines.extend(
            [
                f"  {vjp_value} = fmul double {derivative}, %vjp_cotangent",
                f"  %vjp_out_ptr_{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double {vjp_value}, double* %vjp_out_ptr_{index}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            (
                f"define void @{base_symbol}_batch_value_gradient(double* %values, "
                "i64 %rows, double* %value_out, double* %gradient_out) {"
            ),
            "entry:",
            "  br label %batch_loop",
            "batch_loop:",
            "  %batch_i = phi i64 [0, %entry], [%batch_next, %batch_continue]",
            "  %batch_done = icmp eq i64 %batch_i, %rows",
            "  br i1 %batch_done, label %batch_exit, label %batch_body",
            "batch_body:",
            (
                f"  %batch_row_offset = mul i64 %batch_i, "
                f"{_fmt_llvm_int(len(result.parameter_names))}"
            ),
            "  %row_values = getelementptr double, double* %values, i64 %batch_row_offset",
            *batch_computation_lines,
            "  %batch_value_out_ptr = getelementptr double, double* %value_out, i64 %batch_i",
            f"  store double {batch_final_value}, double* %batch_value_out_ptr",
            (
                f"  %batch_gradient_row_offset = mul i64 %batch_i, "
                f"{_fmt_llvm_int(len(result.parameter_names))}"
            ),
        ]
    )
    for index, derivative in enumerate(batch_final_derivatives):
        lines.extend(
            [
                f"  %batch_gradient_offset_{index} = add i64 %batch_gradient_row_offset, {index}",
                (
                    f"  %batch_gradient_out_ptr_{index} = getelementptr double, "
                    f"double* %gradient_out, i64 %batch_gradient_offset_{index}"
                ),
                f"  store double {derivative}, double* %batch_gradient_out_ptr_{index}",
            ]
        )
    lines.extend(
        [
            "  br label %batch_continue",
            "batch_continue:",
            "  %batch_next = add i64 %batch_i, 1",
            "  br label %batch_loop",
            "batch_exit:",
            "  ret void",
            "}",
            "",
        ]
    )
    lines.extend(
        _emit_whole_program_native_batch_jvp(
            result,
            base_symbol,
            batch_computation_lines,
            batch_final_derivatives,
        )
    )
    lines.extend(
        _emit_whole_program_native_batch_vjp(
            result,
            base_symbol,
            batch_computation_lines,
            batch_final_derivatives,
        )
    )
    return "\n".join(lines)


def _emit_whole_program_native_batch_jvp(
    result: WholeProgramADResult,
    base_symbol: str,
    computation_lines: Sequence[str],
    final_derivatives: Sequence[str],
) -> list[str]:
    parameter_count = len(result.parameter_names)
    lines = [
        (
            f"define void @{base_symbol}_batch_jvp(double* %values, double* %tangents, "
            "i64 %rows, double* %out) {"
        ),
        "entry:",
        "  br label %batch_jvp_loop",
        "batch_jvp_loop:",
        "  %batch_jvp_i = phi i64 [0, %entry], [%batch_jvp_next, %batch_jvp_continue]",
        "  %batch_jvp_done = icmp eq i64 %batch_jvp_i, %rows",
        "  br i1 %batch_jvp_done, label %batch_jvp_exit, label %batch_jvp_body",
        "batch_jvp_body:",
        f"  %batch_jvp_row_offset = mul i64 %batch_jvp_i, {_fmt_llvm_int(parameter_count)}",
        "  %row_values = getelementptr double, double* %values, i64 %batch_jvp_row_offset",
        "  %row_tangents = getelementptr double, double* %tangents, i64 %batch_jvp_row_offset",
        *computation_lines,
    ]
    accumulator = _fmt_llvm_float(0.0)
    for index, derivative in enumerate(final_derivatives):
        tangent_ptr = f"%batch_jvp_tangent_ptr_{index}"
        tangent_value = f"%batch_jvp_tangent_{index}"
        term = f"%batch_jvp_term_{index}"
        next_accumulator = f"%batch_jvp_acc_{index}"
        lines.extend(
            [
                f"  {tangent_ptr} = getelementptr double, double* %row_tangents, i64 {index}",
                f"  {tangent_value} = load double, double* {tangent_ptr}",
                f"  {term} = fmul double {derivative}, {tangent_value}",
                f"  {next_accumulator} = fadd double {accumulator}, {term}",
            ]
        )
        accumulator = next_accumulator
    lines.extend(
        [
            "  %batch_jvp_out_ptr = getelementptr double, double* %out, i64 %batch_jvp_i",
            f"  store double {accumulator}, double* %batch_jvp_out_ptr",
            "  br label %batch_jvp_continue",
            "batch_jvp_continue:",
            "  %batch_jvp_next = add i64 %batch_jvp_i, 1",
            "  br label %batch_jvp_loop",
            "batch_jvp_exit:",
            "  ret void",
            "}",
            "",
        ]
    )
    return lines


def _emit_whole_program_native_batch_vjp(
    result: WholeProgramADResult,
    base_symbol: str,
    computation_lines: Sequence[str],
    final_derivatives: Sequence[str],
) -> list[str]:
    parameter_count = len(result.parameter_names)
    lines = [
        (
            f"define void @{base_symbol}_batch_vjp(double* %values, double* %cotangents, "
            "i64 %rows, double* %out) {"
        ),
        "entry:",
        "  br label %batch_vjp_loop",
        "batch_vjp_loop:",
        "  %batch_vjp_i = phi i64 [0, %entry], [%batch_vjp_next, %batch_vjp_continue]",
        "  %batch_vjp_done = icmp eq i64 %batch_vjp_i, %rows",
        "  br i1 %batch_vjp_done, label %batch_vjp_exit, label %batch_vjp_body",
        "batch_vjp_body:",
        f"  %batch_vjp_row_offset = mul i64 %batch_vjp_i, {_fmt_llvm_int(parameter_count)}",
        "  %row_values = getelementptr double, double* %values, i64 %batch_vjp_row_offset",
        *computation_lines,
        "  %batch_vjp_cotangent_ptr = getelementptr double, double* %cotangents, i64 %batch_vjp_i",
        "  %batch_vjp_cotangent = load double, double* %batch_vjp_cotangent_ptr",
        f"  %batch_vjp_gradient_row_offset = mul i64 %batch_vjp_i, {_fmt_llvm_int(parameter_count)}",
    ]
    for index, derivative in enumerate(final_derivatives):
        vjp_value = f"%batch_vjp_value_{index}"
        lines.extend(
            [
                f"  {vjp_value} = fmul double {derivative}, %batch_vjp_cotangent",
                f"  %batch_vjp_offset_{index} = add i64 %batch_vjp_gradient_row_offset, {index}",
                (
                    f"  %batch_vjp_out_ptr_{index} = getelementptr double, "
                    f"double* %out, i64 %batch_vjp_offset_{index}"
                ),
                f"  store double {vjp_value}, double* %batch_vjp_out_ptr_{index}",
            ]
        )
    lines.extend(
        [
            "  br label %batch_vjp_continue",
            "batch_vjp_continue:",
            "  %batch_vjp_next = add i64 %batch_vjp_i, 1",
            "  br label %batch_vjp_loop",
            "batch_vjp_exit:",
            "  ret void",
            "}",
            "",
        ]
    )
    return lines


def _emit_whole_program_native_computation(
    result: WholeProgramADResult,
    *,
    values_pointer: str,
) -> tuple[list[str], str, tuple[str, ...]]:
    parameter_count = int(result.gradient.size)
    lines: list[str] = []
    names = set(result.parameter_names)
    emission_state = _WholeProgramNativeEmissionState()
    for node in result.ir_nodes:
        value_name = _whole_program_native_value_name(node.index)
        if node.op == "parameter":
            if len(node.inputs) != 1 or node.inputs[0] not in names:
                raise ValueError("native whole-program AD parameter node is malformed")
            parameter_index = result.parameter_names.index(node.inputs[0])
            lines.extend(
                [
                    (
                        f"  %param_ptr_{node.index} = getelementptr double, "
                        f"double* {values_pointer}, i64 {parameter_index}"
                    ),
                    f"  {value_name} = load double, double* %param_ptr_{node.index}",
                ]
            )
            for derivative_index in range(parameter_count):
                seed = (
                    1.0
                    if result.trainable[parameter_index] and derivative_index == parameter_index
                    else 0.0
                )
                lines.append(
                    f"  {_whole_program_native_derivative_name(node.index, derivative_index)} = "
                    f"fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(seed)}"
                )
            continue
        if node.op == "constant":
            lines.append(
                f"  {value_name} = fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(node.value)}"
            )
            for derivative_index in range(parameter_count):
                lines.append(
                    f"  {_whole_program_native_derivative_name(node.index, derivative_index)} = "
                    f"fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(0.0)}"
                )
            continue
        if node.op.startswith("branch:"):
            if node.inputs:
                raise ValueError("native whole-program AD branch node must be signature-only")
            lines.append(
                f"  {value_name} = fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(node.value)}"
            )
            for derivative_index in range(parameter_count):
                lines.append(
                    f"  {_whole_program_native_derivative_name(node.index, derivative_index)} = "
                    f"fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(0.0)}"
                )
            continue
        _emit_whole_program_native_operation(lines, result, node, emission_state)
    if not result.ir_nodes:
        raise ValueError("native whole-program AD lowering requires IR nodes")
    final_index = result.ir_nodes[-1].index
    return (
        lines,
        _whole_program_native_value_name(final_index),
        tuple(
            _whole_program_native_derivative_name(final_index, index)
            for index in range(parameter_count)
        ),
    )


def _emit_whole_program_native_operation(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    emission_state: _WholeProgramNativeEmissionState,
) -> None:
    parameter_count = int(result.gradient.size)
    value_name = _whole_program_native_value_name(node.index)
    inputs = tuple(node.inputs)
    if node.op in {
        "sin",
        "cos",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "tan",
        "tanh",
        "arcsin",
        "arccos",
        "reciprocal",
        "square",
        "abs",
        "neg",
        "negative",
    }:
        if len(inputs) != 1:
            raise ValueError(f"native operation {node.op} expects one input")
        argument = _whole_program_native_operand(inputs[0])
        if node.op == "sin":
            lines.append(f"  {value_name} = call double @llvm.sin.f64(double {argument})")
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = call double @llvm.cos.f64(double {argument})")
        elif node.op == "cos":
            cos_value = value_name
            sin_value = f"%sin_{node.index}"
            lines.append(f"  {cos_value} = call double @llvm.cos.f64(double {argument})")
            lines.append(f"  {sin_value} = call double @llvm.sin.f64(double {argument})")
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fsub double {_fmt_llvm_float(0.0)}, {sin_value}")
        elif node.op == "exp":
            lines.append(f"  {value_name} = call double @llvm.exp.f64(double {argument})")
            local_factor = value_name
        elif node.op == "expm1":
            exp_value = f"%exp_{node.index}"
            lines.append(f"  {exp_value} = call double @llvm.exp.f64(double {argument})")
            lines.append(f"  {value_name} = fsub double {exp_value}, {_fmt_llvm_float(1.0)}")
            local_factor = exp_value
        elif node.op == "log":
            lines.append(f"  {value_name} = call double @llvm.log.f64(double {argument})")
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(1.0)}, {argument}")
        elif node.op == "log1p":
            shifted = f"%log1p_shifted_{node.index}"
            lines.append(f"  {shifted} = fadd double {_fmt_llvm_float(1.0)}, {argument}")
            lines.append(f"  {value_name} = call double @llvm.log.f64(double {shifted})")
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(1.0)}, {shifted}")
        elif node.op == "sqrt":
            lines.append(f"  {value_name} = call double @llvm.sqrt.f64(double {argument})")
            local_factor = f"%factor_{node.index}"
            denominator = f"%sqrt_denominator_{node.index}"
            lines.append(f"  {denominator} = fmul double {_fmt_llvm_float(2.0)}, {value_name}")
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(1.0)}, {denominator}")
        elif node.op == "tan":
            sin_value = f"%tan_sin_{node.index}"
            cos_value = f"%tan_cos_{node.index}"
            cos_squared = f"%tan_cos_squared_{node.index}"
            lines.extend(
                [
                    f"  {sin_value} = call double @llvm.sin.f64(double {argument})",
                    f"  {cos_value} = call double @llvm.cos.f64(double {argument})",
                    f"  {value_name} = fdiv double {sin_value}, {cos_value}",
                    f"  {cos_squared} = fmul double {cos_value}, {cos_value}",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(1.0)}, {cos_squared}")
        elif node.op == "tanh":
            doubled = f"%tanh_doubled_{node.index}"
            exp_value = f"%tanh_exp_{node.index}"
            numerator = f"%tanh_numerator_{node.index}"
            denominator = f"%tanh_denominator_{node.index}"
            squared = f"%tanh_squared_{node.index}"
            lines.extend(
                [
                    f"  {doubled} = fmul double {_fmt_llvm_float(2.0)}, {argument}",
                    f"  {exp_value} = call double @llvm.exp.f64(double {doubled})",
                    f"  {numerator} = fsub double {exp_value}, {_fmt_llvm_float(1.0)}",
                    f"  {denominator} = fadd double {exp_value}, {_fmt_llvm_float(1.0)}",
                    f"  {value_name} = fdiv double {numerator}, {denominator}",
                    f"  {squared} = fmul double {value_name}, {value_name}",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fsub double {_fmt_llvm_float(1.0)}, {squared}")
        elif node.op == "arcsin":
            argument_squared = f"%arcsin_argument_squared_{node.index}"
            radicand = f"%arcsin_radicand_{node.index}"
            root = f"%arcsin_root_{node.index}"
            lines.extend(
                [
                    f"  {value_name} = call double @llvm.asin.f64(double {argument})",
                    f"  {argument_squared} = fmul double {argument}, {argument}",
                    f"  {radicand} = fsub double {_fmt_llvm_float(1.0)}, {argument_squared}",
                    f"  {root} = call double @llvm.sqrt.f64(double {radicand})",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(1.0)}, {root}")
        elif node.op == "arccos":
            argument_squared = f"%arccos_argument_squared_{node.index}"
            radicand = f"%arccos_radicand_{node.index}"
            root = f"%arccos_root_{node.index}"
            positive_factor = f"%arccos_positive_factor_{node.index}"
            lines.extend(
                [
                    f"  {value_name} = call double @llvm.acos.f64(double {argument})",
                    f"  {argument_squared} = fmul double {argument}, {argument}",
                    f"  {radicand} = fsub double {_fmt_llvm_float(1.0)}, {argument_squared}",
                    f"  {root} = call double @llvm.sqrt.f64(double {radicand})",
                    f"  {positive_factor} = fdiv double {_fmt_llvm_float(1.0)}, {root}",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(
                f"  {local_factor} = fsub double {_fmt_llvm_float(0.0)}, {positive_factor}"
            )
        elif node.op == "reciprocal":
            denominator = f"%reciprocal_denominator_{node.index}"
            lines.extend(
                [
                    f"  {value_name} = fdiv double {_fmt_llvm_float(1.0)}, {argument}",
                    f"  {denominator} = fmul double {argument}, {argument}",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(-1.0)}, {denominator}")
        elif node.op == "square":
            lines.append(f"  {value_name} = fmul double {argument}, {argument}")
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fmul double {_fmt_llvm_float(2.0)}, {argument}")
        elif node.op == "abs":
            squared = f"%abs_squared_{node.index}"
            lines.extend(
                [
                    f"  {squared} = fmul double {argument}, {argument}",
                    f"  {value_name} = call double @llvm.sqrt.f64(double {squared})",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {argument}, {value_name}")
        else:
            lines.append(f"  {value_name} = fsub double {_fmt_llvm_float(0.0)}, {argument}")
            local_factor = _fmt_llvm_float(-1.0)
        for derivative_index in range(parameter_count):
            argument_derivative = _whole_program_native_derivative_operand(
                inputs[0], derivative_index
            )
            lines.append(
                f"  {_whole_program_native_derivative_name(node.index, derivative_index)} = "
                f"fmul double {local_factor}, {argument_derivative}"
            )
        return
    if node.op == "linalg:det:2x2":
        if len(inputs) != 4:
            raise ValueError("native linalg:det:2x2 expects four matrix inputs")
        a_value = _whole_program_native_operand(inputs[0])
        b_value = _whole_program_native_operand(inputs[1])
        c_value = _whole_program_native_operand(inputs[2])
        d_value = _whole_program_native_operand(inputs[3])
        diagonal_product = f"%det2_diag_{node.index}"
        off_diagonal_product = f"%det2_offdiag_{node.index}"
        lines.extend(
            [
                f"  {diagonal_product} = fmul double {a_value}, {d_value}",
                f"  {off_diagonal_product} = fmul double {b_value}, {c_value}",
                f"  {value_name} = fsub double {diagonal_product}, {off_diagonal_product}",
            ]
        )
        for derivative_index in range(parameter_count):
            a_derivative = _whole_program_native_derivative_operand(inputs[0], derivative_index)
            b_derivative = _whole_program_native_derivative_operand(inputs[1], derivative_index)
            c_derivative = _whole_program_native_derivative_operand(inputs[2], derivative_index)
            d_derivative = _whole_program_native_derivative_operand(inputs[3], derivative_index)
            left_a = f"%d{node.index}_{derivative_index}_det2_a"
            left_d = f"%d{node.index}_{derivative_index}_det2_d"
            right_b = f"%d{node.index}_{derivative_index}_det2_b"
            right_c = f"%d{node.index}_{derivative_index}_det2_c"
            diagonal_derivative = f"%d{node.index}_{derivative_index}_det2_diag"
            off_diagonal_derivative = f"%d{node.index}_{derivative_index}_det2_offdiag"
            derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
            lines.extend(
                [
                    f"  {left_a} = fmul double {a_derivative}, {d_value}",
                    f"  {left_d} = fmul double {a_value}, {d_derivative}",
                    f"  {diagonal_derivative} = fadd double {left_a}, {left_d}",
                    f"  {right_b} = fmul double {b_derivative}, {c_value}",
                    f"  {right_c} = fmul double {b_value}, {c_derivative}",
                    f"  {off_diagonal_derivative} = fadd double {right_b}, {right_c}",
                    f"  {derivative_name} = fsub double {diagonal_derivative}, "
                    f"{off_diagonal_derivative}",
                ]
            )
        return
    if node.op == "linalg:det:3x3":
        if len(inputs) != 9:
            raise ValueError("native linalg:det:3x3 expects nine matrix inputs")
        _emit_whole_program_native_det3(lines, result, node, value_name, inputs)
        return
    if node.op == "linalg:det:4x4":
        if len(inputs) != 16:
            raise ValueError("native linalg:det:4x4 expects sixteen matrix inputs")
        _emit_whole_program_native_det4(lines, result, node, value_name, inputs)
        return
    if node.op == "linalg:det:5x5":
        if len(inputs) != 25:
            raise ValueError("native linalg:det:5x5 expects twenty-five matrix inputs")
        _emit_whole_program_native_det5(lines, result, node, value_name, inputs)
        return
    wide_det_size = _whole_program_native_wide_det_size(node.op)
    if wide_det_size is not None:
        expected_inputs = wide_det_size * wide_det_size
        if len(inputs) != expected_inputs:
            raise ValueError(f"native {node.op} expects {expected_inputs} matrix inputs")
        if wide_det_size in _WHOLE_PROGRAM_NATIVE_LOOP_HELPER_DET_SIZES:
            _emit_whole_program_native_det_loop_helper_call(
                lines,
                result,
                node,
                value_name,
                inputs,
                size=wide_det_size,
                prefix=f"det{wide_det_size}",
            )
        else:
            _emit_whole_program_native_det_faddeev_leverrier(
                lines,
                result,
                node,
                value_name,
                inputs,
                size=wide_det_size,
                prefix=f"det{wide_det_size}",
            )
        return
    trace_input_count = _whole_program_native_trace_input_count(node.op)
    if trace_input_count is not None:
        if len(inputs) != trace_input_count:
            raise ValueError(f"native {node.op} expects {trace_input_count} diagonal inputs")
        _emit_whole_program_native_sum(
            lines,
            result,
            node,
            value_name,
            inputs,
            "trace",
        )
        return
    diag_input_count = _whole_program_native_diag_input_count(node.op)
    if diag_input_count is not None:
        if len(inputs) != diag_input_count:
            raise ValueError(f"native {node.op} expects {diag_input_count} diagonal input")
        _emit_whole_program_native_sum(
            lines,
            result,
            node,
            value_name,
            inputs,
            "diag",
        )
        return
    if node.op.startswith("linalg:matrix_power:2x2:power:2:"):
        if len(inputs) != 4:
            raise ValueError("native linalg:matrix_power:2x2:power:2 expects four inputs")
        output_inputs = {
            "linalg:matrix_power:2x2:power:2:0:0": (inputs[0], inputs[0], inputs[1], inputs[2]),
            "linalg:matrix_power:2x2:power:2:0:1": (inputs[0], inputs[1], inputs[1], inputs[3]),
            "linalg:matrix_power:2x2:power:2:1:0": (inputs[2], inputs[0], inputs[3], inputs[2]),
            "linalg:matrix_power:2x2:power:2:1:1": (inputs[2], inputs[1], inputs[3], inputs[3]),
        }.get(node.op)
        if output_inputs is None:
            raise ValueError(f"native whole-program AD lowering does not support op {node.op}")
        _emit_whole_program_native_product_sum2(
            lines,
            result,
            node,
            value_name,
            output_inputs,
            "matrix_power2",
        )
        return
    if node.op.startswith("linalg:multi_dot:2x2__2x2:out:2x2:"):
        if len(inputs) != 8:
            raise ValueError("native linalg:multi_dot:2x2__2x2 expects eight inputs")
        output_inputs = {
            "linalg:multi_dot:2x2__2x2:out:2x2:0": (
                inputs[0],
                inputs[4],
                inputs[1],
                inputs[6],
            ),
            "linalg:multi_dot:2x2__2x2:out:2x2:1": (
                inputs[0],
                inputs[5],
                inputs[1],
                inputs[7],
            ),
            "linalg:multi_dot:2x2__2x2:out:2x2:2": (
                inputs[2],
                inputs[4],
                inputs[3],
                inputs[6],
            ),
            "linalg:multi_dot:2x2__2x2:out:2x2:3": (
                inputs[2],
                inputs[5],
                inputs[3],
                inputs[7],
            ),
        }.get(node.op)
        if output_inputs is None:
            raise ValueError(f"native whole-program AD lowering does not support op {node.op}")
        _emit_whole_program_native_product_sum2(
            lines,
            result,
            node,
            value_name,
            output_inputs,
            "multi_dot2",
        )
        return
    if node.op.startswith("linalg:inv:2x2:"):
        if len(inputs) != 4:
            raise ValueError("native linalg:inv:2x2 expects four matrix inputs")
        a_value = _whole_program_native_operand(inputs[0])
        b_value = _whole_program_native_operand(inputs[1])
        c_value = _whole_program_native_operand(inputs[2])
        d_value = _whole_program_native_operand(inputs[3])
        determinant = f"%inv2_det_{node.index}"
        diagonal_product = f"%inv2_diag_{node.index}"
        off_diagonal_product = f"%inv2_offdiag_{node.index}"
        numerator = f"%inv2_num_{node.index}"
        lines.extend(
            [
                f"  {diagonal_product} = fmul double {a_value}, {d_value}",
                f"  {off_diagonal_product} = fmul double {b_value}, {c_value}",
                f"  {determinant} = fsub double {diagonal_product}, {off_diagonal_product}",
            ]
        )
        if node.op == "linalg:inv:2x2:0:0":
            lines.append(f"  {numerator} = fadd double {d_value}, {_fmt_llvm_float(0.0)}")
            numerator_input = inputs[3]
            numerator_sign = 1.0
        elif node.op == "linalg:inv:2x2:0:1":
            lines.append(f"  {numerator} = fsub double {_fmt_llvm_float(0.0)}, {b_value}")
            numerator_input = inputs[1]
            numerator_sign = -1.0
        elif node.op == "linalg:inv:2x2:1:0":
            lines.append(f"  {numerator} = fsub double {_fmt_llvm_float(0.0)}, {c_value}")
            numerator_input = inputs[2]
            numerator_sign = -1.0
        elif node.op == "linalg:inv:2x2:1:1":
            lines.append(f"  {numerator} = fadd double {a_value}, {_fmt_llvm_float(0.0)}")
            numerator_input = inputs[0]
            numerator_sign = 1.0
        else:
            raise ValueError(f"native whole-program AD lowering does not support op {node.op}")
        lines.append(f"  {value_name} = fdiv double {numerator}, {determinant}")
        determinant_squared = f"%inv2_det_sq_{node.index}"
        lines.append(f"  {determinant_squared} = fmul double {determinant}, {determinant}")
        for derivative_index in range(parameter_count):
            a_derivative = _whole_program_native_derivative_operand(inputs[0], derivative_index)
            b_derivative = _whole_program_native_derivative_operand(inputs[1], derivative_index)
            c_derivative = _whole_program_native_derivative_operand(inputs[2], derivative_index)
            d_derivative = _whole_program_native_derivative_operand(inputs[3], derivative_index)
            numerator_derivative_base = _whole_program_native_derivative_operand(
                numerator_input, derivative_index
            )
            numerator_derivative = f"%d{node.index}_{derivative_index}_inv2_num"
            if numerator_sign < 0.0:
                lines.append(
                    f"  {numerator_derivative} = fsub double {_fmt_llvm_float(0.0)}, "
                    f"{numerator_derivative_base}"
                )
            else:
                lines.append(
                    f"  {numerator_derivative} = fadd double {numerator_derivative_base}, "
                    f"{_fmt_llvm_float(0.0)}"
                )
            det_left_a = f"%d{node.index}_{derivative_index}_inv2_det_a"
            det_left_d = f"%d{node.index}_{derivative_index}_inv2_det_d"
            det_right_b = f"%d{node.index}_{derivative_index}_inv2_det_b"
            det_right_c = f"%d{node.index}_{derivative_index}_inv2_det_c"
            det_diag_derivative = f"%d{node.index}_{derivative_index}_inv2_det_diag"
            det_offdiag_derivative = f"%d{node.index}_{derivative_index}_inv2_det_offdiag"
            determinant_derivative = f"%d{node.index}_{derivative_index}_inv2_det"
            numerator_term = f"%d{node.index}_{derivative_index}_inv2_quot_num"
            denominator_term = f"%d{node.index}_{derivative_index}_inv2_quot_den"
            quotient_numerator = f"%d{node.index}_{derivative_index}_inv2_quotient_num"
            derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
            lines.extend(
                [
                    f"  {det_left_a} = fmul double {a_derivative}, {d_value}",
                    f"  {det_left_d} = fmul double {a_value}, {d_derivative}",
                    f"  {det_diag_derivative} = fadd double {det_left_a}, {det_left_d}",
                    f"  {det_right_b} = fmul double {b_derivative}, {c_value}",
                    f"  {det_right_c} = fmul double {b_value}, {c_derivative}",
                    f"  {det_offdiag_derivative} = fadd double {det_right_b}, {det_right_c}",
                    f"  {determinant_derivative} = fsub double {det_diag_derivative}, "
                    f"{det_offdiag_derivative}",
                    f"  {numerator_term} = fmul double {numerator_derivative}, {determinant}",
                    f"  {denominator_term} = fmul double {numerator}, {determinant_derivative}",
                    f"  {quotient_numerator} = fsub double {numerator_term}, {denominator_term}",
                    f"  {derivative_name} = fdiv double {quotient_numerator}, "
                    f"{determinant_squared}",
                ]
            )
        return
    inverse_spec = _whole_program_native_inverse_spec(node.op)
    if inverse_spec is not None:
        inverse_size, output_row, output_col = inverse_spec
        expected_inputs = inverse_size * inverse_size
        if len(inputs) != expected_inputs:
            raise ValueError(f"native {node.op} expects matrix inputs")
        _emit_whole_program_native_inverse_fixed(
            lines,
            result,
            node,
            value_name,
            inputs,
            size=inverse_size,
            output_row=output_row,
            output_col=output_col,
            prefix=f"inv{inverse_size}",
            emission_state=emission_state,
        )
        return
    if node.op.startswith("linalg:solve:2x2:rhs:2:"):
        if len(inputs) != 6:
            raise ValueError("native linalg:solve:2x2:rhs:2 expects matrix and rhs inputs")
        a_value = _whole_program_native_operand(inputs[0])
        b_value = _whole_program_native_operand(inputs[1])
        c_value = _whole_program_native_operand(inputs[2])
        d_value = _whole_program_native_operand(inputs[3])
        rhs0_value = _whole_program_native_operand(inputs[4])
        rhs1_value = _whole_program_native_operand(inputs[5])
        determinant = f"%solve2_det_{node.index}"
        diagonal_product = f"%solve2_diag_{node.index}"
        off_diagonal_product = f"%solve2_offdiag_{node.index}"
        numerator = f"%solve2_num_{node.index}"
        first_product = f"%solve2_num_first_{node.index}"
        second_product = f"%solve2_num_second_{node.index}"
        lines.extend(
            [
                f"  {diagonal_product} = fmul double {a_value}, {d_value}",
                f"  {off_diagonal_product} = fmul double {b_value}, {c_value}",
                f"  {determinant} = fsub double {diagonal_product}, {off_diagonal_product}",
            ]
        )
        if node.op == "linalg:solve:2x2:rhs:2:0":
            lines.extend(
                [
                    f"  {first_product} = fmul double {d_value}, {rhs0_value}",
                    f"  {second_product} = fmul double {b_value}, {rhs1_value}",
                ]
            )
            first_inputs = (inputs[3], inputs[4], d_value, rhs0_value)
            second_inputs = (inputs[1], inputs[5], b_value, rhs1_value)
        elif node.op == "linalg:solve:2x2:rhs:2:1":
            lines.extend(
                [
                    f"  {first_product} = fmul double {a_value}, {rhs1_value}",
                    f"  {second_product} = fmul double {c_value}, {rhs0_value}",
                ]
            )
            first_inputs = (inputs[0], inputs[5], a_value, rhs1_value)
            second_inputs = (inputs[2], inputs[4], c_value, rhs0_value)
        else:
            raise ValueError(f"native whole-program AD lowering does not support op {node.op}")
        lines.extend(
            [
                f"  {numerator} = fsub double {first_product}, {second_product}",
                f"  {value_name} = fdiv double {numerator}, {determinant}",
            ]
        )
        determinant_squared = f"%solve2_det_sq_{node.index}"
        lines.append(f"  {determinant_squared} = fmul double {determinant}, {determinant}")
        for derivative_index in range(parameter_count):
            a_derivative = _whole_program_native_derivative_operand(inputs[0], derivative_index)
            b_derivative = _whole_program_native_derivative_operand(inputs[1], derivative_index)
            c_derivative = _whole_program_native_derivative_operand(inputs[2], derivative_index)
            d_derivative = _whole_program_native_derivative_operand(inputs[3], derivative_index)
            first_left_derivative = _whole_program_native_derivative_operand(
                first_inputs[0], derivative_index
            )
            first_right_derivative = _whole_program_native_derivative_operand(
                first_inputs[1], derivative_index
            )
            second_left_derivative = _whole_program_native_derivative_operand(
                second_inputs[0], derivative_index
            )
            second_right_derivative = _whole_program_native_derivative_operand(
                second_inputs[1], derivative_index
            )
            first_left_term = f"%d{node.index}_{derivative_index}_solve2_first_left"
            first_right_term = f"%d{node.index}_{derivative_index}_solve2_first_right"
            first_derivative = f"%d{node.index}_{derivative_index}_solve2_first"
            second_left_term = f"%d{node.index}_{derivative_index}_solve2_second_left"
            second_right_term = f"%d{node.index}_{derivative_index}_solve2_second_right"
            second_derivative = f"%d{node.index}_{derivative_index}_solve2_second"
            numerator_derivative = f"%d{node.index}_{derivative_index}_solve2_num"
            det_left_a = f"%d{node.index}_{derivative_index}_solve2_det_a"
            det_left_d = f"%d{node.index}_{derivative_index}_solve2_det_d"
            det_right_b = f"%d{node.index}_{derivative_index}_solve2_det_b"
            det_right_c = f"%d{node.index}_{derivative_index}_solve2_det_c"
            det_diag_derivative = f"%d{node.index}_{derivative_index}_solve2_det_diag"
            det_offdiag_derivative = f"%d{node.index}_{derivative_index}_solve2_det_offdiag"
            determinant_derivative = f"%d{node.index}_{derivative_index}_solve2_det"
            numerator_term = f"%d{node.index}_{derivative_index}_solve2_quot_num"
            denominator_term = f"%d{node.index}_{derivative_index}_solve2_quot_den"
            quotient_numerator = f"%d{node.index}_{derivative_index}_solve2_quotient_num"
            derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
            lines.extend(
                [
                    f"  {first_left_term} = fmul double {first_left_derivative}, "
                    f"{first_inputs[3]}",
                    f"  {first_right_term} = fmul double {first_inputs[2]}, "
                    f"{first_right_derivative}",
                    f"  {first_derivative} = fadd double {first_left_term}, {first_right_term}",
                    f"  {second_left_term} = fmul double {second_left_derivative}, "
                    f"{second_inputs[3]}",
                    f"  {second_right_term} = fmul double {second_inputs[2]}, "
                    f"{second_right_derivative}",
                    f"  {second_derivative} = fadd double {second_left_term}, {second_right_term}",
                    f"  {numerator_derivative} = fsub double {first_derivative}, "
                    f"{second_derivative}",
                    f"  {det_left_a} = fmul double {a_derivative}, {d_value}",
                    f"  {det_left_d} = fmul double {a_value}, {d_derivative}",
                    f"  {det_diag_derivative} = fadd double {det_left_a}, {det_left_d}",
                    f"  {det_right_b} = fmul double {b_derivative}, {c_value}",
                    f"  {det_right_c} = fmul double {b_value}, {c_derivative}",
                    f"  {det_offdiag_derivative} = fadd double {det_right_b}, {det_right_c}",
                    f"  {determinant_derivative} = fsub double {det_diag_derivative}, "
                    f"{det_offdiag_derivative}",
                    f"  {numerator_term} = fmul double {numerator_derivative}, {determinant}",
                    f"  {denominator_term} = fmul double {numerator}, {determinant_derivative}",
                    f"  {quotient_numerator} = fsub double {numerator_term}, {denominator_term}",
                    f"  {derivative_name} = fdiv double {quotient_numerator}, "
                    f"{determinant_squared}",
                ]
            )
        return
    solve_spec = _whole_program_native_solve_vector_spec(node.op)
    if solve_spec is not None:
        solve_size, output_row = solve_spec
        expected_inputs = solve_size * solve_size + solve_size
        if len(inputs) != expected_inputs:
            raise ValueError(f"native {node.op} expects matrix and rhs inputs")
        _emit_whole_program_native_solve_fixed(
            lines,
            result,
            node,
            value_name,
            inputs,
            size=solve_size,
            output_row=output_row,
            prefix=f"solve{solve_size}",
            emission_state=emission_state,
        )
        return
    solve_matrix_spec = _whole_program_native_solve_matrix_spec(node.op)
    if solve_matrix_spec is not None:
        solve_size, rhs_cols, output_row, output_col = solve_matrix_spec
        matrix_size = solve_size * solve_size
        expected_inputs = matrix_size + solve_size * rhs_cols
        if len(inputs) != expected_inputs:
            raise ValueError(f"native {node.op} expects matrix and rhs matrix inputs")
        rhs_column = tuple(
            inputs[matrix_size + row * rhs_cols + output_col] for row in range(solve_size)
        )
        _emit_whole_program_native_solve_fixed(
            lines,
            result,
            node,
            value_name,
            (*inputs[:matrix_size], *rhs_column),
            size=solve_size,
            output_row=output_row,
            prefix=f"solve{solve_size}m{rhs_cols}",
            emission_state=emission_state,
        )
        return
    if node.op not in {
        "add",
        "sub",
        "subtract",
        "mul",
        "multiply",
        "div",
        "divide",
        "truediv",
        "pow",
        "power",
        "maximum",
        "minimum",
        "clip",
        "where",
    }:
        raise ValueError(f"native whole-program AD lowering does not support op {node.op}")
    if node.op == "where":
        if len(inputs) != 3:
            raise ValueError(
                "native operation where expects predicate, true value, and false value"
            )
        predicate = _emit_whole_program_native_where_predicate(lines, node.index, inputs[0])
        true_value = _whole_program_native_operand(inputs[1])
        false_value = _whole_program_native_operand(inputs[2])
        lines.append(
            f"  {value_name} = select i1 {predicate}, double {true_value}, double {false_value}"
        )
        for derivative_index in range(parameter_count):
            true_derivative = _whole_program_native_derivative_operand(inputs[1], derivative_index)
            false_derivative = _whole_program_native_derivative_operand(
                inputs[2], derivative_index
            )
            derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
            lines.append(
                f"  {derivative_name} = select i1 {predicate}, "
                f"double {true_derivative}, double {false_derivative}"
            )
        return
    if node.op == "clip":
        if len(inputs) != 3:
            raise ValueError("native operation clip expects value, lower, and upper")
        source_value = _whole_program_native_operand(inputs[0])
        lower_value = _whole_program_native_operand(inputs[1])
        upper_value = _whole_program_native_operand(inputs[2])
        below_predicate = f"%clip_below_pred_{node.index}"
        above_predicate = f"%clip_above_pred_{node.index}"
        lower_stage = f"%clip_lower_stage_{node.index}"
        lines.extend(
            [
                f"  {below_predicate} = fcmp olt double {source_value}, {lower_value}",
                f"  {above_predicate} = fcmp ogt double {source_value}, {upper_value}",
                (
                    f"  {lower_stage} = select i1 {below_predicate}, "
                    f"double {lower_value}, double {source_value}"
                ),
                (
                    f"  {value_name} = select i1 {above_predicate}, "
                    f"double {upper_value}, double {lower_stage}"
                ),
            ]
        )
        for derivative_index in range(parameter_count):
            source_derivative = _whole_program_native_derivative_operand(
                inputs[0], derivative_index
            )
            lower_derivative = _whole_program_native_derivative_operand(
                inputs[1], derivative_index
            )
            upper_derivative = _whole_program_native_derivative_operand(
                inputs[2], derivative_index
            )
            derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
            derivative_lower_stage = f"%d{node.index}_{derivative_index}_clip_lower_stage"
            lines.extend(
                [
                    (
                        f"  {derivative_lower_stage} = select i1 {below_predicate}, "
                        f"double {lower_derivative}, double {source_derivative}"
                    ),
                    (
                        f"  {derivative_name} = select i1 {above_predicate}, "
                        f"double {upper_derivative}, double {derivative_lower_stage}"
                    ),
                ]
            )
        return
    if len(inputs) != 2:
        raise ValueError(f"native operation {node.op} expects two inputs")
    left = _whole_program_native_operand(inputs[0])
    right = _whole_program_native_operand(inputs[1])
    if node.op == "add":
        lines.append(f"  {value_name} = fadd double {left}, {right}")
    elif node.op in {"sub", "subtract"}:
        lines.append(f"  {value_name} = fsub double {left}, {right}")
    elif node.op in {"mul", "multiply"}:
        lines.append(f"  {value_name} = fmul double {left}, {right}")
    elif node.op in {"div", "divide", "truediv"}:
        lines.append(f"  {value_name} = fdiv double {left}, {right}")
    elif node.op == "maximum":
        predicate = f"%select_pred_{node.index}"
        lines.append(f"  {predicate} = fcmp ogt double {left}, {right}")
        lines.append(f"  {value_name} = select i1 {predicate}, double {left}, double {right}")
    elif node.op == "minimum":
        predicate = f"%select_pred_{node.index}"
        lines.append(f"  {predicate} = fcmp olt double {left}, {right}")
        lines.append(f"  {value_name} = select i1 {predicate}, double {left}, double {right}")
    else:
        exponent = _whole_program_native_constant(inputs[1])
        if exponent is None:
            raise ValueError("native whole-program AD pow lowering requires constant exponent")
        lines.append(f"  {value_name} = call double @llvm.pow.f64(double {left}, double {right})")
        pow_factor = f"%pow_factor_{node.index}"
        exponent_minus_one = _fmt_llvm_float(exponent - 1.0)
        lines.append(
            f"  {pow_factor} = call double @llvm.pow.f64(double {left}, double {exponent_minus_one})"
        )
        scaled_factor = f"%pow_scaled_factor_{node.index}"
        lines.append(f"  {scaled_factor} = fmul double {_fmt_llvm_float(exponent)}, {pow_factor}")
    for derivative_index in range(parameter_count):
        left_derivative = _whole_program_native_derivative_operand(inputs[0], derivative_index)
        right_derivative = _whole_program_native_derivative_operand(inputs[1], derivative_index)
        derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
        if node.op == "add":
            lines.append(
                f"  {derivative_name} = fadd double {left_derivative}, {right_derivative}"
            )
        elif node.op in {"sub", "subtract"}:
            lines.append(
                f"  {derivative_name} = fsub double {left_derivative}, {right_derivative}"
            )
        elif node.op in {"mul", "multiply"}:
            left_term = f"%d{node.index}_{derivative_index}_left_mul"
            right_term = f"%d{node.index}_{derivative_index}_right_mul"
            lines.extend(
                [
                    f"  {left_term} = fmul double {left_derivative}, {right}",
                    f"  {right_term} = fmul double {left}, {right_derivative}",
                    f"  {derivative_name} = fadd double {left_term}, {right_term}",
                ]
            )
        elif node.op in {"div", "divide", "truediv"}:
            left_term = f"%d{node.index}_{derivative_index}_left_div"
            right_term = f"%d{node.index}_{derivative_index}_right_div"
            numerator = f"%d{node.index}_{derivative_index}_div_num"
            denominator = f"%d{node.index}_{derivative_index}_div_den"
            lines.extend(
                [
                    f"  {left_term} = fmul double {left_derivative}, {right}",
                    f"  {right_term} = fmul double {left}, {right_derivative}",
                    f"  {numerator} = fsub double {left_term}, {right_term}",
                    f"  {denominator} = fmul double {right}, {right}",
                    f"  {derivative_name} = fdiv double {numerator}, {denominator}",
                ]
            )
        elif node.op in {"maximum", "minimum"}:
            predicate = f"%select_pred_{node.index}"
            lines.append(
                f"  {derivative_name} = select i1 {predicate}, "
                f"double {left_derivative}, double {right_derivative}"
            )
        else:
            lines.append(f"  {derivative_name} = fmul double {scaled_factor}, {left_derivative}")


def _emit_whole_program_native_det3(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    value_name: str,
    inputs: Sequence[str],
) -> None:
    """Emit native 3x3 determinant value and exact cofactor adjoint code."""

    parameter_count = int(result.gradient.size)
    a_value, b_value, c_value, d_value, e_value, f_value, g_value, h_value, i_value = (
        _whole_program_native_operand(token) for token in inputs
    )
    cofactor_specs = (
        ("00", e_value, i_value, f_value, h_value),
        ("01", f_value, g_value, d_value, i_value),
        ("02", d_value, h_value, e_value, g_value),
        ("10", c_value, h_value, b_value, i_value),
        ("11", a_value, i_value, c_value, g_value),
        ("12", b_value, g_value, a_value, h_value),
        ("20", b_value, f_value, c_value, e_value),
        ("21", c_value, d_value, a_value, f_value),
        ("22", a_value, e_value, b_value, d_value),
    )
    cofactors: list[str] = []
    for label, positive_left, positive_right, negative_left, negative_right in cofactor_specs:
        positive_product = f"%det3_cofactor_{label}_positive_{node.index}"
        negative_product = f"%det3_cofactor_{label}_negative_{node.index}"
        cofactor = f"%det3_cofactor_{label}_{node.index}"
        lines.extend(
            [
                f"  {positive_product} = fmul double {positive_left}, {positive_right}",
                f"  {negative_product} = fmul double {negative_left}, {negative_right}",
                f"  {cofactor} = fsub double {positive_product}, {negative_product}",
            ]
        )
        cofactors.append(cofactor)

    value_term_0 = f"%det3_value_term_{node.index}_0"
    value_term_1 = f"%det3_value_term_{node.index}_1"
    value_term_2 = f"%det3_value_term_{node.index}_2"
    value_sum = f"%det3_value_sum_{node.index}"
    lines.extend(
        [
            f"  {value_term_0} = fmul double {a_value}, {cofactors[0]}",
            f"  {value_term_1} = fmul double {b_value}, {cofactors[1]}",
            f"  {value_term_2} = fmul double {c_value}, {cofactors[2]}",
            f"  {value_sum} = fadd double {value_term_0}, {value_term_1}",
            f"  {value_name} = fadd double {value_sum}, {value_term_2}",
        ]
    )

    for derivative_index in range(parameter_count):
        gradient_terms: list[str] = []
        for entry_index, cofactor in enumerate(cofactors):
            entry_derivative = _whole_program_native_derivative_operand(
                inputs[entry_index],
                derivative_index,
            )
            gradient_term = f"%d{node.index}_{derivative_index}_det3_entry_{entry_index}"
            lines.append(f"  {gradient_term} = fmul double {entry_derivative}, {cofactor}")
            gradient_terms.append(gradient_term)

        derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
        accumulator = gradient_terms[0]
        for term_offset, gradient_term in enumerate(gradient_terms[1:], start=1):
            target = (
                derivative_name
                if term_offset == len(gradient_terms) - 1
                else f"%d{node.index}_{derivative_index}_det3_acc_{term_offset}"
            )
            lines.append(f"  {target} = fadd double {accumulator}, {gradient_term}")
            accumulator = target


def _emit_whole_program_native_sum(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    value_name: str,
    inputs: Sequence[str],
    prefix: str,
) -> None:
    """Emit native value and derivative code for a fixed input sum."""

    if not inputs:
        raise ValueError(f"native {prefix} sum requires at least one input")
    parameter_count = int(result.gradient.size)
    value_operands = tuple(_whole_program_native_operand(token) for token in inputs)
    if len(value_operands) == 1:
        identity_value = f"%{prefix}_{node.index}_0"
        lines.extend(
            [
                f"  {identity_value} = fadd double {value_operands[0]}, {_fmt_llvm_float(0.0)}",
                f"  {value_name} = fadd double {identity_value}, {_fmt_llvm_float(0.0)}",
            ]
        )
    else:
        accumulator = value_operands[0]
        for offset, operand in enumerate(value_operands[1:], start=1):
            target = (
                value_name
                if offset == len(value_operands) - 1
                else f"%{prefix}_{node.index}_{offset}"
            )
            lines.append(f"  {target} = fadd double {accumulator}, {operand}")
            accumulator = target

    for derivative_index in range(parameter_count):
        derivative_operands = tuple(
            _whole_program_native_derivative_operand(token, derivative_index) for token in inputs
        )
        derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
        if len(derivative_operands) == 1:
            identity_derivative = f"%d{node.index}_{derivative_index}_{prefix}_0"
            lines.append(
                f"  {identity_derivative} = fadd double {derivative_operands[0]}, "
                f"{_fmt_llvm_float(0.0)}"
            )
            lines.append(
                f"  {derivative_name} = fadd double {identity_derivative}, {_fmt_llvm_float(0.0)}"
            )
            continue
        accumulator = derivative_operands[0]
        for offset, operand in enumerate(derivative_operands[1:], start=1):
            target = (
                derivative_name
                if offset == len(derivative_operands) - 1
                else f"%d{node.index}_{derivative_index}_{prefix}_{offset}"
            )
            lines.append(f"  {target} = fadd double {accumulator}, {operand}")
            accumulator = target


def _emit_whole_program_native_det4(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    value_name: str,
    inputs: Sequence[str],
) -> None:
    """Emit native 4x4 determinant value and exact cofactor adjoint code."""

    _emit_whole_program_native_det_fixed(
        lines,
        result,
        node,
        value_name,
        inputs,
        size=4,
        prefix="det4",
    )


def _emit_whole_program_native_det5(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    value_name: str,
    inputs: Sequence[str],
) -> None:
    """Emit native 5x5 determinant value and exact cofactor adjoint code."""

    _emit_whole_program_native_det_fixed(
        lines,
        result,
        node,
        value_name,
        inputs,
        size=5,
        prefix="det5",
    )


def _emit_whole_program_native_det_loop_helper_call(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    value_name: str,
    inputs: Sequence[str],
    *,
    size: int,
    prefix: str,
) -> None:
    """Emit a compact native helper call plus exact chain-rule derivative code."""

    if size not in _WHOLE_PROGRAM_NATIVE_LOOP_HELPER_DET_SIZES:
        raise ValueError("native determinant helper call requested for unsupported size")
    parameter_count = int(result.gradient.size)
    total = size * size
    output_total = total + 1
    matrix_alloca = f"%{prefix}_helper_matrix_{node.index}"
    output_alloca = f"%{prefix}_helper_output_{node.index}"
    matrix_base = f"%{prefix}_helper_matrix_base_{node.index}"
    output_base = f"%{prefix}_helper_output_base_{node.index}"
    symbol = _whole_program_native_det_loop_helper_symbol(size)
    lines.extend(
        [
            f"  {matrix_alloca} = alloca [{total} x double]",
            f"  {output_alloca} = alloca [{output_total} x double]",
        ]
    )
    for input_index, token in enumerate(inputs):
        matrix_ptr = f"%{prefix}_helper_matrix_ptr_{node.index}_{input_index}"
        lines.extend(
            [
                f"  {matrix_ptr} = getelementptr [{total} x double], "
                f"[{total} x double]* {matrix_alloca}, i64 0, i64 {input_index}",
                f"  store double {_whole_program_native_operand(token)}, double* {matrix_ptr}",
            ]
        )
    lines.extend(
        [
            f"  {matrix_base} = getelementptr [{total} x double], "
            f"[{total} x double]* {matrix_alloca}, i64 0, i64 0",
            f"  {output_base} = getelementptr [{output_total} x double], "
            f"[{output_total} x double]* {output_alloca}, i64 0, i64 0",
            f"  call void @{symbol}(double* {matrix_base}, double* {output_base})",
            f"  %{prefix}_helper_value_ptr_{node.index} = getelementptr "
            f"[{output_total} x double], [{output_total} x double]* {output_alloca}, "
            "i64 0, i64 0",
            f"  {value_name} = load double, double* %{prefix}_helper_value_ptr_{node.index}",
        ]
    )
    partials: list[str] = []
    for input_index in range(total):
        partial_ptr = f"%{prefix}_helper_partial_ptr_{node.index}_{input_index}"
        partial_value = f"%{prefix}_helper_partial_{node.index}_{input_index}"
        lines.extend(
            [
                f"  {partial_ptr} = getelementptr [{output_total} x double], "
                f"[{output_total} x double]* {output_alloca}, i64 0, i64 {input_index + 1}",
                f"  {partial_value} = load double, double* {partial_ptr}",
            ]
        )
        partials.append(partial_value)
    for derivative_index in range(parameter_count):
        derivative_terms: list[str] = []
        for input_index, token in enumerate(inputs):
            input_derivative = _whole_program_native_derivative_operand(token, derivative_index)
            if input_derivative == _fmt_llvm_float(0.0):
                continue
            term = f"%d{node.index}_{derivative_index}_{prefix}_helper_term_{input_index}"
            lines.append(f"  {term} = fmul double {partials[input_index]}, {input_derivative}")
            derivative_terms.append(term)
        _emit_whole_program_native_sum_operands(
            lines,
            derivative_terms,
            _whole_program_native_derivative_name(node.index, derivative_index),
        )


def _emit_whole_program_native_inverse_helper(
    lines: list[str],
    matrix_tokens: Sequence[str],
    *,
    size: int,
    prefix: str,
    emission_state: _WholeProgramNativeEmissionState,
) -> _WholeProgramNativeInverseHelper:
    """Emit or reuse one shared adjugate inverse helper for a static matrix."""

    if size not in _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES:
        raise ValueError("native shared inverse helper requested for unsupported size")
    matrix_key = (size, tuple(matrix_tokens))
    cached = emission_state.inverse_helpers.get(matrix_key)
    if cached is not None:
        return cached

    helper_index = len(emission_state.inverse_helpers)
    helper_prefix = f"{prefix}_shared_{helper_index}"
    total = size * size
    output_total = total + 1
    matrix_alloca = f"%{helper_prefix}_matrix"
    output_alloca = f"%{helper_prefix}_output"
    matrix_base = f"%{helper_prefix}_matrix_base"
    output_base = f"%{helper_prefix}_output_base"
    determinant = f"%{helper_prefix}_det"
    symbol = _whole_program_native_det_loop_helper_symbol(size)

    lines.extend(
        [
            f"  {matrix_alloca} = alloca [{total} x double]",
            f"  {output_alloca} = alloca [{output_total} x double]",
        ]
    )
    for input_index, token in enumerate(matrix_tokens):
        matrix_ptr = f"%{helper_prefix}_matrix_ptr_{input_index}"
        lines.extend(
            [
                f"  {matrix_ptr} = getelementptr [{total} x double], "
                f"[{total} x double]* {matrix_alloca}, i64 0, i64 {input_index}",
                f"  store double {_whole_program_native_operand(token)}, double* {matrix_ptr}",
            ]
        )
    lines.extend(
        [
            f"  {matrix_base} = getelementptr [{total} x double], "
            f"[{total} x double]* {matrix_alloca}, i64 0, i64 0",
            f"  {output_base} = getelementptr [{output_total} x double], "
            f"[{output_total} x double]* {output_alloca}, i64 0, i64 0",
            f"  call void @{symbol}(double* {matrix_base}, double* {output_base})",
            f"  %{helper_prefix}_det_ptr = getelementptr [{output_total} x double], "
            f"[{output_total} x double]* {output_alloca}, i64 0, i64 0",
            f"  {determinant} = load double, double* %{helper_prefix}_det_ptr",
        ]
    )

    partials: list[list[str]] = []
    for row in range(size):
        partial_row: list[str] = []
        for col in range(size):
            entry_index = row * size + col
            partial_ptr = f"%{helper_prefix}_partial_ptr_{entry_index}"
            partial_value = f"%{helper_prefix}_partial_{entry_index}"
            lines.extend(
                [
                    f"  {partial_ptr} = getelementptr [{output_total} x double], "
                    f"[{output_total} x double]* {output_alloca}, i64 0, i64 {entry_index + 1}",
                    f"  {partial_value} = load double, double* {partial_ptr}",
                ]
            )
            partial_row.append(partial_value)
        partials.append(partial_row)

    inverse_entries: list[list[str]] = []
    for row in range(size):
        inverse_row: list[str] = []
        for col in range(size):
            inverse_entry = f"%{helper_prefix}_inverse_{row}_{col}"
            lines.append(f"  {inverse_entry} = fdiv double {partials[col][row]}, {determinant}")
            inverse_row.append(inverse_entry)
        inverse_entries.append(inverse_row)

    helper = _WholeProgramNativeInverseHelper(
        determinant=determinant,
        inverse_entries=tuple(tuple(row) for row in inverse_entries),
    )
    emission_state.inverse_helpers[matrix_key] = helper
    return helper


def _emit_whole_program_native_solve_helper(
    lines: list[str],
    matrix_tokens: Sequence[str],
    rhs_tokens: Sequence[str],
    *,
    size: int,
    prefix: str,
    emission_state: _WholeProgramNativeEmissionState,
) -> _WholeProgramNativeSolveHelper:
    """Emit or reuse one shared solve helper for a static matrix/RHS column."""

    if len(rhs_tokens) != size:
        raise ValueError("native shared solve helper requires one RHS column")
    solve_key = (size, tuple(matrix_tokens), tuple(rhs_tokens))
    cached = emission_state.solve_helpers.get(solve_key)
    if cached is not None:
        return cached

    helper_index = len(emission_state.solve_helpers)
    helper_prefix = f"{prefix}_shared_{helper_index}"
    inverse_helper = _emit_whole_program_native_inverse_helper(
        lines,
        matrix_tokens,
        size=size,
        prefix=f"{helper_prefix}_inverse",
        emission_state=emission_state,
    )
    rhs_values = tuple(_whole_program_native_operand(token) for token in rhs_tokens)
    solution_entries: list[str] = []
    for row in range(size):
        row_terms: list[str] = []
        for col in range(size):
            term = f"%{helper_prefix}_solution_term_{row}_{col}"
            lines.append(
                f"  {term} = fmul double {inverse_helper.inverse_entries[row][col]}, "
                f"{rhs_values[col]}"
            )
            row_terms.append(term)
        solution_entry = f"%{helper_prefix}_solution_{row}"
        _emit_whole_program_native_sum_operands(lines, row_terms, solution_entry)
        solution_entries.append(solution_entry)

    helper = _WholeProgramNativeSolveHelper(solution_entries=tuple(solution_entries))
    emission_state.solve_helpers[solve_key] = helper
    return helper


def _emit_whole_program_native_inverse_fixed(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    value_name: str,
    inputs: Sequence[str],
    *,
    size: int,
    output_row: int,
    output_col: int,
    prefix: str,
    emission_state: _WholeProgramNativeEmissionState,
) -> None:
    """Emit native fixed-size inverse entry value and exact derivative code."""

    if size not in _WHOLE_PROGRAM_NATIVE_INVERSE_SIZES:
        raise ValueError("native fixed inverse requested for unsupported size")
    if output_row < 0 or output_row >= size or output_col < 0 or output_col >= size:
        raise ValueError("native fixed inverse output index is outside the matrix")
    parameter_count = int(result.gradient.size)
    matrix_tokens = tuple(inputs[: size * size])
    if size in _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES:
        helper = _emit_whole_program_native_inverse_helper(
            lines,
            matrix_tokens,
            size=size,
            prefix=prefix,
            emission_state=emission_state,
        )
        lines.append(
            f"  {value_name} = fadd double "
            f"{helper.inverse_entries[output_row][output_col]}, {_fmt_llvm_float(0.0)}"
        )
        zero = _fmt_llvm_float(0.0)
        for derivative_index in range(parameter_count):
            derivative_terms: list[str] = []
            for row in range(size):
                for col in range(size):
                    entry_derivative = _whole_program_native_derivative_operand(
                        matrix_tokens[row * size + col],
                        derivative_index,
                    )
                    if entry_derivative == zero:
                        continue
                    left_term = f"%d{node.index}_{derivative_index}_{prefix}_left_{row}_{col}"
                    product_term = f"%d{node.index}_{derivative_index}_{prefix}_prod_{row}_{col}"
                    lines.extend(
                        [
                            f"  {left_term} = fmul double "
                            f"{helper.inverse_entries[output_row][row]}, {entry_derivative}",
                            f"  {product_term} = fmul double {left_term}, "
                            f"{helper.inverse_entries[col][output_col]}",
                        ]
                    )
                    derivative_terms.append(product_term)
            derivative_sum = f"%d{node.index}_{derivative_index}_{prefix}_sum"
            _emit_whole_program_native_sum_operands(lines, derivative_terms, derivative_sum)
            derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
            lines.append(f"  {derivative_name} = fsub double {zero}, {derivative_sum}")
        return

    matrix = tuple(
        tuple(
            _whole_program_native_operand(matrix_tokens[row * size + col]) for col in range(size)
        )
        for row in range(size)
    )
    numerator_matrix = tuple(
        tuple(
            _fmt_llvm_float(1.0 if row == output_col else 0.0)
            if col == output_row
            else matrix[row][col]
            for col in range(size)
        )
        for row in range(size)
    )
    matrix_derivatives = tuple(
        tuple(
            tuple(
                _whole_program_native_derivative_operand(
                    matrix_tokens[row * size + col],
                    derivative_index,
                )
                for col in range(size)
            )
            for row in range(size)
        )
        for derivative_index in range(parameter_count)
    )
    numerator_derivatives = tuple(
        tuple(
            tuple(
                _fmt_llvm_float(0.0)
                if col == output_row
                else matrix_derivatives[derivative_index][row][col]
                for col in range(size)
            )
            for row in range(size)
        )
        for derivative_index in range(parameter_count)
    )
    denominator, denominator_derivatives = _emit_whole_program_native_det_with_derivatives(
        lines,
        matrix,
        matrix_derivatives,
        prefix=f"{prefix}_{node.index}_den",
    )
    numerator, numerator_derivative_values = _emit_whole_program_native_det_with_derivatives(
        lines,
        numerator_matrix,
        numerator_derivatives,
        prefix=f"{prefix}_{node.index}_{output_row}_{output_col}_num",
    )
    denominator_squared = f"%{prefix}_{node.index}_den_sq"
    lines.extend(
        [
            f"  {value_name} = fdiv double {numerator}, {denominator}",
            f"  {denominator_squared} = fmul double {denominator}, {denominator}",
        ]
    )
    for derivative_index in range(parameter_count):
        numerator_term = f"%d{node.index}_{derivative_index}_{prefix}_quot_num"
        denominator_term = f"%d{node.index}_{derivative_index}_{prefix}_quot_den"
        quotient_numerator = f"%d{node.index}_{derivative_index}_{prefix}_quotient_num"
        derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
        lines.extend(
            [
                f"  {numerator_term} = fmul double "
                f"{numerator_derivative_values[derivative_index]}, {denominator}",
                f"  {denominator_term} = fmul double {numerator}, "
                f"{denominator_derivatives[derivative_index]}",
                f"  {quotient_numerator} = fsub double {numerator_term}, {denominator_term}",
                f"  {derivative_name} = fdiv double {quotient_numerator}, {denominator_squared}",
            ]
        )


def _emit_whole_program_native_solve_fixed(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    value_name: str,
    inputs: Sequence[str],
    *,
    size: int,
    output_row: int,
    prefix: str,
    emission_state: _WholeProgramNativeEmissionState,
) -> None:
    """Emit native fixed-size solve-column value and exact derivative code."""

    if size not in (
        _WHOLE_PROGRAM_NATIVE_SOLVE_VECTOR_SIZES | _WHOLE_PROGRAM_NATIVE_SOLVE_MATRIX_SIZES
    ):
        raise ValueError("native fixed solve requested for unsupported size")
    if output_row < 0 or output_row >= size:
        raise ValueError("native fixed solve output row is outside the solution")
    parameter_count = int(result.gradient.size)
    matrix_tokens = tuple(inputs[: size * size])
    rhs_tokens = tuple(inputs[size * size :])
    if size in _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES:
        inverse_helper = _emit_whole_program_native_inverse_helper(
            lines,
            matrix_tokens,
            size=size,
            prefix=f"{prefix}_inverse",
            emission_state=emission_state,
        )
        solve_helper = _emit_whole_program_native_solve_helper(
            lines,
            matrix_tokens,
            rhs_tokens,
            size=size,
            prefix=prefix,
            emission_state=emission_state,
        )
        lines.append(
            f"  {value_name} = fadd double "
            f"{solve_helper.solution_entries[output_row]}, {_fmt_llvm_float(0.0)}"
        )
        zero = _fmt_llvm_float(0.0)
        for derivative_index in range(parameter_count):
            derivative_terms: list[str] = []
            for row in range(size):
                residual_terms: list[str] = []
                for col in range(size):
                    matrix_derivative = _whole_program_native_derivative_operand(
                        matrix_tokens[row * size + col],
                        derivative_index,
                    )
                    if matrix_derivative == zero:
                        continue
                    residual_term = f"%d{node.index}_{derivative_index}_{prefix}_res_{row}_{col}"
                    lines.append(
                        f"  {residual_term} = fmul double {matrix_derivative}, "
                        f"{solve_helper.solution_entries[col]}"
                    )
                    residual_terms.append(residual_term)
                residual_matrix_sum = f"%d{node.index}_{derivative_index}_{prefix}_res_sum_{row}"
                _emit_whole_program_native_sum_operands(
                    lines,
                    residual_terms,
                    residual_matrix_sum,
                )
                rhs_derivative = _whole_program_native_derivative_operand(
                    rhs_tokens[row],
                    derivative_index,
                )
                residual = f"%d{node.index}_{derivative_index}_{prefix}_rhs_minus_ax_{row}"
                lines.append(f"  {residual} = fsub double {rhs_derivative}, {residual_matrix_sum}")
                derivative_term = f"%d{node.index}_{derivative_index}_{prefix}_term_{row}"
                lines.append(
                    f"  {derivative_term} = fmul double "
                    f"{inverse_helper.inverse_entries[output_row][row]}, {residual}"
                )
                derivative_terms.append(derivative_term)
            _emit_whole_program_native_sum_operands(
                lines,
                derivative_terms,
                _whole_program_native_derivative_name(node.index, derivative_index),
            )
        return

    matrix = tuple(
        tuple(
            _whole_program_native_operand(matrix_tokens[row * size + col]) for col in range(size)
        )
        for row in range(size)
    )
    rhs = tuple(_whole_program_native_operand(token) for token in rhs_tokens)
    numerator_matrix = tuple(
        tuple(rhs[row] if col == output_row else matrix[row][col] for col in range(size))
        for row in range(size)
    )
    matrix_derivatives = tuple(
        tuple(
            tuple(
                _whole_program_native_derivative_operand(
                    matrix_tokens[row * size + col],
                    derivative_index,
                )
                for col in range(size)
            )
            for row in range(size)
        )
        for derivative_index in range(parameter_count)
    )
    numerator_derivatives = tuple(
        tuple(
            tuple(
                _whole_program_native_derivative_operand(rhs_tokens[row], derivative_index)
                if col == output_row
                else matrix_derivatives[derivative_index][row][col]
                for col in range(size)
            )
            for row in range(size)
        )
        for derivative_index in range(parameter_count)
    )
    denominator, denominator_derivatives = _emit_whole_program_native_det_with_derivatives(
        lines,
        matrix,
        matrix_derivatives,
        prefix=f"{prefix}_{node.index}_den",
    )
    numerator, numerator_derivative_values = _emit_whole_program_native_det_with_derivatives(
        lines,
        numerator_matrix,
        numerator_derivatives,
        prefix=f"{prefix}_{node.index}_row{output_row}_num",
    )
    denominator_squared = f"%{prefix}_{node.index}_den_sq"
    lines.extend(
        [
            f"  {value_name} = fdiv double {numerator}, {denominator}",
            f"  {denominator_squared} = fmul double {denominator}, {denominator}",
        ]
    )
    for derivative_index in range(parameter_count):
        numerator_term = f"%d{node.index}_{derivative_index}_{prefix}_quot_num"
        denominator_term = f"%d{node.index}_{derivative_index}_{prefix}_quot_den"
        quotient_numerator = f"%d{node.index}_{derivative_index}_{prefix}_quotient_num"
        derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
        lines.extend(
            [
                f"  {numerator_term} = fmul double "
                f"{numerator_derivative_values[derivative_index]}, {denominator}",
                f"  {denominator_term} = fmul double {numerator}, "
                f"{denominator_derivatives[derivative_index]}",
                f"  {quotient_numerator} = fsub double {numerator_term}, {denominator_term}",
                f"  {derivative_name} = fdiv double {quotient_numerator}, {denominator_squared}",
            ]
        )


def _emit_whole_program_native_det_with_derivatives(
    lines: list[str],
    matrix: tuple[tuple[str, ...], ...],
    derivative_matrices: Sequence[tuple[tuple[str, ...], ...]],
    *,
    prefix: str,
) -> tuple[str, tuple[str, ...]]:
    """Emit a fixed determinant plus exact first derivatives for each matrix tangent."""

    size = len(matrix)
    if size < 1 or any(len(row) != size for row in matrix):
        raise ValueError("native determinant derivative helper requires a square matrix")
    if size in _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES:
        return _emit_whole_program_native_det_helper_with_derivatives(
            lines,
            matrix,
            derivative_matrices,
            prefix=prefix,
        )
    determinant = _emit_whole_program_native_det_expression(lines, matrix, f"{prefix}_det")
    derivatives: list[str] = []
    zero = _fmt_llvm_float(0.0)
    for derivative_index, derivative_matrix in enumerate(derivative_matrices):
        if len(derivative_matrix) != size or any(len(row) != size for row in derivative_matrix):
            raise ValueError("native determinant derivative matrix shape mismatch")
        terms: list[str] = []
        for row in range(size):
            for col in range(size):
                entry_derivative = derivative_matrix[row][col]
                if entry_derivative == zero:
                    continue
                minor = _whole_program_native_det_minor(matrix, row, col)
                cofactor = _emit_whole_program_native_det_expression(
                    lines,
                    minor,
                    f"{prefix}_d{derivative_index}_minor_{row}_{col}",
                )
                if (row + col) % 2:
                    signed_cofactor = f"%{prefix}_d{derivative_index}_cofactor_{row}_{col}"
                    lines.append(f"  {signed_cofactor} = fsub double {zero}, {cofactor}")
                    cofactor = signed_cofactor
                term = f"%{prefix}_d{derivative_index}_term_{row}_{col}"
                lines.append(f"  {term} = fmul double {cofactor}, {entry_derivative}")
                terms.append(term)
        derivative_name = f"%{prefix}_d{derivative_index}"
        _emit_whole_program_native_sum_operands(lines, terms, derivative_name)
        derivatives.append(derivative_name)
    return determinant, tuple(derivatives)


def _emit_whole_program_native_det_helper_with_derivatives(
    lines: list[str],
    matrix: tuple[tuple[str, ...], ...],
    derivative_matrices: Sequence[tuple[tuple[str, ...], ...]],
    *,
    prefix: str,
) -> tuple[str, tuple[str, ...]]:
    """Emit compact determinant helper call plus exact first derivatives."""

    size = len(matrix)
    if size not in _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES:
        raise ValueError("native determinant derivative helper requested for unsupported size")
    if any(len(row) != size for row in matrix):
        raise ValueError("native determinant derivative helper requires a square matrix")
    total = size * size
    output_total = total + 1
    matrix_alloca = f"%{prefix}_helper_matrix"
    output_alloca = f"%{prefix}_helper_output"
    matrix_base = f"%{prefix}_helper_matrix_base"
    output_base = f"%{prefix}_helper_output_base"
    determinant = f"%{prefix}_helper_det"
    symbol = _whole_program_native_det_loop_helper_symbol(size)
    lines.extend(
        [
            f"  {matrix_alloca} = alloca [{total} x double]",
            f"  {output_alloca} = alloca [{output_total} x double]",
        ]
    )
    for row in range(size):
        for col in range(size):
            entry_index = row * size + col
            matrix_ptr = f"%{prefix}_helper_matrix_ptr_{entry_index}"
            lines.extend(
                [
                    f"  {matrix_ptr} = getelementptr [{total} x double], "
                    f"[{total} x double]* {matrix_alloca}, i64 0, i64 {entry_index}",
                    f"  store double {matrix[row][col]}, double* {matrix_ptr}",
                ]
            )
    lines.extend(
        [
            f"  {matrix_base} = getelementptr [{total} x double], "
            f"[{total} x double]* {matrix_alloca}, i64 0, i64 0",
            f"  {output_base} = getelementptr [{output_total} x double], "
            f"[{output_total} x double]* {output_alloca}, i64 0, i64 0",
            f"  call void @{symbol}(double* {matrix_base}, double* {output_base})",
            f"  %{prefix}_helper_det_ptr = getelementptr [{output_total} x double], "
            f"[{output_total} x double]* {output_alloca}, i64 0, i64 0",
            f"  {determinant} = load double, double* %{prefix}_helper_det_ptr",
        ]
    )
    partials: list[str] = []
    for entry_index in range(total):
        partial_ptr = f"%{prefix}_helper_partial_ptr_{entry_index}"
        partial_value = f"%{prefix}_helper_partial_{entry_index}"
        lines.extend(
            [
                f"  {partial_ptr} = getelementptr [{output_total} x double], "
                f"[{output_total} x double]* {output_alloca}, i64 0, i64 {entry_index + 1}",
                f"  {partial_value} = load double, double* {partial_ptr}",
            ]
        )
        partials.append(partial_value)

    zero = _fmt_llvm_float(0.0)
    derivatives: list[str] = []
    for derivative_index, derivative_matrix in enumerate(derivative_matrices):
        if len(derivative_matrix) != size or any(len(row) != size for row in derivative_matrix):
            raise ValueError("native determinant derivative matrix shape mismatch")
        derivative_terms: list[str] = []
        for row in range(size):
            for col in range(size):
                entry_index = row * size + col
                entry_derivative = derivative_matrix[row][col]
                if entry_derivative == zero:
                    continue
                term = f"%{prefix}_helper_d{derivative_index}_term_{entry_index}"
                lines.append(f"  {term} = fmul double {partials[entry_index]}, {entry_derivative}")
                derivative_terms.append(term)
        derivative_name = f"%{prefix}_d{derivative_index}"
        _emit_whole_program_native_sum_operands(lines, derivative_terms, derivative_name)
        derivatives.append(derivative_name)
    return determinant, tuple(derivatives)


def _emit_whole_program_native_det_faddeev_leverrier(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    value_name: str,
    inputs: Sequence[str],
    *,
    size: int,
    prefix: str,
) -> None:
    """Emit native determinant and exact derivative code using Faddeev-LeVerrier."""

    parameter_count = int(result.gradient.size)
    matrix = tuple(
        tuple(_whole_program_native_operand(inputs[row * size + col]) for col in range(size))
        for row in range(size)
    )
    matrix_derivatives = tuple(
        tuple(
            tuple(
                _whole_program_native_derivative_operand(
                    inputs[row * size + col],
                    derivative_index,
                )
                for col in range(size)
            )
            for row in range(size)
        )
        for derivative_index in range(parameter_count)
    )
    identity = tuple(
        tuple(_fmt_llvm_float(1.0 if row == col else 0.0) for col in range(size))
        for row in range(size)
    )
    zero_matrix = tuple(tuple(_fmt_llvm_float(0.0) for _ in range(size)) for _ in range(size))
    b_matrix = identity
    b_derivatives = tuple(zero_matrix for _ in range(parameter_count))
    coefficient = _fmt_llvm_float(0.0)
    coefficient_derivatives = tuple(_fmt_llvm_float(0.0) for _ in range(parameter_count))

    for step in range(1, size + 1):
        product_matrix: list[list[str]] = []
        product_derivatives: list[list[list[str]]] = [[] for _ in range(parameter_count)]
        for row in range(size):
            product_row: list[str] = []
            derivative_rows: list[list[str]] = [[] for _ in range(parameter_count)]
            for col in range(size):
                product_terms: list[str] = []
                for inner in range(size):
                    product_term = f"%{prefix}_fl_s{step}_{row}_{col}_{inner}"
                    lines.append(
                        f"  {product_term} = fmul double "
                        f"{matrix[row][inner]}, {b_matrix[inner][col]}"
                    )
                    product_terms.append(product_term)
                product_value = f"%{prefix}_fl_s{step}_{row}_{col}"
                _emit_whole_program_native_sum_operands(
                    lines,
                    product_terms,
                    product_value,
                )
                product_row.append(product_value)

                for derivative_index in range(parameter_count):
                    derivative_terms: list[str] = []
                    for inner in range(size):
                        left_term = (
                            f"%d{node.index}_{derivative_index}_{prefix}_fl_s"
                            f"{step}_{row}_{col}_{inner}_left"
                        )
                        right_term = (
                            f"%d{node.index}_{derivative_index}_{prefix}_fl_s"
                            f"{step}_{row}_{col}_{inner}_right"
                        )
                        lines.extend(
                            [
                                f"  {left_term} = fmul double "
                                f"{matrix_derivatives[derivative_index][row][inner]}, "
                                f"{b_matrix[inner][col]}",
                                f"  {right_term} = fmul double {matrix[row][inner]}, "
                                f"{b_derivatives[derivative_index][inner][col]}",
                            ]
                        )
                        derivative_terms.extend((left_term, right_term))
                    product_derivative = (
                        f"%d{node.index}_{derivative_index}_{prefix}_fl_s{step}_{row}_{col}"
                    )
                    _emit_whole_program_native_sum_operands(
                        lines,
                        derivative_terms,
                        product_derivative,
                    )
                    derivative_rows[derivative_index].append(product_derivative)
            product_matrix.append(product_row)
            for derivative_index, derivative_row in enumerate(derivative_rows):
                product_derivatives[derivative_index].append(derivative_row)

        trace_terms = [product_matrix[index][index] for index in range(size)]
        trace_value = f"%{prefix}_fl_s{step}_trace"
        _emit_whole_program_native_sum_operands(lines, trace_terms, trace_value)
        coefficient = f"%{prefix}_fl_s{step}_coeff"
        lines.append(
            f"  {coefficient} = fmul double {_fmt_llvm_float(-1.0 / step)}, {trace_value}"
        )

        next_coefficient_derivatives: list[str] = []
        for derivative_index in range(parameter_count):
            derivative_trace_terms = [
                product_derivatives[derivative_index][index][index] for index in range(size)
            ]
            derivative_trace = f"%d{node.index}_{derivative_index}_{prefix}_fl_s{step}_trace"
            _emit_whole_program_native_sum_operands(
                lines,
                derivative_trace_terms,
                derivative_trace,
            )
            derivative_coefficient = f"%d{node.index}_{derivative_index}_{prefix}_fl_s{step}_coeff"
            lines.append(
                f"  {derivative_coefficient} = fmul double "
                f"{_fmt_llvm_float(-1.0 / step)}, {derivative_trace}"
            )
            next_coefficient_derivatives.append(derivative_coefficient)
        coefficient_derivatives = tuple(next_coefficient_derivatives)

        next_b_matrix: list[list[str]] = []
        next_b_derivatives: list[list[list[str]]] = [[] for _ in range(parameter_count)]
        for row in range(size):
            next_b_row: list[str] = []
            next_derivative_rows: list[list[str]] = [[] for _ in range(parameter_count)]
            for col in range(size):
                if row == col:
                    next_b_value = f"%{prefix}_fl_s{step}_b_{row}_{col}"
                    lines.append(
                        f"  {next_b_value} = fadd double {product_matrix[row][col]}, {coefficient}"
                    )
                    for derivative_index in range(parameter_count):
                        next_derivative = (
                            f"%d{node.index}_{derivative_index}_{prefix}_fl_s{step}_b_{row}_{col}"
                        )
                        lines.append(
                            f"  {next_derivative} = fadd double "
                            f"{product_derivatives[derivative_index][row][col]}, "
                            f"{coefficient_derivatives[derivative_index]}"
                        )
                        next_derivative_rows[derivative_index].append(next_derivative)
                else:
                    next_b_value = product_matrix[row][col]
                    for derivative_index in range(parameter_count):
                        next_derivative_rows[derivative_index].append(
                            product_derivatives[derivative_index][row][col]
                        )
                next_b_row.append(next_b_value)
            next_b_matrix.append(next_b_row)
            for derivative_index, derivative_row in enumerate(next_derivative_rows):
                next_b_derivatives[derivative_index].append(derivative_row)
        b_matrix = tuple(tuple(row) for row in next_b_matrix)
        b_derivatives = tuple(
            tuple(tuple(row) for row in derivative_matrix)
            for derivative_matrix in next_b_derivatives
        )

    if size % 2 == 0:
        lines.append(f"  {value_name} = fadd double {coefficient}, {_fmt_llvm_float(0.0)}")
        for derivative_index, derivative_coefficient in enumerate(coefficient_derivatives):
            derivative_name = _whole_program_native_derivative_name(
                node.index,
                derivative_index,
            )
            lines.append(
                f"  {derivative_name} = fadd double "
                f"{derivative_coefficient}, {_fmt_llvm_float(0.0)}"
            )
    else:
        lines.append(f"  {value_name} = fsub double {_fmt_llvm_float(0.0)}, {coefficient}")
        for derivative_index, derivative_coefficient in enumerate(coefficient_derivatives):
            derivative_name = _whole_program_native_derivative_name(
                node.index,
                derivative_index,
            )
            lines.append(
                f"  {derivative_name} = fsub double "
                f"{_fmt_llvm_float(0.0)}, {derivative_coefficient}"
            )


def _emit_whole_program_native_det_fixed(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    value_name: str,
    inputs: Sequence[str],
    *,
    size: int,
    prefix: str,
) -> None:
    """Emit native fixed-size determinant value and exact cofactor adjoint code."""

    parameter_count = int(result.gradient.size)
    entries = tuple(_whole_program_native_operand(token) for token in inputs)
    matrix = tuple(tuple(entries[row * size + col] for col in range(size)) for row in range(size))
    determinant = _emit_whole_program_native_det_expression(
        lines,
        matrix,
        f"{prefix}_value_{node.index}",
    )
    lines.append(f"  {value_name} = fadd double {determinant}, {_fmt_llvm_float(0.0)}")

    cofactors: list[str] = []
    for row in range(size):
        for col in range(size):
            minor = _whole_program_native_det_minor(matrix, row, col)
            cofactor = _emit_whole_program_native_det_expression(
                lines,
                minor,
                f"{prefix}_cofactor_{row}{col}_{node.index}",
            )
            if (row + col) % 2:
                signed_cofactor = f"%{prefix}_cofactor_{row}{col}_signed_{node.index}"
                lines.append(
                    f"  {signed_cofactor} = fsub double {_fmt_llvm_float(0.0)}, {cofactor}"
                )
                cofactors.append(signed_cofactor)
            else:
                cofactors.append(cofactor)

    for derivative_index in range(parameter_count):
        gradient_terms: list[str] = []
        for entry_index, cofactor in enumerate(cofactors):
            entry_derivative = _whole_program_native_derivative_operand(
                inputs[entry_index],
                derivative_index,
            )
            gradient_term = f"%d{node.index}_{derivative_index}_{prefix}_entry_{entry_index}"
            lines.append(f"  {gradient_term} = fmul double {entry_derivative}, {cofactor}")
            gradient_terms.append(gradient_term)

        derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
        accumulator = gradient_terms[0]
        for term_offset, gradient_term in enumerate(gradient_terms[1:], start=1):
            target = (
                derivative_name
                if term_offset == len(gradient_terms) - 1
                else f"%d{node.index}_{derivative_index}_{prefix}_acc_{term_offset}"
            )
            lines.append(f"  {target} = fadd double {accumulator}, {gradient_term}")
            accumulator = target


def _emit_whole_program_native_det_expression(
    lines: list[str],
    matrix: tuple[tuple[str, ...], ...],
    prefix: str,
) -> str:
    """Emit a small fixed determinant expression and return the final LLVM operand."""

    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    if size == 2:
        diagonal_product = f"%{prefix}_diag"
        off_diagonal_product = f"%{prefix}_offdiag"
        determinant = f"%{prefix}_det"
        lines.extend(
            [
                f"  {diagonal_product} = fmul double {matrix[0][0]}, {matrix[1][1]}",
                f"  {off_diagonal_product} = fmul double {matrix[0][1]}, {matrix[1][0]}",
                f"  {determinant} = fsub double {diagonal_product}, {off_diagonal_product}",
            ]
        )
        return determinant

    accumulator = ""
    for col, entry in enumerate(matrix[0]):
        minor = _whole_program_native_det_minor(matrix, 0, col)
        minor_det = _emit_whole_program_native_det_expression(
            lines,
            minor,
            f"{prefix}_minor_{col}",
        )
        term = f"%{prefix}_term_{col}"
        lines.append(f"  {term} = fmul double {entry}, {minor_det}")
        if col == 0:
            accumulator = term
            continue
        updated_accumulator = f"%{prefix}_acc_{col}"
        operation = "fadd" if col % 2 == 0 else "fsub"
        lines.append(f"  {updated_accumulator} = {operation} double {accumulator}, {term}")
        accumulator = updated_accumulator
    return accumulator


def _emit_whole_program_native_sum_operands(
    lines: list[str],
    operands: Sequence[str],
    result_name: str,
) -> None:
    """Emit a deterministic native sum into an existing LLVM result name."""

    if not operands:
        lines.append(
            f"  {result_name} = fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(0.0)}"
        )
        return
    if len(operands) == 1:
        lines.append(f"  {result_name} = fadd double {operands[0]}, {_fmt_llvm_float(0.0)}")
        return
    accumulator = operands[0]
    for offset, operand in enumerate(operands[1:], start=1):
        target = result_name if offset == len(operands) - 1 else f"{result_name}_sum_{offset}"
        lines.append(f"  {target} = fadd double {accumulator}, {operand}")
        accumulator = target


def _whole_program_native_det_minor(
    matrix: tuple[tuple[str, ...], ...],
    row_index: int,
    col_index: int,
) -> tuple[tuple[str, ...], ...]:
    """Return the fixed determinant minor matrix for native code generation."""

    return tuple(
        tuple(value for col, value in enumerate(row_values) if col != col_index)
        for current_row_index, row_values in enumerate(matrix)
        if current_row_index != row_index
    )


def _whole_program_native_wide_det_size(op: str) -> int | None:
    """Return supported helper-backed determinant sizes for compact det ops."""

    prefix = "linalg:det:"
    if not op.startswith(prefix):
        return None
    shape = op[len(prefix) :]
    try:
        row_text, col_text = shape.split("x", 1)
        rows = int(row_text)
        cols = int(col_text)
    except ValueError:
        return None
    if rows != cols or rows not in _WHOLE_PROGRAM_NATIVE_WIDE_DET_SIZES:
        return None
    return rows


def _whole_program_native_inverse_spec(op: str) -> tuple[int, int, int] | None:
    """Return supported static inverse shape and output coordinates."""

    parts = op.split(":")
    if len(parts) != 5 or parts[0] != "linalg" or parts[1] != "inv":
        return None
    try:
        row_text, col_text = parts[2].split("x", 1)
        rows = int(row_text)
        cols = int(col_text)
        output_row = int(parts[3])
        output_col = int(parts[4])
    except ValueError:
        return None
    if (
        rows != cols
        or rows not in _WHOLE_PROGRAM_NATIVE_INVERSE_SIZES
        or output_row < 0
        or output_row >= rows
        or output_col < 0
        or output_col >= rows
    ):
        return None
    return rows, output_row, output_col


def _whole_program_native_solve_vector_spec(op: str) -> tuple[int, int] | None:
    """Return supported static vector solve shape and output row."""

    parts = op.split(":")
    if len(parts) != 6 or parts[0] != "linalg" or parts[1] != "solve" or parts[3] != "rhs":
        return None
    try:
        row_text, col_text = parts[2].split("x", 1)
        rows = int(row_text)
        cols = int(col_text)
        rhs_rows = int(parts[4])
        output_row = int(parts[5])
    except ValueError:
        return None
    if (
        rows != cols
        or rows not in _WHOLE_PROGRAM_NATIVE_SOLVE_VECTOR_SIZES
        or rhs_rows != rows
        or output_row < 0
        or output_row >= rows
    ):
        return None
    return rows, output_row


def _whole_program_native_solve_matrix_spec(op: str) -> tuple[int, int, int, int] | None:
    """Return supported static matrix-RHS solve shape and output coordinates."""

    parts = op.split(":")
    if len(parts) != 7 or parts[0] != "linalg" or parts[1] != "solve" or parts[3] != "rhs":
        return None
    try:
        row_text, col_text = parts[2].split("x", 1)
        rows = int(row_text)
        cols = int(col_text)
        rhs_row_text, rhs_col_text = parts[4].split("x", 1)
        rhs_rows = int(rhs_row_text)
        rhs_cols = int(rhs_col_text)
        output_row = int(parts[5])
        output_col = int(parts[6])
    except ValueError:
        return None
    if (
        rows != cols
        or rows not in _WHOLE_PROGRAM_NATIVE_SOLVE_MATRIX_SIZES
        or rhs_rows != rows
        or rhs_cols < 1
        or rhs_cols > _WHOLE_PROGRAM_NATIVE_SOLVE_MATRIX_MAX_RHS_COLS
        or output_row < 0
        or output_row >= rows
        or output_col < 0
        or output_col >= rhs_cols
    ):
        return None
    return rows, rhs_cols, output_row, output_col


def _whole_program_native_det_derivative_helper_size(op: str) -> int | None:
    """Return determinant-partial helper size required by supported quotient linalg ops."""

    inverse_spec = _whole_program_native_inverse_spec(op)
    if (
        inverse_spec is not None
        and inverse_spec[0] in _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES
    ):
        return inverse_spec[0]
    solve_vector_spec = _whole_program_native_solve_vector_spec(op)
    if (
        solve_vector_spec is not None
        and solve_vector_spec[0] in _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES
    ):
        return solve_vector_spec[0]
    solve_matrix_spec = _whole_program_native_solve_matrix_spec(op)
    if (
        solve_matrix_spec is not None
        and solve_matrix_spec[0] in _WHOLE_PROGRAM_NATIVE_DET_DERIVATIVE_HELPER_SIZES
    ):
        return solve_matrix_spec[0]
    return None


def _whole_program_native_trace_input_count(op: str) -> int | None:
    """Return the number of statically captured diagonal inputs for a trace op."""

    prefix = "linalg:trace:"
    if not op.startswith(prefix):
        return None
    try:
        shape_part, offset_part = op[len(prefix) :].split(":offset:", 1)
        row_text, col_text = shape_part.split("x", 1)
        rows = int(row_text)
        cols = int(col_text)
        offset = int(offset_part)
    except ValueError:
        return None
    if rows <= 0 or cols <= 0:
        return None
    start_row = max(0, -offset)
    start_col = max(0, offset)
    count = min(rows - start_row, cols - start_col)
    if count <= 0:
        return None
    return count


def _whole_program_native_diag_input_count(op: str) -> int | None:
    """Return the scalar input count for compact diag/diagflat passthrough nodes."""

    parts = op.split(":")
    if len(parts) != 7 or parts[0] != "linalg":
        return None
    primitive = parts[1]
    if primitive not in {"diag", "diagflat"}:
        return None
    if parts[3] != "offset":
        return None
    try:
        dimensions = tuple(int(dimension) for dimension in parts[2].split("x"))
        int(parts[4])
        output_index = int(parts[6])
    except ValueError:
        return None
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        return None
    mode = parts[5]
    if primitive == "diag" and mode not in {"extract", "construct"}:
        return None
    if primitive == "diagflat" and mode != "construct":
        return None
    if output_index < 0:
        return None
    return 1


def _emit_whole_program_native_product_sum2(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
    value_name: str,
    inputs: tuple[str, str, str, str],
    prefix: str,
) -> None:
    """Emit native value and derivative code for x*y + z*w."""

    parameter_count = int(result.gradient.size)
    first_left_token, first_right_token, second_left_token, second_right_token = inputs
    first_left = _whole_program_native_operand(first_left_token)
    first_right = _whole_program_native_operand(first_right_token)
    second_left = _whole_program_native_operand(second_left_token)
    second_right = _whole_program_native_operand(second_right_token)
    first_product = f"%{prefix}_first_{node.index}"
    second_product = f"%{prefix}_second_{node.index}"
    lines.extend(
        [
            f"  {first_product} = fmul double {first_left}, {first_right}",
            f"  {second_product} = fmul double {second_left}, {second_right}",
            f"  {value_name} = fadd double {first_product}, {second_product}",
        ]
    )
    for derivative_index in range(parameter_count):
        first_left_derivative = _whole_program_native_derivative_operand(
            first_left_token, derivative_index
        )
        first_right_derivative = _whole_program_native_derivative_operand(
            first_right_token, derivative_index
        )
        second_left_derivative = _whole_program_native_derivative_operand(
            second_left_token, derivative_index
        )
        second_right_derivative = _whole_program_native_derivative_operand(
            second_right_token, derivative_index
        )
        first_left_term = f"%d{node.index}_{derivative_index}_{prefix}_first_left"
        first_right_term = f"%d{node.index}_{derivative_index}_{prefix}_first_right"
        first_derivative = f"%d{node.index}_{derivative_index}_{prefix}_first"
        second_left_term = f"%d{node.index}_{derivative_index}_{prefix}_second_left"
        second_right_term = f"%d{node.index}_{derivative_index}_{prefix}_second_right"
        second_derivative = f"%d{node.index}_{derivative_index}_{prefix}_second"
        derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
        lines.extend(
            [
                f"  {first_left_term} = fmul double {first_left_derivative}, {first_right}",
                f"  {first_right_term} = fmul double {first_left}, {first_right_derivative}",
                f"  {first_derivative} = fadd double {first_left_term}, {first_right_term}",
                f"  {second_left_term} = fmul double {second_left_derivative}, {second_right}",
                f"  {second_right_term} = fmul double {second_left}, {second_right_derivative}",
                f"  {second_derivative} = fadd double {second_left_term}, {second_right_term}",
                f"  {derivative_name} = fadd double {first_derivative}, {second_derivative}",
            ]
        )


def _emit_whole_program_native_where_predicate(
    lines: list[str],
    node_index: int,
    predicate: str,
) -> str:
    predicate_body = _whole_program_native_where_predicate_body(predicate)
    parts = predicate_body.split(":")
    if len(parts) != 3:
        raise ValueError("native whole-program AD where predicate is malformed")
    left_token, op, right_token = parts
    predicate_name = f"%where_pred_{node_index}"
    llvm_predicate = {
        "gt": "ogt",
        "ge": "oge",
        "lt": "olt",
        "le": "ole",
        "eq": "oeq",
        "ne": "one",
    }.get(op)
    if llvm_predicate is None:
        raise ValueError(f"native whole-program AD where predicate op {op} is unsupported")
    left = _whole_program_native_operand(left_token)
    right = _whole_program_native_operand(right_token)
    lines.append(f"  {predicate_name} = fcmp {llvm_predicate} double {left}, {right}")
    return predicate_name


def _whole_program_native_where_predicate_body(predicate: str) -> str:
    if ":truth:" not in predicate:
        raise ValueError("native whole-program AD where predicate requires recorded truth")
    body, truth = predicate.rsplit(":truth:", 1)
    if truth not in {"0", "1"}:
        raise ValueError("native whole-program AD where predicate truth must be 0 or 1")
    return body


def _whole_program_native_operand(token: str) -> str:
    if _whole_program_native_is_ir_value(token):
        return _whole_program_native_value_name(int(token[1:]))
    constant = _whole_program_native_constant(token)
    if constant is None:
        raise ValueError(f"native whole-program AD cannot lower operand {token}")
    return _fmt_llvm_float(constant)


def _whole_program_native_derivative_operand(token: str, derivative_index: int) -> str:
    if _whole_program_native_is_ir_value(token):
        return _whole_program_native_derivative_name(int(token[1:]), derivative_index)
    constant = _whole_program_native_constant(token)
    if constant is None:
        raise ValueError(f"native whole-program AD cannot lower derivative operand {token}")
    return _fmt_llvm_float(0.0)


def _whole_program_native_constant(token: str) -> float | None:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _whole_program_native_is_ir_value(token: str) -> bool:
    return token.startswith("%") and token[1:].isdigit()


def _whole_program_native_value_name(index: int) -> str:
    return f"%n{index}"


def _whole_program_native_derivative_name(node_index: int, derivative_index: int) -> str:
    return f"%d{node_index}_{derivative_index}"


def _fmt_llvm_float(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("LLVM numeric constants must be finite")
    return format(float(value), ".17e")


def _fmt_llvm_int(value: int) -> str:
    if int(value) < 1:
        raise ValueError("LLVM integer constants must be positive")
    return str(int(value))


def _call_native_whole_program_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    output_size: int,
) -> NDArray[np.float64]:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if output_size < 1:
        raise ValueError("native whole-program AD output_size must be positive")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("native whole-program AD output must be finite")
    return cast(NDArray[np.float64], output)


def _call_native_whole_program_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    vector: FloatArray,
    output_size: int,
) -> NDArray[np.float64]:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(_as_finite_vector("vector", vector), dtype=np.float64)
    if output_size < 1:
        raise ValueError("native whole-program AD output_size must be positive")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("native whole-program AD output must be finite")
    return cast(NDArray[np.float64], output)


def _call_native_whole_program_batch_value_gradient(
    function: Callable[[Any, int, Any, Any], None],
    values: FloatArray,
    parameter_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    checked_values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if checked_values.ndim != 2:
        raise ValueError("native whole-program AD batch values must be two-dimensional")
    if checked_values.shape[0] < 1:
        raise ValueError("native whole-program AD batch values must contain at least one row")
    if checked_values.shape[1] != parameter_count:
        raise ValueError("native whole-program AD batch parameter count mismatch")
    if not np.all(np.isfinite(checked_values)):
        raise ValueError("native whole-program AD batch values must be finite")
    rows = int(checked_values.shape[0])
    value_output = np.zeros(rows, dtype=np.float64)
    gradient_output = np.zeros((rows, parameter_count), dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        rows,
        value_output.ctypes.data_as(double_pointer),
        gradient_output.ctypes.data_as(double_pointer),
    )
    if not np.all(np.isfinite(value_output)) or not np.all(np.isfinite(gradient_output)):
        raise ValueError("native whole-program AD batch output must be finite")
    return (
        cast(NDArray[np.float64], value_output),
        cast(NDArray[np.float64], gradient_output),
    )


def _call_native_whole_program_batch_jvp(
    function: Callable[[Any, Any, int, Any], None],
    values: FloatArray,
    tangents: FloatArray,
    parameter_count: int,
) -> NDArray[np.float64]:
    checked_values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    checked_tangents = np.ascontiguousarray(np.asarray(tangents, dtype=np.float64))
    if checked_values.ndim != 2:
        raise ValueError("native whole-program AD batch values must be two-dimensional")
    if checked_values.shape[0] < 1:
        raise ValueError("native whole-program AD batch values must contain at least one row")
    if checked_values.shape != checked_tangents.shape:
        raise ValueError("native whole-program AD batch tangents must match values shape")
    if checked_values.shape[1] != parameter_count:
        raise ValueError("native whole-program AD batch parameter count mismatch")
    if not np.all(np.isfinite(checked_values)) or not np.all(np.isfinite(checked_tangents)):
        raise ValueError("native whole-program AD batch JVP inputs must be finite")
    rows = int(checked_values.shape[0])
    output = np.zeros(rows, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_tangents.ctypes.data_as(double_pointer),
        rows,
        output.ctypes.data_as(double_pointer),
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("native whole-program AD batch JVP output must be finite")
    return cast(NDArray[np.float64], output)


def _call_native_whole_program_batch_vjp(
    function: Callable[[Any, Any, int, Any], None],
    values: FloatArray,
    cotangents: FloatArray,
    parameter_count: int,
) -> NDArray[np.float64]:
    checked_values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    checked_cotangents = np.ascontiguousarray(np.asarray(cotangents, dtype=np.float64))
    if checked_values.ndim != 2:
        raise ValueError("native whole-program AD batch values must be two-dimensional")
    if checked_values.shape[0] < 1:
        raise ValueError("native whole-program AD batch values must contain at least one row")
    if checked_values.shape[1] != parameter_count:
        raise ValueError("native whole-program AD batch parameter count mismatch")
    if checked_cotangents.shape != (checked_values.shape[0],):
        raise ValueError("native whole-program AD batch cotangent row count mismatch")
    if not np.all(np.isfinite(checked_values)) or not np.all(np.isfinite(checked_cotangents)):
        raise ValueError("native whole-program AD batch VJP inputs must be finite")
    rows = int(checked_values.shape[0])
    output = np.zeros((rows, parameter_count), dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_cotangents.ctypes.data_as(double_pointer),
        rows,
        output.ctypes.data_as(double_pointer),
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("native whole-program AD batch VJP output must be finite")
    return cast(NDArray[np.float64], output)


def _verify_native_whole_program_ad_kernel(
    result: WholeProgramADResult,
    native_functions: Mapping[str, Any],
    sample_values: NDArray[np.float64],
) -> CompilerADKernelVerification:
    value = _call_native_whole_program_unary(native_functions["value"], sample_values, 1)
    gradient = _call_native_whole_program_unary(
        native_functions["gradient"],
        sample_values,
        int(result.gradient.size),
    )
    tangent = np.ones(result.gradient.size, dtype=np.float64)
    jvp = _call_native_whole_program_binary(native_functions["jvp"], sample_values, tangent, 1)
    cotangent = np.ones(1, dtype=np.float64)
    vjp = _call_native_whole_program_binary(
        native_functions["vjp"],
        sample_values,
        cotangent,
        int(result.gradient.size),
    )
    batch_values, batch_gradients = _call_native_whole_program_batch_value_gradient(
        native_functions["batch_value_gradient"],
        sample_values.reshape(1, -1),
        int(result.gradient.size),
    )
    batch_jvp = _call_native_whole_program_batch_jvp(
        native_functions["batch_jvp"],
        sample_values.reshape(1, -1),
        tangent.reshape(1, -1),
        int(result.gradient.size),
    )
    batch_vjp = _call_native_whole_program_batch_vjp(
        native_functions["batch_vjp"],
        sample_values.reshape(1, -1),
        cotangent,
        int(result.gradient.size),
    )
    expected_value = np.array([result.value], dtype=np.float64)
    expected_jvp = np.array([float(np.dot(result.gradient, tangent))], dtype=np.float64)
    expected_vjp = result.gradient.copy()
    errors = (
        _max_abs_error(value, expected_value),
        _max_abs_error(gradient, result.gradient),
        _max_abs_error(jvp, expected_jvp),
        _max_abs_error(vjp, expected_vjp),
        _max_abs_error(batch_values, expected_value),
        _max_abs_error(batch_gradients, result.gradient.reshape(1, -1)),
        _max_abs_error(batch_jvp, expected_jvp),
        _max_abs_error(batch_vjp, result.gradient.reshape(1, -1)),
    )
    return CompilerADKernelVerification(
        value_close=bool(np.allclose(value, expected_value, rtol=1.0e-10, atol=1.0e-10)),
        jvp_close=bool(
            np.allclose(jvp, expected_jvp, rtol=1.0e-10, atol=1.0e-10)
            and np.allclose(batch_jvp, expected_jvp, rtol=1.0e-10, atol=1.0e-10)
        ),
        vjp_close=bool(
            np.allclose(vjp, expected_vjp, rtol=1.0e-10, atol=1.0e-10)
            and np.allclose(batch_vjp, result.gradient.reshape(1, -1), rtol=1.0e-10, atol=1.0e-10)
        ),
        gradient_close=bool(
            np.allclose(gradient, result.gradient, rtol=1.0e-10, atol=1.0e-10)
            and np.allclose(
                batch_gradients, result.gradient.reshape(1, -1), rtol=1.0e-10, atol=1.0e-10
            )
            and np.allclose(batch_values, expected_value, rtol=1.0e-10, atol=1.0e-10)
        ),
        max_abs_error=max(errors),
        samples=1,
    )


def _annotate_whole_program_native_mlir(
    module: MLIRModule,
    llvm_ir: str,
    result: WholeProgramADResult,
) -> MLIRModule:
    llvm_sha256 = hashlib.sha256(llvm_ir.encode("utf-8")).hexdigest()
    lowering_report = analyse_whole_program_ad_native_lowering(result)
    if not module.text.endswith("}\n"):
        raise ValueError("whole-program MLIR module must end with a module terminator")
    text = (
        module.text[:-2]
        + "  scpn_diff.native_llvm_jit "
        + '{execution = "native_llvm_jit", '
        + f'gradient = "forward_kernel", llvm_sha256 = "{llvm_sha256}"}}\n'
        + "}\n"
    )
    resource_counts = dict(module.resource_counts)
    resource_counts["native_whole_program_kernels"] = 1
    resource_counts["native_whole_program_batch_kernels"] = 1
    resource_counts["native_whole_program_batch_transform_kernels"] = 2
    resource_counts["native_supported_ops"] = len(lowering_report.lowerable_ops)
    resource_counts["native_lowerable_ops"] = len(lowering_report.lowerable_ops)
    resource_counts["native_unsupported_ops"] = len(lowering_report.unsupported_ops)
    resource_counts["native_supported_elementary_ops"] = sum(
        1
        for op in lowering_report.lowerable_ops
        if op
        in {
            "sin",
            "cos",
            "exp",
            "expm1",
            "log",
            "log1p",
            "sqrt",
            "tan",
            "tanh",
            "arcsin",
            "arccos",
            "reciprocal",
            "square",
            "abs",
        }
    )
    metadata = dict(module.metadata)
    polyglot_targets = dict(metadata.get("polyglot_targets", {}))
    polyglot_targets["llvm"] = (
        "available: native_llvm_jit scalar program AD with expanded elementary ops "
        "and stable executed branch signatures"
    )
    polyglot_targets["jit"] = (
        "available: native_llvm_jit scalar program AD with expanded elementary ops "
        "and stable executed branch signatures"
    )
    metadata.update(
        {
            "claim_boundary": (
                "whole-program AD trace interchange plus native LLVM/JIT lowering "
                "for supported scalar traces with expanded elementary ops and stable "
                "executed branch signatures"
            ),
            "llvm_ir_sha256": llvm_sha256,
            "native_backend": "native_llvm_jit",
            "native_supported_ops": lowering_report.lowerable_ops,
            "native_lowering_report": lowering_report.as_metadata(),
            "native_lowerable_ops": lowering_report.lowerable_ops,
            "native_unsupported_ops": lowering_report.unsupported_ops,
            "native_fail_closed_reason": lowering_report.fail_closed_reason,
            "polyglot_targets": polyglot_targets,
        }
    )
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=module.dialect,
        resource_counts=resource_counts,
        metadata=metadata,
    )


def _whole_program_replay_signature(result: WholeProgramADResult) -> tuple[str, ...]:
    """Return a stable non-numeric signature for supported program AD replay."""

    control_signature = tuple(
        f"{node.index}:{node.op}:{','.join(node.inputs)}"
        for node in result.ir_nodes
        if node.op.startswith(("branch:", "loop:", "control:"))
    )
    if control_signature:
        return control_signature
    return tuple(f"{node.index}:{node.op}:{','.join(node.inputs)}" for node in result.ir_nodes)
