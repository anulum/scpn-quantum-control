# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD Rust bridge wrappers
"""Rust bridge wrappers for bounded Program AD effect IR replay.

The functions in this module are the Python-facing boundary around optional
PyO3 exports from ``scpn_quantum_engine``. They validate JSON payloads,
normalise NumPy inputs, and preserve explicit claim boundaries for scalar,
elementwise-array, structural-array, static-reduction, fixed ``multi_dot``
linalg-array, 2x2 distinct symmetric ``eigvalsh``, 2x2 distinct symmetric
``eigh`` eigenvalues/nonzero-offdiagonal eigenvectors, and 2x2 real-distinct
``eigvals`` replay, plus 2x2 distinct-positive ``svd(..., compute_uv=False)``
singular-value replay and constant-full-rank rank-1/Nx2/2xN ``pinv``
replay without importing the larger differentiable-programming facade.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

_FORWARD_CLAIM_BOUNDARY = "bounded_rust_program_ad_ir_scalar_static_cumulative_and_static_linalg_primitives_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
_VALUE_AND_GRAD_CLAIM_BOUNDARY = "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
_REGISTRY_METADATA_MIRROR_CLAIM_BOUNDARY = (
    "rust_program_ad_registry_metadata_mirror_only_no_execution_promotion"
)


class ProgramADEffectIRLike(Protocol):
    """Structural subset required from Program AD IR objects."""

    @property
    def serialization(self) -> str:  # pragma: no cover - structural typing only.
        """Return the serialized ``program_ad_effect_ir.v1`` payload."""
        ...


@dataclass(frozen=True)
class RustProgramADInterpreterResult:
    """Result from the bounded Rust Program AD IR scalar interpreter."""

    supported: bool
    value: float | None
    effect_count: int
    supported_effect_count: int
    blocked_reasons: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not isinstance(self.supported, bool):
            raise ValueError("Rust Program AD interpreter supported flag must be boolean")
        if self.value is not None:
            _as_real_scalar("Rust Program AD interpreter value", self.value)
        if self.effect_count < 0 or self.supported_effect_count < 0:
            raise ValueError("Rust Program AD interpreter counts must be non-negative")
        if self.supported_effect_count > self.effect_count:
            raise ValueError("Rust Program AD interpreter supported count exceeds effect count")
        if any(not isinstance(reason, str) or not reason for reason in self.blocked_reasons):
            raise ValueError("Rust Program AD interpreter blocked reasons must be non-empty")
        if self.supported != (self.value is not None and not self.blocked_reasons):
            raise ValueError("Rust Program AD interpreter supported state is inconsistent")
        if not self.claim_boundary:
            raise ValueError("Rust Program AD interpreter claim boundary must be non-empty")


@dataclass(frozen=True)
class RustProgramADValueAndGradientResult:
    """Result from bounded Rust Program AD value and gradient replay."""

    supported: bool
    value: float | None
    gradient: NDArray[np.float64]
    parameter_targets: tuple[str, ...]
    effect_count: int
    supported_effect_count: int
    blocked_reasons: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not isinstance(self.supported, bool):
            raise ValueError("Rust Program AD value+gradient supported flag must be boolean")
        if self.value is not None:
            _as_real_scalar("Rust Program AD value+gradient value", self.value)
        checked_gradient = _as_real_numeric_array(
            "Rust Program AD value+gradient gradient",
            self.gradient,
        )
        if checked_gradient.ndim != 1:
            raise ValueError("Rust Program AD value+gradient gradient must be one-dimensional")
        if checked_gradient.size != len(self.parameter_targets):
            raise ValueError("Rust Program AD value+gradient target count must match gradient")
        if self.effect_count < 0 or self.supported_effect_count < 0:
            raise ValueError("Rust Program AD value+gradient counts must be non-negative")
        if self.supported_effect_count > self.effect_count:
            raise ValueError("Rust Program AD value+gradient supported count exceeds effect count")
        if any(not isinstance(target, str) or not target for target in self.parameter_targets):
            raise ValueError("Rust Program AD value+gradient targets must be non-empty")
        if any(not isinstance(reason, str) or not reason for reason in self.blocked_reasons):
            raise ValueError("Rust Program AD value+gradient blocked reasons must be non-empty")
        if self.supported != (
            self.value is not None and checked_gradient.size > 0 and not self.blocked_reasons
        ):
            raise ValueError("Rust Program AD value+gradient supported state is inconsistent")
        if not self.claim_boundary:
            raise ValueError("Rust Program AD value+gradient claim boundary must be non-empty")


@dataclass(frozen=True)
class RustProgramADRegistryMetadataMirrorResult:
    """Result from the Rust Program AD registry metadata mirror."""

    supported: bool
    primitive_count: int
    covered_primitives: int
    family_counts: Mapping[str, int]
    facet_counts: Mapping[str, int]
    executable_operation_count: int
    executable_operations: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not isinstance(self.supported, bool):
            raise ValueError("Rust Program AD registry metadata supported flag must be boolean")
        if self.primitive_count < 0 or self.covered_primitives < 0:
            raise ValueError("Rust Program AD registry metadata counts must be non-negative")
        if self.covered_primitives > self.primitive_count:
            raise ValueError("Rust Program AD registry metadata covered count exceeds total")
        checked_family_counts = _normalise_program_ad_int_mapping(
            "Rust Program AD registry metadata mirror family_counts",
            self.family_counts,
        )
        checked_facet_counts = _normalise_program_ad_int_mapping(
            "Rust Program AD registry metadata mirror facet_counts",
            self.facet_counts,
        )
        if sum(checked_family_counts.values()) != self.primitive_count:
            raise ValueError("Rust Program AD registry metadata family counts must sum to total")
        if self.executable_operation_count != len(self.executable_operations):
            raise ValueError("Rust Program AD registry metadata executable count is inconsistent")
        if any(
            not isinstance(operation, str) or not operation
            for operation in self.executable_operations
        ):
            raise ValueError(
                "Rust Program AD registry metadata executable operations must be non-empty"
            )
        if tuple(sorted(set(self.executable_operations))) != self.executable_operations:
            raise ValueError(
                "Rust Program AD registry metadata executable operations must be sorted and unique"
            )
        if any(not isinstance(reason, str) or not reason for reason in self.blocked_reasons):
            raise ValueError("Rust Program AD registry metadata blocked reasons must be non-empty")
        if self.supported != (
            self.covered_primitives == self.primitive_count and not self.blocked_reasons
        ):
            raise ValueError("Rust Program AD registry metadata supported state is inconsistent")
        if not self.claim_boundary:
            raise ValueError("Rust Program AD registry metadata claim boundary must be non-empty")
        object.__setattr__(self, "family_counts", checked_family_counts)
        object.__setattr__(self, "facet_counts", checked_facet_counts)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready metadata mirror payload."""

        return {
            "supported": self.supported,
            "primitive_count": self.primitive_count,
            "covered_primitives": self.covered_primitives,
            "family_counts": dict(self.family_counts),
            "facet_counts": dict(self.facet_counts),
            "executable_operation_count": self.executable_operation_count,
            "executable_operations": list(self.executable_operations),
            "blocked_reasons": list(self.blocked_reasons),
            "claim_boundary": self.claim_boundary,
        }


def interpret_program_ad_effect_ir_with_rust(
    program_ir: ProgramADEffectIRLike | str,
    inputs: Sequence[float] | NDArray[np.float64],
) -> RustProgramADInterpreterResult:
    """Execute a bounded scalar ``program_ad_effect_ir.v1`` trace in Rust.

    The Rust path is deliberately narrow: every effect row must include the
    opcode-bearing ``operation`` metadata emitted by current Python traces, all
    operations must be scalar and finite. Executed runtime branch metadata is
    accepted only when it is side-effect-free and matched by runtime phi
    provenance; aliases, mutation, arrays, source-level branch semantics, and
    non-executed branch adjoints fail closed instead of falling back to Python
    execution. This is not LLVM, JIT, reverse-mode compiler AD, provider,
    hardware, or performance evidence.
    """

    serialization = _program_ad_serialization(program_ir)
    checked_inputs = _validate_inputs("Rust Program AD interpreter inputs", inputs)
    try:
        import scpn_quantum_engine as engine
    except ModuleNotFoundError:
        return RustProgramADInterpreterResult(
            supported=False,
            value=None,
            effect_count=0,
            supported_effect_count=0,
            blocked_reasons=("scpn_quantum_engine native extension is not built",),
            claim_boundary=_FORWARD_CLAIM_BOUNDARY,
        )
    raw = engine.program_ad_effect_ir_interpret_forward(
        serialization,
        [float(value) for value in checked_inputs],
    )
    if not isinstance(raw, str):
        raise ValueError("Rust Program AD interpreter must return JSON text")
    payload = _decode_payload("Rust Program AD interpreter", raw)
    value = payload.get("value")
    return RustProgramADInterpreterResult(
        supported=bool(payload.get("supported")),
        value=float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else None,
        effect_count=_parse_program_ad_int(
            "Rust Program AD interpreter effect_count",
            payload.get("effect_count"),
        ),
        supported_effect_count=_parse_program_ad_int(
            "Rust Program AD interpreter supported_effect_count",
            payload.get("supported_effect_count"),
        ),
        blocked_reasons=_parse_program_ad_str_tuple(
            "Rust Program AD interpreter blocked reason",
            payload.get("blocked_reasons", []),
        ),
        claim_boundary=_parse_program_ad_str(
            "Rust Program AD interpreter claim_boundary",
            payload.get("claim_boundary"),
        ),
    )


def value_and_grad_program_ad_effect_ir_with_rust(
    program_ir: ProgramADEffectIRLike | str,
    inputs: Sequence[float] | NDArray[np.float64],
) -> RustProgramADValueAndGradientResult:
    """Replay bounded Program AD value and gradient in Rust.

    The replay is intentionally limited to opcode-bearing scalar, shaped
    elementwise, and static structural ``program_ad_effect_ir.v1`` rows,
    including scalar-to-array broadcasting, static ``reshape``/``ravel``,
    reversed-axis ``transpose``, ``broadcast_to``, static-axis ``concatenate``/
    ``stack``, static source-map ``index_map`` indexing, static-axis
    ``sum``/``mean``/``prod``/``var``/``std``/``max``/``min``/``median``
    reductions with static ``ddof``/``correction`` metadata for ``var``/``std``,
    compact static-grid ``trapezoid`` reductions with ``dx``/``x``/``xfull``
    metadata, scalar-``q`` ``quantile``/``percentile`` reductions, and scalar
    all-axis ``sum``/``mean``/``prod``/``var``/``std``/``max``/``min``/
    ``median`` plus compact static-grid ``trapezoid`` and scalar-``q``
    ``quantile``/``percentile`` objective closure.
    Static linalg
    replay remains scalar-SSA only. Aliases, mutation, non-lowered dynamic
    indexing semantics, dynamic axes, dynamic trapezoid-grid metadata, dynamic
    q/method metadata, dynamic ``ddof``/``correction`` metadata, zero-variance ``std`` gradients,
    dynamic structural operations,
    provider execution, hardware execution, LLVM/JIT execution, and performance
    claims fail closed instead of falling back to Python.
    Executed runtime branch metadata is replayed only as provenance for the
    already-executed path; non-executed branch adjoints and source-level
    control-flow lowering remain fail-closed.
    """

    serialization = _program_ad_serialization(program_ir)
    checked_inputs = _validate_inputs("Rust Program AD value+gradient inputs", inputs)
    try:
        import scpn_quantum_engine as engine
    except ModuleNotFoundError:
        return RustProgramADValueAndGradientResult(
            supported=False,
            value=None,
            gradient=np.array([], dtype=np.float64),
            parameter_targets=(),
            effect_count=0,
            supported_effect_count=0,
            blocked_reasons=("scpn_quantum_engine native extension is not built",),
            claim_boundary=_VALUE_AND_GRAD_CLAIM_BOUNDARY,
        )
    interpreter = getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None)
    if not callable(interpreter):
        return RustProgramADValueAndGradientResult(
            supported=False,
            value=None,
            gradient=np.array([], dtype=np.float64),
            parameter_targets=(),
            effect_count=0,
            supported_effect_count=0,
            blocked_reasons=(
                "scpn_quantum_engine native extension lacks Program AD value+gradient replay",
            ),
            claim_boundary=_VALUE_AND_GRAD_CLAIM_BOUNDARY,
        )
    raw = interpreter(
        serialization,
        [float(value) for value in checked_inputs],
    )
    if not isinstance(raw, str):
        raise ValueError("Rust Program AD value+gradient must return JSON text")
    payload = _decode_payload("Rust Program AD value+gradient", raw)
    value = payload.get("value")
    gradient_payload = payload.get("gradient")
    if not isinstance(gradient_payload, list):
        raise ValueError("Rust Program AD value+gradient gradient must be a JSON list")
    return RustProgramADValueAndGradientResult(
        supported=bool(payload.get("supported")),
        value=float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else None,
        gradient=np.array(
            [
                _as_real_scalar("Rust Program AD value+gradient gradient item", item)
                for item in gradient_payload
            ],
            dtype=np.float64,
        ),
        parameter_targets=_parse_program_ad_str_tuple(
            "Rust Program AD value+gradient parameter target",
            payload.get("parameter_targets", []),
        ),
        effect_count=_parse_program_ad_int(
            "Rust Program AD value+gradient effect_count",
            payload.get("effect_count"),
        ),
        supported_effect_count=_parse_program_ad_int(
            "Rust Program AD value+gradient supported_effect_count",
            payload.get("supported_effect_count"),
        ),
        blocked_reasons=_parse_program_ad_str_tuple(
            "Rust Program AD value+gradient blocked reason",
            payload.get("blocked_reasons", []),
        ),
        claim_boundary=_parse_program_ad_str(
            "Rust Program AD value+gradient claim_boundary",
            payload.get("claim_boundary"),
        ),
    )


def mirror_program_ad_registry_metadata_with_rust() -> RustProgramADRegistryMetadataMirrorResult:
    """Mirror Python Program AD registry metadata through the optional Rust extension.

    The mirror serializes the canonical Python registry-dispatch coverage
    report, asks ``scpn_quantum_engine`` to validate and summarize it, and
    returns a typed fail-closed result. A supported result is metadata evidence
    only: it records registry shape, required facet counts, and overlap with
    the currently bounded Rust scalar/static-linalg, array, static
    ``multi_dot``, 2x2 distinct symmetric ``eigvalsh``, 2x2 distinct symmetric
    ``eigh`` eigenvalues/nonzero-offdiagonal eigenvectors, and 2x2
    real-distinct ``eigvals`` replay, plus 2x2 distinct-positive
    ``svd(..., compute_uv=False)`` singular-value replay and
    constant-full-rank rank-1/Nx2/2xN ``pinv`` replay. It does not promote
    registry-dispatched execution, broad linalg/spectral adjoints, LLVM/JIT
    lowering, provider, hardware, or performance evidence.
    """

    from .program_ad_registry import program_ad_registry_dispatch_coverage_report

    report = program_ad_registry_dispatch_coverage_report()
    snapshot = json.dumps(report.to_dict(), sort_keys=True)
    try:
        import scpn_quantum_engine as engine
    except ModuleNotFoundError:
        return _unsupported_registry_metadata_mirror(
            report.total_primitives,
            report.covered_primitives,
            report.family_counts,
            "scpn_quantum_engine native extension is not built",
        )
    mirror = getattr(engine, "program_ad_registry_metadata_mirror", None)
    if not callable(mirror):
        return _unsupported_registry_metadata_mirror(
            report.total_primitives,
            report.covered_primitives,
            report.family_counts,
            "scpn_quantum_engine native extension lacks Program AD registry metadata mirror",
        )
    raw = mirror(snapshot)
    if not isinstance(raw, str):
        raise ValueError("Rust Program AD registry metadata mirror must return JSON text")
    payload = _decode_payload("Rust Program AD registry metadata mirror", raw)
    return RustProgramADRegistryMetadataMirrorResult(
        supported=_parse_program_ad_bool(
            "Rust Program AD registry metadata mirror supported",
            payload.get("supported"),
        ),
        primitive_count=_parse_program_ad_int(
            "Rust Program AD registry metadata mirror primitive_count",
            payload.get("primitive_count"),
        ),
        covered_primitives=_parse_program_ad_int(
            "Rust Program AD registry metadata mirror covered_primitives",
            payload.get("covered_primitives"),
        ),
        family_counts=_parse_program_ad_int_mapping(
            "Rust Program AD registry metadata mirror family_counts",
            payload.get("family_counts"),
        ),
        facet_counts=_parse_program_ad_int_mapping(
            "Rust Program AD registry metadata mirror facet_counts",
            payload.get("facet_counts"),
        ),
        executable_operation_count=_parse_program_ad_int(
            "Rust Program AD registry metadata mirror executable_operation_count",
            payload.get("executable_operation_count"),
        ),
        executable_operations=_parse_program_ad_str_tuple(
            "Rust Program AD registry metadata mirror executable operation",
            payload.get("executable_operations", []),
        ),
        blocked_reasons=_parse_program_ad_str_tuple(
            "Rust Program AD registry metadata mirror blocked reason",
            payload.get("blocked_reasons", []),
        ),
        claim_boundary=_parse_program_ad_str(
            "Rust Program AD registry metadata mirror claim_boundary",
            payload.get("claim_boundary"),
        ),
    )


def _unsupported_registry_metadata_mirror(
    primitive_count: int,
    covered_primitives: int,
    family_counts: Mapping[str, int],
    blocked_reason: str,
) -> RustProgramADRegistryMetadataMirrorResult:
    return RustProgramADRegistryMetadataMirrorResult(
        supported=False,
        primitive_count=primitive_count,
        covered_primitives=covered_primitives,
        family_counts=dict(family_counts),
        facet_counts={},
        executable_operation_count=0,
        executable_operations=(),
        blocked_reasons=(blocked_reason,),
        claim_boundary=_REGISTRY_METADATA_MIRROR_CLAIM_BOUNDARY,
    )


def _program_ad_serialization(program_ir: ProgramADEffectIRLike | str) -> str:
    serialization = program_ir if isinstance(program_ir, str) else program_ir.serialization
    if not isinstance(serialization, str) or not serialization:
        raise ValueError("program AD IR serialization must be a non-empty string")
    return serialization


def _validate_inputs(
    name: str,
    inputs: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    checked_inputs = _as_real_numeric_array(name, inputs)
    if checked_inputs.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(checked_inputs)):
        raise ValueError(f"{name} must contain finite values")
    return checked_inputs


def _decode_payload(name: str, raw: str) -> dict[str, object]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} payload must be a JSON object")
    return payload


def _parse_program_ad_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"program AD IR {name} must be an integer")
    return value


def _parse_program_ad_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"program AD IR {name} must be a boolean")
    return value


def _parse_program_ad_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"program AD IR {name} must be a string")
    return value


def _parse_program_ad_str_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"program AD IR {name} must be a list")
    return tuple(_parse_program_ad_str(name, item) for item in value)


def _parse_program_ad_int_mapping(name: str, value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"program AD IR {name} must be a JSON object")
    checked: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"program AD IR {name} keys must be non-empty strings")
        checked[key] = _parse_program_ad_int(f"{name} {key}", item)
        if checked[key] < 0:
            raise ValueError(f"program AD IR {name} values must be non-negative")
    return checked


def _normalise_program_ad_int_mapping(
    name: str,
    value: Mapping[str, int],
) -> dict[str, int]:
    return _parse_program_ad_int_mapping(name, value)


def _as_real_numeric_array(name: str, values: object) -> NDArray[np.float64]:
    """Return a real numeric array without implicit string/bool/object coercion."""

    try:
        raw = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc

    if raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must contain real numeric scalars")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive NumPy path.
        raise ValueError(f"{name} must contain real numeric scalars") from exc
    return array


def _as_real_scalar(name: str, value: object) -> float:
    """Return an explicit real numeric scalar without implicit coercion."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real numeric scalar")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


__all__ = [
    "ProgramADEffectIRLike",
    "RustProgramADInterpreterResult",
    "RustProgramADRegistryMetadataMirrorResult",
    "RustProgramADValueAndGradientResult",
    "interpret_program_ad_effect_ir_with_rust",
    "mirror_program_ad_registry_metadata_with_rust",
    "value_and_grad_program_ad_effect_ir_with_rust",
]
