# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — immutable native provider request and observation custody
"""Preserve native provider semantics separately from legacy raw HAL records.

These companions describe request wiring and returned native data. They do not
attest a physical device, calibration, uncertainty or scientific qualification.
The original scientific-semantics and HAL raw codecs remain their own owners.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral
from types import MappingProxyType
from typing import Literal

from ._count_integrity import strict_non_negative_count
from .provider_modalities import ModalitySemantics

PROVIDER_SEMANTICS_SCHEMA = "provider_semantics.v1"
"""Version of the independent native provider companion."""


def _integer(value: int, name: str, *, positive: bool = False) -> None:
    """Require an exact nonnegative integer, optionally strictly positive."""
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} must be an {'positive' if positive else 'nonnegative'} integer")


def _digest(value: str, name: str) -> None:
    """Reject anything except a lowercase64-character SHA-256 digest."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _native_count(value: object) -> int:
    """Require native integral count data without the legacy string coercion."""
    if not isinstance(value, Integral):
        raise ValueError("native gate count must be an integer")
    return strict_non_negative_count(value)


@dataclass(frozen=True)
class WorkloadSemantics:
    """Declared native request wiring, bound to the unchanged programme.

    Parameters
    ----------
    program_sha256
        SHA-256 of the exact original UTF-8 encoded HAL programme.
    n_qubits, n_clbits
        Logical and classical widths in bits, not physical-target dimensions.
    measurement_map
        Ordered ``(logical_qubit, classical_bit)`` final measurement occurrences.
    classical_registers
        Native register order and global classical indices within each register.
    parameters
        Native parameter name, identity and ordered ``(instruction, argument)`` uses.
        QPY retains UUIDs; Braket OpenQASM retains its native symbol identity.
        Values and expressions remain in the original native programme codec.
    global_phase_parameters
        Native identities used by Qiskit's original global-phase expression,
        including a parameter used only by global phase and no gate instruction.
    parameter_values
        Explicit native parameter identity/value bindings: Qiskit UUIDs or
        Braket symbol identities. Empty preserves an unbound request; sampled
        execution then refuses rather than inventing a value.
    requested_target
        Explicit target name, or None when the caller has not pinned a device.
    count_bit_order
        Position of classical bit zero in a native count string.
    schema
        Exact supported companion version. Unknown versions refuse.

    Raises
    ------
    ValueError
        If version, digest, widths, indices, registers or parameter identity differ.

    """

    program_sha256: str
    n_qubits: int
    n_clbits: int
    measurement_map: tuple[tuple[int, int], ...] = ()
    classical_registers: tuple[tuple[str, tuple[int, ...]], ...] = ()
    parameters: tuple[tuple[str, str, tuple[tuple[int, int], ...]], ...] = ()
    global_phase_parameters: tuple[str, ...] = ()
    parameter_values: tuple[tuple[str, float], ...] = ()
    requested_target: str | None = None
    count_bit_order: Literal["classical_lsb_right", "classical_lsb_left"] = "classical_lsb_right"
    schema: str = PROVIDER_SEMANTICS_SCHEMA

    def __post_init__(self) -> None:
        """Validate and detach every ordered native declaration."""
        if self.schema != PROVIDER_SEMANTICS_SCHEMA:
            raise ValueError("unsupported provider semantics schema")
        _digest(self.program_sha256, "program_sha256")
        _integer(self.n_qubits, "n_qubits", positive=True)
        _integer(self.n_clbits, "n_clbits")
        measurement = tuple(tuple(pair) for pair in self.measurement_map)
        measured_bits: set[int] = set()
        for qubit, bit in measurement:
            _integer(qubit, "measurement qubit")
            _integer(bit, "measurement classical bit")
            if qubit >= self.n_qubits or bit >= self.n_clbits or bit in measured_bits:
                raise ValueError(
                    "measurement map contains an invalid or overwritten classical bit"
                )
            measured_bits.add(bit)
        registers = tuple((name, tuple(bits)) for name, bits in self.classical_registers)
        seen_names: set[str] = set()
        seen_bits: set[int] = set()
        for name, bits in registers:
            if not isinstance(name, str) or not name or name in seen_names or not bits:
                raise ValueError(
                    "classical register names must be unique with nonempty bit indices"
                )
            seen_names.add(name)
            for bit in bits:
                _integer(bit, "register bit")
                if bit >= self.n_clbits or bit in seen_bits:
                    raise ValueError("classical register indices must be distinct and in range")
                seen_bits.add(bit)
        if len(seen_bits) != self.n_clbits:
            raise ValueError("classical registers must cover the declared classical width")
        parameters = tuple(
            (name, uuid, tuple(tuple(use) for use in uses)) for name, uuid, uses in self.parameters
        )
        phase_ids = tuple(self.global_phase_parameters)
        if any(not isinstance(uuid, str) or not uuid for uuid in phase_ids) or len(
            set(phase_ids)
        ) != len(phase_ids):
            raise ValueError("global phase parameter identities must be nonempty and unique")
        seen_parameters: set[str] = set()
        for name, uuid, uses in parameters:
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(uuid, str)
                or not uuid
                or uuid in seen_parameters
                or (not uses and uuid not in phase_ids)
            ):
                raise ValueError("native parameter identity and uses must be nonempty and unique")
            seen_parameters.add(uuid)
            for instruction, argument in uses:
                _integer(instruction, "parameter instruction")
                _integer(argument, "parameter argument")
        if not set(phase_ids).issubset(seen_parameters):
            raise ValueError("global phase identities must belong to original native parameters")
        parameter_values = tuple((uuid, value) for uuid, value in self.parameter_values)
        bound_ids: set[str] = set()
        for uuid, value in parameter_values:
            if uuid not in seen_parameters or uuid in bound_ids:
                raise ValueError(
                    "parameter binding UUID must identify one original native parameter"
                )
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("parameter binding value must be finite native numeric data")
            try:
                finite = isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "parameter binding value must be finite native numeric data"
                ) from exc
            if not finite:
                raise ValueError("parameter binding value must be finite native numeric data")
            bound_ids.add(uuid)
        if self.requested_target is not None and (
            not isinstance(self.requested_target, str)
            or not self.requested_target
            or self.requested_target.strip() != self.requested_target
            or any(ord(char) < 32 or ord(char) == 127 for char in self.requested_target)
        ):
            raise ValueError("requested_target must be an exact nonempty target name")
        if self.count_bit_order not in {"classical_lsb_right", "classical_lsb_left"}:
            raise ValueError("unsupported native count bit order")
        object.__setattr__(self, "measurement_map", measurement)
        object.__setattr__(self, "classical_registers", registers)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "global_phase_parameters", phase_ids)
        object.__setattr__(self, "parameter_values", parameter_values)

    def require_source(self, program: str, n_qubits: int) -> None:
        """Require exact original programme and logical width before submission.

        Parameters
        ----------
        program
            Unchanged encoded HAL programme.
        n_qubits
            Positive logical width from the public workload.

        Raises
        ------
        ValueError
            If the companion belongs to another source or width.

        """
        if (
            hashlib.sha256(program.encode("utf-8")).hexdigest() != self.program_sha256
            or n_qubits != self.n_qubits
        ):
            raise ValueError("workload semantics source digest or logical width differs")

    def to_payload(self) -> dict[str, object]:
        """Return a detached versioned declaration without inventing observations.

        Returns
        -------
        dict[str, object]
            Exact ordered wiring, native parameter identities and requested target.

        """
        return {
            "schema": self.schema,
            "program_sha256": self.program_sha256,
            "n_qubits": self.n_qubits,
            "n_clbits": self.n_clbits,
            "measurement_map": [list(pair) for pair in self.measurement_map],
            "classical_registers": [[name, list(bits)] for name, bits in self.classical_registers],
            "parameters": [
                [name, uuid, [list(use) for use in uses]] for name, uuid, uses in self.parameters
            ],
            "global_phase_parameters": list(self.global_phase_parameters),
            "parameter_values": [list(pair) for pair in self.parameter_values],
            "requested_target": self.requested_target,
            "count_bit_order": self.count_bit_order,
        }


@dataclass(frozen=True)
class SubmissionSemantics:
    """Actual adapter submission bound to the original request and selected target.

    Parameters
    ----------
    request
        Source-bound native request declaration.
    original_program, ir_format
        Exact original encoded programme and representation identifier.
    requested_shots, effective_shots
        Positive sample counts. The adapter must not silently change them.
    target_name
        Actual selected SDK backend/device name or explicit adapter selector.
    compiled_program_sha256
        Digest of the actual compiled native payload, or None if not compiled.
    compilation
        Provenance category; caller-precompiled data never claims target compilation.
    target_origin
        Distinguish an SDK-reported target from an adapter selector. A selector
        records routing intent without claiming calibrated device identity.

    Raises
    ------
    ValueError
        If source, target, sample counts or compilation provenance are contradictory.

    """

    request: WorkloadSemantics | ModalitySemantics
    original_program: str
    ir_format: str
    requested_shots: int
    effective_shots: int
    target_name: str
    compiled_program_sha256: str | None = None
    compilation: Literal["targeted", "native_provider", "caller_precompiled"] = "native_provider"
    target_origin: Literal["native_sdk", "adapter_selector"] = "native_sdk"

    def __post_init__(self) -> None:
        """Refuse source/target drift before a submission is accepted."""
        self.request.require_source(self.original_program, self.request.n_qubits)
        _integer(self.requested_shots, "requested_shots", positive=True)
        _integer(self.effective_shots, "effective_shots", positive=True)
        if self.requested_shots != self.effective_shots:
            raise ValueError("effective shots differ from requested shots")
        if not self.ir_format or not self.target_name:
            raise ValueError("submission IR and selected target must be nonempty")
        if (
            self.request.requested_target is not None
            and self.target_name != self.request.requested_target
        ):
            raise ValueError("selected target differs from requested target")
        if self.compilation not in {"targeted", "native_provider", "caller_precompiled"}:
            raise ValueError("unsupported compilation provenance")
        if self.compiled_program_sha256 is not None:
            _digest(self.compiled_program_sha256, "compiled_program_sha256")
        if self.compilation == "targeted" and self.compiled_program_sha256 is None:
            raise ValueError("targeted compilation requires its actual native payload digest")
        if self.target_origin not in {"native_sdk", "adapter_selector"}:
            raise ValueError("unsupported target identity origin")

    def to_payload(self) -> dict[str, object]:
        """Export the original program with actual submission provenance.

        Returns
        -------
        dict[str, object]
            Detached native request, exact source, shots and explicit target origin.

        """
        return {
            "schema": PROVIDER_SEMANTICS_SCHEMA,
            "request": self.request.to_payload(),
            "original_program": self.original_program,
            "ir_format": self.ir_format,
            "requested_shots": self.requested_shots,
            "effective_shots": self.effective_shots,
            "target_name": self.target_name,
            "target_origin": self.target_origin,
            "compiled_program_sha256": self.compiled_program_sha256,
            "compilation": self.compilation,
        }

    def require_gate_request(self) -> WorkloadSemantics:
        """Require stored gate wiring before decoding a gate-model result.

        Returns
        -------
        WorkloadSemantics
            Original gate-model request, retaining its native classical order.

        Raises
        ------
        ValueError
            If this submission belongs to a non-gate modality.

        """
        if not isinstance(self.request, WorkloadSemantics):
            raise ValueError("gate-model result requires native gate request wiring")
        return self.request


def modality_submission_semantics(
    request: WorkloadSemantics | ModalitySemantics | None,
    *,
    program: str,
    ir_format: str,
    native_axes: tuple[str | int, ...],
    modality: Literal["photonic", "analog", "annealing"],
    target_name: str,
    shots: int,
    vartype: Literal["SPIN", "BINARY"] | None = None,
    caller_precompiled: bool = False,
) -> SubmissionSemantics | None:
    """Bind a decoded native plan before any builder or provider call.

    Parameters
    ----------
    request
        Optional original source declaration. Legacy requests remain unknown.
    program, ir_format
        Exact original encoded plan and native representation identifier.
    native_axes, modality, vartype
        Actual ordered identities and domain decoded by the existing plan owner.
    target_name
        Adapter's explicitly declared target selector, without SDK attestation.
    shots
        Exact positive requested sample/read count.
    caller_precompiled
        Whether an injected already prepared object requires caller provenance.

    Returns
    -------
    SubmissionSemantics or None
        Source-bound submission, or None for unchanged legacy admission.

    Raises
    ------
    ValueError
        If source, domain, ordered axes or caller target pin disagrees.

    """
    if request is None:
        return None
    if not isinstance(request, ModalitySemantics):
        raise ValueError("native non-gate plan requires modality semantics")
    observed = ModalitySemantics(
        program_sha256=hashlib.sha256(program.encode()).hexdigest(),
        modality=modality,
        native_axes=native_axes,
        vartype=vartype,
        requested_target=request.requested_target,
    )
    if observed != request:
        raise ValueError("decoded native modality source or ordered axes differ")
    return SubmissionSemantics(
        request=request,
        original_program=program,
        ir_format=ir_format,
        requested_shots=shots,
        effective_shots=shots,
        target_name=target_name,
        target_origin="adapter_selector",
        compilation="caller_precompiled" if caller_precompiled else "native_provider",
    )


@dataclass(frozen=True)
class NativeRegisterSamples:
    """Unchanged packed uint8 sample bytes from one native classical register.

    Parameters
    ----------
    name
        Original classical register name.
    num_bits
        Register width in bits.
    shape
        Native two-dimensional ``(shots, packed_bytes_per_shot)`` shape.
    data
        Exact immutable native C-order uint8 bytes, including original padding.

    Raises
    ------
    ValueError
        If identity, shape, width or byte length is incompatible.

    """

    name: str
    num_bits: int
    shape: tuple[int, int]
    data: bytes

    def __post_init__(self) -> None:
        """Preserve native packed bytes without reconstructing count marginals."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("native register name must be nonempty")
        _integer(self.num_bits, "native register num_bits", positive=True)
        shape = tuple(self.shape)
        if len(shape) != 2:
            raise ValueError("native register samples must have a two-dimensional shape")
        for dimension in shape:
            _integer(dimension, "native register dimension", positive=True)
        if (
            shape[1] != (self.num_bits + 7) // 8
            or not isinstance(self.data, bytes)
            or len(self.data) != shape[0] * shape[1]
        ):
            raise ValueError("native register packed width or original byte length differs")
        object.__setattr__(self, "shape", shape)

    def to_payload(self) -> dict[str, object]:
        """Return exact packed samples with explicit dtype, shape and byte digest.

        Returns
        -------
        dict[str, object]
            Detached base64-encoded native bytes, not a synthesized marginal.

        """
        return {
            "name": self.name,
            "num_bits": self.num_bits,
            "shape": list(self.shape),
            "dtype": "uint8",
            "data_base64": base64.b64encode(self.data).decode("ascii"),
            "data_sha256": hashlib.sha256(self.data).hexdigest(),
        }


@dataclass(frozen=True)
class GateModelObservation:
    """Native count output and its explicit classical-register wiring.

    Parameters
    ----------
    request
        Actual stored source declaration; bit positions never come from a later handle.
    raw_counts
        Native count keys, including ASCII register separators, retained unchanged.
    shots
        Positive observed sample count which must equal the exact count sum.
    register_samples
        Optional exact native packed samples in original register order. These
        preserve per-shot correlations which individual count marginals cannot.

    Raises
    ------
    ValueError
        If widths, separators, counts, collisions or shot totals are incompatible.

    """

    request: WorkloadSemantics
    raw_counts: Mapping[str, int]
    shots: int
    register_samples: tuple[NativeRegisterSamples, ...] = ()
    counts: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        """Detach native output and validate its declared bit width and total."""
        _integer(self.shots, "shots", positive=True)
        samples = tuple(self.register_samples)
        if samples:
            if tuple(sample.name for sample in samples) != tuple(
                name for name, _ in self.request.classical_registers
            ):
                raise ValueError("native register sample order differs from stored request")
            for sample, (_, bits) in zip(samples, self.request.classical_registers, strict=True):
                if sample.num_bits != len(bits) or sample.shape[0] != self.shots:
                    raise ValueError("native register sample width or shots differ")
        raw: dict[str, int] = {}
        flattened: dict[str, int] = {}
        for key, value in self.raw_counts.items():
            if (
                not isinstance(key, str)
                or not key
                or key.strip() != key
                or any(char not in "01 " for char in key)
                or "  " in key
            ):
                raise ValueError("native gate count key contains an invalid register separator")
            binary = key.replace(" ", "")
            if " " in key and [len(group) for group in key.split(" ")] != [
                len(bits) for _, bits in reversed(self.request.classical_registers)
            ]:
                raise ValueError("native gate register group widths differ from stored request")
            if len(binary) != self.request.n_clbits or binary in flattened:
                raise ValueError("native gate count width or normalisation collision differs")
            count = _native_count(value)
            raw[key] = count
            flattened[binary] = count
        if not raw or sum(flattened.values()) != self.shots:
            raise ValueError("native gate count shot total differs")
        object.__setattr__(self, "raw_counts", MappingProxyType(raw))
        object.__setattr__(self, "counts", MappingProxyType(flattened))
        object.__setattr__(self, "register_samples", samples)

    @property
    def measurement_map(self) -> tuple[tuple[int, int], ...]:
        """Return the original ordered logical-to-classical measurement map."""
        return self.request.measurement_map

    def to_payload(self) -> dict[str, object]:
        """Return native gate data with explicit wiring and unchanged raw keys.

        Returns
        -------
        dict[str, object]
            Separate provider companion; no old raw or scientific codec mutation.

        """
        return {
            "schema": PROVIDER_SEMANTICS_SCHEMA,
            "modality": "gate_model",
            "request": self.request.to_payload(),
            "raw_counts": dict(self.raw_counts),
            "counts": dict(self.counts),
            "shots": self.shots,
            "register_samples": [sample.to_payload() for sample in self.register_samples],
        }


__all__ = [
    "PROVIDER_SEMANTICS_SCHEMA",
    "WorkloadSemantics",
    "SubmissionSemantics",
    "NativeRegisterSamples",
    "GateModelObservation",
    "modality_submission_semantics",
]
