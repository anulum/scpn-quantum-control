# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native Qiskit measurement and submission declarations
"""Read native circuit wiring without compiling or simulating another answer."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.primitives.containers import BitArray, SamplerPubResult

from scpn_quantum_control.dense_budget import dense_budget_bytes

from .hal import QuantumWorkload
from .provider_semantics import (
    GateModelObservation,
    NativeRegisterSamples,
    SubmissionSemantics,
    WorkloadSemantics,
)


def qiskit_workload_semantics(
    circuit: QuantumCircuit,
    program: str,
    *,
    requested_target: str | None = None,
    parameter_bindings: Mapping[Parameter, float] | None = None,
) -> WorkloadSemantics:
    """Extract final static measurements and shared native parameter identities.

    Parameters
    ----------
    circuit
        Actual native source circuit. Only final static measurements qualify.
    program
        Exact original encoded HAL programme; QPY retains parameter UUIDs.
    requested_target
        Optional exact requested backend name; None leaves target selection open.
    parameter_bindings
        Explicit values keyed by the actual original native Parameter objects.
        Native operator units remain unchanged; no unit conversion is performed.

    Returns
    -------
    WorkloadSemantics
        Source-bound ordered wiring, native registers and shared parameter uses.

    Raises
    ------
    TypeError
        If the supplied owner is not a native Qiskit circuit.
    ValueError
        If control flow, a gate after measurement or ambiguous wiring is present.

    Notes
    -----
    This reads native metadata; it never binds parameters, compiles or simulates.
    Unbound circuits may be preserved but cannot execute through sampled adapters.

    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("native measurement source must be a QuantumCircuit")
    measurements: list[tuple[int, int]] = []
    uses: dict[str, list[tuple[int, int]]] = {}
    measured = False
    for index, instruction in enumerate(circuit.data):
        operation = instruction.operation
        name = operation.name
        if name in {"if_else", "while_loop", "for_loop", "switch_case", "store"}:
            raise ValueError("control flow is outside final measurement semantics")
        if name == "measure":
            measured = True
            measurements.append(
                (
                    int(circuit.find_bit(instruction.qubits[0]).index),
                    int(circuit.find_bit(instruction.clbits[0]).index),
                )
            )
        elif measured and name != "barrier":
            raise ValueError("a gate follows the declared final measurements")
        for argument, expression in enumerate(operation.params):
            if isinstance(expression, ParameterExpression):
                for parameter in expression.parameters:
                    uses.setdefault(str(parameter.uuid), []).append((index, argument))
    registers = tuple(
        (register.name, tuple(int(circuit.find_bit(bit).index) for bit in register))
        for register in circuit.cregs
    )
    phase_expression = circuit.global_phase
    phase_parameters = (
        phase_expression.parameters if isinstance(phase_expression, ParameterExpression) else set()
    )
    phase_ids = tuple(
        str(parameter.uuid) for parameter in circuit.parameters if parameter in phase_parameters
    )
    parameters = tuple(
        (parameter.name, str(parameter.uuid), tuple(uses.get(str(parameter.uuid), ())))
        for parameter in circuit.parameters
    )
    bindings: dict[str, float] = {}
    for parameter, value in (parameter_bindings or {}).items():
        if not isinstance(parameter, Parameter) or parameter not in circuit.parameters:
            raise ValueError("parameter binding must identify an original native Parameter")
        bindings[str(parameter.uuid)] = value
    values = tuple(
        (str(parameter.uuid), bindings[str(parameter.uuid)])
        for parameter in circuit.parameters
        if str(parameter.uuid) in bindings
    )
    return WorkloadSemantics(
        program_sha256=hashlib.sha256(program.encode("utf-8")).hexdigest(),
        n_qubits=int(circuit.num_qubits),
        n_clbits=int(circuit.num_clbits),
        measurement_map=tuple(measurements),
        classical_registers=registers,
        parameters=parameters,
        global_phase_parameters=phase_ids,
        parameter_values=values,
        requested_target=requested_target,
    )


def _require_qiskit_source(
    workload: QuantumWorkload, circuit: QuantumCircuit
) -> WorkloadSemantics | None:
    """Validate actual original source, shared identities and explicit bindings."""
    if int(circuit.num_qubits) != workload.n_qubits:
        raise ValueError("decoded circuit logical width differs from workload")
    request = workload.semantics
    parameters = {str(parameter.uuid): parameter for parameter in circuit.parameters}
    if request is None:
        if parameters:
            raise ValueError("sampled execution requires explicitly bound native parameters")
        return None
    if not isinstance(request, WorkloadSemantics):
        raise ValueError("Qiskit submission requires native gate-model semantics")
    values = dict(request.parameter_values)
    if set(values) != set(parameters):
        raise ValueError("sampled execution requires all original native parameter bindings")
    observed = qiskit_workload_semantics(
        circuit,
        workload.program,
        requested_target=request.requested_target,
        parameter_bindings={parameters[uuid]: value for uuid, value in values.items()},
    )
    if observed != request:
        raise ValueError("decoded circuit differs from declared native measurement semantics")
    return request


def bind_qiskit_workload(workload: QuantumWorkload, circuit: QuantumCircuit) -> QuantumCircuit:
    """Use native Qiskit binding while preserving the original encoded request.

    Parameters
    ----------
    workload
        Source-bound original request with explicit native UUID/value bindings.
    circuit
        Actual original decoded circuit; it remains unchanged.

    Returns
    -------
    QuantumCircuit
        Native bound execution copy, or the already bound original circuit.

    Raises
    ------
    ValueError
        If source identity or binding coverage differs before provider execution.

    """
    request = _require_qiskit_source(workload, circuit)
    if request is None or not circuit.parameters:
        return circuit
    values = dict(request.parameter_values)
    return circuit.assign_parameters(
        {parameter: values[str(parameter.uuid)] for parameter in circuit.parameters}, inplace=False
    )


def qiskit_submission_semantics(
    workload: QuantumWorkload,
    circuit: QuantumCircuit,
    *,
    target_name: str,
    compiled_program: str | None = None,
    compilation: Literal["targeted", "native_provider", "caller_precompiled"] = "native_provider",
) -> SubmissionSemantics | None:
    """Bind an opt-in native declaration before the provider submission.

    Parameters
    ----------
    workload
        Original public workload. Legacy requests without a companion remain unknown.
    circuit
        Actual decoded original circuit, before target compilation.
    target_name
        Actual selected backend name.
    compiled_program
        Actual compiled native encoded programme, when target compilation occurred.
    compilation
        Provenance of the native payload; no inference from a provider's name.

    Returns
    -------
    SubmissionSemantics or None
        Bound source/target/shot record, or None for a legacy request.

    Raises
    ------
    ValueError
        If logical width, source wiring, shared parameters or requested target differ.

    """
    request = _require_qiskit_source(workload, circuit)
    if request is None:
        return None
    return SubmissionSemantics(
        request=request,
        original_program=workload.program,
        ir_format=workload.ir_format,
        requested_shots=workload.shots,
        effective_shots=workload.shots,
        target_name=target_name,
        compilation=compilation,
        compiled_program_sha256=(
            hashlib.sha256(compiled_program.encode("utf-8")).hexdigest()
            if compiled_program is not None
            else None
        ),
    )


def require_runtime_sample_buffers(request: WorkloadSemantics, shots: int) -> int:
    """Admit declared native packed sample buffers before copies or submission.

    Parameters
    ----------
    request
        Original native classical register layout, with at least one bit.
    shots
        Exact positive sample count; parameter broadcast is outside this contract.

    Returns
    -------
    int
        Conservative native byte-buffer estimate, measured in bytes.

    Raises
    ------
    ValueError
        If shots or the classical register layout cannot produce sample buffers.
    MemoryError
        If the estimate exceeds native addressability or the active byte budget.

    Notes
    -----
    For packed register bytes P, total classical bits B and joint packed bytes C,
    the estimate is 10P + shots*B + 9C. It includes original and detached buffers,
    native uint8 unpacking, concatenation, padding and packed joint output.
    Uses the existing SCPN_MAX_DENSE_GIB byte allowance without a Hilbert-space
    dimension. The budget is a snapshot, not a reservation or total process
    memory guarantee; provider allocations and Python count dictionaries are
    outside this native byte-buffer estimate.

    """
    if type(shots) is not int or shots <= 0 or not request.classical_registers:
        raise ValueError("native runtime samples require positive shots and classical registers")
    packed = shots * sum((len(bits) + 7) // 8 for _, bits in request.classical_registers)
    joint = shots * ((request.n_clbits + 7) // 8)
    required = 10 * packed + shots * request.n_clbits + 9 * joint
    if required > sys.maxsize or required > dense_budget_bytes():
        raise MemoryError("native runtime sample buffers exceed the active memory budget")
    return required


def native_runtime_gate_observation(
    pub_result: SamplerPubResult,
    request: WorkloadSemantics,
    *,
    shots: int,
) -> GateModelObservation:
    """Retain actual packed register samples and their native joint counts.

    Parameters
    ----------
    pub_result
        Actual native single-circuit SamplerPubResult, not a transport imitation.
    request
        Stored original source declaration, including native register order.
    shots
        Exact requested and observed shot total.

    Returns
    -------
    GateModelObservation
        Detached original uint8 register bytes and native joint count evidence.

    Raises
    ------
    ValueError
        If result registers, native packed shapes, widths or totals differ.
    TypeError
        If the result does not contain native BitArray register data.
    MemoryError
        If declared native byte buffers exceed the active memory budget.

    Notes
    -----
    Native Qiskit joins registers; this does not synthesize independent marginals.
    Arrays have shape ``(shots, ceil(register_bits / 8))`` for an admitted single
    circuit. Parameter-broadcast result arrays are outside this admission.

    """
    if not isinstance(pub_result, SamplerPubResult):
        raise TypeError("native runtime observation requires SamplerPubResult")
    names = [name for name, _ in request.classical_registers]
    if list(pub_result.data) != names:
        raise ValueError("native runtime register order differs from stored request")
    arrays: list[BitArray] = []
    for name, bits in request.classical_registers:
        array = pub_result.data[name]
        if not isinstance(array, BitArray):
            raise TypeError("native runtime register data must be BitArray")
        native = array.array
        if native.dtype.name != "uint8" or len(native.shape) != 2:
            raise ValueError("native runtime samples require a two-dimensional uint8 array")
        if int(array.num_bits) != len(bits) or native.shape != (shots, (len(bits) + 7) // 8):
            raise ValueError("native runtime register width or shots differ from stored request")
        arrays.append(array)
    require_runtime_sample_buffers(request, shots)
    samples: list[NativeRegisterSamples] = []
    for name, array in zip(names, arrays, strict=True):
        native = array.array
        samples.append(
            NativeRegisterSamples(
                name=name,
                num_bits=int(array.num_bits),
                shape=(int(native.shape[0]), int(native.shape[1])),
                data=bytes(native.tobytes(order="C")),
            )
        )
    joint = BitArray.concatenate_bits(arrays)
    return GateModelObservation(
        request=request,
        raw_counts=joint.get_counts(),
        shots=shots,
        register_samples=tuple(samples),
    )


__all__ = [
    "qiskit_workload_semantics",
    "qiskit_submission_semantics",
    "bind_qiskit_workload",
    "require_runtime_sample_buffers",
    "native_runtime_gate_observation",
]
