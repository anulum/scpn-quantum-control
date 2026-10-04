# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native Qiskit measurement declaration tests
"""Extract final wiring and shared parameter identity from actual native circuits."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import replace
from typing import Literal, cast

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers import BitArray, DataBin, SamplerPubResult

from scpn_quantum_control.hardware.hal_qiskit import qiskit_circuit_to_workload
from scpn_quantum_control.hardware.provider_measurement import (
    bind_qiskit_workload,
    native_runtime_gate_observation,
    qiskit_submission_semantics,
    qiskit_workload_semantics,
    require_runtime_sample_buffers,
)
from scpn_quantum_control.hardware.provider_modalities import ModalitySemantics
from scpn_quantum_control.hardware.provider_semantics import WorkloadSemantics


def test_partial_permutation_and_shared_parameter_identity() -> None:
    """Read exact native q2->c0,q0->c1 occurrences and one shared parameter."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(3, 2)
    circuit.ry(theta, 2)
    circuit.rz(2 * theta, 0)
    circuit.measure(2, 0)
    circuit.measure(0, 1)
    request = qiskit_workload_semantics(circuit, "native-program")
    assert request.program_sha256 == hashlib.sha256(b"native-program").hexdigest()
    assert request.measurement_map == ((2, 0), (0, 1))
    assert request.classical_registers == (("c", (0, 1)),)
    assert request.parameters == (("theta", str(theta.uuid), ((0, 0), (1, 0))),)


def test_native_bound_expression_has_no_remaining_parameter_identity() -> None:
    """A genuinely bound native expression does not acquire a phantom free parameter."""
    theta = Parameter("theta")
    expression = theta.bind({theta: 0.25})
    gate = Gate("bound_rotation", 1, [expression])
    definition = QuantumCircuit(1)
    definition.ry(0.25, 0)
    gate.definition = definition
    circuit = QuantumCircuit(1, 1)
    circuit.append(gate, [0])
    circuit.measure(0, 0)
    assert not expression.parameters
    assert circuit.data[0].operation.params[0] is expression
    request = qiskit_workload_semantics(circuit, "native-program")
    assert request.parameters == ()
    assert request.parameter_values == ()
    assert request.measurement_map == ((0, 0),)


def test_fixed_native_rotation_retains_value_without_a_parameter_identity() -> None:
    """A native numeric angle stays in the original program rather than becoming a binding."""
    circuit = QuantumCircuit(1, 1)
    circuit.ry(0.25, 0)
    circuit.measure(0, 0)
    workload = qiskit_circuit_to_workload(
        circuit, workload_id="fixed_rotation", shots=4, capture_semantics=True
    )
    assert circuit.data[0].operation.params == [0.25]
    assert isinstance(workload.semantics, WorkloadSemantics)
    assert workload.semantics.parameters == ()
    assert workload.semantics.parameter_values == ()
    assert workload.semantics.measurement_map == ((0, 0),)


def test_mid_circuit_gate_is_outside_final_measurement_admission() -> None:
    """A changed measured qubit cannot masquerade as a final static mapping."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    circuit.x(0)
    with pytest.raises(ValueError, match="final measurement"):
        qiskit_workload_semantics(circuit, "native-program")


def test_native_runtime_preserves_original_per_shot_register_correlations() -> None:
    """Actual native Bell samples stay correlated across separately named registers."""
    circuit = QuantumCircuit(2)
    circuit.add_register(ClassicalRegister(1, "alpha"), ClassicalRegister(1, "beta"))
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    native = StatevectorSampler(seed=7).run([circuit], shots=128).result()[0]
    request = qiskit_workload_semantics(circuit, "native-program")
    observation = native_runtime_gate_observation(native, request, shots=128)
    assert set(observation.counts) == {"00", "11"}
    assert sum(observation.counts.values()) == 128
    assert tuple(sample.name for sample in observation.register_samples) == ("alpha", "beta")
    assert observation.register_samples[0].data == bytes(native.data.alpha.array.tobytes())
    assert observation.register_samples[1].data == bytes(native.data.beta.array.tobytes())
    assert observation.register_samples[0].data == observation.register_samples[1].data
    assert len(observation.register_samples[0].data) == 128
    assert observation.raw_counts == native.join_data(["alpha", "beta"]).get_counts()


class _CopySentinel(np.ndarray[tuple[int, ...], np.dtype[np.uint8]]):
    """Actual native array refusing an unexpected copy during malformed admission."""

    def tobytes(self, order: Literal["K", "A", "C", "F"] | None = "C") -> bytes:
        """Fail if the result decoder copies bytes before validating dimensions."""
        raise AssertionError("native bytes copied before result admission")


@pytest.mark.parametrize("returned_bits,returned_shots", [(2, 4), (1, 5)])
def test_native_dimensions_refuse_before_copying_provider_bytes(
    returned_bits: int,
    returned_shots: int,
) -> None:
    """A wrong native width or shot count causes zero original-byte copies."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    request = qiskit_workload_semantics(circuit, "native-program")
    array = np.zeros((returned_shots, 1), dtype=np.uint8).view(_CopySentinel)
    native = SamplerPubResult(DataBin(c=BitArray(array, returned_bits)))
    with pytest.raises(ValueError, match="width or shots"):
        native_runtime_gate_observation(native, request, shots=4)


def test_native_register_buffers_refuse_before_copy_or_joint_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing memory budget applies before native packed result copies."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    request = qiskit_workload_semantics(circuit, "native-program")
    array = np.zeros((4, 1), dtype=np.uint8).view(_CopySentinel)
    native = SamplerPubResult(DataBin(c=BitArray(array, 1)))
    monkeypatch.setenv("SCPN_MAX_DENSE_GIB", str(1 / 1024**3))
    with pytest.raises(MemoryError, match="native runtime.*budget"):
        native_runtime_gate_observation(native, request, shots=4)


def test_native_pub_type_refusal_is_observable() -> None:
    """An opaque transport object cannot claim native packed register evidence."""
    request = qiskit_workload_semantics(QuantumCircuit(1, 1), "native-program")
    with pytest.raises(TypeError, match="SamplerPubResult"):
        native_runtime_gate_observation(cast(SamplerPubResult, object()), request, shots=4)


@pytest.mark.parametrize("control", ["if_else", "while_loop", "for_loop", "switch_case", "store"])
def test_actual_native_dynamic_control_is_outside_static_measurements(control: str) -> None:
    """Real SDK classical control cannot acquire a final static measurement map."""
    circuit = QuantumCircuit(1, 1)
    if control == "if_else":
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.x(0)
    elif control == "while_loop":
        with circuit.while_loop((circuit.clbits[0], True)):
            circuit.x(0)
    elif control == "for_loop":
        with circuit.for_loop(range(2)):
            circuit.x(0)
    elif control == "switch_case":
        with circuit.switch(circuit.cregs[0]) as case, case(0):
            circuit.x(0)
    else:
        circuit.store(circuit.clbits[0], True)
    assert circuit.data[0].operation.name == control
    with pytest.raises(ValueError, match="control flow"):
        qiskit_workload_semantics(circuit, "native-program")


def test_native_extraction_requires_real_source_and_original_binding_keys() -> None:
    """Shared expressions retain both identities and reject replacement Parameters."""
    theta, phi = Parameter("theta"), Parameter("phi")
    circuit = QuantumCircuit(1, 1)
    circuit.ry(theta + phi, 0)
    circuit.measure(0, 0)
    circuit.barrier()
    request = qiskit_workload_semantics(circuit, "native-program")
    assert request.parameters == (
        ("phi", str(phi.uuid), ((0, 0),)),
        ("theta", str(theta.uuid), ((0, 0),)),
    )
    with pytest.raises(TypeError, match="QuantumCircuit"):
        qiskit_workload_semantics(cast(QuantumCircuit, object()), "native-program")
    for keys in ({Parameter("theta"): 0.5}, {"theta": 0.5}):
        with pytest.raises(ValueError, match="original native Parameter"):
            qiskit_workload_semantics(
                circuit,
                "native-program",
                parameter_bindings=cast(Mapping[Parameter, float], keys),
            )


@pytest.mark.parametrize(
    "source_failure", ["width", "wiring", "missing_binding", "legacy_unbound", "modality"]
)
def test_public_native_binding_refuses_decoded_source_drift(source_failure: str) -> None:
    """The actual source must match stored logical width, map, domain and bindings."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(2, 2)
    circuit.ry(theta, 0)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    workload = qiskit_circuit_to_workload(
        circuit,
        workload_id="native_binding_guard",
        shots=4,
        capture_semantics=source_failure != "legacy_unbound",
        parameter_bindings=(
            None if source_failure in {"legacy_unbound", "missing_binding"} else {theta: 0.5}
        ),
    )
    if source_failure == "width":
        decoded = QuantumCircuit(1, 1)
    else:
        decoded = circuit
    if source_failure == "wiring":
        assert isinstance(workload.semantics, WorkloadSemantics)
        workload = replace(
            workload, semantics=replace(workload.semantics, measurement_map=((1, 0), (0, 1)))
        )
    elif source_failure == "modality":
        workload = replace(
            workload,
            semantics=ModalitySemantics(
                hashlib.sha256(workload.program.encode()).hexdigest(),
                "analog",
                (0, 1),
            ),
        )
    with pytest.raises(
        ValueError, match="width|semantics|parameter bindings|bound native parameters"
    ):
        bind_qiskit_workload(workload, decoded)
    assert set(circuit.parameters) == {theta}


def test_bound_legacy_submission_remains_unknown_without_native_promotion() -> None:
    """A legacy source has no companion even when its native circuit is executable."""
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    workload = qiskit_circuit_to_workload(circuit, workload_id="unknown", shots=4)
    assert bind_qiskit_workload(workload, circuit) is circuit
    assert qiskit_submission_semantics(workload, circuit, target_name="native") is None


@pytest.mark.parametrize("native_failure", ["order", "type", "broadcast", "dtype"])
def test_native_register_layout_refuses_malformed_public_sdk_data(native_failure: str) -> None:
    """Native fields, packed dtype and single-circuit shape are checked before joining."""
    circuit = QuantumCircuit(1, 1)
    request = qiskit_workload_semantics(circuit, "native-program")
    array = BitArray(np.zeros((4, 1), dtype=np.uint8), 1)
    if native_failure == "order":
        data = DataBin(other=array)
    elif native_failure == "type":
        data = DataBin(c=np.zeros((4, 1), dtype=np.uint8))
    elif native_failure == "broadcast":
        data = DataBin(c=BitArray(np.zeros((2, 4, 1), dtype=np.uint8), 1))
    else:
        array.array.dtype = np.dtype(np.int8)
        data = DataBin(c=array)
    with pytest.raises((TypeError, ValueError), match="register|uint8"):
        native_runtime_gate_observation(SamplerPubResult(data), request, shots=4)


def test_native_byte_budget_uses_linear_register_storage_and_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packed and temporary native arrays use a byte budget without 2**qubits."""
    circuit = QuantumCircuit(1)
    circuit.add_register(ClassicalRegister(9, "alpha"), ClassicalRegister(1, "beta"))
    request = qiskit_workload_semantics(circuit, "native-program")
    # Four rows: P=12 packed register bytes, B=10 bits and C=8 joint packed bytes.
    required = 10 * 12 + 4 * 10 + 9 * 8
    monkeypatch.setenv("SCPN_MAX_DENSE_GIB", str((required - 1) / 1024**3))
    with pytest.raises(MemoryError, match="budget"):
        require_runtime_sample_buffers(request, 4)
    monkeypatch.setenv("SCPN_MAX_DENSE_GIB", str(required / 1024**3))
    assert require_runtime_sample_buffers(request, 4) == required
    with pytest.raises(MemoryError, match="budget"):
        require_runtime_sample_buffers(request, sys.maxsize)
    for shots in (0, True):
        with pytest.raises(ValueError, match="positive shots"):
            require_runtime_sample_buffers(request, shots)
    with pytest.raises(ValueError, match="classical registers"):
        require_runtime_sample_buffers(
            qiskit_workload_semantics(QuantumCircuit(1), "no-registers"), 4
        )
