# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native provider SDK conformance
"""Exercise native SDK codecs through original public workload and HAL surfaces."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import math
import os
from dataclasses import replace
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm3, qpy
from qiskit.circuit import Parameter
from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult
from qiskit_aer import AerSimulator

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_qiskit import (
    QiskitAerHALAdapter,
    QiskitRuntimeHALAdapter,
    qiskit_circuit_to_workload,
)
from scpn_quantum_control.hardware.provider_semantics import (
    GateModelObservation,
    WorkloadSemantics,
)
from scpn_quantum_control.hardware.provider_smoke import (
    main as provider_smoke_main,
)
from scpn_quantum_control.hardware.provider_smoke import (
    provider_optional_dependency_matrix,
)

_SDK_COHORTS = {
    "ibm": "qiskit-ibm-runtime",
    "braket": "amazon-braket-sdk",
    "azure": "azure-quantum",
    "qbraid": "qbraid",
    "strangeworks": "strangeworks",
    "cirq": "cirq-core",
    "dwave": "dwave-cloud-client",
    "iqm": "iqm-client",
    "oqc": "oqc-qcaas-client",
    "pasqal": "pulser-core",
    "quandela": "perceval-quandela",
    "quantinuum": "pytket-quantinuum",
    "quera": "bloqade",
    "rigetti": "pyquil",
    "pennylane": "pennylane",
}


def _parameterised_source() -> tuple[QuantumCircuit, Parameter, QuantumWorkload]:
    """Build a shared-parameter source whose permuted readout is analytically 10.

    Returns
    -------
    tuple[QuantumCircuit, Parameter, QuantumWorkload]
        Original two-qubit circuit, shared native parameter and encoded QPY.
        Binding theta to pi prepares q0=1, q1=0 and requests sixteen samples.

    """
    theta = Parameter("theta")
    circuit = QuantumCircuit(2, 2)
    circuit.ry(theta, 0)
    circuit.cx(0, 1)
    circuit.ry(theta, 1)
    circuit.measure(0, 1)
    circuit.measure(1, 0)
    workload = qiskit_circuit_to_workload(
        circuit,
        workload_id="sdk-shared-parameter-permuted-readout",
        shots=16,
        capture_semantics=True,
        parameter_bindings={theta: math.pi},
    )
    return circuit, theta, workload


def _qiskit_native_parameter_layout_and_runtime_codec() -> None:
    """Preserve shared UUIDs, permuted measurement and shots through native codecs."""
    from qiskit_ibm_runtime import RuntimeDecoder, RuntimeEncoder

    circuit, theta, workload = _parameterised_source()
    decoded = qpy.load(io.BytesIO(base64.b64decode(workload.program)))[0]
    assert decoded == circuit
    assert tuple(decoded.parameters) == (theta,)
    assert decoded.data[0].operation.params[0] == theta
    assert decoded.data[2].operation.params[0] == theta
    assert isinstance(workload.semantics, WorkloadSemantics)
    assert workload.semantics.measurement_map == ((0, 1), (1, 0))
    assert workload.shots == 16
    wire = json.loads(
        json.dumps({"pubs": [(decoded, [math.pi], 16)]}, cls=RuntimeEncoder),
        cls=RuntimeDecoder,
    )
    assert wire["pubs"][0][0] == circuit
    assert wire["pubs"][0][1] == [math.pi]
    assert wire["pubs"][0][2] == 16
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = AerSimulator(max_parallel_threads=1, seed_simulator=7)
    hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer"), backend=backend))
    hal.register_backend(QiskitRuntimeHALAdapter(hal.profile("ibm_quantum"), backend=backend))
    for route in ("local_qiskit_aer", "ibm_quantum"):
        job = hal.submit(route, workload, approval_id="local-sdk-only")
        first = hal.result(job)
        assert first.counts == {"10": 16}, route
        assert first.shots == 16, route
        assert isinstance(first.provider_observation, GateModelObservation), route
        assert first.provider_observation.measurement_map == ((0, 1), (1, 0)), route
        assert hal.result(replace(job, status="submitted")) is first, route
        assert hal.cancel(job).status == "completed", route
    assert circuit == decoded
    assert circuit.parameters == {theta}


def test_missing_required_sdk_is_not_qualified(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An absent required Runtime import produces a named unavailable CLI row."""
    native_find_spec = importlib.util.find_spec

    def absent_runtime(name: str, package: str | None = None) -> Any:
        if name == "qiskit_ibm_runtime":
            return None
        return native_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", absent_runtime)
    assert provider_smoke_main(["--format", "json", "--backend", "ibm_quantum"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["backend_id"] == "ibm_quantum"
    assert rows[0]["available"] is False
    assert rows[0]["missing_imports"] == ["qiskit_ibm_runtime"]
    assert (
        provider_smoke_main(["--format", "json", "--backend", "ibm_quantum", "--require-all"]) == 1
    )
    required_rows = json.loads(capsys.readouterr().out)
    assert required_rows == rows


def _braket_native_symbols_ir_and_local_hal_roundtrip() -> None:
    """Conserve shared native symbols and requested samples through Braket IR."""
    from braket.circuits import Circuit, FreeParameter
    from braket.ir.openqasm import Program

    from scpn_quantum_control.hardware.hal_braket import (
        BraketLocalHALAdapter,
        braket_circuit_to_workload,
    )

    theta = FreeParameter("theta")
    circuit = Circuit().rx(0, theta).cnot(0, 1).rx(1, theta).measure(0).measure(1)
    workload = braket_circuit_to_workload(
        circuit,
        workload_id="braket-shared-native-symbol",
        shots=16,
        capture_semantics=True,
        parameter_bindings={"theta": math.pi},
    )
    native_program = Program(source=workload.program, inputs={"theta": math.pi})
    decoded = Program.parse_raw(native_program.json())
    assert decoded.source == workload.program
    assert decoded.inputs == {"theta": math.pi}
    assert isinstance(workload.semantics, WorkloadSemantics)
    assert workload.semantics.measurement_map == ((0, 0), (1, 1))
    assert workload.semantics.parameter_values == (("braket:theta", math.pi),)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(BraketLocalHALAdapter(hal.profile("local_braket_sv")))
    job = hal.submit("local_braket_sv", workload)
    result = hal.result(job)
    assert result.counts == {"10": 16}
    assert result.shots == 16
    assert isinstance(result.provider_observation, GateModelObservation)
    assert result.provider_observation.measurement_map == ((0, 0), (1, 1))
    assert hal.result(job) is result
    assert hal.cancel(job).status == "completed"
    assert decoded.source == workload.program
    malformed = replace(workload, program="not-a-Braket-program", semantics=None)
    with pytest.raises(ValueError, match="could not be decoded"):
        hal.submit("local_braket_sv", malformed)
    assert hal.result(job) is result


def _cirq_native_json_and_hal_roundtrip() -> None:
    """Retain a symbolic Cirq circuit and permuted native measurement order."""
    cirq = import_module("cirq")
    sympy = import_module("sympy")
    from scpn_quantum_control.hardware.hal_cirq import CirqLocalHALAdapter, cirq_circuit_workload

    q0, q1 = cirq.LineQubit.range(2)
    theta = sympy.Symbol("theta")
    circuit = cirq.Circuit(
        cirq.ry(theta)(q0), cirq.CNOT(q0, q1), cirq.ry(theta)(q1), cirq.measure(q1, q0, key="m")
    )
    source = cirq.to_json(circuit)
    decoded = cirq.read_json(json_text=source)
    assert decoded == circuit
    assert cirq.parameter_names(decoded) == {"theta"}
    assert tuple(list(decoded.all_operations())[-1].qubits) == (q1, q0)
    workload = cirq_circuit_workload(source, workload_id="cirq-native-codec", n_qubits=2, shots=16)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        CirqLocalHALAdapter(
            hal.profile("local_cirq"),
            simulator=cirq.Simulator(seed=7),
            circuit_factory=lambda program: cirq.resolve_parameters(
                cirq.read_json(json_text=program), {"theta": math.pi}
            ),
        )
    )
    job = hal.submit("local_cirq", workload)
    result = hal.result(job)
    assert result.counts == {"01": 16}
    assert result.shots == 16
    assert hal.result(job) is result
    assert hal.cancel(job).status == "completed"
    assert workload.program == source


def _pennylane_native_export_and_hal_roundtrip() -> None:
    """Decode PennyLane's actual OpenQASM export and execute original native HAL work."""
    qml = import_module("pennylane")
    from qiskit import qasm2

    from scpn_quantum_control.hardware.hal_pennylane import (
        PennyLaneDeviceHALAdapter,
        pennylane_gate_workload,
    )

    tape = qml.tape.QuantumScript(
        [qml.RY(math.pi, 0), qml.CNOT([0, 1]), qml.RY(math.pi, 1)],
        [qml.counts(wires=[0, 1])],
        shots=16,
    )
    exported = qml.to_openqasm(tape, precision=17)
    decoded = qasm2.loads(exported)
    assert decoded.num_qubits == decoded.num_clbits == 2
    assert [item.operation.name for item in decoded.data] == [
        "ry",
        "cx",
        "ry",
        "measure",
        "measure",
    ]
    assert decoded.data[0].operation.params == [math.pi]
    assert decoded.data[2].operation.params == [math.pi]
    workload = pennylane_gate_workload(
        [
            {"gate": "ry", "wires": [0], "params": [math.pi]},
            {"gate": "cx", "wires": [0, 1]},
            {"gate": "ry", "wires": [1], "params": [math.pi]},
        ],
        workload_id="pennylane-native-codec",
        n_qubits=2,
        shots=16,
    )
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        PennyLaneDeviceHALAdapter(hal.profile("local_pennylane"), device_kwargs={"seed": 7})
    )
    job = hal.submit("local_pennylane", workload)
    result = hal.result(job)
    assert result.counts == {"10": 16}
    assert result.shots == 16
    assert hal.result(job) is result


def _quantinuum_native_circuit_and_result_codec() -> None:
    """Retain symbolic tket operations and native ordered result rows without a client."""
    tket = import_module("pytket")
    sympy = import_module("sympy")
    result_module = import_module("pytket.backends.backendresult")
    outcomes = import_module("pytket.utils.outcomearray")
    import_module("pytket.extensions.quantinuum")
    from scpn_quantum_control.hardware.hal_quantinuum import quantinuum_tket_workload

    theta = sympy.Symbol("theta")
    circuit = tket.Circuit(2, 2).Ry(theta, 0).CX(0, 1).Ry(theta, 1).Measure(0, 1).Measure(1, 0)
    workload = quantinuum_tket_workload(
        circuit, workload_id="quantinuum-native-codec", n_qubits=2, shots=16
    )
    decoded = tket.Circuit.from_dict(json.loads(workload.program)["circuit"])
    assert decoded == circuit
    assert decoded.free_symbols() == {theta}
    measurements = [
        command.args
        for command in decoded.get_commands()
        if command.op.type == tket.OpType.Measure
    ]
    assert len(measurements) == 2
    assert [tket.Qubit(0), tket.Bit(1)] in measurements
    assert [tket.Qubit(1), tket.Bit(0)] in measurements
    native = result_module.BackendResult(
        c_bits=[tket.Bit(1), tket.Bit(0)],
        shots=outcomes.OutcomeArray.from_readouts(np.tile([[1, 0]], (16, 1))),
    )
    restored = result_module.BackendResult.from_dict(native.to_dict())
    assert np.array_equal(
        restored.get_shots([tket.Bit(1), tket.Bit(0)]), np.tile([[1, 0]], (16, 1))
    )
    assert workload.shots == 16


def _rigetti_native_quil_roundtrip() -> None:
    """Retain parameter memory, permuted measurements and bounded shot loop in pyQuil."""
    pyquil = import_module("pyquil")
    gates = import_module("pyquil.gates")
    from scpn_quantum_control.hardware.hal_rigetti import rigetti_quil_workload

    native = pyquil.Program()
    theta = native.declare("theta", "REAL", 1)
    ro = native.declare("ro", "BIT", 2)
    native += gates.RY(theta[0], 0)
    native += gates.CNOT(0, 1)
    native += gates.RY(theta[0], 1)
    native += gates.MEASURE(0, ro[1])
    native += gates.MEASURE(1, ro[0])
    workload = rigetti_quil_workload(
        native.out(), workload_id="rigetti-native-codec", n_qubits=2, shots=16
    )
    decoded = pyquil.Program(workload.program)
    assert decoded.out() == native.out()
    assert decoded.declarations["theta"].memory_type == "REAL"
    assert decoded.declarations["ro"].memory_size == 2
    assert decoded.instructions[-2].out() == "MEASURE 0 ro[1]"
    assert decoded.instructions[-1].out() == "MEASURE 1 ro[0]"
    decoded.wrap_in_numshots_loop(workload.shots)
    assert decoded.num_shots == 16
    assert workload.program == native.out()


def _bound_native_qasm() -> tuple[str, QuantumCircuit]:
    """Export the fully bound original permuted-measurement fixture.

    Returns
    -------
    tuple[str, QuantumCircuit]
        Native OpenQASM 3 and the two-qubit source, with both theta uses bound
        to pi and final readout ordered as q0->c1, q1->c0.

    """
    circuit, theta, _ = _parameterised_source()
    bound = circuit.assign_parameters({theta: math.pi})
    return qasm3.dumps(bound), bound


def _azure_native_job_request_codec() -> None:
    """Retain the original program, target and shot request in real Azure SDK models."""
    models = import_module("azure.quantum.job.job")
    from scpn_quantum_control.hardware.hal_azure import azure_openqasm3_to_workload

    source, circuit = _bound_native_qasm()
    workload = azure_openqasm3_to_workload(
        source, workload_id="azure-native-codec", n_qubits=2, shots=16
    )
    native = models.JobDetails(
        name=workload.workload_id,
        id="00000000-0000-4000-8000-000000000001",
        provider_id="quantinuum",
        target="quantinuum.sim.h1-1e",
        container_uri="https://example.invalid/local-codec-container",
        input_params={"shots": workload.shots},
        metadata={"program": workload.program},
    )
    restored = models.JobDetails(json.loads(json.dumps(native.as_dict())))
    assert restored.id == "00000000-0000-4000-8000-000000000001"
    assert restored.name == workload.workload_id
    assert restored.provider_id == "quantinuum"
    assert restored.target == "quantinuum.sim.h1-1e"
    assert restored.input_params == {"shots": 16}
    assert restored.metadata == {"program": source}
    assert qasm3.loads(restored.metadata["program"]) == circuit
    assert workload.program == source


def _qbraid_native_program_codec() -> None:
    """Exercise qBraid's real OpenQASM wrapper and typed runtime Program codec."""
    qasm_module = import_module("qbraid.programs.gate_model.qasm3")
    from scpn_quantum_control.hardware.hal_qbraid import qbraid_program_to_workload

    source, circuit = _bound_native_qasm()
    workload = qbraid_program_to_workload(
        source, workload_id="qbraid-native-codec", ir_format="openqasm3", n_qubits=2, shots=16
    )
    native = qasm_module.OpenQasm3Program(workload.program)
    native.validate()
    assert native.num_qubits == native.num_clbits == 2
    wire = native.serialize()
    restored = type(wire).model_validate_json(wire.model_dump_json())
    assert restored.format == "qasm3"
    assert qasm3.loads(restored.data) == circuit
    assert workload.shots == 16
    assert workload.program == source


def _oqc_native_task_and_compiler_config_codec() -> None:
    """Round-trip actual OQC task/config serialisation without constructing a client."""
    client = import_module("qcaas_client.client")
    from scpn_quantum_control.hardware.hal_oqc import oqc_openqasm3_workload

    source, circuit = _bound_native_qasm()
    workload = oqc_openqasm3_workload(source, workload_id="oqc-native-codec", n_qubits=2, shots=16)
    config = client.CompilerConfig(
        repeats=workload.shots, results_format=client.QuantumResultsFormat().binary_count()
    )
    native = client.QPUTask(
        workload.program, config=config, task_id="local-native-codec", qpu_id="local-codec-target"
    )
    wire = json.loads(json.dumps(native.to_json()))
    restored_config = client.CompilerConfig.create_from_json(wire["config"])
    restored = client.QPUTask(
        wire["program"], config=restored_config, task_id=wire["task_id"], qpu_id=wire["qpu_id"]
    )
    assert restored.config.repeats == 16
    assert restored.qpu_id == "local-codec-target"
    assert restored.task_id == "local-native-codec"
    assert restored.to_json() == native.to_json()
    assert qasm3.loads(restored.program) == circuit
    assert workload.program == source


def _iqm_native_circuit_request_codec() -> None:
    """Preserve native IQM loci, radians, final measurement order and shots in request JSON."""
    pulse = import_module("iqm.pulse")
    models = import_module("iqm.station_control.interface.models.circuit")
    from scpn_quantum_control.hardware.hal_iqm import iqm_qiskit_workload

    theta = Parameter("theta")
    circuit = QuantumCircuit(2, 2)
    circuit.rx(theta, 0)
    circuit.cz(0, 1)
    circuit.rx(theta, 1)
    circuit.measure([1, 0], [0, 1])
    workload = iqm_qiskit_workload(
        circuit,
        workload_id="iqm-native-codec",
        shots=16,
        capture_semantics=True,
        parameter_bindings={theta: math.pi},
    )
    native = pulse.Circuit(
        name=workload.workload_id,
        instructions=(
            pulse.CircuitOperation(
                name="prx", locus=("QB1",), args={"angle": math.pi, "phase": 0.0}
            ),
            pulse.CircuitOperation(name="cz", locus=("QB1", "QB2"), args={}),
            pulse.CircuitOperation(
                name="prx", locus=("QB2",), args={"angle": math.pi, "phase": 0.0}
            ),
            pulse.CircuitOperation(name="measure", locus=("QB2", "QB1"), args={"key": "m"}),
        ),
    )
    request = models.PostJobsRequest(circuits=[native], shots=workload.shots)
    restored = models.PostJobsRequest.model_validate_json(request.model_dump_json())
    assert restored.shots == 16
    assert len(restored.circuits) == 1
    decoded = restored.circuits[0]
    assert isinstance(decoded, pulse.Circuit)
    assert decoded.name == workload.workload_id
    assert tuple(op.name for op in decoded.instructions) == ("prx", "cz", "prx", "measure")
    assert decoded.instructions[0].args == {"angle": math.pi, "phase": 0.0}
    assert decoded.instructions[1].locus == ("QB1", "QB2")
    assert decoded.instructions[2].args == {"angle": math.pi, "phase": 0.0}
    assert decoded.instructions[3].locus == ("QB2", "QB1")
    assert decoded.instructions[3].args == {"key": "m"}
    assert qpy.load(io.BytesIO(base64.b64decode(workload.program)))[0] == circuit
    with pytest.raises(ValueError):
        models.PostJobsRequest(circuits=[native], shots=0)
    assert native.instructions[3].locus == ("QB2", "QB1")


def _dwave_native_bqm_and_sampleset_codec() -> None:
    """Retain SPIN domain, coefficients, native variable order and sixteen occurrences."""
    dimod = import_module("dimod")
    coders = import_module("dwave.cloud.coders")
    from scpn_quantum_control.hardware.hal_dwave import dwave_bqm_workload

    workload = dwave_bqm_workload(
        linear={"left": -1.0, "right": 2.0},
        quadratic={("left", "right"): 0.5},
        offset=0.25,
        vartype="SPIN",
        workload_id="dwave-native-codec",
        n_variables=2,
        reads=16,
        capture_semantics=True,
    )
    plan = json.loads(workload.program)
    native = dimod.BinaryQuadraticModel(
        plan["linear"],
        {(term["u"], term["v"]): term["bias"] for term in plan["quadratic"]},
        plan["offset"],
        dimod.SPIN,
    )
    with native.to_file() as file:
        binary = file.read()
    restored = dimod.BinaryQuadraticModel.from_file(io.BytesIO(binary))
    assert restored == native
    assert restored.vartype is dimod.SPIN
    assert restored.energy({"left": -1, "right": 1}) == 2.75
    assert restored.energy({"left": 1, "right": -1}) == -3.25
    samples = dimod.SampleSet.from_samples(
        [{"right": 1, "left": -1}, {"right": -1, "left": 1}],
        vartype=dimod.SPIN,
        energy=[2.75, -3.25],
        num_occurrences=[7, 9],
        sort_labels=False,
    )
    wire = {"type": "bqm", "answer": {"format": "bq", "data": samples.to_serializable()}}
    decoded = coders.decode_bq(json.loads(json.dumps(wire)))["sampleset"]
    assert tuple(decoded.variables) == ("right", "left")
    assert np.array_equal(decoded.record.sample, [[1, -1], [-1, 1]])
    assert np.array_equal(decoded.record.energy, [2.75, -3.25])
    assert np.array_equal(decoded.record.num_occurrences, [7, 9])
    assert int(decoded.record.num_occurrences.sum()) == workload.shots == 16
    with pytest.raises(ValueError, match="Incomplete answer"):
        coders.decode_bq({"type": "bqm", "answer": {"format": "bq"}})
    assert np.array_equal(decoded.record.num_occurrences, [7, 9])


def _quandela_native_circuit_and_occupation_codec() -> None:
    """Retain actual Perceval components and nonbinary occupations through vendor codecs."""
    pcvl = import_module("perceval")
    codec = import_module("perceval.serialization")
    from scpn_quantum_control.hardware.hal_quandela import quandela_perceval_workload

    plan = {
        "schema": "scpn.quandela.perceval.v1",
        "modes": 2,
        "input_state": [2, 0],
        "components": [
            {"type": "beam_splitter", "modes": [0, 1], "theta": 0.7853981633974483},
            {"type": "phase_shifter", "mode": 1, "phi": 0.25},
        ],
    }
    workload = quandela_perceval_workload(
        plan, workload_id="quandela-native-codec", n_modes=2, shots=16, capture_semantics=True
    )
    native = pcvl.Circuit(2).add((0, 1), pcvl.BS(theta=0.7853981633974483)).add(1, pcvl.PS(0.25))
    decoded = codec.deserialize(codec.serialize(native))
    assert decoded.m == 2
    assert np.array_equal(
        np.asarray(decoded.compute_unitary()), np.asarray(native.compute_unitary())
    )
    state = codec.deserialize(codec.serialize(pcvl.BasicState(plan["input_state"])))
    assert tuple(state) == (2, 0)
    assert state.m == 2
    assert workload.shots == 16
    assert json.loads(workload.program) == plan


def _pasqal_native_sequence_codec() -> None:
    """Preserve ordered Pulser sites, finite pulse parameters and the native nanosecond duration."""
    pulser = import_module("pulser")
    devices = import_module("pulser.devices")
    from scpn_quantum_control.hardware.hal_pasqal import pulser_sequence_workload

    plan = {
        "schema": "pulser_sequence_plan_v1",
        "duration": 1.0,
        "register": {"site-b": [0.0, 0.0], "site-a": [5.0, 0.0]},
        "rydberg_channel": "rydberg_global",
        "rabi_envelope": [
            {"time": 0.0, "amplitude": 1.0, "phase": 0.1},
            {"time": 1.0, "amplitude": 1.0, "phase": 0.1},
        ],
        "local_detunings": [],
        "interaction_terms": [],
        "fim_feedback_terms": [],
    }
    workload = pulser_sequence_workload(
        plan, workload_id="pasqal-native-codec", n_qubits=2, shots=16, capture_semantics=True
    )
    native = pulser.Sequence(
        pulser.Register({"site-b": (0.0, 0.0), "site-a": (5.0, 0.0)}), devices.MockDevice
    )
    native.declare_channel("drive", "rydberg_global")
    native.add(pulser.Pulse.ConstantPulse(1000, 1.0, 0.25, 0.1), "drive")
    native.measure("ground-rydberg")
    source = native.to_abstract_repr()
    decoded = pulser.Sequence.from_abstract_repr(source)
    assert decoded.register.qubit_ids == ("site-b", "site-a")
    assert np.array_equal(decoded.register.qubits["site-a"], [5.0, 0.0])
    assert decoded.get_duration() == 1000
    assert decoded.get_measurement_basis() == "ground-rydberg"
    assert json.loads(decoded.to_abstract_repr()) == json.loads(source)
    assert workload.shots == 16
    assert json.loads(workload.program) == plan


def _quera_native_analog_routine_codec() -> None:
    """Retain Bloqade's native site geometry and waveforms in its genuine routine serializer."""
    analog = import_module("bloqade.analog")
    from scpn_quantum_control.hardware.hal_quera_bloqade import bloqade_ahs_workload

    plan = {
        "schema": "bloqade_ahs_plan_v1",
        "duration": 2.0,
        "atoms": [{"index": 0, "position": [0.0, 0.0]}, {"index": 1, "position": [5.0, 0.0]}],
        "rabi_amplitude_piecewise_linear": [[0.0, 1.0], [2.0, 1.0]],
        "rabi_phase_piecewise_linear": [[0.0, 0.0], [2.0, 0.0]],
        "local_detunings": [],
        "rydberg_interactions": [],
        "fim_feedback_terms": [],
    }
    workload = bloqade_ahs_workload(
        plan, workload_id="quera-native-codec", n_qubits=2, shots=16, capture_semantics=True
    )
    native = analog.rydberg_h(
        [(0.0, 0.0), (5.0, 0.0)],
        amplitude=analog.constant(duration=2.0, value=1.0),
        phase=analog.constant(duration=2.0, value=0.0),
    )
    encoded = analog.dumps(native)
    decoded = analog.loads(encoded)
    assert decoded.circuit == native.circuit
    assert json.loads(analog.dumps(decoded)) == json.loads(encoded)
    assert workload.shots == 16
    assert json.loads(workload.program) == plan


def _strangeworks_native_job_codec() -> None:
    """Decode real Strangeworks job fields while retaining the original workload and samples."""
    jobs = import_module("strangeworks.core.client.jobs")
    from scpn_quantum_control.hardware.hal_strangeworks import strangeworks_program_to_workload

    source, circuit = _bound_native_qasm()
    workload = strangeworks_program_to_workload(
        source,
        workload_id="strangeworks-native-codec",
        ir_format="openqasm3",
        n_qubits=2,
        shots=16,
    )
    native = jobs.Job.from_dict(
        {
            "slug": workload.workload_id,
            "id": "local-sdk-result-no-submit",
            "jobData": {
                "program": workload.program,
                "shots": workload.shots,
                "counts": {"10": 16},
            },
        }
    )
    decoded = jobs.Job.model_validate_json(native.model_dump_json(by_alias=True))
    assert decoded.job_id == "local-sdk-result-no-submit"
    assert decoded.slug == workload.workload_id
    assert decoded.job_data == {"program": source, "shots": 16, "counts": {"10": 16}}
    assert qasm3.loads(decoded.job_data["program"]) == circuit
    assert workload.program == source


def _minimal_install_native_execution_and_optional_absence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise an actual installed base wheel and refuse absent optional SDK cohorts.

    Parameters
    ----------
    capsys
        Native pytest capture for the production metadata CLI output.

    """
    import scpn_quantum_control

    source_owner = Path(__file__).resolve().parents[1] / "src" / "scpn_quantum_control"
    assert Path(scpn_quantum_control.__file__).resolve().parent != source_owner
    assert version("scpn-quantum-control") == scpn_quantum_control.__version__
    for package in _SDK_COHORTS.values():
        with pytest.raises(PackageNotFoundError):
            version(package)
    rows = provider_optional_dependency_matrix()
    assert rows
    for row in rows:
        assert row.available == (row.sdk_package in {"python", "requests", "qiskit-aer"}), (
            row.backend_id
        )
    assert provider_smoke_main(["--format", "json", "--backend", "ibm_quantum"]) == 0
    unavailable = json.loads(capsys.readouterr().out)
    assert len(unavailable) == 1 and unavailable[0]["backend_id"] == "ibm_quantum"
    assert unavailable[0]["available"] is False
    assert unavailable[0]["missing_imports"] == ["qiskit_ibm_runtime"]
    assert (
        provider_smoke_main(["--format", "json", "--backend", "ibm_quantum", "--require-all"]) == 1
    )
    assert json.loads(capsys.readouterr().out) == unavailable
    with pytest.raises(ModuleNotFoundError, match="qiskit_ibm_runtime"):
        import_module("qiskit_ibm_runtime")
    _, _, workload = _parameterised_source()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitAerHALAdapter(
            hal.profile("local_qiskit_aer"),
            backend=AerSimulator(max_parallel_threads=1, seed_simulator=7),
        )
    )
    job = hal.submit("local_qiskit_aer", workload)
    result = hal.result(job)
    assert result.counts == {"10": 16}
    assert result.shots == 16
    assert hal.result(job) is result


@pytest.mark.parametrize(
    "cohort",
    [os.environ["SCPN_PROVIDER_SDK_COHORT"]]
    if "SCPN_PROVIDER_SDK_COHORT" in os.environ
    else ["ibm", "braket"],
)
def test_native_sdk_cohort_roundtrip(
    cohort: str,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A required SDK cohort must execute its native public codecs and original workload path."""
    if cohort == "minimal":
        _minimal_install_native_execution_and_optional_absence(capsys)
        return
    isolated_config = tmp_path / "absent-strangeworks-config.toml"
    if cohort == "strangeworks":
        for key in list(os.environ):
            if key.startswith("STRANGEWORKS_"):
                monkeypatch.delenv(key)
        monkeypatch.setenv("STRANGEWORKS_CONFIG_PATH", str(isolated_config))
    assert (
        provider_smoke_main(
            ["--format", "json", "--sdk-package", _SDK_COHORTS[cohort], "--require-all"]
        )
        == 0
    ), cohort
    rows = json.loads(capsys.readouterr().out)
    assert rows and all(row["available"] for row in rows), cohort
    qualifiers = {
        "ibm": _qiskit_native_parameter_layout_and_runtime_codec,
        "braket": _braket_native_symbols_ir_and_local_hal_roundtrip,
        "cirq": _cirq_native_json_and_hal_roundtrip,
        "pennylane": _pennylane_native_export_and_hal_roundtrip,
        "quantinuum": _quantinuum_native_circuit_and_result_codec,
        "rigetti": _rigetti_native_quil_roundtrip,
        "azure": _azure_native_job_request_codec,
        "qbraid": _qbraid_native_program_codec,
        "oqc": _oqc_native_task_and_compiler_config_codec,
        "iqm": _iqm_native_circuit_request_codec,
        "dwave": _dwave_native_bqm_and_sampleset_codec,
        "quandela": _quandela_native_circuit_and_occupation_codec,
        "pasqal": _pasqal_native_sequence_codec,
        "quera": _quera_native_analog_routine_codec,
        "strangeworks": _strangeworks_native_job_codec,
    }
    qualifiers[cohort]()
    assert not isolated_config.exists()


@pytest.mark.parametrize("fault", ["renamed_register", "width", "shots", "broadcast"])
def test_ibm_incompatible_native_result_preserves_prior_evidence(fault: str) -> None:
    """Reject native SDK channel drift without replacing an earlier valid result."""
    from qiskit_ibm_runtime import RuntimeDecoder, RuntimeEncoder, SamplerV2

    _, _, workload = _parameterised_source()
    sample_shape = (2, 16, 1) if fault == "broadcast" else (15 if fault == "shots" else 16, 1)
    bits = BitArray(
        np.full(sample_shape, 1 if fault == "width" else 2, dtype=np.uint8),
        1 if fault == "width" else 2,
    )
    data = DataBin(**{"readout" if fault == "renamed_register" else "c": bits})
    native_result: PrimitiveResult[SamplerPubResult] = PrimitiveResult(
        [SamplerPubResult(data, metadata={})]
    )
    decoded_result = json.loads(json.dumps(native_result, cls=RuntimeEncoder), cls=RuntimeDecoder)
    dispatches: list[str] = []

    class FaultJob:
        def job_id(self) -> str:
            return f"local-native-result-fault-{fault}"

        def result(self, timeout: float | None = None) -> Any:
            assert timeout == 600.0
            return decoded_result

    class SamplerBoundary:
        def __init__(self, *, mode: AerSimulator) -> None:
            self.native = SamplerV2(mode=mode)
            self.options = self.native.options

        def run(self, pubs: list[QuantumCircuit]) -> Any:
            dispatches.append(workload.workload_id)
            if len(dispatches) == 1:
                return self.native.run(pubs)
            return FaultJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=AerSimulator(max_parallel_threads=1, seed_simulator=7),
            sampler_factory=SamplerBoundary,
        )
    )
    original = hal.submit("ibm_quantum", workload, approval_id="local-sdk-only")
    good = hal.result(original)
    assert good.counts == {"10": 16}
    drifted = hal.submit(
        "ibm_quantum",
        replace(workload, workload_id=f"sdk-drift-{fault}"),
        approval_id="controlled-local-result-fault",
    )
    with pytest.raises((ValueError, TypeError)):
        hal.result(drifted)
    assert len(dispatches) == 2
    assert hal.result(original) is good
    assert good.counts == {"10": 16}
    malformed = replace(workload, program="not-a-QPY-source", semantics=None)
    with pytest.raises(ValueError, match="qiskit_qpy"):
        hal.submit("ibm_quantum", malformed, approval_id="local-sdk-only")
    assert len(dispatches) == 2
    assert hal.result(original) is good
