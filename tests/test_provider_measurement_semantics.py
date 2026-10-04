# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — public provider measurement semantics regressions
"""Exercise native partial measurements and exact-target refusal through HAL."""

from __future__ import annotations

from dataclasses import replace
from math import pi
from typing import NoReturn

import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer
from scpn_quantum_control.hardware.hal_iqm import IQMHALAdapter, iqm_qiskit_workload
from scpn_quantum_control.hardware.hal_qiskit import (
    QiskitAerHALAdapter,
    QiskitRuntimeHALAdapter,
    qiskit_circuit_to_workload,
)
from scpn_quantum_control.hardware.iqm_backend import IQMTargetCompilationError
from scpn_quantum_control.hardware.provider_semantics import (
    GateModelObservation,
    WorkloadSemantics,
)


def _partial_permutation(*, separate_registers: bool = False) -> QuantumCircuit:
    """Prepare q2=1,q0=0 and measure q2->c0,q0->c1 with native Qiskit."""
    circuit = QuantumCircuit(3)
    if separate_registers:
        circuit.add_register(ClassicalRegister(1, "alpha"), ClassicalRegister(1, "beta"))
    else:
        circuit.add_register(ClassicalRegister(2, "c"))
    circuit.x(2)
    circuit.measure(2, 0)
    circuit.measure(0, 1)
    return circuit


@pytest.mark.parametrize("route", ["aer", "iqm", "runtime"])
@pytest.mark.parametrize("separate_registers", [False, True])
def test_provider_measurement_semantics_01(route: str, separate_registers: bool) -> None:
    """Preserve analytic marginals, exact shots and declared q->c wiring."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    native = AerSimulator(max_parallel_threads=1, seed_simulator=7)
    if route == "aer":
        backend_id = "local_qiskit_aer"
        hal.register_backend(QiskitAerHALAdapter(hal.profile(backend_id), backend=native))
    elif route == "iqm":
        backend_id = "iqm_cloud"
        hal.register_backend(IQMHALAdapter(hal.profile(backend_id), backend=native))
    else:
        backend_id = "ibm_quantum"
        hal.register_backend(QiskitRuntimeHALAdapter(hal.profile(backend_id), backend=native))
    circuit = _partial_permutation(separate_registers=separate_registers)
    workload = qiskit_circuit_to_workload(
        circuit, workload_id=f"partial_{route}", shots=32, capture_semantics=True
    )
    job = hal.submit(backend_id, workload, approval_id="offline-native-sdk-only")
    result = hal.result(job)

    assert result.shots == sum(result.counts.values()) == 32
    assert result.counts == {"01": 32}
    assert job.submission is not None
    assert isinstance(job.submission.request, WorkloadSemantics)
    assert job.submission.request.measurement_map == ((2, 0), (0, 1))
    assert isinstance(result.provider_observation, GateModelObservation)
    assert result.provider_observation.measurement_map == ((2, 0), (0, 1))
    # Qiskit count strings put c1 left and c0 right. The oracle is the prepared basis state.
    assert sum((-1 if key[-1] == "1" else 1) * n for key, n in result.counts.items()) == -32
    assert sum((-1 if key[-2] == "1" else 1) * n for key, n in result.counts.items()) == 32
    assert hal.result(job) is result


def test_iqm_failed_target_compilation_never_retries_without_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine one-qubit native Target cannot admit a three-qubit request."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    native = GenericBackendV2(1, basis_gates=["rz", "sx", "x"], noise_info=False, seed=7)
    calls: list[tuple[object, dict[str, object]]] = []

    def forbidden_run(circuits: object, **options: object) -> NoReturn:
        """Record any transport crossing without creating a provider job."""
        calls.append((circuits, options))
        raise AssertionError("invalid selected target reached provider run")

    monkeypatch.setattr(native, "run", forbidden_run)
    hal.register_backend(IQMHALAdapter(hal.profile("iqm_cloud"), backend=native))
    workload = iqm_qiskit_workload(_partial_permutation(), workload_id="too_wide", shots=8)
    with pytest.raises(IQMTargetCompilationError):
        hal.submit("iqm_cloud", workload, approval_id="offline-refusal-only")
    assert calls == []


def test_provider_measurement_semantics_02(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit shot capacity and unsupported IR refuse before any backend run."""
    built_in = HardwareAbstractionLayer.with_builtin_profiles().profile("local_qiskit_aer")
    profile = replace(built_in, capabilities=replace(built_in.capabilities, max_shots=4))
    hal = HardwareAbstractionLayer((profile,))
    native = AerSimulator(max_parallel_threads=1)
    calls: list[object] = []

    def forbidden_run(circuits: object, **options: object) -> NoReturn:
        """Detect a transport crossing independently of admission implementation."""
        calls.append(circuits)
        raise AssertionError("rejected request reached provider run")

    monkeypatch.setattr(native, "run", forbidden_run)
    hal.register_backend(QiskitAerHALAdapter(profile, backend=native))
    workload = qiskit_circuit_to_workload(_partial_permutation(), workload_id="cap", shots=5)
    with pytest.raises(ValueError, match="shots"):
        hal.submit(profile.backend_id, workload)
    with pytest.raises(ValueError, match="IR format"):
        hal.submit(profile.backend_id, replace(workload, ir_format="quil", shots=4))
    assert calls == []


def test_provider_measurement_semantics_03(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing a native target cannot submit the previously admitted payload."""
    workload = iqm_qiskit_workload(_partial_permutation(), workload_id="target_change", shots=8)
    good = HardwareAbstractionLayer.with_builtin_profiles()
    native = AerSimulator(max_parallel_threads=1, seed_simulator=7)
    good.register_backend(IQMHALAdapter(good.profile("iqm_cloud"), backend=native))
    original = good.submit("iqm_cloud", workload, approval_id="offline-native-only")
    evidence = good.result(original)
    assert evidence.counts == {"01": 8}

    changed = HardwareAbstractionLayer.with_builtin_profiles()
    invalid = GenericBackendV2(1, basis_gates=["rz", "sx", "x"], noise_info=False, seed=8)
    calls: list[object] = []

    def forbidden_run(circuits: object, **options: object) -> NoReturn:
        """Detect stale-payload execution on the newly selected native target."""
        calls.append(circuits)
        raise AssertionError("changed target accepted incompatible compiled payload")

    monkeypatch.setattr(invalid, "run", forbidden_run)
    changed.register_backend(IQMHALAdapter(changed.profile("iqm_cloud"), backend=invalid))
    with pytest.raises(IQMTargetCompilationError):
        changed.submit("iqm_cloud", workload, approval_id="offline-refusal-only")
    assert calls == []
    assert good.result(original) is evidence
    assert evidence.counts == {"01": 8}


@pytest.mark.parametrize("route", ["aer", "iqm", "runtime"])
def test_shared_parameter_preserves_original_source_and_executes_one_binding(route: str) -> None:
    """One native theta shared by three instructions remains one parameter identity."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(3, 2)
    circuit.ry(theta, 2)
    circuit.ry(theta, 2)
    circuit.rz(theta, 0)
    circuit.measure(2, 0)
    circuit.measure(0, 1)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    native = AerSimulator(max_parallel_threads=1, seed_simulator=7)
    if route == "aer":
        backend_id = "local_qiskit_aer"
        hal.register_backend(QiskitAerHALAdapter(hal.profile(backend_id), backend=native))
    elif route == "iqm":
        backend_id = "iqm_cloud"
        hal.register_backend(IQMHALAdapter(hal.profile(backend_id), backend=native))
    else:
        backend_id = "ibm_quantum"
        hal.register_backend(QiskitRuntimeHALAdapter(hal.profile(backend_id), backend=native))
    workload = qiskit_circuit_to_workload(
        circuit,
        workload_id=f"shared_{route}",
        shots=32,
        capture_semantics=True,
        parameter_bindings={theta: pi / 2},
    )
    original_program = workload.program
    job = hal.submit(backend_id, workload, approval_id="offline-native-only")
    result = hal.result(job)
    assert result.counts == {"01": 32}  # RY(pi/2) RY(pi/2) flips q2; RZ leaves q0=0.
    assert workload.program == original_program
    assert job.submission is not None
    assert job.submission.original_program == original_program
    assert isinstance(job.submission.request, WorkloadSemantics)
    assert job.submission.request.parameters == (
        ("theta", str(theta.uuid), ((0, 0), (1, 0), (2, 0))),
    )
    assert job.submission.request.parameter_values == ((str(theta.uuid), pi / 2),)


@pytest.mark.parametrize("cloud_profile", [False, True])
def test_braket_preserves_native_partial_measurement_order(cloud_profile: bool) -> None:
    """Native Braket output keeps explicit partial wire order through local/broker HAL."""
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator

    from scpn_quantum_control.hardware.hal_braket import (
        BraketAwsHALAdapter,
        BraketLocalHALAdapter,
        braket_circuit_to_workload,
    )

    circuit = Circuit().i(0).i(1).x(2).measure([2, 0])
    workload = braket_circuit_to_workload(
        circuit, workload_id="braket_partial", shots=16, capture_semantics=True
    )
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    native = LocalSimulator("braket_sv")
    backend_id = "aws_braket_ionq" if cloud_profile else "local_braket_sv"
    if cloud_profile:
        hal.register_backend(BraketAwsHALAdapter(hal.profile(backend_id), device=native))
    else:
        hal.register_backend(BraketLocalHALAdapter(hal.profile(backend_id), device=native))
    job = hal.submit(backend_id, workload, approval_id="offline-native-broker-shape-only")
    result = hal.result(job)
    assert result.counts == {"10": 16}  # Native Braket places b0 on the left.
    assert result.shots == 16
    assert job.submission is not None
    assert isinstance(job.submission.request, WorkloadSemantics)
    assert job.submission.request.measurement_map == ((2, 0), (0, 1))
    assert job.submission.request.count_bit_order == "classical_lsb_left"
    assert isinstance(result.provider_observation, GateModelObservation)
    assert result.provider_observation.raw_counts == {"10": 16}


@pytest.mark.parametrize("cloud_profile", [False, True])
def test_braket_shared_symbol_binds_once_and_preserves_original_source(
    cloud_profile: bool,
) -> None:
    """Two native RY(theta) gates share one symbol and analytically flip q2."""
    from braket.circuits import Circuit, FreeParameter
    from braket.devices import LocalSimulator

    from scpn_quantum_control.hardware.hal_braket import (
        BraketAwsHALAdapter,
        BraketLocalHALAdapter,
        braket_circuit_to_workload,
    )

    theta = FreeParameter("theta")
    circuit = Circuit().i(0).i(1).ry(2, theta).ry(2, theta).rz(0, theta).measure([2, 0])
    workload = braket_circuit_to_workload(
        circuit,
        workload_id="braket_shared",
        shots=16,
        capture_semantics=True,
        parameter_bindings={"theta": pi / 2},
    )
    original_program = circuit.to_ir(ir_type="OPENQASM").source
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    native = LocalSimulator("braket_sv")
    backend_id = "aws_braket_ionq" if cloud_profile else "local_braket_sv"
    adapter = (
        BraketAwsHALAdapter(hal.profile(backend_id), device=native)
        if cloud_profile
        else BraketLocalHALAdapter(hal.profile(backend_id), device=native)
    )
    hal.register_backend(adapter)
    job = hal.submit(backend_id, workload, approval_id="offline-native-only")
    result = hal.result(job)
    assert result.counts == {"10": 16}
    assert workload.program == original_program
    assert job.submission is not None
    assert job.submission.original_program == original_program
    assert isinstance(job.submission.request, WorkloadSemantics)
    assert job.submission.request.parameters == (
        ("theta", "braket:theta", ((2, 0), (3, 0), (4, 0))),
    )
    assert job.submission.request.parameter_values == (("braket:theta", pi / 2),)
    assert hal.result(job) is result


def test_braket_target_capacity_and_instruction_refusal_precede_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned target, shot limit and unsupported source cannot invoke native run."""
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator

    from scpn_quantum_control.hardware.hal_braket import (
        BraketLocalHALAdapter,
        braket_circuit_to_workload,
    )

    built_in = HardwareAbstractionLayer.with_builtin_profiles().profile("local_braket_sv")
    profile = replace(built_in, capabilities=replace(built_in.capabilities, max_shots=4))
    hal = HardwareAbstractionLayer((profile,))
    native = LocalSimulator("braket_sv")
    calls: list[object] = []

    def forbidden_run(circuit: object, **options: object) -> NoReturn:
        """Detect a provider crossing independently of request admission."""
        calls.append(circuit)
        raise AssertionError("invalid request reached native Braket run")

    monkeypatch.setattr(native, "run", forbidden_run)
    hal.register_backend(BraketLocalHALAdapter(profile, device=native))
    workload = braket_circuit_to_workload(
        Circuit().x(0).measure(0),
        workload_id="braket_refusal",
        shots=4,
        capture_semantics=True,
        requested_target="another_native_target",
    )
    with pytest.raises(ValueError, match="target"):
        hal.submit(profile.backend_id, workload)
    with pytest.raises(ValueError, match="shots"):
        hal.submit(profile.backend_id, replace(workload, shots=5))
    with pytest.raises(ValueError, match="IR format"):
        hal.submit(profile.backend_id, replace(workload, ir_format="quil"))
    assert calls == []


@pytest.mark.parametrize("route", ["aer", "iqm", "runtime"])
@pytest.mark.parametrize("shared_with_gate", [False, True])
def test_native_global_phase_parameter_is_retained_and_bound(
    route: str,
    shared_with_gate: bool,
) -> None:
    """Native global phase sharing keeps its identity without changing the count oracle."""
    theta = Parameter("phase_theta")
    circuit = QuantumCircuit(3, 2)
    circuit.global_phase = 2 * theta
    if shared_with_gate:
        circuit.ry(theta, 2)
        circuit.ry(theta, 2)
    else:
        circuit.x(2)
    circuit.measure(2, 0)
    circuit.measure(0, 1)
    native = AerSimulator(max_parallel_threads=1, seed_simulator=7)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    if route == "aer":
        backend_id = "local_qiskit_aer"
        hal.register_backend(QiskitAerHALAdapter(hal.profile(backend_id), backend=native))
    elif route == "iqm":
        backend_id = "iqm_cloud"
        hal.register_backend(IQMHALAdapter(hal.profile(backend_id), backend=native))
    else:
        backend_id = "ibm_quantum"
        hal.register_backend(QiskitRuntimeHALAdapter(hal.profile(backend_id), backend=native))
    workload = qiskit_circuit_to_workload(
        circuit,
        workload_id="phase_shared",
        shots=16,
        capture_semantics=True,
        parameter_bindings={theta: pi / 2},
    )
    job = hal.submit(backend_id, workload, approval_id="offline-native-only")
    result = hal.result(job)
    assert result.counts == {"01": 16}
    assert job.submission is not None and job.submission.original_program == workload.program
    assert isinstance(job.submission.request, WorkloadSemantics)
    request = job.submission.request
    assert request.global_phase_parameters == (str(theta.uuid),)
    assert request.parameters == (
        ("phase_theta", str(theta.uuid), ((0, 0), (1, 0)) if shared_with_gate else ()),
    )
    assert request.parameter_values == ((str(theta.uuid), pi / 2),)


def test_runtime_native_sample_budget_refuses_before_sampler_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized native result refuses transport and preserves prior native evidence."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = AerSimulator(max_parallel_threads=1, seed_simulator=7)
    factory_calls: list[AerSimulator] = []

    def native_sampler(*, mode: AerSimulator) -> SamplerV2:
        """Use the installed Runtime SDK with an independently counted factory crossing."""
        factory_calls.append(mode)
        return SamplerV2(mode=mode)

    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=backend,
            sampler_factory=native_sampler,
        )
    )
    workload = qiskit_circuit_to_workload(
        _partial_permutation(),
        workload_id="native_budget",
        shots=8,
        capture_semantics=True,
    )
    monkeypatch.setenv("SCPN_MAX_DENSE_GIB", "0.001")
    prior = hal.submit("ibm_quantum", workload, approval_id="offline-native-only")
    evidence = hal.result(prior)
    assert evidence.counts == {"01": 8}
    assert factory_calls == [backend]
    monkeypatch.setenv("SCPN_MAX_DENSE_GIB", str(1 / 1024**3))
    with pytest.raises(MemoryError, match="native runtime.*budget"):
        hal.submit(
            "ibm_quantum",
            replace(workload, workload_id="rejected"),
            approval_id="offline-native-only",
        )
    assert factory_calls == [backend]
    assert hal.result(prior) is evidence
    monkeypatch.setenv("SCPN_MAX_DENSE_GIB", "0.001")
    recovered = hal.submit(
        "ibm_quantum",
        replace(workload, workload_id="recovered"),
        approval_id="offline-native-only",
    )
    assert hal.result(recovered).counts == {"01": 8}
    assert factory_calls == [backend, backend]
    assert hal.result(prior) is evidence
