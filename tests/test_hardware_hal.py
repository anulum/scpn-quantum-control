# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- provider-neutral hardware abstraction tests
"""Tests for the provider-neutral hardware abstraction layer."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.hal import (
    BackendCapabilities,
    BackendProfile,
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumBackend,
    QuantumJobRef,
    QuantumJobResult,
    QuantumWorkload,
    built_in_backend_profiles,
)


def test_hal_profiles_export_backend_descriptors_for_selector_metadata() -> None:
    """Every built-in HAL route should have offline selector metadata."""

    from scpn_quantum_control.hardware import (
        describe_hal_backend_profile as exported_describe_hal_backend_profile,
    )
    from scpn_quantum_control.hardware import (
        list_hal_backend_descriptors as exported_list_hal_backend_descriptors,
    )
    from scpn_quantum_control.hardware.backends import (
        describe_hal_backend_profile,
        list_hal_backend_descriptors,
    )

    profiles = built_in_backend_profiles()
    descriptors = list_hal_backend_descriptors()
    by_name = {descriptor.name: descriptor for descriptor in descriptors}

    assert set(by_name) == {profile.backend_id for profile in profiles}
    assert [descriptor.name for descriptor in descriptors] == sorted(by_name)
    assert exported_list_hal_backend_descriptors() == descriptors
    assert exported_describe_hal_backend_profile("quera_bloqade").name == "quera_bloqade"

    quera = describe_hal_backend_profile("quera_bloqade")
    assert quera.provider == "quera"
    assert quera.execution_mode == "cloud_neutral_atom_analog"
    assert quera.adapter_module == "scpn_quantum_control.hardware.hal_quera_bloqade"
    assert quera.can_submit is True
    assert quera.submit_requires_approval is True
    assert quera.capabilities == ("analog", "cancellation", "counts", "shots")
    assert quera.workloads == ("bloqade", "braket_ahs", "mlir")

    aer = describe_hal_backend_profile("local_qiskit_aer")
    assert aer.execution_mode == "local_simulator"
    assert aer.adapter_module == "scpn_quantum_control.hardware.hal_qiskit"
    assert aer.can_simulate is True
    assert aer.can_submit is False
    assert aer.submit_requires_approval is False
    assert "statevector" in aer.capabilities

    iqm = describe_hal_backend_profile("iqm_cloud")
    assert iqm.adapter_module == "scpn_quantum_control.hardware.hal_iqm"
    assert "qiskit_qpy" in iqm.workloads

    pasqal = describe_hal_backend_profile("pasqal_cloud")
    assert pasqal.adapter_module == "scpn_quantum_control.hardware.hal_pasqal"
    assert "pulser" in pasqal.workloads


def test_builtin_hal_profiles_cover_major_current_provider_routes() -> None:
    """Built-in profiles should cover the current major provider families."""

    profiles = built_in_backend_profiles()
    ids = {profile.backend_id for profile in profiles}

    expected = {
        "ibm_quantum",
        "ionq_cloud",
        "aws_braket_ionq",
        "aws_braket_iqm",
        "aws_braket_quera",
        "aws_braket_rigetti",
        "aws_braket_aqt",
        "aws_braket_dm1",
        "aws_braket_sv1",
        "aws_braket_tn1",
        "azure_quantum_quantinuum",
        "azure_quantum_quantinuum_emulator",
        "azure_quantum_ionq",
        "azure_quantum_ionq_simulator",
        "azure_quantum_rigetti",
        "azure_quantum_rigetti_qvm",
        "azure_quantum_pasqal",
        "azure_quantum_pasqal_emulator",
        "azure_quantum_qci_preview",
        "quantinuum_cloud",
        "rigetti_qcs",
        "quera_bloqade",
        "iqm_cloud",
        "pasqal_cloud",
        "oqc_cloud",
        "qbraid_ionq",
        "quandela_cloud",
        "dwave_leap",
        "local_statevector",
        "local_braket_ahs",
        "local_braket_dm",
        "local_braket_sv",
        "local_qiskit_aer",
        "local_cirq",
        "local_pennylane",
    }

    assert expected.issubset(ids)
    assert len(ids) == len(profiles)
    assert all(profile.submit_requires_approval == profile.is_cloud for profile in profiles)
    assert all(
        profile.capabilities.max_qubits is None or profile.capabilities.max_qubits > 0
        for profile in profiles
    )


def test_hal_discovery_is_deterministic_and_does_not_require_sdk_imports() -> None:
    """HAL construction should be metadata-only and offline."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()

    first = [profile.backend_id for profile in hal.list_profiles()]
    second = [profile.backend_id for profile in hal.list_profiles()]

    assert first == sorted(first)
    assert first == second
    assert hal.profile("aws_braket_quera").provider == "quera"
    assert hal.profile("azure_quantum_quantinuum").broker == "azure_quantum"
    assert hal.profile("local_statevector").is_cloud is False


def test_local_deterministic_simulator_round_trip() -> None:
    """The HAL should execute local simulator workloads through the common API."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = LocalDeterministicSimulator(hal.profile("local_statevector"))
    hal.register_backend(backend)
    workload = QuantumWorkload(
        workload_id="w1",
        ir_format="mlir",
        program="module {}",
        n_qubits=3,
        shots=128,
        metadata={"seed": "7"},
    )

    job = hal.submit("local_statevector", workload)
    result = hal.result(job)

    assert isinstance(job, QuantumJobRef)
    assert job.backend_id == "local_statevector"
    assert result.status == "completed"
    assert result.shots == 128
    assert sum(result.counts.values()) == 128
    assert set(result.counts).issubset({"000", "111"})
    assert result.metadata["execution_mode"] == "local_deterministic_simulator"


def test_cloud_profile_fails_closed_without_injected_backend() -> None:
    """Cloud routes must not silently submit or fake results."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    workload = QuantumWorkload(
        workload_id="cloud",
        ir_format="openqasm3",
        program="OPENQASM 3.0;",
        n_qubits=2,
        shots=16,
    )

    with pytest.raises(PermissionError, match="not registered"):
        hal.submit("ionq_cloud", workload, approval_id="approved")


def test_cloud_backend_requires_explicit_approval_before_submit() -> None:
    """Injected cloud adapters should still require approval tokens."""

    profile = BackendProfile(
        backend_id="test_cloud",
        provider="test_provider",
        broker="direct",
        modality="gate_model",
        sdk_package="test-sdk",
        ir_formats=("openqasm3",),
        is_cloud=True,
        submit_requires_approval=True,
        capabilities=BackendCapabilities(
            supports_shots=True,
            supports_counts=True,
            supports_statevector=False,
            supports_mid_circuit_measurement=False,
            supports_analog=False,
            supports_pulse=False,
            max_qubits=4,
        ),
    )

    class ApprovedBackend:
        backend_id = "test_cloud"

        def submit(
            self, workload: QuantumWorkload, *, approval_id: str | None = None
        ) -> QuantumJobRef:
            return QuantumJobRef(
                job_id=f"job-{approval_id}",
                backend_id=self.backend_id,
                workload_id=workload.workload_id,
                status="submitted",
            )

        def status(self, job: QuantumJobRef) -> str:
            return "completed"

        def result(self, job: QuantumJobRef) -> QuantumJobResult:
            return QuantumJobResult(
                job=job,
                status="completed",
                counts={"0": 1},
                shots=1,
            )

        def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
            return QuantumJobRef(
                job_id=job.job_id,
                backend_id=job.backend_id,
                workload_id=job.workload_id,
                status="cancelled",
            )

    hal = HardwareAbstractionLayer((profile,))
    hal.register_backend(ApprovedBackend())
    workload = QuantumWorkload(
        workload_id="w2",
        ir_format="openqasm3",
        program="OPENQASM 3.0;",
        n_qubits=1,
        shots=1,
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("test_cloud", workload)

    job = hal.submit("test_cloud", workload, approval_id="approved-test")
    assert job.job_id == "job-approved-test"
    assert hal.status(job) == "completed"
    assert hal.cancel(job).status == "cancelled"


def test_workload_validation_rejects_bad_ir_qubits_and_shots() -> None:
    """Workload validation should fail before reaching provider code."""

    with pytest.raises(ValueError, match="ir_format"):
        QuantumWorkload(workload_id="bad", ir_format="quil 2", program="x", n_qubits=1)
    with pytest.raises(ValueError, match="n_qubits"):
        QuantumWorkload(workload_id="bad", ir_format="openqasm3", program="x", n_qubits=0)
    with pytest.raises(ValueError, match="shots"):
        QuantumWorkload(workload_id="bad", ir_format="openqasm3", program="x", n_qubits=1, shots=0)


def test_profile_validation_rejects_unsupported_resource_shapes() -> None:
    """Backend profiles should reject inconsistent production metadata."""

    with pytest.raises(ValueError, match="backend_id"):
        BackendProfile(
            backend_id="",
            provider="provider",
            broker="direct",
            modality="gate_model",
            sdk_package="sdk",
            ir_formats=("openqasm3",),
            capabilities=BackendCapabilities(
                supports_shots=True,
                supports_counts=True,
                supports_statevector=False,
                supports_mid_circuit_measurement=False,
                supports_analog=False,
                supports_pulse=False,
                max_qubits=1,
            ),
        )
    with pytest.raises(ValueError, match="max_qubits"):
        BackendCapabilities(
            supports_shots=True,
            supports_counts=True,
            supports_statevector=False,
            supports_mid_circuit_measurement=False,
            supports_analog=False,
            supports_pulse=False,
            max_qubits=0,
        )


def test_hal_protocol_runtime_check() -> None:
    """Registered backends should satisfy the runtime protocol."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = LocalDeterministicSimulator(hal.profile("local_statevector"))

    assert isinstance(backend, QuantumBackend)
    hal.register_backend(backend)
    with pytest.raises(ValueError, match="already registered"):
        hal.register_backend(backend)


def test_hal_rejects_workload_that_backend_cannot_accept() -> None:
    """Routing should validate IR format and qubit limits before submit."""

    profile = BackendProfile(
        backend_id="limited",
        provider="local",
        broker="direct",
        modality="simulator",
        sdk_package="none",
        ir_formats=("openqasm3",),
        is_cloud=False,
        submit_requires_approval=False,
        capabilities=BackendCapabilities(
            supports_shots=True,
            supports_counts=True,
            supports_statevector=False,
            supports_mid_circuit_measurement=False,
            supports_analog=False,
            supports_pulse=False,
            max_qubits=1,
        ),
    )
    hal = HardwareAbstractionLayer((profile,))
    hal.register_backend(LocalDeterministicSimulator(profile))

    with pytest.raises(ValueError, match="IR format"):
        hal.submit(
            "limited",
            QuantumWorkload(
                workload_id="bad-ir",
                ir_format="mlir",
                program="module {}",
                n_qubits=1,
            ),
        )
    with pytest.raises(ValueError, match="qubits"):
        hal.submit(
            "limited",
            QuantumWorkload(
                workload_id="too-large",
                ir_format="openqasm3",
                program="OPENQASM 3.0;",
                n_qubits=2,
            ),
        )
