# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL tests
# scpn-quantum-control -- provider-neutral hardware abstraction tests
"""Tests for the provider-neutral hardware abstraction layer."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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


@pytest.fixture
def local_job_route() -> tuple[
    HardwareAbstractionLayer, LocalDeterministicSimulator, QuantumWorkload
]:
    """Provide a real offline HAL route with a two-qubit workload."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = LocalDeterministicSimulator(hal.profile("local_statevector"))
    hal.register_backend(backend)
    workload = QuantumWorkload("identity-custody", "mlir", "module {}", 2, shots=16)
    return hal, backend, workload


@pytest.mark.parametrize("field", ["backend_id", "workload_id"])
def test_hal_rejects_substituted_submission_identity(
    local_job_route: tuple[HardwareAbstractionLayer, LocalDeterministicSimulator, QuantumWorkload],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Reject a mislabelled handle after actual local adapter submission."""
    hal, backend, workload = local_job_route
    submit = backend.submit
    submitted: list[QuantumJobRef] = []

    def mislabelled_submit(
        request: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        job = submit(request, approval_id=approval_id)
        submitted.append(job)
        return replace(
            job,
            backend_id="different" if field == "backend_id" else job.backend_id,
            workload_id="different" if field == "workload_id" else job.workload_id,
        )

    monkeypatch.setattr(backend, "submit", mislabelled_submit)
    with pytest.raises(ValueError, match=f"submit.*{field}"):
        hal.submit(backend.backend_id, workload)
    assert len(submitted) == 1
    assert backend.result(submitted[0]).shots == workload.shots


@pytest.mark.parametrize("field", ["job_id", "backend_id", "workload_id"])
def test_hal_rejects_substituted_result_identity(
    local_job_route: tuple[HardwareAbstractionLayer, LocalDeterministicSimulator, QuantumWorkload],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Refuse real counts attached to the wrong job, route or workload."""
    hal, backend, workload = local_job_route
    job = hal.submit(backend.backend_id, workload)
    fetch = backend.result

    def mislabelled_result(request: QuantumJobRef) -> QuantumJobResult:
        result = fetch(request)
        return replace(
            result,
            job=replace(
                result.job,
                job_id="different" if field == "job_id" else result.job.job_id,
                backend_id="different" if field == "backend_id" else result.job.backend_id,
                workload_id="different" if field == "workload_id" else result.job.workload_id,
            ),
        )

    monkeypatch.setattr(backend, "result", mislabelled_result)
    with pytest.raises(ValueError, match=f"result.*{field}"):
        hal.result(job)


@pytest.mark.parametrize("field", ["job_id", "backend_id", "workload_id"])
def test_hal_rejects_substituted_cancellation_identity(
    local_job_route: tuple[HardwareAbstractionLayer, LocalDeterministicSimulator, QuantumWorkload],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Reject a cancellation response referring to a different identity."""
    hal, backend, workload = local_job_route
    job = hal.submit(backend.backend_id, workload)
    cancel = backend.cancel

    def mislabelled_cancel(request: QuantumJobRef) -> QuantumJobRef:
        response = cancel(request)
        return replace(
            response,
            job_id="different" if field == "job_id" else response.job_id,
            backend_id="different" if field == "backend_id" else response.backend_id,
            workload_id="different" if field == "workload_id" else response.workload_id,
        )

    monkeypatch.setattr(backend, "cancel", mislabelled_cancel)
    with pytest.raises(ValueError, match=f"cancel.*{field}"):
        hal.cancel(job)
    assert backend.status(job) == "cancelled"


def test_hal_recovered_handle_preserves_result_evidence(
    local_job_route: tuple[HardwareAbstractionLayer, LocalDeterministicSimulator, QuantumWorkload],
) -> None:
    """Allow lifecycle changes and recovery without rewriting raw evidence."""
    hal, backend, workload = local_job_route
    job = hal.submit(backend.backend_id, workload)
    recovered = replace(job, status="submitted", metadata={"recovered": True})
    fresh_router = HardwareAbstractionLayer((backend.profile,))
    fresh_router.register_backend(backend)
    raw = backend.result(job)
    result = fresh_router.result(recovered)
    assert result is raw
    assert dict(result.counts) == dict(raw.counts)
    assert result.metadata == raw.metadata
    assert result.job.metadata != recovered.metadata
    assert result.job.status != recovered.status
    assert fresh_router.cancel(recovered).status == "cancelled"


def test_hal_rejects_recovered_handle_with_wrong_workload(
    local_job_route: tuple[HardwareAbstractionLayer, LocalDeterministicSimulator, QuantumWorkload],
) -> None:
    """Reject a misassociated recovered handle using the unmodified adapter."""
    hal, backend, workload = local_job_route
    job = hal.submit(backend.backend_id, workload)
    misassociated = replace(job, workload_id="another-workload")
    with pytest.raises(ValueError, match="result.*workload_id"):
        hal.result(misassociated)
    assert hal.result(job) is backend.result(job)


@pytest.mark.parametrize(
    "key",
    [
        "approval_id",
        "provider_job_id",
        "execution_mode",
        "backend_name",
        "quantum_computer",
        "ir_format",
        "n_qubits",
        "shots",
        "target",
        "workload_id",
        "broker",
    ],
)
def test_workload_rejects_adapter_owned_metadata(key: str) -> None:
    """Refuse metadata capable of replacing submitted settings or provenance."""
    with pytest.raises(ValueError, match=f"reserved.*{key}"):
        QuantumWorkload("metadata-custody", "mlir", "module {}", 2, metadata={key: "override"})


def test_workload_rejects_duplicate_shot_setting_even_when_equal() -> None:
    """Require one authoritative shot setting instead of duplicate metadata."""
    with pytest.raises(ValueError, match="reserved.*shots"):
        QuantumWorkload(
            "metadata-custody", "mlir", "module {}", 2, shots=16, metadata={"shots": 16}
        )


def test_iqm_adapter_preserves_requested_shots_with_real_local_execution() -> None:
    """Exercise QPY, the IQM adapter and Aer without contacting a provider."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    from scpn_quantum_control.hardware.hal_iqm import IQMHALAdapter, iqm_qiskit_workload

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = AerSimulator(max_parallel_threads=1)
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=backend)
    hal.register_backend(adapter)
    circuit = QuantumCircuit(2)
    circuit.x(1)
    circuit.measure_all()
    workload = iqm_qiskit_workload(
        circuit,
        workload_id="local-metadata-custody",
        shots=16,
        metadata={"campaign": "offline-adapter-contract", "seed": 7},
    )
    job = hal.submit("iqm_cloud", workload, approval_id="local-injected-adapter-only")
    result = hal.result(job)
    assert job.metadata["shots"] == workload.shots == result.shots == 16
    assert result.counts == {"10": 16}
    assert job.metadata["campaign"] == "offline-adapter-contract"
    assert job.metadata["seed"] == 7
    assert result.metadata["provider_job_id"] == job.metadata["provider_job_id"]
    assert result.metadata["backend_name"] == backend.name


class _ReplayProviderJob:
    """Replay locally sampled data at an injected provider SDK boundary."""

    def __init__(self, payload: object) -> None:
        """Retain the SDK-format payload and operation counters."""
        self.payload = payload
        self.id = "local-custody-job"
        self.details = SimpleNamespace(status="Succeeded")
        self.reads = 0
        self.cancels = 0

    def job_id(self) -> str:
        """Return the stable provider identifier."""
        return self.id

    def status(self) -> str:
        """Return a completed provider observation."""
        return "COMPLETED"

    def state(self) -> str:
        """Return a completed Braket observation."""
        return "COMPLETED"

    def result(self, timeout: float | None = None) -> object:
        """Return local evidence using the provider result signature."""
        self.reads += 1
        return self.payload

    def get_results(self) -> object:
        """Return evidence through the Azure SDK result signature."""
        return self.result()

    def cancel(self) -> None:
        """Record the provider cancellation request."""
        self.cancels += 1


class _ReplaySubmission:
    """Inject a recorded local job without a provider network connection."""

    def __init__(self, job: _ReplayProviderJob) -> None:
        """Bind a provider-shaped job and route metadata."""
        self.job = job
        self.name = self.id = "local-custody-route"
        self.options = SimpleNamespace(default_shots=16)

    def run(self, *args: object, **kwargs: object) -> _ReplayProviderJob:
        """Return the recorded job through gate-provider run signatures."""
        return self.job

    def submit(self, *args: object, **kwargs: object) -> _ReplayProviderJob:
        """Return the recorded job through the Azure submission signature."""
        return self.job


@pytest.fixture(params=["qiskit", "braket", "azure", "qbraid", "strangeworks"])
def submitted_provider_record(
    request: pytest.FixtureRequest,
) -> tuple[QuantumBackend, QuantumJobRef, _ReplayProviderJob]:
    """Submit through each real adapter with locally sampled SDK evidence."""
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    from scpn_quantum_control.hardware.hal_azure import AzureQuantumHALAdapter
    from scpn_quantum_control.hardware.hal_braket import BraketAwsHALAdapter
    from scpn_quantum_control.hardware.hal_qbraid import QbraidRuntimeHALAdapter
    from scpn_quantum_control.hardware.hal_qiskit import (
        QiskitRuntimeHALAdapter,
        qiskit_circuit_to_qasm3_workload,
        qiskit_circuit_to_workload,
    )
    from scpn_quantum_control.hardware.hal_strangeworks import StrangeworksComputeHALAdapter

    circuit = QuantumCircuit(2)
    circuit.x(1)
    circuit.measure_all()
    sampled = StatevectorSampler(seed=7).run([circuit], shots=16).result()
    counts = sampled[0].data.meas.get_counts()
    assert counts == {"10": 16}
    kind = request.param
    payload: object = {"counts": counts}
    if kind == "qiskit":
        payload = sampled
    elif kind == "braket":
        payload = SimpleNamespace(measurement_counts=counts)
    provider_job = _ReplayProviderJob(payload)
    transport = _ReplaySubmission(provider_job)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    routes: dict[str, QuantumBackend] = {
        "qiskit": QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=transport,
            sampler_factory=lambda **kwargs: transport,
        ),
        "braket": BraketAwsHALAdapter(hal.profile("aws_braket_ionq"), device=transport),
        "azure": AzureQuantumHALAdapter(
            hal.profile("azure_quantum_ionq_simulator"),
            target=transport,
        ),
        "qbraid": QbraidRuntimeHALAdapter(hal.profile("qbraid_ionq"), device=transport),
        "strangeworks": StrangeworksComputeHALAdapter(
            hal.profile("strangeworks_compute"),
            backend=transport,
        ),
    }
    adapter = routes[kind]
    build = qiskit_circuit_to_workload if kind == "qiskit" else qiskit_circuit_to_qasm3_workload
    workload = build(circuit, workload_id="stored-request", shots=16)
    if kind == "braket":
        from braket.circuits import Circuit

        from scpn_quantum_control.hardware.hal_braket import braket_circuit_to_workload

        workload = braket_circuit_to_workload(
            Circuit().x(0).i(1), workload_id="stored-request", shots=16
        )
    job = adapter.submit(workload, approval_id="local-replay-only")
    return adapter, job, provider_job


def test_provider_result_uses_stored_submission_metadata(
    submitted_provider_record: tuple[QuantumBackend, QuantumJobRef, _ReplayProviderJob],
) -> None:
    """Ignore handle annotations when decoding or caching submitted evidence."""
    adapter, original, provider = submitted_provider_record
    recovered = replace(original, status="queued", metadata={"shots": 1, "n_qubits": 1})
    result = adapter.result(recovered)
    assert result.shots == 16
    assert result.counts == {"10": 16}
    assert result.job is original
    assert result.metadata["approval_id"] == "local-replay-only"
    assert adapter.result(replace(original, metadata={})) is result
    assert provider.reads == 1


def test_provider_cancel_cannot_replace_stored_submission_metadata(
    submitted_provider_record: tuple[QuantumBackend, QuantumJobRef, _ReplayProviderJob],
) -> None:
    """Keep submission settings when a recovered handle is cancelled first."""
    adapter, original, provider = submitted_provider_record
    cancelled = adapter.cancel(replace(original, metadata={"shots": 1, "n_qubits": 1}))
    assert cancelled.metadata == original.metadata
    assert provider.cancels == 1
    result = adapter.result(original)
    assert result.shots == 16
    assert result.job.metadata == original.metadata


@pytest.mark.parametrize("cached", [False, True])
def test_provider_refuses_misassociated_handle_before_access(
    submitted_provider_record: tuple[QuantumBackend, QuantumJobRef, _ReplayProviderJob],
    cached: bool,
) -> None:
    """Reject wrong routes/workloads before SDK calls or cache access."""
    adapter, original, provider = submitted_provider_record
    if cached:
        adapter.result(original)
    reads = provider.reads
    for foreign in (
        replace(original, backend_id="other-route"),
        replace(original, workload_id="other-workload"),
    ):
        for operation in (adapter.result, adapter.cancel, adapter.status):
            with pytest.raises(ValueError, match="different"):
                operation(foreign)
    assert provider.reads == reads
    assert provider.cancels == 0


def test_provider_unknown_handle_does_not_access_sdk(
    submitted_provider_record: tuple[QuantumBackend, QuantumJobRef, _ReplayProviderJob],
) -> None:
    """Refuse jobs absent from this adapter's retained submissions."""
    adapter, original, provider = submitted_provider_record
    unknown = replace(original, job_id="not-retained")
    for operation in (adapter.result, adapter.cancel, adapter.status):
        with pytest.raises(KeyError, match="unknown job_id"):
            operation(unknown)
    assert provider.reads == provider.cancels == 0


def test_provider_handle_cannot_hide_observed_shot_mismatch(
    submitted_provider_record: tuple[QuantumBackend, QuantumJobRef, _ReplayProviderJob],
) -> None:
    """Reject short results even if the caller supplies a matching shot count."""
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    adapter, original, provider = submitted_provider_record
    counts = {"10": 1}
    provider.payload = {"counts": counts}
    if adapter.backend_id == "ibm_quantum":
        circuit = QuantumCircuit(2)
        circuit.x(1)
        circuit.measure_all()
        provider.payload = StatevectorSampler(seed=7).run([circuit], shots=1).result()
    elif adapter.backend_id == "aws_braket_ionq":
        provider.payload = SimpleNamespace(measurement_counts=counts)
    altered = replace(original, metadata={**original.metadata, "shots": 1})
    with pytest.raises(ValueError, match="mismatch"):
        adapter.result(altered)
    assert provider.reads == 1


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

    cirq = describe_hal_backend_profile("local_cirq")
    assert cirq.adapter_module == "scpn_quantum_control.hardware.hal_cirq"
    assert "cirq" in cirq.workloads

    iqm = describe_hal_backend_profile("iqm_cloud")
    assert iqm.adapter_module == "scpn_quantum_control.hardware.hal_iqm"
    assert "qiskit_qpy" in iqm.workloads

    pasqal = describe_hal_backend_profile("pasqal_cloud")
    assert pasqal.adapter_module == "scpn_quantum_control.hardware.hal_pasqal"
    assert "pulser" in pasqal.workloads

    dwave = describe_hal_backend_profile("dwave_leap")
    assert dwave.adapter_module == "scpn_quantum_control.hardware.hal_dwave"
    assert "bqm" in dwave.workloads

    quandela = describe_hal_backend_profile("quandela_cloud")
    assert quandela.adapter_module == "scpn_quantum_control.hardware.hal_quandela"
    assert "perceval" in quandela.workloads

    oqc = describe_hal_backend_profile("oqc_cloud")
    assert oqc.adapter_module == "scpn_quantum_control.hardware.hal_oqc"
    assert "openqasm3" in oqc.workloads


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
        "qbraid_runtime",
        "quandela_cloud",
        "dwave_leap",
        "strangeworks_compute",
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


def test_dynamic_aggregator_profiles_are_first_class_catalog_routes() -> None:
    """Provider-agnostic aggregators should not collapse to one provider."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()

    qbraid = hal.profile("qbraid_runtime")
    assert qbraid.provider == "dynamic"
    assert qbraid.broker == "qbraid"
    assert qbraid.target_family == "dynamic_catalog"
    assert "dynamic_catalog" in qbraid.notes
    assert qbraid.submit_requires_approval is True
    assert {
        "openqasm3",
        "qiskit",
        "cirq",
        "quil",
        "braket_ir",
        "pennylane",
        "pyqubo",
        "tket",
        "mlir",
    } <= set(qbraid.ir_formats)

    strangeworks = hal.profile("strangeworks_compute")
    assert strangeworks.provider == "dynamic"
    assert strangeworks.broker == "strangeworks"
    assert strangeworks.target_family == "dynamic_catalog"
    assert "dynamic_catalog" in strangeworks.notes
    assert strangeworks.submit_requires_approval is True
    assert {"openqasm3", "qiskit", "quil", "braket_ir", "mlir"} <= set(strangeworks.ir_formats)


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
