# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL iqm adapters tests
# scpn-quantum-control -- IQM HAL adapter tests
"""Tests for the direct IQM HAL adapter."""

from __future__ import annotations

import types
from dataclasses import replace
from typing import Any

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import GenericBackendV2

from scpn_quantum_control.hardware.hal import (
    HardwareAbstractionLayer,
    QuantumJobRef,
    QuantumWorkload,
)
from scpn_quantum_control.hardware.hal_iqm import IQMHALAdapter, iqm_qiskit_workload
from scpn_quantum_control.hardware.provider_modalities import ModalitySemantics
from scpn_quantum_control.hardware.provider_semantics import WorkloadSemantics


class _FakeIQMResult:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _FakeIQMJob:
    def __init__(
        self,
        counts: dict[str, int],
        *,
        status: str = "DONE",
        expected_timeout: float | None = 9.5,
    ) -> None:
        self._counts = counts
        self._status = status
        self._expected_timeout = expected_timeout
        self.cancelled = False

    def job_id(self) -> str:
        return "iqm-job-123"

    def status(self) -> str:
        return self._status

    def result(self, timeout: float | None = None) -> _FakeIQMResult:
        if self._expected_timeout is not None:
            assert timeout == self._expected_timeout
        return _FakeIQMResult(self._counts)

    def cancel(self) -> None:
        self.cancelled = True


class _NativeIQMTarget:
    """Delegate compiler metadata to a genuine native two-qubit Target."""

    def __init__(self) -> None:
        """Retain native compiler metadata without inheriting an untyped SDK class."""
        self._native = GenericBackendV2(2, noise_info=False, seed=7)
        self.name = "fake_garnet"

    def __getattr__(self, name: str) -> object:
        """Return unchanged native target properties requested by the compiler."""
        return getattr(self._native, name)


class _FakeIQMBackend(_NativeIQMTarget):
    def __init__(self) -> None:
        super().__init__()
        self.jobs: list[_FakeIQMJob] = []
        self.received_shots: list[int] = []

    def run(self, circuits: list[QuantumCircuit], *, shots: int) -> _FakeIQMJob:
        assert len(circuits) == 1
        assert circuits[0].num_qubits == 2
        self.received_shots.append(shots)
        job = _FakeIQMJob({"00": 7, "11": 9})
        self.jobs.append(job)
        return job


class _FakeIQMProvider:
    def __init__(self, url: str, *, quantum_computer: str | None = None) -> None:
        self.url = url
        self.quantum_computer = quantum_computer

    def get_backend(self) -> _FakeIQMBackend:
        return _FakeIQMBackend()


def _bell_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def test_iqm_hal_adapter_executes_injected_backend_with_approval() -> None:
    """An approved native-compatible route preserves shots, job identity and lifecycle."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = _FakeIQMBackend()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=backend, timeout_s=9.5)
    hal.register_backend(adapter)
    workload = iqm_qiskit_workload(
        _bell_circuit(),
        workload_id="iqm_bell",
        shots=16,
        metadata={"campaign": "hal"},
    )

    job = hal.submit("iqm_cloud", workload, approval_id="approved-iqm")
    status = hal.status(job)
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert isinstance(job, QuantumJobRef)
    assert job.job_id.startswith("iqm_cloud:iqm_bell:")
    assert job.metadata["provider_job_id"] == "iqm-job-123"
    assert job.metadata["approval_id"] == "approved-iqm"
    assert job.metadata["execution_mode"] == "iqm_qiskit"
    assert job.metadata["backend_name"] == "fake_garnet"
    assert status == "completed"
    assert result.counts == {"00": 7, "11": 9}
    assert result.shots == 16
    assert result.metadata["backend_name"] == "fake_garnet"
    assert cancelled.status == "cancelled"
    assert backend.jobs[0].cancelled is True


def test_iqm_hal_adapter_requires_cloud_approval() -> None:
    """HAL and direct adapter entry points refuse before provider execution."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = _FakeIQMBackend()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=backend)
    hal.register_backend(adapter)
    workload = iqm_qiskit_workload(_bell_circuit(), workload_id="needs_approval", shots=8)

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("iqm_cloud", workload)
    with pytest.raises(PermissionError, match="approval"):
        adapter.submit(workload)
    assert backend.received_shots == [] and backend.jobs == []


def test_iqm_hal_adapter_rejects_wrong_profile_and_ir() -> None:
    """IQM refuses a foreign backend profile or source format before execution."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    with pytest.raises(ValueError, match="iqm_cloud"):
        IQMHALAdapter(hal.profile("ibm_quantum"), backend=_FakeIQMBackend())

    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=_FakeIQMBackend())
    with pytest.raises(ValueError, match="qiskit_qpy"):
        adapter.submit(
            iqm_qiskit_workload(_bell_circuit(), workload_id="bad_ir", shots=4).__class__(
                workload_id="bad_ir",
                ir_format="openqasm3",
                program="OPENQASM 3.0;",
                n_qubits=2,
                shots=4,
            ),
            approval_id="approved",
        )


def test_iqm_hal_adapter_uses_lazy_remote_provider_factory() -> None:
    """The lazy provider receives the explicit endpoint and computer selector."""
    captured: dict[str, str | None] = {}

    class Provider(_FakeIQMProvider):
        def __init__(self, url: str, *, quantum_computer: str | None = None) -> None:
            captured["url"] = url
            captured["quantum_computer"] = quantum_computer
            super().__init__(url, quantum_computer=quantum_computer)

    def import_module(name: str) -> Any:
        if name == "iqm.qiskit_iqm.iqm_provider":
            return types.SimpleNamespace(IQMProvider=Provider)
        raise ModuleNotFoundError(name)

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IQMHALAdapter(
        hal.profile("iqm_cloud"),
        server_url="https://example.iqm.invalid",
        quantum_computer="garnet",
        import_module=import_module,
    )
    job = adapter.submit(
        iqm_qiskit_workload(_bell_circuit(), workload_id="remote_iqm", shots=16),
        approval_id="approved",
    )

    assert job.status == "submitted"
    assert captured == {
        "url": "https://example.iqm.invalid",
        "quantum_computer": "garnet",
    }


def test_iqm_hal_adapter_fails_closed_without_backend_or_server_url() -> None:
    """An unconfigured route cannot select a default IQM endpoint."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"))

    with pytest.raises(RuntimeError, match="server_url"):
        adapter.submit(
            iqm_qiskit_workload(_bell_circuit(), workload_id="no_route", shots=4),
            approval_id="approved",
        )


def test_iqm_hal_adapter_reports_unknown_jobs_and_bad_counts() -> None:
    """Unknown handles and negative provider counts never become completed results."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=_FakeIQMBackend())

    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.result(
            QuantumJobRef(
                job_id="missing",
                backend_id="iqm_cloud",
                workload_id="missing",
                status="submitted",
            )
        )

    class BadBackend(_FakeIQMBackend):
        def run(self, circuits: list[QuantumCircuit], *, shots: int) -> _FakeIQMJob:
            return _FakeIQMJob({"00": -1}, expected_timeout=None)

    bad = IQMHALAdapter(hal.profile("iqm_cloud"), backend=BadBackend())
    job = bad.submit(
        iqm_qiskit_workload(_bell_circuit(), workload_id="bad_counts", shots=4),
        approval_id="approved",
    )
    with pytest.raises(ValueError, match="non-negative"):
        bad.result(job)


def test_iqm_hal_adapter_rejects_provider_job_without_id() -> None:
    """IQM adapter should fail closed when backend job id is missing."""

    class MissingIdJob:
        def status(self) -> str:
            return "DONE"

        def result(self, timeout: float | None = None) -> _FakeIQMResult:
            del timeout
            return _FakeIQMResult({"0": 1})

        def cancel(self) -> None:
            self.cancelled = True

    class MissingIdBackend(_NativeIQMTarget):
        def run(self, circuits: list[QuantumCircuit], *, shots: int) -> MissingIdJob:
            del circuits, shots
            return MissingIdJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=MissingIdBackend())

    with pytest.raises(ValueError, match="provider job id"):
        adapter.submit(
            iqm_qiskit_workload(_bell_circuit(), workload_id="missing_iqm_job_id", shots=4),
            approval_id="approved",
        )


def test_iqm_provider_job_id_rejects_control_characters() -> None:
    """IQM provider identifiers must reject control-character payloads."""
    from scpn_quantum_control.hardware import hal_iqm as iqm_mod

    class BadJob:
        job_id = "iqm-provider-\n2"

    with pytest.raises(ValueError, match="provider job id"):
        iqm_mod._job_id(BadJob())


def test_iqm_provider_job_id_trims_padding() -> None:
    """IQM provider identifiers should be canonicalised by trimming padding."""
    from scpn_quantum_control.hardware import hal_iqm as iqm_mod

    class PaddedJob:
        job_id = "  iqm-provider-2  "

    assert iqm_mod._job_id(PaddedJob()) == "iqm-provider-2"


def test_iqm_backend_name_rejects_control_characters() -> None:
    """IQM backend names must reject control-character payloads."""
    from scpn_quantum_control.hardware import hal_iqm as iqm_mod

    class BadBackend:
        name = "iqm-\nbackend"

    with pytest.raises(ValueError, match="backend name"):
        iqm_mod._backend_name(BadBackend())


def test_iqm_backend_name_trims_padding() -> None:
    """IQM backend names should be canonicalised by trimming padding."""
    from scpn_quantum_control.hardware import hal_iqm as iqm_mod

    class PaddedBackend:
        name = "  iqm-backend  "

    assert iqm_mod._backend_name(PaddedBackend()) == "iqm-backend"


def test_iqm_status_normalisation_maps_provider_tokens() -> None:
    """IQM status values should map to canonical HAL status values."""
    from scpn_quantum_control.hardware import hal_iqm as iqm_mod

    assert iqm_mod._normalise_status("SUCCEEDED") == "completed"
    assert iqm_mod._normalise_status("CANCELED") == "cancelled"
    assert iqm_mod._normalise_status("IN-PROGRESS") == "running"
    assert iqm_mod._normalise_status("INPROGRESS") == "running"
    assert iqm_mod._normalise_status("INITIALIZING") == "submitted"
    assert iqm_mod._normalise_status("STARTING") == "submitted"
    assert iqm_mod._normalise_status("CREATING") == "submitted"
    assert iqm_mod._normalise_status("ABORTING") == "cancelled"
    assert iqm_mod._normalise_status("CANCELLING") == "cancelled"


def test_iqm_adapter_rejects_shot_mismatch() -> None:
    """IQM adapter must fail closed when decoded counts diverge from requested shots."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = _FakeIQMBackend()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=backend, timeout_s=9.5)
    hal.register_backend(adapter)
    workload = iqm_qiskit_workload(
        _bell_circuit(),
        workload_id="iqm_shot_mismatch",
        shots=15,
    )

    job = hal.submit("iqm_cloud", workload, approval_id="approved-iqm")
    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.result(job)


def test_iqm_quantum_computer_rejects_control_characters() -> None:
    """Control characters in an explicit computer selector refuse locally."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("iqm_cloud")
    with pytest.raises(ValueError, match="IQM quantum computer"):
        IQMHALAdapter(profile, backend=_FakeIQMBackend(), quantum_computer="garnet\nbad")


def test_iqm_quantum_computer_trims_padding() -> None:
    """Surrounding selector whitespace is canonicalised without changing the computer."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("iqm_cloud")
    adapter = IQMHALAdapter(
        profile,
        backend=_FakeIQMBackend(),
        quantum_computer="  garnet  ",
    )
    assert adapter._quantum_computer == "garnet"


@pytest.mark.parametrize(
    "case", ["corrupt", "unbound_legacy", "missing_bindings", "wrong_map", "foreign_modality"]
)
def test_iqm_source_and_binding_refusal_precedes_lazy_provider_construction(case: str) -> None:
    """Malformed native source or shared binding cannot construct a remote provider."""
    imports: list[str] = []

    def forbidden_import(name: str) -> object:
        imports.append(name)
        raise AssertionError("native source refusal must precede provider construction")

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IQMHALAdapter(
        hal.profile("iqm_cloud"),
        server_url="https://source-refusal.iqm.invalid",
        import_module=forbidden_import,
    )
    if case == "corrupt":
        workload = QuantumWorkload(
            workload_id="corrupt_native_qpy",
            ir_format="qiskit_qpy",
            program="invalid-base64!",
            n_qubits=2,
            shots=4,
        )
    else:
        shared = Parameter("shared")
        circuit = QuantumCircuit(2, 2)
        circuit.ry(shared, 0)
        circuit.ry(shared, 1)
        circuit.measure([0, 1], [0, 1])
        workload = iqm_qiskit_workload(
            circuit,
            workload_id="bad_native_request",
            shots=4,
            capture_semantics=case != "unbound_legacy",
            parameter_bindings={shared: 0.0}
            if case in {"wrong_map", "foreign_modality"}
            else None,
        )
        if case == "wrong_map":
            assert isinstance(workload.semantics, WorkloadSemantics)
            workload = replace(
                workload,
                semantics=replace(
                    workload.semantics,
                    measurement_map=((1, 0), (0, 1)),
                ),
            )
        elif case == "foreign_modality":
            assert isinstance(workload.semantics, WorkloadSemantics)
            workload = replace(
                workload,
                semantics=ModalitySemantics(
                    program_sha256=workload.semantics.program_sha256,
                    modality="analog",
                    native_axes=(0, 1),
                ),
            )
    with pytest.raises(ValueError, match="decoded|binding|bound native|wiring|gate-model"):
        adapter.submit(workload, approval_id="fault-contract-only")
    assert imports == []


@pytest.mark.parametrize(
    "channel",
    [
        "indexed",
        "single_map",
        "two_maps",
        "zero_maps",
        "not_mapping",
        "record",
        "empty_record",
        "two_records",
        "bad_record",
        "empty_counts",
    ],
)
def test_iqm_native_result_channel_contract_and_recovery(channel: str) -> None:
    """SDK channel faults refuse without caching; original count evidence can be recovered."""
    original_counts = {"00": 4}
    reply: object
    indexed_reads: list[int] = []
    if channel == "indexed":

        def indexed_counts(index: int) -> dict[str, int]:
            indexed_reads.append(index)
            return original_counts

        reply = types.SimpleNamespace(get_counts=indexed_counts)
    elif channel in {"single_map", "two_maps", "zero_maps", "not_mapping", "empty_counts"}:
        payload: object = {
            "single_map": [original_counts],
            "two_maps": [original_counts, original_counts],
            "zero_maps": [],
            "not_mapping": "invalid",
            "empty_counts": {},
        }[channel]
        reply = types.SimpleNamespace(get_counts=lambda: payload)
    else:
        row = types.SimpleNamespace(
            data=types.SimpleNamespace(
                counts=original_counts if channel != "bad_record" else "invalid",
            )
        )
        rows = {
            "record": [row],
            "empty_record": [],
            "two_records": [row, row],
            "bad_record": [row],
        }[channel]
        reply = types.SimpleNamespace(results=rows)

    class ReplyJob:
        def __init__(self) -> None:
            self.reply = reply
            self.reads = 0

        def job_id(self) -> str:
            return "iqm-native-reply-channel"

        def result(self, timeout: float | None = None) -> object:
            assert timeout == 9.5
            self.reads += 1
            return self.reply

    provider_job = ReplyJob()

    class ReplyBackend(_NativeIQMTarget):
        def run(self, circuits: list[QuantumCircuit], *, shots: int) -> ReplyJob:
            assert len(circuits) == 1 and shots == 4
            return provider_job

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        IQMHALAdapter(hal.profile("iqm_cloud"), backend=ReplyBackend(), timeout_s=9.5)
    )
    job = hal.submit(
        "iqm_cloud",
        iqm_qiskit_workload(
            _bell_circuit(),
            workload_id="native_reply_channel",
            shots=4,
        ),
        approval_id="fault-contract-only",
    )
    if channel in {"indexed", "single_map", "record"}:
        result = hal.result(job)
        assert result.counts == original_counts and provider_job.reads == 1
        if channel == "indexed":
            assert indexed_reads == [0]
    else:
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            hal.result(job)
        provider_job.reply = types.SimpleNamespace(get_counts=lambda: original_counts)
        result = hal.result(job)
        assert result.counts == original_counts and provider_job.reads == 2
    assert hal.result(job) is result
    original_counts["00"] = 99
    assert result.counts == {"00": 4}


@pytest.mark.parametrize("timeout,level", [(0.0, 1), (-1.0, 1), (9.5, -1), (9.5, 4)])
def test_iqm_invalid_retrieval_and_compilation_settings_refuse_locally(
    timeout: float, level: int
) -> None:
    """Invalid timeout or compilation level cannot construct an IQM client."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("iqm_cloud")
    with pytest.raises(ValueError, match="timeout_s|optimisation_level"):
        IQMHALAdapter(profile, timeout_s=timeout, optimisation_level=level)


def test_iqm_missing_optional_sdk_preserves_loader_cause() -> None:
    """A valid local request surfaces missing IQM SDK without any fallback client."""
    original = ModuleNotFoundError("controlled absent optional IQM SDK")
    calls: list[str] = []

    def missing_sdk(name: str) -> object:
        calls.append(name)
        raise original

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("iqm_cloud")
    adapter = IQMHALAdapter(
        profile, server_url="https://sdk-absent.iqm.invalid", import_module=missing_sdk
    )
    with pytest.raises(ImportError, match="isolated runner") as error:
        adapter.submit(
            iqm_qiskit_workload(_bell_circuit(), workload_id="sdk_absent", shots=4),
            approval_id="fault-contract-only",
        )
    assert error.value.__cause__ is original and calls == ["iqm.qiskit_iqm.iqm_provider"]


def test_iqm_native_backend_factory_legacy_accessor_keeps_explicit_selector() -> None:
    """The legacy provider accessor uses the configured endpoint and compiler target exactly once."""
    calls: list[tuple[str, str | None]] = []
    backend = _FakeIQMBackend()

    class LegacyProvider:
        def __init__(self, url: str, *, quantum_computer: str | None) -> None:
            calls.append((url, quantum_computer))

        def backend(self) -> _FakeIQMBackend:
            return backend

    def importer(name: str) -> object:
        assert name == "iqm.qiskit_iqm.iqm_provider"
        return types.SimpleNamespace(IQMProvider=LegacyProvider)

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("iqm_cloud")
    adapter = IQMHALAdapter(
        profile,
        server_url="https://accessor.iqm.invalid",
        quantum_computer="garnet",
        import_module=importer,
        timeout_s=9.5,
    )
    job = adapter.submit(
        iqm_qiskit_workload(_bell_circuit(), workload_id="legacy_accessor", shots=16),
        approval_id="fault-contract-only",
    )
    assert adapter.result(job).shots == 16
    assert calls == [("https://accessor.iqm.invalid", "garnet")]
    assert job.metadata["quantum_computer"] == "garnet"


@pytest.mark.parametrize("fault", ["result", "cancel", "status", "unknown_status"])
def test_iqm_missing_provider_operations_are_observable_and_recover(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Operation absence refuses without discarding stored source or later count retrieval."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = _FakeIQMBackend()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=backend, timeout_s=9.5)
    hal.register_backend(adapter)
    job = hal.submit(
        "iqm_cloud",
        iqm_qiskit_workload(_bell_circuit(), workload_id="operation_absence", shots=16),
        approval_id="fault-contract-only",
    )
    provider = backend.jobs[0]
    with monkeypatch.context() as injected:
        if fault == "result":
            injected.setattr(provider, "result", None)
            with pytest.raises(TypeError, match="result"):
                hal.result(job)
        elif fault == "cancel":
            injected.setattr(provider, "cancel", None)
            with pytest.raises(ValueError, match="cancellation"):
                hal.cancel(job)
        else:
            injected.setattr(provider, "status", None)
            if fault == "unknown_status":
                injected.delattr(provider, "_status")
            assert hal.status(job) == ("unknown" if fault == "unknown_status" else "completed")
    result = hal.result(job)
    assert result.counts == {"00": 7, "11": 9} and result.shots == 16
    assert hal.result(job) is result


@pytest.mark.parametrize("kind", ["callable", "class", "empty", "control"])
def test_iqm_backend_identity_forms_admit_or_refuse_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """Legacy backend identities retain canonical names; malformed present names never run."""
    backend = _FakeIQMBackend()
    value: object = {
        "callable": lambda: "fake_garnet",
        "class": None,
        "empty": "",
        "control": "bad\nname",
    }[kind]
    monkeypatch.setattr(backend, "name", value)
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("iqm_cloud")
    adapter = IQMHALAdapter(profile, backend=backend, timeout_s=9.5, compile_circuit=False)
    workload = iqm_qiskit_workload(_bell_circuit(), workload_id="native_identity_form", shots=16)
    if kind in {"empty", "control"}:
        with pytest.raises(ValueError, match="IQM backend name"):
            adapter.submit(workload, approval_id="fault-contract-only")
        assert backend.received_shots == []
    else:
        job = adapter.submit(workload, approval_id="fault-contract-only")
        assert job.metadata["backend_name"] == (
            "_FakeIQMBackend" if kind == "class" else "fake_garnet"
        )
        assert adapter.result(job).shots == 16 and backend.received_shots == [16]


@pytest.mark.parametrize("ir", ["openqasm3", "quil"])
def test_iqm_direct_decoder_refuses_foreign_ir_even_when_custom_profile_advertises_it(
    ir: str,
) -> None:
    """A broader declaration cannot bypass the direct adapter's native QPY-only boundary."""
    original = HardwareAbstractionLayer.with_builtin_profiles().profile("iqm_cloud")
    profile = replace(original, ir_formats=(*original.ir_formats, ir))
    imports: list[str] = []

    def forbidden_import(name: str) -> object:
        imports.append(name)
        raise AssertionError("foreign IR cannot construct a provider")

    adapter = IQMHALAdapter(
        profile, server_url="https://ir-refusal.iqm.invalid", import_module=forbidden_import
    )
    workload = QuantumWorkload(
        workload_id="unsupported_native_ir",
        ir_format=ir,
        program="original foreign source",
        n_qubits=2,
        shots=4,
    )
    with pytest.raises(ValueError, match="IQM direct adapter requires qiskit_qpy"):
        adapter.submit(workload, approval_id="fault-contract-only")
    assert imports == []
