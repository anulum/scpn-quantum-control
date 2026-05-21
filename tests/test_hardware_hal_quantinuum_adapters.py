# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Quantinuum direct HAL adapter tests
"""Tests for Quantinuum direct execution behind the provider-neutral HAL."""

from __future__ import annotations

from typing import Any

import pytest

from scpn_quantum_control.hardware import hal_quantinuum as quantinuum_mod
from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_quantinuum import (
    QuantinuumCloudHALAdapter,
    quantinuum_tket_workload,
)


class _FakeQuantinuumStatus:
    def __init__(self, status: str) -> None:
        self.status = status


class _FakeQuantinuumResult:
    def __init__(self, counts: dict[tuple[int, ...], int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[tuple[int, ...], int]:
        return self._counts


class _FakeQuantinuumBackend:
    def __init__(self) -> None:
        self.compiled: list[Any] = []
        self.processed: list[dict[str, Any]] = []
        self.cancelled: list[str] = []

    def get_compiled_circuit(self, circuit: Any) -> str:
        self.compiled.append(circuit)
        return f"compiled::{circuit}"

    def process_circuit(self, circuit: Any, *, n_shots: int) -> str:
        self.processed.append({"circuit": circuit, "n_shots": n_shots})
        return "quantinuum-handle-1"

    def circuit_status(self, handle: str) -> _FakeQuantinuumStatus:
        assert handle == "quantinuum-handle-1"
        return _FakeQuantinuumStatus("COMPLETED")

    def get_result(self, handle: str) -> _FakeQuantinuumResult:
        assert handle == "quantinuum-handle-1"
        return _FakeQuantinuumResult({(0, 1): 2, (1, 1): 1})

    def cancel(self, handle: str) -> None:
        self.cancelled.append(handle)


class _FakeNoCompileQuantinuumBackend:
    def __init__(self) -> None:
        self.processed: list[dict[str, Any]] = []

    def process_circuit(self, circuit: Any, *, n_shots: int) -> str:
        self.processed.append({"circuit": circuit, "n_shots": n_shots})
        return "quantinuum-precompiled-handle"

    def get_result(self, handle: str) -> _FakeQuantinuumResult:
        assert handle == "quantinuum-precompiled-handle"
        return _FakeQuantinuumResult({(1,): 1})


class _FakeCircuitObject:
    def to_dict(self) -> dict[str, object]:
        return {"name": "from_object", "qubits": 1}


class _FakeCircuitClass:
    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> str:
        return f"circuit::{payload['name']}"


class _FakePytketCircuitModule:
    Circuit = _FakeCircuitClass


class _FakeQuantinuumModule:
    def __init__(self) -> None:
        self.backend = _FakeQuantinuumBackend()
        self.machines: list[str] = []

    def QuantinuumBackend(self, machine: str) -> _FakeQuantinuumBackend:
        self.machines.append(machine)
        return self.backend


class _FakeOutcomeKey:
    def to_list(self) -> list[int]:
        return [1, 0]


class _FakeStatusName:
    name = "QUEUED"


def test_quantinuum_adapter_submits_status_results_and_cancel() -> None:
    """Quantinuum adapter should compile, submit, track handle, and normalise counts."""

    backend = _FakeQuantinuumBackend()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuantinuumCloudHALAdapter(
            hal.profile("quantinuum_cloud"),
            backend=backend,
            machine="H1-1E",
            circuit_factory=lambda workload: workload.program,
        )
    )
    workload = quantinuum_tket_workload(
        {"name": "bell", "qubits": 2, "ops": ["H 0", "CX 0 1"]},
        workload_id="quantinuum_bell",
        n_qubits=2,
        shots=3,
        metadata={"lane": "hal"},
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("quantinuum_cloud", workload)

    job = hal.submit("quantinuum_cloud", workload, approval_id="approved-quantinuum")
    status = hal.status(job)
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert job.status == "submitted"
    assert status == "completed"
    assert result.counts == {"01": 2, "11": 1}
    assert result.shots == 3
    assert result.metadata["execution_mode"] == "quantinuum_pytket"
    assert cancelled.status == "cancelled"
    assert backend.compiled == [workload.program]
    assert backend.processed == [{"circuit": f"compiled::{workload.program}", "n_shots": 3}]
    assert backend.cancelled == ["quantinuum-handle-1"]
    assert job.metadata == {
        "approval_id": "approved-quantinuum",
        "provider_job_id": "quantinuum-handle-1",
        "execution_mode": "quantinuum_pytket",
        "machine": "H1-1E",
        "ir_format": "tket",
        "n_qubits": 2,
        "shots": 3,
    }


def test_quantinuum_adapter_rejects_non_tket_payloads() -> None:
    """Direct Quantinuum execution should fail closed until translation into pytket is explicit."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuantinuumCloudHALAdapter(
            hal.profile("quantinuum_cloud"),
            backend=_FakeQuantinuumBackend(),
            circuit_factory=lambda workload: workload.program,
        )
    )
    workload = QuantumWorkload(
        workload_id="bad_quantinuum",
        ir_format="openqasm3",
        program="OPENQASM 3.0;\nqubit[1] q;\nh q[0];",
        n_qubits=1,
        shots=8,
    )

    with pytest.raises(ValueError, match="tket workloads"):
        hal.submit("quantinuum_cloud", workload, approval_id="approved-quantinuum")


def test_quantinuum_adapter_rejects_wrong_profile() -> None:
    """The concrete adapter must not attach to a non-Quantinuum profile."""

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("local_statevector")

    with pytest.raises(ValueError, match="quantinuum_cloud"):
        QuantinuumCloudHALAdapter(profile, backend=_FakeQuantinuumBackend())


def test_quantinuum_default_lazy_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default route should load pytket factories lazily and construct circuits from JSON."""

    fake_quantinuum = _FakeQuantinuumModule()

    def fake_import(name: str) -> object:
        if name == "pytket.extensions.quantinuum":
            return fake_quantinuum
        if name == "pytket.circuit":
            return _FakePytketCircuitModule()
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(quantinuum_mod, "import_module", fake_import)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuantinuumCloudHALAdapter(
            hal.profile("quantinuum_cloud"),
            machine="H1-1E",
        )
    )

    result = hal.result(
        hal.submit(
            "quantinuum_cloud",
            quantinuum_tket_workload(
                '{"name":"json_circuit","qubits":2}',
                workload_id="quantinuum_default",
                n_qubits=2,
                shots=3,
            ),
            approval_id="approved-quantinuum",
        )
    )

    assert fake_quantinuum.machines == ["H1-1E"]
    assert fake_quantinuum.backend.compiled == ["circuit::json_circuit"]
    assert result.counts == {"01": 2, "11": 1}


def test_quantinuum_workload_accepts_circuit_objects() -> None:
    """Circuit objects should be serialised through their pytket-style to_dict method."""

    workload = quantinuum_tket_workload(
        _FakeCircuitObject(),
        workload_id="quantinuum_object",
        n_qubits=1,
        shots=1,
    )

    assert '"from_object"' in workload.program


def test_quantinuum_adapter_can_submit_precompiled_payloads() -> None:
    """compile_circuit=False should forward already compiled provider payloads."""

    backend = _FakeNoCompileQuantinuumBackend()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuantinuumCloudHALAdapter(
            hal.profile("quantinuum_cloud"),
            backend=backend,
            circuit_factory=lambda workload: workload.program,
            compile_circuit=False,
        )
    )

    result = hal.result(
        hal.submit(
            "quantinuum_cloud",
            quantinuum_tket_workload(
                {"name": "precompiled", "qubits": 1},
                workload_id="quantinuum_precompiled",
                n_qubits=1,
                shots=1,
            ),
            approval_id="approved-quantinuum",
        )
    )

    assert result.counts == {"1": 1}
    assert backend.processed == [
        {
            "circuit": quantinuum_tket_workload(
                {"name": "precompiled", "qubits": 1},
                workload_id="quantinuum_precompiled",
                n_qubits=1,
                shots=1,
            ).program,
            "n_shots": 1,
        }
    ]


def test_quantinuum_adapter_rejects_missing_compile_when_enabled() -> None:
    """Compilation remains explicit because Quantinuum expects compiled circuits."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuantinuumCloudHALAdapter(
            hal.profile("quantinuum_cloud"),
            backend=_FakeNoCompileQuantinuumBackend(),
            circuit_factory=lambda workload: workload.program,
        )
    )

    with pytest.raises(TypeError, match="get_compiled_circuit"):
        hal.submit(
            "quantinuum_cloud",
            quantinuum_tket_workload(
                {"name": "no_compile", "qubits": 1},
                workload_id="quantinuum_no_compile",
                n_qubits=1,
                shots=1,
            ),
            approval_id="approved-quantinuum",
        )


def test_quantinuum_adapter_rejects_unknown_jobs() -> None:
    """Unknown job handles should not fabricate provider state."""

    adapter = QuantinuumCloudHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("quantinuum_cloud"),
        backend=_FakeQuantinuumBackend(),
    )
    unknown = quantinuum_mod.QuantumJobRef(
        job_id="quantinuum_cloud:missing:000000000000",
        backend_id="quantinuum_cloud",
        workload_id="missing",
        status="submitted",
    )

    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.status(unknown)
    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.result(unknown)
    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.cancel(unknown)


def test_quantinuum_adapter_rejects_shot_mismatch() -> None:
    """Quantinuum result decoding must fail closed when counts disagree with expected shots."""

    backend = _FakeQuantinuumBackend()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = QuantinuumCloudHALAdapter(
        hal.profile("quantinuum_cloud"),
        backend=backend,
        machine="H1-1E",
        circuit_factory=lambda workload: workload.program,
    )
    hal.register_backend(adapter)
    workload = quantinuum_tket_workload(
        {"name": "shot_mismatch", "qubits": 2},
        workload_id="quantinuum_shot_mismatch",
        n_qubits=2,
        shots=3,
    )
    job = hal.submit("quantinuum_cloud", workload, approval_id="approved-quantinuum")
    adapter._jobs[job.job_id] = adapter._jobs[job.job_id].__class__(
        job_id=job.job_id,
        backend_id=job.backend_id,
        workload_id=job.workload_id,
        status=job.status,
        metadata={**job.metadata, "shots": 4},
    )

    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.result(job)


def test_quantinuum_adapter_rejects_invalid_provider_handle() -> None:
    """Quantinuum adapter should fail closed when provider handle is missing."""

    class InvalidHandleBackend(_FakeQuantinuumBackend):
        def process_circuit(self, circuit: Any, *, n_shots: int) -> str:
            del circuit, n_shots
            return None  # type: ignore[return-value]

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuantinuumCloudHALAdapter(
            hal.profile("quantinuum_cloud"),
            backend=InvalidHandleBackend(),
            machine="H1-1E",
            circuit_factory=lambda workload: workload.program,
        )
    )
    workload = quantinuum_tket_workload(
        {"name": "bad_handle", "qubits": 2, "ops": ["H 0"]},
        workload_id="quantinuum_bad_handle",
        n_qubits=2,
        shots=1,
    )

    with pytest.raises(ValueError, match="invalid provider handle"):
        hal.submit("quantinuum_cloud", workload, approval_id="approved-quantinuum")


def test_quantinuum_adapter_rejects_opaque_provider_handle_objects() -> None:
    """Opaque object repr handles must fail closed to preserve lineage quality."""

    class OpaqueHandleBackend(_FakeQuantinuumBackend):
        def process_circuit(self, circuit: Any, *, n_shots: int) -> object:
            del circuit, n_shots
            return object()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuantinuumCloudHALAdapter(
            hal.profile("quantinuum_cloud"),
            backend=OpaqueHandleBackend(),
            machine="H1-1E",
            circuit_factory=lambda workload: workload.program,
        )
    )
    workload = quantinuum_tket_workload(
        {"name": "opaque_handle", "qubits": 2, "ops": ["H 0"]},
        workload_id="quantinuum_opaque_handle",
        n_qubits=2,
        shots=1,
    )

    with pytest.raises(ValueError, match="invalid provider handle"):
        hal.submit("quantinuum_cloud", workload, approval_id="approved-quantinuum")


def test_quantinuum_adapter_accepts_structured_handle_identifier() -> None:
    """Structured provider handles should expose lineage via id attributes."""

    class HandleWithId:
        def __init__(self) -> None:
            self.id = "quantinuum-structured-handle-1"

    class StructuredHandleBackend(_FakeQuantinuumBackend):
        def process_circuit(self, circuit: Any, *, n_shots: int) -> HandleWithId:
            del circuit, n_shots
            return HandleWithId()

        def circuit_status(self, handle: HandleWithId) -> _FakeQuantinuumStatus:
            assert handle.id == "quantinuum-structured-handle-1"
            return _FakeQuantinuumStatus("COMPLETED")

        def get_result(self, handle: HandleWithId) -> _FakeQuantinuumResult:
            assert handle.id == "quantinuum-structured-handle-1"
            return _FakeQuantinuumResult({(0,): 1})

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuantinuumCloudHALAdapter(
            hal.profile("quantinuum_cloud"),
            backend=StructuredHandleBackend(),
            machine="H1-1E",
            circuit_factory=lambda workload: workload.program,
        )
    )
    workload = quantinuum_tket_workload(
        {"name": "structured_handle", "qubits": 1, "ops": ["X 0"]},
        workload_id="quantinuum_structured_handle",
        n_qubits=1,
        shots=1,
    )

    job = hal.submit("quantinuum_cloud", workload, approval_id="approved-quantinuum")
    assert job.metadata["provider_job_id"] == "quantinuum-structured-handle-1"
    assert job.job_id.startswith("quantinuum_cloud:quantinuum_structured_handle:")


@pytest.mark.parametrize(
    ("raw_counts", "message"),
    [
        ({}, "did not contain counts"),
        ({(0,): -1}, "non-negative"),
        ({(): 1}, "must not be empty"),
        ({(2,): 1}, "binary"),
        ({object(): 1}, "bitstrings"),
    ],
)
def test_quantinuum_count_validation_rejects_malformed_provider_results(
    raw_counts: dict[object, int], message: str
) -> None:
    """Malformed provider counts should fail before HAL results are reported."""

    with pytest.raises(ValueError, match=message):
        quantinuum_mod._normalise_counts(raw_counts)


def test_quantinuum_counts_accept_provider_key_variants() -> None:
    """pytket tuple, string, and to_list keys should normalise into bitstrings."""

    counts = quantinuum_mod._normalise_counts({(0, 1): 1, "11": 2, _FakeOutcomeKey(): 3})

    assert counts == {"01": 1, "11": 2, "10": 3}


def test_quantinuum_status_normalisation_accepts_enum_names() -> None:
    """Provider status objects should be reduced to HAL-safe status tokens."""

    assert quantinuum_mod._normalise_status(_FakeStatusName()) == "queued"
    assert quantinuum_mod._normalise_status("SUCCEEDED") == "completed"
    assert quantinuum_mod._normalise_status("CANCELED") == "cancelled"


def test_quantinuum_provider_job_id_rejects_control_characters() -> None:
    """Quantinuum provider identifiers must reject control-character payloads."""

    from scpn_quantum_control.hardware import hal_quantinuum as quantinuum_mod

    class BadHandle:
        job_id = "quantinuum-job-\n1"

    with pytest.raises(ValueError, match="provider job id"):
        quantinuum_mod._provider_job_id(BadHandle())


def test_quantinuum_provider_job_id_trims_padding() -> None:
    """Quantinuum provider identifiers should be canonicalised by trimming padding."""

    from scpn_quantum_control.hardware import hal_quantinuum as quantinuum_mod

    class PaddedHandle:
        job_id = "  quantinuum-job-42  "

    assert quantinuum_mod._provider_job_id(PaddedHandle()) == "quantinuum-job-42"


def test_quantinuum_default_dependency_errors_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing pytket packages should produce route-specific dependency errors."""

    def fail_import(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(quantinuum_mod, "import_module", fail_import)
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("quantinuum_cloud")

    with pytest.raises(RuntimeError, match="pytket"):
        QuantinuumCloudHALAdapter(profile, machine="H1-1E").submit(
            quantinuum_tket_workload(
                {"name": "missing_pytket", "qubits": 1},
                workload_id="quantinuum_missing_pytket",
                n_qubits=1,
                shots=1,
            ),
            approval_id="approved-quantinuum",
        )
    with pytest.raises(RuntimeError, match="pytket-quantinuum"):
        QuantinuumCloudHALAdapter(
            profile,
            machine="H1-1E",
            circuit_factory=lambda workload: workload.program,
        ).submit(
            quantinuum_tket_workload(
                {"name": "missing_backend", "qubits": 1},
                workload_id="quantinuum_missing_backend",
                n_qubits=1,
                shots=1,
            ),
            approval_id="approved-quantinuum",
        )


def test_quantinuum_default_circuit_factory_rejects_malformed_payloads() -> None:
    """Malformed tket payloads should fail before provider submission."""

    with pytest.raises(ValueError, match="valid JSON"):
        quantinuum_mod._default_circuit_factory(
            QuantumWorkload(
                workload_id="bad_json",
                ir_format="tket",
                program="{",
                n_qubits=1,
                shots=1,
            )
        )
    with pytest.raises(ValueError, match="schema"):
        quantinuum_mod._default_circuit_factory(
            QuantumWorkload(
                workload_id="bad_schema",
                ir_format="tket",
                program='{"schema":"bad","circuit":{}}',
                n_qubits=1,
                shots=1,
            )
        )
    with pytest.raises(ValueError, match="JSON object"):
        quantinuum_mod._default_circuit_factory(
            QuantumWorkload(
                workload_id="bad_circuit",
                ir_format="tket",
                program='{"schema":"scpn.quantinuum.tket_circuit.v1","circuit":[]}',
                n_qubits=1,
                shots=1,
            )
        )


def test_quantinuum_adapter_requires_execution_route() -> None:
    """Construction should fail closed without an injected backend or machine name."""

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("quantinuum_cloud")

    with pytest.raises(ValueError, match="backend or machine"):
        QuantinuumCloudHALAdapter(profile)
