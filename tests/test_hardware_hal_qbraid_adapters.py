# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- qBraid HAL adapter tests
"""Tests for qBraid runtime execution behind the provider-neutral HAL."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer
from scpn_quantum_control.hardware.hal_qbraid import (
    QbraidRuntimeHALAdapter,
    qbraid_program_to_workload,
)


def test_qbraid_adapter_uses_injected_device_and_approval_gate() -> None:
    """qBraid cloud routes should be injectable and approval-gated."""

    class FakeResultData:
        def get_counts(self) -> dict[str, int]:
            return {"00": 5, "11": 7}

    class FakeResult:
        data = FakeResultData()

    class FakeJob:
        id = "qbraid-job-1"

        def status(self) -> str:
            return "COMPLETED"

        def result(self) -> FakeResult:
            return FakeResult()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeDevice:
        id = "ionq_qpu.aria-1"

        def run(self, run_input: str, *, shots: int, name: str, metadata: dict[str, object]):
            assert run_input.startswith("OPENQASM 3.0")
            assert shots == 12
            assert name == "qbraid_bell"
            assert metadata["workload_id"] == "qbraid_bell"
            return FakeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QbraidRuntimeHALAdapter(hal.profile("qbraid_ionq"), device=FakeDevice()))
    workload = qbraid_program_to_workload(
        "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];",
        workload_id="qbraid_bell",
        ir_format="openqasm3",
        n_qubits=2,
        shots=12,
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("qbraid_ionq", workload)

    job = hal.submit("qbraid_ionq", workload, approval_id="approved-qbraid")
    result = hal.result(job)

    assert job.status == "submitted"
    assert result.counts == {"00": 5, "11": 7}
    assert result.shots == 12
    assert result.metadata["execution_mode"] == "qbraid_runtime"
    assert result.metadata["approval_id"] == "approved-qbraid"
    assert hal.status(job) == "completed"


def test_qbraid_adapter_can_load_device_from_provider() -> None:
    """Provider injection should resolve devices by qBraid device id."""

    class FakeResult:
        def get_counts(self) -> dict[str, int]:
            return {"0": 3}

    class FakeJob:
        job_id = "provider-job-1"

        def result(self) -> FakeResult:
            return FakeResult()

    class FakeDevice:
        id = "ionq_qpu.aria-1"

        def run(self, run_input: str, *, shots: int, name: str, metadata: dict[str, object]):
            assert run_input == "OPENQASM 3.0;\nqubit[1] q;"
            assert shots == 3
            assert name == "provider_route"
            assert metadata["approval_id"] == "approved-provider"
            return FakeJob()

    class FakeProvider:
        def get_device(self, device_id: str) -> FakeDevice:
            assert device_id == "ionq_qpu.aria-1"
            return FakeDevice()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QbraidRuntimeHALAdapter(
            hal.profile("qbraid_ionq"),
            provider=FakeProvider(),
            device_id="ionq_qpu.aria-1",
        )
    )
    workload = qbraid_program_to_workload(
        "OPENQASM 3.0;\nqubit[1] q;",
        workload_id="provider_route",
        ir_format="openqasm3",
        n_qubits=1,
        shots=3,
    )

    result = hal.result(hal.submit("qbraid_ionq", workload, approval_id="approved-provider"))

    assert result.counts == {"0": 3}


def test_qbraid_adapter_requires_device_route() -> None:
    """qBraid construction should fail closed without a device or provider route."""

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("qbraid_ionq")

    with pytest.raises(ValueError, match="device"):
        QbraidRuntimeHALAdapter(profile)


def test_qbraid_adapter_rejects_provider_job_without_id() -> None:
    """qBraid adapter should fail closed when device.run() omits job id."""

    class FakeResultData:
        def get_counts(self) -> dict[str, int]:
            return {"0": 1}

    class FakeResult:
        data = FakeResultData()

    class FakeJob:
        def status(self) -> str:
            return "COMPLETED"

        def result(self) -> FakeResult:
            return FakeResult()

    class FakeDevice:
        def run(self, run_input: str, *, shots: int, name: str, metadata: dict[str, object]):
            del run_input, shots, name, metadata
            return FakeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QbraidRuntimeHALAdapter(hal.profile("qbraid_ionq"), device=FakeDevice()))
    workload = qbraid_program_to_workload(
        "OPENQASM 3.0;\nqubit[1] q;",
        workload_id="qbraid_missing_job_id",
        ir_format="openqasm3",
        n_qubits=1,
        shots=1,
    )

    with pytest.raises(ValueError, match="without an id"):
        hal.submit("qbraid_ionq", workload, approval_id="approved-qbraid")


def test_qbraid_adapter_normalises_provider_status_tokens() -> None:
    """qBraid status values should map to canonical HAL status values."""

    from scpn_quantum_control.hardware import hal_qbraid as qbraid_mod

    assert qbraid_mod._normalise_status("CANCELED") == "cancelled"
    assert qbraid_mod._normalise_status("FINISHED") == "completed"


def test_qbraid_provider_job_id_rejects_control_characters() -> None:
    """qBraid provider job identifiers must reject control-character payloads."""

    from scpn_quantum_control.hardware import hal_qbraid as qbraid_mod

    class BadJob:
        id = "qbraid-job-\n1"

    with pytest.raises(ValueError, match="provider job id"):
        qbraid_mod._job_id(BadJob())


def test_qbraid_provider_job_id_trims_padding() -> None:
    """qBraid provider job identifiers should be canonicalised by trimming padding."""

    from scpn_quantum_control.hardware import hal_qbraid as qbraid_mod

    class PaddedJob:
        id = "  qbraid-job-42  "

    assert qbraid_mod._job_id(PaddedJob()) == "qbraid-job-42"


def test_qbraid_adapter_rejects_shot_mismatch() -> None:
    """qBraid adapter must fail closed when decoded counts diverge from requested shots."""

    class FakeResultData:
        def get_counts(self) -> dict[str, int]:
            return {"00": 5, "11": 7}

    class FakeResult:
        data = FakeResultData()

    class FakeJob:
        id = "qbraid-job-shot-mismatch"

        def status(self) -> str:
            return "COMPLETED"

        def result(self) -> FakeResult:
            return FakeResult()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeDevice:
        id = "ionq_qpu.aria-1"

        def run(self, run_input: str, *, shots: int, name: str, metadata: dict[str, object]):
            del run_input, name, metadata
            assert shots == 13
            return FakeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QbraidRuntimeHALAdapter(hal.profile("qbraid_ionq"), device=FakeDevice()))
    workload = qbraid_program_to_workload(
        "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\n",
        workload_id="qbraid_shot_mismatch",
        ir_format="openqasm3",
        n_qubits=2,
        shots=13,
    )

    job = hal.submit("qbraid_ionq", workload, approval_id="approved-qbraid")
    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.result(job)
