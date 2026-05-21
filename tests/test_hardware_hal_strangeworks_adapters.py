# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Strangeworks HAL adapter tests
"""Tests for Strangeworks Compute execution behind the provider-neutral HAL."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer
from scpn_quantum_control.hardware.hal_strangeworks import (
    StrangeworksComputeHALAdapter,
    strangeworks_program_to_workload,
)


def test_strangeworks_adapter_uses_injected_backend_and_approval_gate() -> None:
    """Strangeworks Compute routes should stay injected and approval-gated."""

    class FakeJob:
        id = "sw-job-1"

        def status(self) -> str:
            return "COMPLETED"

        def result(self) -> dict[str, dict[str, int]]:
            return {"counts": {"00": 4, "11": 6}}

        def cancel(self) -> None:
            self.cancelled = True

    class FakeBackend:
        id = "rigetti.qvm"

        def run(self, run_input: str, *, shots: int, name: str, metadata: dict[str, object]):
            assert run_input == "DECLARE ro BIT[2]"
            assert shots == 10
            assert name == "sw_rigetti"
            assert metadata["approval_id"] == "approved-sw"
            assert metadata["broker"] == "strangeworks"
            return FakeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        StrangeworksComputeHALAdapter(
            hal.profile("strangeworks_compute"),
            backend=FakeBackend(),
        )
    )
    workload = strangeworks_program_to_workload(
        "DECLARE ro BIT[2]",
        workload_id="sw_rigetti",
        ir_format="quil",
        n_qubits=2,
        shots=10,
        metadata={"provider": "rigetti"},
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("strangeworks_compute", workload)

    job = hal.submit("strangeworks_compute", workload, approval_id="approved-sw")
    result = hal.result(job)

    assert job.status == "submitted"
    assert result.counts == {"00": 4, "11": 6}
    assert result.shots == 10
    assert result.metadata["execution_mode"] == "strangeworks_compute"
    assert result.metadata["backend_id"] == "rigetti.qvm"
    assert hal.status(job) == "completed"


def test_strangeworks_adapter_can_load_backend_from_workspace() -> None:
    """Workspace injection should resolve dynamic Strangeworks resources by id."""

    class FakeJob:
        job_id = "workspace-job-1"

        def result(self) -> dict[str, dict[str, int]]:
            return {"measurement_counts": {"0": 3}}

    class FakeBackend:
        name = "ionq.simulator"

        def run(self, run_input: str, *, shots: int, name: str, metadata: dict[str, object]):
            assert run_input == "OPENQASM 3.0;"
            assert shots == 3
            assert name == "workspace_route"
            assert metadata["approval_id"] == "approved-workspace"
            return FakeJob()

    class FakeWorkspace:
        def get_backend(self, backend_id: str) -> FakeBackend:
            assert backend_id == "ionq.simulator"
            return FakeBackend()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        StrangeworksComputeHALAdapter(
            hal.profile("strangeworks_compute"),
            workspace=FakeWorkspace(),
            backend_id="ionq.simulator",
        )
    )
    workload = strangeworks_program_to_workload(
        "OPENQASM 3.0;",
        workload_id="workspace_route",
        ir_format="openqasm3",
        n_qubits=1,
        shots=3,
    )

    result = hal.result(
        hal.submit("strangeworks_compute", workload, approval_id="approved-workspace")
    )

    assert result.counts == {"0": 3}


def test_strangeworks_adapter_requires_backend_route() -> None:
    """Strangeworks construction should fail closed without a backend route."""

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("strangeworks_compute")

    with pytest.raises(ValueError, match="backend"):
        StrangeworksComputeHALAdapter(profile)


def test_strangeworks_adapter_rejects_provider_job_without_id() -> None:
    """Strangeworks adapter should fail closed when backend.run() omits job id."""

    class FakeJob:
        def result(self) -> dict[str, dict[str, int]]:
            return {"counts": {"0": 1}}

    class FakeBackend:
        def run(self, run_input: str, *, shots: int, name: str, metadata: dict[str, object]):
            del run_input, shots, name, metadata
            return FakeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        StrangeworksComputeHALAdapter(
            hal.profile("strangeworks_compute"),
            backend=FakeBackend(),
        )
    )
    workload = strangeworks_program_to_workload(
        "OPENQASM 3.0;",
        workload_id="sw_missing_job_id",
        ir_format="openqasm3",
        n_qubits=1,
        shots=1,
    )

    with pytest.raises(ValueError, match="without an id"):
        hal.submit("strangeworks_compute", workload, approval_id="approved-sw")


def test_strangeworks_adapter_normalises_provider_status_tokens() -> None:
    """Strangeworks status values should map to canonical HAL status values."""

    from scpn_quantum_control.hardware import hal_strangeworks as sw_mod

    assert sw_mod._normalise_status("CANCELED") == "cancelled"
    assert sw_mod._normalise_status("SUCCESS") == "completed"
