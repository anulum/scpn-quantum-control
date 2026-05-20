# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Azure Quantum HAL adapter tests
"""Tests for Azure Quantum adapters behind the provider-neutral HAL."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer
from scpn_quantum_control.hardware.hal_azure import (
    AzureQuantumHALAdapter,
    azure_openqasm3_to_workload,
)


def test_azure_quantum_adapter_uses_injected_target_and_approval_gate() -> None:
    """Azure targets should be injectable and approval-gated."""

    class FakeDetails:
        status = "Succeeded"

    class FakeJob:
        id = "azure-job-1"
        details = FakeDetails()

        def get_results(self):
            return {"histogram": {"00": 7, "11": 9}}

        def cancel(self) -> None:
            self.cancelled = True

    class FakeTarget:
        name = "ionq.simulator"

        def submit(self, input_data, name: str, shots: int, input_params=None, **kwargs):
            assert input_data.startswith("OPENQASM 3.0")
            assert name == "azure_bell"
            assert shots == 16
            assert input_params == {"shots": 16}
            assert kwargs["provider"] == "ionq"
            return FakeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        AzureQuantumHALAdapter(
            hal.profile("azure_quantum_ionq_simulator"),
            target=FakeTarget(),
            input_params_factory=lambda workload: {"shots": workload.shots},
            submit_kwargs={"provider": "ionq"},
        )
    )
    workload = azure_openqasm3_to_workload(
        "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];",
        workload_id="azure_bell",
        n_qubits=2,
        shots=16,
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("azure_quantum_ionq_simulator", workload)

    job = hal.submit("azure_quantum_ionq_simulator", workload, approval_id="approved-azure")
    result = hal.result(job)

    assert job.status == "submitted"
    assert result.status == "completed"
    assert result.counts == {"00": 7, "11": 9}
    assert result.metadata["execution_mode"] == "azure_quantum"
    assert result.metadata["approval_id"] == "approved-azure"
    assert hal.status(job) == "succeeded"


def test_azure_adapter_requires_target_or_factory() -> None:
    """Azure adapter construction should fail closed without a target route."""

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("azure_quantum_rigetti_qvm")

    with pytest.raises(ValueError, match="target"):
        AzureQuantumHALAdapter(profile)
