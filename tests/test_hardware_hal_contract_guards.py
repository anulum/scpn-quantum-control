# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Contract-guard tests for the hardware abstraction layer
"""Fail-closed contract tests for the provider-neutral hardware abstraction layer.

Covers the token/tuple/metadata validators, profile and workload guards, the
job-result count contract, the local simulator job-handle errors, and the
HAL registry, approval and delegation guards.
"""

from __future__ import annotations

from typing import Any

import pytest

from scpn_quantum_control.hardware.hal import (
    BackendCapabilities,
    BackendProfile,
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumJobRef,
    QuantumJobResult,
    QuantumWorkload,
)


def _capabilities(
    *, supports_shots: bool = True, supports_cancellation: bool = True
) -> BackendCapabilities:
    return BackendCapabilities(
        supports_shots=supports_shots,
        supports_counts=True,
        supports_statevector=False,
        supports_mid_circuit_measurement=False,
        supports_analog=False,
        supports_pulse=False,
        max_qubits=4,
        supports_cancellation=supports_cancellation,
    )


def _local_profile(
    backend_id: str = "local_test",
    *,
    supports_shots: bool = True,
    supports_cancellation: bool = True,
) -> BackendProfile:
    return BackendProfile(
        backend_id=backend_id,
        provider="test_provider",
        broker="direct",
        modality="gate_model",
        sdk_package="test-sdk",
        ir_formats=("openqasm3",),
        capabilities=_capabilities(
            supports_shots=supports_shots, supports_cancellation=supports_cancellation
        ),
    )


def _cloud_profile() -> BackendProfile:
    return BackendProfile(
        backend_id="cloud_test",
        provider="test_provider",
        broker="direct",
        modality="gate_model",
        sdk_package="test-sdk",
        ir_formats=("openqasm3",),
        is_cloud=True,
        submit_requires_approval=True,
        capabilities=_capabilities(),
    )


def _workload(workload_id: str = "w1", *, shots: int = 8) -> QuantumWorkload:
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="openqasm3",
        program="OPENQASM 3.0;",
        n_qubits=2,
        shots=shots,
    )


@pytest.mark.parametrize("field", ["n_qubits", "shots"])
@pytest.mark.parametrize("bad", [True, False, 1.0, float("nan"), float("inf"), None, "1", 0, -1])
def test_workload_rejects_nonpositive_or_noninteger_resources(field: str, bad: Any) -> None:
    """Reject malformed resources before any workload can reach an adapter."""
    arguments: dict[str, Any] = {
        "workload_id": "resource-admission",
        "ir_format": "openqasm3",
        "program": "OPENQASM 3;",
        "n_qubits": 2,
        "shots": 16,
    }
    arguments[field] = bad
    with pytest.raises(ValueError, match=field):
        QuantumWorkload(**arguments)


@pytest.mark.parametrize("bad", [True, False, 0.0, 1.0, float("nan"), float("inf"), None, "1", -1])
def test_result_rejects_noninteger_or_negative_shots(bad: Any) -> None:
    """Result metadata must not admit bool, float, nonfinite or missing shot counts."""
    job = QuantumJobRef("resource-job", "local_test", "resource-admission", "completed")
    with pytest.raises(ValueError, match="shots"):
        QuantumJobResult(job, "completed", shots=bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, float("nan"), None, "1", -1])
def test_result_rejects_noninteger_or_negative_count_values(bad: Any) -> None:
    """Counts remain nonnegative integers without boolean observations."""
    job = QuantumJobRef("resource-job", "local_test", "resource-admission", "completed")
    with pytest.raises(ValueError, match="counts"):
        QuantumJobResult(job, "completed", counts={"0": bad})


def test_result_preserves_zero_shot_unknown_total_convention() -> None:
    """Stricter types preserve the existing zero-shot sentinel and integer counts."""
    job = QuantumJobRef("resource-job", "local_test", "resource-admission", "completed")
    result = QuantumJobResult(job, "completed", counts={"0": 0, "1": 2}, shots=0)
    assert result.shots == 0
    assert dict(result.counts) == {"0": 0, "1": 2}


def test_result_rejects_positive_shots_with_no_counts() -> None:
    """A positive observed shot total requires a complete histogram."""
    job = QuantumJobRef("resource-job", "local_test", "resource-admission", "completed")
    with pytest.raises(ValueError, match="counts must sum to shots"):
        QuantumJobResult(job, "completed", counts={}, shots=16)


def test_profile_rejects_empty_ir_formats() -> None:
    """An empty IR-format tuple is rejected."""
    with pytest.raises(ValueError, match="ir_formats must not be empty"):
        BackendProfile(
            backend_id="local_test",
            provider="test_provider",
            broker="direct",
            modality="gate_model",
            sdk_package="test-sdk",
            ir_formats=(),
            capabilities=_capabilities(),
        )


def test_workload_metadata_rejects_non_scalar_value() -> None:
    """Workload metadata values must be JSON-scalar compatible."""
    with pytest.raises(ValueError, match="metadata values must be JSON-scalar compatible"):
        QuantumWorkload(
            workload_id="w1",
            ir_format="openqasm3",
            program="OPENQASM 3.0;",
            n_qubits=1,
            metadata={"bad": [1, 2]},
        )


def test_profile_rejects_invalid_region_token() -> None:
    """A region that is not an identifier token is rejected."""
    with pytest.raises(ValueError, match="region must be a non-empty identifier token"):
        BackendProfile(
            backend_id="local_test",
            provider="test_provider",
            broker="direct",
            modality="gate_model",
            sdk_package="test-sdk",
            ir_formats=("openqasm3",),
            region="bad region",
            capabilities=_capabilities(),
        )


def test_cloud_profile_must_require_approval() -> None:
    """A cloud profile that does not require submission approval is rejected."""
    with pytest.raises(
        ValueError, match="cloud profiles must require explicit submission approval"
    ):
        BackendProfile(
            backend_id="cloud_test",
            provider="test_provider",
            broker="direct",
            modality="gate_model",
            sdk_package="test-sdk",
            ir_formats=("openqasm3",),
            is_cloud=True,
            submit_requires_approval=False,
            capabilities=_capabilities(),
        )


def test_workload_rejects_blank_program() -> None:
    """A whitespace-only program is rejected."""
    with pytest.raises(ValueError, match="program must be non-empty"):
        QuantumWorkload(workload_id="w1", ir_format="openqasm3", program="   ", n_qubits=1)


def _job_ref() -> QuantumJobRef:
    return QuantumJobRef(
        job_id="job1", backend_id="local_test", workload_id="w1", status="submitted"
    )


def test_job_result_rejects_negative_shots() -> None:
    """A negative shot count on a result is rejected."""
    with pytest.raises(ValueError, match="shots must be non-negative"):
        QuantumJobResult(job=_job_ref(), status="completed", shots=-1)


def test_job_result_rejects_non_bitstring_count_key() -> None:
    """A count key that is not a non-empty bitstring is rejected."""
    with pytest.raises(ValueError, match="counts keys must be non-empty bitstrings"):
        QuantumJobResult(job=_job_ref(), status="completed", counts={"": 1}, shots=1)


def test_job_result_rejects_negative_count_value() -> None:
    """A negative count value is rejected."""
    with pytest.raises(ValueError, match="counts values must be non-negative integers"):
        QuantumJobResult(job=_job_ref(), status="completed", counts={"0": -1}, shots=1)


def test_job_result_rejects_count_sum_mismatch() -> None:
    """Counts must sum to the declared shot total."""
    with pytest.raises(ValueError, match="counts must sum to shots"):
        QuantumJobResult(job=_job_ref(), status="completed", counts={"00": 5}, shots=10)


def test_local_simulator_rejects_cloud_profile() -> None:
    """The local simulator refuses a cloud profile."""
    with pytest.raises(ValueError, match="requires a non-cloud profile"):
        LocalDeterministicSimulator(_cloud_profile())


def test_simulator_status_unknown_job() -> None:
    """Querying status for an unknown job id raises."""
    sim = LocalDeterministicSimulator(_local_profile())
    with pytest.raises(KeyError, match="unknown job_id"):
        sim.status(_job_ref())


def test_simulator_result_unknown_job() -> None:
    """Fetching a result for an unknown job id raises."""
    sim = LocalDeterministicSimulator(_local_profile())
    with pytest.raises(KeyError, match="unknown job_id"):
        sim.result(_job_ref())


def test_simulator_cancel_unknown_job() -> None:
    """Cancelling an unknown job id raises."""
    sim = LocalDeterministicSimulator(_local_profile())
    with pytest.raises(KeyError, match="unknown job_id"):
        sim.cancel(_job_ref())


def test_simulator_submit_then_cancel_round_trip() -> None:
    """A synchronous completed job retains its evidence after late cancellation."""
    sim = LocalDeterministicSimulator(_local_profile())
    job = sim.submit(_workload())
    assert sim.status(job) == job.status
    result = sim.result(job)
    assert sum(result.counts.values()) == result.shots
    cancelled = sim.cancel(job)
    assert cancelled is job
    assert sim.status(cancelled) == "completed"
    assert sim.result(cancelled) is result


def test_hal_rejects_duplicate_profile() -> None:
    """Two profiles sharing a backend id are rejected by the registry."""
    with pytest.raises(ValueError, match="duplicate backend profile"):
        HardwareAbstractionLayer((_local_profile(), _local_profile()))


def test_hal_profile_unknown_id() -> None:
    """Requesting an unknown profile id raises."""
    hal = HardwareAbstractionLayer((_local_profile(),))
    with pytest.raises(KeyError, match="unknown backend_id"):
        hal.profile("missing")


def test_hal_register_rejects_non_backend() -> None:
    """Registering an object that is not a backend adapter is rejected."""
    hal = HardwareAbstractionLayer((_local_profile(),))
    with pytest.raises(TypeError, match="backend must satisfy QuantumBackend"):
        hal.register_backend(object())  # type: ignore[arg-type]


def test_hal_register_rejects_unregistered_profile() -> None:
    """Registering an adapter without a matching profile is rejected."""
    hal = HardwareAbstractionLayer((_local_profile(),))
    foreign = LocalDeterministicSimulator(_local_profile(backend_id="other_local"))
    with pytest.raises(ValueError, match="backend profile is not registered"):
        hal.register_backend(foreign)


def test_hal_cancel_rejected_when_unsupported() -> None:
    """Cancellation is refused when the profile disallows it."""
    profile = _local_profile(supports_cancellation=False)
    hal = HardwareAbstractionLayer((profile,))
    hal.register_backend(LocalDeterministicSimulator(profile))
    job = QuantumJobRef(
        job_id="job1", backend_id="local_test", workload_id="w1", status="submitted"
    )
    with pytest.raises(ValueError, match="does not support cancellation"):
        hal.cancel(job)


def test_hal_delegation_requires_registered_backend() -> None:
    """Delegating to a known profile without a registered adapter is refused."""
    hal = HardwareAbstractionLayer((_local_profile(),))
    with pytest.raises(PermissionError, match="backend is not registered"):
        hal.status(_job_ref())


def test_hal_rejects_shots_without_shot_support() -> None:
    """A shot workload on a no-shot backend is rejected during validation."""
    profile = _local_profile(supports_shots=False)
    hal = HardwareAbstractionLayer((profile,))
    hal.register_backend(LocalDeterministicSimulator(profile))
    with pytest.raises(ValueError, match="does not support shot workloads"):
        hal.submit("local_test", _workload(shots=8))
