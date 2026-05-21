# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- IonQ direct HAL adapter tests
"""Tests for direct IonQ Cloud execution behind the provider-neutral HAL."""

from __future__ import annotations

from typing import Any

import pytest

from scpn_quantum_control.hardware import hal_ionq as ionq_mod
from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer
from scpn_quantum_control.hardware.hal_ionq import (
    IonQCloudHALAdapter,
    ionq_qis_workload,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeIonQClient:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self.puts: list[str] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> _FakeResponse:
        self.posts.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _FakeResponse({"id": "617a1f8b-59d4-435d-aa33-695433d7155e", "status": "submitted"})

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> _FakeResponse:
        del headers, timeout
        self.gets.append(url)
        if url.endswith("/results/probabilities"):
            return _FakeResponse({"0": 0.5, "3": 0.5})
        return _FakeResponse({"id": "617a1f8b-59d4-435d-aa33-695433d7155e", "status": "completed"})

    def put(self, url: str, *, headers: dict[str, str], timeout: float) -> _FakeResponse:
        del headers, timeout
        self.puts.append(url)
        return _FakeResponse({"id": "617a1f8b-59d4-435d-aa33-695433d7155e", "status": "canceled"})


class _FakeIonQShotMismatchClient(_FakeIonQClient):
    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> _FakeResponse:
        del headers, timeout
        self.gets.append(url)
        if url.endswith("/results/probabilities"):
            return _FakeResponse({"0": 0.75, "3": 0.75})
        return _FakeResponse({"id": "617a1f8b-59d4-435d-aa33-695433d7155e", "status": "completed"})


def test_ionq_direct_adapter_submits_status_results_and_cancel() -> None:
    """IonQ direct adapter should implement the v0.4 job lifecycle."""

    client = _FakeIonQClient()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        IonQCloudHALAdapter(
            hal.profile("ionq_cloud"),
            client=client,
            api_key="test-key",
            backend="simulator",
            settings={"error_mitigation": {"debiasing": False}},
            timeout_s=12.5,
        )
    )
    workload = ionq_qis_workload(
        [
            {"gate": "h", "target": 0},
            {"gate": "cnot", "control": 0, "target": 1},
        ],
        workload_id="ionq_bell",
        n_qubits=2,
        shots=10,
        metadata={"lane": "hal"},
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("ionq_cloud", workload)

    job = hal.submit("ionq_cloud", workload, approval_id="approved-ionq")
    status = hal.status(job)
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert job.status == "submitted"
    assert status == "completed"
    assert result.counts == {"00": 5, "11": 5}
    assert result.shots == 10
    assert result.metadata["execution_mode"] == "ionq_cloud_api_v0.4"
    assert cancelled.status == "cancelled"
    assert client.posts[0]["url"] == "https://api.ionq.co/v0.4/jobs"
    assert client.posts[0]["headers"]["Authorization"] == "apiKey test-key"
    assert client.posts[0]["json"] == {
        "type": "ionq.circuit.v1",
        "name": "ionq_bell",
        "metadata": {"approval_id": "approved-ionq", "lane": "hal", "workload_id": "ionq_bell"},
        "shots": 10,
        "backend": "simulator",
        "input": {
            "qubits": 2,
            "gateset": "qis",
            "circuit": [
                {"gate": "h", "target": 0},
                {"gate": "cnot", "control": 0, "target": 1},
            ],
        },
        "settings": {"error_mitigation": {"debiasing": False}},
    }
    assert client.gets == [
        "https://api.ionq.co/v0.4/jobs/617a1f8b-59d4-435d-aa33-695433d7155e",
        "https://api.ionq.co/v0.4/jobs/617a1f8b-59d4-435d-aa33-695433d7155e/results/probabilities",
    ]
    assert client.puts == [
        "https://api.ionq.co/v0.4/jobs/617a1f8b-59d4-435d-aa33-695433d7155e/status/cancel"
    ]


def test_ionq_direct_adapter_rejects_malformed_qis_payload() -> None:
    """IonQ helper should fail closed on malformed circuit instructions."""

    with pytest.raises(ValueError, match="target"):
        ionq_qis_workload(
            [{"gate": "h"}],
            workload_id="bad_ionq",
            n_qubits=1,
            shots=10,
        )


def test_ionq_direct_adapter_requires_route_credentials() -> None:
    """IonQ construction should fail closed without client credentials."""

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("ionq_cloud")

    with pytest.raises(ValueError, match="api_key"):
        IonQCloudHALAdapter(profile, client=_FakeIonQClient())


def test_ionq_direct_adapter_rejects_missing_n_qubits_metadata() -> None:
    """IonQ result decoding should fail closed when job metadata is corrupted."""

    client = _FakeIonQClient()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IonQCloudHALAdapter(
        hal.profile("ionq_cloud"),
        client=client,
        api_key="test-key",
    )
    hal.register_backend(adapter)
    workload = ionq_qis_workload(
        [{"gate": "h", "target": 0}],
        workload_id="ionq_missing_n_qubits",
        n_qubits=1,
        shots=10,
    )
    job = hal.submit("ionq_cloud", workload, approval_id="approved-ionq")
    adapter._jobs[job.job_id] = adapter._jobs[job.job_id].__class__(
        job_id=job.job_id,
        backend_id=job.backend_id,
        workload_id=job.workload_id,
        status=job.status,
        metadata={k: v for k, v in job.metadata.items() if k != "n_qubits"},
    )

    with pytest.raises(ValueError, match="n_qubits"):
        hal.result(job)


def test_ionq_direct_adapter_normalises_provider_status_tokens() -> None:
    """IonQ status fields should map to canonical HAL status values."""

    from scpn_quantum_control.hardware import hal_ionq as ionq_mod

    assert ionq_mod._normalise_status("CANCELED") == "cancelled"
    assert ionq_mod._normalise_status("COMPLETE") == "completed"
    assert ionq_mod._normalise_status("IN-PROGRESS") == "running"
    assert ionq_mod._normalise_status("INPROGRESS") == "running"
    assert ionq_mod._normalise_status("INITIALIZING") == "submitted"
    assert ionq_mod._normalise_status("STARTING") == "submitted"


def test_ionq_direct_adapter_rejects_control_characters_in_provider_job_id() -> None:
    """IonQ job creation response ids must pass strict provider-id validation."""

    with pytest.raises(ValueError, match="provider job id"):
        ionq_mod._provider_job_id_from_response({"id": "job-\n42"})


def test_ionq_provider_job_id_rejects_control_characters() -> None:
    """IonQ provider identifiers must reject control-character payloads."""

    with pytest.raises(ValueError, match="provider job id"):
        ionq_mod._provider_job_id_from_response({"id": "ionq-provider-\n2"})


def test_ionq_direct_adapter_trims_provider_job_id_padding() -> None:
    """IonQ provider ids should be trimmed to canonical form."""

    assert ionq_mod._provider_job_id_from_response({"id": "  ionq-job-42  "}) == "ionq-job-42"


def test_ionq_direct_adapter_rejects_non_positive_expected_shots() -> None:
    """IonQ result decoding must fail closed without a positive expected shot count."""

    client = _FakeIonQClient()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IonQCloudHALAdapter(
        hal.profile("ionq_cloud"),
        client=client,
        api_key="test-key",
    )
    hal.register_backend(adapter)
    workload = ionq_qis_workload(
        [{"gate": "h", "target": 0}], workload_id="ionq_bad_shots", n_qubits=1, shots=10
    )
    job = hal.submit("ionq_cloud", workload, approval_id="approved-ionq")
    adapter._jobs[job.job_id] = adapter._jobs[job.job_id].__class__(
        job_id=job.job_id,
        backend_id=job.backend_id,
        workload_id=job.workload_id,
        status=job.status,
        metadata={**job.metadata, "shots": 0},
    )

    with pytest.raises(ValueError, match="positive shot count"):
        hal.result(job)


def test_ionq_direct_adapter_rejects_shot_mismatch() -> None:
    """IonQ result decoding must fail closed when counts overshoot expected shots."""

    client = _FakeIonQShotMismatchClient()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        IonQCloudHALAdapter(
            hal.profile("ionq_cloud"),
            client=client,
            api_key="test-key",
        )
    )
    workload = ionq_qis_workload(
        [{"gate": "h", "target": 0}],
        workload_id="ionq_shot_mismatch",
        n_qubits=2,
        shots=10,
    )
    job = hal.submit("ionq_cloud", workload, approval_id="approved-ionq")

    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.result(job)


def test_ionq_backend_rejects_control_characters() -> None:
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("ionq_cloud")
    with pytest.raises(ValueError, match="IonQ backend"):
        IonQCloudHALAdapter(profile, api_key="test-key", backend="qpu.\naria-1")


def test_ionq_backend_trims_padding() -> None:
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("ionq_cloud")
    adapter = IonQCloudHALAdapter(profile, api_key="test-key", backend="  simulator  ")
    assert adapter._backend == "simulator"
