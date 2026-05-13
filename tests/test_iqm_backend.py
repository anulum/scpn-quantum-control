# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM backend adapter tests
"""Tests for the IQM Qiskit backend adapter."""

from __future__ import annotations

import types
from typing import Any

import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.hardware import backends as be
from scpn_quantum_control.hardware.iqm_backend import (
    IQMBackendConfig,
    IQMQuantumBackend,
    IQMRunResult,
    is_iqm_available,
)


class _FakeResult:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _FakeJob:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def job_id(self) -> str:
        return "fake-iqm-job-1"

    def result(self, timeout: float | None = None) -> _FakeResult:
        assert timeout == 12.5
        return _FakeResult(self._counts)


class _FakeIQMBackend:
    name = "fake_garnet"
    num_qubits = 20

    def __init__(self) -> None:
        self.received_shots: int | None = None

    def run(self, circuits: list[QuantumCircuit], *, shots: int) -> _FakeJob:
        assert len(circuits) == 1
        assert circuits[0].num_qubits == 2
        self.received_shots = shots
        return _FakeJob({"00": 31, "11": 33})


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


def test_is_iqm_available_returns_bool() -> None:
    assert isinstance(is_iqm_available(), bool)


def test_iqm_config_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError, match="shots must be positive"):
        IQMBackendConfig(shots=0)
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        IQMBackendConfig(timeout_s=0.0)
    with pytest.raises(ValueError, match="server_url is required"):
        IQMBackendConfig(mode="remote")


def test_iqm_descriptor_is_registered_and_approval_gated() -> None:
    descriptor = be.describe_backend("iqm")
    assert descriptor.name == "iqm"
    assert descriptor.provider == "iqm"
    assert descriptor.can_submit is True
    assert descriptor.submit_requires_approval is True
    assert descriptor.supports_shots is True
    assert "kuramoto_xy" in descriptor.workloads
    assert "superconducting_qpu" in descriptor.capabilities


def test_iqm_backend_requires_dependency_for_backend_resolution(monkeypatch) -> None:
    def missing_module(name: str) -> Any:
        raise ModuleNotFoundError(name)

    backend = IQMQuantumBackend(import_module=missing_module)
    assert backend.is_available() is False
    with pytest.raises(ImportError, match=r"iqm-client\[qiskit\]"):
        backend.resolve_backend(IQMBackendConfig(mode="fake", fake_backend="garnet"))


def test_iqm_fake_backend_resolution_and_count_run(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(IQMFakeGarnet=_FakeIQMBackend)

    def import_module(name: str) -> Any:
        if name == "iqm.qiskit_iqm.fake_backends.fake_garnet":
            return fake_module
        if name == "iqm.qiskit_iqm.iqm_provider":
            return types.SimpleNamespace(IQMProvider=_FakeIQMProvider)
        raise ModuleNotFoundError(name)

    adapter = IQMQuantumBackend(import_module=import_module)
    result = adapter.run_counts(
        _bell_circuit(),
        IQMBackendConfig(mode="fake", fake_backend="garnet", shots=64, timeout_s=12.5),
    )

    assert isinstance(result, IQMRunResult)
    assert result.job_id == "fake-iqm-job-1"
    assert result.backend_name == "fake_garnet"
    assert result.counts == {"00": 31, "11": 33}
    assert result.metadata["mode"] == "fake"
    assert result.metadata["shots"] == 64


def test_iqm_remote_backend_uses_provider_url_and_quantum_computer() -> None:
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

    adapter = IQMQuantumBackend(import_module=import_module)
    backend = adapter.resolve_backend(
        IQMBackendConfig(
            mode="remote",
            server_url="https://example.iqm.invalid",
            quantum_computer="garnet",
        )
    )

    assert backend.name == "fake_garnet"
    assert captured == {
        "url": "https://example.iqm.invalid",
        "quantum_computer": "garnet",
    }
