# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- provider-neutral hardware abstraction layer
"""Provider-neutral hardware abstraction layer.

This module separates SCPN workload routing from provider SDKs. Discovery is
metadata-only: constructing profiles does not import Qiskit, Braket, Azure,
IonQ, Rigetti, QuEra, IQM, Pasqal, OQC, D-Wave, or simulator packages. Live
execution is available only through an injected backend adapter that satisfies
``QuantumBackend`` and, for cloud profiles, carries an explicit approval token.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/-]+$")


def _validate_token(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or not _TOKEN_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty identifier token")


def _normalise_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    for value in result:
        _validate_token(value, field_name)
    return result


def _freeze_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    frozen: dict[str, object] = {}
    for key, value in metadata.items():
        _validate_token(key, "metadata key")
        if value is not None and not isinstance(value, str | int | float | bool):
            raise ValueError("metadata values must be JSON-scalar compatible")
        frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class BackendCapabilities:
    """Provider route capabilities used for fail-fast workload validation."""

    supports_shots: bool
    supports_counts: bool
    supports_statevector: bool
    supports_mid_circuit_measurement: bool
    supports_analog: bool
    supports_pulse: bool
    max_qubits: int | None = None
    supports_cancellation: bool = True
    supports_cost_estimate: bool = False

    def __post_init__(self) -> None:
        if self.max_qubits is not None and self.max_qubits <= 0:
            raise ValueError("max_qubits must be positive when provided")


@dataclass(frozen=True)
class BackendProfile:
    """Static profile for a concrete provider, broker, or simulator route."""

    backend_id: str
    provider: str
    broker: str
    modality: str
    sdk_package: str
    ir_formats: Sequence[str]
    capabilities: BackendCapabilities
    is_cloud: bool = False
    submit_requires_approval: bool = False
    region: str | None = None
    target_family: str | None = None
    notes: Sequence[str] = ()

    def __post_init__(self) -> None:
        _validate_token(self.backend_id, "backend_id")
        _validate_token(self.provider, "provider")
        _validate_token(self.broker, "broker")
        _validate_token(self.modality, "modality")
        _validate_token(self.sdk_package, "sdk_package")
        object.__setattr__(self, "ir_formats", _normalise_tuple(self.ir_formats, "ir_formats"))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        if self.region is not None:
            _validate_token(self.region, "region")
        if self.target_family is not None:
            _validate_token(self.target_family, "target_family")
        if self.is_cloud and not self.submit_requires_approval:
            raise ValueError("cloud profiles must require explicit submission approval")


@dataclass(frozen=True)
class QuantumWorkload:
    """Provider-neutral workload handed to an injected backend adapter."""

    workload_id: str
    ir_format: str
    program: str
    n_qubits: int
    shots: int = 1024
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_token(self.workload_id, "workload_id")
        _validate_token(self.ir_format, "ir_format")
        if not isinstance(self.program, str) or not self.program.strip():
            raise ValueError("program must be non-empty")
        if not isinstance(self.n_qubits, int) or self.n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")
        if not isinstance(self.shots, int) or self.shots <= 0:
            raise ValueError("shots must be a positive integer")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True)
class QuantumJobRef:
    """Stable handle returned by a backend adapter after submission."""

    job_id: str
    backend_id: str
    workload_id: str
    status: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_token(self.job_id, "job_id")
        _validate_token(self.backend_id, "backend_id")
        _validate_token(self.workload_id, "workload_id")
        _validate_token(self.status, "status")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True)
class QuantumJobResult:
    """Provider-neutral result payload for shot-count workloads."""

    job: QuantumJobRef
    status: str
    counts: Mapping[str, int] = field(default_factory=dict)
    shots: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_token(self.status, "status")
        if self.shots < 0:
            raise ValueError("shots must be non-negative")
        frozen_counts: dict[str, int] = {}
        for bitstring, count in self.counts.items():
            if not isinstance(bitstring, str) or not bitstring:
                raise ValueError("counts keys must be non-empty bitstrings")
            if not isinstance(count, int) or count < 0:
                raise ValueError("counts values must be non-negative integers")
            frozen_counts[bitstring] = count
        if frozen_counts and self.shots and sum(frozen_counts.values()) != self.shots:
            raise ValueError("counts must sum to shots")
        object.__setattr__(self, "counts", MappingProxyType(frozen_counts))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@runtime_checkable
class QuantumBackend(Protocol):
    """Runtime protocol for injected provider adapters."""

    backend_id: str

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a validated workload and return a job handle."""

    def status(self, job: QuantumJobRef) -> str:
        """Return the provider status for a job handle."""

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return a completed result payload."""

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Cancel a job when the provider supports cancellation."""


class LocalDeterministicSimulator:
    """Offline simulator adapter used to verify the HAL execution contract."""

    def __init__(self, profile: BackendProfile) -> None:
        if profile.is_cloud:
            raise ValueError("LocalDeterministicSimulator requires a non-cloud profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        del approval_id
        _validate_workload_for_profile(self.profile, workload)
        digest = hashlib.sha256(
            f"{self.backend_id}|{workload.workload_id}|{workload.ir_format}|"
            f"{workload.n_qubits}|{workload.shots}|{workload.program}".encode()
        ).hexdigest()[:12]
        job = QuantumJobRef(
            job_id=f"{self.backend_id}:{workload.workload_id}:{digest}",
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            metadata={"execution_mode": "local_deterministic_simulator"},
        )
        counts = self._deterministic_counts(workload)
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=workload.shots,
            metadata={
                "execution_mode": "local_deterministic_simulator",
                "profile_modality": self.profile.modality,
            },
        )
        self._jobs[job.job_id] = job
        self._results[job.job_id] = result
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored.status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        result = self._results.get(job.job_id)
        if result is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        if job.job_id not in self._jobs:
            raise KeyError(f"unknown job_id: {job.job_id}")
        cancelled = QuantumJobRef(
            job_id=job.job_id,
            backend_id=job.backend_id,
            workload_id=job.workload_id,
            status="cancelled",
            metadata=job.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    @staticmethod
    def _deterministic_counts(workload: QuantumWorkload) -> dict[str, int]:
        seed_value = workload.metadata.get("seed")
        if seed_value is None:
            seed_material = f"{workload.workload_id}|{workload.program}"
            seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
        else:
            seed = int(str(seed_value))
        zero_state = "0" * workload.n_qubits
        one_state = "1" * workload.n_qubits
        zero_count = workload.shots // 2 + (seed % 2)
        zero_count = min(max(zero_count, 0), workload.shots)
        one_count = workload.shots - zero_count
        return {zero_state: zero_count, one_state: one_count}


class HardwareAbstractionLayer:
    """Profile registry plus approval-gated execution router."""

    def __init__(self, profiles: Sequence[BackendProfile]) -> None:
        by_id: dict[str, BackendProfile] = {}
        for profile in profiles:
            if profile.backend_id in by_id:
                raise ValueError(f"duplicate backend profile: {profile.backend_id}")
            by_id[profile.backend_id] = profile
        self._profiles = by_id
        self._backends: dict[str, QuantumBackend] = {}

    @classmethod
    def with_builtin_profiles(cls) -> HardwareAbstractionLayer:
        """Construct a HAL with all built-in provider route profiles."""

        return cls(built_in_backend_profiles())

    def list_profiles(self) -> tuple[BackendProfile, ...]:
        """Return profiles in deterministic backend-id order."""

        return tuple(self._profiles[key] for key in sorted(self._profiles))

    def profile(self, backend_id: str) -> BackendProfile:
        """Return one backend profile by id."""

        try:
            return self._profiles[backend_id]
        except KeyError as exc:
            raise KeyError(f"unknown backend_id: {backend_id}") from exc

    def register_backend(self, backend: QuantumBackend) -> None:
        """Inject an executable adapter for one known profile."""

        if not isinstance(backend, QuantumBackend):
            raise TypeError("backend must satisfy QuantumBackend")
        if backend.backend_id not in self._profiles:
            raise ValueError(f"backend profile is not registered: {backend.backend_id}")
        if backend.backend_id in self._backends:
            raise ValueError(f"backend already registered: {backend.backend_id}")
        self._backends[backend.backend_id] = backend

    def submit(
        self,
        backend_id: str,
        workload: QuantumWorkload,
        *,
        approval_id: str | None = None,
    ) -> QuantumJobRef:
        """Validate and submit a workload through an injected adapter."""

        profile = self.profile(backend_id)
        _validate_workload_for_profile(profile, workload)
        backend = self._backends.get(backend_id)
        if backend is None:
            raise PermissionError(f"backend is not registered: {backend_id}")
        if profile.submit_requires_approval and not approval_id:
            raise PermissionError(f"submission approval required for backend: {backend_id}")
        return backend.submit(workload, approval_id=approval_id)

    def status(self, job: QuantumJobRef) -> str:
        """Return current status by delegating to the owning adapter."""

        return self._backend_for_job(job).status(job)

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return a result by delegating to the owning adapter."""

        return self._backend_for_job(job).result(job)

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Cancel a job by delegating to the owning adapter."""

        profile = self.profile(job.backend_id)
        if not profile.capabilities.supports_cancellation:
            raise ValueError(f"backend does not support cancellation: {job.backend_id}")
        return self._backend_for_job(job).cancel(job)

    def _backend_for_job(self, job: QuantumJobRef) -> QuantumBackend:
        self.profile(job.backend_id)
        backend = self._backends.get(job.backend_id)
        if backend is None:
            raise PermissionError(f"backend is not registered: {job.backend_id}")
        return backend


def _validate_workload_for_profile(profile: BackendProfile, workload: QuantumWorkload) -> None:
    if workload.ir_format not in profile.ir_formats:
        accepted = ", ".join(profile.ir_formats)
        raise ValueError(
            f"IR format {workload.ir_format!r} is not supported by {profile.backend_id}; expected {accepted}"
        )
    max_qubits = profile.capabilities.max_qubits
    if max_qubits is not None and workload.n_qubits > max_qubits:
        raise ValueError(
            f"workload qubits exceed {profile.backend_id} limit: {workload.n_qubits} > {max_qubits}"
        )
    if workload.shots and not profile.capabilities.supports_shots:
        raise ValueError(f"backend does not support shot workloads: {profile.backend_id}")


def _profile(
    backend_id: str,
    *,
    provider: str,
    broker: str,
    modality: str,
    sdk_package: str,
    ir_formats: Sequence[str],
    max_qubits: int | None,
    is_cloud: bool,
    supports_statevector: bool = False,
    supports_mid_circuit_measurement: bool = False,
    supports_analog: bool = False,
    supports_pulse: bool = False,
    target_family: str | None = None,
    notes: Sequence[str] = (),
) -> BackendProfile:
    return BackendProfile(
        backend_id=backend_id,
        provider=provider,
        broker=broker,
        modality=modality,
        sdk_package=sdk_package,
        ir_formats=ir_formats,
        capabilities=BackendCapabilities(
            supports_shots=True,
            supports_counts=True,
            supports_statevector=supports_statevector,
            supports_mid_circuit_measurement=supports_mid_circuit_measurement,
            supports_analog=supports_analog,
            supports_pulse=supports_pulse,
            max_qubits=max_qubits,
        ),
        is_cloud=is_cloud,
        submit_requires_approval=is_cloud,
        target_family=target_family,
        notes=notes,
    )


def built_in_backend_profiles() -> tuple[BackendProfile, ...]:
    """Return built-in provider and simulator route profiles.

    Profiles intentionally describe routes rather than perform SDK discovery.
    Provider-specific credentials, queues, regions, and pricing are left to
    injected adapters so offline tooling remains deterministic and auditable.
    """

    profiles = (
        _profile(
            "aws_braket_aqt",
            provider="aqt",
            broker="aws_braket",
            modality="trapped_ion_gate_model",
            sdk_package="amazon-braket-sdk",
            ir_formats=("openqasm3", "braket_ir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="aqt_ion_trap",
        ),
        _profile(
            "aws_braket_dm1",
            provider="aws",
            broker="aws_braket",
            modality="managed_density_matrix_simulator",
            sdk_package="amazon-braket-sdk",
            ir_formats=("openqasm3", "braket_ir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="braket_dm1",
        ),
        _profile(
            "aws_braket_ionq",
            provider="ionq",
            broker="aws_braket",
            modality="trapped_ion_gate_model",
            sdk_package="amazon-braket-sdk",
            ir_formats=("openqasm3", "braket_ir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="ionq_ion_trap",
        ),
        _profile(
            "aws_braket_iqm",
            provider="iqm",
            broker="aws_braket",
            modality="superconducting_gate_model",
            sdk_package="amazon-braket-sdk",
            ir_formats=("openqasm3", "braket_ir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="iqm_superconducting",
        ),
        _profile(
            "aws_braket_quera",
            provider="quera",
            broker="aws_braket",
            modality="neutral_atom_analog",
            sdk_package="amazon-braket-sdk",
            ir_formats=("braket_ahs", "bloqade", "mlir"),
            max_qubits=None,
            is_cloud=True,
            supports_analog=True,
            target_family="quera_neutral_atom",
        ),
        _profile(
            "aws_braket_rigetti",
            provider="rigetti",
            broker="aws_braket",
            modality="superconducting_gate_model",
            sdk_package="amazon-braket-sdk",
            ir_formats=("openqasm3", "quil", "braket_ir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="rigetti_superconducting",
        ),
        _profile(
            "aws_braket_sv1",
            provider="aws",
            broker="aws_braket",
            modality="managed_statevector_simulator",
            sdk_package="amazon-braket-sdk",
            ir_formats=("openqasm3", "braket_ir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            supports_statevector=True,
            target_family="braket_sv1",
        ),
        _profile(
            "aws_braket_tn1",
            provider="aws",
            broker="aws_braket",
            modality="managed_tensor_network_simulator",
            sdk_package="amazon-braket-sdk",
            ir_formats=("openqasm3", "braket_ir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="braket_tn1",
        ),
        _profile(
            "azure_quantum_ionq",
            provider="ionq",
            broker="azure_quantum",
            modality="trapped_ion_gate_model",
            sdk_package="azure-quantum",
            ir_formats=("openqasm3", "qir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="ionq_ion_trap",
        ),
        _profile(
            "azure_quantum_ionq_simulator",
            provider="ionq",
            broker="azure_quantum",
            modality="managed_gate_model_simulator",
            sdk_package="azure-quantum",
            ir_formats=("openqasm3", "qir", "mlir"),
            max_qubits=36,
            is_cloud=True,
            supports_statevector=True,
            target_family="ionq_simulator",
        ),
        _profile(
            "azure_quantum_pasqal",
            provider="pasqal",
            broker="azure_quantum",
            modality="neutral_atom_analog",
            sdk_package="azure-quantum",
            ir_formats=("pasqal_ir", "openqasm3", "qir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            supports_analog=True,
            target_family="pasqal_neutral_atom",
        ),
        _profile(
            "azure_quantum_pasqal_emulator",
            provider="pasqal",
            broker="azure_quantum",
            modality="managed_neutral_atom_emulator",
            sdk_package="azure-quantum",
            ir_formats=("pasqal_ir", "openqasm3", "qir", "mlir"),
            max_qubits=80,
            is_cloud=True,
            supports_statevector=True,
            supports_analog=True,
            target_family="pasqal_emulator",
        ),
        _profile(
            "azure_quantum_qci_preview",
            provider="quantum_circuits",
            broker="azure_quantum",
            modality="superconducting_feedback_gate_model",
            sdk_package="azure-quantum",
            ir_formats=("openqasm3", "qir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            supports_mid_circuit_measurement=True,
            target_family="qci_private_preview",
            notes=("private_preview",),
        ),
        _profile(
            "azure_quantum_quantinuum",
            provider="quantinuum",
            broker="azure_quantum",
            modality="trapped_ion_gate_model",
            sdk_package="azure-quantum",
            ir_formats=("openqasm3", "qir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            supports_mid_circuit_measurement=True,
            target_family="quantinuum_ion_trap",
        ),
        _profile(
            "azure_quantum_quantinuum_emulator",
            provider="quantinuum",
            broker="azure_quantum",
            modality="managed_trapped_ion_emulator",
            sdk_package="azure-quantum",
            ir_formats=("openqasm3", "qir", "mlir"),
            max_qubits=32,
            is_cloud=True,
            supports_mid_circuit_measurement=True,
            target_family="quantinuum_emulator",
        ),
        _profile(
            "azure_quantum_rigetti",
            provider="rigetti",
            broker="azure_quantum",
            modality="superconducting_gate_model",
            sdk_package="azure-quantum",
            ir_formats=("quil", "openqasm3", "qir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="rigetti_superconducting",
        ),
        _profile(
            "azure_quantum_rigetti_qvm",
            provider="rigetti",
            broker="azure_quantum",
            modality="managed_gate_model_simulator",
            sdk_package="azure-quantum",
            ir_formats=("quil", "openqasm3", "qir", "mlir"),
            max_qubits=30,
            is_cloud=True,
            supports_statevector=True,
            target_family="rigetti_qvm",
        ),
        _profile(
            "dwave_leap",
            provider="dwave",
            broker="direct",
            modality="quantum_annealing",
            sdk_package="dwave-cloud-client",
            ir_formats=("bqm", "ising", "qubo", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="annealing",
        ),
        _profile(
            "ibm_quantum",
            provider="ibm",
            broker="direct",
            modality="superconducting_gate_model",
            sdk_package="qiskit-ibm-runtime",
            ir_formats=("qiskit_qpy", "openqasm3", "qiskit", "qir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            supports_mid_circuit_measurement=True,
            supports_pulse=True,
            target_family="ibm_quantum",
        ),
        _profile(
            "ionq_cloud",
            provider="ionq",
            broker="direct",
            modality="trapped_ion_gate_model",
            sdk_package="requests",
            ir_formats=("openqasm3", "ionq_json", "qir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="ionq_ion_trap",
        ),
        _profile(
            "iqm_cloud",
            provider="iqm",
            broker="direct",
            modality="superconducting_gate_model",
            sdk_package="iqm-client",
            ir_formats=("qiskit_qpy", "openqasm3", "qiskit", "circuit", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="iqm_superconducting",
        ),
        _profile(
            "local_braket_ahs",
            provider="aws",
            broker="local",
            modality="analog_hamiltonian_simulator",
            sdk_package="amazon-braket-sdk",
            ir_formats=("braket_ahs", "bloqade", "mlir"),
            max_qubits=None,
            is_cloud=False,
            supports_statevector=True,
            supports_analog=True,
            target_family="braket_local_ahs",
        ),
        _profile(
            "local_braket_dm",
            provider="aws",
            broker="local",
            modality="density_matrix_simulator",
            sdk_package="amazon-braket-sdk",
            ir_formats=("openqasm3", "braket_ir", "mlir"),
            max_qubits=None,
            is_cloud=False,
            target_family="braket_local_dm",
        ),
        _profile(
            "local_braket_sv",
            provider="aws",
            broker="local",
            modality="statevector_simulator",
            sdk_package="amazon-braket-sdk",
            ir_formats=("openqasm3", "braket_ir", "mlir"),
            max_qubits=None,
            is_cloud=False,
            supports_statevector=True,
            target_family="braket_local_sv",
        ),
        _profile(
            "local_cirq",
            provider="cirq",
            broker="local",
            modality="simulator",
            sdk_package="cirq-core",
            ir_formats=("cirq", "openqasm3", "mlir"),
            max_qubits=None,
            is_cloud=False,
            supports_statevector=True,
        ),
        _profile(
            "local_pennylane",
            provider="pennylane",
            broker="local",
            modality="simulator",
            sdk_package="pennylane",
            ir_formats=("pennylane", "openqasm3", "mlir"),
            max_qubits=None,
            is_cloud=False,
            supports_statevector=True,
        ),
        _profile(
            "local_qiskit_aer",
            provider="qiskit_aer",
            broker="local",
            modality="simulator",
            sdk_package="qiskit-aer",
            ir_formats=("qiskit_qpy", "qiskit", "openqasm3", "mlir"),
            max_qubits=None,
            is_cloud=False,
            supports_statevector=True,
        ),
        _profile(
            "local_statevector",
            provider="scpn",
            broker="local",
            modality="simulator",
            sdk_package="python",
            ir_formats=("mlir", "openqasm3"),
            max_qubits=None,
            is_cloud=False,
            supports_statevector=True,
        ),
        _profile(
            "oqc_cloud",
            provider="oqc",
            broker="direct",
            modality="superconducting_gate_model",
            sdk_package="oqc-qcaas-client",
            ir_formats=("openqasm3", "qir", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="oqc_superconducting",
        ),
        _profile(
            "pasqal_cloud",
            provider="pasqal",
            broker="direct",
            modality="neutral_atom_analog",
            sdk_package="pulser-core",
            ir_formats=("pulser", "pasqal_ir", "openqasm3", "mlir"),
            max_qubits=None,
            is_cloud=True,
            supports_analog=True,
            target_family="pasqal_neutral_atom",
        ),
        _profile(
            "qbraid_ionq",
            provider="ionq",
            broker="qbraid",
            modality="trapped_ion_gate_model",
            sdk_package="qbraid",
            ir_formats=(
                "openqasm3",
                "qiskit",
                "cirq",
                "quil",
                "braket_ir",
                "pennylane",
                "pyqubo",
                "tket",
                "mlir",
            ),
            max_qubits=None,
            is_cloud=True,
            target_family="ionq_ion_trap",
            notes=("broker_runtime",),
        ),
        _profile(
            "qbraid_runtime",
            provider="dynamic",
            broker="qbraid",
            modality="provider_agnostic_runtime",
            sdk_package="qbraid",
            ir_formats=(
                "openqasm3",
                "qiskit",
                "cirq",
                "quil",
                "braket_ir",
                "pennylane",
                "pyqubo",
                "tket",
                "mlir",
            ),
            max_qubits=None,
            is_cloud=True,
            target_family="dynamic_catalog",
            notes=("broker_runtime", "dynamic_catalog"),
        ),
        _profile(
            "quandela_cloud",
            provider="quandela",
            broker="direct",
            modality="photonic_gate_model",
            sdk_package="perceval-quandela",
            ir_formats=("perceval", "openqasm3", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="quandela_photonic",
        ),
        _profile(
            "quantinuum_cloud",
            provider="quantinuum",
            broker="direct",
            modality="trapped_ion_gate_model",
            sdk_package="pytket-quantinuum",
            ir_formats=("openqasm3", "qir", "tket", "mlir"),
            max_qubits=None,
            is_cloud=True,
            supports_mid_circuit_measurement=True,
            target_family="quantinuum_ion_trap",
        ),
        _profile(
            "quera_bloqade",
            provider="quera",
            broker="direct",
            modality="neutral_atom_analog",
            sdk_package="bloqade",
            ir_formats=("bloqade", "braket_ahs", "mlir"),
            max_qubits=None,
            is_cloud=True,
            supports_analog=True,
            target_family="quera_neutral_atom",
        ),
        _profile(
            "rigetti_qcs",
            provider="rigetti",
            broker="direct",
            modality="superconducting_gate_model",
            sdk_package="pyquil",
            ir_formats=("quil", "openqasm3", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="rigetti_superconducting",
        ),
        _profile(
            "strangeworks_compute",
            provider="dynamic",
            broker="strangeworks",
            modality="provider_agnostic_compute",
            sdk_package="strangeworks",
            ir_formats=("openqasm3", "qiskit", "quil", "braket_ir", "cirq", "mlir"),
            max_qubits=None,
            is_cloud=True,
            target_family="dynamic_catalog",
            notes=("broker_runtime", "dynamic_catalog"),
        ),
    )
    return tuple(sorted(profiles, key=lambda profile: profile.backend_id))


__all__ = [
    "BackendCapabilities",
    "BackendProfile",
    "HardwareAbstractionLayer",
    "LocalDeterministicSimulator",
    "QuantumBackend",
    "QuantumJobRef",
    "QuantumJobResult",
    "QuantumWorkload",
    "built_in_backend_profiles",
]
