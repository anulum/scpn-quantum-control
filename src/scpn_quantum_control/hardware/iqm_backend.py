# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM quantum backend adapter
"""IQM backend adapter for Qiskit-compatible circuit execution.

The adapter follows IQM's documented Qiskit integration:
``IQMProvider(url, quantum_computer=...)`` creates a remote backend, fake
backend classes provide local architecture checks, and submitted circuits use
the normal Qiskit ``backend.run(..., shots=...)`` result path. This module never
discovers credentials, contacts IQM, or submits hardware jobs during registry
lookup.
"""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Literal

from qiskit import QuantumCircuit, transpile

from .backends import QuantumBackendDescriptor

IQMMode = Literal["fake", "remote"]

_FAKE_BACKENDS: Mapping[str, tuple[str, str, int]] = {
    "adonis": ("iqm.qiskit_iqm.fake_backends.fake_adonis", "IQMFakeAdonis", 5),
    "deneb": ("iqm.qiskit_iqm.fake_backends.fake_deneb", "IQMFakeDeneb", 6),
    "apollo": ("iqm.qiskit_iqm.fake_backends.fake_apollo", "IQMFakeApollo", 20),
    "garnet": ("iqm.qiskit_iqm.fake_backends.fake_garnet", "IQMFakeGarnet", 20),
    "aphrodite": ("iqm.qiskit_iqm.fake_backends.fake_aphrodite", "IQMFakeAphrodite", 54),
}


@dataclass(frozen=True)
class IQMBackendConfig:
    """Configuration for one IQM circuit execution path.

    ``mode="fake"`` uses local IQM fake backends and never contacts a remote
    service. ``mode="remote"`` requires an explicit ``server_url`` so accidental
    cloud submission cannot happen through defaults or hidden environment state.
    Shots and optimisation levels require integers, excluding booleans.
    Timeout is a finite positive real number in seconds, excluding booleans.
    Malformed values raise ValueError before any backend resolution.
    """

    mode: IQMMode = "fake"
    fake_backend: str = "garnet"
    server_url: str | None = None
    quantum_computer: str | None = None
    shots: int = 1024
    timeout_s: float = 600.0
    optimisation_level: int = 1

    def __post_init__(self) -> None:
        """Validate execution mode, budgets, and explicit remote routing."""
        if self.mode not in {"fake", "remote"}:
            raise ValueError("mode must be 'fake' or 'remote'")
        if (
            isinstance(self.shots, bool)
            or not isinstance(self.shots, (int, Integral))
            or self.shots <= 0
        ):
            raise ValueError("shots must be positive integers, excluding booleans")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float, Real))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0.0
        ):
            raise ValueError("timeout_s must be positive and finite, excluding booleans")
        if (
            isinstance(self.optimisation_level, bool)
            or not isinstance(self.optimisation_level, (int, Integral))
            or self.optimisation_level not in {0, 1, 2, 3}
        ):
            raise ValueError("optimisation_level must be 0, 1, 2, or 3")
        if self.mode == "remote" and not self.server_url:
            raise ValueError("server_url is required for remote IQM execution")
        if self.mode == "fake" and self.fake_backend.lower() not in _FAKE_BACKENDS:
            known = ", ".join(sorted(_FAKE_BACKENDS))
            raise ValueError(
                f"unknown IQM fake backend {self.fake_backend!r}; expected one of {known}"
            )


@dataclass(frozen=True)
class IQMRunResult:
    """Counts result from one IQM-backed circuit execution."""

    job_id: str
    backend_name: str
    counts: dict[str, int]
    wall_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


def is_iqm_available() -> bool:
    """Return whether the Qiskit-on-IQM package is importable."""
    try:
        importlib.import_module("iqm.qiskit_iqm.iqm_provider")
    except Exception:
        return False
    return True


class IQMTargetCompilationError(RuntimeError):
    """A circuit could not be compiled for the resolved IQM backend.

    Raised instead of falling back to a target-free transpilation. A circuit
    compiled without a backend has no coupling map, basis gates or qubit
    layout, so submitting it would run something other than what the caller
    asked for on a device it was never mapped to.
    """


class IQMQuantumBackend:
    """IQM provider adapter for SCPN circuit-replication workloads.

    The class satisfies the repository's ``BackendProtocol`` and exposes a
    richer execution method for approved callers. It is deliberately small:
    readiness routing, budget approval, and artefact ledgers live in the common
    hardware scheduler layers.
    """

    name = "iqm"

    def __init__(
        self,
        *,
        import_module: Callable[[str], Any] = importlib.import_module,
    ) -> None:
        self._import_module = import_module

    def is_available(self) -> bool:
        """Return whether Qiskit-on-IQM is importable in this environment.

        Returns
        -------
        bool
            ``True`` when the Qiskit-on-IQM provider module imports
            successfully.

        """
        try:
            self._import_module("iqm.qiskit_iqm.iqm_provider")
        except Exception:
            return False
        return True

    def descriptor(self) -> QuantumBackendDescriptor:
        """Return the non-submitting IQM capability descriptor."""
        return QuantumBackendDescriptor(
            name=self.name,
            provider="iqm",
            execution_mode="cloud_qpu_or_local_fake",
            sdk_package="iqm-client[qiskit]",
            adapter_module="scpn_quantum_control.hardware.iqm_backend",
            available=self.is_available(),
            can_simulate=True,
            can_submit=True,
            submit_requires_approval=True,
            supports_shots=True,
            supports_statevector=False,
            supports_mid_circuit_measurement=False,
            supports_pulse=False,
            max_qubits=None,
            capabilities=(
                "qiskit_iqm_provider",
                "iqm_fake_backend",
                "superconducting_qpu",
                "hardware_counts",
            ),
            workloads=(
                "kuramoto_xy",
                "dla_parity",
                "fim_feedback",
                "cross_platform_hardware",
            ),
            notes=(
                "Registry descriptor does not create an IQMProvider or contact IQM services.",
                "Remote execution requires IQMBackendConfig.server_url and external approval.",
            ),
        )

    def resolve_backend(self, config: IQMBackendConfig | None = None) -> Any:
        """Resolve an IQM backend without submitting a circuit."""
        cfg = config or IQMBackendConfig()
        if cfg.mode == "fake":
            return self._resolve_fake_backend(cfg.fake_backend)
        return self._resolve_remote_backend(cfg)

    def transpile_circuit(
        self,
        circuit: QuantumCircuit,
        config: IQMBackendConfig | None = None,
        *,
        backend: Any | None = None,
    ) -> QuantumCircuit:
        """Transpile ``circuit`` for one resolved IQM backend.

        The compilation target is always a concrete backend. A circuit that
        cannot be compiled for that device is rejected here rather than being
        recompiled without a target, because a target-free circuit carries no
        coupling map, basis gates or qubit layout and must never reach a
        submission.

        Parameters
        ----------
        circuit
            Logical circuit to compile.
        config
            Execution configuration; defaults to :class:`IQMBackendConfig`.
        backend
            Already-resolved backend to compile for. When omitted the backend
            is resolved from ``config``. Callers that also submit should pass
            the backend they will submit to, so compile and submit cannot
            diverge.

        Returns
        -------
        qiskit.QuantumCircuit
            Circuit compiled for the resolved backend.

        Raises
        ------
        IQMTargetCompilationError
            If the circuit cannot be compiled for that backend.

        """
        cfg = config or IQMBackendConfig()
        target = self.resolve_backend(cfg) if backend is None else backend
        try:
            return transpile(circuit, backend=target, optimization_level=cfg.optimisation_level)
        except Exception as exc:
            raise IQMTargetCompilationError(
                f"circuit cannot be compiled for IQM backend {_backend_name(target)!r} "
                f"at optimisation level {cfg.optimisation_level}: {exc}"
            ) from exc

    def run_counts(
        self,
        circuit: QuantumCircuit,
        config: IQMBackendConfig | None = None,
    ) -> IQMRunResult:
        """Run one measured circuit on a fake or approved remote IQM backend.

        Parameters
        ----------
        circuit:
            Measured logical circuit; compiled and submitted to one resolved target.
        config:
            Validated routing, shots and timeout. Defaults to the local fake mode.
            Remote execution still requires external approval.

        Returns
        -------
        IQMRunResult
            Provider counts with job identity and compilation provenance.

        Raises
        ------
        IQMTargetCompilationError
            Targeted compilation failed; nothing is submitted.
        ValueError
            Provider counts contain invalid labels or non-integer/negative values.
        RuntimeError
            The single-circuit provider result cannot be decoded.

        Notes
        -----
        Provider errors propagate without automatic resubmission. This synchronous
        adapter does not implement durable job recovery or certify hardware results.

        """
        cfg = config or IQMBackendConfig()
        backend = self.resolve_backend(cfg)
        isa_circuit = self.transpile_circuit(circuit, cfg, backend=backend)

        started = time.time()
        job = backend.run([isa_circuit], shots=cfg.shots)
        result = job.result(timeout=cfg.timeout_s)
        counts = _extract_counts(result)
        wall = time.time() - started

        return IQMRunResult(
            job_id=_job_id(job),
            backend_name=_backend_name(backend),
            counts=counts,
            wall_time_s=wall,
            metadata={
                "provider": "iqm",
                "mode": cfg.mode,
                "shots": cfg.shots,
                "timeout_s": cfg.timeout_s,
                "optimisation_level": cfg.optimisation_level,
                "quantum_computer": cfg.quantum_computer,
                "fake_backend": cfg.fake_backend if cfg.mode == "fake" else None,
                "compiled_for": _backend_name(backend),
                "circuit_depth": isa_circuit.depth(),
                "n_qubits": isa_circuit.num_qubits,
                "total_gates": sum(isa_circuit.count_ops().values()),
            },
        )

    def _resolve_fake_backend(self, fake_backend: str) -> Any:
        key = fake_backend.lower()
        try:
            module_name, class_name, _ = _FAKE_BACKENDS[key]
        except KeyError as exc:
            known = ", ".join(sorted(_FAKE_BACKENDS))
            raise ValueError(
                f"unknown IQM fake backend {fake_backend!r}; expected one of {known}"
            ) from exc
        try:
            module = self._import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ImportError(
                "iqm-client[qiskit] is required for IQM fake backends; install it in "
                "an isolated runner environment such as `.venv-iqm` because current "
                "IQM client releases pin Qiskit below the repository's main Qiskit floor."
            ) from exc
        backend_cls = getattr(module, class_name)
        return backend_cls()

    def _resolve_remote_backend(self, cfg: IQMBackendConfig) -> Any:
        try:
            provider_module = self._import_module("iqm.qiskit_iqm.iqm_provider")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "iqm-client[qiskit] is required for IQM remote execution; install it in "
                "an isolated runner environment such as `.venv-iqm` because current "
                "IQM client releases pin Qiskit below the repository's main Qiskit floor."
            ) from exc
        provider = provider_module.IQMProvider(
            cfg.server_url,
            quantum_computer=cfg.quantum_computer,
        )
        get_backend = getattr(provider, "get_backend", None)
        if callable(get_backend):
            return get_backend()
        return provider.backend()


def iqm_factory() -> IQMQuantumBackend:
    """Entry-point target for the IQM backend."""
    return IQMQuantumBackend()


def _extract_counts(result: Any) -> dict[str, int]:
    get_counts = getattr(result, "get_counts", None)
    if callable(get_counts):
        try:
            raw = get_counts()
        except TypeError:
            raw = get_counts(0)
        if isinstance(raw, list):
            if len(raw) != 1:
                raise RuntimeError("IQM single-circuit execution returned multiple count maps")
            raw = raw[0]
        return _validated_counts(raw)
    results = getattr(result, "results", None)
    if isinstance(results, list) and len(results) == 1:
        data = getattr(results[0], "data", None)
        counts = getattr(data, "counts", None)
        if isinstance(counts, dict):
            return _validated_counts(counts)
    raise RuntimeError("Could not extract IQM counts from backend result")


def _validated_counts(raw: object) -> dict[str, int]:
    """Copy count maps without changing labels or truncating measurements."""
    if not isinstance(raw, Mapping):
        raise ValueError("counts must be a mapping")
    counts = {}
    for label, value in raw.items():
        if not isinstance(label, str) or not label:
            raise ValueError("count labels must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("counts must be nonnegative integers, excluding booleans")
        counts[label] = int(value)
    return counts


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        return str(name())
    if name:
        return str(name)
    return type(backend).__name__


def _job_id(job: Any) -> str:
    job_id = getattr(job, "job_id", None)
    if callable(job_id):
        return str(job_id())
    if job_id:
        return str(job_id)
    return "iqm_job_id_unavailable"


__all__ = [
    "IQMTargetCompilationError",
    "IQMBackendConfig",
    "IQMQuantumBackend",
    "IQMRunResult",
    "iqm_factory",
    "is_iqm_available",
]
