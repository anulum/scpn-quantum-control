# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Backend plugin registry
# Language policy: EXEMPT from the Rust-path rule. This module is a
# `dict[str, Callable]` registry with no numeric compute. See
# docs/language_policy.md §"Current-state audit" for the rationale.
"""Plugin / backend extension API.

Closes audit item C10. Third parties may now register additional quantum
backends without editing this repository by declaring an entry point in
the ``scpn_quantum_control.backends`` group:

.. code-block:: toml

    [project.entry-points."scpn_quantum_control.backends"]
    acme_trapped_ion = "acme_plugin:AcmeBackend"
    analog_kuramoto = "scpn_quantum_control.hardware.analog_kuramoto:analog_kuramoto_factory"
    hybrid_digital_analog = "scpn_quantum_control.hardware.hybrid_digital_analog:hybrid_digital_analog_factory"

The entry-point target must be a zero-argument callable or a class that
returns an object satisfying the :class:`BackendProtocol` interface when
instantiated.

Internal backends (Qiskit runtime, PennyLane) register themselves via
this repository's own ``pyproject.toml`` entry points; they are loaded
the same way every third-party backend is, which means a broken plugin
cannot take down the rest of the registry.

Discovery is lazy: call :func:`discover_backends` once per process. The
module also exposes a manual :func:`register_backend` hatch for tests
and notebooks that want to exercise a specific class without round-trip
through entry points.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

ENTRY_POINT_GROUP = "scpn_quantum_control.backends"

logger = logging.getLogger(__name__)

_HAL_PROFILE_ADAPTER_MODULES: dict[str, str] = {
    "aws_braket_aqt": "scpn_quantum_control.hardware.hal_braket",
    "aws_braket_dm1": "scpn_quantum_control.hardware.hal_braket",
    "aws_braket_ionq": "scpn_quantum_control.hardware.hal_braket",
    "aws_braket_iqm": "scpn_quantum_control.hardware.hal_braket",
    "aws_braket_quera": "scpn_quantum_control.hardware.hal_braket",
    "aws_braket_rigetti": "scpn_quantum_control.hardware.hal_braket",
    "aws_braket_sv1": "scpn_quantum_control.hardware.hal_braket",
    "aws_braket_tn1": "scpn_quantum_control.hardware.hal_braket",
    "azure_quantum_ionq": "scpn_quantum_control.hardware.hal_azure",
    "azure_quantum_ionq_simulator": "scpn_quantum_control.hardware.hal_azure",
    "azure_quantum_pasqal": "scpn_quantum_control.hardware.hal_azure",
    "azure_quantum_pasqal_emulator": "scpn_quantum_control.hardware.hal_azure",
    "azure_quantum_qci_preview": "scpn_quantum_control.hardware.hal_azure",
    "azure_quantum_quantinuum": "scpn_quantum_control.hardware.hal_azure",
    "azure_quantum_quantinuum_emulator": "scpn_quantum_control.hardware.hal_azure",
    "azure_quantum_rigetti": "scpn_quantum_control.hardware.hal_azure",
    "azure_quantum_rigetti_qvm": "scpn_quantum_control.hardware.hal_azure",
    "dwave_leap": "scpn_quantum_control.hardware.hal_dwave",
    "ibm_quantum": "scpn_quantum_control.hardware.hal_qiskit",
    "ionq_cloud": "scpn_quantum_control.hardware.hal_ionq",
    "iqm_cloud": "scpn_quantum_control.hardware.hal_iqm",
    "local_braket_ahs": "scpn_quantum_control.hardware.hal_braket",
    "local_braket_dm": "scpn_quantum_control.hardware.hal_braket",
    "local_braket_sv": "scpn_quantum_control.hardware.hal_braket",
    "local_cirq": "scpn_quantum_control.hardware.hal_cirq",
    "local_pennylane": "scpn_quantum_control.hardware.hal_pennylane",
    "local_qiskit_aer": "scpn_quantum_control.hardware.hal_qiskit",
    "local_statevector": "scpn_quantum_control.hardware.hal",
    "oqc_cloud": "scpn_quantum_control.hardware.hal_oqc",
    "pasqal_cloud": "scpn_quantum_control.hardware.hal_pasqal",
    "qbraid_ionq": "scpn_quantum_control.hardware.hal_qbraid",
    "qbraid_runtime": "scpn_quantum_control.hardware.hal_qbraid",
    "quandela_cloud": "scpn_quantum_control.hardware.hal_quandela",
    "quantinuum_cloud": "scpn_quantum_control.hardware.hal_quantinuum",
    "quera_bloqade": "scpn_quantum_control.hardware.hal_quera_bloqade",
    "rigetti_qcs": "scpn_quantum_control.hardware.hal_rigetti",
    "strangeworks_compute": "scpn_quantum_control.hardware.hal_strangeworks",
}


@runtime_checkable
class BackendProtocol(Protocol):
    """Minimum contract every registered backend must satisfy.

    The registry intentionally asks for very little — enough to identify
    the backend and check whether it is runnable in the current
    environment. Anything beyond that (circuit submission, retrieval
    semantics) is delegated to adapter-specific subtypes.
    """

    name: str

    def is_available(self) -> bool:
        """True iff the backend can run in the current environment."""
        ...  # pragma: no cover - Protocol declaration only.


BackendFactory = Callable[[], BackendProtocol]


@dataclass(frozen=True)
class QuantumBackendDescriptor:
    """Provider-neutral execution contract for a quantum backend.

    Registry lookup must never authenticate, touch the network, or queue
    paid work. This static descriptor gives routing code enough
    information to distinguish local simulation from approval-gated cloud
    submission before execution-specific adapters are invoked.
    """

    name: str
    provider: str
    execution_mode: str
    sdk_package: str
    adapter_module: str
    available: bool
    can_simulate: bool
    can_submit: bool
    submit_requires_approval: bool
    supports_shots: bool
    supports_statevector: bool
    supports_mid_circuit_measurement: bool
    supports_pulse: bool
    max_qubits: int | None
    capabilities: tuple[str, ...]
    workloads: tuple[str, ...]
    notes: tuple[str, ...] = ()


class BackendRegistrationError(RuntimeError):
    """Raised when a backend fails to load or violates the contract."""


class BackendRegistry:
    """In-memory backend registry.

    Backends are keyed by their ``name`` attribute. Registration is
    idempotent (same-class re-registration is a no-op); registering a
    different factory under an existing name raises
    :class:`BackendRegistrationError` so silent overrides do not happen
    in production.
    """

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}
        self._discovered: bool = False

    # -- public API ---------------------------------------------------------

    def register(self, name: str, factory: BackendFactory) -> None:
        """Register a backend factory under ``name``."""
        if not isinstance(name, str) or not name:
            raise BackendRegistrationError(f"Backend name must be a non-empty str, got {name!r}")
        if not callable(factory):
            raise BackendRegistrationError(
                f"Backend factory for {name!r} is not callable",
            )
        existing = self._factories.get(name)
        if existing is not None and existing is not factory:
            raise BackendRegistrationError(
                f"Backend {name!r} already registered with a different factory",
            )
        self._factories[name] = factory

    def unregister(self, name: str) -> None:
        """Remove a backend from the registry (useful for tests)."""
        self._factories.pop(name, None)

    def get(self, name: str) -> BackendProtocol:
        """Instantiate the backend registered under ``name``.

        Raises :class:`KeyError` if no such backend is registered.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(
                f"Unknown backend {name!r}; known backends: {sorted(self._factories)}",
            )
        backend = factory()
        if not isinstance(backend, BackendProtocol):
            raise BackendRegistrationError(
                f"Backend {name!r} factory returned {type(backend).__name__}, "
                "which does not satisfy BackendProtocol",
            )
        return backend

    def names(self) -> list[str]:
        """All registered backend names in insertion order."""
        return list(self._factories)

    def clear(self) -> None:
        """Wipe the registry (tests only)."""
        self._factories.clear()
        self._discovered = False

    # -- discovery ----------------------------------------------------------

    def discover(self, *, force: bool = False) -> list[str]:
        """Load every backend advertised under the entry-point group.

        Returns the list of names successfully registered during this
        call. A plugin that raises at load time is logged and skipped —
        one broken plugin never blocks the rest.
        """
        if self._discovered and not force:
            return []

        loaded: list[str] = []
        # ``requires-python = ">=3.10"`` — the ``group=`` kwarg is always
        # available, no legacy fallback needed.
        entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)

        for ep in entries:
            try:
                factory = ep.load()
            except Exception as exc:  # pragma: no cover - diagnostic branch
                logger.warning(
                    "Failed to load backend plugin %r from %s: %s",
                    ep.name,
                    getattr(ep, "value", "<unknown>"),
                    exc,
                )
                continue
            try:
                self.register(ep.name, factory)
                loaded.append(ep.name)
            except BackendRegistrationError as exc:
                logger.warning("Rejected backend %r: %s", ep.name, exc)

        self._discovered = True
        return loaded


# Module-level singleton — tests can reach in via :func:`get_registry`.
_registry = BackendRegistry()


def get_registry() -> BackendRegistry:
    """Return the process-wide backend registry."""
    return _registry


def register_backend(name: str, factory: BackendFactory) -> None:
    """Convenience wrapper for :meth:`BackendRegistry.register`."""
    _registry.register(name, factory)


def unregister_backend(name: str) -> None:
    """Convenience wrapper for :meth:`BackendRegistry.unregister`."""
    _registry.unregister(name)


def get_backend(name: str) -> BackendProtocol:
    """Convenience wrapper for :meth:`BackendRegistry.get`."""
    return _registry.get(name)


def describe_backend(name: str) -> QuantumBackendDescriptor:
    """Return the provider-neutral descriptor for ``name``.

    Third-party backends that have not implemented ``descriptor()`` get a
    conservative descriptor: no advertised submit/simulator capability
    and explicit approval required before production routing.
    """
    backend = get_backend(name)
    descriptor = getattr(backend, "descriptor", None)
    if callable(descriptor):
        value = descriptor()
        if not isinstance(value, QuantumBackendDescriptor):
            raise BackendRegistrationError(
                f"Backend {name!r} descriptor returned {type(value).__name__}, "
                "expected QuantumBackendDescriptor",
            )
        return value

    return QuantumBackendDescriptor(
        name=name,
        provider="external",
        execution_mode="unknown",
        sdk_package="unknown",
        adapter_module=type(backend).__module__,
        available=backend.is_available(),
        can_simulate=False,
        can_submit=False,
        submit_requires_approval=True,
        supports_shots=False,
        supports_statevector=False,
        supports_mid_circuit_measurement=False,
        supports_pulse=False,
        max_qubits=None,
        capabilities=(),
        workloads=(),
        notes=("Legacy backend: implement descriptor() before production routing.",),
    )


def discover_backends(*, force: bool = False) -> list[str]:
    """Convenience wrapper for :meth:`BackendRegistry.discover`."""
    return _registry.discover(force=force)


def list_backends(*, auto_discover: bool = True) -> list[str]:
    """Return every known backend name.

    When ``auto_discover`` is True (default) this triggers a one-shot
    entry-point scan so users do not have to remember to call
    :func:`discover_backends` first.
    """
    if auto_discover:
        discover_backends()
    return _registry.names()


def list_quantum_backends(*, auto_discover: bool = True) -> list[QuantumBackendDescriptor]:
    """Return sorted provider-neutral descriptors for every known backend."""
    if auto_discover:
        discover_backends()
    return sorted((describe_backend(name) for name in _registry.names()), key=lambda d: d.name)


def describe_hal_backend_profile(backend_id: str) -> QuantumBackendDescriptor:
    """Return selector metadata for one built-in HAL profile.

    The descriptor is constructed from static HAL profile metadata only. It
    does not import provider SDKs, authenticate, inspect queues, or create any
    executable adapter. Runtime availability remains the responsibility of the
    injected adapter route.
    """

    from .hal import built_in_backend_profiles

    for profile in built_in_backend_profiles():
        if profile.backend_id == backend_id:
            return _hal_profile_descriptor(profile)
    raise KeyError(f"unknown HAL backend profile: {backend_id}")


def list_hal_backend_descriptors() -> list[QuantumBackendDescriptor]:
    """Return selector metadata for all built-in HAL profiles."""

    from .hal import built_in_backend_profiles

    return sorted(
        (_hal_profile_descriptor(profile) for profile in built_in_backend_profiles()),
        key=lambda descriptor: descriptor.name,
    )


def _hal_profile_descriptor(profile: object) -> QuantumBackendDescriptor:
    from .hal import BackendProfile

    if not isinstance(profile, BackendProfile):
        raise TypeError("profile must be a BackendProfile")
    capabilities = _hal_profile_capability_tokens(profile)
    return QuantumBackendDescriptor(
        name=profile.backend_id,
        provider=profile.provider,
        execution_mode=_hal_profile_execution_mode(profile),
        sdk_package=profile.sdk_package,
        adapter_module=_HAL_PROFILE_ADAPTER_MODULES[profile.backend_id],
        available=profile.sdk_package == "python",
        can_simulate=_hal_profile_can_simulate(profile),
        can_submit=profile.is_cloud,
        submit_requires_approval=profile.submit_requires_approval,
        supports_shots=profile.capabilities.supports_shots,
        supports_statevector=profile.capabilities.supports_statevector,
        supports_mid_circuit_measurement=profile.capabilities.supports_mid_circuit_measurement,
        supports_pulse=profile.capabilities.supports_pulse,
        max_qubits=profile.capabilities.max_qubits,
        capabilities=capabilities,
        workloads=tuple(profile.ir_formats),
        notes=(
            "Metadata-only HAL descriptor; runtime availability is checked by injected adapters.",
            *profile.notes,
        ),
    )


def _hal_profile_execution_mode(profile: object) -> str:
    from .hal import BackendProfile

    if not isinstance(profile, BackendProfile):
        raise TypeError("profile must be a BackendProfile")
    if not profile.is_cloud:
        if "simulator" in profile.modality or "statevector" in profile.modality:
            return "local_simulator"
        return "local_adapter"
    if profile.capabilities.supports_analog:
        return "cloud_neutral_atom_analog"
    if "simulator" in profile.modality or "emulator" in profile.modality:
        return "cloud_managed_simulator"
    if profile.modality == "quantum_annealing":
        return "cloud_quantum_annealing"
    return "cloud_qpu"


def _hal_profile_can_simulate(profile: object) -> bool:
    from .hal import BackendProfile

    if not isinstance(profile, BackendProfile):
        raise TypeError("profile must be a BackendProfile")
    return (
        not profile.is_cloud
        or profile.capabilities.supports_statevector
        or "simulator" in profile.modality
        or "emulator" in profile.modality
    )


def _hal_profile_capability_tokens(profile: object) -> tuple[str, ...]:
    from .hal import BackendProfile

    if not isinstance(profile, BackendProfile):
        raise TypeError("profile must be a BackendProfile")
    capabilities: list[str] = []
    if profile.capabilities.supports_analog:
        capabilities.append("analog")
    if profile.capabilities.supports_counts:
        capabilities.append("counts")
    if profile.capabilities.supports_mid_circuit_measurement:
        capabilities.append("mid_circuit_measurement")
    if profile.capabilities.supports_pulse:
        capabilities.append("pulse")
    if profile.capabilities.supports_shots:
        capabilities.append("shots")
    if profile.capabilities.supports_statevector:
        capabilities.append("statevector")
    if profile.capabilities.supports_cancellation:
        capabilities.append("cancellation")
    if profile.capabilities.supports_cost_estimate:
        capabilities.append("cost_estimate")
    return tuple(sorted(capabilities))


# ---------------------------------------------------------------------------
# Built-in backend entries — zero-argument factories that the repository
# registers via its own pyproject.toml entry points. Third-party plugins
# follow the same pattern.
# ---------------------------------------------------------------------------


class _QiskitIBMBackend:
    """Built-in backend for IBM Quantum hardware via qiskit-ibm-runtime.

    ``is_available`` checks whether ``qiskit_ibm_runtime`` is importable
    — it deliberately does not authenticate because that would require a
    token and network access at registry-lookup time.
    """

    name = "qiskit_ibm"

    def is_available(self) -> bool:
        """Return whether the IBM Runtime package is importable."""
        try:
            import qiskit_ibm_runtime  # noqa: F401
        except Exception:
            return False
        return True

    def descriptor(self) -> QuantumBackendDescriptor:
        """Return the non-submitting IBM Runtime capability descriptor."""
        return QuantumBackendDescriptor(
            name=self.name,
            provider="ibm_quantum",
            execution_mode="cloud_qpu",
            sdk_package="qiskit-ibm-runtime",
            adapter_module="scpn_quantum_control.hardware.runner",
            available=self.is_available(),
            can_simulate=False,
            can_submit=True,
            submit_requires_approval=True,
            supports_shots=True,
            supports_statevector=False,
            supports_mid_circuit_measurement=True,
            supports_pulse=False,
            max_qubits=None,
            capabilities=(
                "runtime_sampler",
                "runtime_estimator",
                "dynamic_circuits",
                "hardware_counts",
            ),
            workloads=("kuramoto_xy", "dla_parity", "fim_feedback", "qec_feedback"),
            notes=(
                "Descriptor does not authenticate or submit jobs.",
                "Live submission remains gated by the hardware approval scheduler.",
            ),
        )


class _QiskitAerBackend:
    """Built-in backend for local Qiskit Aer simulation."""

    name = "qiskit_aer"

    def is_available(self) -> bool:
        """Return whether Qiskit Aer is importable."""
        try:
            import qiskit_aer  # noqa: F401
        except Exception:
            return False
        return True

    def descriptor(self) -> QuantumBackendDescriptor:
        """Return the local Aer simulator capability descriptor."""
        return QuantumBackendDescriptor(
            name=self.name,
            provider="local_qiskit_aer",
            execution_mode="local_simulator",
            sdk_package="qiskit-aer",
            adapter_module="scpn_quantum_control.hardware.runner",
            available=self.is_available(),
            can_simulate=True,
            can_submit=False,
            submit_requires_approval=False,
            supports_shots=True,
            supports_statevector=True,
            supports_mid_circuit_measurement=False,
            supports_pulse=False,
            max_qubits=None,
            capabilities=("aer_sampler", "noise_model", "statevector"),
            workloads=("kuramoto_xy", "dla_parity", "fim_feedback", "mitigation"),
            notes=("Local simulator path; no cloud credentials or QPU submission.",),
        )


class _CirqBackend:
    """Built-in backend for Cirq simulator/circuit export capability."""

    name = "cirq"

    def is_available(self) -> bool:
        """Return whether the Cirq adapter can be imported."""
        try:
            from .cirq_adapter import is_cirq_available
        except Exception:
            return False
        return is_cirq_available()

    def descriptor(self) -> QuantumBackendDescriptor:
        """Return the Cirq capability descriptor."""
        return QuantumBackendDescriptor(
            name=self.name,
            provider="google_cirq",
            execution_mode="local_simulator",
            sdk_package="cirq-core",
            adapter_module="scpn_quantum_control.hardware.cirq_adapter",
            available=self.is_available(),
            can_simulate=True,
            can_submit=False,
            submit_requires_approval=False,
            supports_shots=True,
            supports_statevector=True,
            supports_mid_circuit_measurement=False,
            supports_pulse=False,
            max_qubits=None,
            capabilities=("cirq_circuit", "cirq_simulator"),
            workloads=("kuramoto_xy", "cross_platform_export"),
            notes=("Current built-in Cirq path is local simulation/export.",),
        )


class _BraketBackend:
    """Built-in capability descriptor for Amazon Braket integrations."""

    name = "braket"

    def is_available(self) -> bool:
        """Return whether the Amazon Braket SDK is importable."""
        try:
            import braket.aws  # noqa: F401
        except Exception:
            return False
        return True

    def descriptor(self) -> QuantumBackendDescriptor:
        """Return the Amazon Braket capability descriptor."""
        return QuantumBackendDescriptor(
            name=self.name,
            provider="aws_braket",
            execution_mode="cloud_qpu_or_managed_simulator",
            sdk_package="amazon-braket-sdk",
            adapter_module="scpn_quantum_control.hardware.pennylane_adapter",
            available=self.is_available(),
            can_simulate=True,
            can_submit=True,
            submit_requires_approval=True,
            supports_shots=True,
            supports_statevector=False,
            supports_mid_circuit_measurement=False,
            supports_pulse=False,
            max_qubits=None,
            capabilities=("braket_device", "managed_simulator", "qpu_submission"),
            workloads=("kuramoto_xy", "cross_platform_hardware"),
            notes=(
                "Current repository route is through PennyLane Braket device strings.",
                "Live AWS submission requires explicit approval and external credentials.",
            ),
        )


class _PennyLaneBackend:
    """Built-in backend for the PennyLane adapter."""

    name = "pennylane"

    def is_available(self) -> bool:
        """Return whether the PennyLane adapter reports availability."""
        from .pennylane_adapter import is_pennylane_available

        return is_pennylane_available()

    def descriptor(self) -> QuantumBackendDescriptor:
        """Return the PennyLane multi-provider router descriptor."""
        return QuantumBackendDescriptor(
            name=self.name,
            provider="pennylane",
            execution_mode="adapter_router",
            sdk_package="pennylane",
            adapter_module="scpn_quantum_control.hardware.pennylane_adapter",
            available=self.is_available(),
            can_simulate=True,
            can_submit=True,
            submit_requires_approval=False,
            supports_shots=True,
            supports_statevector=False,
            supports_mid_circuit_measurement=False,
            supports_pulse=False,
            max_qubits=None,
            capabilities=("device_router", "autodiff", "vqe", "trotter"),
            workloads=("kuramoto_xy", "differentiable_circuits", "cross_platform_hardware"),
            notes=("Provider-specific PennyLane plugins may impose their own approval gates.",),
        )


def qiskit_ibm_factory() -> BackendProtocol:
    """Entry-point target for the IBM runtime backend."""
    return _QiskitIBMBackend()


def qiskit_aer_factory() -> BackendProtocol:
    """Entry-point target for the local Qiskit Aer backend."""
    return _QiskitAerBackend()


def cirq_factory() -> BackendProtocol:
    """Entry-point target for the Cirq backend."""
    return _CirqBackend()


def braket_factory() -> BackendProtocol:
    """Entry-point target for the Amazon Braket backend."""
    return _BraketBackend()


def pennylane_factory() -> BackendProtocol:
    """Entry-point target for the PennyLane backend."""
    return _PennyLaneBackend()


def iqm_factory() -> BackendProtocol:
    """Entry-point target for the IQM Qiskit backend."""
    from .iqm_backend import iqm_factory as _factory

    return _factory()


def analog_kuramoto_factory() -> BackendProtocol:
    """Entry-point target for the analog Kuramoto compiler backend."""
    from .analog_kuramoto import analog_kuramoto_factory as _factory

    return _factory()


def hybrid_digital_analog_factory() -> BackendProtocol:
    """Entry-point target for the hybrid digital-analog compiler backend."""
    from .hybrid_digital_analog import hybrid_digital_analog_factory as _factory

    return _factory()


# Pre-register the built-ins so the registry is useful even on a source
# checkout where the entry points haven't been installed.
_registry.register("qiskit_ibm", qiskit_ibm_factory)
_registry.register("qiskit_aer", qiskit_aer_factory)
_registry.register("cirq", cirq_factory)
_registry.register("braket", braket_factory)
_registry.register("pennylane", pennylane_factory)
_registry.register("iqm", iqm_factory)
_registry.register("analog_kuramoto", analog_kuramoto_factory)
_registry.register("hybrid_digital_analog", hybrid_digital_analog_factory)


__all__ = [
    "ENTRY_POINT_GROUP",
    "BackendFactory",
    "BackendProtocol",
    "BackendRegistrationError",
    "BackendRegistry",
    "QuantumBackendDescriptor",
    "analog_kuramoto_factory",
    "braket_factory",
    "cirq_factory",
    "describe_backend",
    "discover_backends",
    "get_backend",
    "get_registry",
    "hybrid_digital_analog_factory",
    "iqm_factory",
    "list_backends",
    "list_quantum_backends",
    "pennylane_factory",
    "qiskit_aer_factory",
    "qiskit_ibm_factory",
    "register_backend",
    "unregister_backend",
]
