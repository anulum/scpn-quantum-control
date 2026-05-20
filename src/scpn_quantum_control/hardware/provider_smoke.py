# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — HAL provider optional-dependency smoke matrix
"""Metadata-only optional dependency smoke checks for HAL provider routes."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from .backends import list_hal_backend_descriptors

_SDK_IMPORTS: dict[str, tuple[str, ...]] = {
    "amazon-braket-sdk": ("braket",),
    "azure-quantum": ("azure.quantum",),
    "cirq-core": ("cirq",),
    "dimod": ("dimod",),
    "dwave-cloud-client": ("dimod", "dwave.system"),
    "ionq": ("requests",),
    "iqm-client": ("iqm", "qiskit_iqm"),
    "oqc-qcaas-client": ("qcaas_client",),
    "pennylane": ("pennylane",),
    "perceval-quandela": ("perceval",),
    "pulser-core": ("pulser",),
    "pyquil": ("pyquil",),
    "pytket-quantinuum": ("pytket", "pytket.extensions.quantinuum"),
    "qbraid": ("qbraid",),
    "qiskit-aer": ("qiskit_aer",),
    "qiskit-ibm-runtime": ("qiskit", "qiskit_ibm_runtime"),
    "python": ("scpn_quantum_control",),
}


@dataclass(frozen=True)
class ProviderOptionalDependencyRow:
    """Offline import-probe result for one built-in HAL backend route."""

    backend_id: str
    provider: str
    sdk_package: str
    adapter_module: str
    import_names: tuple[str, ...]
    available: bool
    missing_imports: tuple[str, ...]


def provider_optional_dependency_matrix() -> tuple[ProviderOptionalDependencyRow, ...]:
    """Return metadata-only import availability for every built-in HAL route.

    The probe uses ``importlib.util.find_spec`` only. It does not import provider
    SDKs, read credentials, create clients, authenticate, or touch the network.
    """

    rows: list[ProviderOptionalDependencyRow] = []
    for descriptor in list_hal_backend_descriptors():
        import_names = _SDK_IMPORTS.get(descriptor.sdk_package, (descriptor.sdk_package,))
        missing = tuple(name for name in import_names if not _module_available(name))
        rows.append(
            ProviderOptionalDependencyRow(
                backend_id=descriptor.name,
                provider=descriptor.provider,
                sdk_package=descriptor.sdk_package,
                adapter_module=descriptor.adapter_module,
                import_names=import_names,
                available=not missing,
                missing_imports=missing,
            )
        )
    return tuple(rows)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


__all__ = ["ProviderOptionalDependencyRow", "provider_optional_dependency_matrix"]
