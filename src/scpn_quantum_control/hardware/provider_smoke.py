# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — HAL provider optional-dependency smoke matrix
"""Metadata-only optional dependency smoke checks for HAL provider routes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .backends import list_hal_backend_descriptors

_SDK_IMPORTS: dict[str, tuple[str, ...]] = {
    "amazon-braket-sdk": ("braket",),
    "azure-quantum": ("azure.quantum",),
    "cirq-core": ("cirq",),
    "dimod": ("dimod",),
    "dwave-cloud-client": ("dimod", "dwave.system"),
    "iqm-client": ("iqm.iqm_client",),
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
    "requests": ("requests",),
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


def main(argv: Sequence[str] | None = None) -> int:
    """Print the offline provider optional-dependency matrix.

    The command is intentionally metadata-only: it imports no provider SDK,
    reads no credentials, creates no clients, performs no authentication, and
    touches no network endpoint. Use ``--require-all`` in provider-pack CI lanes
    after installing ``scpn-quantum-control[providers]``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Return exit code 1 when any declared provider import is missing.",
    )
    args = parser.parse_args(argv)

    rows = provider_optional_dependency_matrix()
    if args.format == "json":
        print(json.dumps([asdict(row) for row in rows], sort_keys=True))
    else:
        print(_format_table(rows))
    if args.require_all and any(not row.available for row in rows):
        return 1
    return 0


def _format_table(rows: Sequence[ProviderOptionalDependencyRow]) -> str:
    lines = ["backend_id\tprovider\tsdk_package\tavailable\tmissing_imports"]
    lines.extend(
        "\t".join(
            (
                row.backend_id,
                row.provider,
                row.sdk_package,
                str(row.available).lower(),
                ",".join(row.missing_imports),
            )
        )
        for row in rows
    )
    return "\n".join(lines)


__all__ = ["ProviderOptionalDependencyRow", "main", "provider_optional_dependency_matrix"]
