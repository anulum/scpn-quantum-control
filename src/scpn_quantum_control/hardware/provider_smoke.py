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
import sys
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

_ISOLATED_PROVIDER_EXTRAS: dict[str, tuple[str, ...]] = {
    "dwave": ("dwave_leap",),
    "iqm": ("iqm_cloud",),
    "quera": ("quera_bloqade",),
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


@dataclass(frozen=True)
class IsolatedProviderSmokeLane:
    """Deterministic command lane for an SDK family isolated from ``[providers]``."""

    extra: str
    backend_ids: tuple[str, ...]
    sdk_packages: tuple[str, ...]
    venv_path: str
    create_command: tuple[str, ...]
    install_command: tuple[str, ...]
    smoke_command: tuple[str, ...]
    rationale: str


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


def isolated_provider_smoke_lanes() -> tuple[IsolatedProviderSmokeLane, ...]:
    """Return isolated smoke commands for provider SDKs excluded from ``[providers]``.

    D-Wave, IQM, and QuEra SDK families are intentionally kept out of the
    portable provider bundle because their current dependency trees can conflict
    with shared development and application extras. These lanes are offline:
    they only install one provider extra and run the metadata import probe.
    """

    rows_by_backend = {row.backend_id: row for row in provider_optional_dependency_matrix()}
    lanes: list[IsolatedProviderSmokeLane] = []
    for extra, backend_ids in _ISOLATED_PROVIDER_EXTRAS.items():
        sdk_packages = tuple(
            dict.fromkeys(rows_by_backend[backend_id].sdk_package for backend_id in backend_ids)
        )
        venv_path = f".venv-provider-{extra}"
        smoke_command = _smoke_command(venv_path, backend_ids)
        lanes.append(
            IsolatedProviderSmokeLane(
                extra=extra,
                backend_ids=backend_ids,
                sdk_packages=sdk_packages,
                venv_path=venv_path,
                create_command=("python", "-m", "venv", venv_path),
                install_command=(
                    f"{venv_path}/bin/python",
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    f".[{extra}]",
                ),
                smoke_command=smoke_command,
                rationale=(
                    "kept outside the portable providers extra because this SDK family "
                    "currently requires an isolated resolver environment"
                ),
            )
        )
    return tuple(lanes)


def _smoke_command(venv_path: str, backend_ids: Sequence[str]) -> tuple[str, ...]:
    command: list[str] = [f"{venv_path}/bin/scpn-provider-smoke", "--format", "table"]
    for backend_id in backend_ids:
        command.extend(("--backend", backend_id))
    command.append("--require-all")
    return tuple(command)


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
    parser.add_argument(
        "--backend",
        action="append",
        dest="backend_ids",
        help="Restrict the matrix to one backend id. May be supplied more than once.",
    )
    parser.add_argument(
        "--sdk-package",
        action="append",
        dest="sdk_packages",
        help="Restrict the matrix to one SDK package. May be supplied more than once.",
    )
    parser.add_argument(
        "--plan-isolated",
        action="store_true",
        help="Print isolated smoke lanes for SDK families excluded from [providers].",
    )
    args = parser.parse_args(argv)

    if args.plan_isolated:
        lanes = isolated_provider_smoke_lanes()
        if args.format == "json":
            print(json.dumps([asdict(lane) for lane in lanes], sort_keys=True))
        else:
            print(_format_isolated_plan(lanes))
        return 0

    rows = _select_rows(
        provider_optional_dependency_matrix(),
        backend_ids=args.backend_ids or (),
        sdk_packages=args.sdk_packages or (),
    )
    if not rows:
        print("no provider smoke rows matched the requested filters", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps([asdict(row) for row in rows], sort_keys=True))
    else:
        print(_format_table(rows))
    if args.require_all and any(not row.available for row in rows):
        return 1
    return 0


def _select_rows(
    rows: Sequence[ProviderOptionalDependencyRow],
    *,
    backend_ids: Sequence[str],
    sdk_packages: Sequence[str],
) -> tuple[ProviderOptionalDependencyRow, ...]:
    selected = tuple(rows)
    if backend_ids:
        requested = set(backend_ids)
        selected = tuple(row for row in selected if row.backend_id in requested)
    if sdk_packages:
        requested = set(sdk_packages)
        selected = tuple(row for row in selected if row.sdk_package in requested)
    return selected


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


def _format_isolated_plan(rows: Sequence[IsolatedProviderSmokeLane]) -> str:
    lines = [
        "extra\tbackend_ids\tsdk_packages\tvenv_path\tcreate_command\tinstall_command\tsmoke_command"
    ]
    lines.extend(
        "\t".join(
            (
                lane.extra,
                ",".join(lane.backend_ids),
                ",".join(lane.sdk_packages),
                lane.venv_path,
                " ".join(lane.create_command),
                " ".join(lane.install_command),
                " ".join(lane.smoke_command),
            )
        )
        for lane in rows
    )
    return "\n".join(lines)


__all__ = [
    "IsolatedProviderSmokeLane",
    "ProviderOptionalDependencyRow",
    "isolated_provider_smoke_lanes",
    "main",
    "provider_optional_dependency_matrix",
]
