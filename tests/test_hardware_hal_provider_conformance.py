# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — HAL provider conformance tests
"""Cross-provider invariants for the built-in HAL route matrix."""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from scpn_quantum_control.hardware.backends import list_hal_backend_descriptors
from scpn_quantum_control.hardware.hal import built_in_backend_profiles
from scpn_quantum_control.hardware.provider_smoke import (
    main as provider_smoke_main,
)
from scpn_quantum_control.hardware.provider_smoke import (
    provider_optional_dependency_matrix,
)


def test_every_builtin_hal_route_has_importable_adapter_module() -> None:
    descriptors = list_hal_backend_descriptors()

    assert descriptors
    for descriptor in descriptors:
        module = importlib.import_module(descriptor.adapter_module)
        assert module is not None, descriptor.name


def test_cloud_routes_are_approval_gated_and_local_routes_are_not() -> None:
    profile_by_id = {profile.backend_id: profile for profile in built_in_backend_profiles()}

    for descriptor in list_hal_backend_descriptors():
        profile = profile_by_id[descriptor.name]
        assert descriptor.submit_requires_approval is profile.is_cloud
        assert descriptor.can_submit is profile.is_cloud
        if not profile.is_cloud:
            assert descriptor.can_simulate is True


def test_direct_and_local_provider_routes_do_not_fall_back_to_generic_hal_module() -> None:
    allowed_generic = {"local_statevector"}

    for descriptor in list_hal_backend_descriptors():
        if descriptor.name in allowed_generic:
            continue
        assert descriptor.adapter_module != "scpn_quantum_control.hardware.hal", descriptor.name


def test_every_dedicated_hal_adapter_has_focused_adapter_tests() -> None:
    tests_dir = Path(__file__).resolve().parent
    available = {path.name for path in tests_dir.glob("test_hardware_hal*_adapters.py")}

    expected = {
        "test_hardware_hal_azure_adapters.py",
        "test_hardware_hal_braket_adapters.py",
        "test_hardware_hal_cirq_adapters.py",
        "test_hardware_hal_dwave_adapters.py",
        "test_hardware_hal_ionq_adapters.py",
        "test_hardware_hal_iqm_adapters.py",
        "test_hardware_hal_oqc_adapters.py",
        "test_hardware_hal_pasqal_adapters.py",
        "test_hardware_hal_pennylane_adapters.py",
        "test_hardware_hal_qbraid_adapters.py",
        "test_hardware_hal_qiskit_adapters.py",
        "test_hardware_hal_quandela_adapters.py",
        "test_hardware_hal_quantinuum_adapters.py",
        "test_hardware_hal_quera_bloqade_adapters.py",
        "test_hardware_hal_rigetti_adapters.py",
    }

    assert expected <= available


def test_optional_dependency_matrix_covers_all_non_builtin_provider_modules() -> None:
    matrix = provider_optional_dependency_matrix()
    by_backend = {row.backend_id: row for row in matrix}

    for descriptor in list_hal_backend_descriptors():
        if descriptor.name == "local_statevector":
            continue
        row = by_backend[descriptor.name]
        assert row.adapter_module == descriptor.adapter_module
        assert row.sdk_package == descriptor.sdk_package
        assert row.import_names
        assert isinstance(row.available, bool)


def test_hal_sdk_packages_are_exposed_as_install_extras() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    provider_packages = {
        descriptor.sdk_package
        for descriptor in list_hal_backend_descriptors()
        if descriptor.sdk_package != "python"
    }
    isolated_provider_packages = {"bloqade", "dwave-cloud-client", "iqm-client"}
    declared_packages = {
        _normalise_requirement(requirement)
        for requirements in extras.values()
        for requirement in requirements
        if not requirement.startswith("scpn-quantum-control[")
    }

    assert provider_packages <= declared_packages

    provider_extra_requirements = {
        _normalise_requirement(requirement)
        for requirement in extras["providers"]
        if not requirement.startswith("scpn-quantum-control[")
    }
    assert provider_packages - isolated_provider_packages <= provider_extra_requirements
    assert isolated_provider_packages.isdisjoint(provider_extra_requirements)
    isolated_extra_packages = {
        _normalise_requirement(requirement)
        for extra_name in ("dwave", "iqm", "quera")
        for requirement in extras[extra_name]
        if not requirement.startswith("scpn-quantum-control[")
    }
    assert isolated_provider_packages <= isolated_extra_packages

    all_extra_references = {
        extra
        for requirement in extras["all"]
        for extra in re.findall(r"scpn-quantum-control\[([^\]]+)\]", requirement)
    }
    assert "providers" in {
        part.strip() for reference in all_extra_references for part in reference.split(",")
    }


def _normalise_requirement(requirement: str) -> str:
    return re.split(r"\s*(?:[<>=!~;\[])", requirement, maxsplit=1)[0].strip().lower()


def test_provider_smoke_cli_emits_offline_json_matrix(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = provider_smoke_main(["--format", "json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert {
        "backend_id",
        "provider",
        "sdk_package",
        "adapter_module",
        "import_names",
        "available",
        "missing_imports",
    } <= set(row)
