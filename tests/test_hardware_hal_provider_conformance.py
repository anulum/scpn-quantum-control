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

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from scpn_quantum_control.hardware.aggregators import (
    aggregator_provider_routes_for,
    built_in_aggregator_provider_routes,
    resolve_aggregator_provider_route,
)
from scpn_quantum_control.hardware.backends import list_hal_backend_descriptors
from scpn_quantum_control.hardware.hal import built_in_backend_profiles
from scpn_quantum_control.hardware.provider_smoke import (
    aggregator_provider_optional_dependency_matrix,
    isolated_provider_smoke_lanes,
    provider_optional_dependency_matrix,
)
from scpn_quantum_control.hardware.provider_smoke import (
    main as provider_smoke_main,
)


def test_every_builtin_hal_route_has_importable_adapter_module() -> None:
    descriptors = list_hal_backend_descriptors()

    assert descriptors
    for descriptor in descriptors:
        module = importlib.import_module(descriptor.adapter_module)
        assert module is not None, descriptor.name


def test_aggregator_provider_routes_resolve_to_first_class_hal_profiles() -> None:
    """Every declared aggregator-provider combination must hit a real HAL route."""

    descriptors = {descriptor.name: descriptor for descriptor in list_hal_backend_descriptors()}
    profiles = {profile.backend_id: profile for profile in built_in_backend_profiles()}
    routes = built_in_aggregator_provider_routes()

    assert routes
    assert [route.route_id for route in routes] == sorted(route.route_id for route in routes)
    assert len({route.route_id for route in routes}) == len(routes)

    for route in routes:
        assert route.backend_id in profiles
        assert route.backend_id in descriptors
        assert route.adapter_module == descriptors[route.backend_id].adapter_module
        assert route.sdk_package == descriptors[route.backend_id].sdk_package
        assert set(route.ir_formats) <= set(profiles[route.backend_id].ir_formats)
        if profiles[route.backend_id].is_cloud:
            assert route.submit_requires_approval is True


def test_aggregator_provider_matrix_covers_source_grounded_current_brokers() -> None:
    """The matrix should expose concrete current aggregator/provider combinations."""

    route_ids = {route.route_id for route in built_in_aggregator_provider_routes()}

    expected = {
        "aws_braket/aqt",
        "aws_braket/ionq",
        "aws_braket/iqm",
        "aws_braket/quera",
        "aws_braket/rigetti",
        "aws_braket/amazon_simulators",
        "direct/dwave",
        "direct/ibm_quantum",
        "direct/iqm",
        "direct/ionq",
        "direct/oqc",
        "direct/pasqal",
        "direct/quantinuum",
        "direct/quera",
        "direct/rigetti",
        "azure_quantum/ionq",
        "azure_quantum/quantinuum",
        "azure_quantum/rigetti",
        "azure_quantum/pasqal",
        "azure_quantum/qci_preview",
        "qbraid/ionq",
        "qbraid/aws_braket",
        "qbraid/azure_quantum",
        "qbraid/ibm_quantum",
        "qbraid/equal1",
        "qbraid/iqm",
        "qbraid/qir_simulator",
        "qbraid/nec_vector_annealer",
        "qbraid/oqc",
        "qbraid/pasqal",
        "qbraid/quantinuum",
        "qbraid/quera",
        "qbraid/rigetti",
        "strangeworks/aqt",
        "strangeworks/ionq",
        "strangeworks/iqm",
        "strangeworks/rigetti",
        "strangeworks/ibm_quantum",
        "strangeworks/aws_braket",
        "strangeworks/azure_quantum",
        "strangeworks/classical_hpc",
        "strangeworks/qiskit_runtime",
        "strangeworks/quantinuum",
        "strangeworks/quera",
    }

    assert expected <= route_ids
    assert {route.provider for route in aggregator_provider_routes_for(aggregator="qbraid")} >= {
        "aws_braket",
        "azure_quantum",
        "ibm_quantum",
        "ionq",
        "iqm",
        "nec_vector_annealer",
        "oqc",
        "pasqal",
        "quantinuum",
        "quera",
        "rigetti",
    }
    assert [route.route_id for route in aggregator_provider_routes_for(provider="dwave")] == [
        "direct/dwave",
    ]
    assert [route.route_id for route in aggregator_provider_routes_for(provider="rigetti")] == [
        "aws_braket/rigetti",
        "azure_quantum/rigetti",
        "direct/rigetti",
        "qbraid/rigetti",
        "strangeworks/rigetti",
    ]
    assert [route.route_id for route in aggregator_provider_routes_for(provider="iqm")] == [
        "aws_braket/iqm",
        "direct/iqm",
        "qbraid/iqm",
        "strangeworks/iqm",
    ]
    assert [route.route_id for route in aggregator_provider_routes_for(provider="quera")] == [
        "aws_braket/quera",
        "direct/quera",
        "qbraid/quera",
        "strangeworks/quera",
    ]
    assert [route.route_id for route in aggregator_provider_routes_for(provider="oqc")] == [
        "direct/oqc",
        "qbraid/oqc",
    ]
    assert [route.route_id for route in aggregator_provider_routes_for(provider="pasqal")] == [
        "azure_quantum/pasqal",
        "direct/pasqal",
        "qbraid/pasqal",
    ]


def test_aggregator_provider_selector_resolves_profile_and_ir() -> None:
    """Routing code should resolve matrix rows to executable HAL profiles."""

    resolved = resolve_aggregator_provider_route(
        aggregator="qbraid",
        provider="aws_braket",
        ir_format="braket_ir",
    )

    assert resolved.route.route_id == "qbraid/aws_braket"
    assert resolved.profile.backend_id == "qbraid_runtime"
    assert resolved.descriptor.name == "qbraid_runtime"
    assert resolved.route.submit_requires_approval is True

    direct = resolve_aggregator_provider_route(
        aggregator="aws_braket",
        provider="ionq",
        ir_format="openqasm3",
    )
    assert direct.route.backend_id == "aws_braket_ionq"
    assert direct.profile.provider == "ionq"

    ibm = resolve_aggregator_provider_route(
        aggregator="direct",
        provider="ibm_quantum",
        ir_format="qiskit_qpy",
    )
    assert ibm.route.route_id == "direct/ibm_quantum"
    assert ibm.route.backend_id == "ibm_quantum"
    assert ibm.profile.provider == "ibm"

    dwave = resolve_aggregator_provider_route(
        aggregator="direct",
        provider="dwave",
        ir_format="bqm",
    )
    assert dwave.route.route_id == "direct/dwave"
    assert dwave.route.backend_id == "dwave_leap"
    assert dwave.profile.provider == "dwave"

    ionq = resolve_aggregator_provider_route(
        aggregator="direct",
        provider="ionq",
        ir_format="ionq_json",
    )
    assert ionq.route.route_id == "direct/ionq"
    assert ionq.route.backend_id == "ionq_cloud"
    assert ionq.profile.provider == "ionq"

    quantinuum = resolve_aggregator_provider_route(
        aggregator="direct",
        provider="quantinuum",
        ir_format="tket",
    )
    assert quantinuum.route.route_id == "direct/quantinuum"
    assert quantinuum.route.backend_id == "quantinuum_cloud"
    assert quantinuum.profile.provider == "quantinuum"

    rigetti = resolve_aggregator_provider_route(
        aggregator="direct",
        provider="rigetti",
        ir_format="quil",
    )
    assert rigetti.route.route_id == "direct/rigetti"
    assert rigetti.route.backend_id == "rigetti_qcs"
    assert rigetti.profile.provider == "rigetti"

    iqm = resolve_aggregator_provider_route(
        aggregator="direct",
        provider="iqm",
        ir_format="qiskit_qpy",
    )
    assert iqm.route.route_id == "direct/iqm"
    assert iqm.route.backend_id == "iqm_cloud"
    assert iqm.profile.provider == "iqm"

    quera = resolve_aggregator_provider_route(
        aggregator="direct",
        provider="quera",
        ir_format="bloqade",
    )
    assert quera.route.route_id == "direct/quera"
    assert quera.route.backend_id == "quera_bloqade"
    assert quera.profile.provider == "quera"

    oqc = resolve_aggregator_provider_route(
        aggregator="direct",
        provider="oqc",
        ir_format="openqasm3",
    )
    assert oqc.route.route_id == "direct/oqc"
    assert oqc.route.backend_id == "oqc_cloud"
    assert oqc.profile.provider == "oqc"

    pasqal = resolve_aggregator_provider_route(
        aggregator="direct",
        provider="pasqal",
        ir_format="pulser",
    )
    assert pasqal.route.route_id == "direct/pasqal"
    assert pasqal.route.backend_id == "pasqal_cloud"
    assert pasqal.profile.provider == "pasqal"

    with pytest.raises(LookupError, match="unsupported IR"):
        resolve_aggregator_provider_route(
            aggregator="qbraid",
            provider="aws_braket",
            ir_format="quil",
        )

    with pytest.raises(LookupError, match="no aggregator/provider route"):
        resolve_aggregator_provider_route(aggregator="qbraid", provider="missing_provider")


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
        "test_hardware_hal_strangeworks_adapters.py",
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


def test_aggregator_provider_dependency_matrix_covers_every_route() -> None:
    """Aggregator routes should carry executable dependency evidence."""

    rows = aggregator_provider_optional_dependency_matrix()
    by_route = {row.route_id: row for row in rows}
    routes = built_in_aggregator_provider_routes()

    assert len(rows) == len(routes)
    assert set(by_route) == {route.route_id for route in routes}

    qbraid_rigetti = by_route["qbraid/rigetti"]
    assert qbraid_rigetti.aggregator == "qbraid"
    assert qbraid_rigetti.provider == "rigetti"
    assert qbraid_rigetti.backend_id == "qbraid_runtime"
    assert qbraid_rigetti.sdk_package == "qbraid"
    assert "quil" in qbraid_rigetti.ir_formats
    assert qbraid_rigetti.dynamic_catalog_target is True
    assert isinstance(qbraid_rigetti.available, bool)

    braket_ionq = by_route["aws_braket/ionq"]
    assert braket_ionq.backend_id == "aws_braket_ionq"
    assert braket_ionq.target_family == "ionq"
    assert braket_ionq.submit_requires_approval is True

    direct_dwave = by_route["direct/dwave"]
    assert direct_dwave.aggregator == "direct"
    assert direct_dwave.provider == "dwave"
    assert direct_dwave.backend_id == "dwave_leap"
    assert direct_dwave.target_family == "annealing"
    assert "bqm" in direct_dwave.ir_formats
    assert direct_dwave.submit_requires_approval is True

    direct_ionq = by_route["direct/ionq"]
    assert direct_ionq.aggregator == "direct"
    assert direct_ionq.provider == "ionq"
    assert direct_ionq.backend_id == "ionq_cloud"
    assert direct_ionq.target_family == "ionq"
    assert "ionq_json" in direct_ionq.ir_formats
    assert direct_ionq.submit_requires_approval is True

    direct_quantinuum = by_route["direct/quantinuum"]
    assert direct_quantinuum.aggregator == "direct"
    assert direct_quantinuum.provider == "quantinuum"
    assert direct_quantinuum.backend_id == "quantinuum_cloud"
    assert direct_quantinuum.target_family == "quantinuum"
    assert "tket" in direct_quantinuum.ir_formats
    assert direct_quantinuum.submit_requires_approval is True

    direct_rigetti = by_route["direct/rigetti"]
    assert direct_rigetti.aggregator == "direct"
    assert direct_rigetti.provider == "rigetti"
    assert direct_rigetti.backend_id == "rigetti_qcs"
    assert direct_rigetti.target_family == "rigetti"
    assert "quil" in direct_rigetti.ir_formats
    assert direct_rigetti.submit_requires_approval is True

    direct_iqm = by_route["direct/iqm"]
    assert direct_iqm.aggregator == "direct"
    assert direct_iqm.provider == "iqm"
    assert direct_iqm.backend_id == "iqm_cloud"
    assert direct_iqm.target_family == "iqm"
    assert "qiskit_qpy" in direct_iqm.ir_formats
    assert direct_iqm.submit_requires_approval is True

    direct_quera = by_route["direct/quera"]
    assert direct_quera.aggregator == "direct"
    assert direct_quera.provider == "quera"
    assert direct_quera.backend_id == "quera_bloqade"
    assert direct_quera.target_family == "quera"
    assert "bloqade" in direct_quera.ir_formats
    assert direct_quera.submit_requires_approval is True

    direct_oqc = by_route["direct/oqc"]
    assert direct_oqc.aggregator == "direct"
    assert direct_oqc.provider == "oqc"
    assert direct_oqc.backend_id == "oqc_cloud"
    assert direct_oqc.target_family == "oqc"
    assert "openqasm3" in direct_oqc.ir_formats
    assert direct_oqc.submit_requires_approval is True

    direct_pasqal = by_route["direct/pasqal"]
    assert direct_pasqal.aggregator == "direct"
    assert direct_pasqal.provider == "pasqal"
    assert direct_pasqal.backend_id == "pasqal_cloud"
    assert direct_pasqal.target_family == "pasqal_neutral_atom"
    assert "pulser" in direct_pasqal.ir_formats
    assert direct_pasqal.submit_requires_approval is True

    direct_ibm = by_route["direct/ibm_quantum"]
    assert direct_ibm.aggregator == "direct"
    assert direct_ibm.provider == "ibm_quantum"
    assert direct_ibm.backend_id == "ibm_quantum"
    assert "qiskit_qpy" in direct_ibm.ir_formats
    assert direct_ibm.dynamic_catalog_target is False


def test_aggregator_provider_dependency_matrix_filters_by_route_request() -> None:
    """Operators need deterministic scoped preflight rows before live work."""

    rows = aggregator_provider_optional_dependency_matrix(
        aggregator="qbraid",
        provider="rigetti",
        ir_format="quil",
    )

    assert [row.route_id for row in rows] == ["qbraid/rigetti"]

    with pytest.raises(LookupError, match="unsupported IR"):
        aggregator_provider_optional_dependency_matrix(
            aggregator="qbraid",
            provider="rigetti",
            ir_format="braket_ahs",
        )


def test_hal_sdk_packages_are_exposed_as_install_extras() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    provider_packages = {
        descriptor.sdk_package
        for descriptor in list_hal_backend_descriptors()
        if descriptor.sdk_package != "python"
    }
    isolated_provider_packages = {
        "bloqade",
        "dwave-cloud-client",
        "iqm-client",
        "strangeworks",
    }
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
        for extra_name in ("dwave", "iqm", "quera", "strangeworks")
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


def test_provider_smoke_cli_filters_to_one_backend(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = provider_smoke_main(["--format", "json", "--backend", "dwave_leap"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert [row["backend_id"] for row in payload] == ["dwave_leap"]
    assert payload[0]["sdk_package"] == "dwave-cloud-client"


def test_provider_smoke_cli_emits_aggregator_route_dependency_matrix(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = provider_smoke_main(
        [
            "--format",
            "json",
            "--aggregator-routes",
            "--aggregator",
            "qbraid",
            "--provider",
            "rigetti",
            "--ir-format",
            "quil",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert [row["route_id"] for row in payload] == ["qbraid/rigetti"]
    assert payload[0]["backend_id"] == "qbraid_runtime"
    assert payload[0]["dynamic_catalog_target"] is True


def test_provider_smoke_cli_rejects_empty_filter(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = provider_smoke_main(["--format", "json", "--backend", "not_a_route"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "no provider smoke rows matched" in captured.err


def test_provider_smoke_cli_requires_only_selected_sdk(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = provider_smoke_main(
        ["--format", "json", "--sdk-package", "requests", "--require-all"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload
    assert {row["sdk_package"] for row in payload} == {"requests"}


def test_isolated_provider_smoke_lanes_cover_conflict_prone_extras() -> None:
    lanes = isolated_provider_smoke_lanes()
    by_extra = {lane.extra: lane for lane in lanes}

    assert set(by_extra) == {"dwave", "iqm", "quera", "strangeworks"}
    assert by_extra["dwave"].backend_ids == ("dwave_leap",)
    assert by_extra["iqm"].backend_ids == ("iqm_cloud",)
    assert by_extra["quera"].backend_ids == ("quera_bloqade",)
    assert by_extra["strangeworks"].backend_ids == ("strangeworks_compute",)
    assert by_extra["dwave"].sdk_packages == ("dwave-cloud-client",)
    assert by_extra["iqm"].sdk_packages == ("iqm-client",)
    assert by_extra["quera"].sdk_packages == ("bloqade",)
    assert by_extra["strangeworks"].sdk_packages == ("strangeworks",)

    for lane in lanes:
        assert lane.venv_path == f".venv-provider-{lane.extra}"
        assert f".[${lane.extra}]" not in " ".join(lane.install_command)
        assert f".[{lane.extra}]" in " ".join(lane.install_command)
        assert "--require-all" in lane.smoke_command
        for backend_id in lane.backend_ids:
            assert backend_id in lane.smoke_command


def test_provider_smoke_cli_emits_isolated_plan(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = provider_smoke_main(["--format", "json", "--plan-isolated"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert {row["extra"] for row in payload} == {"dwave", "iqm", "quera", "strangeworks"}
    assert all(row["install_command"] for row in payload)
