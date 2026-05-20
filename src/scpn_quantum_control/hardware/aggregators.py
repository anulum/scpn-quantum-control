# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- aggregator/provider route matrix
"""First-class aggregator/provider route matrix for the hardware HAL."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .backends import QuantumBackendDescriptor, describe_hal_backend_profile
from .hal import BackendProfile, built_in_backend_profiles

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/-]+$")


@dataclass(frozen=True)
class AggregatorProviderRoute:
    """A declared aggregator/provider combination resolved to a HAL backend."""

    route_id: str
    aggregator: str
    provider: str
    backend_id: str
    adapter_module: str
    sdk_package: str
    ir_formats: tuple[str, ...]
    submit_requires_approval: bool
    target_family: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("route_id", "aggregator", "provider", "backend_id"):
            _validate_token(str(getattr(self, field_name)), field_name)
        _validate_token(self.sdk_package, "sdk_package")
        _validate_token(self.target_family, "target_family")
        if not self.ir_formats:
            raise ValueError("ir_formats must not be empty")
        for ir_format in self.ir_formats:
            _validate_token(ir_format, "ir_formats")
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))


@dataclass(frozen=True)
class ResolvedAggregatorProviderRoute:
    """Executable resolution of an aggregator/provider route."""

    route: AggregatorProviderRoute
    profile: BackendProfile
    descriptor: QuantumBackendDescriptor


def built_in_aggregator_provider_routes() -> tuple[AggregatorProviderRoute, ...]:
    """Return the metadata-only aggregator/provider coverage matrix."""

    profiles = {profile.backend_id: profile for profile in built_in_backend_profiles()}
    routes = (
        _route("aws_braket/aqt", "aws_braket", "aqt", "aws_braket_aqt", ("openqasm3",)),
        _route("aws_braket/ionq", "aws_braket", "ionq", "aws_braket_ionq", ("openqasm3",)),
        _route("aws_braket/iqm", "aws_braket", "iqm", "aws_braket_iqm", ("openqasm3",)),
        _route("aws_braket/quera", "aws_braket", "quera", "aws_braket_quera", ("braket_ahs",)),
        _route(
            "aws_braket/rigetti",
            "aws_braket",
            "rigetti",
            "aws_braket_rigetti",
            ("openqasm3",),
        ),
        _route(
            "aws_braket/amazon_simulators",
            "aws_braket",
            "amazon",
            "aws_braket_sv1",
            ("braket_ir", "openqasm3"),
            notes=("managed_simulator_family", "sv1_dm1_tn1"),
        ),
        _route(
            "direct/ibm_quantum",
            "direct",
            "ibm_quantum",
            "ibm_quantum",
            ("qiskit_qpy", "openqasm3"),
        ),
        _route(
            "direct/dwave",
            "direct",
            "dwave",
            "dwave_leap",
            ("bqm", "ising", "qubo", "mlir"),
            target_family="annealing",
        ),
        _route(
            "direct/iqm",
            "direct",
            "iqm",
            "iqm_cloud",
            ("qiskit_qpy", "openqasm3", "qiskit"),
        ),
        _route(
            "direct/ionq",
            "direct",
            "ionq",
            "ionq_cloud",
            ("ionq_json", "openqasm3", "qir"),
        ),
        _route(
            "direct/oqc",
            "direct",
            "oqc",
            "oqc_cloud",
            ("openqasm3", "qir"),
        ),
        _route(
            "direct/pasqal",
            "direct",
            "pasqal",
            "pasqal_cloud",
            ("pulser", "pasqal_ir", "openqasm3", "mlir"),
            target_family="pasqal_neutral_atom",
        ),
        _route(
            "direct/quandela",
            "direct",
            "quandela",
            "quandela_cloud",
            ("perceval", "openqasm3", "mlir"),
            target_family="quandela_photonic",
        ),
        _route(
            "direct/quantinuum",
            "direct",
            "quantinuum",
            "quantinuum_cloud",
            ("tket", "openqasm3", "qir"),
        ),
        _route(
            "direct/quera",
            "direct",
            "quera",
            "quera_bloqade",
            ("bloqade", "braket_ahs"),
        ),
        _route(
            "direct/rigetti",
            "direct",
            "rigetti",
            "rigetti_qcs",
            ("quil", "openqasm3"),
        ),
        _route(
            "azure_quantum/ionq",
            "azure_quantum",
            "ionq",
            "azure_quantum_ionq",
            ("openqasm3",),
        ),
        _route(
            "azure_quantum/quantinuum",
            "azure_quantum",
            "quantinuum",
            "azure_quantum_quantinuum",
            ("openqasm3",),
        ),
        _route(
            "azure_quantum/rigetti",
            "azure_quantum",
            "rigetti",
            "azure_quantum_rigetti",
            ("quil",),
        ),
        _route(
            "azure_quantum/pasqal",
            "azure_quantum",
            "pasqal",
            "azure_quantum_pasqal",
            ("pasqal_ir", "openqasm3"),
        ),
        _route(
            "azure_quantum/qci_preview",
            "azure_quantum",
            "qci",
            "azure_quantum_qci_preview",
            ("openqasm3", "qir"),
            notes=("private_preview",),
        ),
        _route("qbraid/ionq", "qbraid", "ionq", "qbraid_ionq", ("openqasm3", "qiskit")),
        _route(
            "qbraid/aws_braket",
            "qbraid",
            "aws_braket",
            "qbraid_runtime",
            ("braket_ir", "openqasm3", "qiskit", "cirq"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "qbraid/azure_quantum",
            "qbraid",
            "azure_quantum",
            "qbraid_runtime",
            ("openqasm3", "qiskit", "cirq"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "qbraid/ibm_quantum",
            "qbraid",
            "ibm_quantum",
            "qbraid_runtime",
            ("qiskit", "openqasm3"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "qbraid/equal1",
            "qbraid",
            "equal1",
            "qbraid_runtime",
            ("openqasm3",),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "qbraid/iqm",
            "qbraid",
            "iqm",
            "qbraid_runtime",
            ("qiskit", "openqasm3"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "qbraid/qir_simulator",
            "qbraid",
            "qbraid",
            "qbraid_runtime",
            ("mlir", "openqasm3"),
            notes=("managed_simulator_family", "dynamic_catalog_target"),
        ),
        _route(
            "qbraid/nec_vector_annealer",
            "qbraid",
            "nec_vector_annealer",
            "qbraid_runtime",
            ("pyqubo",),
            notes=("optimisation_target", "dynamic_catalog_target"),
        ),
        _route(
            "qbraid/oqc",
            "qbraid",
            "oqc",
            "qbraid_runtime",
            ("openqasm3", "qiskit"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "qbraid/pasqal",
            "qbraid",
            "pasqal",
            "qbraid_runtime",
            ("openqasm3",),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "qbraid/quantinuum",
            "qbraid",
            "quantinuum",
            "qbraid_runtime",
            ("openqasm3", "tket"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "qbraid/quera",
            "qbraid",
            "quera",
            "qbraid_runtime",
            ("openqasm3", "braket_ir"),
            notes=("qasm_simulator", "dynamic_catalog_target"),
        ),
        _route(
            "qbraid/rigetti",
            "qbraid",
            "rigetti",
            "qbraid_runtime",
            ("quil", "openqasm3"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "strangeworks/aqt",
            "strangeworks",
            "aqt",
            "strangeworks_compute",
            ("openqasm3",),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "strangeworks/ionq",
            "strangeworks",
            "ionq",
            "strangeworks_compute",
            ("openqasm3", "qiskit"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "strangeworks/iqm",
            "strangeworks",
            "iqm",
            "strangeworks_compute",
            ("openqasm3", "qiskit"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "strangeworks/rigetti",
            "strangeworks",
            "rigetti",
            "strangeworks_compute",
            ("quil", "openqasm3"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "strangeworks/ibm_quantum",
            "strangeworks",
            "ibm_quantum",
            "strangeworks_compute",
            ("qiskit", "openqasm3"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "strangeworks/qiskit_runtime",
            "strangeworks",
            "ibm_quantum",
            "strangeworks_compute",
            ("qiskit", "openqasm3"),
            notes=("qiskit_runtime", "dynamic_catalog_target"),
        ),
        _route(
            "strangeworks/quantinuum",
            "strangeworks",
            "quantinuum",
            "strangeworks_compute",
            ("openqasm3", "qiskit"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "strangeworks/quera",
            "strangeworks",
            "quera",
            "strangeworks_compute",
            ("braket_ir", "openqasm3"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "strangeworks/aws_braket",
            "strangeworks",
            "aws_braket",
            "strangeworks_compute",
            ("braket_ir", "openqasm3"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "strangeworks/azure_quantum",
            "strangeworks",
            "azure_quantum",
            "strangeworks_compute",
            ("openqasm3", "qiskit"),
            notes=("dynamic_catalog_target",),
        ),
        _route(
            "strangeworks/classical_hpc",
            "strangeworks",
            "classical_hpc",
            "strangeworks_compute",
            ("mlir",),
            notes=("compute_target_metadata_only",),
        ),
    )
    for route in routes:
        profile = profiles[route.backend_id]
        if not set(route.ir_formats) <= set(profile.ir_formats):
            raise ValueError(
                f"route {route.route_id!r} advertises IR formats not supported by "
                f"profile {profile.backend_id!r}"
            )
    return tuple(sorted(routes, key=lambda route: route.route_id))


def aggregator_provider_routes_for(
    *, aggregator: str | None = None, provider: str | None = None
) -> tuple[AggregatorProviderRoute, ...]:
    """Return declared routes filtered by aggregator and/or provider."""

    routes = built_in_aggregator_provider_routes()
    if aggregator is not None:
        _validate_token(aggregator, "aggregator")
        routes = tuple(route for route in routes if route.aggregator == aggregator)
    if provider is not None:
        _validate_token(provider, "provider")
        routes = tuple(route for route in routes if route.provider == provider)
    return routes


def resolve_aggregator_provider_route(
    *,
    aggregator: str,
    provider: str,
    ir_format: str | None = None,
    route_id: str | None = None,
) -> ResolvedAggregatorProviderRoute:
    """Resolve a broker/provider request to one executable HAL profile."""

    routes = aggregator_provider_routes_for(aggregator=aggregator, provider=provider)
    if route_id is not None:
        _validate_token(route_id, "route_id")
        routes = tuple(route for route in routes if route.route_id == route_id)
    if not routes:
        raise LookupError(f"no aggregator/provider route for {aggregator}/{provider}")
    if ir_format is not None:
        _validate_token(ir_format, "ir_format")
        supported_by_pair = tuple(route for route in routes if ir_format in route.ir_formats)
        if not supported_by_pair:
            raise LookupError(f"unsupported IR {ir_format!r} for {aggregator}/{provider}")
        routes = supported_by_pair
    if len(routes) != 1:
        choices = ", ".join(route.route_id for route in routes)
        raise LookupError(
            f"ambiguous aggregator/provider route for {aggregator}/{provider}: {choices}"
        )

    route = routes[0]
    profiles = {profile.backend_id: profile for profile in built_in_backend_profiles()}
    profile = profiles[route.backend_id]
    return ResolvedAggregatorProviderRoute(
        route=route,
        profile=profile,
        descriptor=describe_hal_backend_profile(route.backend_id),
    )


def _route(
    route_id: str,
    aggregator: str,
    provider: str,
    backend_id: str,
    ir_formats: Sequence[str],
    *,
    notes: Sequence[str] = (),
    target_family: str | None = None,
) -> AggregatorProviderRoute:
    descriptor = describe_hal_backend_profile(backend_id)
    resolved_target_family = (
        target_family
        if target_family is not None
        else descriptor.provider
        if descriptor.provider != "dynamic"
        else provider
    )
    return AggregatorProviderRoute(
        route_id=route_id,
        aggregator=aggregator,
        provider=provider,
        backend_id=backend_id,
        adapter_module=descriptor.adapter_module,
        sdk_package=descriptor.sdk_package,
        ir_formats=tuple(ir_formats),
        submit_requires_approval=descriptor.submit_requires_approval,
        target_family=resolved_target_family,
        notes=tuple(notes),
    )


def _validate_token(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or not _TOKEN_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty identifier token")


__all__ = [
    "AggregatorProviderRoute",
    "ResolvedAggregatorProviderRoute",
    "aggregator_provider_routes_for",
    "built_in_aggregator_provider_routes",
    "resolve_aggregator_provider_route",
]
