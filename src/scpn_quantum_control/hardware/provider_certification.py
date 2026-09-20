# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — HAL provider certification gate
"""Fail-closed certification of the publicly claimed HAL provider matrix.

A provider route may be claimed on a public surface only once every
certification criterion below is satisfied by repository evidence. This module
derives the criteria from the live declared matrix rather than from a frozen
allow-list, so a newly declared backend is refused until its documentation,
focused adapter tests, dependency evidence and approval semantics exist.

The surface is metadata-only: it imports adapter modules, reads repository
files, and resolves declared routes. It reads no credentials, opens no provider
session, contacts no network target and submits no job.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from .aggregators import (
    AggregatorProviderRoute,
    built_in_aggregator_provider_routes,
    resolve_aggregator_provider_route,
)
from .backends import QuantumBackendDescriptor, list_hal_backend_descriptors
from .hal import BackendProfile, built_in_backend_profiles
from .provider_capability_core import build_provider_route_catalogue
from .provider_smoke import provider_optional_dependency_matrix

CriterionStatus = Literal["satisfied", "failed", "not_applicable"]
"""Outcome of one certification criterion for one declared backend."""

CERTIFICATION_CRITERIA: Final[tuple[str, ...]] = (
    "hal_profile_resolution",
    "adapter_module_import",
    "broker_route_consistency",
    "capability_catalogue_row",
    "optional_dependency_smoke",
    "public_documentation",
    "focused_adapter_tests",
    "approval_gated_submission",
)
"""Criteria evaluated for every declared backend, in report order."""

PUBLIC_BACKEND_TABLE_DOCUMENT: Final[str] = "docs/backends.md"
"""Public document owning the claimed backend-identifier table."""

PUBLIC_BACKEND_TABLE_HEADER: Final[str] = "| Backend id | Provider | Broker | Modality |"
"""Exact header row that opens the claimed backend table."""

LOCAL_SDK_PACKAGE: Final[str] = "python"
"""SDK marker of the built-in backend that needs no optional dependency."""

GENERIC_HAL_ADAPTER_TEST: Final[str] = "tests/test_hardware_hal.py"
"""Focused owner of the generic in-repository HAL adapter."""

_ADAPTER_MODULE_PREFIX: Final[str] = "scpn_quantum_control.hardware."
_ADAPTER_FAMILY_PREFIX: Final[str] = "hal_"
_GENERIC_ADAPTER_MODULE: Final[str] = "hal"
_TABLE_ROW = re.compile(r"^\|\s*`(?P<backend_id>[A-Za-z0-9_]+)`\s*\|")
_CATALOGUE_OBSERVATION_DATE: Final[str] = "1970-01-01"


@dataclass(frozen=True)
class CertificationCriterion:
    """Evidence outcome of one criterion for one declared backend.

    Parameters
    ----------
    criterion
        Name from :data:`CERTIFICATION_CRITERIA`.
    status
        ``satisfied`` when repository evidence proves the criterion, ``failed``
        when the evidence is missing or contradictory, and ``not_applicable``
        when the criterion cannot apply to this backend class.
    detail
        Specific reason naming the evidence that was found or missing. A
        ``not_applicable`` detail must name why the class is excluded, never
        merely that nothing was checked.

    """

    criterion: str
    status: CriterionStatus
    detail: str

    def __post_init__(self) -> None:
        """Refuse an unknown criterion name or an unexplained outcome."""
        if self.criterion not in CERTIFICATION_CRITERIA:
            raise ValueError(f"unknown certification criterion: {self.criterion}")
        if not self.detail.strip():
            raise ValueError(f"{self.criterion}: detail must explain the outcome")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable mapping of this criterion outcome."""
        return {"criterion": self.criterion, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class ProviderCertificationRecord:
    """Certification evidence for one declared HAL backend.

    Parameters
    ----------
    backend_id
        Declared HAL backend identifier.
    provider
        Provider identity carried by the backend profile.
    adapter_module
        Module that owns execution or export for the backend.
    broker_route_ids
        Declared aggregator/provider routes naming this backend, in sorted
        order. Empty means the backend is reached directly, not through a
        broker row.
    criteria
        One outcome per name in :data:`CERTIFICATION_CRITERIA`, in that order.

    """

    backend_id: str
    provider: str
    adapter_module: str
    broker_route_ids: tuple[str, ...]
    criteria: tuple[CertificationCriterion, ...]

    def __post_init__(self) -> None:
        """Require exactly one outcome per criterion, in canonical order."""
        observed = tuple(criterion.criterion for criterion in self.criteria)
        if observed != CERTIFICATION_CRITERIA:
            raise ValueError(f"{self.backend_id}: criteria must be complete and in order")

    @property
    def certified(self) -> bool:
        """Return whether no criterion failed for this backend."""
        return all(criterion.status != "failed" for criterion in self.criteria)

    @property
    def failures(self) -> tuple[CertificationCriterion, ...]:
        """Return every failed criterion, preserving report order."""
        return tuple(criterion for criterion in self.criteria if criterion.status == "failed")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping of this backend record."""
        return {
            "backend_id": self.backend_id,
            "provider": self.provider,
            "adapter_module": self.adapter_module,
            "broker_route_ids": list(self.broker_route_ids),
            "certified": self.certified,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
        }


@dataclass(frozen=True)
class ProviderCertificationReport:
    """Certification outcome for the whole declared provider matrix.

    Parameters
    ----------
    records
        One record per declared backend, ordered by backend identifier.
    undeclared_documented_backends
        Backend identifiers claimed by the public table that the declared
        matrix does not implement. Any entry blocks certification.

    """

    records: tuple[ProviderCertificationRecord, ...]
    undeclared_documented_backends: tuple[str, ...]

    @property
    def certified(self) -> bool:
        """Return whether every backend passed and no claim lacks an implementation."""
        return not self.undeclared_documented_backends and all(
            record.certified for record in self.records
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping of the whole report."""
        return {
            "certified": self.certified,
            "undeclared_documented_backends": list(self.undeclared_documented_backends),
            "records": [record.to_dict() for record in self.records],
        }


def resolve_source_root(candidate: Path) -> Path:
    """Validate a repository checkout root that owns the certification evidence.

    Parameters
    ----------
    candidate
        Directory expected to contain the public backend table and the focused
        adapter test directory.

    Returns
    -------
    Path
        The resolved absolute checkout root.

    Raises
    ------
    ValueError
        If the path is not an existing directory, or does not carry both the
        public backend document and a ``tests`` directory.

    """
    root = candidate.expanduser()
    if not root.is_dir():
        raise ValueError(f"source root is not a directory: {candidate}")
    root = root.resolve()
    if not (root / PUBLIC_BACKEND_TABLE_DOCUMENT).is_file():
        raise ValueError(f"source root lacks {PUBLIC_BACKEND_TABLE_DOCUMENT}: {root}")
    if not (root / "tests").is_dir():
        raise ValueError(f"source root lacks a tests directory: {root}")
    return root


def documented_backend_ids(source_root: Path) -> tuple[str, ...]:
    """Read the publicly claimed backend identifiers from the backend table.

    Parameters
    ----------
    source_root
        Validated checkout root, as returned by :func:`resolve_source_root`.

    Returns
    -------
    tuple of str
        Claimed backend identifiers in document order.

    Raises
    ------
    ValueError
        If the claimed table header is absent or the table carries no row. A
        gate that cannot read its evidence must not report a pass.

    """
    document = (source_root / PUBLIC_BACKEND_TABLE_DOCUMENT).read_text(encoding="utf-8")
    lines = document.splitlines()
    try:
        header_index = lines.index(PUBLIC_BACKEND_TABLE_HEADER)
    except ValueError as error:
        raise ValueError(
            f"{PUBLIC_BACKEND_TABLE_DOCUMENT} has no claimed backend table header"
        ) from error
    claimed: list[str] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        match = _TABLE_ROW.match(line)
        if match is None:
            break
        claimed.append(match.group("backend_id"))
    if not claimed:
        raise ValueError(f"{PUBLIC_BACKEND_TABLE_DOCUMENT} claimed backend table has no row")
    return tuple(claimed)


def focused_adapter_test_path(adapter_module: str) -> str:
    """Derive the focused adapter test owner required for one adapter module.

    Parameters
    ----------
    adapter_module
        Fully qualified adapter module declared by a backend descriptor.

    Returns
    -------
    str
        Repository-relative path of the required focused adapter test file.

    Raises
    ------
    ValueError
        If the module is outside the hardware package or does not follow the
        ``hal_<family>`` adapter naming contract. An adapter that cannot name
        its test owner is refused rather than exempted.

    """
    if not adapter_module.startswith(_ADAPTER_MODULE_PREFIX):
        raise ValueError(f"adapter module outside the hardware package: {adapter_module}")
    leaf = adapter_module[len(_ADAPTER_MODULE_PREFIX) :]
    if leaf == _GENERIC_ADAPTER_MODULE:
        return GENERIC_HAL_ADAPTER_TEST
    if not leaf.startswith(_ADAPTER_FAMILY_PREFIX) or leaf == _ADAPTER_FAMILY_PREFIX:
        raise ValueError(f"adapter module does not follow the hal_<family> contract: {leaf}")
    family = leaf[len(_ADAPTER_FAMILY_PREFIX) :]
    return f"tests/test_hardware_hal_{family}_adapters.py"


def certify_provider_matrix(
    *,
    source_root: Path,
    descriptors: Sequence[QuantumBackendDescriptor] | None = None,
    routes: Sequence[AggregatorProviderRoute] | None = None,
    profiles: Sequence[BackendProfile] | None = None,
) -> ProviderCertificationReport:
    """Certify every declared backend against the public-claim criteria.

    Parameters
    ----------
    source_root
        Validated checkout root owning the documentation and test evidence.
    descriptors
        Declared backend descriptors. Defaults to the built-in HAL matrix.
    routes
        Declared aggregator/provider routes. Defaults to the built-in matrix.
    profiles
        Authoritative HAL profiles. Defaults to the built-in profiles.

    Returns
    -------
    ProviderCertificationReport
        One record per declared backend plus any public claim without an
        implementation.

    Raises
    ------
    ValueError
        If the claimed backend table cannot be read, the declared matrix
        repeats a backend identifier, or a declared route cannot be inventoried
        against an authoritative profile carrying the same SDK.

    Notes
    -----
    The broker-route criterion uses the production route selector. A supplied
    route that the selector does not own is refused rather than accepted on the
    strength of the caller's own table.

    """
    descriptor_rows = (
        tuple(descriptors) if descriptors is not None else tuple(list_hal_backend_descriptors())
    )
    profile_rows = tuple(profiles) if profiles is not None else tuple(built_in_backend_profiles())
    route_rows = (
        tuple(routes) if routes is not None else tuple(built_in_aggregator_provider_routes())
    )
    profile_by_id = {profile.backend_id: profile for profile in profile_rows}
    declared_ids = [descriptor.name for descriptor in descriptor_rows]
    if len(set(declared_ids)) != len(declared_ids):
        raise ValueError("declared descriptors repeat a backend identifier")
    documented = documented_backend_ids(source_root)
    dependency_backend_ids = frozenset(
        row.backend_id for row in provider_optional_dependency_matrix()
    )
    records = tuple(
        _certify_backend(
            descriptor=descriptor,
            profile=profile_by_id.get(descriptor.name),
            routes=tuple(route for route in route_rows if route.backend_id == descriptor.name),
            documented=frozenset(documented),
            dependency_backend_ids=dependency_backend_ids,
            profiles=profile_rows,
            source_root=source_root,
        )
        for descriptor in sorted(descriptor_rows, key=lambda row: row.name)
    )
    return ProviderCertificationReport(
        records=records,
        undeclared_documented_backends=tuple(
            sorted(frozenset(documented) - frozenset(declared_ids))
        ),
    )


def _certify_backend(
    *,
    descriptor: QuantumBackendDescriptor,
    profile: BackendProfile | None,
    routes: tuple[AggregatorProviderRoute, ...],
    documented: frozenset[str],
    dependency_backend_ids: frozenset[str],
    profiles: tuple[BackendProfile, ...],
    source_root: Path,
) -> ProviderCertificationRecord:
    criteria = (
        _check_hal_profile(descriptor, profile),
        _check_adapter_import(descriptor),
        _check_broker_routes(descriptor, routes),
        _check_capability_catalogue(descriptor, routes, profiles),
        _check_optional_dependency(descriptor, dependency_backend_ids),
        _check_documentation(descriptor, documented),
        _check_focused_tests(descriptor, source_root),
        _check_approval_semantics(descriptor, profile),
    )
    return ProviderCertificationRecord(
        backend_id=descriptor.name,
        provider=descriptor.provider,
        adapter_module=descriptor.adapter_module,
        broker_route_ids=tuple(sorted(route.route_id for route in routes)),
        criteria=criteria,
    )


def _check_hal_profile(
    descriptor: QuantumBackendDescriptor, profile: BackendProfile | None
) -> CertificationCriterion:
    if profile is None:
        return CertificationCriterion(
            criterion="hal_profile_resolution",
            status="failed",
            detail=f"{descriptor.name} has no HAL backend profile",
        )
    mismatches = [
        f"{field}: descriptor {descriptor_value!r} vs profile {profile_value!r}"
        for field, descriptor_value, profile_value in (
            ("provider", descriptor.provider, profile.provider),
            ("sdk_package", descriptor.sdk_package, profile.sdk_package),
        )
        if descriptor_value != profile_value
    ]
    if mismatches:
        return CertificationCriterion(
            criterion="hal_profile_resolution",
            status="failed",
            detail=f"{descriptor.name} disagrees with its profile: {'; '.join(mismatches)}",
        )
    return CertificationCriterion(
        criterion="hal_profile_resolution",
        status="satisfied",
        detail=f"{descriptor.name} resolves to a matching HAL profile",
    )


def _check_adapter_import(descriptor: QuantumBackendDescriptor) -> CertificationCriterion:
    try:
        importlib.import_module(descriptor.adapter_module)
    except ImportError as error:
        return CertificationCriterion(
            criterion="adapter_module_import",
            status="failed",
            detail=f"{descriptor.adapter_module} failed to import: {error}",
        )
    return CertificationCriterion(
        criterion="adapter_module_import",
        status="satisfied",
        detail=f"{descriptor.adapter_module} imports without a provider SDK",
    )


def _check_broker_routes(
    descriptor: QuantumBackendDescriptor, routes: tuple[AggregatorProviderRoute, ...]
) -> CertificationCriterion:
    if not routes:
        return CertificationCriterion(
            criterion="broker_route_consistency",
            status="not_applicable",
            detail=f"{descriptor.name} is a direct HAL target named by no broker route",
        )
    for route in routes:
        for ir_format in route.ir_formats:
            try:
                resolved = resolve_aggregator_provider_route(
                    aggregator=route.aggregator,
                    provider=route.provider,
                    ir_format=ir_format,
                    route_id=route.route_id,
                )
            except LookupError as error:
                return CertificationCriterion(
                    criterion="broker_route_consistency",
                    status="failed",
                    detail=f"{route.route_id} is unresolvable by the route selector: {error}",
                )
            if resolved.descriptor.name != descriptor.name:
                return CertificationCriterion(
                    criterion="broker_route_consistency",
                    status="failed",
                    detail=(
                        f"{route.route_id} with IR {ir_format} resolves to "
                        f"{resolved.descriptor.name}, not {descriptor.name}"
                    ),
                )
    return CertificationCriterion(
        criterion="broker_route_consistency",
        status="satisfied",
        detail=f"{len(routes)} broker route(s) resolve to {descriptor.name} for every declared IR",
    )


def _check_capability_catalogue(
    descriptor: QuantumBackendDescriptor,
    routes: tuple[AggregatorProviderRoute, ...],
    profiles: tuple[BackendProfile, ...],
) -> CertificationCriterion:
    if not routes:
        return CertificationCriterion(
            criterion="capability_catalogue_row",
            status="not_applicable",
            detail=f"{descriptor.name} has no broker route to inventory",
        )
    try:
        entries = build_provider_route_catalogue(
            observed_at=_CATALOGUE_OBSERVATION_DATE,
            routes=routes,
            profiles=profiles,
        )
    except ValueError as error:
        return CertificationCriterion(
            criterion="capability_catalogue_row",
            status="failed",
            detail=f"{descriptor.name} cannot be inventoried without submission: {error}",
        )
    return CertificationCriterion(
        criterion="capability_catalogue_row",
        status="satisfied",
        detail=(
            f"{len(entries)} no-submit capability row(s) cover {descriptor.name} "
            "with unknown-by-default support"
        ),
    )


def _check_optional_dependency(
    descriptor: QuantumBackendDescriptor, dependency_backend_ids: frozenset[str]
) -> CertificationCriterion:
    if descriptor.sdk_package == LOCAL_SDK_PACKAGE:
        return CertificationCriterion(
            criterion="optional_dependency_smoke",
            status="not_applicable",
            detail=f"{descriptor.name} ships in-repository and declares no provider SDK",
        )
    if descriptor.name not in dependency_backend_ids:
        return CertificationCriterion(
            criterion="optional_dependency_smoke",
            status="failed",
            detail=f"{descriptor.name} has no offline optional-dependency smoke row",
        )
    return CertificationCriterion(
        criterion="optional_dependency_smoke",
        status="satisfied",
        detail=f"{descriptor.sdk_package} availability is probed offline for {descriptor.name}",
    )


def _check_documentation(
    descriptor: QuantumBackendDescriptor, documented: frozenset[str]
) -> CertificationCriterion:
    if descriptor.name not in documented:
        return CertificationCriterion(
            criterion="public_documentation",
            status="failed",
            detail=(
                f"{descriptor.name} is absent from the claimed backend table in "
                f"{PUBLIC_BACKEND_TABLE_DOCUMENT}"
            ),
        )
    return CertificationCriterion(
        criterion="public_documentation",
        status="satisfied",
        detail=f"{descriptor.name} is claimed in {PUBLIC_BACKEND_TABLE_DOCUMENT}",
    )


def _check_focused_tests(
    descriptor: QuantumBackendDescriptor, source_root: Path
) -> CertificationCriterion:
    try:
        relative = focused_adapter_test_path(descriptor.adapter_module)
    except ValueError as error:
        return CertificationCriterion(
            criterion="focused_adapter_tests",
            status="failed",
            detail=f"{descriptor.name}: {error}",
        )
    if not (source_root / relative).is_file():
        return CertificationCriterion(
            criterion="focused_adapter_tests",
            status="failed",
            detail=f"{descriptor.name} requires a focused adapter suite at {relative}",
        )
    return CertificationCriterion(
        criterion="focused_adapter_tests",
        status="satisfied",
        detail=f"{relative} owns the focused adapter suite for {descriptor.name}",
    )


def _check_approval_semantics(
    descriptor: QuantumBackendDescriptor, profile: BackendProfile | None
) -> CertificationCriterion:
    if profile is None:
        return CertificationCriterion(
            criterion="approval_gated_submission",
            status="failed",
            detail=f"{descriptor.name} has no profile declaring a submission policy",
        )
    if descriptor.submit_requires_approval is not profile.is_cloud:
        return CertificationCriterion(
            criterion="approval_gated_submission",
            status="failed",
            detail=(
                f"{descriptor.name} approval flag {descriptor.submit_requires_approval} "
                f"disagrees with cloud status {profile.is_cloud}"
            ),
        )
    if descriptor.can_submit is not profile.is_cloud:
        return CertificationCriterion(
            criterion="approval_gated_submission",
            status="failed",
            detail=(
                f"{descriptor.name} submission capability {descriptor.can_submit} "
                f"disagrees with cloud status {profile.is_cloud}"
            ),
        )
    if not profile.is_cloud and not descriptor.can_simulate:
        return CertificationCriterion(
            criterion="approval_gated_submission",
            status="failed",
            detail=f"{descriptor.name} is local but exposes no simulation path",
        )
    policy = "approval-gated cloud submission" if profile.is_cloud else "local simulation only"
    return CertificationCriterion(
        criterion="approval_gated_submission",
        status="satisfied",
        detail=f"{descriptor.name} declares {policy}",
    )


def render_report_table(report: ProviderCertificationReport) -> str:
    """Render one operator-readable line per backend plus a closing verdict.

    Parameters
    ----------
    report
        Report returned by :func:`certify_provider_matrix`.

    Returns
    -------
    str
        Newline-separated table without a trailing newline.

    """
    width = max((len(record.backend_id) for record in report.records), default=0)
    lines = [
        f"{record.backend_id:<{width}}  {'CERTIFIED' if record.certified else 'REFUSED'}"
        + (
            ""
            if record.certified
            else "  " + "; ".join(criterion.detail for criterion in record.failures)
        )
        for record in report.records
    ]
    for backend_id in report.undeclared_documented_backends:
        lines.append(f"{backend_id:<{width}}  REFUSED  claimed publicly but never declared")
    lines.append(f"verdict: {'certified' if report.certified else 'refused'}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the provider certification gate over a checkout.

    Parameters
    ----------
    argv
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` when every declared backend is certified, ``1`` when a backend is
        refused, and ``2`` when the evidence itself cannot be read.

    """
    parser = argparse.ArgumentParser(
        prog="scpn-provider-certification",
        description=(
            "Certify the declared HAL provider matrix against the public-claim "
            "criteria. Metadata only: no credential read, no network call, no "
            "job submission."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="Repository checkout owning the documentation and test evidence.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Report rendering (default: table).",
    )
    args = parser.parse_args(argv)
    try:
        report = certify_provider_matrix(source_root=resolve_source_root(args.source_root))
    except ValueError as error:
        print(f"provider certification evidence unavailable: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_report_table(report))
    return 0 if report.certified else 1


__all__ = [
    "CERTIFICATION_CRITERIA",
    "GENERIC_HAL_ADAPTER_TEST",
    "LOCAL_SDK_PACKAGE",
    "PUBLIC_BACKEND_TABLE_DOCUMENT",
    "PUBLIC_BACKEND_TABLE_HEADER",
    "CertificationCriterion",
    "CriterionStatus",
    "ProviderCertificationRecord",
    "ProviderCertificationReport",
    "certify_provider_matrix",
    "documented_backend_ids",
    "focused_adapter_test_path",
    "main",
    "render_report_table",
    "resolve_source_root",
]
