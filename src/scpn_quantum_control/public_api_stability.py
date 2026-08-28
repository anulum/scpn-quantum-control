# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — public API stability programme
"""Fail-closed public-vs-internal API stability catalogue.

Productises a **narrow** SemVer-intent surface (``stable_core`` + curated
entry points) and explicit workbench / internal classes so the entire research
export surface (~717 top-level / ~1900 package symbols) is **not** silently
reported as guaranteed-stable.

Composes policy from ``DEPRECATIONS.md`` and the internal v1 stability gate
without freezing the whole workbench. Unknown or blank symbol ids fail closed.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Final, Literal, TypeVar, cast

StabilityClass = Literal[
    "semver_stable",
    "documented_public",
    "experimental_workbench",
    "internal",
    "deprecated",
]
"""Stability class vocabulary for declared symbols and path classifications."""

DeprecationState = Literal["active", "deprecated", "not_applicable"]
"""Deprecation state on a catalogue row."""

Visibility = Literal["public", "internal"]
"""Coarse public-vs-internal visibility used by path classification."""

BreakingChangeKind = Literal["remove", "rename", "signature_break"]
"""Kinds of breaking change the validator can assess."""

PUBLIC_API_STABILITY_SCHEMA: Final[str] = "public_api_stability.v1"
"""JSON schema identifier for serialised stability payloads."""

PUBLIC_API_STABILITY_CLAIM_BOUNDARY: Final[str] = (
    "public API stability programme only; semver_stable covers the narrow "
    "declared durable contract (stable_core + curated entry points); "
    "experimental_workbench and bulk package exports are not guaranteed-stable; "
    "unknown/blank symbol ids fail closed; removal of public symbols without a "
    "deprecation record is refused"
)
"""Shared claim boundary for inventory rows and probe results."""

_INTERNAL_PATH_MARKERS: Final[tuple[str, ...]] = (
    "._",
    "/_",
    "tests.",
    "tests/",
    "docs.internal",
    "docs/internal",
    ".coordination",
    "private.",
    "fixtures.",
)

_F = TypeVar("_F", bound=Callable[..., object])


@dataclass(frozen=True, slots=True)
class PublicApiSymbolRecord:
    """One declared symbol or path under the stability programme.

    Attributes
    ----------
    symbol_id
        Fully-qualified symbol or dotted path identifier.
    stability_class
        Stability vocabulary class.
    owner_surface
        Owning surface label (e.g. ``stable_core``, ``cli``, ``workbench``).
    deprecation_state
        Whether the symbol is active, deprecated, or N/A (internal).
    visibility
        Coarse public vs internal classification.
    summary
        Short description.
    replacement_target
        Preferred replacement path when deprecated (empty when not).
    removal_horizon
        Planned removal release label when deprecated (empty when not).
    as_of
        Inventory date label (ISO-like string, not a runtime clock claim).
    claim_boundary
        Non-promotional claim boundary.

    """

    symbol_id: str
    stability_class: StabilityClass
    owner_surface: str
    deprecation_state: DeprecationState
    visibility: Visibility
    summary: str
    replacement_target: str = ""
    removal_horizon: str = ""
    as_of: str = "2026-07-23"
    claim_boundary: str = PUBLIC_API_STABILITY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate symbol-record invariants."""
        if not self.symbol_id or not self.symbol_id.strip():
            raise ValueError("symbol_id must be non-empty")
        if self.stability_class not in {
            "semver_stable",
            "documented_public",
            "experimental_workbench",
            "internal",
            "deprecated",
        }:
            raise ValueError(f"unknown stability_class: {self.stability_class!r}")
        if self.deprecation_state not in {"active", "deprecated", "not_applicable"}:
            raise ValueError(f"unknown deprecation_state: {self.deprecation_state!r}")
        if self.visibility not in {"public", "internal"}:
            raise ValueError(f"unknown visibility: {self.visibility!r}")
        if not self.owner_surface or not self.owner_surface.strip():
            raise ValueError("owner_surface must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        if self.stability_class == "internal" and self.visibility != "internal":
            raise ValueError("internal stability_class requires visibility=internal")
        if self.stability_class == "deprecated":
            if self.deprecation_state != "deprecated":
                raise ValueError(
                    "deprecated stability_class requires deprecation_state=deprecated"
                )
            if not self.replacement_target or not self.replacement_target.strip():
                raise ValueError("deprecated symbols require replacement_target")
            if not self.removal_horizon or not self.removal_horizon.strip():
                raise ValueError("deprecated symbols require removal_horizon")
        if self.deprecation_state == "deprecated" and self.stability_class not in {
            "deprecated",
            "documented_public",
            "experimental_workbench",
            "semver_stable",
        }:
            raise ValueError("deprecated state is invalid for this stability_class")
        if self.visibility == "internal" and self.deprecation_state == "active":
            raise ValueError("internal symbols use deprecation_state=not_applicable")
        if self.visibility == "internal" and self.deprecation_state != "not_applicable":
            raise ValueError("internal symbols must use deprecation_state=not_applicable")
        if self.replacement_target and not self.replacement_target.strip():
            raise ValueError("replacement_target must be non-empty when present")
        if self.removal_horizon and not self.removal_horizon.strip():
            raise ValueError("removal_horizon must be non-empty when present")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this symbol record."""
        return {
            "symbol_id": self.symbol_id,
            "stability_class": self.stability_class,
            "owner_surface": self.owner_surface,
            "deprecation_state": self.deprecation_state,
            "visibility": self.visibility,
            "summary": self.summary,
            "replacement_target": self.replacement_target,
            "removal_horizon": self.removal_horizon,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathClassification:
    """Result of classifying a path or symbol against the programme.

    Attributes
    ----------
    path_id
        Input path or symbol string (stripped).
    visibility
        Public vs internal classification.
    stability_class
        Best-fit stability class (never invents semver_stable for unknowns).
    guaranteed_stable
        True only for declared ``semver_stable`` catalogue rows that are active.
    reason
        Human-readable classification reason.
    in_catalogue
        Whether the path matches a declared catalogue row exactly.

    """

    path_id: str
    visibility: Visibility
    stability_class: StabilityClass
    guaranteed_stable: bool
    reason: str
    in_catalogue: bool

    def __post_init__(self) -> None:
        """Validate classification invariants."""
        if not self.path_id or not self.path_id.strip():
            raise ValueError("path_id must be non-empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.guaranteed_stable and self.stability_class != "semver_stable":
            raise ValueError("guaranteed_stable requires stability_class=semver_stable")
        if self.guaranteed_stable and self.visibility != "public":
            raise ValueError("guaranteed_stable requires visibility=public")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this classification."""
        return {
            "path_id": self.path_id,
            "visibility": self.visibility,
            "stability_class": self.stability_class,
            "guaranteed_stable": self.guaranteed_stable,
            "reason": self.reason,
            "in_catalogue": self.in_catalogue,
        }


@dataclass(frozen=True, slots=True)
class DeprecationProbe:
    """Structured deprecation probe result.

    Attributes
    ----------
    symbol_id
        Queried symbol.
    is_deprecated
        Whether the symbol is staged for removal.
    replacement_target
        Replacement path when deprecated.
    removal_horizon
        Removal release label when deprecated.
    warning_message
        Message that would be emitted by the deprecation decorator/policy.
    reason
        Decision reason.

    """

    symbol_id: str
    is_deprecated: bool
    replacement_target: str
    removal_horizon: str
    warning_message: str
    reason: str
    claim_boundary: str = PUBLIC_API_STABILITY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate deprecation probe invariants."""
        if not self.symbol_id or not self.symbol_id.strip():
            raise ValueError("symbol_id must be non-empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.is_deprecated:
            if not self.replacement_target or not self.replacement_target.strip():
                raise ValueError("deprecated probes require replacement_target")
            if not self.removal_horizon or not self.removal_horizon.strip():
                raise ValueError("deprecated probes require removal_horizon")
            if not self.warning_message or not self.warning_message.strip():
                raise ValueError("deprecated probes require warning_message")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "symbol_id": self.symbol_id,
            "is_deprecated": self.is_deprecated,
            "replacement_target": self.replacement_target,
            "removal_horizon": self.removal_horizon,
            "warning_message": self.warning_message,
            "reason": self.reason,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class BreakingChangeDecision:
    """Fail-closed decision for a proposed breaking change.

    Attributes
    ----------
    symbol_id
        Target symbol.
    change_kind
        Kind of breaking change proposed.
    allowed
        Whether the change is permitted under the programme.
    reason
        Human-readable decision reason.
    requires_deprecation
        Whether a prior deprecation record is required.

    """

    symbol_id: str
    change_kind: BreakingChangeKind
    allowed: bool
    reason: str
    requires_deprecation: bool
    claim_boundary: str = PUBLIC_API_STABILITY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate breaking-change decision invariants."""
        if not self.symbol_id or not self.symbol_id.strip():
            raise ValueError("symbol_id must be non-empty")
        if self.change_kind not in {"remove", "rename", "signature_break"}:
            raise ValueError(f"unknown change_kind: {self.change_kind!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.requires_deprecation:
            # allowed removals after deprecation still required it historically
            pass

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "symbol_id": self.symbol_id,
            "change_kind": self.change_kind,
            "allowed": self.allowed,
            "reason": self.reason,
            "requires_deprecation": self.requires_deprecation,
            "claim_boundary": self.claim_boundary,
        }


def _stable(
    symbol_id: str,
    *,
    owner_surface: str,
    summary: str,
) -> PublicApiSymbolRecord:
    """Build one active semver_stable public row."""
    return PublicApiSymbolRecord(
        symbol_id=symbol_id,
        stability_class="semver_stable",
        owner_surface=owner_surface,
        deprecation_state="active",
        visibility="public",
        summary=summary,
    )


def _workbench(
    symbol_id: str,
    *,
    owner_surface: str,
    summary: str,
) -> PublicApiSymbolRecord:
    """Build one experimental workbench public (not SemVer-guaranteed) row."""
    return PublicApiSymbolRecord(
        symbol_id=symbol_id,
        stability_class="experimental_workbench",
        owner_surface=owner_surface,
        deprecation_state="active",
        visibility="public",
        summary=summary,
    )


def _internal(
    symbol_id: str,
    *,
    owner_surface: str,
    summary: str,
) -> PublicApiSymbolRecord:
    """Build one internal (not public contract) row."""
    return PublicApiSymbolRecord(
        symbol_id=symbol_id,
        stability_class="internal",
        owner_surface=owner_surface,
        deprecation_state="not_applicable",
        visibility="internal",
        summary=summary,
    )


def _deprecated(
    symbol_id: str,
    *,
    owner_surface: str,
    summary: str,
    replacement_target: str,
    removal_horizon: str,
) -> PublicApiSymbolRecord:
    """Build one deprecated public row with mandatory staging fields."""
    return PublicApiSymbolRecord(
        symbol_id=symbol_id,
        stability_class="deprecated",
        owner_surface=owner_surface,
        deprecation_state="deprecated",
        visibility="public",
        summary=summary,
        replacement_target=replacement_target,
        removal_horizon=removal_horizon,
    )


# Narrow durable contract (stable_core intent) + curated examples + policy rows.
# Does NOT freeze the full package __all__ — that would contradict the v1 gate.
_CANONICAL_SYMBOLS: Final[tuple[PublicApiSymbolRecord, ...]] = (
    _stable(
        "scpn_quantum_control.stable_core.Problem",
        owner_surface="stable_core",
        summary="Durable Problem dataclass (stable_core contract).",
    ),
    _stable(
        "scpn_quantum_control.stable_core.Backend",
        owner_surface="stable_core",
        summary="Durable Backend dataclass (stable_core contract).",
    ),
    _stable(
        "scpn_quantum_control.stable_core.Experiment",
        owner_surface="stable_core",
        summary="Durable Experiment dataclass (stable_core contract).",
    ),
    _stable(
        "scpn_quantum_control.stable_core.Result",
        owner_surface="stable_core",
        summary="Durable Result dataclass (stable_core contract).",
    ),
    _stable(
        "scpn_quantum_control.stable_core.build_problem",
        owner_surface="stable_core",
        summary="Problem builder for the durable contract.",
    ),
    _stable(
        "scpn_quantum_control.stable_core.build_backend",
        owner_surface="stable_core",
        summary="Backend builder for the durable contract.",
    ),
    _stable(
        "scpn_quantum_control.stable_core.build_experiment",
        owner_surface="stable_core",
        summary="Experiment builder for the durable contract.",
    ),
    _stable(
        "scpn_quantum_control.stable_core.build_result",
        owner_surface="stable_core",
        summary="Result builder for the durable contract.",
    ),
    _stable(
        "scpn_quantum_control.stable_core.backend_capability_matrix",
        owner_surface="stable_core",
        summary="Backend capability matrix for the durable contract.",
    ),
    _stable(
        "scpn_quantum_control.stable_core.stable_core_capability_payload",
        owner_surface="stable_core",
        summary="Serialisable stable_core capability payload.",
    ),
    _stable(
        "scpn_quantum_control.stable_core_preflight.run_stable_core_preflight",
        owner_surface="stable_core",
        summary="Eligibility preflight gate for stable_core backends.",
    ),
    _stable(
        "project.scripts",
        owner_surface="cli",
        summary="CLI entry points declared in pyproject.toml project.scripts.",
    ),
    _workbench(
        "scpn_quantum_control.scorecard_acceptance_engine",
        owner_surface="workbench",
        summary=(
            "Scorecard acceptance engine — documented public module, "
            "not SemVer-guaranteed until CEO v1 scope freezes it."
        ),
    ),
    _workbench(
        "scpn_quantum_control.governed_route_matrix",
        owner_surface="workbench",
        summary="Governed route matrix — experimental workbench surface.",
    ),
    _workbench(
        "scpn_quantum_control.public_api_stability",
        owner_surface="workbench",
        summary="Public API stability programme surface (this module).",
    ),
    _deprecated(
        "scpn_quantum_control.kuramoto",
        owner_surface="kuramoto_shim",
        summary="Kuramoto toolkit re-export shim; prefer oscillatools.",
        replacement_target="oscillatools",
        removal_horizon="v2.0.0 (no earlier than next major after deprecation)",
    ),
    _deprecated(
        "scpn_quantum_control.accel",
        owner_surface="kuramoto_shim",
        summary="Accel re-export shim; prefer oscillatools.accel.",
        replacement_target="oscillatools.accel",
        removal_horizon="v2.0.0 (no earlier than next major after deprecation)",
    ),
    _deprecated(
        "scpn_quantum_control.forecasting.kuramoto_neural_operator",
        owner_surface="kuramoto_shim",
        summary="Kuramoto neural operator shim; prefer oscillatools.neural_operator.",
        replacement_target="oscillatools.neural_operator",
        removal_horizon="v2.0.0 (no earlier than next major after deprecation)",
    ),
    _internal(
        "scpn_quantum_control._private_helpers",
        owner_surface="internal",
        summary="Illustrative private helper module pattern (leading underscore).",
    ),
    _internal(
        "tests.test_public_api_stability",
        owner_surface="tests",
        summary="Test module — not part of the public SemVer contract.",
    ),
    _internal(
        "docs.internal.v1_api_stability_gate",
        owner_surface="docs_internal",
        summary="Internal planning record for the v1.0 stability gate.",
    ),
)


def _catalogue_map() -> dict[str, PublicApiSymbolRecord]:
    """Return symbol_id → record map; refuse blank or duplicate ids."""
    mapping: dict[str, PublicApiSymbolRecord] = {}
    for row in _CANONICAL_SYMBOLS:
        key = row.symbol_id.strip()
        if not key:
            raise RuntimeError("public API stability catalogue contains blank symbol_id")
        if key in mapping:
            raise RuntimeError(f"duplicate public API symbol_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("public API stability catalogue must be non-empty")
    return mapping


_SYMBOL_BY_ID: Final[Mapping[str, PublicApiSymbolRecord]] = _catalogue_map()


def list_public_api_symbol_ids() -> tuple[str, ...]:
    """Return all declared symbol identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered symbol identifiers.

    """
    return tuple(row.symbol_id for row in _CANONICAL_SYMBOLS)


def get_public_api_symbol(symbol_id: str) -> PublicApiSymbolRecord:
    """Return one catalogue row or raise for blank/unknown identifiers.

    Parameters
    ----------
    symbol_id
        Fully-qualified symbol or path key.

    Returns
    -------
    PublicApiSymbolRecord
        Matching catalogue row.

    Raises
    ------
    ValueError
        If ``symbol_id`` is blank or unknown (fail closed — never invent-stable).

    """
    if not symbol_id or not str(symbol_id).strip():
        raise ValueError("symbol_id must be a non-empty string")
    key = str(symbol_id).strip()
    try:
        return _SYMBOL_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown public API symbol_id {key!r}; refuse invent-stable claim "
            f"(known_count={len(_SYMBOL_BY_ID)})"
        ) from exc


def iter_public_api_symbols(
    *,
    stability_class: StabilityClass | None = None,
    deprecation_state: DeprecationState | None = None,
    visibility: Visibility | None = None,
) -> tuple[PublicApiSymbolRecord, ...]:
    """Return filtered catalogue rows in stable order.

    Parameters
    ----------
    stability_class
        Optional stability class filter.
    deprecation_state
        Optional deprecation state filter.
    visibility
        Optional public/internal filter.

    Returns
    -------
    tuple[PublicApiSymbolRecord, ...]
        Matching rows.

    """
    rows: Iterable[PublicApiSymbolRecord] = _CANONICAL_SYMBOLS
    if stability_class is not None:
        rows = (row for row in rows if row.stability_class == stability_class)
    if deprecation_state is not None:
        rows = (row for row in rows if row.deprecation_state == deprecation_state)
    if visibility is not None:
        rows = (row for row in rows if row.visibility == visibility)
    return tuple(rows)


def _looks_internal_path(path_id: str) -> bool:
    """Heuristic: private/internal path patterns (underscore modules, tests, …)."""
    normalised = path_id.replace("\\", "/")
    if normalised.startswith("_") or "/_" in f"/{normalised}":
        return True
    # dotted private segment: foo._bar
    parts = normalised.replace("/", ".").split(".")
    if any(part.startswith("_") and part != "_" for part in parts):
        return True
    return any(marker in normalised for marker in _INTERNAL_PATH_MARKERS)


def classify_api_path(path_id: str) -> PathClassification:
    """Classify a path/symbol as public or internal with stability semantics.

    Catalogue hits win. Unknown paths that match private/internal patterns are
    classified internal (not guaranteed-stable). Other unknowns are treated as
    experimental workbench **public-ish** but **not** guaranteed-stable — never
    invent ``semver_stable`` for undeclared symbols.

    Parameters
    ----------
    path_id
        Path or fully-qualified symbol string.

    Returns
    -------
    PathClassification
        Classification result.

    Raises
    ------
    ValueError
        If ``path_id`` is blank.

    """
    if not path_id or not str(path_id).strip():
        raise ValueError("path_id must be a non-empty string")
    key = str(path_id).strip()
    record = _SYMBOL_BY_ID.get(key)
    if record is not None:
        guaranteed = (
            record.stability_class == "semver_stable"
            and record.deprecation_state == "active"
            and record.visibility == "public"
        )
        return PathClassification(
            path_id=key,
            visibility=record.visibility,
            stability_class=record.stability_class,
            guaranteed_stable=guaranteed,
            reason=(f"catalogue hit: {record.stability_class} / {record.owner_surface}"),
            in_catalogue=True,
        )
    if _looks_internal_path(key):
        return PathClassification(
            path_id=key,
            visibility="internal",
            stability_class="internal",
            guaranteed_stable=False,
            reason=(
                "internal path pattern (private module, tests, or internal docs); "
                "not guaranteed-stable"
            ),
            in_catalogue=False,
        )
    return PathClassification(
        path_id=key,
        visibility="public",
        stability_class="experimental_workbench",
        guaranteed_stable=False,
        reason=(
            "undeclared path treated as experimental workbench; "
            "not guaranteed-stable (refuse invent-stable)"
        ),
        in_catalogue=False,
    )


def probe_deprecation(symbol_id: str) -> DeprecationProbe:
    """Probe deprecation state for a declared or blank-rejected symbol.

    Parameters
    ----------
    symbol_id
        Catalogue symbol key.

    Returns
    -------
    DeprecationProbe
        Structured deprecation status (replacement + horizon when deprecated).

    Raises
    ------
    ValueError
        If ``symbol_id`` is blank or unknown.

    """
    record = get_public_api_symbol(symbol_id)
    if record.deprecation_state == "deprecated":
        warning = (
            f"{record.symbol_id} is deprecated; use {record.replacement_target} "
            f"instead (removal horizon: {record.removal_horizon})"
        )
        return DeprecationProbe(
            symbol_id=record.symbol_id,
            is_deprecated=True,
            replacement_target=record.replacement_target,
            removal_horizon=record.removal_horizon,
            warning_message=warning,
            reason="catalogue deprecation record present",
        )
    return DeprecationProbe(
        symbol_id=record.symbol_id,
        is_deprecated=False,
        replacement_target="",
        removal_horizon="",
        warning_message="",
        reason=f"not deprecated (deprecation_state={record.deprecation_state})",
    )


def validate_breaking_change(
    symbol_id: str,
    *,
    change_kind: BreakingChangeKind = "remove",
) -> BreakingChangeDecision:
    """Assess a proposed breaking change under fail-closed deprecation policy.

    Public active (non-deprecated) symbols cannot be removed/renamed/broken
    without a prior deprecation record. Already-deprecated public symbols may
    proceed at the stated removal horizon. Internal symbols are out of SemVer
    contract (allowed with reason).

    Parameters
    ----------
    symbol_id
        Catalogue symbol key.
    change_kind
        Kind of breaking change.

    Returns
    -------
    BreakingChangeDecision
        Allowed or refused decision.

    Raises
    ------
    ValueError
        If ``symbol_id`` / ``change_kind`` are invalid.

    """
    if change_kind not in {"remove", "rename", "signature_break"}:
        raise ValueError(f"unknown change_kind: {change_kind!r}")
    record = get_public_api_symbol(symbol_id)

    if record.visibility == "internal":
        return BreakingChangeDecision(
            symbol_id=record.symbol_id,
            change_kind=change_kind,
            allowed=True,
            reason="internal surface is outside SemVer contract; change allowed",
            requires_deprecation=False,
        )

    if record.deprecation_state == "deprecated":
        return BreakingChangeDecision(
            symbol_id=record.symbol_id,
            change_kind=change_kind,
            allowed=True,
            reason=(
                f"deprecation record present; {change_kind} permitted at "
                f"removal horizon {record.removal_horizon!r}"
            ),
            requires_deprecation=True,
        )

    # public active — refuse silent removal / break
    return BreakingChangeDecision(
        symbol_id=record.symbol_id,
        change_kind=change_kind,
        allowed=False,
        reason=(
            f"refuse {change_kind} of public symbol without deprecation record; "
            "stage DeprecationWarning + DEPRECATIONS.md entry first "
            f"(stability_class={record.stability_class})"
        ),
        requires_deprecation=True,
    )


def deprecated_public(
    *,
    symbol_id: str,
    replacement_target: str,
    removal_horizon: str,
) -> Callable[[_F], _F]:
    """Emit ``DeprecationWarning`` from a policy-bound decorator.

    Does not mutate the static catalogue; call sites that decorate live callables
    still need a catalogue row (or pack slice) for inventory integrity. Use
    :func:`probe_deprecation` for structured status without invoking the callable.

    Parameters
    ----------
    symbol_id
        Public symbol being deprecated (for the warning text).
    replacement_target
        Preferred replacement path.
    removal_horizon
        Planned removal release label.

    Returns
    -------
    Callable
        Decorator that wraps a callable.

    Raises
    ------
    ValueError
        If any required field is blank.

    """
    if not symbol_id or not str(symbol_id).strip():
        raise ValueError("symbol_id must be a non-empty string")
    if not replacement_target or not str(replacement_target).strip():
        raise ValueError("replacement_target must be a non-empty string")
    if not removal_horizon or not str(removal_horizon).strip():
        raise ValueError("removal_horizon must be a non-empty string")
    sid = str(symbol_id).strip()
    repl = str(replacement_target).strip()
    horizon = str(removal_horizon).strip()
    message = f"{sid} is deprecated; use {repl} instead (removal horizon: {horizon})"

    def decorator(func: _F) -> _F:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return cast(_F, wrapper)

    return decorator


def build_public_api_stability_registry() -> dict[str, object]:
    """Build the full serialisable public API stability registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every catalogue row (no blanks).

    """
    rows = [row.to_dict() for row in _CANONICAL_SYMBOLS]
    semver = sum(1 for row in _CANONICAL_SYMBOLS if row.stability_class == "semver_stable")
    workbench = sum(
        1 for row in _CANONICAL_SYMBOLS if row.stability_class == "experimental_workbench"
    )
    deprecated = sum(1 for row in _CANONICAL_SYMBOLS if row.deprecation_state == "deprecated")
    internal = sum(1 for row in _CANONICAL_SYMBOLS if row.visibility == "internal")
    public = sum(1 for row in _CANONICAL_SYMBOLS if row.visibility == "public")
    return {
        "schema": PUBLIC_API_STABILITY_SCHEMA,
        "claim_boundary": PUBLIC_API_STABILITY_CLAIM_BOUNDARY,
        "symbol_count": len(rows),
        "semver_stable_count": semver,
        "experimental_workbench_count": workbench,
        "deprecated_count": deprecated,
        "public_count": public,
        "internal_count": internal,
        "blank_entry_count": 0,
        "symbols": rows,
        "policy_note": (
            "Narrow durable contract only; bulk package __all__ is not frozen as "
            "semver_stable. See DEPRECATIONS.md and docs/internal/v1_api_stability_gate.md."
        ),
    }


def assert_public_api_stability_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry has no blanks and no invent-stable unknown classes.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_public_api_stability_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-stable rows appear.

    """
    registry = dict(payload) if payload is not None else build_public_api_stability_registry()
    symbols = registry.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("public API stability registry must contain a non-empty symbols list")
    seen: set[str] = set()
    blank = 0
    for index, row in enumerate(symbols):
        if not isinstance(row, Mapping):
            raise ValueError(f"symbol row {index} must be a mapping")
        symbol_id = row.get("symbol_id")
        stability_class = row.get("stability_class")
        visibility = row.get("visibility")
        if not symbol_id or not str(symbol_id).strip():
            blank += 1
            continue
        sid = str(symbol_id).strip()
        if sid in seen:
            raise ValueError(f"duplicate symbol_id in registry: {sid!r}")
        seen.add(sid)
        if stability_class not in {
            "semver_stable",
            "documented_public",
            "experimental_workbench",
            "internal",
            "deprecated",
        }:
            blank += 1
            continue
        if visibility not in {"public", "internal"}:
            blank += 1
            continue
        if stability_class == "deprecated" and (
            not row.get("replacement_target") or not row.get("removal_horizon")
        ):
            raise ValueError(
                f"deprecated symbol {sid!r} missing replacement_target or removal_horizon"
            )
        if stability_class == "semver_stable" and visibility != "public":
            raise ValueError(f"semver_stable symbol {sid!r} must be visibility=public")
    if blank:
        raise ValueError(f"public API stability registry has {blank} blank or invalid entries")
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    symbol_count = registry.get("symbol_count", -1)
    if not isinstance(symbol_count, int) or symbol_count != len(symbols):
        raise ValueError("symbol_count does not match symbols list length")
    expected = set(list_public_api_symbol_ids())
    if seen != expected:
        raise ValueError(
            f"registry symbol set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    return registry


def version_compatibility_note() -> dict[str, object]:
    """Return a structured version-compatibility and migration note.

    Returns
    -------
    dict[str, object]
        Compatibility matrix note linking DEPRECATIONS.md policy to this surface.

    """
    return {
        "schema": "public_api_version_compatibility.v1",
        "pre_v1": (
            "Until CEO-scoped v1.0 stable freeze lands, pre-1.0 SemVer applies: "
            "anything may change in a 0.y.z release. DEPRECATIONS.md still records "
            "staging for known shims."
        ),
        "v1_intent": (
            "v1.0 SemVer contract is intended to bind the narrow durable surface "
            "(stable_core + curated CLI/schema entry points), not the full workbench."
        ),
        "deprecation_policy": "DEPRECATIONS.md",
        "migration_note": (
            "Kuramoto/accel shims: rewrite imports to oscillatools / "
            "oscillatools.accel / oscillatools.neural_operator before the next major."
        ),
        "claim_boundary": PUBLIC_API_STABILITY_CLAIM_BOUNDARY,
        "as_of": "2026-07-23",
    }


__all__ = [
    "PUBLIC_API_STABILITY_CLAIM_BOUNDARY",
    "PUBLIC_API_STABILITY_SCHEMA",
    "BreakingChangeDecision",
    "BreakingChangeKind",
    "DeprecationProbe",
    "DeprecationState",
    "PathClassification",
    "PublicApiSymbolRecord",
    "StabilityClass",
    "Visibility",
    "assert_public_api_stability_integrity",
    "build_public_api_stability_registry",
    "classify_api_path",
    "deprecated_public",
    "get_public_api_symbol",
    "iter_public_api_symbols",
    "list_public_api_symbol_ids",
    "probe_deprecation",
    "validate_breaking_change",
    "version_compatibility_note",
]
