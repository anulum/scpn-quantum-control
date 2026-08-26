# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — custom derivatives product surface
"""Fail-closed **custom / registered derivatives** product surface.

Productises the safe extension point for third-party and domain JVP/VJP rules:
versioned registration contract, public register/query/list helpers, fail-closed
blank/unknown/duplicate paths, and an example linear rule.

Composes ambient :class:`~scpn_quantum_control.program_ad_registry.CustomDerivativeRegistry`
and :class:`~scpn_quantum_control.program_ad_registry.CustomDerivativeRule` —
does **not** rewrite the full transform algebra stack or invent-green full
transform-algebra/route-matrix CI. Complete transform-algebra interaction
coverage and per-rule metamorphic verification remain open honestly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .program_ad_registry import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
)

ContractKind = Literal[
    "registration_contract",
    "public_register_api",
    "example_rule",
    "transform_algebra_boundary",
    "metamorphic_boundary",
]
"""Product contract kinds in the catalogue."""

CUSTOM_DERIVATIVES_PRODUCT_SCHEMA: Final[str] = "custom_derivatives_product.v2"
"""JSON schema identifier for serialised product payloads."""

DEFAULT_PRODUCT_NAMESPACE: Final[str] = "scpn.product.custom_derivatives"
"""Namespace for product-managed demo / extension rules."""

CUSTOM_DERIVATIVES_CLAIM_BOUNDARY: Final[str] = (
    "Custom derivatives product surface only; versioned registration contract "
    "and fail-closed register/query over ambient CustomDerivativeRegistry; "
    "does not invent-green complete transform-algebra and governed route-matrix "
    "CI or mass rule migration; transform-algebra interaction coverage and "
    "per-rule metamorphic verification remain open honestly"
)
"""Shared claim boundary for contracts, registrations, and decisions."""


@dataclass(frozen=True, slots=True)
class CustomDerivativeContractRow:
    """One product contract row for the custom-derivatives surface.

    Attributes
    ----------
    contract_id
        Stable catalogue identifier.
    kind
        Contract kind.
    title
        Human-readable title.
    summary
        Short description.
    module_path
        Primary ambient module path.
    symbol_name
        Primary ambient symbol.
    metamorphic_verification_pointer
        Optional metamorphic-verification boundary pointer.
    api_stability_class
        Stability honesty class.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    contract_id: str
    kind: ContractKind
    title: str
    summary: str
    module_path: str
    symbol_name: str
    metamorphic_verification_pointer: str = "metamorphic_ad_verification.custom_rule_residual"
    api_stability_class: str = "experimental_workbench"
    as_of: str = "2026-07-24"
    claim_boundary: str = CUSTOM_DERIVATIVES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate contract row invariants."""
        if not self.contract_id or not self.contract_id.strip():
            raise ValueError("contract_id must be non-empty")
        if self.kind not in {
            "registration_contract",
            "public_register_api",
            "example_rule",
            "transform_algebra_boundary",
            "metamorphic_boundary",
        }:
            raise ValueError(f"unknown contract kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.module_path or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if not self.symbol_name or not self.symbol_name.strip():
            raise ValueError("symbol_name must be non-empty")
        if not self.api_stability_class or not self.api_stability_class.strip():
            raise ValueError("api_stability_class must be non-empty")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this contract row."""
        return {
            "contract_id": self.contract_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "module_path": self.module_path,
            "symbol_name": self.symbol_name,
            "metamorphic_verification_pointer": self.metamorphic_verification_pointer,
            "api_stability_class": self.api_stability_class,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """Result of a product-level custom derivative registration.

    Attributes
    ----------
    identity_key
        Canonical ``namespace:name@version`` key.
    rule_name
        Registered rule name.
    registered
        Whether registration succeeded.
    overwrite
        Whether overwrite was requested.
    claim_boundary
        Non-promotional claim boundary.

    """

    identity_key: str
    rule_name: str
    registered: bool
    overwrite: bool
    claim_boundary: str = CUSTOM_DERIVATIVES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate registration result invariants."""
        if not self.identity_key or not self.identity_key.strip():
            raise ValueError("identity_key must be non-empty")
        if not self.rule_name or not self.rule_name.strip():
            raise ValueError("rule_name must be non-empty")
        if not self.registered:
            raise ValueError("registered must be True for successful results")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this result."""
        return {
            "identity_key": self.identity_key,
            "rule_name": self.rule_name,
            "registered": self.registered,
            "overwrite": self.overwrite,
            "claim_boundary": self.claim_boundary,
        }


def _row(
    contract_id: str,
    *,
    kind: ContractKind,
    title: str,
    summary: str,
    module_path: str,
    symbol_name: str,
) -> CustomDerivativeContractRow:
    """Build one catalogue row."""
    return CustomDerivativeContractRow(
        contract_id=contract_id,
        kind=kind,
        title=title,
        summary=summary,
        module_path=module_path,
        symbol_name=symbol_name,
    )


_CANONICAL_CONTRACTS: Final[tuple[CustomDerivativeContractRow, ...]] = (
    _row(
        "registration_contract",
        kind="registration_contract",
        title="Registration contract",
        summary=(
            "Versioned PrimitiveIdentity + CustomDerivativeRule binding contract "
            "with conflict-safe register (fail closed on duplicates)."
        ),
        module_path="scpn_quantum_control.program_ad_registry",
        symbol_name="CustomDerivativeRegistry",
    ),
    _row(
        "public_register_api",
        kind="public_register_api",
        title="Public register / query API",
        summary=(
            "Product wrappers: register_product_custom_rule, "
            "require_product_custom_rule, list_product_registered_identities."
        ),
        module_path="scpn_quantum_control.custom_derivatives_product",
        symbol_name="register_product_custom_rule",
    ),
    _row(
        "example_linear_rule",
        kind="example_rule",
        title="Example scaled-linear custom rule",
        summary=("Documented example CustomDerivativeRule with exact JVP/VJP for y = scale * x."),
        module_path="scpn_quantum_control.custom_derivatives_product",
        symbol_name="build_example_scaled_linear_rule",
    ),
    _row(
        "transform_algebra_boundary",
        kind="transform_algebra_boundary",
        title="Transform-algebra interaction coverage boundary",
        summary=(
            "Boundary-only product row: full transform-algebra matrix CI for "
            "custom rules remains open."
        ),
        module_path="scpn_quantum_control.differentiable_custom_derivatives",
        symbol_name="custom_jvp",
    ),
    _row(
        "metamorphic_boundary",
        kind="metamorphic_boundary",
        title="Per-rule metamorphic verification boundary",
        summary=(
            "Boundary-only product row: full metamorphic automation for every "
            "new custom rule remains open."
        ),
        module_path="scpn_quantum_control.metamorphic_ad_verification",
        symbol_name="probe_metamorphic_law",
    ),
)


def _catalogue_map() -> dict[str, CustomDerivativeContractRow]:
    """Return contract_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, CustomDerivativeContractRow] = {}
    for row in _CANONICAL_CONTRACTS:
        key = row.contract_id.strip()
        if not key:
            raise RuntimeError("custom derivatives catalogue contains blank contract_id")
        if key in mapping:
            raise RuntimeError(f"duplicate contract_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("custom derivatives catalogue must be non-empty")
    return mapping


_CONTRACT_BY_ID: Final[Mapping[str, CustomDerivativeContractRow]] = _catalogue_map()


def list_custom_derivative_contract_ids() -> tuple[str, ...]:
    """Return all product contract identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered contract identifiers.

    """
    return tuple(row.contract_id for row in _CANONICAL_CONTRACTS)


def get_custom_derivative_contract(contract_id: str) -> CustomDerivativeContractRow:
    """Return one contract row or raise for blank/unknown identifiers.

    Parameters
    ----------
    contract_id
        Catalogue contract key.

    Returns
    -------
    CustomDerivativeContractRow
        Matching row.

    Raises
    ------
    ValueError
        If ``contract_id`` is blank or unknown (fail closed).

    """
    if not contract_id or not str(contract_id).strip():
        raise ValueError("contract_id must be a non-empty string")
    key = str(contract_id).strip()
    try:
        return _CONTRACT_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown contract_id {key!r}; refuse invent-green custom derivatives "
            f"product claim (known_count={len(_CONTRACT_BY_ID)})"
        ) from exc


def iter_custom_derivative_contracts(
    *,
    kind: ContractKind | None = None,
) -> tuple[CustomDerivativeContractRow, ...]:
    """Return filtered contract rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.

    Returns
    -------
    tuple[CustomDerivativeContractRow, ...]
        Matching rows.

    """
    rows: Sequence[CustomDerivativeContractRow] = _CANONICAL_CONTRACTS
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def registration_contract_policy() -> dict[str, object]:
    """Return the versioned registration contract policy.

    Returns
    -------
    dict[str, object]
        Policy describing identity keys, fail-closed duplicates, and residuals.

    """
    return {
        "product_schema": CUSTOM_DERIVATIVES_PRODUCT_SCHEMA,
        "default_namespace": DEFAULT_PRODUCT_NAMESPACE,
        "identity_key_format": "namespace:name@version",
        "fail_closed_blank_identity": True,
        "fail_closed_duplicate_without_overwrite": True,
        "require_jvp_or_vjp": True,
        "transform_algebra_ci_residual": "transform-algebra-interaction-coverage",
        "metamorphic_verification_residual": "custom-rule-metamorphic-verification",
        "claim_boundary": CUSTOM_DERIVATIVES_CLAIM_BOUNDARY,
    }


def parse_product_identity(identity: PrimitiveIdentity | str) -> PrimitiveIdentity:
    """Parse a product identity key fail-closed.

    Parameters
    ----------
    identity
        Primitive identity or ``namespace:name[@version]`` string.

    Returns
    -------
    PrimitiveIdentity
        Parsed identity.

    Raises
    ------
    ValueError
        If blank or malformed.

    """
    if identity is None:
        raise ValueError("identity must be a non-empty string or PrimitiveIdentity")
    if isinstance(identity, str) and not identity.strip():
        raise ValueError("identity must be a non-empty string")
    return PrimitiveIdentity.parse(identity)


def build_example_scaled_linear_rule(
    *,
    scale: float = 2.0,
    name: str = "scaled_linear",
) -> CustomDerivativeRule:
    """Build the documented example scaled-linear custom rule.

    Implements ``y = scale * x`` with exact JVP/VJP:

    - JVP: ``scale * tangent``
    - VJP: ``scale * cotangent`` (same for real linear map)

    Parameters
    ----------
    scale
        Finite real scale factor (must be non-zero for a useful demo).
    name
        Rule name token.

    Returns
    -------
    CustomDerivativeRule
        Exact JVP+VJP rule.

    Raises
    ------
    ValueError
        If ``scale`` is non-finite or zero, or ``name`` is blank.

    """
    if not name or not str(name).strip():
        raise ValueError("name must be a non-empty string")
    scale_f = float(scale)
    if not np.isfinite(scale_f):
        raise ValueError("scale must be finite")
    if scale_f == 0.0:
        raise ValueError("scale must be non-zero for the example rule")

    def value_fn(values: ArrayLike) -> NDArray[np.float64]:
        arr = np.asarray(values, dtype=np.float64)
        return scale_f * arr

    def jvp_rule(values: ArrayLike, tangent: ArrayLike) -> NDArray[np.float64]:
        del values  # linear map independent of position
        t = np.asarray(tangent, dtype=np.float64)
        return scale_f * t

    def vjp_rule(values: ArrayLike, cotangent: ArrayLike) -> NDArray[np.float64]:
        del values
        ct = np.asarray(cotangent, dtype=np.float64)
        return scale_f * ct

    return CustomDerivativeRule(
        name=str(name).strip(),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
        parameter_names=(),
        trainable=(),
    )


def new_product_registry() -> CustomDerivativeRegistry:
    """Return an isolated empty product registry (does not touch the default).

    Returns
    -------
    CustomDerivativeRegistry
        Fresh registry for product-scoped demos and tests.

    """
    return CustomDerivativeRegistry()


def register_product_custom_rule(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    overwrite: bool = False,
    registry: CustomDerivativeRegistry | None = None,
) -> RegistrationResult:
    """Register a custom derivative rule fail-closed.

    Parameters
    ----------
    identity
        Primitive identity or key string.
    rule
        Custom derivative rule to bind.
    overwrite
        Whether an existing different rule may be replaced.
    registry
        Optional registry; when omitted a **new isolated** registry is used
        (never silently mutates the process default). Callers that need the
        default ambient registry must pass it explicitly.

    Returns
    -------
    RegistrationResult
        Successful registration metadata.

    Raises
    ------
    ValueError
        If identity/rule is invalid or a duplicate exists without overwrite.

    """
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    target = new_product_registry() if registry is None else registry
    if not isinstance(target, CustomDerivativeRegistry):
        raise ValueError("registry must be a CustomDerivativeRegistry")
    parsed = parse_product_identity(identity)
    registered = target.register(parsed, rule, overwrite=overwrite)
    return RegistrationResult(
        identity_key=parsed.key,
        rule_name=registered.name,
        registered=True,
        overwrite=overwrite,
    )


def require_product_custom_rule(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry,
) -> CustomDerivativeRule:
    """Require a registered rule or fail closed.

    Parameters
    ----------
    identity
        Primitive identity or key.
    registry
        Registry to query (must be provided explicitly).

    Returns
    -------
    CustomDerivativeRule
        Registered rule.

    Raises
    ------
    ValueError
        If identity is blank/unknown or rule is missing.

    """
    if not isinstance(registry, CustomDerivativeRegistry):
        raise ValueError("registry must be a CustomDerivativeRegistry")
    parsed = parse_product_identity(identity)
    return registry.require(parsed)


def list_product_registered_identities(
    *,
    registry: CustomDerivativeRegistry,
) -> tuple[str, ...]:
    """List registered identity keys in sorted order.

    Parameters
    ----------
    registry
        Registry to inspect.

    Returns
    -------
    tuple[str, ...]
        Sorted identity keys.

    """
    if not isinstance(registry, CustomDerivativeRegistry):
        raise ValueError("registry must be a CustomDerivativeRegistry")
    keys = [identity.key for identity in registry.snapshot()]
    return tuple(sorted(keys))


def probe_example_rule_round_trip(
    *,
    scale: float = 2.0,
    values: ArrayLike | None = None,
    tangent: ArrayLike | None = None,
) -> dict[str, object]:
    """Exercise the example rule with ambient custom_jvp (non-empty observables).

    Parameters
    ----------
    scale
        Example scale factor.
    values
        Parameter vector (default ``[1.0, 2.0]``).
    tangent
        Tangent vector (default ones like values).

    Returns
    -------
    dict[str, object]
        Value and JVP arrays plus registration identity for the demo.

    Raises
    ------
    ValueError
        If shapes mismatch or rule construction fails.

    """
    from .differentiable_custom_derivatives import value_and_custom_jvp

    rule = build_example_scaled_linear_rule(scale=scale)
    x = np.asarray([1.0, 2.0] if values is None else values, dtype=np.float64)
    t = np.asarray(np.ones_like(x) if tangent is None else tangent, dtype=np.float64)
    if x.shape != t.shape:
        raise ValueError("values and tangent must have the same shape")
    registry = new_product_registry()
    identity = PrimitiveIdentity(
        namespace=DEFAULT_PRODUCT_NAMESPACE,
        name="scaled_linear",
        version="1",
    )
    register_product_custom_rule(identity, rule, registry=registry)
    result = value_and_custom_jvp(rule, x, t)
    value = np.asarray(result.value, dtype=np.float64)
    jvp = np.asarray(result.jvp, dtype=np.float64)
    expected_value = float(scale) * x
    expected_jvp = float(scale) * t
    if not np.allclose(value, expected_value):
        raise ValueError("example rule value does not match scale * values")
    if not np.allclose(jvp, expected_jvp):
        raise ValueError("example rule JVP does not match scale * tangent")
    return {
        "identity_key": identity.key,
        "rule_name": rule.name,
        "scale": float(scale),
        "value": value.tolist(),
        "jvp": jvp.tolist(),
        "registered_identities": list(list_product_registered_identities(registry=registry)),
        "claim_boundary": CUSTOM_DERIVATIVES_CLAIM_BOUNDARY,
    }


def map_custom_derivatives_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of custom-derivatives product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for contract in _CANONICAL_CONTRACTS:
        path = contract.module_path
        if path in seen:
            continue
        seen.add(path)
        rows.append(
            {
                "module_path": path,
                "role": "custom_derivatives_product_surface",
                "api_stability_class": contract.api_stability_class,
                "kind": contract.kind,
                "contract_ids": [
                    c.contract_id for c in _CANONICAL_CONTRACTS if c.module_path == path
                ],
                "claim_boundary": CUSTOM_DERIVATIVES_CLAIM_BOUNDARY,
            }
        )
    return tuple(rows)


def build_custom_derivatives_product_registry() -> dict[str, object]:
    """Build the full serialisable custom-derivatives product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with contracts and policy (no blanks).

    """
    contracts = [row.to_dict() for row in _CANONICAL_CONTRACTS]
    return {
        "schema": CUSTOM_DERIVATIVES_PRODUCT_SCHEMA,
        "claim_boundary": CUSTOM_DERIVATIVES_CLAIM_BOUNDARY,
        "contract_count": len(contracts),
        "blank_entry_count": 0,
        "default_contract_id": "registration_contract",
        "registration_policy": registration_contract_policy(),
        "public_surfaces": list(map_custom_derivatives_public_surfaces()),
        "contracts": contracts,
        "policy_note": (
            "Custom derivatives product catalogue only; ambient registry remains "
            "the implementation; complete transform-algebra interaction coverage "
            "and per-rule metamorphic verification remain open honestly."
        ),
    }


def assert_custom_derivatives_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers contracts without blanks.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_custom_derivatives_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage or blanks appear.

    """
    registry = (
        dict(payload) if payload is not None else build_custom_derivatives_product_registry()
    )
    contracts = registry.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        raise ValueError(
            "custom derivatives product registry must contain a non-empty contracts list"
        )
    seen: set[str] = set()
    blank = 0
    default_found = False
    for index, row in enumerate(contracts):
        if not isinstance(row, Mapping):
            raise ValueError(f"contract row {index} must be a mapping")
        contract_id = row.get("contract_id")
        kind = row.get("kind")
        symbol_name = row.get("symbol_name")
        if not contract_id or not str(contract_id).strip():
            blank += 1
            continue
        cid = str(contract_id).strip()
        if cid in seen:
            raise ValueError(f"duplicate contract_id in registry: {cid!r}")
        seen.add(cid)
        if cid == "registration_contract":
            default_found = True
        if kind not in {
            "registration_contract",
            "public_register_api",
            "example_rule",
            "transform_algebra_boundary",
            "metamorphic_boundary",
        }:
            blank += 1
            continue
        if not symbol_name or not str(symbol_name).strip():
            raise ValueError(f"contract {cid!r} must have symbol_name")
    if blank:
        raise ValueError(
            f"custom derivatives product registry has {blank} blank or invalid entries"
        )
    if not default_found:
        raise ValueError("custom derivatives product registry missing registration_contract")
    expected = set(list_custom_derivative_contract_ids())
    if seen != expected:
        raise ValueError(
            f"registry contract set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    contract_count = registry.get("contract_count", -1)
    if not isinstance(contract_count, int) or contract_count != len(contracts):
        raise ValueError("contract_count does not match contracts list length")
    policy = registry.get("registration_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("registration_policy must be a mapping")
    if policy.get("fail_closed_duplicate_without_overwrite") is not True:
        raise ValueError("registration_policy must fail closed on duplicates without overwrite")
    return registry


__all__ = [
    "CUSTOM_DERIVATIVES_CLAIM_BOUNDARY",
    "CUSTOM_DERIVATIVES_PRODUCT_SCHEMA",
    "DEFAULT_PRODUCT_NAMESPACE",
    "ContractKind",
    "CustomDerivativeContractRow",
    "RegistrationResult",
    "assert_custom_derivatives_product_integrity",
    "build_custom_derivatives_product_registry",
    "build_example_scaled_linear_rule",
    "get_custom_derivative_contract",
    "iter_custom_derivative_contracts",
    "list_custom_derivative_contract_ids",
    "list_product_registered_identities",
    "map_custom_derivatives_public_surfaces",
    "new_product_registry",
    "parse_product_identity",
    "probe_example_rule_round_trip",
    "register_product_custom_rule",
    "registration_contract_policy",
    "require_product_custom_rule",
]
